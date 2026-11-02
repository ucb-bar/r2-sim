//! The memory store of the functional plane: one anonymous mapping per
//! memory block that holds data, shared by every backend.
//!
//! Each block of a numeric subtree (each HBM, SMEM bank and tmem, and
//! each linear range of an address hash) is one private anonymous mapping
//! of the block's size, reserved at configure (`MAP_NORESERVE`): the
//! operating system supplies a page at its first write, and a byte that no
//! op wrote reads as zero. A block does not move until the store is
//! dropped, which unmaps it. A block of a null subtree has no mapping: a
//! read from it is an error, a write to it has no effect, and a copy from
//! it copies nothing. A write into the sync-only space has no effect.
//!
//! The blocks are sorted by base and do not overlap; a lookup is a binary
//! search. A range must lie in one block. During a run only the worker
//! thread uses the store.

use nu_model::{Dst, Range};

use crate::config::MemBlock;
use crate::func::FuncError;
use crate::profile::Rows;

/// One memory block of the store.
#[derive(Debug)]
pub struct StoreBlock {
    /// The global address of the first byte.
    pub base: u64,
    /// The bytes of the block.
    pub size: u64,
    /// The bytes of the block in this process; null for a block of a null
    /// subtree (no data).
    pub ptr: *mut u8,
    /// The block's name, for example `root.gpc0.hbm`.
    pub name: String,
}

/// The memory of the functional plane.
#[derive(Debug, Default)]
pub struct Store {
    blocks: Vec<StoreBlock>,
    sync_only: Vec<Range>,
    granule: u64,
}

// SAFETY: the store owns its mappings, and one thread at a time uses it
// (the worker thread during the run).
unsafe impl Send for Store {}

fn err<T>(msg: String) -> Result<T, FuncError> {
    Err(FuncError(msg))
}

impl Store {
    /// Maps the blocks of `memories` that hold data (a block listed twice
    /// is mapped once). `granule`: the Neutrino granule, the unit of a
    /// destination range's mask. An error when two blocks overlap or a
    /// mapping fails.
    pub fn new(
        memories: &[MemBlock],
        sync_only: &[Range],
        granule: u64,
    ) -> Result<Self, FuncError> {
        let mut s = Store {
            blocks: Vec::new(),
            sync_only: sync_only.to_vec(),
            granule: granule.max(1),
        };
        let mut list: Vec<&MemBlock> = Vec::new();
        for m in memories {
            if !m.range.is_empty() && !list.iter().any(|b| b.range == m.range) {
                list.push(m);
            }
        }
        list.sort_by_key(|m| m.range.base);
        for m in list {
            if let Some(prev) = s.blocks.last() {
                if prev.base + prev.size > m.range.base {
                    return err(format!(
                        "memory blocks {} and {} overlap",
                        prev.name, m.name
                    ));
                }
            }
            let size = m.range.len();
            let ptr = if m.numeric {
                map(size)?
            } else {
                std::ptr::null_mut()
            };
            s.blocks.push(StoreBlock {
                base: m.range.base,
                size,
                ptr,
                name: m.name.clone(),
            });
        }
        Ok(s)
    }

    /// The blocks, sorted by base.
    pub fn blocks(&self) -> &[StoreBlock] {
        &self.blocks
    }

    /// The sync-only space.
    pub fn sync_only(&self) -> &[Range] {
        &self.sync_only
    }

    /// The Neutrino granule.
    pub fn granule(&self) -> u64 {
        self.granule.max(1)
    }

    /// The index of the block that holds `[addr, addr + len)`.
    fn find(&self, addr: u64, len: u64) -> Result<usize, FuncError> {
        let i = self.blocks.partition_point(|b| b.base <= addr);
        if i > 0 {
            let b = &self.blocks[i - 1];
            let off = addr - b.base;
            if off <= b.size && len <= b.size - off {
                return Ok(i - 1);
            }
        }
        err(format!(
            "{} is not inside one memory block",
            Range::from_len(addr, len)
        ))
    }

    /// True when `[addr, addr + len)` lies in a block of a null subtree.
    pub fn is_null(&self, addr: u64, len: u64) -> bool {
        self.find(addr, len)
            .is_ok_and(|i| self.blocks[i].ptr.is_null())
    }

    /// True when `[addr, addr + len)` lies in a block that holds data.
    pub fn has_data(&self, addr: u64, len: u64) -> bool {
        self.find(addr, len)
            .is_ok_and(|i| !self.blocks[i].ptr.is_null())
    }

    fn is_sync_only(&self, addr: u64, len: u64) -> bool {
        let r = Range::from_len(addr, len);
        self.sync_only.iter().any(|s| s.overlaps(&r))
    }

    /// The bytes `[addr, addr + len)`. An error outside one block and in a
    /// block of a null subtree.
    pub fn bytes(&self, addr: u64, len: u64) -> Result<&[u8], FuncError> {
        let b = &self.blocks[self.find(addr, len)?];
        if b.ptr.is_null() {
            return err(format!(
                "read of {} in a null-fidelity block",
                Range::from_len(addr, len)
            ));
        }
        // SAFETY: the range lies in the block's mapping, which lives as
        // long as the store; `&self` excludes a writer.
        Ok(
            unsafe {
                std::slice::from_raw_parts(b.ptr.add((addr - b.base) as usize), len as usize)
            },
        )
    }

    /// The bytes `[addr, addr + len)` to write; `None` when a write has no
    /// effect there (the sync-only space, or a block of a null subtree).
    /// An error outside one block.
    pub fn bytes_mut(&mut self, addr: u64, len: u64) -> Result<Option<&mut [u8]>, FuncError> {
        if self.is_sync_only(addr, len) {
            return Ok(None);
        }
        let b = &self.blocks[self.find(addr, len)?];
        if b.ptr.is_null() {
            return Ok(None);
        }
        // SAFETY: as in `bytes`; `&mut self` makes the slice unique.
        Ok(Some(unsafe {
            std::slice::from_raw_parts_mut(b.ptr.add((addr - b.base) as usize), len as usize)
        }))
    }

    /// A copy of `[addr, addr + len)`.
    pub fn read(&self, addr: u64, len: u64) -> Result<Vec<u8>, FuncError> {
        self.bytes(addr, len).map(<[u8]>::to_vec)
    }

    /// Writes `bytes` at `addr` (no effect in the sync-only space or a
    /// block of a null subtree).
    pub fn write(&mut self, addr: u64, bytes: &[u8]) -> Result<(), FuncError> {
        if let Some(m) = self.bytes_mut(addr, bytes.len() as u64)? {
            m.copy_from_slice(bytes);
        }
        Ok(())
    }

    /// The DMA of a task with dst `task_dst`: `bytes` from `src` to `dst`,
    /// or with `rows` the rows of a 2D copy, each as a memmove (a row's
    /// ranges may overlap). Each written row must lie in `task_dst`. A row
    /// from a block of a null subtree copies nothing; a row into the
    /// sync-only space or a block of a null subtree writes nothing.
    pub fn dma(
        &mut self,
        task_dst: &Dst,
        src: u64,
        dst: u64,
        bytes: u64,
        rows: Option<&Rows>,
    ) -> Result<(), FuncError> {
        let (count, len, ss, ds) = match rows {
            None => (1, bytes, 0, 0),
            Some(r) => (r.count, r.bytes, r.src_stride, r.dst_stride),
        };
        for r in 0..count {
            let (s, d) = (src + r * ss, dst + r * ds);
            check_in_dst(task_dst, self.granule(), &Range::from_len(d, len))?;
            let from = self.raw(s, len)?;
            if from.is_null() || self.is_sync_only(d, len) {
                continue;
            }
            let to = self.raw(d, len)?;
            if !to.is_null() {
                // SAFETY: both ranges lie in mappings of the store (`raw`),
                // and no reference to their bytes exists; `copy` allows
                // them to overlap.
                unsafe { std::ptr::copy(from, to, len as usize) };
            }
        }
        Ok(())
    }

    /// A raw pointer to the first byte of `[addr, addr + len)`; null for a
    /// block of a null subtree. An error outside one block. It makes no
    /// reference to the bytes.
    fn raw(&self, addr: u64, len: u64) -> Result<*mut u8, FuncError> {
        let b = &self.blocks[self.find(addr, len)?];
        if b.ptr.is_null() {
            return Ok(std::ptr::null_mut());
        }
        // SAFETY: `find` checked that the range lies in the mapping.
        Ok(unsafe { b.ptr.add((addr - b.base) as usize) })
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        for b in &self.blocks {
            if !b.ptr.is_null() {
                // SAFETY: `ptr` is a mapping of `size` bytes made by `map`.
                unsafe { libc::munmap(b.ptr.cast(), b.size as usize) };
            }
        }
    }
}

/// A private anonymous mapping of `size` bytes, without swap reservation.
fn map(size: u64) -> Result<*mut u8, FuncError> {
    // SAFETY: an anonymous mapping at an address the kernel chooses.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return err(format!(
            "mapping {size} bytes: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(p.cast())
}

/// Checks that a task writes only granules that its masked `dst` holds.
pub fn check_in_dst(dst: &Dst, granule: u64, written: &Range) -> Result<(), FuncError> {
    if dst.covers_bytes(granule, written) {
        Ok(())
    } else if dst.mask == nu_model::FULL_MASK {
        err(format!(
            "task writes {written} outside its dst {}",
            dst.span
        ))
    } else {
        err(format!(
            "task writes {written} outside its dst {} with mask {:#x}",
            dst.span, dst.mask
        ))
    }
}
