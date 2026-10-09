//! Sparse byte memory shared by the Numeric and Trace backends: 64 KiB pages
//! per memory block, allocated on first write, zero when never written.
//! Writes into sync-only ranges are ignored; a block in a `null`-fidelity
//! subtree drops writes and rejects loads, and a DMA from it copies nothing.

use std::collections::HashMap;

use nu_model::Range;
use sim_core::{FuncError, MemBlock};

const PAGE: u64 = 1 << 16;

struct Block {
    range: Range,
    numeric: bool,
    pages: HashMap<u64, Box<[u8]>>,
}

#[derive(Default)]
pub(crate) struct Memory {
    blocks: Vec<Block>,
    sync_only: Vec<Range>,
}

pub(crate) fn err<T>(msg: String) -> Result<T, FuncError> {
    Err(FuncError(msg))
}

impl Memory {
    pub fn configure(&mut self, memories: &[MemBlock], sync_only: &[Range]) {
        // One store per memory: the slices of a sliced memory share its
        // range.
        self.blocks = Vec::new();
        for m in memories {
            if !self.blocks.iter().any(|b| b.range == m.range) {
                self.blocks.push(Block {
                    range: m.range,
                    numeric: m.numeric,
                    pages: HashMap::new(),
                });
            }
        }
        self.sync_only = sync_only.to_vec();
    }

    /// True when `range` lies in a `null`-fidelity block: it has no data,
    /// so a DMA from it copies nothing.
    pub fn is_null(&self, range: &Range) -> bool {
        self.blocks
            .iter()
            .any(|b| !b.numeric && b.range.covers(range))
    }

    fn is_sync_only(&self, range: &Range) -> bool {
        self.sync_only.iter().any(|s| s.overlaps(range))
    }

    fn block(&mut self, range: &Range) -> Result<&mut Block, FuncError> {
        match self.blocks.iter_mut().find(|b| b.range.covers(range)) {
            Some(b) => Ok(b),
            None => err(format!("{range} is not inside one memory block")),
        }
    }

    pub fn read(&mut self, addr: u64, len: u64) -> Result<Vec<u8>, FuncError> {
        let b = self.block(&Range::from_len(addr, len))?;
        if !b.numeric {
            return err(format!(
                "read of [{addr:#x}, +{len}) in a null-fidelity block"
            ));
        }
        let mut out = vec![0u8; len as usize];
        let mut off = 0u64;
        while off < len {
            let a = addr + off;
            let (page, in_page) = (a / PAGE, a % PAGE);
            let n = (PAGE - in_page).min(len - off);
            if let Some(p) = b.pages.get(&page) {
                out[off as usize..(off + n) as usize]
                    .copy_from_slice(&p[in_page as usize..(in_page + n) as usize]);
            }
            off += n;
        }
        Ok(out)
    }

    pub fn write(&mut self, addr: u64, bytes: &[u8]) -> Result<(), FuncError> {
        let len = bytes.len() as u64;
        let range = Range::from_len(addr, len);
        if self.is_sync_only(&range) {
            return Ok(());
        }
        let b = self.block(&range)?;
        if !b.numeric {
            return Ok(());
        }
        let mut off = 0u64;
        while off < len {
            let a = addr + off;
            let (page, in_page) = (a / PAGE, a % PAGE);
            let n = (PAGE - in_page).min(len - off);
            let p = b
                .pages
                .entry(page)
                .or_insert_with(|| vec![0u8; PAGE as usize].into_boxed_slice());
            p[in_page as usize..(in_page + n) as usize]
                .copy_from_slice(&bytes[off as usize..(off + n) as usize]);
            off += n;
        }
        Ok(())
    }
}

impl Memory {
    /// The DMA of a task with dst `task_dst`: `bytes` from `src` to `dst`.
    /// A copy from a `null`-fidelity block copies nothing.
    pub fn dma(
        &mut self,
        task_dst: &Range,
        src: u64,
        dst: u64,
        bytes: u64,
    ) -> Result<(), FuncError> {
        check_in_dst(task_dst, &Range::from_len(dst, bytes))?;
        if self.is_null(&Range::from_len(src, bytes)) {
            return Ok(());
        }
        let data = self.read(src, bytes)?;
        self.write(dst, &data)
    }
}

/// Checks that a task writes only inside its `dst`.
pub(crate) fn check_in_dst(dst: &Range, written: &Range) -> Result<(), FuncError> {
    if written.is_empty() || dst.covers(written) {
        Ok(())
    } else {
        err(format!("task writes {written} outside its dst {dst}"))
    }
}
