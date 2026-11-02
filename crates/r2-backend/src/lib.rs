//! The interface of a functional backend plug-in: the `#[repr(C)]` types
//! of `include/r2_backend.h`, with the same layout. A plug-in is a shared
//! library that exports [`OPEN`]; r2-sim calls it, checks
//! [`Backend::abi`] against [`ABI`], then calls `configure` once, `apply`
//! once per task in the order of `seq`, and `close` after the last task.

use std::ffi::c_char;

/// `R2_BACKEND_ABI`.
pub const ABI: u32 = 1;

/// The name of the one exported symbol, with its terminating NUL.
pub const OPEN: &[u8] = b"r2_backend_open\0";

/// `r2_block`: one memory block of the store.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Block {
    /// The global byte address of the first byte.
    pub base: u64,
    /// The bytes of the block.
    pub size: u64,
    /// The bytes of the block; null for a block of a null subtree.
    pub ptr: *mut u8,
    /// The block's name, NUL-terminated.
    pub name: *const c_char,
}

/// `r2_range`: a global byte range `[base, bound)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub base: u64,
    pub bound: u64,
}

/// `r2_memory`: the store, blocks sorted by base, and the sync-only space.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Memory {
    pub blocks: *const Block,
    pub count: usize,
    /// The Neutrino granule in bytes.
    pub granule: u64,
    pub sync_only: *const ByteRange,
    pub sync_only_count: usize,
}

/// `r2_task`: one task.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Task {
    /// 1, 2, 3, ... in the order of emission.
    pub seq: u64,
    /// The cycle of the issue.
    pub cycle: u64,
    /// The task id.
    pub gid: u64,
    /// The Neutrino that owns the task.
    pub neutrino: u32,
    /// The executing PE (an index of the PE table); [`NO_PE`] without one.
    pub pe: u32,
    /// The descriptor bytes (Neutrino spec layout).
    pub desc: *const u8,
    pub desc_bytes: usize,
    /// The argument block of a Muon task, or null.
    pub args: *const u8,
    pub args_bytes: usize,
}

/// The `pe` of a task without a PE.
pub const NO_PE: u32 = u32::MAX;

/// `configure`: the memory, the PE names by index, the backend's entry of
/// the configuration as JSON, and the error buffer.
pub type ConfigureFn = unsafe extern "C" fn(
    this: *mut std::ffi::c_void,
    memory: *const Memory,
    pes: *const *const c_char,
    pe_count: usize,
    config: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> i32;

/// `apply`: one task, and the error buffer.
pub type ApplyFn = unsafe extern "C" fn(
    this: *mut std::ffi::c_void,
    task: *const Task,
    err: *mut c_char,
    err_len: usize,
) -> i32;

/// `close`: frees the backend. The blocks of the store are still mapped
/// during the call: a backend can write back its state.
pub type CloseFn = unsafe extern "C" fn(this: *mut std::ffi::c_void);

/// `r2_backend`: the backend that `r2_backend_open` returns.
#[repr(C)]
#[derive(Debug)]
pub struct Backend {
    pub abi: u32,
    pub this: *mut std::ffi::c_void,
    pub configure: ConfigureFn,
    pub apply: ApplyFn,
    pub close: CloseFn,
}

/// `r2_backend_open`.
pub type OpenFn = unsafe extern "C" fn(err: *mut c_char, err_len: usize) -> *const Backend;

/// Writes `msg` into the error buffer `err` of `len` bytes, truncated and
/// NUL-terminated.
///
/// # Safety
/// `err` must be null or point to `len` writable bytes.
pub unsafe fn set_err(err: *mut c_char, len: usize, msg: &str) {
    if err.is_null() || len == 0 {
        return;
    }
    let n = msg.len().min(len - 1);
    std::ptr::copy_nonoverlapping(msg.as_ptr(), err.cast::<u8>(), n);
    *err.add(n) = 0;
}

/// The bytes `[addr, addr + len)` of `memory`, or null when they are not
/// in one block or the block has no data (`r2_at`).
///
/// # Safety
/// `memory` must describe a valid block table.
pub unsafe fn at(memory: &Memory, addr: u64, len: u64) -> *mut u8 {
    let blocks = std::slice::from_raw_parts(memory.blocks, memory.count);
    let i = blocks.partition_point(|b| b.base <= addr);
    if i == 0 {
        return std::ptr::null_mut();
    }
    let b = &blocks[i - 1];
    let off = addr - b.base;
    if b.ptr.is_null() || off > b.size || len > b.size - off {
        return std::ptr::null_mut();
    }
    b.ptr.add(off as usize)
}
