//! A functional backend plug-in for the tests of the loader. Its entry of
//! the configuration (JSON) has:
//! * `log`: a file; each task appends one line `seq cycle gid neutrino pe
//!   desc`, the descriptor bytes in hex, and ` args <hex>` for a task with
//!   an argument block;
//! * `fail`: `configure` or `apply` makes that call fail with a message;
//! * `close_read`: `close` reads the first byte of the first block with
//!   data and logs it (`close <byte>`): the blocks are still mapped then.
//!
//! A DMA task (opcode 2) copies its bytes on the store through the block
//! table, so a test can see the data move. `r2_test_set_abi` changes the
//! ABI version that the next `r2_backend_open` returns.
//!
//! The rlib also gives the C layout of include/r2_backend.h ([`c_layout`])
//! and the C lookup `r2_at` ([`c_at`]).

use std::ffi::{c_char, c_void, CStr};
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};

use r2_backend::{set_err, Backend, Memory, Task, ABI};

static OPEN_ABI: AtomicU32 = AtomicU32::new(ABI);

/// Sets the ABI version that the next `r2_backend_open` returns.
#[no_mangle]
pub extern "C" fn r2_test_set_abi(abi: u32) {
    OPEN_ABI.store(abi, Ordering::SeqCst);
}

struct State {
    log: Option<std::fs::File>,
    fail_apply: bool,
    close_read: bool,
    memory: Option<Memory>,
}

/// # Safety
/// `err` must point to `err_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn r2_backend_open(_err: *mut c_char, _err_len: usize) -> *const Backend {
    let state = Box::new(State {
        log: None,
        fail_apply: false,
        close_read: false,
        memory: None,
    });
    Box::into_raw(Box::new(Backend {
        abi: OPEN_ABI.load(Ordering::SeqCst),
        this: Box::into_raw(state).cast(),
        configure,
        apply,
        close,
    }))
}

unsafe extern "C" fn configure(
    this: *mut c_void,
    memory: *const Memory,
    pes: *const *const c_char,
    pe_count: usize,
    config: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    let s = &mut *this.cast::<State>();
    let text = CStr::from_ptr(config).to_string_lossy();
    let cfg: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            set_err(err, err_len, &format!("config: {e}"));
            return 1;
        }
    };
    match cfg["fail"].as_str() {
        Some("configure") => {
            set_err(err, err_len, "the test backend fails at configure");
            return 1;
        }
        Some("apply") => s.fail_apply = true,
        _ => {}
    }
    s.close_read = cfg["close_read"].as_bool() == Some(true);
    if let Some(p) = cfg["log"].as_str() {
        match std::fs::File::create(p) {
            Ok(mut f) => {
                let names: Vec<String> = (0..pe_count)
                    .map(|i| CStr::from_ptr(*pes.add(i)).to_string_lossy().into_owned())
                    .collect();
                let _ = writeln!(f, "pes {}", names.join(" "));
                s.log = Some(f);
            }
            Err(e) => {
                set_err(err, err_len, &format!("{p}: {e}"));
                return 1;
            }
        }
    }
    s.memory = Some(*memory);
    0
}

unsafe extern "C" fn apply(
    this: *mut c_void,
    task: *const Task,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    let s = &mut *this.cast::<State>();
    let t = &*task;
    if s.fail_apply {
        set_err(
            err,
            err_len,
            &format!("the test backend fails at seq {}", t.seq),
        );
        return 1;
    }
    let desc = std::slice::from_raw_parts(t.desc, t.desc_bytes);
    if let Some(f) = &mut s.log {
        let hex = |b: &[u8]| -> String { b.iter().map(|b| format!("{b:02x}")).collect() };
        let args = if t.args.is_null() {
            &[][..]
        } else {
            std::slice::from_raw_parts(t.args, t.args_bytes)
        };
        let _ = write!(
            f,
            "{} {} {} {} {} {}",
            t.seq,
            t.cycle,
            t.gid,
            t.neutrino,
            t.pe,
            hex(desc)
        );
        let _ = if args.is_empty() {
            writeln!(f)
        } else {
            writeln!(f, " args {}", hex(args))
        };
    }
    // A DMA: the header is 24 bytes and 8 per key; the arguments are src,
    // dst, bytes (a contiguous copy).
    if desc.len() >= 24 && u16::from_le_bytes([desc[0], desc[1]]) == 2 {
        let keys = u16::from_le_bytes([desc[6], desc[7]]) as usize;
        let a = &desc[24 + 8 * keys..];
        let word = |o: usize| u64::from_le_bytes(a[o..o + 8].try_into().unwrap());
        let (src, dst, n) = (word(0), word(8), word(16));
        let m = s.memory.as_ref().unwrap();
        let (from, to) = (r2_backend::at(m, src, n), r2_backend::at(m, dst, n));
        if !from.is_null() && !to.is_null() {
            std::ptr::copy(from, to, n as usize);
        }
    }
    0
}

unsafe extern "C" fn close(this: *mut c_void) {
    let mut s = Box::from_raw(this.cast::<State>());
    if !s.close_read {
        return;
    }
    let Some(m) = s.memory else { return };
    let blocks = std::slice::from_raw_parts(m.blocks, m.count);
    if let Some(b) = blocks.iter().find(|b| !b.ptr.is_null()) {
        let byte = std::ptr::read_volatile(b.ptr);
        if let Some(f) = &mut s.log {
            let _ = writeln!(f, "close {byte}");
        }
    }
}

extern "C" {
    fn r2_test_layout(out: *mut usize, n: usize) -> usize;
    fn r2_test_at(m: *const Memory, addr: u64, len: u64) -> *mut u8;
}

/// The sizes and field offsets of the C types, in the order of layout.c.
pub fn c_layout() -> Vec<usize> {
    let mut v = vec![0usize; 64];
    // SAFETY: the C function writes at most 64 values.
    let n = unsafe { r2_test_layout(v.as_mut_ptr(), v.len()) };
    v.truncate(n);
    v
}

/// `r2_at` of the header, compiled by the C compiler.
///
/// # Safety
/// `m` must describe a valid block table.
pub unsafe fn c_at(m: &Memory, addr: u64, len: u64) -> *mut u8 {
    r2_test_at(m, addr, len)
}
