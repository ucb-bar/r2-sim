//! Functional backends (r2sim spec Section 3.1).
//!
//! * [`Numeric`]: executes ops on byte memory (DMA copy, GEMM, stores).
//! * [`Trace`]: the same memory without compute; labeled tasks write
//!   recorded outputs (router decisions).
//! * [`Process`]: a plane in a child process (for example the Python plane,
//!   `python -m r2sim.func`), over a framed binary protocol.
//!
//! Memory is 64 KiB pages per memory block, allocated on first write, zero
//! when never written; writes into sync-only ranges are ignored. `Null`
//! lives in `sim-core`.

mod dtype;
mod memory;
mod numeric;
pub mod process;
mod trace;

pub use dtype::{bf16_to_f32, f32_to_bf16};
pub use numeric::Numeric;
pub use process::Process;
pub use trace::{decode_trace, encode_trace, load_trace, save_trace, Records, Trace};
