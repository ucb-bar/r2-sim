//! Functional backends (architecture Section 4.4).
//!
//! * [`Numeric`]: executes ops on byte memory (DMA copy, GEMM, stores).
//! * [`Trace`]: the same memory without compute; labeled tasks write
//!   recorded outputs (router decisions).
//!
//! Memory is 64 KiB pages per memory block, allocated on first write, zero
//! when never written; writes into sync-only ranges are ignored. `Null`
//! lives in `sim-core`.

mod dtype;
mod memory;
mod numeric;
mod trace;

pub use dtype::{bf16_to_f32, f32_to_bf16};
pub use numeric::Numeric;
pub use trace::{decode_trace, encode_trace, load_trace, save_trace, Records, Trace};
