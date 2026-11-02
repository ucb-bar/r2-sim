//! Functional backends (r2sim spec Section 3.1).
//!
//! * [`Numeric`]: executes tasks on the store of the functional plane
//!   (DMA copy, GEMM).
//! * [`Trace`]: DMA copies only.
//! * [`Process`]: a plane in a child process that keeps its own memory
//!   (for example the Python plane, `python -m r2sim.func`), over a framed
//!   binary protocol: an external plane for the whole run.
//!
//! The store and `Null` live in `sim-core`.

mod dtype;
mod numeric;
pub mod process;
mod trace;

pub use dtype::{bf16_to_f32, f32_to_bf16};
pub use numeric::Numeric;
pub use process::Process;
pub use trace::Trace;
