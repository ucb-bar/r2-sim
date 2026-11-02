//! Functional backends (r2sim spec Section 3.1).
//!
//! * [`Numeric`]: executes tasks on the store of the functional plane
//!   (DMA copy, GEMM).
//! * [`Trace`]: DMA copies only.
//! * [`Plugin`]: a backend in a plug-in library with the C interface of
//!   `include/r2_backend.h` (for example the Python backend,
//!   `libr2_python.so`).
//!
//! The store and `Null` live in `sim-core`.

mod dtype;
mod numeric;
pub mod plugin;
mod trace;

pub use dtype::{bf16_to_f32, f32_to_bf16};
pub use numeric::Numeric;
pub use plugin::Plugin;
pub use trace::Trace;
