//! Functional backends (r2sim spec Section 3.1).
//!
//! * [`Builtin`]: executes tasks on the store of the functional plane
//!   (DMA copy, GEMM).
//! * [`Plugin`]: a backend in a plug-in library with the C interface of
//!   `include/r2_backend.h` (for example the Python backend,
//!   `libr2_python.so`, or a trace backend).
//!
//! The store and `Null` live in `sim-core`.

mod builtin;
mod dtype;
pub mod plugin;

pub use builtin::Builtin;
pub use dtype::{bf16_to_f32, f32_to_bf16};
pub use plugin::Plugin;
