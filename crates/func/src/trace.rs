//! The Trace backend (r2sim spec Section 3.1): a DMA copies bytes on the
//! store; a compute task writes nothing.

use sim_core::{Args, FuncError, FuncOp, FunctionalBackend, OpKind, Store};

/// DMA copies, no compute.
#[derive(Default)]
pub struct Trace;

impl Trace {
    /// Builds the backend.
    pub fn new() -> Self {
        Self
    }
}

impl FunctionalBackend for Trace {
    fn name(&self) -> &'static str {
        "trace"
    }

    fn apply(&mut self, op: &FuncOp, mem: &mut Store) -> Result<(), FuncError> {
        match &op.kind {
            OpKind::Store { addr, bytes } => mem.write(addr.0, bytes),
            OpKind::Task(desc) => match desc.args {
                Args::Dma {
                    src, dst, bytes, ..
                } => mem.dma(&desc.held(), src.0, dst.0, bytes),
                _ => Ok(()),
            },
        }
    }
}
