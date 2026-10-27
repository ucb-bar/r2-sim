//! The Trace backend (r2sim spec Section 3.1): Numeric memory without
//! compute. A DMA copies bytes and a store writes them; a compute task
//! writes nothing. Loads read memory, as with Numeric.

use nu_model::{Addr, Range};
use sim_core::{Args, FuncError, FuncOp, FunctionalBackend, MemBlock, OpKind};

use crate::memory::Memory;

/// Numeric memory, no compute.
#[derive(Default)]
pub struct Trace {
    mem: Memory,
}

impl Trace {
    /// Builds a backend with empty memory.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FunctionalBackend for Trace {
    fn configure(&mut self, memories: &[MemBlock], sync_only: &[Range], granule: u64) {
        self.mem.configure(memories, sync_only, granule);
    }

    fn has_data(&self) -> bool {
        true
    }

    fn apply(&mut self, op: &FuncOp) -> Result<(), FuncError> {
        match &op.kind {
            OpKind::Store { addr, bytes } => self.mem.write(addr.0, bytes),
            OpKind::Task(desc) => match desc.args {
                Args::Dma {
                    src, dst, bytes, ..
                } => self.mem.dma(&desc.held(), src.0, dst.0, bytes),
                _ => Ok(()),
            },
        }
    }

    fn read(&mut self, addr: Addr, len: u64) -> Result<Vec<u8>, FuncError> {
        self.mem.read(addr.0, len)
    }

    fn write(&mut self, addr: Addr, bytes: &[u8]) -> Result<(), FuncError> {
        self.mem.write(addr.0, bytes)
    }
}
