//! The Trace backend (architecture Section 4.4) and the trace file format.
//!
//! Trace: Numeric memory without compute. A DMA copies bytes and a store
//! writes them. A task whose descriptor `label` has a record writes the
//! recorded bytes at `dst.base` (an error if they exceed `dst`). Any other
//! compute task writes nothing. Loads read memory, as with Numeric.
//!
//! File format, little-endian: the magic `R2TRACE1` (8 bytes), a `u64`
//! record count, then per record a `u64` label, a `u32` length and that many
//! bytes. Labels are unique.

use std::collections::BTreeMap;
use std::path::Path;

use nu_model::{Addr, Range};
use sim_core::{Args, FuncError, FuncOp, FunctionalBackend, MemBlock, OpKind};

use crate::memory::{err, Memory};

const MAGIC: &[u8; 8] = b"R2TRACE1";

/// Label -> recorded output bytes.
pub type Records = BTreeMap<u64, Vec<u8>>;

/// Parses a trace file image.
pub fn decode_trace(bytes: &[u8]) -> Result<Records, FuncError> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8], FuncError> {
        let s = bytes
            .get(at..at + n)
            .ok_or_else(|| FuncError(format!("trace truncated at byte {at}")))?;
        at += n;
        Ok(s)
    };
    if take(8)? != MAGIC {
        return err("not a trace file (bad magic)".into());
    }
    let count = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
    let mut out = Records::new();
    for _ in 0..count {
        let label = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        let len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let data = take(len)?.to_vec();
        if out.insert(label, data).is_some() {
            return err(format!("trace has label {label:#x} twice"));
        }
    }
    if at != bytes.len() {
        return err(format!("{} trailing bytes in trace", bytes.len() - at));
    }
    Ok(out)
}

/// Serializes records in label order.
pub fn encode_trace(records: &Records) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for (label, data) in records {
        out.extend_from_slice(&label.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
    }
    out
}

/// Reads and parses the trace file at `path`.
pub fn load_trace(path: &Path) -> Result<Records, FuncError> {
    let bytes =
        std::fs::read(path).map_err(|e| FuncError(format!("reading {}: {e}", path.display())))?;
    decode_trace(&bytes)
}

/// Writes `records` to the trace file at `path`.
pub fn save_trace(path: &Path, records: &Records) -> Result<(), FuncError> {
    std::fs::write(path, encode_trace(records))
        .map_err(|e| FuncError(format!("writing {}: {e}", path.display())))
}

/// Numeric memory, recorded outputs for labeled tasks, no compute.
#[derive(Default)]
pub struct Trace {
    mem: Memory,
    records: Records,
    hits: u64,
}

impl Trace {
    /// Builds a backend with empty memory that replays `records`.
    pub fn new(records: Records) -> Self {
        Self {
            mem: Memory::default(),
            records,
            hits: 0,
        }
    }

    /// Labeled tasks that found a record so far.
    pub fn hits(&self) -> u64 {
        self.hits
    }
}

impl FunctionalBackend for Trace {
    fn configure(&mut self, memories: &[MemBlock], sync_only: &[Range]) {
        self.mem.configure(memories, sync_only);
    }

    fn has_data(&self) -> bool {
        true
    }

    fn apply(&mut self, op: &FuncOp) -> Result<(), FuncError> {
        match &op.kind {
            OpKind::Store { addr, bytes } => self.mem.write(addr.0, bytes),
            OpKind::Task(desc) => {
                if let Args::Dma {
                    src, dst, bytes, ..
                } = desc.args
                {
                    return self.mem.dma(&desc.dst, src.0, dst.0, bytes);
                }
                let Some(data) = desc.label.and_then(|l| self.records.get(&l)) else {
                    return Ok(());
                };
                if data.len() as u64 > desc.dst.len() {
                    return err(format!(
                        "trace record {:#x} of {} B exceeds dst {}",
                        desc.label.unwrap_or(0),
                        data.len(),
                        desc.dst
                    ));
                }
                self.hits += 1;
                let data = data.clone();
                self.mem.write(desc.dst.base, &data)
            }
        }
    }

    fn read(&mut self, addr: Addr, len: u64) -> Result<Vec<u8>, FuncError> {
        self.mem.read(addr.0, len)
    }

    fn write(&mut self, addr: Addr, bytes: &[u8]) -> Result<(), FuncError> {
        self.mem.write(addr.0, bytes)
    }
}
