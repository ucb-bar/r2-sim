//! Descriptors of ABI version 1, as an RV program builds them in its data
//! scratch: a 24-byte header (opcode u16, pe u16, flags u16, ndeps u16,
//! dst base u64 in granules, bound u32 in granules, mask u32), `ndeps`
//! keys of 8 bytes, then the arguments of the opcode. All fields are
//! little-endian; addresses in the arguments are byte addresses.

use nu_model::{Addr, NeutrinoId, Range};
use sim_core::{Args, Descriptor, Dtype, OpcodeId};

use crate::Abi;

/// Opcodes.
pub const OP_JOIN: u16 = 0;
pub const OP_SPAWN: u16 = 1;
pub const OP_DMA: u16 = 2;
pub const OP_MATMUL: u16 = 3;
pub const OP_VECTOR: u16 = 4;
pub const OP_IMEM_COPY: u16 = 5;
/// The PE field of an IMEM_COPY: the instruction DMA of the invoking node.
pub const PE_IMEM: u16 = 0xffff;
const FLAG_VERSIONED: u16 = 1;
const FLAG_UNTRACKED: u16 = 2;
const MM_ACCUMULATE: u32 = 1;
const MM_KEEP: u32 = 2;
const HEADER: u64 = 24;

/// Bytes of the arguments of each opcode, by opcode.
pub const ARGS_BYTES: [u64; 6] = [0, 16, 32, 72, 32, 24];

/// A decoded descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decoded {
    /// A task: JOIN, DMA, MATMUL or VECTOR.
    Task(Descriptor),
    /// SPAWN: start `entry` on direct child `child` (its index among the
    /// direct children) in epoch `epoch` with the arguments at `args`;
    /// `done` completes when the child's program returns.
    Spawn {
        done: Range,
        deps: Vec<nu_model::Dep>,
        child: u16,
        epoch: u16,
        entry: u32,
        args: u64,
    },
    /// IMEM_COPY: `bytes` from offset `src` of the invoker's IMEM to
    /// offset `offset` of the IMEM of direct child `child`; `record`
    /// completes with the copy.
    ImemCopy {
        record: Range,
        deps: Vec<nu_model::Dep>,
        child: u16,
        src: u64,
        offset: u32,
        bytes: u64,
    },
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"))
}

fn dtype(code: u32) -> Result<Dtype, String> {
    match code {
        0 => Ok(Dtype::Bf16),
        1 => Ok(Dtype::F32),
        2 => Ok(Dtype::Fp8),
        3 => Ok(Dtype::Mxfp4),
        c => Err(format!("unknown matmul element type {c}")),
    }
}

/// Decodes the descriptor at global address `addr`, in the data scratch
/// `scratch` (its bytes) at global range `at`, that node `invoker` invokes
/// with `nu.invoke` at `pc`. The PE of a task is an index into the PEs of
/// the owner of its `dst` (of the invoker for an untracked task).
pub fn decode(
    scratch: &[u8],
    at: Range,
    addr: u64,
    invoker: NeutrinoId,
    pc: u64,
    abi: &Abi,
) -> Result<Decoded, String> {
    let fits = |len: u64| addr >= at.base && addr.checked_add(len).is_some_and(|e| e <= at.bound);
    if !fits(HEADER) {
        return Err(format!(
            "descriptor at {addr:#x} outside the own data scratch {at}"
        ));
    }
    let h = &scratch[(addr - at.base) as usize..];
    let (opcode, pe, flags, ndeps) = (u16_at(h, 0), u16_at(h, 2), u16_at(h, 4), u16_at(h, 6));
    let Some(&args_bytes) = ARGS_BYTES.get(opcode as usize) else {
        return Err(format!("descriptor at {addr:#x}: unknown opcode {opcode}"));
    };
    let len = HEADER + 8 * ndeps as u64 + args_bytes;
    if !fits(len) {
        return Err(format!(
            "descriptor at {addr:#x} ({len} bytes) outside the own data scratch {at}"
        ));
    }
    let shift = abi.granule_shift;
    let base = u64_at(h, 8);
    let dst = Range::from_len(base << shift, (u32_at(h, 16) as u64) << shift);
    let mask = u32_at(h, 20);
    let deps: Vec<nu_model::Dep> = (0..ndeps as usize)
        .map(|i| abi.dep(u64_at(h, 24 + 8 * i)))
        .collect();
    let a = &h[(HEADER + 8 * ndeps as u64) as usize..len as usize];
    let (versioned, untracked) = (flags & FLAG_VERSIONED != 0, flags & FLAG_UNTRACKED != 0);
    let task = |opcode: OpcodeId, pe: Option<sim_core::PeId>, args: Args| {
        Decoded::Task(Descriptor {
            opcode,
            pe,
            dst,
            versioned,
            untracked,
            dst_mask: mask,
            inferred_version: None,
            deps: deps.clone(),
            label: None,
            args,
            pc: Some(pc),
            call_path: Vec::new(),
        })
    };
    // The PE: an index into the PEs of the node it runs on.
    let pe_of = || -> Result<sim_core::PeId, String> {
        let node = if untracked {
            invoker
        } else {
            abi.owner_of(dst.base).ok_or_else(|| {
                format!("descriptor at {addr:#x}: dst {dst} is in no node's own space")
            })?
        };
        let pes = &abi.node(node).pes;
        pes.get(pe as usize).copied().ok_or_else(|| {
            format!(
                "descriptor at {addr:#x}: PE {pe} of {}, which has {} PEs",
                abi.node(node).path,
                pes.len()
            )
        })
    };
    Ok(match opcode {
        OP_JOIN => task(OpcodeId::JOIN, None, Args::Join),
        OP_DMA => task(
            OpcodeId::DMA,
            Some(pe_of()?),
            Args::Dma {
                src: Addr(u64_at(a, 0)),
                dst: Addr(u64_at(a, 8)),
                bytes: u64_at(a, 16),
                tile_bytes: u64_at(a, 24),
            },
        ),
        OP_MATMUL => {
            let types = u32_at(a, 56);
            let mmflags = u32_at(a, 60);
            let opt = |x: u64| (x != 0).then_some(Addr(x));
            task(
                OpcodeId::MATMUL,
                Some(pe_of()?),
                Args::Matmul {
                    a: Addr(u64_at(a, 0)),
                    b: Addr(u64_at(a, 8)),
                    a2: opt(u64_at(a, 16)),
                    b2: opt(u64_at(a, 24)),
                    lda: u32_at(a, 32) as u64,
                    ldb: u32_at(a, 36) as u64,
                    ldc: u32_at(a, 40) as u64,
                    m: u32_at(a, 44) as u64,
                    n: u32_at(a, 48) as u64,
                    k: u32_at(a, 52) as u64,
                    in_dtype: dtype(types & 0xff)?,
                    b_dtype: dtype((types >> 8) & 0xff)?,
                    out_dtype: dtype((types >> 16) & 0xff)?,
                    accumulate: mmflags & MM_ACCUMULATE != 0,
                    keep: mmflags & MM_KEEP != 0,
                    tile: [u16_at(a, 64) as u64, u16_at(a, 66) as u64],
                },
            )
        }
        OP_VECTOR => task(
            OpcodeId::VECTOR,
            Some(pe_of()?),
            Args::Vector {
                ops: u64_at(a, 0),
                bytes_in: u64_at(a, 8),
                bytes_out: u64_at(a, 16),
                src: Some(u64_at(a, 24)).filter(|&s| s != 0).map(Addr),
            },
        ),
        OP_SPAWN => Decoded::Spawn {
            done: dst,
            deps,
            child: u16_at(a, 0),
            epoch: u16_at(a, 2),
            entry: u32_at(a, 4),
            args: u64_at(a, 8),
        },
        OP_IMEM_COPY => {
            if pe != PE_IMEM {
                return Err(format!(
                    "descriptor at {addr:#x}: IMEM_COPY on PE {pe}, not the instruction DMA {PE_IMEM:#x}"
                ));
            }
            Decoded::ImemCopy {
                record: dst,
                deps,
                child: u16_at(a, 8),
                src: u64_at(a, 0),
                offset: u32_at(a, 12),
                bytes: u64_at(a, 16),
            }
        }
        _ => unreachable!("ARGS_BYTES has an entry per opcode"),
    })
}
