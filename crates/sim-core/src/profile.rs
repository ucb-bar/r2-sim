//! Descriptors and demand profiles (architecture Section 5.5).
//!
//! Every leaf opcode registers one function from a descriptor and the node
//! configuration to a [`Profile`]: the chunks of the task, each with a
//! demand vector and a latency, and the occupancy held from issue to
//! completion. A profile never reads functional data.

use nu_model::{Addr, Dep, NeutrinoId, Range};

use crate::config::{MemBlock, PeConfig, PeParams};
use crate::resource::{Demand, ResourceId};
use crate::time::{Cycle, Ratio};
use crate::topology::Topology;

/// Index of an opcode in the [`ProfileRegistry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct OpcodeId(pub u16);

impl OpcodeId {
    /// The dummy task. It has no PE and no profile; the Neutrino charges
    /// only its own cost.
    pub const JOIN: OpcodeId = OpcodeId(0);
    /// `spawn_child`: a task in the parent's tables that completes when the
    /// child program returns. No PE, no profile.
    pub const SPAWN: OpcodeId = OpcodeId(1);
}

/// Index of a PE or DMA engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub struct PeId(pub u32);

/// Request data of a descriptor. The resource plane reads it to derive
/// demand; it carries shapes, never data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Args {
    /// Copy `bytes` from `src` to `dst` in chunks of `tile_bytes`.
    Dma {
        /// Source address.
        src: Addr,
        /// Destination address.
        dst: Addr,
        /// Bytes to copy.
        bytes: u64,
        /// Bytes per chunk (greater than 0).
        tile_bytes: u64,
    },
    /// `C[m, n] (+)= A[m, k] * B[k, n]`, row-major with leading dimensions
    /// in elements; C is at `dst.base`. One chunk per `tile[0] x tile[1]`
    /// output tile over the full `k`.
    Matmul {
        /// Address of A.
        a: Addr,
        /// Address of B.
        b: Addr,
        /// Leading dimension of A, in elements.
        lda: u64,
        /// Leading dimension of B, in elements.
        ldb: u64,
        /// Leading dimension of C, in elements.
        ldc: u64,
        /// Rows of A and C.
        m: u64,
        /// Columns of B and C.
        n: u64,
        /// Columns of A and rows of B.
        k: u64,
        /// Element type of A.
        in_dtype: Dtype,
        /// Element type of B (weights may be quantized; MACs do not change).
        b_dtype: Dtype,
        /// Element type of C.
        out_dtype: Dtype,
        /// `C = C + A * B` instead of `C = A * B`.
        accumulate: bool,
        /// Leave C in the Atlas accumulator for the next accumulate task on
        /// the same dst (H3); without an accumulator memory, no effect.
        keep: bool,
        /// Output tile shape `[rows, columns]`: one chunk per tile.
        tile: [u64; 2],
    },
    /// No request data (`join`).
    Join,
    /// A vector task on a Muon: `ops` lane-ops, reading `bytes_in` and
    /// writing `bytes_out` in SMEM.
    Vector {
        /// Lane-ops of the task.
        ops: u64,
        /// Bytes read, from the block that holds `src`.
        bytes_in: u64,
        /// Bytes written, to the block that holds `dst`.
        bytes_out: u64,
        /// The address the task reads; `None`: the block of `dst`.
        src: Option<Addr>,
    },
    /// Start a program on the child Neutrino.
    Spawn {
        /// The child Neutrino that runs the program.
        child: NeutrinoId,
    },
}

/// Element types of the numeric operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// bfloat16, 2 B per element.
    Bf16,
    /// 32-bit float, 4 B per element.
    F32,
    /// 8-bit float (E4M3), 1 B per element.
    Fp8,
    /// OCP MX FP4 (E2M1): 0.5 B per element plus one 1 B E8M0 scale per
    /// block of 32 elements.
    Mxfp4,
}

impl Dtype {
    /// Elements per MX scale block.
    pub const MX_BLOCK: u64 = 32;

    /// Bytes per element. Panics for `Mxfp4`, which has no whole-byte
    /// element size; use `bytes_for`.
    pub fn bytes(self) -> u64 {
        match self {
            Dtype::Bf16 => 2,
            Dtype::F32 => 4,
            Dtype::Fp8 => 1,
            Dtype::Mxfp4 => panic!("mxfp4 has no whole-byte element size"),
        }
    }

    /// Bytes of `n` elements, scales included (rounded up).
    pub fn bytes_for(self, n: u64) -> u64 {
        match self {
            Dtype::Mxfp4 => n.div_ceil(2) + n.div_ceil(Self::MX_BLOCK),
            d => n * d.bytes(),
        }
    }
}

/// A task descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    /// Selects the profile function.
    pub opcode: OpcodeId,
    /// The executing PE. `None` only for `join` and `spawn`.
    pub pe: Option<PeId>,
    /// The range the task writes; later lookups of it depend on the task.
    pub dst: Range,
    /// `dst` is a versioned data range (Section 3.2, Versions); false for
    /// a signal in the sync-only space.
    pub versioned: bool,
    /// Simulator check, not hardware state: the version the program infers
    /// for this invoke. In debug builds the owner compares it with the
    /// version it assigns and stops the run on a mismatch.
    pub inferred_version: Option<u64>,
    /// The task's dependencies: it is ready when their producers complete.
    pub deps: Vec<Dep>,
    /// Software-assigned identity of this invocation; no hardware meaning.
    /// The tracer names tasks by it and the Trace backend keys records by it.
    pub label: Option<u64>,
    /// Opcode-specific request data.
    pub args: Args,
}

impl Descriptor {
    /// A `join` task on `dst` that depends on `deps`.
    pub fn join(dst: Range, deps: Vec<Dep>) -> Self {
        Self {
            opcode: OpcodeId::JOIN,
            pe: None,
            dst,
            versioned: false,
            inferred_version: None,
            deps,
            label: None,
            args: Args::Join,
        }
    }

    /// Sets `label`.
    pub fn with_label(mut self, label: u64) -> Self {
        self.label = Some(label);
        self
    }

    /// Marks `dst` as a versioned data range.
    pub fn versioned(mut self) -> Self {
        self.versioned = true;
        self
    }

    /// Bytes of the descriptor in an `Invoke` message (Section 3.3):
    /// opcode 2, PE 2, `dst` base and length 16, 8 per dependency (address
    /// and version packed in 64 bits), and the request data (DMA 32,
    /// matmul 52, vector 24, join 0, spawn 4). The label is not sent.
    pub fn wire_bytes(&self) -> u64 {
        let args = match self.args {
            Args::Dma { .. } => 32,
            Args::Matmul { .. } => 52,
            Args::Vector { src: None, .. } => 24,
            Args::Vector { src: Some(_), .. } => 32,
            Args::Join => 0,
            Args::Spawn { .. } => 4,
        };
        2 + 2 + 16 + 8 * self.deps.len() as u64 + args
    }

    /// Marks `dst` as versioned and records the version the program
    /// infers for it ([`Descriptor::inferred_version`]).
    pub fn with_inferred_version(mut self, v: u64) -> Self {
        self.versioned = true;
        self.inferred_version = Some(v);
        self
    }
}

/// Occupancy held from issue to completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occupancy {
    /// The occupancy resource.
    pub resource: ResourceId,
    /// Units held.
    pub amount: u64,
}

/// One timing-only subdivision of a task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Co-charged over one interval (Section 5.4).
    pub demand: Vec<Demand>,
    /// Cycles after the request before the demand may start (Section 5.7).
    pub latency: Cycle,
}

/// The timing of one task: its chunks and the occupancy it holds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Profile {
    /// Chunks in request order; with none, the task completes at issue.
    pub chunks: Vec<Chunk>,
    /// Occupancy held from issue to completion.
    pub holds: Vec<Occupancy>,
}

/// What a profile function sees of the hardware.
#[derive(Debug, Clone, Copy)]
pub struct NodeConfig<'a> {
    /// The executing PE.
    pub pe: &'a PeConfig,
    /// All memory blocks.
    pub memories: &'a [MemBlock],
    /// The fabric topology.
    pub topology: &'a Topology,
    /// Atlas accumulator memory (H3) decisions for this task.
    pub acc: AccFlags,
}

/// What the Atlas accumulator memory does for one task (H3): the C read
/// comes from it, the C write stays in it, and bytes of evicted tiles are
/// written back to SMEM before the task.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AccFlags {
    /// Read C from the accumulator memory instead of SMEM.
    pub skip_read: bool,
    /// Keep C in the accumulator memory instead of writing it to SMEM.
    pub skip_write: bool,
    /// Bytes of evicted tiles written back to SMEM, charged on the first
    /// chunk.
    pub flush_bytes: u64,
}

impl NodeConfig<'_> {
    /// The memory block that holds `addr`.
    pub fn memory(&self, addr: Addr) -> &MemBlock {
        self.memories
            .iter()
            .find(|m| m.range.contains(addr))
            .unwrap_or_else(|| panic!("address {:#x} is in no memory block", addr.0))
    }
}

/// A profile function: from a descriptor and the node configuration to a
/// [`Profile`].
pub type ProfileFn = fn(&Descriptor, &NodeConfig) -> Profile;

/// Opcode -> profile function. Opcodes 0 (`join`) and 1 (`spawn`) have
/// none.
#[derive(Debug, Clone)]
pub struct ProfileRegistry {
    fns: Vec<(String, Option<ProfileFn>)>,
}

impl Default for ProfileRegistry {
    fn default() -> Self {
        Self {
            fns: vec![("join".to_string(), None), ("spawn".to_string(), None)],
        }
    }
}

impl ProfileRegistry {
    /// The registry with the phase-1 profiles: `dma`, `atlas.matmul` and
    /// `muon.vector`.
    pub fn with_builtins() -> Self {
        let mut r = Self::default();
        r.register("dma", dma_profile);
        r.register("atlas.matmul", matmul_profile);
        r.register("muon.vector", vector_profile);
        r
    }

    /// Registers `f` under `name` and returns its opcode. Panics if `name` is
    /// already registered.
    pub fn register(&mut self, name: &str, f: ProfileFn) -> OpcodeId {
        assert!(
            self.lookup(name).is_none(),
            "opcode {name} registered twice"
        );
        self.fns.push((name.to_string(), Some(f)));
        OpcodeId(self.fns.len() as u16 - 1)
    }

    /// The opcode registered under `name`.
    pub fn lookup(&self, name: &str) -> Option<OpcodeId> {
        self.fns
            .iter()
            .position(|(n, _)| n == name)
            .map(|i| OpcodeId(i as u16))
    }

    /// The profile of `desc` on `node`. Panics for an opcode without a
    /// profile function.
    pub fn profile(&self, desc: &Descriptor, node: &NodeConfig) -> Profile {
        let f = self
            .fns
            .get(desc.opcode.0 as usize)
            .and_then(|(_, f)| *f)
            .unwrap_or_else(|| panic!("opcode {:?} has no profile", desc.opcode));
        f(desc, node)
    }
}

/// DMA: one chunk per `tile_bytes`; bandwidth on the source port, the
/// engine's extra links, the fabric route from the source block to the
/// engine and from the engine to the destination block (each over the
/// engine's master bus with the fewest bridges), and the destination port.
/// Chunk latency is the engine latency plus the two route latencies.
pub(crate) fn dma_profile(desc: &Descriptor, node: &NodeConfig) -> Profile {
    let Args::Dma {
        src,
        dst,
        bytes,
        tile_bytes,
    } = desc.args
    else {
        panic!("dma with args {:?}", desc.args);
    };
    let PeParams::Dma {
        links,
        latency,
        masters,
    } = &node.pe.params
    else {
        panic!("dma opcode on PE {}", node.pe.name);
    };
    assert!(tile_bytes > 0, "dma with tile_bytes 0");
    let (sm, dm) = (node.memory(src), node.memory(dst));
    let (src_port, dst_port) = (sm.port, dm.port);
    let route =
        |from: &[crate::topology::AttachId], to: &[crate::topology::AttachId], what: &str| {
            node.topology
                .best_path(from, to)
                .unwrap_or_else(|| panic!("{}: no route {what}", node.pe.name))
        };
    let (inp, out) = (
        route(&sm.attach, masters, "from the source"),
        route(masters, &dm.attach, "to the destination"),
    );
    let mut chunks = Vec::with_capacity(bytes.div_ceil(tile_bytes) as usize);
    let mut left = bytes;
    while left > 0 {
        let n = left.min(tile_bytes);
        left -= n;
        let mut demand = vec![Demand::new(src_port, n)];
        demand.extend(links.iter().map(|l| Demand::new(*l, n)));
        demand.extend(inp.resources.iter().map(|r| Demand::new(*r, n)));
        demand.extend(out.resources.iter().map(|r| Demand::new(*r, n)));
        demand.push(Demand::new(dst_port, n));
        chunks.push(Chunk {
            demand,
            latency: *latency + inp.latency + out.latency,
        });
    }
    Profile {
        chunks,
        holds: Vec::new(),
    }
}

/// Atlas matmul: one chunk per output tile; tensor ops on the array and
/// operand plus C bytes on the SMEM port.
pub(crate) fn matmul_profile(desc: &Descriptor, node: &NodeConfig) -> Profile {
    let Args::Matmul {
        m,
        n,
        k,
        in_dtype,
        b_dtype,
        out_dtype,
        accumulate,
        tile,
        ..
    } = desc.args
    else {
        panic!("matmul with args {:?}", desc.args);
    };
    let PeParams::Atlas {
        array,
        smem_port,
        utilization,
        quantize,
        row_panel,
        ref rates,
        ..
    } = node.pe.params
    else {
        panic!("matmul opcode on PE {}", node.pe.name);
    };
    // The slower operand type sets the rate: the array charges
    // MACs * max / rate, as a utilization of rate / max.
    let utilization = match rates.iter().map(|r| r.1).max_by(ratio_cmp) {
        None => utilization,
        Some(max) => {
            let rate = |d: Dtype| {
                rates
                    .iter()
                    .find(|r| r.0 == d)
                    .unwrap_or_else(|| panic!("PE {} has no {d:?} rate", node.pe.name))
                    .1
            };
            let r = [rate(in_dtype), rate(b_dtype)]
                .into_iter()
                .min_by(ratio_cmp)
                .expect("two rates");
            Ratio::new(
                utilization.num * r.num * max.den,
                utilization.den * r.den * max.num,
            )
        }
    };
    let acc = node.acc;
    let mut chunks = Vec::new();
    for mi in (0..m).step_by(tile[0] as usize) {
        let tm = tile[0].min(m - mi);
        for ni in (0..n).step_by(tile[1] as usize) {
            let tn = tile[1].min(n - ni);
            // Quantized: the array works on whole tiles.
            let (qm, qn) = if quantize {
                (tile[0], tile[1])
            } else {
                (tm, tn)
            };
            let mut ops = Demand::new(array, qm * qn * k);
            ops.utilization = utilization;
            // Operands in, C out, and C in when accumulating. C is exact.
            let c_bytes = tm * tn * out_dtype.bytes();
            // With a row panel, A is read at the first output tile of the row.
            let a_bytes = if row_panel && ni > 0 {
                0
            } else {
                in_dtype.bytes_for(qm * k)
            };
            let c_out = if acc.skip_write { 0 } else { c_bytes };
            let c_in = if accumulate && !acc.skip_read {
                c_bytes
            } else {
                0
            };
            let flush = if chunks.is_empty() {
                acc.flush_bytes
            } else {
                0
            };
            let bytes = a_bytes + b_dtype.bytes_for(k * qn) + c_out + c_in + flush;
            chunks.push(Chunk {
                demand: vec![ops, Demand::new(smem_port, bytes)],
                latency: 0,
            });
        }
    }
    Profile {
        chunks,
        holds: Vec::new(),
    }
}

/// Muon vector task: one chunk; lane-ops on the vector resource at the
/// configured utilization, plus the task's bytes on the port of the block
/// that holds `dst` and on the route to it, after the fixed overhead and
/// the route latency.
pub(crate) fn vector_profile(desc: &Descriptor, node: &NodeConfig) -> Profile {
    let Args::Vector {
        ops,
        bytes_in,
        bytes_out,
        src,
    } = desc.args
    else {
        panic!("vector op with args {:?}", desc.args);
    };
    let PeParams::Muon {
        vector,
        ref masters,
        utilization,
        overhead,
    } = node.pe.params
    else {
        panic!("vector opcode on PE {}", node.pe.name);
    };
    // The blocks it reads and writes: the SM's SMEM over lbus, a remote
    // range over the SM's data bus. A resource on both routes is charged
    // once for both byte counts.
    let mut v = Demand::new(vector, ops);
    v.utilization = utilization;
    let mut demand = vec![v];
    let mut latency = 0;
    let out_block = node.memory(Addr(desc.dst.base));
    let in_block = src.map_or(out_block, |a| node.memory(a));
    for (block, bytes, to) in [(in_block, bytes_in, false), (out_block, bytes_out, true)] {
        let route = if to {
            node.topology.best_path(masters, &block.attach)
        } else {
            node.topology.best_path(&block.attach, masters)
        }
        .unwrap_or_else(|| panic!("{}: no route to {}", node.pe.name, block.name));
        latency = latency.max(route.latency);
        for r in route.resources.iter().chain([&block.port]) {
            match demand.iter_mut().find(|d| d.resource == *r) {
                Some(d) => d.amount += bytes,
                None => demand.push(Demand::new(*r, bytes)),
            }
        }
    }
    Profile {
        chunks: vec![Chunk {
            demand,
            latency: overhead + latency,
        }],
        holds: Vec::new(),
    }
}

fn ratio_cmp(a: &Ratio, b: &Ratio) -> std::cmp::Ordering {
    (a.num as u128 * b.den as u128).cmp(&(b.num as u128 * a.den as u128))
}
