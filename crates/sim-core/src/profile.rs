//! Descriptors and demand profiles (architecture Section 5.5).
//!
//! Every leaf opcode registers one function from a descriptor and the node
//! configuration to a [`Profile`]: the chunks of the task, each with a
//! demand vector and a latency, and the occupancy held from issue to
//! completion. A profile never reads functional data.

use std::rc::Rc;

use nu_model::{Addr, AddressHash, Dep, NeutrinoId, Range};

use crate::config::{MemBlock, PeConfig, PeParams};
use crate::resource::{Demand, Dir, PathHop, ResourceId};
use crate::time::{Cycle, Ratio};
use crate::topology::{AttachId, Topology};
use crate::tracer::RouteUse;

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

/// One timing-only subdivision of a task: a compute chunk (`demand`,
/// co-charged from one start) or a transfer chunk (`path`, reserved hop by
/// hop in route order; Section 5.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Compute: co-charged over one interval.
    pub demand: Vec<Demand>,
    /// Transfer: its two route halves and bytes; `None` for a compute
    /// chunk.
    pub transfer: Option<TransferChunk>,
    /// Transfer: the master's beat rate, the start of the narrowing chain.
    pub rate: Ratio,
    /// Cycles after the request before the demand (or the first hop) may
    /// start (Section 5.7).
    pub latency: Cycle,
    /// The fabric routes the chunk's bytes take, for a tracer (the
    /// unlimited links of a route have no demand).
    pub routes: [Option<RouteUse>; 2],
}

/// One half of a transfer's route between a memory block and a DMA
/// engine (Section 5.3, routing function), computed once per engine and
/// block: the hops with no amount, and the half's transaction grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Half {
    /// Read half: the block's port, the route to the engine and the
    /// engine's links. Write half: the route from the engine and the
    /// block's write port. A latency before the route's first rated hop
    /// is on the hop before it.
    pub hops: Rc<[PathHop]>,
    /// Write half: the latency before its first rated hop, added to the
    /// last hop of the read half.
    pub lead: Cycle,
    /// The attachments the bytes move between, for a tracer.
    pub pair: Option<(AttachId, AttachId)>,
    /// The smallest `block_bytes` on the half's buses, links and port
    /// (`u64::MAX`: none); all are powers of two, so its multiples are
    /// every boundary of the half.
    pub grid: u64,
}

/// A transfer chunk: `bytes` over the read half then the write half.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferChunk {
    pub read: Rc<Half>,
    pub write: Rc<Half>,
    pub bytes: u64,
}

impl TransferChunk {
    /// The hops of the chunk in route order with its bytes, into `buf`.
    pub fn hops_into(&self, buf: &mut Vec<PathHop>) {
        buf.clear();
        buf.extend(self.read.hops.iter().map(|h| PathHop {
            amount: self.bytes,
            ..*h
        }));
        if let Some(last) = buf.last_mut() {
            last.latency += self.write.lead;
        }
        buf.extend(self.write.hops.iter().map(|h| PathHop {
            amount: self.bytes,
            ..*h
        }));
    }
}

/// The routing function of transfers (Section 5.3): memory blocks by
/// address, a hash link's linear range as one entry rewritten onto its
/// physical block in one step, and per DMA engine and block the two route
/// halves, in dense tables built once per run.
#[derive(Debug, Clone, Default)]
pub struct TransferRoutes {
    /// Blocks by range, sorted by base.
    blocks: Vec<(Range, usize)>,
    /// Hash links' rules by linear range, sorted by base.
    hashes: Vec<AddressHash>,
    nblocks: usize,
    /// `[pe * nblocks + block]`: the read and write halves; `None` for a
    /// PE that is not a DMA engine, or a block without a route.
    halves: Vec<Halves>,
}

/// The read and write halves of one engine and block.
type Halves = (Option<Rc<Half>>, Option<Rc<Half>>);

impl TransferRoutes {
    /// The tables for `pes` (by id), `memories` (by index) and the hash
    /// links' rules.
    pub fn build(
        pes: &[&PeConfig],
        memories: &[MemBlock],
        hashes: &[&AddressHash],
        topo: &Topology,
    ) -> Self {
        let mut blocks: Vec<(Range, usize)> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| (m.range, i))
            .collect();
        blocks.sort_by_key(|b| b.0.base);
        let mut hashes: Vec<AddressHash> = hashes.iter().map(|h| (*h).clone()).collect();
        hashes.sort_by_key(|h| h.range.base);
        let nblocks = memories.len();
        let mut halves = Vec::with_capacity(pes.len() * nblocks);
        for pe in pes {
            let PeParams::Dma { links, masters, .. } = &pe.params else {
                halves.extend((0..nblocks).map(|_| (None, None)));
                continue;
            };
            for m in memories {
                let hop = |resource, dir, latency| PathHop {
                    resource,
                    amount: 0,
                    dir,
                    latency,
                };
                let grid = |blocks: &[u64]| {
                    blocks
                        .iter()
                        .chain(m.block.as_ref())
                        .copied()
                        .min()
                        .unwrap_or(u64::MAX)
                };
                let read = topo.best_path(&m.attach, masters).map(|p| {
                    let mut hops = vec![hop(m.port, Dir::Read, p.lead)];
                    hops.extend(p.hops.iter().map(|h| hop(h.lane, Dir::Read, h.latency)));
                    hops.extend(links.iter().map(|l| hop(*l, Dir::Read, 0)));
                    Rc::new(Half {
                        hops: hops.into(),
                        lead: 0,
                        pair: topo.best_pair(&m.attach, masters),
                        grid: grid(&p.blocks),
                    })
                });
                // The write half on the write channels (Section 3.5).
                let write = topo.best_path(masters, &m.attach).map(|p| {
                    let mut hops: Vec<PathHop> = p
                        .hops_for(Dir::Write)
                        .iter()
                        .map(|h| hop(h.lane, Dir::Write, h.latency))
                        .collect();
                    hops.push(hop(m.port_for(Dir::Write), Dir::Write, 0));
                    Rc::new(Half {
                        hops: hops.into(),
                        lead: p.lead,
                        pair: topo.best_pair(masters, &m.attach),
                        grid: grid(&p.blocks),
                    })
                });
                halves.push((read, write));
            }
        }
        Self {
            blocks,
            hashes,
            nblocks,
            halves,
        }
    }

    /// The block that holds physical address `a` and the bytes to its end.
    fn physical(&self, a: u64) -> Option<(usize, u64)> {
        let i = self.blocks.partition_point(|b| b.0.base <= a);
        let (r, b) = self.blocks.get(i.checked_sub(1)?)?;
        r.contains(Addr(a)).then(|| (*b, r.bound - a))
    }

    /// The block that holds `a` and the bytes from `a` that stay in it: a
    /// physical address as it is; an address in a hash link's linear range
    /// rewritten onto its physical block, the bytes then at most to the
    /// end of the hash granule (a chunk never straddles two slices).
    pub fn locate(&self, a: u64) -> Option<(usize, u64)> {
        let i = self.hashes.partition_point(|h| h.range.base <= a);
        if let Some(h) = i
            .checked_sub(1)
            .map(|i| &self.hashes[i])
            .filter(|h| h.range.contains(Addr(a)))
        {
            let (b, run) = self.physical(h.rewrite(a))?;
            let g = h.granule();
            return Some((b, run.min(g - (a & (g - 1)))));
        }
        self.physical(a)
    }

    /// The read half from block `b` to DMA engine `pe`.
    pub fn read(&self, pe: PeId, b: usize) -> Option<&Rc<Half>> {
        self.halves[pe.0 as usize * self.nblocks + b].0.as_ref()
    }

    /// The write half from DMA engine `pe` to block `b`.
    pub fn write(&self, pe: PeId, b: usize) -> Option<&Rc<Half>> {
        self.halves[pe.0 as usize * self.nblocks + b].1.as_ref()
    }
}

/// The bytes from `a` to the next multiple of `grid` (`u64::MAX`: none).
fn grid_run(a: u64, grid: u64) -> u64 {
    if grid == u64::MAX {
        u64::MAX
    } else {
        grid - (a & (grid - 1))
    }
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
    /// Its id.
    pub pe_id: PeId,
    /// The routing function of transfers.
    pub routes: &'a TransferRoutes,
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
            .find(|m| m.holds(addr))
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

    /// The opcode names, by opcode id.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.fns.iter().map(|(n, _)| n.as_str())
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

/// DMA: one transfer chunk per `tile_bytes` (or per burst); its hops in
/// route order are the source port, the fabric route from the source
/// block to the engine, the engine's extra links, the route from the
/// engine to the destination block (each over the engine's master bus
/// with the fewest bridges), and the destination port. The chunk latency
/// is the engine latency; each hop carries the route latency to the next
/// hop. The engine's beat rate starts the narrowing chain.
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
        latency,
        burst,
        beat,
        ..
    } = &node.pe.params
    else {
        panic!("dma opcode on PE {}", node.pe.name);
    };
    // The engine's burst, when set, is the chunk size: a fabric
    // transaction, whatever the program's tile.
    let tile_bytes = burst.unwrap_or(tile_bytes);
    assert!(tile_bytes > 0, "dma with tile_bytes 0");
    // The chunk loop is arithmetic (Section 5.3, routing function): the
    // block and route half of a side are looked up only when its address
    // leaves the block's run (a block or a hash granule), and a chunk
    // ends at the first block, grid or tile boundary of either side.
    let r = node.routes;
    let side = |a: u64, read: bool| {
        let (b, run) = r
            .locate(a)
            .unwrap_or_else(|| panic!("address {a:#x} is in no memory block"));
        let (half, what) = if read {
            (r.read(node.pe_id, b), "from the source")
        } else {
            (r.write(node.pe_id, b), "to the destination")
        };
        let half = half.unwrap_or_else(|| panic!("{}: no route {what}", node.pe.name));
        (half.clone(), run)
    };
    let (mut rh, mut src_run) = side(src.0, true);
    let (mut wh, mut dst_run) = side(dst.0, false);
    let mut chunks = Vec::with_capacity(bytes.div_ceil(tile_bytes) as usize);
    let mut done = 0u64;
    while done < bytes {
        let (sa, da) = (src.0 + done, dst.0 + done);
        if src_run == 0 {
            (rh, src_run) = side(sa, true);
        }
        if dst_run == 0 {
            (wh, dst_run) = side(da, false);
        }
        let n = (bytes - done)
            .min(tile_bytes)
            .min(src_run)
            .min(dst_run)
            .min(grid_run(sa, rh.grid))
            .min(grid_run(da, wh.grid));
        done += n;
        src_run -= n;
        dst_run -= n;
        let use_of = |p: Option<(AttachId, AttachId)>, dir| {
            p.map(|(from, to)| RouteUse {
                from,
                to,
                bytes: n,
                dir,
            })
        };
        chunks.push(Chunk {
            demand: Vec::new(),
            routes: [use_of(rh.pair, Dir::Read), use_of(wh.pair, Dir::Write)],
            transfer: Some(TransferChunk {
                read: rh.clone(),
                write: wh.clone(),
                bytes: n,
            }),
            rate: Ratio::int(*beat),
            latency: *latency,
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
        smem_wport,
        utilization,
        quantize,
        row_panel,
        ref rates,
        ref masters,
        ..
    } = node.pe.params
    else {
        panic!("matmul opcode on PE {}", node.pe.name);
    };
    // The route between the SMEM block of `dst` and the array (over
    // `lbus`), for a tracer only: (reads, writes).
    let smem_route = node
        .memories
        .iter()
        .find(|m| m.covers(&desc.dst))
        .map_or((None, None), |m| {
            (
                node.topology.best_pair(&m.attach, masters),
                node.topology.best_pair(masters, &m.attach),
            )
        });
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
            let mut ops = Demand::work(array, qm * qn * k);
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
            // A and B and C in are reads; C out and the accumulator flush
            // are writes (one resource: charged as their sum).
            let read = a_bytes + b_dtype.bytes_for(k * qn) + c_in;
            let write = c_out + flush;
            let use_of = |p: Option<(AttachId, AttachId)>, bytes, dir| {
                p.map(|(from, to)| RouteUse {
                    from,
                    to,
                    bytes,
                    dir,
                })
            };
            chunks.push(Chunk {
                demand: vec![
                    ops,
                    Demand::read(smem_port, read),
                    Demand::write(smem_wport.unwrap_or(smem_port), write),
                ],
                transfer: None,
                rate: Ratio::ONE,
                latency: 0,
                routes: [
                    use_of(smem_route.0, read, Dir::Read),
                    use_of(smem_route.1, write, Dir::Write),
                ],
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
    // range over the SM's data bus. bytes_in are reads, bytes_out writes;
    // a resource on both routes is charged once for both.
    let mut v = Demand::work(vector, ops);
    v.utilization = utilization;
    let mut demand = vec![v];
    let mut latency = 0;
    let mut routes = [None; 2];
    let out_block = node.memory(Addr(desc.dst.base));
    let in_block = src.map_or(out_block, |a| node.memory(a));
    for (block, bytes, to) in [(in_block, bytes_in, false), (out_block, bytes_out, true)] {
        let (from, dest): (&[AttachId], &[AttachId]) = if to {
            (masters, &block.attach)
        } else {
            (&block.attach, masters)
        };
        let route = node
            .topology
            .best_path(from, dest)
            .unwrap_or_else(|| panic!("{}: no route to {}", node.pe.name, block.name));
        latency = latency.max(route.latency);
        let dir = if to { Dir::Write } else { Dir::Read };
        routes[to as usize] = node
            .topology
            .best_pair(from, dest)
            .map(|(from, to)| RouteUse {
                from,
                to,
                bytes,
                dir,
            });
        for r in route
            .resources_for(dir)
            .iter()
            .chain([&block.port_for(dir)])
        {
            demand.push(Demand::new(*r, bytes, dir));
        }
    }
    Profile {
        chunks: vec![Chunk {
            demand,
            transfer: None,
            rate: Ratio::ONE,
            latency: overhead + latency,
            routes,
        }],
        holds: Vec::new(),
    }
}

fn ratio_cmp(a: &Ratio, b: &Ratio) -> std::cmp::Ordering {
    (a.num as u128 * b.den as u128).cmp(&(b.num as u128 * a.den as u128))
}

#[cfg(test)]
mod tests {
    use nu_model::{Addr, Range};

    use nu_model::AddressHash;

    use super::{dma_profile, Args, Descriptor, NodeConfig, PathHop, TransferRoutes};
    use crate::config::{MemBlock, PeConfig, PeParams};
    use crate::profile::{AccFlags, PeId, ProfileRegistry};
    use crate::resource::Dir;
    use crate::topology::{grid_run, Topology};

    /// A root with an HBM of 4 banks behind a hash link (512 B granules),
    /// two adjacent dram blocks and a bus grid of 1 KiB, and one SM whose
    /// local bus has a grid of 256 B.
    const CONFIG: &str = "global:
  neutrino_granule: 512
  address: {fields: [{name: sm, bits: 2}], offset_bits: 24}
device:
  name: root
  fabric:
    - {name: sbus, block_bytes: 1024}
    - {name: cbus, control: true}
    - {name: hbmbus}
    - {name: sbus_to_hbmbus, topology: link, from: $this.sbus, to: $this.hbmbus,
       hash: {range: [0x0, 0x10000], to_range: [0x40000, 0x50000], index: [0x200, 0x400]}}
  memory:
    - {name: hbm, type: dram, addr: [[0x40000, 0x50000]], banks: 4, bw_bytes: 64, slave: $this.hbmbus}
    - {name: d0, type: dram, addr: [[0x10000, 0x18000]], bw_bytes: 64, slave: $this.sbus}
    - {name: d1, type: dram, addr: [[0x18000, 0x20000]], bw_bytes: 64, block_bytes: 2048, slave: $this.sbus}
    - {name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}
  neutrino: {c_base: 0, c_dep: 0, c_word: 0}
  children:
    - name: sm0
      fabric:
        - {name: smsbus}
        - {name: smcbus, control: true}
        - {name: lbus, block_bytes: 256}
        - {name: smsbus_to_sbus, topology: link, from: $this.smsbus, to: $parent.sbus, bw_bytes: 1024}
        - {name: smcbus_to_cbus, topology: link, from: $this.smcbus, to: $parent.cbus, bw_bytes: 128, control: true}
      memory:
        - {name: smem, type: scratch, addr: [[0x0, 0x10000]], bw_bytes: 1024, slave: [$this.lbus, $this.smsbus]}
        - {name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}
      compute:
        - {name: dma, type: dma, beat_bytes: 1024, master: [$this.lbus, $this.smsbus]}
      neutrino: {c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}
";

    /// The chunks of a DMA by the lookup per chunk that the routing tables
    /// replace: the block of each address (through the hash link's
    /// rewrite), its run and port grid, the hash granule, and the grid of
    /// the buses and links of each route half.
    fn by_lookup(
        mems: &[MemBlock],
        hash: &AddressHash,
        topo: &Topology,
        pe: &PeConfig,
        (src, dst, bytes, tile): (u64, u64, u64, u64),
    ) -> Vec<Vec<PathHop>> {
        let PeParams::Dma { links, masters, .. } = &pe.params else {
            unreachable!()
        };
        // The block and the bytes that stay in it from address `a`.
        let block = |a: u64| {
            let (p, cap) = if hash.range.contains(Addr(a)) {
                let g = hash.granule();
                (hash.rewrite(a), g - a % g)
            } else {
                (a, u64::MAX)
            };
            let m = mems.iter().find(|m| m.holds(Addr(p))).unwrap();
            (
                m,
                m.run(p).min(cap),
                m.block.map_or(u64::MAX, |b| b - a % b),
            )
        };
        let mut out = Vec::new();
        let mut done = 0;
        while done < bytes {
            let (sa, da) = (src + done, dst + done);
            let ((sm, srun, sgrid), (dm, drun, dgrid)) = (block(sa), block(da));
            let inp = topo.best_path(&sm.attach, masters).unwrap();
            let outp = topo.best_path(masters, &dm.attach).unwrap();
            let n = (bytes - done)
                .min(tile)
                .min(srun)
                .min(drun)
                .min(sgrid)
                .min(dgrid)
                .min(grid_run(sa, &inp.blocks))
                .min(grid_run(da, &outp.blocks));
            let hop = |resource, dir, latency| PathHop {
                resource,
                amount: n,
                dir,
                latency,
            };
            let mut path = vec![hop(sm.port, Dir::Read, inp.lead)];
            path.extend(inp.hops.iter().map(|h| hop(h.lane, Dir::Read, h.latency)));
            path.extend(links.iter().map(|l| hop(*l, Dir::Read, 0)));
            path.last_mut().unwrap().latency += outp.lead;
            path.extend(
                outp.hops_for(Dir::Write)
                    .iter()
                    .map(|h| hop(h.lane, Dir::Write, h.latency)),
            );
            path.push(hop(dm.port_for(Dir::Write), Dir::Write, 0));
            out.push(path);
            done += n;
        }
        out
    }

    #[test]
    fn the_chunk_loop_matches_the_lookup_per_chunk_on_random_spans() {
        let mut b = crate::config::build(CONFIG, ProfileRegistry::with_builtins()).unwrap();
        b.sim.finish_fabric().unwrap();
        let pe_id = b.pe("root.sm0.dma");
        let sim = &b.sim;
        let pes: Vec<&PeConfig> = (0..sim.pe_count())
            .map(|i| sim.pe_config(PeId(i as u32)))
            .collect();
        let hashes: Vec<&AddressHash> = sim.hashes().collect();
        let routes = TransferRoutes::build(&pes, sim.memories(), &hashes, sim.topology());
        let smem = b.memory("root.sm0.smem").base;
        let node = NodeConfig {
            pe: sim.pe_config(pe_id),
            pe_id,
            routes: &routes,
            memories: sim.memories(),
            topology: sim.topology(),
            acc: AccFlags::default(),
        };
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |n: u64| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1442695040888963407);
            (x >> 33) % n
        };
        let mut buf = Vec::new();
        for _ in 0..2000 {
            let bytes = 1 + rnd(9000);
            let tile = [64, 512, 1500, 4096, 65536][rnd(5) as usize];
            // From the hashed HBM or across the two dram blocks into SMEM,
            // or back.
            let far = if rnd(2) == 0 {
                rnd(0x10000 - bytes)
            } else {
                0x10000 + rnd(0x10000 - bytes)
            };
            let near = smem + rnd(0x10000 - bytes);
            let (src, dst) = if rnd(2) == 0 {
                (far, near)
            } else {
                (near, far)
            };
            let desc = Descriptor {
                opcode: crate::profile::OpcodeId(2),
                pe: Some(pe_id),
                dst: Range::from_len(dst, bytes),
                versioned: false,
                inferred_version: None,
                deps: vec![],
                label: None,
                args: Args::Dma {
                    src: Addr(src),
                    dst: Addr(dst),
                    bytes,
                    tile_bytes: tile,
                },
            };
            let p = dma_profile(&desc, &node);
            let want = by_lookup(
                sim.memories(),
                hashes[0],
                sim.topology(),
                node.pe,
                (src, dst, bytes, tile),
            );
            let got: Vec<Vec<PathHop>> = p
                .chunks
                .iter()
                .map(|c| {
                    c.transfer.as_ref().unwrap().hops_into(&mut buf);
                    buf.clone()
                })
                .collect();
            assert_eq!(got, want, "{src:#x} -> {dst:#x}, {bytes} B, tile {tile}");
        }
    }
}
