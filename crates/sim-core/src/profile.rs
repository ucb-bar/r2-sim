//! Descriptors and demand profiles (r2sim spec Section 6).
//!
//! Every leaf opcode registers one function from a descriptor and the node
//! configuration to a [`Profile`]: the chunks of the task, each with a
//! demand vector and a latency, and the occupancy held from issue to
//! completion. A profile never reads functional data.

use std::rc::Rc;

use nu_model::{Addr, AddressHash, Dep, NeutrinoId, Range, FULL_MASK};

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
    /// `dma`, the first built-in profile of
    /// [`ProfileRegistry::with_builtins`].
    pub const DMA: OpcodeId = OpcodeId(2);
    /// `atlas.matmul`, a built-in profile.
    pub const MATMUL: OpcodeId = OpcodeId(3);
    /// `muon.vector`, a built-in profile.
    pub const VECTOR: OpcodeId = OpcodeId(4);
    /// `imem_copy`, a built-in opcode without a profile (RV programs,
    /// 2026-10-23): a copy into a child's IMEM on the instruction DMA of
    /// the invoking Neutrino.
    pub const IMEM_COPY: OpcodeId = OpcodeId(5);
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
    ///
    /// Split operands (banked SMEM, Neutrino spec Section 4.2.2): with `a2`,
    /// the rows of A are in two half matrices. Row `r` of A, with
    /// `t = r / tile[0]` and `j = r % tile[0]`, is row `t * h + j` of the
    /// matrix at `a` when `j < h = tile[0] / 2`, else row `t * h + j - h`
    /// of the matrix at `a2`; both have leading dimension `lda`. With
    /// `b2`, the columns of B are split in the same way over `tile[1]`
    /// (the matrices at `b` and `b2`, leading dimension `ldb`).
    Matmul {
        /// Address of A (with `a2`: of its low half rows).
        a: Addr,
        /// Address of B (with `b2`: of its low half columns).
        b: Addr,
        /// Address of the high half rows of A; `None`: A is one matrix.
        a2: Option<Addr>,
        /// Address of the high half columns of B; `None`: B is one matrix.
        b2: Option<Addr>,
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
        /// the same dst; without an accumulator memory, no effect.
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
    /// Copy `bytes` from offset `src` of the invoking Neutrino's IMEM to
    /// offset `offset` of the IMEM of its direct child `child`, on the
    /// invoker's instruction DMA (RV programs, 2026-10-23).
    ImemCopy {
        child: NeutrinoId,
        src: u64,
        offset: u64,
        bytes: u64,
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
    /// `dst` is a versioned data range (Neutrino spec Section 7); false for
    /// a signal in the sync-only space.
    pub versioned: bool,
    /// An untracked task (2026-10-20): it runs on a PE of the invoking
    /// Neutrino and is never forwarded; it gets no version, no lookup finds
    /// it, and only a fence or the program's return waits for it. Its
    /// invoke returns no handle ([`crate::NuCtx::invoke_untracked`]).
    pub untracked: bool,
    /// The granules of `dst` the task holds, as in Rocket's address sets
    /// (2026-10-21): granule `dst.base / g + i` belongs to the range when
    /// `i & !dst_mask == 0`; [`nu_model::FULL_MASK`] holds the whole span.
    pub dst_mask: u32,
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
    /// The PC of the `nu.invoke` of an RV program whose instructions built
    /// this descriptor (2026-10-23): its assembly is in the program's
    /// instruction cycles, so the invoke charges no `c_desc`. `None` for
    /// the descriptors of the Rust programs.
    pub pc: Option<u64>,
    /// The call path of the RV program at `pc`: the PCs of the call
    /// instructions from the program function to the frame that made the
    /// task, outermost first (raw trace 1.11). Empty for a task made in
    /// the program function, and for the Rust programs.
    pub call_path: Vec<u32>,
}

impl Descriptor {
    /// A versioned task on `pe` that writes `dst`: full mask, no
    /// dependencies, no label.
    fn task(opcode: OpcodeId, pe: PeId, dst: Range, args: Args) -> Self {
        Self {
            opcode,
            pe: Some(pe),
            dst,
            versioned: true,
            untracked: false,
            dst_mask: FULL_MASK,
            inferred_version: None,
            pc: None,
            call_path: Vec::new(),
            deps: Vec::new(),
            label: None,
            args,
        }
    }

    /// A DMA on engine `pe` of `bytes` from `src` to the start of `dst`, in
    /// one tile. The engine must belong to the owner of `dst` (a write into
    /// another node's memory is a pull by its owner; Neutrino spec), unless
    /// the task is [untracked](Descriptor::untracked).
    pub fn dma(pe: PeId, src: u64, dst: Range, bytes: u64) -> Self {
        Self::task(
            OpcodeId::DMA,
            pe,
            dst,
            Args::Dma {
                src: Addr(src),
                dst: Addr(dst.base),
                bytes,
                tile_bytes: bytes,
            },
        )
    }

    /// A matmul on the Atlas `pe`: C (`m` x `n`, at the start of `c`) = A
    /// (`m` x `k` at `a`) B (`k` x `n` at `b`), all of type `dtype`,
    /// row-major with leading dimensions `k`, `n` and `n`; no
    /// accumulation, C not kept; output tiles of at most 128 x 128.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        pe: PeId,
        a: u64,
        b: u64,
        c: Range,
        m: u64,
        n: u64,
        k: u64,
        dtype: Dtype,
    ) -> Self {
        Self::task(
            OpcodeId::MATMUL,
            pe,
            c,
            Args::Matmul {
                a: Addr(a),
                b: Addr(b),
                a2: None,
                b2: None,
                lda: k,
                ldb: n,
                ldc: n,
                m,
                n,
                k,
                in_dtype: dtype,
                b_dtype: dtype,
                out_dtype: dtype,
                accumulate: false,
                keep: false,
                tile: [m.min(128), n.min(128)],
            },
        )
    }

    /// A vector task on the Muon `pe` of `ops` lane-ops that reads
    /// `bytes_in` (from `dst` unless [`Descriptor::with_src`]) and writes
    /// `bytes_out` to `dst`.
    pub fn vector(pe: PeId, dst: Range, ops: u64, bytes_in: u64, bytes_out: u64) -> Self {
        Self::task(
            OpcodeId::VECTOR,
            pe,
            dst,
            Args::Vector {
                ops,
                bytes_in,
                bytes_out,
                src: None,
            },
        )
    }

    /// Sets the dependencies.
    pub fn with_deps(mut self, deps: Vec<Dep>) -> Self {
        self.deps = deps;
        self
    }

    /// Marks `dst` as an unversioned range (a signal, or a writer whose
    /// consumers name an address).
    pub fn unversioned(mut self) -> Self {
        self.versioned = false;
        self
    }

    /// A DMA's chunk size in bytes.
    pub fn with_tile_bytes(mut self, n: u64) -> Self {
        if let Args::Dma { tile_bytes, .. } = &mut self.args {
            *tile_bytes = n;
        }
        self
    }

    /// A matmul's output tile, `[rows, columns]`.
    pub fn with_tile(mut self, rows: u64, cols: u64) -> Self {
        if let Args::Matmul { tile, .. } = &mut self.args {
            *tile = [rows, cols];
        }
        self
    }

    /// A matmul's leading dimensions, in elements.
    pub fn with_ld(mut self, a: u64, b: u64, c: u64) -> Self {
        if let Args::Matmul { lda, ldb, ldc, .. } = &mut self.args {
            (*lda, *ldb, *ldc) = (a, b, c);
        }
        self
    }

    /// A matmul's operand and output types.
    pub fn with_types(mut self, a: Dtype, b: Dtype, out: Dtype) -> Self {
        if let Args::Matmul {
            in_dtype,
            b_dtype,
            out_dtype,
            ..
        } = &mut self.args
        {
            (*in_dtype, *b_dtype, *out_dtype) = (a, b, out);
        }
        self
    }

    /// A matmul that adds to C (`accumulate`) and keeps C in the
    /// accumulator (`keep`).
    pub fn with_acc(mut self, accumulate: bool, keep: bool) -> Self {
        if let Args::Matmul {
            accumulate: a,
            keep: k,
            ..
        } = &mut self.args
        {
            (*a, *k) = (accumulate, keep);
        }
        self
    }

    /// A vector task's source address.
    pub fn with_src(mut self, src: u64) -> Self {
        if let Args::Vector { src: s, .. } = &mut self.args {
            *s = Some(Addr(src));
        }
        self
    }

    /// A `join` task on `dst` that depends on `deps`.
    pub fn join(dst: Range, deps: Vec<Dep>) -> Self {
        Self {
            opcode: OpcodeId::JOIN,
            pe: None,
            dst,
            versioned: false,
            untracked: false,
            dst_mask: nu_model::FULL_MASK,
            inferred_version: None,
            pc: None,
            call_path: Vec::new(),
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

    /// Sets the mask of `dst` (in granules; [`nu_model::FULL_MASK`]: the
    /// whole span).
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.dst_mask = mask;
        self
    }

    /// The masked destination range.
    pub fn held(&self) -> nu_model::Dst {
        nu_model::Dst {
            span: self.dst,
            mask: self.dst_mask,
        }
    }

    /// Marks the task as untracked; invoke it with
    /// [`crate::NuCtx::invoke_untracked`].
    pub fn untracked(mut self) -> Self {
        self.untracked = true;
        self
    }

    /// Bytes of the descriptor in an `Invoke` message: its size in the
    /// Neutrino ABI (2026-10-26), a 24-byte header (opcode, PE, flags,
    /// dependency count, `dst` base, bound and mask), 8 per dependency, and
    /// the arguments (DMA 32, matmul 72, vector 32, spawn 16, IMEM copy 24,
    /// join 0). The label is not sent.
    pub fn wire_bytes(&self) -> u64 {
        let args = match self.args {
            Args::Dma { .. } => 32,
            Args::Matmul { .. } => 72,
            Args::Vector { .. } => 32,
            Args::Join => 0,
            Args::Spawn { .. } => 16,
            Args::ImemCopy { .. } => 24,
        };
        24 + 8 * self.deps.len() as u64 + args
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
/// hop in route order; r2sim spec Section 5.6).
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
    /// start (r2sim spec Section 5.9).
    pub latency: Cycle,
    /// The fabric routes the chunk's bytes take, for a tracer (the
    /// unlimited links of a route have no demand).
    pub routes: [Option<RouteUse>; 2],
}

/// One half of a transfer's route between a memory block and a DMA
/// engine (r2sim spec Section 5.5), computed once per engine and
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
    /// Read half: a pool whose link the route crosses after its last hop
    /// (r2sim spec Section 5.6, Rule S5.11); its hold ends with the write half's first
    /// hop.
    pub pool_after: Option<crate::resource::PoolMark>,
}

/// A transfer chunk: `bytes` over the read half then the write half.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferChunk {
    pub read: Rc<Half>,
    pub write: Rc<Half>,
    pub bytes: u64,
    /// The engine's beat in bytes (the start of the beat chain, Section
    /// 5.4); the chunk's `rate` is the engine's bytes per core cycle.
    pub beat: u64,
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
        let first = buf.len();
        buf.extend(self.write.hops.iter().map(|h| PathHop {
            amount: self.bytes,
            ..*h
        }));
        // A pool crossed after the read half's last hop ends with the
        // write half's first hop.
        if let (Some(m), Some(h)) = (self.read.pool_after, buf.get_mut(first)) {
            debug_assert!(h.pool.is_none(), "two pools on one hop");
            h.pool = Some(crate::resource::PoolMark {
                skip: m.skip + self.write.lead,
                ..m
            });
        }
    }
}

/// The routing function of transfers (r2sim spec Section 5.5): memory blocks by
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
                    pool: None,
                };
                let from = |h: &crate::topology::Hop, dir| PathHop {
                    pool: h.pool,
                    ..hop(h.lane, dir, h.latency)
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
                    hops.extend(p.hops.iter().map(|h| from(h, Dir::Read)));
                    let after = hops.len();
                    hops.extend(links.iter().map(|l| hop(*l, Dir::Read, 0)));
                    // A pool crossed after the route's last hop ends with
                    // the engine's first link, else with the write half.
                    let mut pool_after = p.pool_after;
                    if let Some(h) = hops.get_mut(after) {
                        h.pool = pool_after.take();
                    }
                    Rc::new(Half {
                        hops: hops.into(),
                        lead: 0,
                        pair: topo.best_pair(&m.attach, masters),
                        grid: grid(&p.blocks),
                        pool_after,
                    })
                });
                // The write half on the write channels (Radiance 2 spec Section 3.4).
                let write = topo.best_path(masters, &m.attach).map(|p| {
                    let mut hops: Vec<PathHop> = p
                        .hops_for(Dir::Write)
                        .iter()
                        .map(|h| from(h, Dir::Write))
                        .collect();
                    // A pool crossed after the route's last hop ends with
                    // the block's write port.
                    hops.push(PathHop {
                        pool: p.pool_after,
                        ..hop(m.port_for(Dir::Write), Dir::Write, 0)
                    });
                    Rc::new(Half {
                        hops: hops.into(),
                        lead: p.lead,
                        pair: topo.best_pair(masters, &m.attach),
                        grid: grid(&p.blocks),
                        pool_after: None,
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

    /// The pieces of `r` in memory blocks, as (block, bytes) in address
    /// order (physical addresses; bytes in no block are left out).
    pub fn pieces(&self, r: Range) -> impl Iterator<Item = (usize, u64)> + '_ {
        let mut a = r.base;
        std::iter::from_fn(move || {
            while a < r.bound {
                match self.physical(a) {
                    Some((b, run)) => {
                        let n = run.min(r.bound - a);
                        a += n;
                        return Some((b, n));
                    }
                    None => {
                        // To the next block's first byte.
                        let i = self.blocks.partition_point(|x| x.0.base <= a);
                        a = self
                            .blocks
                            .get(i)
                            .map_or(r.bound, |x| x.0.base.min(r.bound));
                    }
                }
            }
            None
        })
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
    /// A transfer's pipeline fill (r2sim spec Section 5.9, source
    /// latency): no chunk starts before the task's issue plus this many
    /// cycles plus the chunk's own latency. 0 for a compute task.
    pub fill: Cycle,
    /// The part of `fill` that is the read request's trip to the source
    /// block (2026-10-21); the rest is the block's latency, which the
    /// in-flight pools take as the source latency.
    pub request: Cycle,
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
    /// Atlas accumulator memory decisions for this task.
    pub acc: AccFlags,
}

/// What the Atlas accumulator memory does for one task: the C read
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
    /// The memory block that holds the physical address `a` (by the
    /// routing function's block table).
    pub fn block_of(&self, a: u64) -> &MemBlock {
        match self.routes.physical(a) {
            Some((b, _)) => &self.memories[b],
            None => self.memory(Addr(a)),
        }
    }

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
    /// `muon.vector`, and the `imem_copy` opcode.
    pub fn with_builtins() -> Self {
        let mut r = Self::default();
        assert_eq!(r.register("dma", dma_profile), OpcodeId::DMA);
        assert_eq!(r.register("atlas.matmul", matmul_profile), OpcodeId::MATMUL);
        assert_eq!(r.register("muon.vector", vector_profile), OpcodeId::VECTOR);
        r.fns.push(("imem_copy".to_string(), None));
        assert_eq!(r.lookup("imem_copy"), Some(OpcodeId::IMEM_COPY));
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
/// hop. The engine's beat rate starts the narrowing chain. The transfer's
/// pipeline fill is the latency of the block of its first source byte
/// (r2sim spec Section 5.9, source latency).
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
        clock,
        ..
    } = &node.pe.params
    else {
        panic!("dma opcode on PE {}", node.pe.name);
    };
    // The engine's burst, when set, is the chunk size: a fabric
    // transaction, whatever the program's tile.
    let tile_bytes = burst.unwrap_or(tile_bytes);
    assert!(tile_bytes > 0, "dma with tile_bytes 0");
    // The chunk loop is arithmetic (r2sim spec Section 5.5): the
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
                beat: *beat,
            }),
            rate: Ratio::new(*beat * clock.num, clock.den),
            latency: *latency,
        });
    }
    // The pipeline fill: the request's trip to the block of the first
    // source byte (propagation only, 2026-10-21), then the block's
    // latency.
    let (request, fill) = r.locate(src.0).map_or((0, 0), |(b, _)| {
        let leg = request_leg(node.topology, node.pe, &node.memories[b]);
        (leg, leg + node.memories[b].load_latency)
    });
    Profile {
        chunks,
        holds: Vec::new(),
        fill,
        request,
    }
}

/// Atlas matmul: one chunk per output tile; tensor ops on the array and
/// operand plus C bytes on SMEM. With one SMEM port (`smem_port`), all
/// bytes are on that port. On a banked SMEM (`smem_port` is `None`), each
/// operand and C range of the chunk charges the ports of the blocks it
/// overlaps, in proportion to its bytes in each (r2sim spec Section
/// 6.2, banked SMEM); the demands of one chunk are co-charged.
pub(crate) fn matmul_profile(desc: &Descriptor, node: &NodeConfig) -> Profile {
    let Args::Matmul {
        a,
        b,
        a2,
        b2,
        lda,
        ldb,
        ldc,
        m,
        n,
        k,
        in_dtype,
        b_dtype,
        out_dtype,
        accumulate,
        keep,
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
        beat,
        acc_buffer,
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
        .or_else(|| {
            smem_port
                .is_none()
                .then(|| node.memories.iter().find(|m| m.holds(Addr(desc.dst.base))))
                .flatten()
        })
        .map_or((None, None), |m| {
            (
                node.topology.best_pair(&m.attach, masters),
                node.topology.best_pair(masters, &m.attach),
            )
        });
    // The class a tile runs at, for a tracer: the slower operand type's
    // index in `rates` (the faster one's on a tie).
    let class = (!rates.is_empty()).then(|| {
        let at = |d: Dtype| rates.iter().position(|r| r.0 == d).unwrap_or(0);
        let (a, b) = (at(in_dtype), at(b_dtype));
        (if ratio_cmp(&rates[b].1, &rates[a].1).is_lt() {
            b
        } else {
            a
        }) as u8
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
    let es = out_dtype.bytes();
    // On an SM with a tmem, C is in the tmem (tmem model, 2026-10-21): a
    // partial C that is not in the accumulator is fp32 (read when
    // accumulating, written when kept); the last task writes C in its
    // output type. The task pays the tmem's latency once, before its first
    // output tile.
    let tmem = node
        .memories
        .iter()
        .find(|m| m.is_tmem() && m.holds(Addr(desc.dst.base)));
    let (es_in, es_out) = match tmem {
        Some(_) => (4, if keep { 4 } else { es }),
        None => (es, es),
    };
    // Output stationary (2026-10-21): an accumulator that holds one fp32
    // output tile frees it only once its C is written, so the last task of
    // a chain writes C after the tile's compute, as a tail that holds the
    // Atlas. With room for two or more tiles the next tile accumulates
    // while C drains: the write overlaps the compute.
    let tail = tmem.is_some() && !keep && acc_buffer > 0 && acc_buffer < 2 * tile[0] * tile[1] * 4;
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
            ops.class = class;
            // Operands in, C out, and C in when accumulating. C is exact.
            let (c_bytes, c_in_bytes) = (tm * tn * es_out, tm * tn * es_in);
            // With a row panel, A is read at the first output tile of the row.
            let a_bytes = if row_panel && ni > 0 {
                0
            } else {
                in_dtype.bytes_for(qm * k)
            };
            let b_bytes = b_dtype.bytes_for(k * qn);
            let c_out = if acc.skip_write { 0 } else { c_bytes };
            let c_in = if accumulate && !acc.skip_read {
                c_in_bytes
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
            let read = a_bytes + b_bytes + c_in;
            let write = c_out + flush;
            let mut demand = vec![ops];
            // A tmem's banks take the bytes, whatever the SMEM's ports.
            match smem_port.filter(|_| tmem.is_none()) {
                Some(port) => demand.extend([
                    Demand::read(port, read),
                    Demand::write(smem_wport.unwrap_or(port), write),
                ]),
                None => {
                    // A: the tile's rows; B: all K rows of the tile's
                    // columns; C: the tile. A half of a split operand is
                    // in one bank by layout: its bytes go to the bank of
                    // its first byte.
                    let c_at = |es: u64| {
                        Range::from_len(
                            desc.dst.base + (mi * ldc + ni) * es,
                            ((tm - 1) * ldc + tn) * es,
                        )
                    };
                    for (part, bytes) in rows_split(a, a2, mi, qm, tile[0], a_bytes)
                        .into_iter()
                        .flatten()
                    {
                        let r = Range::from_len(
                            part.0 + in_dtype.bytes_for(part.1 * lda),
                            in_dtype.bytes_for((part.2 - 1) * lda + k),
                        );
                        operand(node, r, a2.is_some(), bytes, &mut demand);
                    }
                    for (part, bytes) in rows_split(b, b2, ni, qn, tile[1], b_bytes)
                        .into_iter()
                        .flatten()
                    {
                        let r = Range::from_len(
                            part.0 + b_dtype.bytes_for(part.1),
                            b_dtype.bytes_for((k - 1) * ldb + part.2),
                        );
                        operand(node, r, b2.is_some(), bytes, &mut demand);
                    }
                    spread(node, c_at(es_in), c_in, Dir::Read, &mut demand);
                    spread(node, c_at(es_out), write, Dir::Write, &mut demand);
                }
            }
            let use_of = |p: Option<(AttachId, AttachId)>, bytes, dir| {
                p.map(|(from, to)| RouteUse {
                    from,
                    to,
                    bytes,
                    dir,
                })
            };
            // The array's operand and C bytes move in its beats.
            for d in demand.iter_mut().filter(|d| d.dir != Dir::Work) {
                d.beat = beat;
            }
            if tail {
                // The compute and the reads, then the C write.
                let (writes, rest): (Vec<Demand>, Vec<Demand>) =
                    demand.into_iter().partition(|d| d.dir == Dir::Write);
                chunks.push(Chunk {
                    demand: rest,
                    transfer: None,
                    rate: Ratio::ONE,
                    latency: 0,
                    routes: [use_of(smem_route.0, read, Dir::Read), None],
                });
                chunks.push(Chunk {
                    demand: writes,
                    transfer: None,
                    rate: Ratio::ONE,
                    latency: 0,
                    routes: [None, use_of(smem_route.1, write, Dir::Write)],
                });
                continue;
            }
            chunks.push(Chunk {
                demand,
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
        fill: tmem.map_or(0, |m| m.load_latency),
        request: 0,
    }
}

/// Where row (or column) `r` of a split operand with tile `t` is
/// ([`Args::Matmul`]): `(true, i)` for row `i` of the high half matrix,
/// `(false, i)` for row `i` of the low half matrix.
pub fn split_row(r: u64, t: u64) -> (bool, u64) {
    let h = t / 2;
    let (ti, j) = (r / t, r % t);
    if j < h {
        (false, ti * h + j)
    } else {
        (true, ti * h + j - h)
    }
}

/// A part of an operand tile: (matrix address, first row in it, rows).
type Part = (u64, u64, u64);

/// The parts of `rows` rows (or columns) of one tile of an operand from
/// row `first` (a multiple of the tile height `t`), as `((matrix address,
/// first row in it, rows), bytes)` with `bytes` of the chunk split between
/// them in proportion to the rows: one part without `hi`, else the low and
/// the high half (Args::Matmul). With a quantized tile, `rows` is `t`.
fn rows_split(
    lo: Addr,
    hi: Option<Addr>,
    first: u64,
    rows: u64,
    t: u64,
    bytes: u64,
) -> [Option<(Part, u64)>; 2] {
    let Some(hi) = hi else {
        return [Some(((lo.0, first, rows), bytes)), None];
    };
    let h = t / 2;
    let row = first / t * h;
    let low = rows.min(h);
    let low_bytes = bytes * low / rows;
    [
        (low > 0).then_some(((lo.0, row, low), low_bytes)),
        (rows > low).then_some(((hi.0, row, rows - low), bytes - low_bytes)),
    ]
}

/// The read of `bytes` of an operand range `r`: a half of a split operand
/// (`split`) is in one bank by layout, so its bytes go to the bank of its
/// first byte; an unsplit range spreads over the blocks it overlaps.
fn operand(node: &NodeConfig, r: Range, split: bool, bytes: u64, out: &mut Vec<Demand>) {
    if split {
        let port = node.block_of(r.base).port_for(Dir::Read);
        out.push(Demand::read(port, bytes));
    } else {
        spread(node, r, bytes, Dir::Read, out);
    }
}

/// Adds `amount` in direction `dir` on the ports of the memory blocks that
/// `range` overlaps, in proportion to its bytes in each (the parts sum to
/// `amount`; bytes of `range` in no block do not count); a range in no
/// block charges the block of its base.
fn spread(node: &NodeConfig, range: Range, amount: u64, dir: Dir, out: &mut Vec<Demand>) {
    if amount == 0 {
        return;
    }
    let total: u64 = node.routes.pieces(range).map(|p| p.1).sum();
    if total == 0 {
        out.push(Demand::new(
            node.block_of(range.base).port_for(dir),
            amount,
            dir,
        ));
        return;
    }
    let (mut covered, mut given) = (0u64, 0u64);
    let mut last: Option<(usize, usize)> = None;
    for (b, n) in node.routes.pieces(range) {
        covered += n;
        let part = (amount as u128 * covered as u128 / total as u128) as u64 - given;
        given += part;
        match last {
            // The next set of the same block: one demand.
            Some((lb, i)) if lb == b => out[i].amount += part,
            _ => {
                last = Some((b, out.len()));
                out.push(Demand::new(node.memories[b].port_for(dir), part, dir));
            }
        }
    }
}

/// The propagation latency of a read request from the PE `pe` to the
/// block `m`: the sum of the hop latencies of the route from the PE's
/// master buses to the block (2026-10-21). The request takes no resource,
/// no beat time, and waits for nothing.
pub(crate) fn request_leg(topo: &Topology, pe: &PeConfig, m: &MemBlock) -> Cycle {
    let (PeParams::Dma { masters, .. }
    | PeParams::Muon { masters, .. }
    | PeParams::Atlas { masters, .. }) = &pe.params;
    topo.best_path(masters, &m.attach).map_or(0, |p| p.latency)
}

/// Muon vector task: one chunk; lane-ops on the vector resource at the
/// configured utilization, plus the task's bytes on the port of the block
/// that holds `dst` and on the route to it, after the fixed overhead and
/// the longer path latency: the input path is the latency of the block
/// that holds `src` (its bank on a banked memory; 2026-10-21) plus the
/// route's, the output path the route's only (a write pays no block
/// latency).
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
        beat,
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
        // A read pays its request's trip to the source block and the
        // block's latency, as a DMA's fill and a load do (2026-10-21); a
        // task that reads nothing has no input path.
        let block_latency = if to || bytes == 0 {
            0
        } else {
            request_leg(node.topology, node.pe, block) + block.load_latency
        };
        latency = latency.max(block_latency + route.latency);
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
        for r in route.resources_for(dir) {
            demand.push(Demand::new(*r, bytes, dir));
        }
        // A banked block: the bytes spread over the banks of the range
        // (r2sim spec Section 6.3); otherwise the block's port.
        let base = if to {
            desc.dst.base
        } else {
            src.map_or(desc.dst.base, |a| a.0)
        };
        if block.banked && bytes > 0 {
            spread(node, Range::from_len(base, bytes), bytes, dir, &mut demand);
        } else {
            demand.push(Demand::new(block.port_for(dir), bytes, dir));
        }
    }
    for d in demand.iter_mut().filter(|d| d.dir != Dir::Work) {
        d.beat = beat;
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
        fill: 0,
        request: 0,
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

    /// A root with an HBM of 4 banks behind a hash link (512 B granules;
    /// linear at 0x20000, off the null address),
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
       hash: {range: [0x20000, 0x30000], to_range: [0x40000, 0x50000], index: [0x200, 0x400]}}
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
                pool: None,
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
                0x20000 + rnd(0x10000 - bytes)
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
                untracked: false,
                dst_mask: nu_model::FULL_MASK,
                inferred_version: None,
                pc: None,
                call_path: Vec::new(),
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
