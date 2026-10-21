//! Kernel sequences: a run of GPU-style kernels in launch order, to compare
//! the model with a card that runs the same kernels with the same tiling
//! (the RTX PRO 6000 configurations, architecture Section 8.3). The test
//! definitions stay outside the repository; they pass the parameters.
//!
//! Parameters (TOML):
//!
//! * `tensors`: `{name, bytes, home}`, `home` = `dram` (default) or `l2`.
//!   Tensors are laid out in the linear ranges of the memories `dram` and
//!   `l2` (by default `root.hbm` and `root.l2`; the linear range is the
//!   source range of the hash link that maps onto the memory, if any).
//! * `cache`: `first_touch` (default), `home`, `per_kernel` or `sim`. Under `first_touch`, the
//!   first read of a tile range in launch order comes from the tensor's
//!   home, and later reads of that range come from the L2; writes go to
//!   the L2. This stands for an L2 that keeps a working set that fits;
//!   there is no cache model, and the decision is made when the program is
//!   built (a read that follows a first read of the same range in another
//!   CTA reads the L2 even while the first read is still in flight). Every
//!   tensor of home `dram` then has a copy in the L2 range. Under `home`,
//!   every access goes to the tensor's home. Under `per_kernel` the L2 is
//!   empty at each kernel's start (a measurement tool's flush): every
//!   tensor's home is the dram, and the first-touch rule applies within
//!   each kernel. Under `sim` every tensor's home is the dram and every
//!   access goes there; the simulator's cache of the dram (a `cache` key
//!   on the l2 memory, `sim_core::sim::cache`) decides what the L2 holds,
//!   and the device empties it before each kernel's start mark; the
//!   `kernels` object then has each kernel's cache counters.
//! * `sms`: the SMs the CTAs run on, the first n taken round-robin over the
//!   gpcs (default 0: every SM).
//! * `kernels`: `{name, op, inputs, output, ctas_per_sm, ..}` with `op` one
//!   of `gemm`, `softmax`, `layernorm`, `elementwise`, `attention` and the
//!   shape and tile fields of [`Kernel`].
//!
//! Execution: a kernel is a grid of CTAs in launch order. CTA i runs on SM
//! i mod n (the SMs taken round-robin over the gpcs), at most
//! `ctas_per_sm` at once per SM, each in its own part of the SMEM. A CTA
//! starts on an SM when the CTA `ctas_per_sm` before it on that SM is
//! done. All CTAs of a kernel end before the next kernel starts: per
//! kernel the device starts a program on each gpc, which starts one on
//! each of its SMs, and waits for them. The device marks `<kernel> start`
//! and `<kernel> end` (`--phase-stats`, the raw dump's marks).
//!
//! A tile of a matrix is one contiguous range: a tensor of R x C elements
//! tiled by h x w is laid out by tile rows, and each tile's elements are
//! contiguous (the layout of the kernel that touches it; another kernel
//! with another tiling sees the same bytes in other tiles).
//!
//! Output stores: a CTA's output tile goes from its SMEM to the L2 or the
//! GDDR7 by a DMA on its own SM's copy engine. The task's range is a store
//! record in the SM's sync-only space (one granule per slot and output
//! buffer), not the written memory: the SM's Neutrino tracks only the
//! store's completion, and the device's Neutrino never sees it. This is a
//! correlation-only convention for the card, whose SMs store into global
//! memory: a Radiance 2 program writes into another node's memory only
//! through a pull by the owner's engine (architecture Section 3.3).
//!
//! CTA contents:
//!
//! * `gemm` (CUTLASS 2.x multistage): C = A B, A M x K, B K x N, tile `bm`
//!   x `bn`, k-tile `bk`, `stages` SMEM buffers. Per k-tile, DMAs of the A
//!   (bm x bk) and B (bk x bn) tiles into buffer kt mod `stages`, then an
//!   Atlas matmul that accumulates into C (it keeps C in the Atlas
//!   accumulator buffer, `--acc-buffer`, until the last k-tile). The DMAs
//!   of k-tile kt + `stages` depend on the matmul of kt. With `split_k` S
//!   (CUTLASS serial split-K) the grid is S slices of every tile, slice
//!   by slice, each over its part of the k-tiles; the CTA of slice s > 0,
//!   after its last matmul and after slice s - 1's store of the tile (a
//!   dependency on that store's record at slice s - 1's SM, which stands
//!   for the card's split-K semaphore in global memory), reads that
//!   partial tile into its SMEM and adds it on the Muon (`split_ops`
//!   lane-ops per element); the slices therefore store the tile in slice
//!   order. Epilogue of the last slice: a Muon task of
//!   `epilogue_ops` lane-ops per element (none at 0). Then the C tile goes
//!   to its memory. When the buffers and C do not fit in the CTA's SMEM,
//!   C shares the buffers' space (CUTLASS's epilogue reuses it).
//! * `softmax`, `layernorm`, `elementwise`: a CTA of `rows_per_cta` rows of
//!   `cols` elements: a DMA per input (`elementwise` has 1 or 2), a Muon
//!   task of rows x cols x `ops_per_elem` lane-ops, the rows out. With
//!   `piece` P > 0 the CTA's rows stream in pieces of P elements, each a
//!   DMA per input, a Muon task of its share of the lane-ops (after its
//!   own loads only) and a store (after its Muon task), through two piece
//!   buffers per input and two output piece buffers per slot (the smallest
//!   SMEM without false dependencies; `row_pieces`). The second pass of a
//!   normalization (the whole row's statistics before any output) is not
//!   modeled. `simt_util` u (0 < u <= 1, default 1) charges the row op's
//!   Muon tasks lane-ops / u, as a utilization of u (exclusive,
//!   architecture Section 5.6); `softmax_util` does the same for
//!   attention's Muon tasks. The kernels report keeps the unscaled
//!   lane-ops (`lane_ops`), the charged ones (`lane_ops_charged`) and the
//!   value (`simt_util`).
//! * `attention` (FlashAttention-2 forward): inputs Q, K, V; Q and O are
//!   batch x heads x seq x head_dim, K and V batch x kv_heads x seq_kv x
//!   head_dim. A CTA is (batch, head, Q tile of `bq` rows), with the Q tile
//!   index fastest. A DMA of Q; per KV tile of `bkv` rows (only those at or
//!   before the Q tile's last row when `causal`), DMAs of K and V into
//!   buffer j mod `stages`, an Atlas S = Q K^T (bq, bkv, head_dim), a
//!   Muon task of `softmax_ops` lane-ops per S element plus `rescale_ops`
//!   per O element (the online softmax), and an Atlas O += P V (bq, head_dim, bkv) kept in the
//!   accumulator buffer until the last KV tile; then a Muon task of
//!   `norm_ops` lane-ops per O element and O out. With `s_in = "acc"`
//!   (default; the card's registers) S (fp32) and O stay in the Atlas
//!   accumulator, the online softmax moves no SMEM bytes, and P V reads P
//!   from SMEM over the K and V space; only Q and the K and V buffers have
//!   SMEM of their own. With `s_in = "smem"` S goes to SMEM, the online
//!   softmax reads it and writes P in place, and O has SMEM of its own.
//!
//! Tensor units: an SM has one Atlas (`atlas`) or one per sub-partition
//! (`atlas0`, `atlas1`, ...); warp w of a CTA runs its matmuls on unit w
//! mod the units. Each unit has its own accumulator (`--acc-buffer` sets
//! every unit's), shared by the SM's CTAs like the SMEM: on a unit, each
//! CTA has `acc_buffer / ctas_per_sm` bytes for the C tiles (gemm) or S
//! and O (attention) of its warps on that unit; more is an error.
//!
//! Every task writes a versioned range; the program sets each version.
//! A task's label (nu_prog::label) is (its role: load, store, matmul,
//! softmax, norm, eltwise or rescale; layer = the kernel's index; unit =
//! the SM; idx = the CTA's index in the kernel).
//! Version counters start again with each kernel: the SMs' and gpcs'
//! programs start again with it. The device's Neutrino holds no versions.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use nu_model::{Addr, Dep, NeutrinoId, Range};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sim_core::{Args, Built, Descriptor, Dtype, NuCtx, OpcodeId, PeId};

use crate::label::{self, Kind};
use crate::micro::{pick, sms_of, Sm};

/// The alignment of every range a task writes (the Neutrino granule).
const GRANULE: u64 = 512;

/// The alignment of a tensor in memory.
const TENSOR_ALIGN: u64 = 4096;

/// Parameters of a kernel sequence (TOML, or `--param key=value`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Params {
    pub tensors: Vec<Tensor>,
    pub cache: String,
    pub kernels: Vec<Kernel>,
    /// The memories of `home = dram` and `home = l2`.
    pub dram: String,
    pub l2: String,
    /// A CTA's slot takes the next CTA once the CTA's compute is done; its
    /// output write is in flight from one of two output buffers (default
    /// true). False: the slot waits for the write.
    pub posted_outputs: bool,
    /// The SMs the CTAs run on: the first `sms` taken round-robin over the
    /// gpcs (0, the default: every SM).
    pub sms: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            tensors: Vec::new(),
            cache: "first_touch".into(),
            kernels: Vec::new(),
            dram: "root.hbm".into(),
            l2: "root.l2".into(),
            posted_outputs: true,
            sms: 0,
        }
    }
}

/// A tensor: its size and the memory it lives in.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub name: String,
    pub bytes: u64,
    #[serde(default = "default_home")]
    pub home: String,
    /// A view: `bytes` at `offset` of the tensor `of` (listed before it),
    /// whose home and L2 copy it shares.
    #[serde(default)]
    pub of: Option<String>,
    #[serde(default)]
    pub offset: u64,
}

fn default_home() -> String {
    "dram".into()
}

/// One kernel. The fields an op does not use are ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Kernel {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub ctas_per_sm: usize,
    /// Element type of every tensor of the kernel (bf16, fp8, f32).
    pub dtype: String,
    // gemm
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub bm: u64,
    pub bn: u64,
    pub bk: u64,
    pub stages: u64,
    /// CTA order over the tiles: `row` (N tiles fastest) or `col`.
    pub raster: String,
    /// Lane-ops per output element of the epilogue (bias, activation), on
    /// the last K slice.
    pub epilogue_ops: u64,
    /// Serial split-K: K slices, each a CTA per tile; slice s > 0 waits
    /// for slice s - 1's tile, reads it and adds it (`split_ops` lane-ops
    /// per element).
    pub split_k: u64,
    pub split_ops: u64,
    // softmax, layernorm, elementwise
    pub rows: u64,
    pub cols: u64,
    pub rows_per_cta: u64,
    pub ops_per_elem: u64,
    /// Elements per piece: a CTA's rows stream through its SMEM in pieces
    /// (one DMA per input, one Muon task and one store each); 0, the
    /// default: the CTA's rows in one piece.
    pub piece: u64,
    /// The row op's Muon utilization (0 < u <= 1, default 1): its tasks
    /// are charged lane-ops / u.
    pub simt_util: f64,
    // attention
    pub batch: u64,
    pub heads: u64,
    /// 0: `heads`.
    pub kv_heads: u64,
    pub seq: u64,
    /// 0: `seq`.
    pub seq_kv: u64,
    pub head_dim: u64,
    pub bq: u64,
    pub bkv: u64,
    pub causal: bool,
    /// Where S, P and O live: `acc` (the card's registers: S and O in the
    /// Atlas accumulator, P read from SMEM over the K and V space) or
    /// `smem` (S, P and O in SMEM of their own).
    pub s_in: String,
    /// When a K or V buffer takes the next tile: `fa2` (K once every warp's
    /// S read it, V once every warp's P V did) or `after_pv` (both after
    /// P V).
    pub kv_release: String,
    /// The CTA's warps. Attention: row groups, each a chain over bq /
    /// warps rows (S, online softmax, P V) on the CTA's K and V tiles
    /// (default 1). Gemm: the CTA's warp tiles (default bm x bn / 4096,
    /// CUTLASS sm80's 64 x 64). Warp w runs its matmuls on tensor unit w
    /// mod the SM's units.
    pub warps: u64,
    pub softmax_ops: u64,
    /// Attention's Muon utilization (0 < u <= 1, default 1).
    pub softmax_util: f64,
    /// Lane-ops per O element per KV tile (the online softmax's rescale).
    pub rescale_ops: u64,
    pub norm_ops: u64,
}

impl Default for Kernel {
    fn default() -> Self {
        Self {
            name: String::new(),
            op: String::new(),
            inputs: Vec::new(),
            output: String::new(),
            ctas_per_sm: 1,
            dtype: "bf16".into(),
            m: 0,
            n: 0,
            k: 0,
            bm: 128,
            bn: 128,
            bk: 32,
            stages: 3,
            raster: "row".into(),
            epilogue_ops: 0,
            split_k: 1,
            split_ops: 2,
            rows: 0,
            cols: 0,
            rows_per_cta: 1,
            ops_per_elem: 1,
            piece: 0,
            simt_util: 1.0,
            batch: 1,
            heads: 1,
            kv_heads: 0,
            seq: 0,
            seq_kv: 0,
            head_dim: 0,
            bq: 128,
            bkv: 64,
            causal: false,
            s_in: "acc".into(),
            kv_release: "fa2".into(),
            warps: 0,
            softmax_ops: 12,
            softmax_util: 1.0,
            rescale_ops: 1,
            norm_ops: 1,
        }
    }
}

/// What one kernel did, by the program's own count.
#[derive(Debug, Clone, Default, Serialize)]
pub struct KernelRec {
    pub name: String,
    pub op: String,
    pub ctas: usize,
    /// The device's marks: the kernel's start, and its end after every CTA.
    pub start: u64,
    pub end: u64,
    pub macs: u64,
    /// Lane-ops of the kernel's Muon tasks: the work, and as charged (the
    /// work over `simt_util` or `softmax_util`).
    pub lane_ops: u64,
    pub lane_ops_charged: u64,
    /// The utilization of the row ops' Muon tasks (`simt_util`) or of
    /// attention's (`softmax_util`); 1 for the others.
    pub simt_util: f64,
    /// Bytes read from the home memory `dram`, and from the L2 range.
    pub dram_read_bytes: u64,
    pub l2_read_bytes: u64,
    /// Bytes written to `dram` and to the L2 range.
    pub dram_write_bytes: u64,
    pub l2_write_bytes: u64,
    /// DMAs that read `dram`.
    pub dram_reads: u64,
}

/// What the program records (the `kernels` object of `--stats`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub cache: String,
    pub sms: usize,
    pub kernels: Vec<KernelRec>,
    /// The end of the last kernel.
    pub end: u64,
    /// Per tensor: (name, bytes read from dram, from the L2, written).
    pub tensors: Vec<TensorRec>,
}

/// A tensor's traffic over the run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TensorRec {
    pub name: String,
    pub home: String,
    /// Its linear address at home, and of its L2 copy (if any).
    pub base: u64,
    pub l2_base: Option<u64>,
    pub dram_read_bytes: u64,
    pub l2_read_bytes: u64,
    pub write_bytes: u64,
}

/// The `kernels` object of `--stats`.
pub fn result(r: &Report, stats: &sim_core::Stats) -> Value {
    let mut out = json!(r);
    // cache = sim: per kernel, each cache's counters between the flush
    // before it and the next flush (or the run's end).
    if r.cache == "sim" {
        for (k, kr) in out["kernels"]
            .as_array_mut()
            .into_iter()
            .flatten()
            .enumerate()
        {
            let per: Vec<Value> = stats
                .caches
                .iter()
                .filter_map(|c| {
                    let from = c.flushes.get(k)?.counts;
                    let to = c.flushes.get(k + 1).map_or(c.counts, |f| f.counts);
                    Some(json!({
                        "name": c.name,
                        "read_hits": to.read_hits - from.read_hits,
                        "read_misses": to.read_misses - from.read_misses,
                        "write_hits": to.write_hits - from.write_hits,
                        "write_allocs": to.write_allocs - from.write_allocs,
                        "allocs": to.allocs - from.allocs,
                        "clean_evictions": to.clean_evictions - from.clean_evictions,
                        "dirty_evictions": to.dirty_evictions - from.dirty_evictions,
                        "writeback_bytes": to.writeback_bytes - from.writeback_bytes,
                        "peak_lines": to.peak_lines,
                    }))
                })
                .collect();
            kr["cache"] = json!(per);
        }
    }
    out
}

/// A task of a CTA and its role (the label's kind).
struct Task {
    desc: Descriptor,
    kind: Kind,
}

/// A CTA: its tasks in order, the versions that end its compute, and the
/// version of its output write (its last task).
struct Cta {
    tasks: Vec<Task>,
    compute: Vec<(u64, u64)>,
    /// Its output writes (one per piece of a row op in pieces).
    done: Vec<(u64, u64)>,
}

/// A kernel ready to run: per SM (index into the run's SMs), its CTAs in
/// launch order.
struct Plan {
    per_sm: Vec<Vec<Cta>>,
    ctas_per_sm: usize,
    /// Posted outputs: a slot's next CTA starts once the CTA's compute is
    /// done, not its output write.
    posted: bool,
}

/// A tensor's place: home `l2`, and the bases. A view has only `view`
/// and `bytes`; the other fields are its root tensor's.
struct Place {
    /// (the tensor it is a view of, its offset there).
    view: Option<(usize, u64)>,
    in_l2: bool,
    base: u64,
    l2_base: Option<u64>,
    bytes: u64,
    /// Under `first_touch`: the byte ranges of its L2 copy that hold data,
    /// sorted and disjoint.
    covered: Vec<(u64, u64)>,
}

impl Place {
    fn is_covered(&self, off: u64, len: u64) -> bool {
        self.covered
            .iter()
            .any(|&(a, b)| a <= off && off + len <= b)
    }

    fn cover(&mut self, off: u64, len: u64) {
        let (mut a, mut b) = (off, off + len);
        self.covered.retain(|&(x, y)| {
            if y < a || x > b {
                true
            } else {
                a = a.min(x);
                b = b.max(y);
                false
            }
        });
        let i = self.covered.partition_point(|&(x, _)| x < a);
        self.covered.insert(i, (a, b));
    }
}

/// The builder's state.
struct Builder {
    sms: Vec<Sm>,
    /// Per SM: its sync-only range, where the store records are.
    records: Vec<Range>,
    dma: OpcodeId,
    matmul: Option<OpcodeId>,
    vector: Option<OpcodeId>,
    places: Vec<Place>,
    names: HashMap<String, usize>,
    tensor_recs: Vec<TensorRec>,
    /// Version counters of the current kernel, by range base.
    versions: HashMap<u64, u64>,
    /// The current kernel's index: the layer field of its tasks' labels.
    index: u64,
    /// The current CTA: (its index in the kernel, its SM): the idx and unit
    /// fields of its tasks' labels.
    cta: (u32, u32),
    /// The kind of the Muon tasks being built.
    muon_kind: Kind,
    /// The SMs' Atlas accumulator buffer (bytes).
    acc_buffer: u64,
    /// The utilization of the Muon tasks being built (`simt_util`,
    /// `softmax_util`; 1 otherwise).
    util: f64,
    /// Posted outputs (two output buffers per slot).
    posted: bool,
    /// Per (SM, slot): the output writes of the current kernel's CTAs.
    outs: HashMap<(usize, usize), Vec<Vec<Dep>>>,
    /// Row ops in pieces, per (SM, slot): the pieces so far and the store
    /// that last read each of the two output piece buffers.
    rings: HashMap<(usize, usize), Ring>,
    /// `cache = per_kernel`: the L2 range in which each kernel's copies
    /// are placed.
    per_kernel: Option<Range>,
    rec: KernelRec,
}

/// A slot's output piece buffers (row ops in pieces): the pieces so far
/// and the store that last read each of the two buffers.
type Ring = (u64, [Option<Dep>; 2]);

/// A gpc of the run: its Neutrino, its sync range (the `done` of its SMs'
/// programs) and its SMs (index into the run's SMs, Neutrino).
type Gpc = (NeutrinoId, Range, Vec<(usize, NeutrinoId)>);

/// A built run.
pub struct Run {
    plans: Vec<(String, Plan)>,
    /// `cache = sim`: the device empties the caches before each kernel.
    flush: bool,
    pub report: Rc<RefCell<Report>>,
    /// Per gpc: its Neutrino, its sync range, its SMs (index, Neutrino).
    gpcs: Vec<Gpc>,
    root_sync: Range,
}

/// The range a task writes for `bytes` at `base`: at least one granule.
fn dst_range(base: u64, bytes: u64) -> Range {
    Range::from_len(base, bytes.max(1).div_ceil(GRANULE) * GRANULE)
}

/// The first base at or after `at`, in granules, that is in neither
/// `fixed` nor `taken`.
fn own(mut at: u64, fixed: &[u64], taken: &[u64]) -> u64 {
    while fixed.contains(&at) || taken.contains(&at) {
        at += GRANULE;
    }
    at
}

/// A version handle's (address, version).
fn handle(d: Dep) -> (u64, u64) {
    match d {
        Dep::Versioned { addr, version } => (addr, version),
        _ => unreachable!("a versioned handle"),
    }
}

fn elem_bytes(dtype: &str) -> Result<(Dtype, u64), String> {
    match dtype {
        "bf16" => Ok((Dtype::Bf16, 2)),
        "fp8" => Ok((Dtype::Fp8, 1)),
        "f32" => Ok((Dtype::F32, 4)),
        d => Err(format!("unknown dtype {d:?} (bf16, fp8, f32)")),
    }
}

/// The linear range of memory `name`: the source range of a hash link
/// that maps onto it, else its own range.
fn linear(b: &Built, name: &str) -> Result<Range, String> {
    let r = b
        .try_memory(name)
        .ok_or_else(|| format!("kernels: no memory {name:?}"))?;
    Ok(b.layout
        .links
        .iter()
        .filter_map(|l| l.hash.as_ref())
        .find(|h| h.to_range.base <= r.base && r.bound <= h.to_range.bound)
        .map_or(r, |h| h.range))
}

/// The warp grid (warps_m, warps_n) of a bm x bn block with `warps`
/// warps: 64 x 64 warp tiles when they make up the warps (CUTLASS sm80),
/// else the factorization whose warp tile is closest to square; `None`
/// when no factorization tiles the block evenly.
fn warp_grid(bm: u64, bn: u64, warps: u64) -> Option<(u64, u64)> {
    if bm.is_multiple_of(64) && bn.is_multiple_of(64) && (bm / 64) * (bn / 64) == warps {
        return Some((bm / 64, bn / 64));
    }
    (1..=warps)
        .filter(|&wm| {
            warps.is_multiple_of(wm) && bm.is_multiple_of(wm) && bn.is_multiple_of(warps / wm)
        })
        .min_by_key(|wm| (bm / wm).abs_diff(bn / (warps / wm)))
        .map(|wm| (wm, warps / wm))
}

/// Tile (ti, tj) of an R x C matrix tiled by h x w: (offset, rows, cols)
/// in elements, with the tiles laid out by tile rows and each contiguous.
fn tile(rows: u64, cols: u64, h: u64, w: u64, ti: u64, tj: u64) -> (u64, u64, u64) {
    let (r0, c0) = (ti * h, tj * w);
    let (th, tw) = (h.min(rows - r0), w.min(cols - c0));
    (r0 * cols + th * c0, th, tw)
}

impl Builder {
    /// Sets the next version of `desc.dst`; returns its handle.
    fn version(&mut self, desc: &mut Descriptor) -> (u64, u64) {
        let base = desc.dst.base;
        let v = self.versions.entry(base).or_insert(0);
        *v += 1;
        desc.versioned = true;
        desc.inferred_version = Some(*v);
        (base, *v)
    }

    fn tensor(&self, name: &str) -> Result<usize, String> {
        self.names
            .get(name)
            .copied()
            .ok_or_else(|| format!("kernels: unknown tensor {name:?}"))
    }

    /// Tensor `t`'s root tensor and its offset there.
    fn root_of(&self, mut t: usize) -> (usize, u64) {
        let mut off = 0;
        while let Some((parent, o)) = self.places[t].view {
            off += o;
            t = parent;
        }
        (t, off)
    }

    /// Checks that `[off, off + len)` is inside tensor `t`; its root and
    /// the offset there.
    fn locate(&self, t: usize, off: u64, len: u64, what: &str) -> Result<(usize, u64), String> {
        if off + len > self.places[t].bytes {
            return Err(format!(
                "kernels: a {what} of [{off}, {}) exceeds tensor {} of {} B",
                off + len,
                self.tensor_recs[t].name,
                self.places[t].bytes
            ));
        }
        let (r, o) = self.root_of(t);
        Ok((r, o + off))
    }

    /// The address a read of `len` bytes at offset `off` of tensor `t`
    /// comes from, counted.
    fn read_at(&mut self, t: usize, off: u64, len: u64) -> Result<u64, String> {
        let (r, off) = self.locate(t, off, len, "read")?;
        let p = &mut self.places[r];
        let from_l2 = p.in_l2 || (p.l2_base.is_some() && p.is_covered(off, len));
        let rec = &mut self.tensor_recs[t];
        if from_l2 {
            rec.l2_read_bytes += len;
            self.rec.l2_read_bytes += len;
        } else {
            rec.dram_read_bytes += len;
            self.rec.dram_read_bytes += len;
            self.rec.dram_reads += 1;
            if p.l2_base.is_some() {
                p.cover(off, len);
            }
        }
        Ok(match (from_l2, p.l2_base) {
            (true, Some(l2)) => l2 + off,
            _ => p.base + off,
        })
    }

    /// The address a write of `len` bytes at offset `off` of tensor `t`
    /// goes to, counted.
    fn write_at(&mut self, t: usize, off: u64, len: u64) -> Result<u64, String> {
        let (r, off) = self.locate(t, off, len, "write")?;
        self.tensor_recs[t].write_bytes += len;
        let p = &mut self.places[r];
        if p.in_l2 || p.l2_base.is_some() {
            self.rec.l2_write_bytes += len;
        } else {
            self.rec.dram_write_bytes += len;
        }
        Ok(match p.l2_base {
            Some(l2) => {
                p.cover(off, len);
                l2 + off
            }
            None => p.base + off,
        })
    }

    /// A DMA of `len` bytes from `src` into `dst` on `pe`; its range
    /// is `dst_len` bytes (an SMEM buffer keeps one length).
    fn dma(
        &mut self,
        pe: PeId,
        src: u64,
        (dst, dst_len): (u64, u64),
        len: u64,
        deps: Vec<Dep>,
    ) -> Task {
        Task {
            kind: Kind::Load,
            desc: Descriptor {
                opcode: self.dma,
                pe: Some(pe),
                dst: dst_range(dst, dst_len),
                versioned: true,
                inferred_version: None,
                deps,
                label: None,
                args: Args::Dma {
                    src: Addr(src),
                    dst: Addr(dst),
                    bytes: len,
                    tile_bytes: len,
                },
            },
        }
    }

    /// The store of `len` bytes from SMEM `src` to `dst` (in the L2 or the
    /// GDDR7) on SM `s`'s copy engine, for output buffer `parity` of slot
    /// `slot`: its range is that buffer's store record in the SM's sync
    /// space.
    #[allow(clippy::too_many_arguments)]
    fn store(
        &mut self,
        s: usize,
        slot: usize,
        parity: usize,
        src: u64,
        dst: u64,
        len: u64,
        deps: Vec<Dep>,
    ) -> Result<Task, String> {
        let sync = self.records[s];
        let at = sync.base + (2 * slot + parity) as u64 * GRANULE;
        if at + GRANULE > sync.bound {
            return Err(format!(
                "kernels: {}: no sync space for the store record of slot {slot}",
                self.sms[s].path
            ));
        }
        let mut t = self.dma(self.sms[s].dma, src, (dst, len), len, deps);
        t.desc.dst = Range::from_len(at, GRANULE);
        t.kind = Kind::Store;
        Ok(t)
    }

    /// Appends `t` to `tasks` with its version; returns its handle.
    fn push(&mut self, tasks: &mut Vec<Task>, mut t: Task) -> Dep {
        let (addr, version) = self.version(&mut t.desc);
        // kind, layer = the kernel's index, unit = the SM, idx = the CTA.
        let (cta, sm) = self.cta;
        t.desc.label = Some(label::label(t.kind, 0, self.index as u32, sm, cta));
        tasks.push(t);
        Dep::Versioned { addr, version }
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &mut self,
        (sm, unit): (&Sm, usize),
        (a, lda): (u64, u64),
        (b, ldb): (u64, u64),
        (c, c_len, ldc): (u64, u64, u64),
        (m, n, k): (u64, u64, u64),
        (dtype, out_dtype): (Dtype, Dtype),
        accumulate: bool,
        keep: bool,
        deps: Vec<Dep>,
    ) -> Result<Task, String> {
        if sm.atlas.is_empty() {
            return Err(format!("{}: no atlas", sm.path));
        }
        let atlas = sm.atlas[unit % sm.atlas.len()];
        let opcode = self.matmul.ok_or("kernels: no atlas.matmul profile")?;
        self.rec.macs += m * n * k;
        Ok(Task {
            kind: Kind::Matmul,
            desc: Descriptor {
                opcode,
                pe: Some(atlas),
                dst: dst_range(c, c_len),
                versioned: true,
                inferred_version: None,
                deps,
                label: None,
                args: Args::Matmul {
                    a: Addr(a),
                    b: Addr(b),
                    a2: None,
                    b2: None,
                    lda,
                    ldb,
                    ldc,
                    m,
                    n,
                    k,
                    in_dtype: dtype,
                    b_dtype: dtype,
                    out_dtype,
                    accumulate,
                    keep,
                    tile: [m, n],
                },
            },
        })
    }

    /// A Muon task of `ops` lane-ops that reads `bytes_in` at `src` and
    /// writes `bytes_out` at `dst`.
    #[allow(clippy::too_many_arguments)]
    fn vector(
        &mut self,
        sm: &Sm,
        ops: u64,
        src: u64,
        bytes_in: u64,
        (dst, dst_len): (u64, u64),
        bytes_out: u64,
        deps: Vec<Dep>,
    ) -> Result<Task, String> {
        let muon = sm.muon.ok_or_else(|| format!("{}: no muon", sm.path))?;
        let opcode = self.vector.ok_or("kernels: no muon.vector profile")?;
        // A utilization below 1 charges the lane-ops over it (exclusive
        // utilization, architecture Section 5.6); the record keeps the work.
        self.rec.lane_ops += ops;
        let ops = (ops as f64 / self.util).ceil() as u64;
        self.rec.lane_ops_charged += ops;
        Ok(Task {
            kind: self.muon_kind,
            desc: Descriptor {
                opcode,
                pe: Some(muon),
                dst: dst_range(dst, dst_len),
                versioned: true,
                inferred_version: None,
                deps,
                label: None,
                args: Args::Vector {
                    ops,
                    bytes_in,
                    bytes_out,
                    src: Some(Addr(src)),
                },
            },
        })
    }

    /// The SMEM of slot `slot` of `ctas_per_sm` on `sm`: (base, bytes).
    fn slot(&self, sm: &Sm, slot: usize, ctas_per_sm: usize) -> (u64, u64) {
        let size = sm.smem.len() / ctas_per_sm as u64 / GRANULE * GRANULE;
        (sm.smem.base + slot as u64 * size, size)
    }

    /// Checks that a CTA's share of a tensor unit's accumulator
    /// (`acc_buffer / ctas_per_sm`) holds the `bytes` its warps on that
    /// unit keep there.
    fn acc_fits(&self, k: &Kernel, bytes: u64, what: &str) -> Result<(), String> {
        let share = self.acc_buffer / k.ctas_per_sm as u64;
        if bytes > share {
            return Err(format!(
                "{}: {what} of {bytes} B per tensor unit do not fit in the CTA's {share} B of the unit's accumulator (--acc-buffer {} per unit over {} CTAs per SM)",
                k.name, self.acc_buffer, k.ctas_per_sm
            ));
        }
        Ok(())
    }

    /// The output buffer (0 or 1) of the next CTA in slot `slot` of SM
    /// `s`, and the output write it waits for (the CTA two back there).
    /// Without posted outputs: buffer 0, no wait (the slot waits for the
    /// write).
    fn out_slot(&self, s: usize, slot: usize) -> (usize, Vec<Dep>) {
        if !self.posted {
            return (0, Vec::new());
        }
        let h = self.outs.get(&(s, slot)).map_or(&[][..], |v| v.as_slice());
        let n = h.len();
        (n % 2, n.checked_sub(2).map_or(Vec::new(), |i| h[i].clone()))
    }

    /// A CTA of slot `slot` of SM `s`.
    fn cta(
        &mut self,
        s: usize,
        slot: usize,
        tasks: Vec<Task>,
        compute: Vec<Dep>,
        done: Vec<Dep>,
    ) -> Cta {
        self.outs.entry((s, slot)).or_default().push(done.clone());
        Cta {
            tasks,
            compute: compute.into_iter().map(handle).collect(),
            done: done.into_iter().map(handle).collect(),
        }
    }

    /// `cache = per_kernel`, before kernel `k`: the L2 holds nothing, and
    /// the root tensors `need` (every tensor `k` names, or its root, when
    /// `None`, at addresses not checked against `l2`) get an L2 copy in
    /// `l2`.
    fn cold_l2(&mut self, k: &Kernel, l2: Range, need: Option<&[usize]>) -> Result<(), String> {
        for p in &mut self.places {
            p.covered.clear();
            p.l2_base = None;
        }
        let mut at = l2.base;
        for name in k.inputs.iter().chain(std::iter::once(&k.output)) {
            let (r, _) = self.root_of(self.tensor(name)?);
            if self.places[r].l2_base.is_some() || need.is_some_and(|n| !n.contains(&r)) {
                continue;
            }
            let base = at.div_ceil(TENSOR_ALIGN) * TENSOR_ALIGN;
            let bytes = self.places[r].bytes;
            if need.is_some() && base + bytes > l2.bound {
                return Err(format!(
                    "{}: the L2 copies of its tensors do not fit in {} B",
                    k.name,
                    l2.len()
                ));
            }
            at = base + bytes;
            self.places[r].l2_base = Some(base);
            self.tensor_recs[r].l2_base = Some(base);
        }
        Ok(())
    }

    /// `cache = per_kernel`: before kernel `k`, the root tensors that `k`
    /// writes or reads again (a plan of `k` with an L2 copy of every
    /// tensor; its counts are discarded).
    fn kernel_needs(&mut self, k: &Kernel, l2: Range) -> Result<Vec<usize>, String> {
        let saved = self.tensor_recs.clone();
        self.cold_l2(k, l2, None)?;
        self.versions.clear();
        self.kernel(k)?;
        self.versions.clear();
        self.rec = KernelRec::default();
        let mut need = Vec::new();
        for (t, (now, before)) in self.tensor_recs.iter().zip(&saved).enumerate() {
            if now.l2_read_bytes > before.l2_read_bytes || now.write_bytes > before.write_bytes {
                let (r, _) = self.root_of(t);
                if !need.contains(&r) {
                    need.push(r);
                }
            }
        }
        self.tensor_recs = saved;
        Ok(need)
    }

    /// A row op's CTA in pieces (`piece` elements each) on slot `slot` of
    /// SM `s` (SMEM `[base, base + size)`): rows `r0 .. r0 + rh`. The SMEM
    /// holds two piece buffers per input and two output piece buffers
    /// (overlapping, at bases of their own, when they do not fit side by
    /// side) that the slot's pieces use in turn: piece i's loads go into input
    /// buffer i mod 2 after piece i - 2's Muon task read it; a piece's
    /// Muon task writes the slot's next output buffer after the store
    /// that last read it (of this CTA or the one before); a piece's store
    /// follows its Muon task.
    #[allow(clippy::too_many_arguments)]
    fn row_pieces(
        &mut self,
        k: &Kernel,
        ins: &[usize],
        out: usize,
        (s, slot): (usize, usize),
        (base, size): (u64, u64),
        r0: u64,
        rh: u64,
        es: u64,
    ) -> Result<Cta, String> {
        let sm = self.sms[s].clone();
        let n_in = ins.len() as u64;
        let pb = (k.piece * es).div_ceil(GRANULE) * GRANULE;
        let nb = 2 * n_in + 2;
        // The buffers side by side, or, when they do not fit, at bases
        // spread over the slot so that each ends inside it (overlapping:
        // the program models time, not data).
        let step = if nb * pb <= size {
            pb
        } else if pb <= size && (size - pb) / (nb - 1) >= GRANULE {
            (size - pb) / (nb - 1) / GRANULE * GRANULE
        } else {
            return Err(format!(
                "{}: {nb} piece buffers of {pb} B do not fit in {size} B of SMEM per CTA",
                k.name
            ));
        };
        let in_at = |x: u64, b: u64| base + (2 * x + b) * step;
        let out_at = |b: u64| base + (2 * n_in + b) * step;
        let total = rh * k.cols;
        let mut tasks = Vec::new();
        let (mut muons, mut stores): (Vec<Dep>, Vec<Dep>) = (Vec::new(), Vec::new());
        for i in 0..total.div_ceil(k.piece) {
            let e0 = i * k.piece;
            let ne = k.piece.min(total - e0);
            let (off, len, b) = ((r0 * k.cols + e0) * es, ne * es, i % 2);
            let war: Vec<Dep> = i
                .checked_sub(2)
                .map_or(Vec::new(), |j| vec![muons[j as usize]]);
            let mut deps = Vec::new();
            for (x, &t_in) in ins.iter().enumerate() {
                let src = self.read_at(t_in, off, len)?;
                let t = self.dma(sm.dma, src, (in_at(x as u64, b), pb), len, war.clone());
                deps.push(self.push(&mut tasks, t));
            }
            let ring = self.rings.entry((s, slot)).or_default();
            let j = (ring.0 % 2) as usize;
            ring.0 += 1;
            deps.extend(ring.1[j]);
            let t = self.vector(
                &sm,
                ne * k.ops_per_elem,
                in_at(0, b),
                len * n_in,
                (out_at(j as u64), pb),
                len,
                deps,
            )?;
            let m = self.push(&mut tasks, t);
            muons.push(m);
            let dst = self.write_at(out, off, len)?;
            let t = self.store(s, slot, j, out_at(j as u64), dst, len, vec![m])?;
            let st = self.push(&mut tasks, t);
            stores.push(st);
            self.rings.get_mut(&(s, slot)).expect("ring").1[j] = Some(st);
        }
        Ok(self.cta(s, slot, tasks, muons, stores))
    }

    /// Plans kernel `k`.
    fn kernel(&mut self, k: &Kernel) -> Result<Plan, String> {
        for (what, u) in [("simt_util", k.simt_util), ("softmax_util", k.softmax_util)] {
            if !(u > 0.0 && u <= 1.0) {
                return Err(format!("{}: {what} {u} is not in (0, 1]", k.name));
            }
        }
        self.util = match k.op.as_str() {
            "softmax" | "layernorm" | "elementwise" => k.simt_util,
            "attention" => k.softmax_util,
            _ => 1.0,
        };
        self.rec.simt_util = self.util;
        self.muon_kind = match k.op.as_str() {
            "softmax" | "attention" => Kind::Softmax,
            "layernorm" => Kind::Norm,
            _ => Kind::Eltwise,
        };
        self.outs.clear();
        self.rings.clear();
        if k.ctas_per_sm == 0 {
            return Err(format!("{}: ctas_per_sm is positive", k.name));
        }
        let (dtype, es) = elem_bytes(&k.dtype)?;
        let n_sms = self.sms.len();
        let mut per_sm: Vec<Vec<Cta>> = (0..n_sms).map(|_| Vec::new()).collect();
        let out = self.tensor(&k.output)?;
        let ins: Vec<usize> = k
            .inputs
            .iter()
            .map(|i| self.tensor(i))
            .collect::<Result<_, _>>()?;
        let need = |want: usize| -> Result<(), String> {
            if ins.len() != want {
                return Err(format!(
                    "{}: {} takes {want} input(s), got {}",
                    k.name,
                    k.op,
                    ins.len()
                ));
            }
            Ok(())
        };
        let mut n_ctas = 0usize;
        // The next CTA's SM and slot.
        let place = |n_ctas: &mut usize| {
            let i = *n_ctas;
            *n_ctas += 1;
            let sm = i % n_sms;
            let local = i / n_sms;
            (sm, local % k.ctas_per_sm)
        };
        match k.op.as_str() {
            "gemm" => {
                need(2)?;
                let (m, n, kk) = (k.m, k.n, k.k);
                if m == 0 || n == 0 || kk == 0 || k.bm == 0 || k.bn == 0 || k.bk == 0 {
                    return Err(format!("{}: m, n, k, bm, bn, bk are positive", k.name));
                }
                let stages = k.stages.max(1);
                let split = k.split_k.max(1);
                let (mt, nt, kt_n) = (m.div_ceil(k.bm), n.div_ceil(k.bn), kk.div_ceil(k.bk));
                if split > kt_n {
                    return Err(format!(
                        "{}: split_k {split} exceeds {kt_n} k-tiles",
                        k.name
                    ));
                }
                let per_slice = kt_n.div_ceil(split);
                let tiles: Vec<(u64, u64)> = match k.raster.as_str() {
                    "row" => (0..mt).flat_map(|i| (0..nt).map(move |j| (i, j))).collect(),
                    "col" => (0..nt).flat_map(|j| (0..mt).map(move |i| (i, j))).collect(),
                    r => return Err(format!("{}: unknown raster {r:?} (row, col)", k.name)),
                };
                // Serial split-K: every tile of slice 0, then of slice 1, ...
                // The last write of each output tile.
                let mut written: HashMap<(u64, u64), Dep> = HashMap::new();
                let a_buf = (k.bm * k.bk * es).div_ceil(GRANULE) * GRANULE;
                let b_buf = (k.bk * k.bn * es).div_ceil(GRANULE) * GRANULE;
                let c_buf = (k.bm * k.bn * es).div_ceil(GRANULE) * GRANULE;
                let bufs = stages * (a_buf + b_buf);
                // The CTA's warps (CUTLASS sm80: 64 x 64 warp tiles by
                // default): warp w covers C rows (w mod warps_m) x wtm and
                // columns (w div warps_m) x wtn, on tensor unit w mod units.
                let warps = if k.warps == 0 {
                    (k.bm * k.bn / 4096).max(1)
                } else {
                    k.warps
                };
                let (warps_m, warps_n) = warp_grid(k.bm, k.bn, warps).ok_or_else(|| {
                    format!(
                        "{}: {warps} warps do not tile a {} x {} block",
                        k.name, k.bm, k.bn
                    )
                })?;
                let (wtm, wtn) = (k.bm / warps_m, k.bn / warps_n);
                let units = self.sms[0].atlas.len().max(1) as u64;
                let w_buf = (wtm * wtn * es).div_ceil(GRANULE) * GRANULE;
                self.acc_fits(
                    k,
                    warps.div_ceil(units) * wtm * wtn * es,
                    "the warps' C tiles",
                )?;
                for sl in 0..split {
                    let (kt0, kt1) = (sl * per_slice, ((sl + 1) * per_slice).min(kt_n));
                    for &(i, j) in &tiles {
                        let (s, slot) = place(&mut n_ctas);
                        self.cta = ((n_ctas - 1) as u32, s as u32);
                        let sm = self.sms[s].clone();
                        let (base, size) = self.slot(&sm, slot, k.ctas_per_sm);
                        let bases: Vec<u64> = (0..stages)
                            .flat_map(|st| {
                                let a = base + st * (a_buf + b_buf);
                                [a, a + a_buf]
                            })
                            .collect();
                        // C after the buffers, or over them (one granule
                        // in, so that its base is not a buffer's).
                        let c0 = if bufs + c_buf <= size {
                            base + bufs
                        } else if bufs <= size && GRANULE + c_buf <= size {
                            base + GRANULE
                        } else {
                            return Err(format!(
                                "{}: {stages} stages of {} B and C of {c_buf} B do not fit in {size} B of SMEM per CTA",
                                k.name,
                                a_buf + b_buf
                            ));
                        };
                        // The second output buffer: after the first, or
                        // over the stage space (a base of its own).
                        let c1 = if c0 + 2 * c_buf <= base + size {
                            c0 + c_buf
                        } else {
                            own(base + GRANULE, &bases, &[c0])
                        };
                        if c1 + c_buf > base + size {
                            return Err(format!(
                                "{}: a second C buffer of {c_buf} B does not fit in {size} B of SMEM per CTA",
                                k.name
                            ));
                        }
                        // The split-K partial tile: one base for both
                        // output buffers' CTAs.
                        let top = c0.max(c1);
                        let mut p_at = if top + 2 * c_buf <= base + size {
                            top + c_buf
                        } else {
                            base + (size - c_buf) / GRANULE * GRANULE
                        };
                        while p_at == c0 || p_at == c1 || bases.contains(&p_at) {
                            p_at -= GRANULE;
                        }
                        // Each warp's C handle, for both output buffers:
                        // bases of their own, over the C space (the program
                        // models time, not data).
                        let mut fixed = bases.clone();
                        fixed.extend([c0, c1, p_at]);
                        let mut taken = Vec::new();
                        let mut handles = [Vec::new(), Vec::new()];
                        for (par, c) in [c0, c1].into_iter().enumerate() {
                            for w in 0..warps {
                                let h = own(c + w * GRANULE, &fixed, &taken);
                                if h + w_buf > base + size {
                                    return Err(format!(
                                        "{}: the warps' C handles do not fit in {size} B of SMEM per CTA",
                                        k.name
                                    ));
                                }
                                taken.push(h);
                                handles[par].push(h);
                            }
                        }
                        let (parity, two_back) = self.out_slot(s, slot);
                        let c_at = [c0, c1][parity];
                        let mut tasks = Vec::new();
                        let (c_off, tm, tn) = tile(m, n, k.bm, k.bn, i, j);
                        // Per k-tile: every warp's matmul (None for a warp
                        // outside an edge tile).
                        let mut mm: Vec<Vec<Option<Dep>>> = Vec::new();
                        for kt in kt0..kt1 {
                            let x = kt - kt0;
                            let st = x % stages;
                            let a_at = bases[2 * st as usize];
                            let b_at = a_at + a_buf;
                            // Every warp's matmul of k-tile x - stages read
                            // these buffers.
                            let war: Vec<Dep> = if x >= stages {
                                mm[(x - stages) as usize]
                                    .iter()
                                    .flatten()
                                    .copied()
                                    .collect()
                            } else {
                                Vec::new()
                            };
                            let (a_off, _, tk) = tile(m, kk, k.bm, k.bk, i, kt);
                            let a_src = self.read_at(ins[0], a_off * es, tm * tk * es)?;
                            let t =
                                self.dma(sm.dma, a_src, (a_at, a_buf), tm * tk * es, war.clone());
                            let a_dep = self.push(&mut tasks, t);
                            let (b_off, _, _) = tile(kk, n, k.bk, k.bn, kt, j);
                            let b_src = self.read_at(ins[1], b_off * es, tk * tn * es)?;
                            let t = self.dma(sm.dma, b_src, (b_at, b_buf), tk * tn * es, war);
                            let b_dep = self.push(&mut tasks, t);
                            let last = kt + 1 == kt1;
                            let mut row = Vec::new();
                            for w in 0..warps {
                                let (r0, c0w) = ((w % warps_m) * wtm, (w / warps_m) * wtn);
                                if r0 >= tm || c0w >= tn {
                                    row.push(None);
                                    continue;
                                }
                                let (rows, cols) = (wtm.min(tm - r0), wtn.min(tn - c0w));
                                let mut deps = vec![a_dep, b_dep];
                                deps.extend(mm.last().and_then(|p| p[w as usize]));
                                // The last matmul writes C: after the write
                                // of the CTA two back in this slot read it.
                                if last {
                                    deps.extend(two_back.iter().copied());
                                }
                                let t = self.matmul(
                                    (&sm, w as usize),
                                    (a_at + r0 * k.bk * es, k.bk),
                                    (b_at + c0w * es, k.bn),
                                    (handles[parity][w as usize], w_buf, k.bn),
                                    (rows, cols, tk),
                                    (dtype, dtype),
                                    x > 0,
                                    !last,
                                    deps,
                                )?;
                                row.push(Some(self.push(&mut tasks, t)));
                            }
                            mm.push(row);
                        }
                        let mut c_deps: Vec<Dep> = mm
                            .last()
                            .expect("a k-tile")
                            .iter()
                            .flatten()
                            .copied()
                            .collect();
                        let c_bytes = tm * tn * es;
                        if sl > 0 {
                            // The partial tile of slice sl - 1, after it is
                            // stored and after this slice's last matmuls.
                            let prev = written[&(i, j)];
                            let src = self.read_at(out, c_off * es, c_bytes)?;
                            let mut deps = c_deps.clone();
                            deps.push(prev);
                            let t = self.dma(sm.dma, src, (p_at, c_buf), c_bytes, deps);
                            let p_dep = self.push(&mut tasks, t);
                            let mut deps = c_deps.clone();
                            deps.push(p_dep);
                            let t = self.vector(
                                &sm,
                                tm * tn * k.split_ops,
                                p_at,
                                2 * c_bytes,
                                (c_at, c_buf),
                                c_bytes,
                                deps,
                            )?;
                            c_deps = vec![self.push(&mut tasks, t)];
                        }
                        if k.epilogue_ops > 0 && sl + 1 == split {
                            let t = self.vector(
                                &sm,
                                tm * tn * k.epilogue_ops,
                                c_at,
                                c_bytes,
                                (c_at, c_buf),
                                c_bytes,
                                c_deps,
                            )?;
                            c_deps = vec![self.push(&mut tasks, t)];
                        }
                        let dst = self.write_at(out, c_off * es, c_bytes)?;
                        let t = self.store(s, slot, parity, c_at, dst, c_bytes, c_deps.clone())?;
                        let done = self.push(&mut tasks, t);
                        written.insert((i, j), done);
                        per_sm[s].push(self.cta(s, slot, tasks, c_deps, vec![done]));
                    }
                }
            }
            "softmax" | "layernorm" | "elementwise" => {
                if k.op == "elementwise" {
                    if ins.is_empty() || ins.len() > 2 {
                        return Err(format!("{}: elementwise takes 1 or 2 inputs", k.name));
                    }
                } else {
                    need(1)?;
                }
                if k.rows == 0 || k.cols == 0 || k.rows_per_cta == 0 {
                    return Err(format!("{}: rows, cols, rows_per_cta are positive", k.name));
                }
                for c in 0..k.rows.div_ceil(k.rows_per_cta) {
                    let (s, slot) = place(&mut n_ctas);
                    self.cta = ((n_ctas - 1) as u32, s as u32);
                    let sm = self.sms[s].clone();
                    let (base, size) = self.slot(&sm, slot, k.ctas_per_sm);
                    let r0 = c * k.rows_per_cta;
                    let rh = k.rows_per_cta.min(k.rows - r0);
                    let len = rh * k.cols * es;
                    let buf = (k.rows_per_cta * k.cols * es).div_ceil(GRANULE) * GRANULE;
                    let n_in = ins.len() as u64;
                    if k.piece > 0 {
                        let cta =
                            self.row_pieces(k, &ins, out, (s, slot), (base, size), r0, rh, es)?;
                        per_sm[s].push(cta);
                        continue;
                    }
                    if buf * n_in > size {
                        return Err(format!(
                            "{}: {} input tile(s) of {len} B do not fit in {size} B of SMEM per CTA",
                            k.name,
                            ins.len()
                        ));
                    }
                    let in_bases: Vec<u64> = (0..n_in).map(|x| base + x * buf).collect();
                    // The output buffers: after the inputs, or over them
                    // (bases of their own).
                    let (parity, two_back) = self.out_slot(s, slot);
                    let ob = if !self.posted {
                        base
                    } else if (n_in + 2) * buf <= size {
                        base + (n_in + parity as u64) * buf
                    } else {
                        let o0 = own(base + GRANULE, &in_bases, &[]);
                        let o1 = own(o0 + GRANULE, &in_bases, &[o0]);
                        let o = [o0, o1][parity];
                        if o + buf > base + size {
                            return Err(format!(
                                "{}: an output tile of {len} B does not fit in {size} B of SMEM per CTA",
                                k.name
                            ));
                        }
                        o
                    };
                    let mut tasks = Vec::new();
                    let mut deps = Vec::new();
                    for (x, &t_in) in ins.iter().enumerate() {
                        let src = self.read_at(t_in, r0 * k.cols * es, len)?;
                        let t = self.dma(sm.dma, src, (in_bases[x], buf), len, Vec::new());
                        deps.push(self.push(&mut tasks, t));
                    }
                    deps.extend(two_back);
                    let t = self.vector(
                        &sm,
                        rh * k.cols * k.ops_per_elem,
                        base,
                        len * n_in,
                        (ob, buf),
                        len,
                        deps,
                    )?;
                    let v = self.push(&mut tasks, t);
                    let dst = self.write_at(out, r0 * k.cols * es, len)?;
                    let t = self.store(s, slot, parity, ob, dst, len, vec![v])?;
                    let done = self.push(&mut tasks, t);
                    per_sm[s].push(self.cta(s, slot, tasks, vec![v], vec![done]));
                }
            }
            "attention" => {
                need(3)?;
                let (bt, h, d) = (k.batch, k.heads, k.head_dim);
                let hkv = if k.kv_heads == 0 { h } else { k.kv_heads };
                let (sq, skv) = (k.seq, if k.seq_kv == 0 { k.seq } else { k.seq_kv });
                if bt == 0 || h == 0 || d == 0 || sq == 0 || k.bq == 0 || k.bkv == 0 {
                    return Err(format!(
                        "{}: batch, heads, seq, head_dim, bq, bkv are positive",
                        k.name
                    ));
                }
                if h % hkv != 0 {
                    return Err(format!("{}: heads is a multiple of kv_heads", k.name));
                }
                let warps = k.warps.max(1);
                if warps > k.bq {
                    return Err(format!("{}: more warps than bq rows", k.name));
                }
                let stages = k.stages.max(1);
                let acc = match k.s_in.as_str() {
                    "acc" => true,
                    "smem" => false,
                    x => return Err(format!("{}: unknown s_in {x:?} (acc, smem)", k.name)),
                };
                let fa2 = match k.kv_release.as_str() {
                    "fa2" => true,
                    "after_pv" => false,
                    x => {
                        return Err(format!(
                            "{}: unknown kv_release {x:?} (fa2, after_pv)",
                            k.name
                        ))
                    }
                };
                let q_buf = (k.bq * d * es).div_ceil(GRANULE) * GRANULE;
                let kv_buf = (k.bkv * d * es).div_ceil(GRANULE) * GRANULE;
                let s_buf = (k.bq * k.bkv * es).div_ceil(GRANULE) * GRANULE;
                let o_buf = q_buf;
                let kv_space = stages * 2 * kv_buf;
                // A warp's rows, and its part of the P and O spaces.
                let wq = k.bq.div_ceil(warps);
                let p_part = (wq * k.bkv * es).div_ceil(GRANULE) * GRANULE;
                let o_part = (wq * d * es).div_ceil(GRANULE) * GRANULE;
                // s_in = acc: S (fp32) and O in the accumulator; only Q and
                // the K and V buffers have SMEM of their own. s_in = smem:
                // S and O in SMEM too, O in the accumulator until the last
                // KV tile.
                // Per tensor unit: the warps on it (warp w on unit w mod
                // units) hold their S and O there.
                let on_unit = warps.div_ceil(self.sms[0].atlas.len().max(1) as u64);
                let (total, s_dtype) = if acc {
                    self.acc_fits(k, on_unit * (wq * k.bkv * 4 + wq * d * es), "S and O")?;
                    (q_buf + kv_space, Dtype::F32)
                } else {
                    self.acc_fits(k, on_unit * wq * d * es, "O")?;
                    (q_buf + kv_space + s_buf + 2 * o_buf, dtype)
                };
                for b in 0..bt {
                    for hh in 0..h {
                        let g = hh / (h / hkv);
                        for qi in 0..sq.div_ceil(k.bq) {
                            let (s, slot) = place(&mut n_ctas);
                            self.cta = ((n_ctas - 1) as u32, s as u32);
                            let sm = self.sms[s].clone();
                            let (base, size) = self.slot(&sm, slot, k.ctas_per_sm);
                            if total > size {
                                return Err(format!(
                                    "{}: Q {q_buf} B and {stages} x K and V {kv_buf} B{} do not fit in {size} B of SMEM per CTA",
                                    k.name,
                                    if acc { String::new() } else { format!(", S {s_buf} B and 2 x O {o_buf} B") }
                                ));
                            }
                            let q_at = base;
                            let kv_at = q_at + q_buf;
                            let fixed: Vec<u64> = std::iter::once(q_at)
                                .chain((0..2 * stages).map(|x| kv_at + x * kv_buf))
                                .collect();
                            let (parity, two_back) = self.out_slot(s, slot);
                            // Per warp: (S handle, P, O); bases of their own,
                            // laid out for both output buffers at once, so
                            // that a base has one length in every CTA of the
                            // slot. s_in = acc: P over the K and V space, the
                            // two O spaces over Q, S a handle after them: the
                            // program models time, not data. s_in = smem: S
                            // and P after K and V, the two O spaces after
                            // them.
                            let mut taken: Vec<u64> = Vec::new();
                            let grab = |at: u64, len: u64, taken: &mut Vec<u64>| {
                                let x = own(at, &fixed, taken);
                                taken.push(x);
                                (x + len <= base + size).then_some(x).ok_or_else(|| {
                                    format!(
                                        "{}: P {s_buf} B or O {o_buf} B do not fit in {size} B of SMEM per CTA",
                                        k.name
                                    )
                                })
                            };
                            let (p0, o0) = if acc {
                                (kv_at, q_at)
                            } else {
                                let s_at = kv_at + kv_space;
                                (s_at, s_at + s_buf)
                            };
                            let mut ps = Vec::new();
                            for w in 0..warps {
                                ps.push(grab(p0 + w * p_part, p_part, &mut taken)?);
                            }
                            let mut os = Vec::new();
                            for par in 0..2u64 {
                                let start = if acc {
                                    o0 + par * GRANULE
                                } else {
                                    o0 + par * o_buf
                                };
                                for w in 0..warps {
                                    os.push(grab(start + w * o_part, o_part, &mut taken)?);
                                }
                            }
                            let mut parts: Vec<(u64, u64, u64)> = Vec::new();
                            for w in 0..warps {
                                let s_at = if acc {
                                    grab(q_at + GRANULE, p_part, &mut taken)?
                                } else {
                                    ps[w as usize]
                                };
                                let o_at = os[(parity as u64 * warps + w) as usize];
                                parts.push((s_at, ps[w as usize], o_at));
                            }
                            let q0 = qi * k.bq;
                            let tq = k.bq.min(sq - q0);
                            let mut tasks = Vec::new();
                            let q_off = ((b * h + hh) * sq + q0) * d * es;
                            let src = self.read_at(ins[0], q_off, tq * d * es)?;
                            let t = self.dma(sm.dma, src, (q_at, q_buf), tq * d * es, Vec::new());
                            let q_dep = self.push(&mut tasks, t);
                            let last_row = q0 + tq - 1;
                            let n_kv = if k.causal {
                                (last_row / k.bkv + 1).min(skv.div_ceil(k.bkv))
                            } else {
                                skv.div_ceil(k.bkv)
                            };
                            // Per KV tile: every warp's S and P V.
                            let mut s_ver: Vec<Vec<Dep>> = Vec::new();
                            let mut pv_ver: Vec<Vec<Dep>> = Vec::new();
                            for j in 0..n_kv {
                                let st = j % stages;
                                let k_at = kv_at + st * 2 * kv_buf;
                                let v_at = k_at + kv_buf;
                                // K is free once every warp's S of the tile
                                // `stages` back read it, V once every P V
                                // did (kv_release = fa2); both after P V
                                // under after_pv.
                                let (k_war, v_war) = if j >= stages {
                                    let back = (j - stages) as usize;
                                    let pv = pv_ver[back].clone();
                                    if fa2 {
                                        (s_ver[back].clone(), pv)
                                    } else {
                                        (pv.clone(), pv)
                                    }
                                } else {
                                    (Vec::new(), Vec::new())
                                };
                                let k0 = j * k.bkv;
                                let tk = k.bkv.min(skv - k0);
                                let kv_off = ((b * hkv + g) * skv + k0) * d * es;
                                let src = self.read_at(ins[1], kv_off, tk * d * es)?;
                                let t = self.dma(sm.dma, src, (k_at, kv_buf), tk * d * es, k_war);
                                let k_dep = self.push(&mut tasks, t);
                                let src = self.read_at(ins[2], kv_off, tk * d * es)?;
                                let t = self.dma(sm.dma, src, (v_at, kv_buf), tk * d * es, v_war);
                                let v_dep = self.push(&mut tasks, t);
                                let (mut ss, mut pvs) = (Vec::new(), Vec::new());
                                for (w, &(s_at, p_at, o_at)) in parts.iter().enumerate() {
                                    let r0 = w as u64 * wq;
                                    if r0 >= tq {
                                        continue;
                                    }
                                    let rows = wq.min(tq - r0);
                                    let prev = j.checked_sub(1).map(|x| pv_ver[x as usize][w]);
                                    // S = Q K^T after the warp's last P V.
                                    let mut deps = vec![q_dep, k_dep];
                                    deps.extend(prev);
                                    let t = self.matmul(
                                        (&sm, w),
                                        (q_at + r0 * d * es, d),
                                        (k_at, tk),
                                        (s_at, p_part, tk),
                                        (rows, tk, d),
                                        (dtype, s_dtype),
                                        false,
                                        acc,
                                        deps,
                                    )?;
                                    let s_dep = self.push(&mut tasks, t);
                                    ss.push(s_dep);
                                    // The online softmax: per S element, and
                                    // the rescale of O per O element; on the
                                    // accumulator (no SMEM bytes) or S in
                                    // SMEM to P in place.
                                    let s_bytes = if acc { 0 } else { rows * tk * es };
                                    let t = self.vector(
                                        &sm,
                                        rows * tk * k.softmax_ops + rows * d * k.rescale_ops,
                                        s_at,
                                        s_bytes,
                                        (p_at, p_part),
                                        s_bytes,
                                        vec![s_dep],
                                    )?;
                                    let p_dep = self.push(&mut tasks, t);
                                    let mut deps = vec![p_dep, v_dep];
                                    deps.extend(prev);
                                    // The last P V writes O: after the write
                                    // of the CTA two back in this slot.
                                    if j + 1 == n_kv {
                                        deps.extend(two_back.iter().copied());
                                    }
                                    let t = self.matmul(
                                        (&sm, w),
                                        (p_at, tk),
                                        (v_at, d),
                                        (o_at, o_part, d),
                                        (rows, d, tk),
                                        (dtype, dtype),
                                        j > 0,
                                        j + 1 < n_kv,
                                        deps,
                                    )?;
                                    pvs.push(self.push(&mut tasks, t));
                                }
                                s_ver.push(ss);
                                pv_ver.push(pvs);
                            }
                            let last = pv_ver
                                .last()
                                .ok_or_else(|| format!("{}: a Q tile without a KV tile", k.name))?
                                .clone();
                            let mut norms = Vec::new();
                            for (w, &o) in last.iter().enumerate() {
                                let o_at = parts[w].2;
                                let rows = wq.min(tq - w as u64 * wq);
                                self.muon_kind = Kind::Rescale;
                                let t = self.vector(
                                    &sm,
                                    rows * d * k.norm_ops,
                                    o_at,
                                    rows * d * es,
                                    (o_at, o_part),
                                    rows * d * es,
                                    vec![o],
                                )?;
                                norms.push(self.push(&mut tasks, t));
                                self.muon_kind = Kind::Softmax;
                            }
                            let o_bytes = tq * d * es;
                            let dst = self.write_at(out, q_off, o_bytes)?;
                            let t = self.store(
                                s,
                                slot,
                                parity,
                                parts[0].2,
                                dst,
                                o_bytes,
                                norms.clone(),
                            )?;
                            let done = self.push(&mut tasks, t);
                            // Compute is done when every warp's normalize is.
                            per_sm[s].push(self.cta(s, slot, tasks, norms, vec![done]));
                        }
                    }
                }
            }
            o => {
                return Err(format!(
                    "{}: unknown op {o:?} (gemm, softmax, layernorm, elementwise, attention)",
                    k.name
                ))
            }
        }
        self.rec.ctas = n_ctas;
        Ok(Plan {
            per_sm,
            ctas_per_sm: k.ctas_per_sm,
            posted: self.posted,
        })
    }
}

/// Builds the run of `p` on `b`.
pub fn build(b: &Built, p: Params) -> Result<Run, String> {
    let all = sms_of(b);
    if all.is_empty() {
        return Err("kernels: no SM (a node with smem and dma)".into());
    }
    let sms = pick(&all, if p.sms == 0 { all.len() } else { p.sms }, "spread")?;
    if p.kernels.len() > 256 {
        return Err("kernels: at most 256 kernels (the label's layer field)".into());
    }
    let (first_touch, per_kernel, sim_cache) = match p.cache.as_str() {
        "first_touch" => (true, false, false),
        "home" => (false, false, false),
        "per_kernel" => (false, true, false),
        "sim" => (false, false, true),
        c => {
            return Err(format!(
                "kernels: unknown cache {c:?} (first_touch, home, per_kernel, sim)"
            ))
        }
    };
    let dram = linear(b, &p.dram)?;
    let l2 = linear(b, &p.l2)?;
    let mut names: HashMap<String, usize> = HashMap::new();
    for t in &p.tensors {
        if names.insert(t.name.clone(), names.len()).is_some() {
            return Err(format!("kernels: tensor {:?} twice", t.name));
        }
    }
    let profiles = b.sim.profiles();
    let base = Builder {
        sms: sms.clone(),
        records: sms
            .iter()
            .map(|sm| {
                b.sync_only(sm.nu)
                    .ok_or_else(|| format!("kernels: {} has no sync space", sm.path))
            })
            .collect::<Result<_, _>>()?,
        dma: profiles.lookup("dma").ok_or("kernels: no dma profile")?,
        matmul: profiles.lookup("atlas.matmul"),
        vector: profiles.lookup("muon.vector"),
        places: Vec::new(),
        names,
        tensor_recs: Vec::new(),
        versions: HashMap::new(),
        index: 0,
        cta: (0, 0),
        muon_kind: Kind::Vec,
        acc_buffer: sms
            .iter()
            .flat_map(|s| s.atlas.clone())
            .map(|a| match b.sim.pe_config(a).params {
                sim_core::PeParams::Atlas { acc_buffer, .. } => acc_buffer,
                _ => 0,
            })
            .min()
            .unwrap_or(0),
        util: 1.0,
        posted: p.posted_outputs,
        outs: HashMap::new(),
        rings: HashMap::new(),
        per_kernel: None,
        rec: KernelRec::default(),
    };
    // Pass 1, with an L2 copy of every tensor (addresses not checked):
    // which tensors are read again or written. Pass 2: L2 copies only for
    // those, placed by lifetime.
    let (plans, recs, tensor_recs) = if sim_cache {
        // Every tensor at home in the dram range, every access there: the
        // simulator's cache decides what the L2 holds.
        plan(&p, base.with(layout(&p, false, true, dram, l2, Some(&[]))?))?
    } else if per_kernel {
        // Every tensor at home in the dram range; each kernel's L2 copies
        // are placed when it is planned.
        let places = layout(&p, false, per_kernel, dram, l2, Some(&[]))?;
        let mut bld = base.with(places);
        bld.per_kernel = Some(l2);
        plan(&p, bld)?
    } else {
        let probe = layout(&p, first_touch, false, dram, l2, None)?;
        let (_, _, seen) = plan(&p, base.with(probe))?;
        let roots = layout(&p, first_touch, false, dram, l2, None)?;
        let mut need = vec![false; p.tensors.len()];
        for (t, rec) in seen.iter().enumerate() {
            if rec.l2_read_bytes > 0 || rec.write_bytes > 0 {
                need[root(&roots, t)] = true;
            }
        }
        let places = layout(&p, first_touch, false, dram, l2, Some(&need))?;
        plan(&p, base.with(places))?
    };
    // The spawn tree over the gpcs.
    let topo = b.sim.topology();
    let root_sync = b
        .sync_only(NeutrinoId(0))
        .ok_or("kernels: the device has no sync space")?;
    let mut gpcs: Vec<Gpc> = Vec::new();
    for (j, s) in sms.iter().enumerate() {
        let g = topo
            .node(s.nu)
            .parent
            .ok_or("kernels: an SM without a gpc")?;
        if topo.node(g).parent != Some(NeutrinoId(0)) {
            return Err("kernels: an SM's parent is not a child of the device".into());
        }
        match gpcs.iter_mut().find(|x| x.0 == g) {
            Some(x) => x.2.push((j, s.nu)),
            None => {
                let sync = b
                    .sync_only(g)
                    .ok_or_else(|| format!("kernels: {} has no sync space", topo.path_name(g)))?;
                gpcs.push((g, sync, vec![(j, s.nu)]));
            }
        }
    }
    let report = Rc::new(RefCell::new(Report {
        cache: p.cache.clone(),
        sms: sms.len(),
        kernels: recs,
        end: 0,
        tensors: tensor_recs,
    }));
    Ok(Run {
        plans,
        flush: sim_cache,
        report,
        gpcs,
        root_sync,
    })
}

impl Builder {
    /// A copy of the builder's fixed parts with `places`.
    fn with(&self, places: Vec<Place>) -> Builder {
        Builder {
            sms: self.sms.clone(),
            records: self.records.clone(),
            dma: self.dma,
            matmul: self.matmul,
            vector: self.vector,
            tensor_recs: Vec::new(),
            places,
            names: self.names.clone(),
            versions: HashMap::new(),
            index: 0,
            cta: (0, 0),
            muon_kind: Kind::Vec,
            acc_buffer: self.acc_buffer,
            util: 1.0,
            posted: self.posted,
            outs: HashMap::new(),
            rings: HashMap::new(),
            per_kernel: None,
            rec: KernelRec::default(),
        }
    }
}

/// The root tensor of `t` among `places`.
fn root(places: &[Place], mut t: usize) -> usize {
    while let Some((parent, _)) = places[t].view {
        t = parent;
    }
    t
}

/// The tensors' places. Home `dram` tensors go to the dram range. Home
/// `l2` tensors, and under `first_touch` the L2 copies of the tensors in
/// `need` (every tensor when `need` is `None`, at addresses not checked
/// against the range), go to the L2 range, each for its lifetime: from
/// the first kernel that names it (or a view of it) to the last.
fn layout(
    p: &Params,
    first_touch: bool,
    all_dram: bool,
    dram: Range,
    l2: Range,
    need: Option<&[bool]>,
) -> Result<Vec<Place>, String> {
    let index: HashMap<&str, usize> = p
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();
    let mut places: Vec<Place> = Vec::new();
    for (i, t) in p.tensors.iter().enumerate() {
        let view = match &t.of {
            None => None,
            Some(of) => {
                let parent = *index
                    .get(of.as_str())
                    .filter(|&&x| x < i)
                    .ok_or_else(|| format!("kernels: {}: no tensor {of:?} before it", t.name))?;
                if t.offset + t.bytes > p.tensors[parent].bytes {
                    return Err(format!("kernels: view {} exceeds {of}", t.name));
                }
                Some((parent, t.offset))
            }
        };
        let in_l2 = match t.home.as_str() {
            "dram" => false,
            "l2" => !all_dram,
            h => {
                return Err(format!(
                    "kernels: {}: unknown home {h:?} (dram, l2)",
                    t.name
                ))
            }
        };
        places.push(Place {
            view,
            in_l2,
            base: 0,
            l2_base: None,
            bytes: t.bytes,
            covered: Vec::new(),
        });
    }
    // Lifetimes of the roots, in kernels.
    let n = p.kernels.len();
    let mut life: Vec<Option<(usize, usize)>> = vec![None; places.len()];
    for (ki, k) in p.kernels.iter().enumerate() {
        for name in k.inputs.iter().chain(std::iter::once(&k.output)) {
            if let Some(&t) = index.get(name.as_str()) {
                let r = root(&places, t);
                let l = life[r].get_or_insert((ki, ki));
                l.1 = ki;
            }
        }
    }
    let mut at_dram = dram.base;
    // L2 blocks in use: (base, bound, last kernel).
    let mut used: Vec<(u64, u64, usize)> = Vec::new();
    let mut virtual_l2 = l2.base;
    let mut order: Vec<usize> = (0..places.len())
        .filter(|&i| places[i].view.is_none())
        .collect();
    order.sort_by_key(|&i| (life[i].map_or(0, |l| l.0), i));
    for i in order {
        let bytes = places[i].bytes;
        let align = |x: u64| x.div_ceil(TENSOR_ALIGN) * TENSOR_ALIGN;
        if !places[i].in_l2 {
            places[i].base = align(at_dram);
            at_dram = places[i].base + bytes;
            if places[i].base + bytes > dram.bound {
                return Err(format!("kernels: the tensors do not fit in {}", p.dram));
            }
        }
        let wants_l2 = places[i].in_l2 || (first_touch && need.is_none_or(|nd| nd[i]));
        if !wants_l2 {
            continue;
        }
        let at = if need.is_none() {
            let at = align(virtual_l2);
            virtual_l2 = at + bytes;
            at
        } else {
            let (first, last) = life[i].unwrap_or((0, n.saturating_sub(1)));
            used.retain(|u| u.2 >= first);
            used.sort_unstable();
            let mut at = align(l2.base);
            for u in &used {
                if at + bytes <= u.0 {
                    break;
                }
                at = at.max(align(u.1));
            }
            if at + bytes > l2.bound {
                return Err(format!(
                    "kernels: {} ({bytes} B) does not fit in {} with the tensors live in kernel {first}; cache = \"home\" needs no L2 copies",
                    p.tensors[i].name, p.l2
                ));
            }
            used.push((at, at + bytes, last));
            at
        };
        if places[i].in_l2 {
            places[i].base = at;
        } else {
            places[i].l2_base = Some(at);
        }
    }
    Ok(places)
}

/// Plans every kernel of `p` with `bld`.
#[allow(clippy::type_complexity)]
fn plan(
    p: &Params,
    mut bld: Builder,
) -> Result<(Vec<(String, Plan)>, Vec<KernelRec>, Vec<TensorRec>), String> {
    bld.tensor_recs = p
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let r = root(&bld.places, i);
            let (off, rp) = (bld.root_of(i).1, &bld.places[r]);
            TensorRec {
                name: t.name.clone(),
                home: p.tensors[r].home.clone(),
                base: rp.base + off,
                l2_base: rp.l2_base.map(|b| b + off),
                ..TensorRec::default()
            }
        })
        .collect();
    let mut plans = Vec::new();
    let mut recs = Vec::new();
    for (ki, k) in p.kernels.iter().enumerate() {
        bld.index = ki as u64;
        if k.name.is_empty() {
            return Err("kernels: a kernel without a name".into());
        }
        bld.versions.clear();
        if let Some(l2) = bld.per_kernel {
            let need = bld.kernel_needs(k, l2)?;
            bld.cold_l2(k, l2, Some(&need))?;
        }
        bld.rec = KernelRec {
            name: k.name.clone(),
            op: k.op.clone(),
            ..KernelRec::default()
        };
        let plan = bld.kernel(k)?;
        recs.push(std::mem::take(&mut bld.rec));
        plans.push((k.name.clone(), plan));
    }
    Ok((plans, recs, bld.tensor_recs))
}

/// A CTA in flight on an SM: its compute versions and its output writes.
type OpenCta = (Vec<(u64, u64)>, Vec<(u64, u64)>);

/// One SM's part of a kernel: its CTAs in order, at most `limit` at once.
/// With posted outputs a slot is free when its CTA's compute is done, and
/// at the end the program waits for the last two output writes of each
/// slot (each earlier write is a dependency of a later CTA's output);
/// otherwise a slot is free when its CTA's output write is done.
async fn sm_program(ctx: NuCtx, ctas: Vec<Cta>, limit: usize, posted: bool) {
    let mut open: std::collections::VecDeque<OpenCta> = Default::default();
    let mut dones = Vec::new();
    for cta in ctas {
        if open.len() >= limit {
            let (compute, done) = open.pop_front().expect("open CTA");
            if posted {
                for (a, v) in compute {
                    ctx.wait(a, v).await;
                }
            } else {
                for (a, v) in done {
                    ctx.wait(a, v).await;
                }
            }
        }
        for t in cta.tasks {
            ctx.invoke(t.desc).await;
        }
        dones.push(cta.done.clone());
        open.push_back((cta.compute, cta.done));
    }
    let tail = if posted { 2 * limit } else { limit };
    for &(a, v) in dones[dones.len().saturating_sub(tail)..].iter().flatten() {
        ctx.wait(a, v).await;
    }
}

impl Run {
    /// The device's program: per kernel, a program on each gpc that starts
    /// one on each of its SMs with CTAs, then the barrier.
    pub async fn run(self, ctx: NuCtx) {
        for (ki, (name, plan)) in self.plans.into_iter().enumerate() {
            // The kernel's gpc and SM program instances carry its name (a
            // tracer's program bars, asm listings); one leaked string per
            // kernel of the run.
            let prog: &'static str = Box::leak(name.clone().into_boxed_str());
            if self.flush {
                ctx.cache_flush().await;
            }
            ctx.mark(&format!("{name} start")).await;
            let start = ctx.now();
            let mut per_sm: Vec<Option<Vec<Cta>>> = plan.per_sm.into_iter().map(Some).collect();
            let (limit, posted) = (plan.ctas_per_sm, plan.posted);
            let mut dones = Vec::new();
            for (k, (g, sync, sms)) in self.gpcs.iter().enumerate() {
                let lists: Vec<(NeutrinoId, Vec<Cta>)> = sms
                    .iter()
                    .filter_map(|&(j, nu)| {
                        let ctas = per_sm[j].take().unwrap_or_default();
                        (!ctas.is_empty()).then_some((nu, ctas))
                    })
                    .collect();
                if lists.is_empty() {
                    continue;
                }
                let sync = *sync;
                let done = Range::from_len(self.root_sync.base + k as u64 * GRANULE, GRANULE);
                let gpc = move |gctx: NuCtx| async move {
                    let mut sm_dones = Vec::new();
                    for (i, (nu, ctas)) in lists.into_iter().enumerate() {
                        let d = Range::from_len(sync.base + i as u64 * GRANULE, GRANULE);
                        let sm = move |sctx: NuCtx| sm_program(sctx, ctas, limit, posted);
                        gctx.spawn_named(nu, prog, d, vec![], sm).await;
                        sm_dones.push(d.base);
                    }
                    for d in sm_dones {
                        gctx.wait_unversioned(d).await;
                    }
                };
                ctx.spawn_named(*g, prog, done, vec![], gpc).await;
                dones.push(done.base);
            }
            for d in dones {
                ctx.wait_unversioned(d).await;
            }
            ctx.mark(&format!("{name} end")).await;
            let end = ctx.now();
            let mut r = self.report.borrow_mut();
            r.kernels[ki].start = start;
            r.kernels[ki].end = end;
            r.end = end;
        }
    }
}
