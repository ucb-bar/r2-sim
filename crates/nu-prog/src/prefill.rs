//! Prefix pass builders (5b item 3): the SigLIP vision encoder and the
//! PaliGemma VLM prefill, as GEMM and flash-attention phases over
//! activations in HBM (or in cmem when the configuration has it).
//!
//! Structure:
//! * One driver program on the home SM invokes every task; each phase (one
//!   GEMM or one attention) is one graph window closed by a barrier.
//! * GEMM `out[N, C] = A[N, K] x W[K, C]`: a spatial grid of token groups x
//!   column groups over the workers ([`choose_grid`]). Each SM owns an out
//!   block of `T` rows and `Cs` columns and runs column passes of width `Cw`
//!   (the widest that fits the SMEM budget). A pass streams K in `kc`
//!   chunks (A tile and W tile per chunk, two staging slots each), then an
//!   epilogue on the Muon, then writes the out tile. RMSNorm and LayerNorm
//!   are folded: row statistics from the A chunks of the first pass, the
//!   row scale in the epilogue, the gain in W (exact).
//! * Flash attention: work units of one query tile and all query heads;
//!   K/V blocks of `key_block` keys are streamed once per KV head per unit
//!   (MQA: once for all 8 VLM heads), with an online softmax on the Muon.
//! * Writes into HBM or cmem are tasks of the memory's owner on its own
//!   copy engine, which pulls the SM out tile (Section 3.3): each depends on
//!   a fresh signal at the SM; the next writer of that tile depends on a
//!   fresh signal at the owner (the write is done).
//! * Tensor layout in HBM and cmem: column-block-major, `[C/128][N][128]`,
//!   so an A chunk and an out block are contiguous. Weights likewise,
//!   `[C/128][K][128]` in the weight type. The VLM K and V go row-major to
//!   the [`KvLayout`] the action expert reads.
//!
//! Reads carry no deps: the producers are in earlier windows. A read of a
//! head's columns (SigLIP, 72 of a 128-column block) is charged its exact
//! bytes at the block's address.

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::{NeutrinoId, Range};
use serde::{Deserialize, Serialize};
use sim_core::{Args, Descriptor, Dtype, NuCtx, PeId};

use crate::graph::{excess_deps, with_deps, Graph, MAX_JOIN_DEPS};
use crate::kv::KvLayout;
use crate::label::{self, Kind};
use crate::machine::{
    parse_dtype, with_b_dtype, worker_list, Layout, Machine, MemNode, SmRates, SyncPool,
};

/// Program parameters (a TOML file, every key optional).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Params {
    /// "vlm" (PaliGemma 2B) or "vision" (SigLIP So400m).
    pub model: String,
    /// Layers to run; 0 = the model's.
    pub layers: u32,
    /// Shape overrides for small tests; 0 = the model's.
    pub d: u64,
    /// Query heads; 0 = the model's.
    pub heads: u64,
    /// KV heads; 0 = the model's.
    pub kv_heads: u64,
    /// Elements per head; 0 = the model's.
    pub head_dim: u64,
    /// FFN hidden width; 0 = the model's.
    pub ffn: u64,
    /// The action expert whose HBM map comes first and whose KV layout the
    /// VLM writes.
    pub action: crate::adamoe::Params,
    /// Images in the prefix.
    pub images: u64,
    /// Tokens per image.
    pub tokens_per_image: u64,
    /// Text tokens after the image tokens (VLM only).
    pub text_tokens: u64,
    /// "bf16", "fp8" or "mxfp4".
    pub weight_dtype: String,
    /// The VLM's last layer: "full", "no_ffn" (skip its FFN) or "kv" (only
    /// its K/V projection; the rest feeds nothing downstream). The vision
    /// encoder always runs its last layer in full.
    pub last_layer: String,
    /// Force the grid of every GEMM, [token_groups, column_groups]; [0, 0]
    /// = the minimum-cost grid.
    pub grid: [u64; 2],
    /// K chunk of the GEMM stream; 0 = chosen with the grid (128 to 512).
    pub kc: u64,
    /// Pass width of every GEMM; 0 = the widest that fits (experiments).
    pub cw: u64,
    /// GEMM loop order: "auto" (the cost model picks per GEMM), "os"
    /// (output-stationary only) or "ws" (weight-stationary for every GEMM
    /// where a weight-stationary grid fits, else output-stationary).
    pub order: String,
    /// The cost model's view of Atlas tile quantization (match the config).
    pub quantize: bool,
    /// Cost model: fixed cycles per GEMM pass (Muon task overhead plus the
    /// out write's DMA latency).
    pub pass_overhead: f64,
    /// One accumulator per SM (no ping-pong; default): wider passes, the
    /// next pass waits for this pass's out writes. Faster on one VLM layer
    /// on all four item-4 configurations.
    pub single_accumulator: bool,
    /// Query rows per attention unit; 0 = the largest that fits.
    pub q_tile: u64,
    /// Keys per K/V block in flash attention.
    pub key_block: u64,
    /// DMA chunk, bytes.
    pub dma_chunk: u64,
    /// With activations in cmem: a copy in every cmem cluster (reads stay
    /// in the cluster; writes go to every copy). Default off: one VLM layer
    /// +2 %, one vision layer -3 % (3b iii), and each copy costs cmem.
    pub act_copies: bool,
    /// Where activations live: "auto" (cmem of the first cmem cluster when
    /// present, else HBM), "hbm" or "cmem".
    pub act_mem: String,
    /// Index of the driver SM in configuration order.
    pub home: usize,
    /// Worker SMs; empty = all but home.
    pub workers: Vec<usize>,
    /// SMEM bytes per SM for the builder; 0 = the whole SMEM.
    pub smem_budget: u64,
    /// Atlas tile (m, n) of every matmul.
    pub tile: [u64; 2],
    /// Muon lane-ops per A element for the row statistics of a folded norm.
    pub norm_ops: u64,
    /// Muon lane-ops per out element for the FFN activation.
    pub act_ops: u64,
    /// Muon lane-ops per out element for RoPE (Q and KV GEMMs of the VLM).
    pub rope_ops: u64,
    /// Muon lane-ops per attention score for the online softmax.
    pub softmax_ops: u64,
    /// Muon lane-ops per out element for a residual add (and the vision
    /// position embedding).
    pub add_ops: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            model: "vlm".into(),
            layers: 0,
            d: 0,
            heads: 0,
            kv_heads: 0,
            head_dim: 0,
            ffn: 0,
            action: crate::adamoe::Params::default(),
            images: 3,
            tokens_per_image: 256,
            text_tokens: 48,
            weight_dtype: "bf16".into(),
            last_layer: "full".into(),
            grid: [0, 0],
            kc: 0,
            cw: 0,
            order: "auto".into(),
            quantize: true,
            pass_overhead: 150.0,
            single_accumulator: true,
            q_tile: 0,
            key_block: 128,
            dma_chunk: 64 << 10,
            act_mem: "auto".into(),
            act_copies: false,
            home: 0,
            workers: Vec::new(),
            smem_budget: 0,
            tile: [128, 128],
            norm_ops: 4,
            act_ops: 8,
            rope_ops: 4,
            softmax_ops: 6,
            add_ops: 1,
        }
    }
}

/// Transformer shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Model {
    /// Transformer layers.
    pub layers: u32,
    /// Model width.
    pub d: u64,
    /// Query heads.
    pub heads: u64,
    /// KV heads.
    pub kv_heads: u64,
    /// Elements per head.
    pub head_dim: u64,
    /// FFN hidden width.
    pub ffn: u64,
    /// Gated FFN (GeGLU): gate and up in one pass.
    pub gated: bool,
    /// Rotary position embedding on Q and K.
    pub rope: bool,
}

pub(crate) const PALIGEMMA_2B: Model = Model {
    layers: 18,
    d: 2048,
    heads: 8,
    kv_heads: 1,
    head_dim: 256,
    ffn: 16384,
    gated: true,
    rope: true,
};

pub(crate) const SIGLIP_SO400M: Model = Model {
    layers: 27,
    d: 1152,
    heads: 16,
    kv_heads: 16,
    head_dim: 72,
    ffn: 4304,
    gated: false,
    rope: false,
};

/// Patch pixels (14 x 14 x 3) and the projector width (the VLM's d).
pub(crate) const PATCH_DIM: u64 = 588;
pub(crate) const PROJ_OUT: u64 = 2048;

/// One grid choice, for the log.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GridChoice {
    /// Loop order: weight-stationary (W slice resident, token tiles
    /// stream) instead of output-stationary (accumulators resident).
    pub ws: bool,
    /// Token groups.
    pub tg: u64,
    /// Column groups.
    pub cg: u64,
    /// Rows per token group (the last may be fewer).
    pub t: u64,
    /// Columns per column group.
    pub cs: u64,
    /// Column pass width.
    pub cw: u64,
    /// K chunk.
    pub kc: u64,
    /// Passes per SM (the most of any SM).
    pub passes: u64,
    /// Bytes from the activation memory (A, residual, out).
    pub act_bytes: u64,
    /// Weight bytes from HBM.
    pub w_bytes: u64,
    /// Bytes through the HBM port (weights, and activations in HBM).
    pub hbm_bytes: u64,
    /// SMEM port bytes of all used SMs (the busiest SM's times the SMs).
    pub smem_bytes: u64,
    /// Estimated cycles: the slowest of the busiest SM (link in, link out,
    /// SMEM port, Atlas), the HBM port and the activation memory port.
    pub cost: f64,
    /// Which of those set `cost`.
    pub bound: &'static str,
}

/// The shape of one GEMM for the grid choice.
#[derive(Debug, Clone, Copy)]
pub struct GemmShape {
    /// Rows (tokens) of A and out.
    pub n: u64,
    /// Inner dimension.
    pub k: u64,
    /// Out columns.
    pub c: u64,
    /// Weight element type.
    pub w: Dtype,
    /// Two weight matrices streamed in the same pass (gate and up).
    pub gated: bool,
    /// A residual tile read in the epilogue.
    pub residual: bool,
    /// Column passes must be exactly this wide (the KV GEMM: one head).
    pub cw_fixed: Option<u64>,
    /// Row statistics of a folded norm in the first pass.
    pub norm: bool,
    /// One accumulator (set) per SM instead of two: wider passes, no
    /// overlap of a pass's writes with the next pass.
    pub single_acc: bool,
}

impl GemmShape {
    /// Weight matrices per pass (2 for gate and up).
    fn mats(&self) -> u64 {
        if self.gated {
            2
        } else {
            1
        }
    }
}

/// GEMM loop order of the grid search (`Params::order`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// The cheaper order per GEMM.
    Auto,
    /// Output-stationary only.
    Os,
    /// Weight-stationary where a grid fits.
    Ws,
}

impl Order {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "os" => Ok(Self::Os),
            "ws" => Ok(Self::Ws),
            _ => Err(format!("unknown order {s:?} (auto, os, ws)")),
        }
    }
}

/// What the grid cost model knows about the machine.
#[derive(Debug, Clone)]
pub struct CostEnv {
    /// Worker SMs.
    pub workers: u64,
    /// SMEM bytes per SM.
    pub budget: u64,
    /// Allocation granule, bytes.
    pub granule: u64,
    /// Candidate K chunks (the largest that fits wins on SMEM traffic).
    pub kcs: Vec<u64>,
    /// Rates of the first worker SM.
    pub sm: SmRates,
    /// HBM port rate, bytes per cycle.
    pub hbm_rate: f64,
    /// Activation memory port rate, bytes per cycle.
    pub act_rate: f64,
    /// Activations share the HBM port.
    pub act_in_hbm: bool,
    /// The Atlas works on whole 128 x 128 tiles.
    pub quantize: bool,
    /// Fixed cycles per pass on an SM's critical path.
    pub pass_overhead: f64,
    /// Force the pass width (0 = the widest that fits).
    pub cw: u64,
    /// Per worker (in grid order): its cluster index, and per cluster: the
    /// link rates to the parent each way and whether the activations live
    /// in that cluster (else they cross the link, as the weights from HBM
    /// always do). Empty: no cluster link term.
    pub worker_cluster: Vec<usize>,
    /// Per cluster: (link rate down, link rate up, activations in the
    /// cluster); rates in bytes per cycle.
    pub cluster_links: Vec<(f64, f64, bool)>,
    /// Copies of the activations (one per cmem cluster when copied).
    pub act_copies: usize,
    /// The GEMM loop orders the grid search may use.
    pub order: Order,
    /// The tasks the out owner can track for the driver (its scoreboard
    /// less its issue headroom):
    /// the out writes and done signals in flight at the owner must fit it,
    /// or the owner's tables deadlock (protocol lesson 7).
    pub owner_grant: Option<u64>,
}

/// The lower bounds of one GEMM on the whole machine, in cycles.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Roofline {
    /// Every weight byte once from HBM.
    pub weights_once: f64,
    /// Every (tile-padded) MAC on all workers' arrays.
    pub atlas: f64,
    /// The Atlas operand and C traffic of one full-K task per output tile
    /// on all workers' SMEM ports.
    pub smem: f64,
    /// A and W once into the SMs, out once back, on all workers' links.
    pub links: f64,
    /// The largest of the bounds.
    pub bound: f64,
    /// Which bound is the largest.
    pub bound_by: &'static str,
}

/// The roofline of `s` on `e` (independent of the grid).
pub(crate) fn roofline(s: &GemmShape, e: &CostEnv) -> Roofline {
    let mats = s.mats();
    let w = e.workers as f64;
    let (mt, nt) = (s.n.div_ceil(128), s.c.div_ceil(128));
    let rows = if e.quantize { mt * 128 } else { s.n };
    let w_once = mats * nt * s.w.bytes_for(s.k * 128);
    let macs = mats * rows * nt * 128 * s.k;
    let a = if e.sm.row_panel { 1 } else { nt } * rows * s.k * 2;
    let b = mt * nt * s.w.bytes_for(s.k * 128);
    let c = s.n * s.c * 2;
    let smem = mats * (a + b + c);
    let a_once = s.n * s.k * 2;
    let into = a_once + w_once + if s.residual { c } else { 0 };
    let parts = [
        (w_once as f64 / e.hbm_rate, "weights once"),
        (macs as f64 / (w * e.sm.mac.max(1e-9)), "Atlas"),
        (smem as f64 / (w * e.sm.smem.max(1e-9)), "SMEM port"),
        (
            (into as f64 / e.sm.link_down.max(1e-9)).max(c as f64 / e.sm.link_up.max(1e-9)) / w,
            "SM links",
        ),
    ];
    let (bound, bound_by) = parts
        .iter()
        .copied()
        .fold((0.0, ""), |a, x| if x.0 > a.0 { x } else { a });
    Roofline {
        weights_once: parts[0].0,
        atlas: parts[1].0,
        smem: parts[2].0,
        links: parts[3].0,
        bound,
        bound_by,
    }
}

/// SMEM bytes of one SM for `t` rows, pass width `cw` and K chunk `kc`:
/// the accumulators (one set, or two that ping-pong over passes without
/// `single_acc`; two per set when gated), two A and two W staging slots,
/// the residual tile and the row statistics.
pub(crate) fn smem_need(s: &GemmShape, t: u64, cw: u64, kc: u64, g: u64) -> u64 {
    let r = |b: u64| b.div_ceil(g) * g;
    let mats = s.mats();
    let acc = if s.single_acc { 1 } else { 2 } * mats * r(t * cw * 2);
    let a = 2 * r(t * kc * 2);
    let w = 2 * mats * (cw / 128) * r(s.w.bytes_for(kc * 128));
    let res = if s.residual { r(t * cw * 2) } else { 0 };
    acc + a + w + res + r(t * 8)
}

/// Token rows per tile of the weight-stationary order.
pub(crate) const WS_TILE: u64 = 128;

/// SMEM bytes of one SM in the weight-stationary order: the W slice of the
/// pass resident (every K chunk), two out tiles of `WS_TILE` rows (ping-pong
/// over token tiles), two A staging slots, the residual tile and the row
/// statistics of every token tile.
pub(crate) fn smem_need_ws(s: &GemmShape, t: u64, cw: u64, kc: u64, g: u64) -> u64 {
    let r = |b: u64| b.div_ceil(g) * g;
    let mats = s.mats();
    let tt = t.min(WS_TILE);
    let nk = s.k.div_ceil(kc);
    let w = mats * (cw / 128) * nk * r(s.w.bytes_for(kc * 128));
    let out = 2 * mats * r(tt * cw * 2);
    let a = 2 * r(tt * kc * 2);
    let res = if s.residual { r(tt * cw * 2) } else { 0 };
    w + out + a + res + t.div_ceil(tt) * r(tt * 8)
}

/// Estimated cycles of one candidate; see [`GridChoice::cost`].
#[allow(clippy::too_many_arguments)]
fn estimate(
    ws: bool,
    s: &GemmShape,
    e: &CostEnv,
    tg: u64,
    cg: u64,
    t: u64,
    cs: u64,
    cw: u64,
    kc: u64,
) -> GridChoice {
    let mats = s.mats();
    let blocks = s.c.div_ceil(128);
    let passes = cs.div_ceil(cw);
    let nk = s.k.div_ceil(kc);
    // The busiest SM: t rows, cs columns.
    let w_sm = mats * (cs / 128) * s.w.bytes_for(s.k * 128);
    let res = if s.residual { t * cs * 2 } else { 0 };
    let a_in = passes * t * s.k * 2;
    let link_in = a_in + w_sm + res;
    let link_out = t * cs * 2;
    // Matmul SMEM traffic as the Atlas profile charges it: per 128 x 128
    // output tile, the A tile and the B tile over the chunk, C out, and C
    // in when accumulating (every chunk but the first of a pass).
    let (mt, nt) = (t.div_ceil(128), cw / 128);
    let rows = if e.quantize { mt * 128 } else { t };
    // With a row panel (H1) the A tile is read once per row of tiles.
    let a_mm = if e.sm.row_panel { 1 } else { nt } * rows * kc * 2;
    let b_mm = mt * nt * s.w.bytes_for(kc * 128);
    // C out per chunk and in per accumulate, unless the Atlas keeps the
    // pass's accumulators (H3): then C goes out once.
    // Weight-stationary keeps one token tile's accumulators at a time.
    let acc_rows = if ws { t.min(WS_TILE) } else { t };
    let c_pass = if mats * acc_rows * cw * 2 <= e.sm.acc_buffer {
        t * cw * 2
    } else {
        (2 * nk - 1) * t * cw * 2
    };
    let mm_bytes = passes * mats * (nk * (a_mm + b_mm) + c_pass);
    // Epilogue on the Muon: the accumulators and residual in, the out tile
    // back; the row statistics read A once in the first pass.
    let epi = passes * (mats + u64::from(s.residual) + 1) * t * cw * 2;
    let stats = if s.norm { t * s.k * 2 } else { 0 };
    let smem = link_in + link_out + mm_bytes + epi + stats;
    let macs = passes * nk * mats * rows * cw * kc;
    // Per-pass costs on the SM's critical path. With two staging slots the
    // last K chunk of a pass loads only after the previous pass's last
    // matmul, so one chunk's transfer is exposed per pass (at the SM's
    // share of its links), or the fixed overhead (the epilogue task and the
    // out write's latency) when that is longer. Ping-pong accumulators hide
    // it except once.
    let chunk_in = t * kc * 2 + mats * (cw / 128) * s.w.bytes_for(kc * 128);
    let mut r_in = e.sm.link_down.max(1e-9);
    for (c, &(dr, _, local)) in e.cluster_links.iter().enumerate() {
        let n = e
            .worker_cluster
            .iter()
            .take((tg * cg) as usize)
            .filter(|&&x| x == c)
            .count();
        if n > 0 && !local {
            r_in = r_in.min(dr / n as f64);
        }
    }
    let bubble = (chunk_in as f64 / r_in).max(e.pass_overhead);
    // Weight-stationary: token tiles ping-pong, W chunks reload one by one.
    let fixed = if ws {
        passes as f64 * e.pass_overhead
    } else {
        (if s.single_acc { passes } else { 1 }) as f64 * bubble
    };
    let sm_times = [
        (
            link_in as f64 / e.sm.link_down.max(1e-9) + fixed,
            "SM link in",
        ),
        (
            link_out as f64 / e.sm.link_up.max(1e-9) + fixed,
            "SM link out",
        ),
        (smem as f64 / e.sm.smem.max(1e-9) + fixed, "SMEM port"),
        (macs as f64 / e.sm.mac.max(1e-9) + fixed, "Atlas"),
    ];
    // Whole-GEMM traffic.
    let a = passes * s.n * s.k * 2 * cg;
    let out = s.n * s.c * 2 * if s.residual { 2 } else { 1 };
    let w = mats * blocks * s.w.bytes_for(s.k * 128) * tg;
    let copies = e.act_copies.max(1) as u64;
    let (hbm, act) = if e.act_in_hbm {
        ((w + a + out) as f64 / e.hbm_rate, 0.0)
    } else {
        // Reads spread over the copies; every copy takes every write.
        (
            w as f64 / e.hbm_rate,
            (a + out * copies) as f64 / (e.act_rate * copies as f64),
        )
    };
    // Cluster links: each used SM's weights come from HBM through its
    // cluster's link; its activations too unless they live in the cluster.
    let mut down = vec![0u64; e.cluster_links.len()];
    let mut up = vec![0u64; e.cluster_links.len()];
    let used = (tg * cg) as usize;
    let mut out_of = vec![0u64; e.cluster_links.len()];
    for &c in e.worker_cluster.iter().take(used) {
        let local = e.cluster_links[c].2;
        down[c] += w_sm + if local { 0 } else { a_in + res };
        up[c] += if local {
            // Writes to the other clusters' copies.
            link_out * (copies - 1)
        } else {
            link_out
        };
        out_of[c] += link_out;
    }
    if copies > 1 {
        // Each cluster with a copy takes the other clusters' writes.
        let total: u64 = out_of.iter().sum();
        for c in 0..down.len() {
            if e.cluster_links[c].2 {
                down[c] += total - out_of[c];
            }
        }
    }
    let xlink = (0..e.cluster_links.len())
        .map(|c| {
            let (dr, ur, _) = e.cluster_links[c];
            (down[c] as f64 / dr.max(1e-9)).max(up[c] as f64 / ur.max(1e-9))
        })
        .fold(0.0, f64::max);
    let mut worst = (hbm, "HBM port");
    for x in sm_times
        .into_iter()
        .chain([(act, "activation memory port"), (xlink, "cluster link")])
    {
        if x.0 > worst.0 {
            worst = x;
        }
    }
    GridChoice {
        ws,
        tg,
        cg,
        t,
        cs,
        cw,
        kc,
        passes,
        act_bytes: a + out,
        w_bytes: w,
        hbm_bytes: if e.act_in_hbm { w + a + out } else { w },
        smem_bytes: smem * tg * cg,
        cost: worst.0,
        bound: worst.1,
    }
}

/// Grids whose estimates lie within this fraction of the best estimate are
/// treated as equal: the GEMM phases measure 1.05 to 1.30 times their
/// estimate, so a smaller difference is inside the model's uncertainty.
pub const NEAR: f64 = 0.10;

/// The grid over token_groups x column_groups <= workers (or `forced`) and
/// the candidate K chunks. For each, the pass width is the widest that fits
/// the SMEM budget; infeasible candidates are skipped. Among the grids
/// within [`NEAR`] of the lowest estimate, the one with the fewest HBM
/// bytes, then the fewest SMEM bytes, then the lowest estimate. Under
/// order ws, a weight-stationary grid beats every other.
pub fn choose_grid(
    s: &GemmShape,
    e: &CostEnv,
    forced: Option<(u64, u64)>,
) -> Result<GridChoice, String> {
    let blocks = s.c.div_ceil(128);
    let mut all: Vec<GridChoice> = Vec::new();
    let cands: Vec<(u64, u64)> = match forced {
        Some(f) => vec![f],
        None => (1..=e.workers)
            .flat_map(|tg| (1..=e.workers / tg).map(move |cg| (tg, cg)))
            .collect(),
    };
    for (tg, cg) in cands {
        if cg > blocks || tg > s.n || tg * cg > e.workers {
            continue;
        }
        // Even rows keep every 128-column block of a group granule-aligned.
        let t = s.n.div_ceil(tg).next_multiple_of(2);
        let cs = blocks.div_ceil(cg) * 128;
        for (ws, &kc) in [false, true]
            .into_iter()
            .filter(|&ws| e.order != Order::Os || !ws)
            .flat_map(|ws| e.kcs.iter().map(move |k| (ws, k)))
        {
            if kc > s.k.next_multiple_of(128) {
                continue;
            }
            let fits = |cw: u64| {
                let need = if ws {
                    smem_need_ws(s, t, cw, kc, e.granule)
                } else {
                    smem_need(s, t, cw, kc, e.granule)
                };
                // Tracked at the out owner per SM: the out tiles in flight
                // (ws ping-pong: 2; os: the accumulator sets), each with its
                // writes (one per 128-column block; one for the KV) and its
                // done signal.
                let sets = if ws || !s.single_acc { 2 } else { 1 };
                let writes = if s.cw_fixed.is_some() { 1 } else { cw / 128 };
                let tracked = sets * (writes + 1) * tg * cg;
                // Weight-stationary keeps every W chunk of the pass done in
                // the execution table until the pass's last token tile reads it; half
                // the execution table stays for the rest of the pipeline.
                let nk = s.k.div_ceil(kc);
                let mats = s.mats();
                let execution_ok =
                    !ws || e.sm.execution == 0 || mats * (cw / 128) * nk <= e.sm.execution / 2;
                need <= e.budget && execution_ok && e.owner_grant.is_none_or(|g| tracked <= g)
            };
            let cw = match s.cw_fixed {
                Some(f) if cs % f == 0 && fits(f) => f,
                Some(_) => continue,
                None if e.cw > 0 => {
                    if e.cw > cs || !fits(e.cw) {
                        continue;
                    }
                    e.cw
                }
                None => match (1..=cs / 128).rev().map(|b| b * 128).find(|&cw| fits(cw)) {
                    Some(cw) => cw,
                    None => continue,
                },
            };
            all.push(estimate(ws, s, e, tg, cg, t, cs, cw, kc));
        }
    }
    if e.order == Order::Ws && all.iter().any(|g| g.ws) {
        all.retain(|g| g.ws);
    }
    let low = all.iter().map(|g| g.cost).fold(f64::INFINITY, f64::min);
    all.into_iter()
        .filter(|g| g.cost <= low * (1.0 + NEAR))
        .min_by(|a, b| {
            (a.hbm_bytes, a.smem_bytes)
                .cmp(&(b.hbm_bytes, b.smem_bytes))
                .then(a.cost.total_cmp(&b.cost))
        })
        .ok_or_else(|| {
            format!(
                "no grid fits: n {} k {} c {} in {} B of SMEM",
                s.n, s.k, s.c, e.budget
            )
        })
}

/// A column-block-major tensor `[cols/128][rows][128]`, bf16.
#[derive(Debug, Clone, Copy)]
struct Tensor {
    base: u64,
    rows: u64,
    cols: u64,
}

impl Tensor {
    fn block(&self, cb: u64, r0: u64, rows: u64) -> Range {
        Range::from_len(self.base + (cb * self.rows + r0) * 256, rows * 256)
    }
    fn bytes(&self) -> u64 {
        self.cols.div_ceil(128) * self.rows * 256
    }
}

/// Weights `[C/128][K][128]` in the weight type.
#[derive(Debug, Clone, Copy)]
struct Weights {
    base: u64,
    k: u64,
    c: u64,
    /// Bytes per 128-column block (granule-rounded).
    block: u64,
}

impl Weights {
    fn chunk(&self, w: Dtype, cb: u64, k0: u64, kw: u64) -> (u64, u64) {
        (
            self.base + cb * self.block + w.bytes_for(k0 * 128),
            w.bytes_for(kw * 128),
        )
    }
}

/// Where a GEMM writes.
#[derive(Debug, Clone, Copy)]
enum Out {
    Tensor(Tensor),
    /// Columns [0, head_dim) to K, the rest to V, row-major, of one layer.
    Kv(KvLayout, u32),
}

#[derive(Debug, Clone)]
struct Gemm {
    name: &'static str,
    layer: u32,
    n: u64,
    a: Tensor,
    w: Weights,
    w2: Option<Weights>,
    out: Out,
    residual: Option<Tensor>,
    norm: bool,
    /// Extra lane-ops per out element in the epilogue.
    epi_ops: u64,
}

/// One logged phase.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseLog {
    /// Phase name: "patch_embed", "q", "kv", "attn", "o", "up", "down"
    /// or "projector".
    pub name: String,
    /// Layer of the phase.
    pub layer: u32,
    /// Start cycle.
    pub start: u64,
    /// End cycle (the driver's waits are done).
    pub end: u64,
    /// GEMM: the grid choice.
    pub grid: Option<GridChoice>,
    /// Measured cycles (end - start) over the grid's estimate.
    pub measured_over_estimate: Option<f64>,
    /// The GEMM's lower bounds, and measured over the largest.
    pub roofline: Option<Roofline>,
    /// Measured cycles over the roofline's bound.
    pub measured_over_roofline: Option<f64>,
    /// Attention: query rows per unit and units.
    pub q_tile: Option<u64>,
    /// Attention: work units.
    pub units: Option<u64>,
    /// Tasks invoked in the phase.
    pub tasks: u64,
    /// Writes into HBM or cmem in the phase.
    pub writes: u64,
    /// Joins (fresh signals) in the phase.
    pub joins: u64,
}

/// What the program measured, for the CLI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    /// One entry per phase, in run order.
    pub phases: Vec<PhaseLog>,
    /// Cycle at which each layer's last phase was complete.
    pub layer_end: Vec<u64>,
    /// Bytes read from or written to HBM.
    pub hbm_bytes: u64,
    /// Bytes read from or written to cmem.
    pub cmem_bytes: u64,
    /// Weight bytes streamed, and the weights of every GEMM run counted
    /// once.
    pub w_bytes: u64,
    /// Weight bytes of every GEMM run, counted once.
    pub w_once: u64,
    /// Tasks invoked, all windows.
    pub tasks: u64,
    /// Writes into HBM or cmem.
    pub writes: u64,
    /// Joins (fresh signals).
    pub joins: u64,
    /// Graph windows run (one per phase).
    pub windows: u64,
    /// Ranges written into the KV cache: (layer, base, len).
    pub kv_writes: Vec<(u32, u64, u64)>,
    /// The parameters used.
    pub params: Option<Params>,
    /// The model shape after the overrides.
    pub model: Option<Model>,
}

/// Graph node index: SMs first, then memory owner nodes.
type Node = usize;

/// Owner-side signals of a phase's out writes.
type Dones = Vec<(Node, Range)>;

/// The prefill driver program on the home SM: one graph window per phase.
pub struct Driver {
    m: Machine,
    p: Params,
    model: Model,
    w_dtype: Dtype,
    order: Order,
    g: Graph,
    workers: Vec<usize>,
    /// SMs then memory nodes: sync pool.
    pools: Vec<SyncPool>,
    /// Memory nodes by graph node index (index - sms).
    mem_nodes: Vec<MemNode>,
    /// Graph node of the activation memory.
    act_node: Node,
    /// The activation memory's block (the primary copy).
    act_region: Range,
    /// Copies of the activations: (cluster, address delta from the
    /// primary, owner node); the primary first.
    copies: Vec<(Option<NeutrinoId>, i64, Node)>,
    hbm_node: Node,
    kv: KvLayout,
    budget: u64,
    /// Per-window counters.
    writes: u64,
    joins: u64,
    /// What the run measured.
    pub report: Rc<RefCell<Report>>,
    /// Weights and activations.
    plan: Plan,
    place: Place,
    /// The roofline of the GEMM being built.
    roof: Option<Roofline>,
    /// Ends of this driver's HBM weights and of its activations.
    ends: (u64, u64),
}

/// Where a driver's weights and activations start, and where the vision
/// projector writes (the VLM input); `None` = the standalone defaults.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Place {
    pub hbm_base: Option<u64>,
    pub act_base: Option<u64>,
    pub out_x: Option<u64>,
}

/// Addresses of one model's weights and activations.
#[derive(Debug, Clone)]
struct Plan {
    layers: Vec<LayerW>,
    /// Vision: patch embedding and projector.
    embed: Option<Weights>,
    proj: Option<Weights>,
    pixels: Option<Tensor>,
    /// Residual stream ping-pong.
    x: [Tensor; 2],
    q: Tensor,
    o: Tensor,
    h: Tensor,
    /// The VLM input the projector writes (vision).
    vlm_x: Option<Tensor>,
}

#[derive(Debug, Clone)]
struct LayerW {
    q: Weights,
    /// VLM: K and V (one head); vision: none (in `q`, fused QKV).
    kv: Option<Weights>,
    o: Weights,
    up: Weights,
    gate: Option<Weights>,
    down: Weights,
}

impl Driver {
    /// Builds the standalone driver for `p` on `m`: its weights after the
    /// action expert's HBM map, its activations from the start of the
    /// activation memory.
    pub fn new(m: Machine, p: Params) -> Result<Self, String> {
        Self::new_at(m, p, Place::default())
    }

    /// As [`Driver::new`], with the weights and activations placed by the
    /// caller (the full forward pass).
    pub(crate) fn new_at(m: Machine, p: Params, place: Place) -> Result<Self, String> {
        let model = match p.model.as_str() {
            "vlm" => PALIGEMMA_2B,
            "vision" => SIGLIP_SO400M,
            s => return Err(format!("unknown model {s:?} (vlm, vision)")),
        };
        let or = |v: u64, d: u64| if v == 0 { d } else { v };
        let model = Model {
            layers: if p.layers == 0 {
                model.layers
            } else {
                p.layers
            },
            d: or(p.d, model.d),
            heads: or(p.heads, model.heads),
            kv_heads: or(p.kv_heads, model.kv_heads),
            head_dim: or(p.head_dim, model.head_dim),
            ffn: or(p.ffn, model.ffn),
            ..model
        };
        let workers = worker_list(m.sms.len(), p.home, &p.workers)?;
        let w_dtype = parse_dtype(&p.weight_dtype)?;
        let order = Order::parse(&p.order)?;
        let hbm = m
            .hbm_node
            .clone()
            .ok_or("no owner node with a sync space for HBM")?;
        let mut mem_nodes = vec![hbm];
        mem_nodes.extend(m.cmem_nodes.iter().cloned());
        let act_idx = match p.act_mem.as_str() {
            "hbm" => 0,
            "cmem" if m.cmem_nodes.is_empty() => return Err("act_mem cmem without cmem".into()),
            "cmem" => 1,
            "auto" => usize::from(!m.cmem_nodes.is_empty()),
            s => return Err(format!("unknown act_mem {s:?}")),
        };
        if mem_nodes.iter().any(|x| x.dma.is_none()) {
            return Err("the prefill needs a copy engine at each memory owner (HBM, cmem)".into());
        }
        let g = m.granule;
        let mut own: Vec<Range> = m.sms.iter().map(|s| s.own).collect();
        let mut pools: Vec<SyncPool> = m.sms.iter().map(|s| SyncPool::new(s.sync, g)).collect();
        for x in &mem_nodes {
            own.push(x.own);
            pools.push(SyncPool::new(x.sync, g));
        }
        let budget = if p.smem_budget == 0 {
            m.sms[workers[0]].smem.len()
        } else {
            p.smem_budget
        };
        // HBM map: the action expert's weights and KV first, then ours.
        let (_, _, kv) = crate::adamoe::hbm_map(&m, &p.action)?;
        if p.model == "vlm" {
            let n_tok = vlm_tokens(&p);
            if kv.prefix != n_tok
                || kv.head_dim != model.head_dim
                || kv.layers < model.layers
                || model.kv_heads != 1
            {
                return Err(format!(
                    "VLM prefix {n_tok} x {} x {} layers ({} KV heads) does not match the action KV layout {:?}",
                    model.head_dim, model.layers, model.kv_heads, kv
                ));
            }
        }
        let sms = m.sms.len();
        let driver_own = own[p.home];
        let versions = m.versions.clone();
        let mut d = Self {
            act_node: sms + act_idx,
            act_region: Range::from_len(0, 0),
            copies: Vec::new(),
            hbm_node: sms,
            plan: Plan::empty(),
            place,
            roof: None,
            ends: (0, 0),
            m,
            p,
            model,
            w_dtype,
            order,
            g: Graph::new(driver_own, versions),
            workers,
            pools,
            mem_nodes,
            kv,
            budget,
            writes: 0,
            joins: 0,
            report: Rc::new(RefCell::new(Report::default())),
        };
        // Activation copies: one per cmem cluster, at the same offset.
        let am = d.mem_nodes[act_idx].clone();
        d.act_region = am.mem;
        d.copies = vec![((act_idx > 0).then_some(am.nu), 0, d.act_node)];
        if act_idx > 0 && d.p.act_copies {
            for (i, mn) in d.mem_nodes.iter().enumerate().skip(1) {
                if i != act_idx {
                    d.copies.push((
                        Some(mn.nu),
                        mn.mem.base as i64 - am.mem.base as i64,
                        sms + i,
                    ));
                }
            }
        }
        let (plan, hbm_end, act_end) = d.make_plan()?;
        let used = act_end - am.mem.base;
        for c in &d.copies[1..] {
            if used > d.mem_nodes[c.2 - sms].mem.len() {
                return Err("activation copy does not fit a cluster's cmem".into());
            }
        }
        d.plan = plan;
        d.ends = (hbm_end, act_end);
        {
            let mut r = d.report.borrow_mut();
            r.params = Some(d.p.clone());
            r.model = Some(d.model);
        }
        Ok(d)
    }

    /// The Neutrino of the home SM.
    pub fn home_nu(&self) -> NeutrinoId {
        self.m.sms[self.p.home].nu
    }

    fn tokens(&self) -> u64 {
        match self.p.model.as_str() {
            "vlm" => vlm_tokens(&self.p),
            _ => self.p.images * self.p.tokens_per_image,
        }
    }

    /// The memory node of the activations.
    fn act_node_mem(&self) -> &MemNode {
        &self.mem_nodes[self.act_node - self.m.sms.len()]
    }

    /// Allocates this model's weights in HBM and its activations in the
    /// activation memory: after the KV cache and from the start of the
    /// activation memory, or where `place` says (the forward pass lays out
    /// both models in one map).
    fn make_plan(&self) -> Result<(Plan, u64, u64), String> {
        let g = self.m.granule;
        let w = self.w_dtype;
        let md = self.model;
        let hbm_base = self.place.hbm_base.unwrap_or(self.kv.end());
        let mut hbm = Bump::new(hbm_base, self.m.hbm.bound, g, "HBM");
        let mut weights = |k: u64, c: u64| -> Result<Weights, String> {
            let block = w.bytes_for(k * 128).div_ceil(g) * g;
            let base = hbm.take(c.div_ceil(128) * block)?;
            Ok(Weights { base, k, c, block })
        };
        let vlm = self.p.model == "vlm";
        let qd = md.heads * md.head_dim;
        let embed = if vlm {
            None
        } else {
            Some(weights(PATCH_DIM, md.d)?)
        };
        let mut layers = Vec::new();
        for _ in 0..md.layers {
            layers.push(LayerW {
                q: weights(md.d, if vlm { qd } else { 3 * qd })?,
                kv: if vlm {
                    Some(weights(md.d, 2 * md.kv_heads * md.head_dim)?)
                } else {
                    None
                },
                o: weights(qd, md.d)?,
                gate: if md.gated {
                    Some(weights(md.d, md.ffn)?)
                } else {
                    None
                },
                up: weights(md.d, md.ffn)?,
                down: weights(md.ffn, md.d)?,
            });
        }
        let proj = if vlm {
            None
        } else {
            Some(weights(md.d, PROJ_OUT)?)
        };
        let hbm_end = hbm.next;
        let am = self.act_node_mem();
        let act_base = self.place.act_base.unwrap_or(am.mem.base);
        let mut act = Bump::new(act_base, am.mem.bound, g, "activation memory");
        let mut tensor = |rows: u64, cols: u64| -> Result<Tensor, String> {
            let t = Tensor {
                base: 0,
                rows,
                cols,
            };
            Ok(Tensor {
                base: act.take(t.bytes())?,
                ..t
            })
        };
        let n = self.tokens();
        let n_vlm = vlm_tokens(&self.p);
        let vlm_x = if vlm {
            None
        } else {
            Some(match self.place.out_x {
                Some(base) => Tensor {
                    base,
                    rows: n_vlm,
                    cols: PROJ_OUT,
                },
                None => tensor(n_vlm, PROJ_OUT)?,
            })
        };
        let plan = Plan {
            layers,
            embed,
            proj,
            pixels: if vlm {
                None
            } else {
                Some(tensor(n, PATCH_DIM)?)
            },
            x: [tensor(n, md.d)?, tensor(n, md.d)?],
            q: tensor(n, if vlm { qd } else { 3 * qd })?,
            o: tensor(n, qd)?,
            h: tensor(n, md.ffn)?,
            vlm_x,
        };
        Ok((plan, hbm_end, act.next))
    }

    /// The KV cache layout this driver writes (VLM) or would write.
    pub(crate) fn kv(&self) -> KvLayout {
        self.kv
    }

    /// End of this driver's weights in HBM.
    pub(crate) fn hbm_end(&self) -> u64 {
        self.ends.0
    }

    /// End of this driver's activations.
    pub(crate) fn act_end(&self) -> u64 {
        self.ends.1
    }

    /// The cmem blocks holding a copy of the activations (empty in HBM).
    pub(crate) fn act_copy_mems(&self) -> Vec<Range> {
        if self.act_node == self.hbm_node {
            return Vec::new();
        }
        self.copies
            .iter()
            .map(|c| self.mem_nodes[c.2 - self.m.sms.len()].mem)
            .collect()
    }

    pub(crate) fn act_mem(&self) -> Range {
        self.act_node_mem().mem
    }

    /// The VLM input (the residual stream at layer 0).
    pub(crate) fn x_in(&self) -> u64 {
        self.plan.x[0].base
    }

    // ---- graph helpers ----

    /// Adds a task that `node` owns (the owner follows from `dst`).
    fn add(&mut self, node: Node, mut desc: Descriptor) -> usize {
        // Deps beyond one message go through a join at the same node.
        let rest = excess_deps(&mut desc);
        if !rest.is_empty() {
            let j = self.signal(node, &rest, desc.label.unwrap_or(0));
            desc.deps.push(nu_model::Dep::Unversioned(j.base));
        }
        self.g.add(desc)
    }

    /// A signal for later consumers over `deps`: a join at `node` for
    /// several deps (fan-in), or a pin of the one producer
    /// ([`Graph::pin`]): versions make a remote dependency on data
    /// unambiguous, so a single dep needs no join.
    fn signal(&mut self, node: Node, deps: &[Option<Range>], label: u64) -> Range {
        let mut present: Vec<Range> = deps.iter().flatten().copied().collect();
        if let [r] = present[..] {
            return self.g.pin(r);
        }
        // A join message holds at most MAX_JOIN_DEPS deps (4 beats): a
        // wider fan-in is a tree of joins.
        while present.len() > MAX_JOIN_DEPS {
            present = present
                .chunks(MAX_JOIN_DEPS)
                .map(|c| {
                    let c: Vec<Option<Range>> = c.iter().map(|r| Some(*r)).collect();
                    self.signal(node, &c, label)
                })
                .collect();
        }
        let deps: Vec<Option<Range>> = present.iter().map(|r| Some(*r)).collect();
        let r = self.pools[node].alloc();
        let d = with_deps(Descriptor::join(r, vec![]), &deps, label);
        let ix = self.add(node, d);
        self.joins += 1;
        self.g.pin_task(ix)
    }

    /// True for an address in the primary activation region.
    fn in_act(&self, addr: u64) -> bool {
        self.act_region.covers(&Range::from_len(addr, 1))
    }

    /// The copy of activation address `addr` that `sm` reads: its own
    /// cluster's, if it has one.
    fn local(&self, sm: usize, addr: u64) -> u64 {
        if !self.in_act(addr) {
            return addr;
        }
        let cl = self.m.sms[sm].cluster;
        match self.copies.iter().find(|c| c.0 == cl) {
            Some(&(_, dl, _)) => (addr as i64 + dl) as u64,
            None => addr,
        }
    }

    /// The done signals of out writes `dsts` of `sm`: one per owner node,
    /// each chained to the SM's previous one there; with several owners a
    /// join at `sm` over them. Returns the signal the next writer waits on.
    fn done_signal(
        &mut self,
        sm: usize,
        dsts: &[Range],
        chain: &mut std::collections::BTreeMap<Node, Range>,
        lab: u64,
    ) -> Range {
        let mut by: std::collections::BTreeMap<Node, Vec<Option<Range>>> = Default::default();
        for d in dsts {
            by.entry(self.node_of_mem(d.base))
                .or_default()
                .push(Some(*d));
        }
        let mut sigs = Vec::new();
        for (owner, mut deps) in by {
            deps.push(chain.get(&owner).copied());
            let od = self.signal(owner, &deps, lab);
            chain.insert(owner, od);
            sigs.push(Some(od));
        }
        if sigs.len() == 1 {
            sigs[0].expect("signal")
        } else {
            self.signal(sm, &sigs, lab)
        }
    }

    fn node_of_mem(&self, addr: u64) -> Node {
        let sms = self.m.sms.len();
        (0..self.mem_nodes.len())
            .find(|&i| self.mem_nodes[i].mem.covers(&Range::from_len(addr, 1)))
            .map(|i| sms + i)
            .expect("address in a memory node")
    }

    fn count_bytes(&mut self, addr: u64, bytes: u64) {
        let node = self.node_of_mem(addr);
        let mut r = self.report.borrow_mut();
        if node == self.hbm_node {
            r.hbm_bytes += bytes;
        } else {
            r.cmem_bytes += bytes;
        }
    }

    /// A copy from a shared memory into `sm`'s SMEM.
    fn get(
        &mut self,
        sm: usize,
        src: u64,
        dst: Range,
        bytes: u64,
        deps: &[Option<Range>],
        lab: u64,
    ) {
        let src = self.local(sm, src);
        self.count_bytes(src, bytes);
        let d = self.m.dma(sm, src, dst, bytes, self.p.dma_chunk);
        let d = with_deps(d, deps, lab);
        self.add(sm, d);
    }

    /// A write of `bytes` from `sm`'s SMEM at `src` into `dst` (HBM or
    /// cmem), after `ready` (a signal at `sm`), into every copy of the
    /// activation memory: a pull by each memory owner's copy engine.
    /// Returns the dst ranges.
    fn write_out(
        &mut self,
        sm: usize,
        src: u64,
        dst: Range,
        bytes: u64,
        ready: Range,
        lab: u64,
    ) -> Vec<Range> {
        let deltas: Vec<i64> = if self.in_act(dst.base) {
            self.copies.iter().map(|c| c.1).collect()
        } else {
            vec![0]
        };
        deltas
            .into_iter()
            .map(|dl| {
                let d = Range::from_len((dst.base as i64 + dl) as u64, dst.len());
                self.write_one(sm, src, d, bytes, ready, lab)
            })
            .collect()
    }

    fn write_one(
        &mut self,
        sm: usize,
        src: u64,
        dst: Range,
        bytes: u64,
        ready: Range,
        lab: u64,
    ) -> Range {
        self.count_bytes(dst.base, bytes);
        let owner = self.node_of_mem(dst.base);
        let pe: PeId = self.mem_nodes[owner - self.m.sms.len()]
            .dma
            .expect("copy engine");
        let mut d = self.m.dma(sm, src, dst, bytes, self.p.dma_chunk);
        d.pe = Some(pe);
        let d = with_deps(d, &[Some(ready)], lab);
        self.add(owner, d);
        self.writes += 1;
        dst
    }

    // ---- GEMM ----

    fn gemm_shape(&self, gm: &Gemm) -> GemmShape {
        let (c, cw_fixed) = match gm.out {
            Out::Tensor(t) => (t.cols.min(gm.w.c), None),
            Out::Kv(kv, _) => (2 * kv.head_dim, Some(kv.head_dim)),
        };
        GemmShape {
            n: gm.n,
            k: gm.w.k,
            c,
            w: self.w_dtype,
            gated: gm.w2.is_some(),
            residual: gm.residual.is_some(),
            cw_fixed,
            norm: gm.norm,
            single_acc: self.p.single_accumulator,
        }
    }

    fn cost_env(&self) -> CostEnv {
        let am = self.act_node_mem();
        let act_cluster = (self.act_node != self.hbm_node).then_some(am.nu);
        let mut clusters: Vec<Option<NeutrinoId>> = Vec::new();
        let mut cluster_links = Vec::new();
        let worker_cluster = self
            .workers
            .iter()
            .map(|&w| {
                let s = &self.m.sms[w];
                match clusters.iter().position(|&c| c == s.cluster) {
                    Some(i) => i,
                    None => {
                        clusters.push(s.cluster);
                        let local =
                            act_cluster.is_some() && self.copies.iter().any(|c| c.0 == s.cluster);
                        cluster_links.push((s.rates.cluster_down, s.rates.cluster_up, local));
                        clusters.len() - 1
                    }
                }
            })
            .collect();
        let kcs = if self.p.kc > 0 {
            vec![self.p.kc]
        } else {
            // 1024 is left out: with K 2048 a pass has two chunks, and the
            // measured phases run 1.6 to 1.8x the estimate (unmodeled).
            vec![128, 256, 512]
        };
        CostEnv {
            workers: self.workers.len() as u64,
            budget: self.budget,
            granule: self.m.granule,
            kcs,
            sm: self.m.sms[self.workers[0]].rates,
            hbm_rate: self.hbm_rate(),
            act_rate: self.act_rate(),
            act_in_hbm: self.act_node == self.hbm_node,
            quantize: self.p.quantize,
            pass_overhead: self.p.pass_overhead,
            cw: self.p.cw,
            worker_cluster,
            cluster_links,
            act_copies: self.copies.len(),
            order: self.order,
            // The KV GEMM writes to HBM: the smaller of the two grants.
            owner_grant: [am.grant, self.mem_nodes[0].grant]
                .into_iter()
                .flatten()
                .min(),
        }
    }

    fn act_rate(&self) -> f64 {
        self.act_node_mem().rate
    }

    fn hbm_rate(&self) -> f64 {
        self.mem_nodes[0].rate
    }

    /// Builds one GEMM phase into the window; returns the grid and the
    /// owner-side signals of every out write. The SMs' tasks are emitted
    /// round-robin, one K chunk (or one epilogue) per SM per round, so the
    /// in-order driver does not wait on one SM's tables while the others
    /// idle.
    fn gemm(&mut self, gm: &Gemm) -> Result<(GridChoice, Dones), String> {
        let shape = self.gemm_shape(gm);
        let forced = (self.p.grid != [0, 0]).then_some((self.p.grid[0], self.p.grid[1]));
        let env = self.cost_env();
        let gc = choose_grid(&shape, &env, forced)?;
        self.roof = Some(roofline(&shape, &env));
        {
            // Each weight byte of this GEMM once (the weights-once bound).
            let once: u64 = [Some(gm.w), gm.w2]
                .iter()
                .flatten()
                .map(|w| w.c.div_ceil(128) * w.block)
                .sum();
            self.report.borrow_mut().w_once += once;
        }
        let (kc, g) = (gc.kc, self.m.granule);
        let blocks = shape.c.div_ceil(128);
        let bpc = gc.cs / 128;
        let mats = if gm.w2.is_some() { 2 } else { 1 };
        let wseg = self.m.round(self.w_dtype.bytes_for(kc * 128));
        let mut sms = Vec::new();
        let mut wi = 0;
        for tgi in 0..gc.tg {
            let r0 = tgi * gc.t;
            if r0 >= gm.n {
                break;
            }
            let t = gc.t.min(gm.n - r0);
            for cgi in 0..gc.cg {
                let cb0 = cgi * bpc;
                if cb0 >= blocks {
                    break;
                }
                let sm = self.workers[wi];
                wi += 1;
                let mut lay = Layout::new(self.m.sms[sm].smem, g);
                // Weight-stationary: accumulators, A slots and residual of
                // one token tile; the W slice resident (in w_slot[0]).
                let tt = if gc.ws { t.min(WS_TILE) } else { t };
                let nk = gm.w.k.div_ceil(kc);
                let sets = if self.p.single_accumulator && !gc.ws {
                    1
                } else {
                    2
                };
                let acc = (0..sets)
                    .map(|_| (0..mats).map(|_| lay.take(tt * gc.cw * 2)).collect())
                    .collect::<Result<Vec<Vec<Range>>, String>>()?;
                let a_slot = [lay.take(tt * kc * 2)?, lay.take(tt * kc * 2)?];
                let w_slot = if gc.ws {
                    let wres = lay.take(mats * (gc.cw / 128) * nk * wseg)?;
                    [wres, wres]
                } else {
                    [
                        lay.take(mats * (gc.cw / 128) * wseg)?,
                        lay.take(mats * (gc.cw / 128) * wseg)?,
                    ]
                };
                let res = if gm.residual.is_some() {
                    Some(lay.take(tt * gc.cw * 2)?)
                } else {
                    None
                };
                let ntt = t.div_ceil(tt);
                let stats = lay.take(ntt * self.m.round(tt * 8))?;
                sms.push(GemmSm {
                    sm,
                    r0,
                    t,
                    cb: cb0,
                    cb1: (cb0 + bpc).min(blocks),
                    acc,
                    a_slot,
                    w_slot,
                    res,
                    stats,
                    free: [None, None],
                    owner_done: Vec::new(),
                    chain: Default::default(),
                    prev_epi: None,
                    pass: 0,
                    i: 0,
                    tt,
                    tti: 0,
                    wfree: vec![None; nk as usize],
                    tile_done: Vec::new(),
                    parity: 0,
                });
            }
        }
        let mut dones = Vec::new();
        loop {
            let mut any = false;
            for st in &mut sms {
                if st.cb < st.cb1 {
                    any = true;
                    if gc.ws {
                        self.gemm_ws_step(gm, &gc, st, &mut dones);
                    } else {
                        self.gemm_step(gm, &gc, st, &mut dones);
                    }
                }
            }
            if !any {
                break;
            }
        }
        Ok((gc, dones))
    }

    /// Weight-stationary: one K chunk of the current token tile of the
    /// current pass of `st` (the first token tile also loads that chunk of
    /// the pass's W slice, which the later tiles reuse), or the tile's
    /// epilogue and writes.
    fn gemm_ws_step(&mut self, gm: &Gemm, gc: &GridChoice, st: &mut GemmSm, dones: &mut Dones) {
        let (sm, kc, tt) = (st.sm, gc.kc, st.tt);
        let k = gm.w.k;
        let nk = k.div_ceil(kc);
        let mats = st.acc[0].len();
        let wseg = self.m.round(self.w_dtype.bytes_for(kc * 128));
        let pb = (gc.cw / 128).min(st.cb1 - st.cb);
        let cw = pb * 128;
        let ntt = st.t.div_ceil(tt);
        let (pass, cb, tti) = (st.pass, st.cb, st.tti);
        let r = st.r0 + tti * tt;
        let rows = tt.min(st.r0 + st.t - r);
        let gtile = pass * ntt + tti;
        let lab = |kind, idx: u64| {
            label::label(
                kind,
                0,
                gm.layer,
                sm as u32,
                ((gtile % 4096) * 4096 + idx) as u32,
            )
        };
        let set = (gtile % 2) as usize;
        let acc_p = st.acc[set].clone();
        // Ranges are sized for a full token tile, so a short last tile and
        // the next pass's full first tile name the same range.
        let stats = Range::from_len(
            st.stats.base + tti * self.m.round(tt * 8),
            self.m.round(tt * 8),
        );
        let wslot = |mi: u64, j: u64, i: u64| {
            Range::from_len(st.w_slot[0].base + ((mi * pb + j) * nk + i) * wseg, wseg)
        };
        if st.i < nk {
            let i = st.i;
            st.i += 1;
            let k0 = i * kc;
            let kw = kc.min(k - k0);
            let s = st.parity;
            st.parity ^= 1;
            let free = st.free[s];
            // WAR on this out tile set: the writes of the tile two back.
            let war = if gtile >= 2 {
                st.tile_done[(gtile - 2) as usize]
            } else {
                None
            };
            let mut asub = Vec::new();
            for b in 0..kw.div_ceil(128) {
                let cols = 128.min(kw - b * 128);
                let a_src = gm.a.block(k0 / 128 + b, r, rows);
                let dst = Range::from_len(st.a_slot[s].base + b * tt * 256, tt * 256);
                asub.push(Some(dst));
                self.get(
                    sm,
                    a_src.base,
                    dst,
                    rows * cols * 2,
                    &[free],
                    lab(Kind::DmaAct, i),
                );
            }
            let mut wsub: Vec<Vec<Option<Range>>> = vec![Vec::new(); mats];
            for (mi, wm) in [Some(gm.w), gm.w2].into_iter().flatten().enumerate() {
                for j in 0..pb {
                    let dst = wslot(mi as u64, j, i);
                    wsub[mi].push(Some(dst));
                    if tti == 0 {
                        let (src, bytes) = wm.chunk(self.w_dtype, cb + j, k0, kw);
                        let wf = st.wfree[i as usize];
                        self.get(sm, src, dst, bytes, &[wf], lab(Kind::DmaW, i));
                        self.report.borrow_mut().w_bytes += bytes;
                    }
                }
            }
            let mut readers = Vec::new();
            if gm.norm && pass == 0 {
                let v = self.m.vector(
                    sm,
                    stats,
                    rows * kw * self.p.norm_ops,
                    rows * kw * 2,
                    rows * 8,
                );
                let deps = [asub.clone(), vec![Some(stats)]].concat();
                let v = with_deps(v, &deps, lab(Kind::Vec, i));
                self.add(sm, v);
                readers.push(Some(stats));
            }
            let mut outs = Vec::new();
            for (mi, &out) in acc_p.iter().enumerate() {
                let b = wslot(mi as u64, 0, i).base;
                let a = st.a_slot[s].base;
                let mm = self
                    .m
                    .matmul(sm, a, b, out, rows, cw, kw, i > 0, self.p.tile);
                let mm = keep(with_b_dtype(self.w_dtype, mm), i + 1 < nk);
                let chain = if i == 0 { war } else { Some(out) };
                let mut deps = asub.clone();
                deps.push(chain);
                deps.extend(wsub[mi].iter().copied());
                let mm = with_deps(mm, &deps, lab(Kind::FfnUp, i));
                self.add(sm, mm);
                readers.push(Some(out));
                outs.push(Some(out));
            }
            st.free[s] = Some(self.signal(sm, &readers, lab(Kind::Join, i)));
            if tti + 1 == ntt {
                // The last token tile read this W chunk: the next pass may
                // overwrite it.
                st.wfree[i as usize] = Some(self.signal(sm, &outs, lab(Kind::Join, 3000 + i)));
            }
            return;
        }
        // Epilogue of the token tile.
        let od = self.epilogue(gm, st, &acc_p, stats, (cb, pb), (r, rows, tt), &lab);
        st.tile_done.push(Some(od));
        st.i = 0;
        st.tti += 1;
        if st.tti == ntt {
            st.tti = 0;
            st.cb += pb;
            st.pass += 1;
            if st.cb >= st.cb1 {
                dones.extend(st.chain.iter().map(|(&n, &r)| (n, r)));
            }
        }
    }

    /// One K chunk of the current pass of `st`, or its epilogue and writes.
    fn gemm_step(&mut self, gm: &Gemm, gc: &GridChoice, st: &mut GemmSm, dones: &mut Dones) {
        let (sm, t, r0, kc) = (st.sm, st.t, st.r0, gc.kc);
        let k = gm.w.k;
        let nk = k.div_ceil(kc);
        let mats = st.acc[0].len();
        let wseg = self.m.round(self.w_dtype.bytes_for(kc * 128));
        let pb = (gc.cw / 128).min(st.cb1 - st.cb);
        let cw = pb * 128;
        let (pass, cb) = (st.pass, st.cb);
        let lab =
            |kind, idx: u64| label::label(kind, 0, gm.layer, sm as u32, (pass * 4096 + idx) as u32);
        let sets = st.acc.len() as u64;
        let acc_p = st.acc[(pass % sets) as usize].clone();
        if st.i < nk {
            let i = st.i;
            st.i += 1;
            let k0 = i * kc;
            let kw = kc.min(k - k0);
            let s = (i % 2) as usize;
            let free = st.free[s];
            // WAR: the out writes of pass - sets read this accumulator.
            let war = if pass >= sets {
                st.owner_done[(pass - sets) as usize]
            } else {
                None
            };
            // A chunk: rows r0..r0+t of the column blocks of [k0, k0 + kw).
            let mut asub = Vec::new();
            for b in 0..kw.div_ceil(128) {
                let cols = 128.min(kw - b * 128);
                let a_src = gm.a.block(k0 / 128 + b, r0, t);
                let dst = Range::from_len(st.a_slot[s].base + b * t * 256, t * 256);
                asub.push(Some(dst));
                self.get(
                    sm,
                    a_src.base,
                    dst,
                    t * cols * 2,
                    &[free],
                    lab(Kind::DmaAct, i),
                );
            }
            // W chunks of this pass's blocks.
            let mut wsub: Vec<Vec<Option<Range>>> = vec![Vec::new(); mats];
            for (mi, wm) in [Some(gm.w), gm.w2].into_iter().flatten().enumerate() {
                for j in 0..pb {
                    let (src, bytes) = wm.chunk(self.w_dtype, cb + j, k0, kw);
                    let off = (mi as u64 * pb + j) * wseg;
                    let dst = Range::from_len(st.w_slot[s].base + off, wseg);
                    wsub[mi].push(Some(dst));
                    self.get(sm, src, dst, bytes, &[free], lab(Kind::DmaW, i));
                    self.report.borrow_mut().w_bytes += bytes;
                }
            }
            let mut readers = Vec::new();
            if gm.norm && pass == 0 {
                let v = self
                    .m
                    .vector(sm, st.stats, t * kw * self.p.norm_ops, t * kw * 2, t * 8);
                let deps = [asub.clone(), vec![Some(st.stats)]].concat();
                let v = with_deps(v, &deps, lab(Kind::Vec, i));
                self.add(sm, v);
                readers.push(Some(st.stats));
            }
            for (mi, &out) in acc_p.iter().enumerate() {
                let b = st.w_slot[s].base + mi as u64 * pb * wseg;
                let a = st.a_slot[s].base;
                let mm = self.m.matmul(sm, a, b, out, t, cw, kw, i > 0, self.p.tile);
                let mm = keep(with_b_dtype(self.w_dtype, mm), i + 1 < nk);
                let chain = if i == 0 { war } else { Some(out) };
                let mut deps = asub.clone();
                deps.push(chain);
                deps.extend(wsub[mi].iter().copied());
                let mm = with_deps(mm, &deps, lab(Kind::FfnUp, i));
                self.add(sm, mm);
                readers.push(Some(out));
            }
            // Slot s is free when this chunk's readers are done.
            st.free[s] = Some(self.signal(sm, &readers, lab(Kind::Join, i)));
            return;
        }
        let stats = st.stats;
        let od = self.epilogue(gm, st, &acc_p, stats, (cb, pb), (r0, t, t), &lab);
        st.owner_done.push(Some(od));
        if st.cb + pb >= st.cb1 {
            dones.extend(st.chain.iter().map(|(&n, &r)| (n, r)));
        }
        st.cb += pb;
        st.pass += 1;
        st.i = 0;
    }

    /// The epilogue of one output tile, rows `r..r + rows` of the pass's
    /// column blocks `cb..cb + pb` (SMEM tiles `stride` rows apart): norm
    /// scale, activation or RoPE and residual on the Muon, then the out
    /// writes. Returns the tile's done signal.
    #[allow(clippy::too_many_arguments)]
    fn epilogue(
        &mut self,
        gm: &Gemm,
        st: &mut GemmSm,
        acc_p: &[Range],
        stats: Range,
        (cb, pb): (u64, u64),
        (r, rows, stride): (u64, u64, u64),
        lab: &dyn Fn(Kind, u64) -> u64,
    ) -> Range {
        let sm = st.sm;
        let cw = pb * 128;
        let mut rsub = Vec::new();
        if let (Some(rt), Some(rb)) = (gm.residual, st.res) {
            for j in 0..pb {
                let src = rt.block(cb + j, r, rows);
                let dst = Range::from_len(rb.base + j * stride * 256, stride * 256);
                rsub.push(Some(dst));
                self.get(
                    sm,
                    src.base,
                    dst,
                    rows * 256,
                    &[st.prev_epi],
                    lab(Kind::DmaAct, 9000 + j),
                );
            }
        }
        let resid = u64::from(gm.residual.is_some());
        let per = u64::from(gm.norm) + gm.epi_ops + resid * self.p.add_ops;
        let ins = (acc_p.len() as u64 + resid) * rows * cw * 2;
        let v = self
            .m
            .vector(sm, acc_p[0], rows * cw * per.max(1), ins, rows * cw * 2);
        let mut deps: Vec<Option<Range>> = acc_p.iter().map(|r| Some(*r)).collect();
        if gm.norm {
            deps.push(Some(stats));
        }
        deps.extend(rsub);
        let v = with_deps(v, &deps, lab(Kind::Vec, 9999));
        self.add(sm, v);
        let ready = self.signal(sm, &[Some(acc_p[0])], lab(Kind::Join, 9999));
        st.prev_epi = Some(ready);
        // Out writes.
        let mut dsts = Vec::new();
        match gm.out {
            Out::Tensor(ot) => {
                for j in 0..pb {
                    let dst = ot.block(cb + j, r, rows);
                    let src = acc_p[0].base + j * rows * 256;
                    dsts.extend(self.write_out(
                        sm,
                        src,
                        dst,
                        rows * 256,
                        ready,
                        lab(Kind::DmaAct, 8000 + j),
                    ));
                }
            }
            Out::Kv(kv, l) => {
                let whole = if cb == 0 { kv.k(l) } else { kv.v(l) };
                let bytes = rows * kv.head_dim * 2;
                let dst = Range::from_len(whole.base + r * kv.head_dim * 2, bytes);
                let dst = self.write_one(
                    sm,
                    acc_p[0].base,
                    dst,
                    bytes,
                    ready,
                    lab(Kind::DmaAct, 8000),
                );
                self.report
                    .borrow_mut()
                    .kv_writes
                    .push((l, dst.base, dst.len()));
                dsts.push(dst);
            }
        }
        self.done_signal(sm, &dsts, &mut st.chain, lab(Kind::Join, 9998))
    }

    // ---- attention ----

    /// SMEM bytes of one attention unit of `qt` query rows.
    fn attn_need(&self, qt: u64) -> u64 {
        let md = self.model;
        let r = |b: u64| self.m.round(b);
        let kb = self.p.key_block;
        let q = md.heads * r(qt * md.head_dim * 2);
        let o = md.heads * r(qt * md.head_dim * 2);
        let kvb = 2 * 2 * r(kb * md.head_dim * 2);
        let s = 2 * r(qt * kb * 2);
        let st = md.heads * r(qt * 8);
        q + o + kvb + s + st
    }

    /// Flash attention over `seqs` sequences of `seq` tokens: units of one
    /// query tile and all heads, dealt round-robin to the workers; each SM
    /// runs its units in order. Emission is round-robin over the SMs, one
    /// K/V block per SM per round.
    fn attention(&mut self, layer: u32, seqs: u64, seq: u64) -> Result<(u64, u64, Dones), String> {
        let md = self.model;
        let kb = self.p.key_block;
        let qt = if self.p.q_tile > 0 {
            self.p.q_tile
        } else {
            // The fitting tile with the least estimated time: rounds of
            // units over the workers, each unit bounded by its link
            // transfer (K/V blocks, Q in, O out) or its Atlas work.
            let w = self.workers.len() as u64;
            let r = self.m.sms[self.workers[0]].rates;
            let est = |q: u64| {
                let units = seqs * seq.div_ceil(q);
                let kv = 2 * md.kv_heads * seq * md.head_dim * 2;
                let qo = 2 * md.heads * q * md.head_dim * 2;
                let rows = if self.p.quantize {
                    q.div_ceil(128) * 128
                } else {
                    q
                };
                let macs = 2 * md.heads * rows * seq.next_multiple_of(128) * md.head_dim;
                let unit =
                    ((kv + qo) as f64 / r.link_down.max(1e-9)).max(macs as f64 / r.mac.max(1e-9));
                units.div_ceil(w) as f64 * unit
            };
            [256, 128, 64, 32, 16]
                .into_iter()
                .filter(|&q| q <= seq.next_power_of_two() && self.attn_need(q) <= self.budget)
                .min_by(|&a, &b| est(a).total_cmp(&est(b)))
                .ok_or("no attention tile fits SMEM")?
        };
        if self.attn_need(qt) > self.budget {
            return Err(format!("q_tile {qt} does not fit SMEM"));
        }
        let g = self.m.granule;
        let mut units = Vec::new();
        for sidx in 0..seqs {
            let mut r = 0;
            while r < seq {
                units.push((sidx, r, qt.min(seq - r)));
                r += qt;
            }
        }
        let nunits = units.len() as u64;
        let mut sms: Vec<AttnSm> = Vec::new();
        for (ui, u) in units.into_iter().enumerate() {
            let w = ui % self.workers.len();
            if w == sms.len() {
                let sm = self.workers[w];
                let mut lay = Layout::new(self.m.sms[sm].smem, g);
                let mut take = |b: u64| lay.take(b);
                let qb = (0..md.heads)
                    .map(|_| take(qt * md.head_dim * 2))
                    .collect::<Result<_, _>>()?;
                let ob = (0..md.heads)
                    .map(|_| take(qt * md.head_dim * 2))
                    .collect::<Result<_, _>>()?;
                let kslot = [take(kb * md.head_dim * 2)?, take(kb * md.head_dim * 2)?];
                let vslot = [take(kb * md.head_dim * 2)?, take(kb * md.head_dim * 2)?];
                let sbuf = [take(qt * kb * 2)?, take(qt * kb * 2)?];
                let st = (0..md.heads)
                    .map(|_| take(qt * 8))
                    .collect::<Result<_, _>>()?;
                sms.push(AttnSm {
                    sm,
                    bufs: AttnBufs {
                        qb,
                        ob,
                        kslot,
                        vslot,
                        sbuf,
                        st,
                    },
                    units: Vec::new(),
                    next: 0,
                    step: 0,
                    last_done: None,
                    chain: Default::default(),
                    free: [None, None],
                    sfree: [None, None],
                    si: 0,
                });
            }
            sms[w].units.push((ui as u64, u));
        }
        let mut dones = Vec::new();
        loop {
            let mut any = false;
            for st in &mut sms {
                if st.next < st.units.len() {
                    any = true;
                    self.attn_step(layer, seq, st, &mut dones);
                }
            }
            if !any {
                break;
            }
        }
        Ok((qt, nunits, dones))
    }

    /// One step of the current unit of `st`: its Q loads (step 0), one K/V
    /// block (steps 1..=kv_heads x blocks), or its normalize and writes.
    fn attn_step(&mut self, layer: u32, seq: u64, st: &mut AttnSm, dones: &mut Dones) {
        let md = self.model;
        let kb = self.p.key_block;
        let nkb = seq.div_ceil(kb);
        let sm = st.sm;
        let (ui, (sidx, q0, rows)) = st.units[st.next];
        let lab =
            |kind, idx: u64| label::label(kind, 0, layer, sm as u32, (ui * 65536 + idx) as u32);
        let row0 = sidx * seq + q0;
        let b = st.bufs.clone();
        let war = st.last_done;
        let blocks_total = md.kv_heads * nkb;
        if st.step == 0 {
            for h in 0..md.heads {
                let cb = h * md.head_dim / 128;
                let src = self.plan.q.block(cb, row0, rows);
                self.get(
                    sm,
                    src.base,
                    b.qb[h as usize],
                    rows * md.head_dim * 2,
                    &[war],
                    lab(Kind::DmaAct, h),
                );
            }
            st.free = [war, war];
            st.sfree = [war, war];
            st.step = 1;
            return;
        }
        if st.step <= blocks_total {
            let x = st.step - 1;
            st.step += 1;
            let (kvh, j) = (x / nkb, x % nkb);
            let qpk = md.heads / md.kv_heads;
            let keys = kb.min(seq - j * kb);
            let s = (j % 2) as usize;
            let (ksrc, vsrc) = self.kv_block(layer, kvh, sidx * seq + j * kb);
            let kv_bytes = keys * md.head_dim * 2;
            self.get(
                sm,
                ksrc,
                b.kslot[s],
                kv_bytes,
                &[st.free[s]],
                lab(Kind::DmaAct, 100 + x),
            );
            self.get(
                sm,
                vsrc,
                b.vslot[s],
                kv_bytes,
                &[st.free[s]],
                lab(Kind::DmaAct, 200 + x),
            );
            let mut readers = Vec::new();
            for hq in 0..qpk {
                let h = (kvh * qpk + hq) as usize;
                let scoreboard = b.sbuf[st.si % 2];
                let sdep = st.sfree[st.si % 2];
                let (qh, kslot) = (b.qb[h], b.kslot[s]);
                let mm = self.m.matmul(
                    sm,
                    qh.base,
                    kslot.base,
                    scoreboard,
                    rows,
                    keys,
                    md.head_dim,
                    false,
                    self.p.tile,
                );
                let mm = with_deps(
                    mm,
                    &[Some(qh), Some(kslot), sdep],
                    lab(Kind::Attn, 1000 + x),
                );
                self.add(sm, mm);
                let ops = rows * keys * self.p.softmax_ops + rows * md.head_dim;
                let v = self.m.vector(sm, b.st[h], ops, rows * keys * 2, rows * 8);
                let v = with_deps(
                    v,
                    &[Some(scoreboard), Some(b.st[h])],
                    lab(Kind::Vec, 2000 + x),
                );
                self.add(sm, v);
                let (oh, vslot) = (b.ob[h], b.vslot[s]);
                let pv = self.m.matmul(
                    sm,
                    scoreboard.base,
                    vslot.base,
                    oh,
                    rows,
                    md.head_dim,
                    keys,
                    j > 0,
                    self.p.tile,
                );
                let pv = keep(pv, j + 1 < nkb);
                let first = if j == 0 { war } else { Some(oh) };
                let pv = with_deps(
                    pv,
                    &[Some(scoreboard), Some(vslot), Some(b.st[h]), first],
                    lab(Kind::Attn, 3000 + x),
                );
                self.add(sm, pv);
                readers.push(Some(oh));
                st.sfree[st.si % 2] = Some(self.signal(sm, &[Some(oh)], lab(Kind::Join, 4000 + x)));
                st.si += 1;
            }
            st.free[s] = Some(self.signal(sm, &readers, lab(Kind::Join, 5000 + x)));
            return;
        }
        // Normalize and write O (column blocks of the o tensor).
        let obytes = rows * md.head_dim * 2;
        let deps: Vec<Option<Range>> = b.ob.iter().chain(&b.st).map(|r| Some(*r)).collect();
        let v = self.m.vector(
            sm,
            b.ob[0],
            md.heads * rows * md.head_dim,
            md.heads * obytes,
            md.heads * obytes,
        );
        let v = with_deps(v, &deps, lab(Kind::Vec, 9999));
        self.add(sm, v);
        let odeps: Vec<Option<Range>> = b.ob.iter().map(|r| Some(*r)).collect();
        let ready = self.signal(sm, &odeps, lab(Kind::Join, 9999));
        let mut dsts = Vec::new();
        for cb in 0..(md.heads * md.head_dim).div_ceil(128) {
            let dst = self.plan.o.block(cb, row0, rows);
            let src = b.ob[((cb * 128 / md.head_dim) as usize).min(b.ob.len() - 1)].base;
            dsts.extend(self.write_out(
                sm,
                src,
                dst,
                rows * 256,
                ready,
                lab(Kind::DmaAct, 8000 + cb),
            ));
        }
        let od = self.done_signal(sm, &dsts, &mut st.chain, lab(Kind::Join, 9998));
        st.last_done = Some(od);
        st.next += 1;
        if st.next == st.units.len() {
            dones.extend(st.chain.iter().map(|(&n, &r)| (n, r)));
        }
        st.step = 0;
    }

    /// Addresses of the K and V block of KV head `kvh` starting at key
    /// `key` of layer `layer`.
    fn kv_block(&self, layer: u32, kvh: u64, key: u64) -> (u64, u64) {
        let md = self.model;
        if self.p.model == "vlm" {
            let off = key * md.head_dim * 2;
            (self.kv.k(layer).base + off, self.kv.v(layer).base + off)
        } else {
            // Fused QKV tensor: K columns after the Q columns, then V.
            let qkv = self.plan.q;
            let kc = (md.heads * md.head_dim + kvh * md.head_dim) / 128;
            let vc = (2 * md.heads * md.head_dim + kvh * md.head_dim) / 128;
            (qkv.block(kc, key, 1).base, qkv.block(vc, key, 1).base)
        }
    }

    // ---- driver ----

    #[allow(clippy::too_many_arguments)]
    async fn close(
        &mut self,
        ctx: &NuCtx,
        name: &str,
        layer: u32,
        start: u64,
        grid: Option<GridChoice>,
        attn: Option<(u64, u64)>,
        dones: Vec<(Node, Range)>,
    ) {
        // One join per owner node over its signals; the driver waits on each.
        let mut by: std::collections::BTreeMap<Node, Vec<Option<Range>>> = Default::default();
        for (n, r) in dones {
            by.entry(n).or_default().push(Some(r));
        }
        let mut waits = Vec::new();
        for (n, deps) in by {
            let r = self.signal(
                n,
                &deps,
                label::label(Kind::Join, 0, layer, n as u32, 0xffff),
            );
            waits.push(r);
        }
        let before = self.g.invoked;
        self.g.flush(ctx).await;
        for r in waits {
            if let Some(k) = self.g.key(&r) {
                ctx.wait_key(k).await;
            }
        }
        let end = ctx.now();
        ctx.mark(&format!("{name} L{layer}")).await;
        let roof = self.roof.take();
        let mut rep = self.report.borrow_mut();
        rep.windows += 1;
        rep.tasks = self.g.invoked;
        rep.writes += self.writes;
        rep.joins += self.joins;
        rep.phases.push(PhaseLog {
            name: name.into(),
            layer,
            start,
            end,
            grid,
            measured_over_estimate: grid.map(|g| (end - start) as f64 / g.cost.max(1.0)),
            roofline: roof,
            measured_over_roofline: roof.map(|r| (end - start) as f64 / r.bound.max(1.0)),
            q_tile: attn.map(|a| a.0),
            units: attn.map(|a| a.1),
            tasks: self.g.invoked - before,
            writes: self.writes,
            joins: self.joins,
        });
        self.writes = 0;
        self.joins = 0;
    }

    async fn run_gemm(&mut self, ctx: &NuCtx, gm: Gemm) {
        let start = ctx.now();
        match self.gemm(&gm) {
            Ok((gc, dones)) => {
                self.close(ctx, gm.name, gm.layer, start, Some(gc), None, dones)
                    .await
            }
            Err(e) => panic!("{}: {e}", gm.name),
        }
    }

    /// Runs every phase of the model and fills `report`.
    pub async fn run(mut self, ctx: NuCtx) {
        ctx.mark(&format!("{} start", self.p.model)).await;
        let md = self.model;
        let n = self.tokens();
        let plan = self.plan.clone();
        let vlm = self.p.model == "vlm";
        // Residual stream: x[0] in, x[1] after attention, x[0] after the FFN.
        if !vlm {
            // Patch embedding (+ position embedding in the epilogue).
            let gm = Gemm {
                name: "patch_embed",
                layer: 0,
                n,
                a: plan.pixels.expect("pixels"),
                w: plan.embed.expect("embed"),
                w2: None,
                out: Out::Tensor(plan.x[0]),
                residual: None,
                norm: false,
                epi_ops: self.p.add_ops,
            };
            self.run_gemm(&ctx, gm).await;
        }
        for l in 0..md.layers {
            let lw = plan.layers[l as usize].clone();
            let (xi, xo) = (plan.x[0], plan.x[1]);
            let rope = if md.rope { self.p.rope_ops } else { 0 };
            // The VLM's last layer feeds only its KV downstream.
            let last = vlm && l + 1 == md.layers;
            let kv_only = last && self.p.last_layer == "kv";
            let no_ffn = kv_only || (last && self.p.last_layer == "no_ffn");
            if !kv_only {
                self.run_gemm(
                    &ctx,
                    Gemm {
                        name: "q",
                        layer: l,
                        n,
                        a: xi,
                        w: lw.q,
                        w2: None,
                        out: Out::Tensor(plan.q),
                        residual: None,
                        norm: true,
                        epi_ops: rope,
                    },
                )
                .await;
            }
            if let Some(kvw) = lw.kv {
                self.run_gemm(
                    &ctx,
                    Gemm {
                        name: "kv",
                        layer: l,
                        n,
                        a: xi,
                        w: kvw,
                        w2: None,
                        out: Out::Kv(self.kv, l),
                        residual: None,
                        norm: true,
                        epi_ops: rope,
                    },
                )
                .await;
            }
            if kv_only {
                self.report.borrow_mut().layer_end.push(ctx.now());
                continue;
            }
            let start = ctx.now();
            let (seqs, seq) = if vlm {
                (1, n)
            } else {
                (self.p.images, self.p.tokens_per_image)
            };
            match self.attention(l, seqs, seq) {
                Ok((qt, units, dones)) => {
                    self.close(&ctx, "attn", l, start, None, Some((qt, units)), dones)
                        .await
                }
                Err(e) => panic!("attention: {e}"),
            }
            self.run_gemm(
                &ctx,
                Gemm {
                    name: "o",
                    layer: l,
                    n,
                    a: plan.o,
                    w: lw.o,
                    w2: None,
                    out: Out::Tensor(xo),
                    residual: Some(xi),
                    norm: false,
                    epi_ops: 0,
                },
            )
            .await;
            if no_ffn {
                self.report.borrow_mut().layer_end.push(ctx.now());
                continue;
            }
            self.run_gemm(
                &ctx,
                Gemm {
                    name: "up",
                    layer: l,
                    n,
                    a: xo,
                    w: lw.gate.unwrap_or(lw.up),
                    w2: lw.gate.map(|_| lw.up),
                    out: Out::Tensor(plan.h),
                    residual: None,
                    norm: true,
                    epi_ops: self.p.act_ops,
                },
            )
            .await;
            self.run_gemm(
                &ctx,
                Gemm {
                    name: "down",
                    layer: l,
                    n,
                    a: plan.h,
                    w: lw.down,
                    w2: None,
                    out: Out::Tensor(xi),
                    residual: Some(xo),
                    norm: false,
                    epi_ops: 0,
                },
            )
            .await;
            self.report.borrow_mut().layer_end.push(ctx.now());
        }
        if !vlm {
            // Post-LN and the projector into the VLM input rows.
            let gm = Gemm {
                name: "projector",
                layer: md.layers,
                n,
                a: plan.x[0],
                w: plan.proj.expect("projector"),
                w2: None,
                out: Out::Tensor(plan.vlm_x.expect("vlm x")),
                residual: None,
                norm: true,
                epi_ops: 0,
            };
            self.run_gemm(&ctx, gm).await;
        }
    }
}

impl Plan {
    fn empty() -> Self {
        let t = Tensor {
            base: 0,
            rows: 0,
            cols: 0,
        };
        Self {
            layers: Vec::new(),
            embed: None,
            proj: None,
            pixels: None,
            x: [t, t],
            q: t,
            o: t,
            h: t,
            vlm_x: None,
        }
    }
}

/// One SM's state in a GEMM phase.
struct GemmSm {
    sm: usize,
    r0: u64,
    t: u64,
    /// Next column block and the end of the group's blocks.
    cb: u64,
    cb1: u64,
    acc: Vec<Vec<Range>>,
    a_slot: [Range; 2],
    w_slot: [Range; 2],
    res: Option<Range>,
    stats: Range,
    /// Per staging slot: the signal that its last readers are done.
    free: [Option<Range>; 2],
    owner_done: Vec<Option<Range>>,
    /// Per owner node: the SM's latest done signal there (chained).
    chain: std::collections::BTreeMap<Node, Range>,
    prev_epi: Option<Range>,
    pass: u64,
    /// Next K chunk of the pass.
    i: u64,
    /// Weight-stationary: rows per token tile, the current token tile, per
    /// K chunk the signal that the resident W chunk's readers are done, the
    /// done signal of every token tile written, the A slot parity.
    tt: u64,
    tti: u64,
    wfree: Vec<Option<Range>>,
    tile_done: Vec<Option<Range>>,
    parity: usize,
}

/// One SM's state in an attention phase.
struct AttnSm {
    sm: usize,
    bufs: AttnBufs,
    /// (unit index, (sequence, first query row, rows)).
    units: Vec<(u64, (u64, u64, u64))>,
    next: usize,
    step: u64,
    /// The owner signal of the previous unit's O writes.
    last_done: Option<Range>,
    chain: std::collections::BTreeMap<Node, Range>,
    free: [Option<Range>; 2],
    sfree: [Option<Range>; 2],
    si: usize,
}

/// One SM's attention buffers.
#[derive(Debug, Clone)]
struct AttnBufs {
    qb: Vec<Range>,
    ob: Vec<Range>,
    kslot: [Range; 2],
    vslot: [Range; 2],
    sbuf: [Range; 2],
    st: Vec<Range>,
}

/// Bump allocator over a shared memory.
struct Bump {
    next: u64,
    bound: u64,
    g: u64,
    what: &'static str,
}

impl Bump {
    fn new(base: u64, bound: u64, g: u64, what: &'static str) -> Self {
        Self {
            next: base,
            bound,
            g,
            what,
        }
    }
    fn take(&mut self, bytes: u64) -> Result<u64, String> {
        let b = self.next;
        let len = bytes.div_ceil(self.g) * self.g;
        if b + len > self.bound {
            return Err(format!("{} too small for the prefill map", self.what));
        }
        self.next += len;
        Ok(b)
    }
}

/// Tokens of the VLM prefix: the image tokens and the text tokens.
fn vlm_tokens(p: &Params) -> u64 {
    p.images * p.tokens_per_image + p.text_tokens
}

/// Marks a matmul to leave C in the Atlas accumulator for the next task of
/// its chain (H3).
fn keep(mut d: Descriptor, on: bool) -> Descriptor {
    if let Args::Matmul { keep, .. } = &mut d.args {
        *keep = on;
    }
    d
}
