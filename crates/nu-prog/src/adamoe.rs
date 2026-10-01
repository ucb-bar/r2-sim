//! The AdaMoE action-expert stack (pi0 action expert, MoE FFN): `layers`
//! transformer layers per denoise step, `steps` steps, `tokens` suffix
//! tokens attending to a `prefix`-entry KV cache, and a top-1 MoE FFN with
//! `experts` routed experts plus one shared expert.
//!
//! Structure (step 5a of the architecture plan):
//! * One driver program on the home SM. Every other task is forwarded to
//!   the SM that owns its `dst` (compute follows data).
//! * Weights stream from HBM through two staging slots per SM, one block
//!   (`block_cols` columns or rows, full K) per DMA; the next DMA into a slot
//!   depends on the last reader of that slot.
//! * Attention: heads round-robin over `attn_sms`; each head SM computes its
//!   own K/V projection (MQA, duplicated; reported) and streams the prefix
//!   KV in blocks of `key_block` keys. Wo is row-parallel into a partial sum.
//! * FFN (shared and routed): hidden blocks round-robin over the expert's
//!   SMs; gate/up column blocks, GeGLU on the Muon, down row blocks
//!   accumulated into a partial sum per SM.
//! * Partial sums are pulled to the home SM and added into the residual.
//! * Router: a matmul at home whose output (top-1 expert id per token) comes
//!   from the trace; the driver loads it (the data-dependent point) and
//!   dispatches only the experts that received tokens.
//! * Windows: the driver ends each window with a barrier (the router wait
//!   and load, and the step end), so a window's graph needs no deps on
//!   earlier windows. Cross-SM deps name fresh signals (joins in the
//!   producer's sync-only space).
//!
//! The vision encoder and the VLM prefill that write the KV cache are in
//! [`crate::prefill`]; [`crate::forward`] runs all three. Left out: the
//! action and state projections, the time embedding, the action output
//! projection and the Euler update.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use nu_model::{Addr, Range};
use serde::{Deserialize, Serialize};
use sim_core::{Descriptor, Dtype, NuCtx};

use crate::graph::{excess_deps, with_deps, Graph, TaskIx, MAX_JOIN_DEPS};
use crate::kv::KvLayout;
use crate::label::{self, Kind};
use crate::machine::{parse_dtype, with_b_dtype, worker_list, Layout, Machine, SyncPool};

/// Program parameters (a TOML file, every key optional).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Params {
    /// Denoise steps.
    pub steps: u32,
    /// Transformer layers per step.
    pub layers: u32,
    /// Suffix tokens (rows of the residual stream).
    pub tokens: u64,
    /// Model width.
    pub d: u64,
    /// Hidden width of each expert's FFN (shared and routed).
    pub ffn: u64,
    /// Query heads (one MQA KV head).
    pub heads: u64,
    /// Elements per head.
    pub head_dim: u64,
    /// Prefix KV cache entries (keys) per layer.
    pub prefix: u64,
    /// Routed experts (the shared expert is extra).
    pub experts: u64,
    /// Run the shared expert (off only for isolated dispatch tests).
    pub shared_expert: bool,
    /// Invoke the shared expert before the router barrier, so its weight
    /// stream overlaps the barrier (it does not depend on the router).
    pub shared_before_barrier: bool,
    /// Index of the driver SM in configuration order.
    pub home: usize,
    /// SMs that run attention and FFN work; empty = all but home.
    pub workers: Vec<usize>,
    /// Attention SMs; empty = workers.
    pub attn_sms: Vec<usize>,
    /// How head SMs get the prefix KV and the MQA K/V projection:
    /// "per_head" (every head SM loads them from HBM) or "multicast" (one
    /// loader SM per cluster and round loads them; the others pull from it).
    pub attn_kv: String,
    /// Under multicast, rotate the loader role over a cluster's head SMs by
    /// (layer + round), so the loader's extra SMEM traffic is spread. Off:
    /// the cluster's first head SM of the round loads.
    pub rotate_loader: bool,
    /// Offset the head -> attention SM map by the layer: head h of layer l
    /// runs on attn_sms[(h + l) mod n], so the extra heads (heads > n) move.
    pub rotate_heads: bool,
    /// Offset the FFN hidden-block round-robin by the layer: block b of
    /// layer l runs on sms[(b + l) mod len], so the extra blocks move.
    pub rotate_ffn: bool,
    /// Shared-expert SMs; empty = workers.
    pub shared_sms: Vec<usize>,
    /// SMs per routed expert; empty = every expert on all workers.
    pub expert_sms: Vec<Vec<usize>>,
    /// Element type of the projection and FFN weights: "bf16", "fp8" or
    /// "mxfp4" (0.5 B per element plus one scale byte per 32). Weight blocks,
    /// staging slots and B operand bytes scale; MACs do not. Activations,
    /// the KV cache and the router weights stay bf16.
    pub weight_dtype: String,
    /// Columns (or rows) per weight block; blocks are `d x block_cols`.
    pub block_cols: u64,
    /// Keys per prefix KV block.
    pub key_block: u64,
    /// DMA chunk (Section 5.3).
    pub dma_chunk: u64,
    /// Gather buffers at home for partial sums.
    pub gather_bufs: usize,
    /// SMEM per SM kept for resident weight blocks (no DMA per use).
    pub resident_bytes_per_sm: u64,
    /// Weight classes filled into the resident region, in order:
    /// "shared", "routed", "attn".
    pub resident_order: Vec<String>,
    /// Weight placement in the cluster memory tier (a `cmem` block at each
    /// SM's cluster; nothing happens without one). Blocks stream from the
    /// cmem of the fetching SM's cluster instead of HBM. "prob": fill in
    /// descending activation probability (attention and shared weights 1,
    /// routed expert e of layer l `expert_prob[l][e]`); ties keep
    /// `resident_order`, layer-major. "reverse": ascending (a control).
    /// "none": off. Placement is free (no step-0 preload charge).
    pub cmem_residency: String,
    /// cmem bytes per cluster for weights; 0 = the whole block.
    pub cmem_bytes: u64,
    /// Bytes kept free at the start of each cluster's cmem (in the order of
    /// the configuration's cmem clusters), e.g. for prefill activations.
    pub cmem_reserve: Vec<u64>,
    /// Per layer, per routed expert: probability that the expert gets at
    /// least one token in a step. Missing entries count as 1. The CLI fills
    /// it from the trace.
    pub expert_prob: Vec<Vec<f64>>,
    /// Atlas tile (m, n) of every matmul.
    pub tile: [u64; 2],
    /// Muon lane-ops per element.
    pub rms_ops: u64,
    /// Muon lane-ops per hidden element for GeGLU.
    pub geglu_ops: u64,
    /// Muon lane-ops per attention score for the softmax.
    pub softmax_ops: u64,
    /// Muon lane-ops per element to add a partial sum into the residual.
    pub add_ops: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            steps: 10,
            layers: 18,
            tokens: 51,
            d: 1024,
            ffn: 4096,
            heads: 8,
            head_dim: 256,
            prefix: 816,
            experts: 4,
            shared_expert: true,
            shared_before_barrier: true,
            home: 0,
            workers: Vec::new(),
            attn_sms: Vec::new(),
            attn_kv: "multicast".into(),
            rotate_loader: true,
            rotate_heads: false,
            rotate_ffn: false,
            shared_sms: Vec::new(),
            expert_sms: Vec::new(),
            weight_dtype: "bf16".into(),
            block_cols: 128,
            key_block: 128,
            dma_chunk: 64 << 10,
            gather_bufs: 4,
            resident_bytes_per_sm: 0,
            resident_order: vec!["shared".into(), "routed".into(), "attn".into()],
            cmem_residency: "prob".into(),
            cmem_bytes: 0,
            cmem_reserve: Vec::new(),
            expert_prob: Vec::new(),
            tile: [128, 128],
            rms_ops: 4,
            geglu_ops: 8,
            softmax_ops: 6,
            add_ops: 1,
        }
    }
}

/// What the program measured, for the CLI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    /// Cycle at which each denoise step's output was complete.
    pub step_end: Vec<u64>,
    /// Bytes streamed from HBM (weights and KV).
    pub hbm_bytes: u64,
    /// Of those, bytes loaded again by more than one attention SM (MQA K/V
    /// projection weights and the prefix KV).
    pub mqa_dup_bytes: u64,
    /// Weight blocks served from the resident region.
    pub resident_uses: u64,
    /// Bytes streamed from cmem (weights placed there).
    pub cmem_bytes: u64,
    /// Weight bytes placed in cmem, per class, summed over clusters.
    pub cmem_placed: std::collections::BTreeMap<String, u64>,
    /// Weight bytes of the model, per class (one copy).
    pub weight_bytes: std::collections::BTreeMap<String, u64>,
    /// Tasks invoked, all windows.
    pub tasks: u64,
    /// Graph windows run.
    pub windows: u64,
    /// Joins (fan-in signals), all windows.
    pub joins: u64,
    /// Deps on producers of earlier windows (satisfied by the barrier).
    pub dropped_deps: u64,
    /// Tokens per expert, per step and layer.
    pub tokens_per_expert: Vec<Vec<Vec<u32>>>,
    /// The parameters used.
    pub params: Option<Params>,
}

#[derive(Debug, Clone, Copy)]
struct Block {
    /// HBM address (the block's identity).
    src: u64,
    /// Address the DMA reads: `src`, or the block's cmem copy.
    from: u64,
    cmem: bool,
    bytes: u64,
    resident: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
enum Src {
    Slot(usize),
    Resident(u64),
}

/// The weight stream of one SM: two staging slots.
struct Stream {
    slots: [Range; 2],
    next: usize,
    /// Per slot: the outputs the next DMA into it must wait for (its last
    /// reader; under KV multicast also every puller's signal).
    last_reader: [Vec<Range>; 2],
    queue: VecDeque<Block>,
    ready: VecDeque<Src>,
    in_flight: usize,
}

struct SmBufs {
    xc: Range,
    p: Range,
    q: Vec<Range>,
    k: Vec<Range>,
    v: Vec<Range>,
    s: Vec<Range>,
    stats: Range,
    o: Range,
    g: [Range; 2],
    u: [Range; 2],
    h: [Range; 2],
    resident: Layout,
}

struct HomeBufs {
    x: Range,
    xn: Range,
    xn2: Range,
    gbuf: Vec<Range>,
    rout: Range,
    wr: Range,
}

/// HBM addresses of the weights and the KV cache.
struct Hbm {
    base: u64,
    layer_stride: u64,
    bb: u64,
    hpb: u64,
    nb: u64,
    kv: KvLayout,
}

impl Hbm {
    fn block(&self, layer: u32, index: u64) -> u64 {
        self.base + layer as u64 * self.layer_stride + index * self.bb
    }
    fn wq(&self, l: u32, head: u64, j: u64) -> u64 {
        self.block(l, head * self.hpb + j)
    }
    fn wk(&self, l: u32, heads: u64, j: u64) -> u64 {
        self.block(l, heads * self.hpb + j)
    }
    fn wv(&self, l: u32, heads: u64, j: u64) -> u64 {
        self.block(l, (heads + 1) * self.hpb + j)
    }
    fn wo(&self, l: u32, heads: u64, head: u64, j: u64) -> u64 {
        self.block(l, (heads + 2 + head) * self.hpb + j)
    }
    /// Expert `x` (0..experts routed, `experts` = shared), matrix 0 gate,
    /// 1 up, 2 down, hidden block `b`.
    fn ffn(&self, l: u32, heads: u64, x: u64, mat: u64, b: u64) -> u64 {
        let attn = (2 * heads + 2) * self.hpb;
        self.block(l, attn + (x * 3 + mat) * self.nb + b)
    }
    fn k_prefix(&self, l: u32) -> u64 {
        self.kv.k(l).base
    }
    fn v_prefix(&self, l: u32) -> u64 {
        self.kv.v(l).base
    }
}

/// The action-expert driver program on the home SM.
pub struct Driver {
    m: Machine,
    p: Params,
    g: Graph,
    pools: Vec<SyncPool>,
    streams: Vec<Stream>,
    bufs: Vec<Option<SmBufs>>,
    home: HomeBufs,
    hbm: Hbm,
    attn_sms: Vec<usize>,
    shared_sms: Vec<usize>,
    expert_sms: Vec<Vec<usize>>,
    /// Per SM: the signal that the previous phase's work is done.
    prev_end: Vec<Option<Range>>,
    /// Per SM: the signal that home has pulled the previous partial sum.
    prev_pull: Vec<Option<Range>>,
    /// Pulls in this window (selects the gather buffer).
    pulls: usize,
    /// Signals carried across the last barrier (the shared expert still
    /// running): per SM, and at home.
    carried: Vec<(usize, Range)>,
    /// Gate for home tasks that follow carried work (gather buffers, x).
    home_gate: Option<Range>,
    /// Element type of the weight blocks.
    w_dtype: Dtype,
    /// Weight blocks in cmem: (cluster, hbm address) -> cmem address.
    cmem: std::collections::HashMap<(nu_model::NeutrinoId, u64), u64>,
    /// Resident weight blocks: (sm, hbm address) -> SMEM address.
    resident: std::collections::HashMap<(usize, u64), u64>,
    /// What the run measured.
    pub report: Rc<RefCell<Report>>,
}

impl Driver {
    /// Builds the driver for `p` on `m`: SM lists, SMEM buffers, HBM map,
    /// resident and cmem weight placement. Errors on a bad parameter or a
    /// layout that does not fit.
    pub fn new(m: Machine, p: Params) -> Result<Self, String> {
        let n = m.sms.len();
        if p.attn_kv != "per_head" && p.attn_kv != "multicast" {
            return Err(format!(
                "unknown attn_kv {:?} (per_head, multicast)",
                p.attn_kv
            ));
        }
        let workers = worker_list(n, p.home, &p.workers)?;
        let or = |v: &Vec<usize>| {
            if v.is_empty() {
                workers.clone()
            } else {
                v.clone()
            }
        };
        let attn_sms = or(&p.attn_sms);
        let shared_sms = or(&p.shared_sms);
        let expert_sms = if p.expert_sms.is_empty() {
            vec![workers.clone(); p.experts as usize]
        } else {
            p.expert_sms.clone()
        };
        if expert_sms.len() != p.experts as usize {
            return Err("expert_sms needs one list per routed expert".into());
        }
        if !p.head_dim.is_multiple_of(p.block_cols)
            || !p.ffn.is_multiple_of(p.block_cols)
            || !p.d.is_multiple_of(p.block_cols)
        {
            return Err("head_dim, ffn and d must be multiples of block_cols".into());
        }
        let g = m.granule;
        let t = p.tokens;
        let w_dtype = parse_dtype(&p.weight_dtype)?;
        let bb = m.round(w_dtype.bytes_for(p.d * p.block_cols));
        let unit = t * p.block_cols * 2;
        let act = t * p.d * 2;
        let hpb = p.head_dim / p.block_cols;
        let nkb = p.prefix.div_ceil(p.key_block);

        // Worker layouts.
        let mut bufs: Vec<Option<SmBufs>> = (0..n).map(|_| None).collect();
        let mut streams = Vec::with_capacity(n);
        for (i, sm) in m.sms.iter().enumerate() {
            let mut lay = Layout::new(sm.smem, g);
            // The home SM streams no weights (the router weights are
            // resident); its SMEM holds the residual stream and the gather
            // buffers.
            let slots = if i == p.home {
                let none = Range::from_len(sm.smem.base, 0);
                [none, none]
            } else {
                [lay.take(bb)?, lay.take(bb)?]
            };
            streams.push(Stream {
                slots,
                next: 0,
                last_reader: [Vec::new(), Vec::new()],
                queue: VecDeque::new(),
                ready: VecDeque::new(),
                in_flight: 0,
            });
            if i == p.home {
                continue;
            }
            let units = |lay: &mut Layout, k: u64| -> Result<Vec<Range>, String> {
                (0..k).map(|_| lay.take(unit)).collect()
            };
            let xc = lay.take(act)?;
            let pp = lay.take(act)?;
            let q = units(&mut lay, hpb)?;
            let k = units(&mut lay, hpb)?;
            let v = units(&mut lay, hpb)?;
            let s = (0..=nkb)
                .map(|_| lay.take(t * p.key_block * 2))
                .collect::<Result<Vec<_>, _>>()?;
            let stats = lay.take(t * 8)?;
            let o = lay.take(t * p.head_dim * 2)?;
            let gg = [lay.take(unit)?, lay.take(unit)?];
            let uu = [lay.take(unit)?, lay.take(unit)?];
            let hh = [lay.take(unit)?, lay.take(unit)?];
            let resident = if p.resident_bytes_per_sm > 0 {
                Layout::new(lay.take(p.resident_bytes_per_sm)?, g)
            } else {
                Layout::new(Range::from_len(sm.smem.bound, 0), g)
            };
            bufs[i] = Some(SmBufs {
                xc,
                p: pp,
                q,
                k,
                v,
                s,
                stats,
                o,
                g: gg,
                u: uu,
                h: hh,
                resident,
            });
        }
        let mut lay = Layout::new(m.sms[p.home].smem, g);
        let home = HomeBufs {
            x: lay.take(act)?,
            xn: lay.take(act)?,
            xn2: lay.take(act)?,
            gbuf: (0..p.gather_bufs.max(1))
                .map(|_| lay.take(act))
                .collect::<Result<Vec<_>, _>>()?,
            rout: lay.take(t)?,
            wr: lay.take(p.d * p.experts * 2)?,
        };

        let nb = p.ffn / p.block_cols;
        let (base, layer_stride, kv) = hbm_map(&m, &p)?;
        let hbm = Hbm {
            base,
            layer_stride,
            bb,
            hpb,
            nb,
            kv,
        };
        let pools = m.sms.iter().map(|s| SyncPool::new(s.sync, g)).collect();
        let mut d = Self {
            prev_end: vec![None; n],
            prev_pull: vec![None; n],
            pulls: 0,
            carried: Vec::new(),
            home_gate: None,
            w_dtype,
            cmem: Default::default(),
            resident: Default::default(),
            report: Rc::new(RefCell::new(Report {
                params: Some(p.clone()),
                ..Report::default()
            })),
            g: Graph::new(m.sms[p.home].own, m.versions.clone()),
            m,
            pools,
            streams,
            bufs,
            home,
            hbm,
            attn_sms,
            shared_sms,
            expert_sms,
            p,
        };
        d.place_resident();
        d.place_cmem()?;
        Ok(d)
    }

    /// The prefix KV cache layout the attention reads.
    pub(crate) fn kv(&self) -> KvLayout {
        self.hbm.kv
    }

    /// The Neutrino of the home SM.
    pub fn home_nu(&self) -> nu_model::NeutrinoId {
        self.m.sms[self.p.home].nu
    }

    /// Fills each SM's resident region with its weight blocks in the
    /// configured class order; within a class, layer-major. Residency is per
    /// block address, so the blocks of every layer compete for the region.
    fn place_resident(&mut self) {
        if self.p.resident_bytes_per_sm == 0 {
            return;
        }
        let order = self.p.resident_order.clone();
        for class in order {
            for l in 0..self.p.layers {
                for (sm, blk) in self.blocks_of_class(&class, l) {
                    let Some(b) = self.bufs[sm].as_mut() else {
                        continue;
                    };
                    if self.resident.contains_key(&(sm, blk)) {
                        continue;
                    }
                    if let Ok(r) = b.resident.take(self.hbm.bb) {
                        self.resident.insert((sm, blk), r.base);
                    }
                }
            }
        }
    }

    /// Places weight blocks in each cluster's cmem (see `cmem_residency`).
    /// A block goes to the cmem of every cluster with an SM that fetches it.
    fn place_cmem(&mut self) -> Result<(), String> {
        let classes = ["shared", "routed", "attn"];
        {
            let mut r = self.report.borrow_mut();
            for class in classes {
                let n: usize = (0..self.p.layers)
                    .map(|l| {
                        let mut v: Vec<u64> = self
                            .blocks_of_class(class, l)
                            .into_iter()
                            .map(|(_, b)| b)
                            .collect();
                        v.sort();
                        v.dedup();
                        v.len()
                    })
                    .sum();
                r.weight_bytes.insert(class.into(), n as u64 * self.hbm.bb);
            }
        }
        let desc = match self.p.cmem_residency.as_str() {
            "none" => return Ok(()),
            "prob" => true,
            "reverse" => false,
            s => {
                return Err(format!(
                    "unknown cmem_residency {s:?} (prob, reverse, none)"
                ))
            }
        };
        // Candidates: (probability, tie rank, cluster, class, block).
        let mut order: Vec<String> = self.p.resident_order.clone();
        for c in classes {
            if !order.iter().any(|o| o == c) {
                order.push(c.into());
            }
        }
        let mut cand = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (rank, class) in order.iter().enumerate() {
            for l in 0..self.p.layers {
                for (i, (sm, blk)) in self.blocks_of_class(class, l).into_iter().enumerate() {
                    let (Some(cl), Some(_)) = (self.m.sms[sm].cluster, self.m.sms[sm].cmem) else {
                        continue;
                    };
                    if !seen.insert((cl, blk)) {
                        continue;
                    }
                    let prob = if class == "routed" {
                        // blocks_of_class lists routed experts in order, 3
                        // blocks per hidden block.
                        let e = i as u64 / (3 * self.hbm.nb);
                        self.p
                            .expert_prob
                            .get(l as usize)
                            .and_then(|v| v.get(e as usize))
                            .copied()
                            .unwrap_or(1.0)
                    } else {
                        1.0
                    };
                    cand.push((prob, rank, cl, class.clone(), blk));
                }
            }
        }
        // Stable: ties keep the class order, layer-major.
        cand.sort_by(|a, b| {
            let o = a.0.total_cmp(&b.0);
            (if desc { o.reverse() } else { o }).then(a.1.cmp(&b.1))
        });
        let mut lays: std::collections::HashMap<nu_model::NeutrinoId, Layout> = Default::default();
        for s in &self.m.sms {
            if let (Some(cl), Some(cm)) = (s.cluster, s.cmem) {
                let i = self.m.cmem_nodes.iter().position(|c| c.nu == cl);
                let skip = i
                    .and_then(|i| self.p.cmem_reserve.get(i))
                    .copied()
                    .unwrap_or(0)
                    .min(cm.len());
                let free = cm.len() - skip;
                let len = if self.p.cmem_bytes == 0 {
                    free
                } else {
                    self.p.cmem_bytes.min(free)
                };
                let g = self.m.granule;
                lays.entry(cl)
                    .or_insert_with(|| Layout::new(Range::from_len(cm.base + skip, len), g));
            }
        }
        let mut r = self.report.borrow_mut();
        for (_, _, cl, class, blk) in cand {
            let lay = lays.get_mut(&cl).expect("cluster layout");
            if let Ok(a) = lay.take(self.hbm.bb) {
                self.cmem.insert((cl, blk), a.base);
                *r.cmem_placed.entry(class).or_default() += self.hbm.bb;
            }
        }
        Ok(())
    }

    /// The attention SM of `head` in layer `l`.
    fn head_sm(&self, l: u32, head: u64) -> usize {
        let off = if self.p.rotate_heads { l as u64 } else { 0 };
        self.attn_sms[((head + off) % self.attn_sms.len() as u64) as usize]
    }

    /// Index into an expert's SM list of hidden block `b` in layer `l`.
    fn ffn_slot(&self, l: u32, b: u64, len: usize) -> usize {
        let off = if self.p.rotate_ffn { l as u64 } else { 0 };
        ((b + off) % len as u64) as usize
    }

    fn blocks_of_class(&self, class: &str, l: u32) -> Vec<(usize, u64)> {
        let p = &self.p;
        let h = &self.hbm;
        let mut v = Vec::new();
        let ffn = |x: u64, sms: &[usize], v: &mut Vec<(usize, u64)>| {
            for b in 0..h.nb {
                let sm = sms[self.ffn_slot(l, b, sms.len())];
                for mat in 0..3 {
                    v.push((sm, h.ffn(l, p.heads, x, mat, b)));
                }
            }
        };
        match class {
            "shared" => ffn(p.experts, &self.shared_sms, &mut v),
            "routed" => {
                for e in 0..p.experts {
                    ffn(e, &self.expert_sms[e as usize], &mut v);
                }
            }
            "attn" => {
                for head in 0..p.heads {
                    let sm = self.head_sm(l, head);
                    for j in 0..h.hpb {
                        v.push((sm, h.wq(l, head, j)));
                        v.push((sm, h.wk(l, p.heads, j)));
                        v.push((sm, h.wv(l, p.heads, j)));
                        v.push((sm, h.wo(l, p.heads, head, j)));
                    }
                }
            }
            _ => {}
        }
        v
    }

    // ---- graph helpers ----

    /// Adds a task that `sm` owns (the owner follows from `dst`).
    fn add(&mut self, sm: usize, mut desc: Descriptor) -> TaskIx {
        // Deps beyond one message go through a join at the same SM.
        let rest = excess_deps(&mut desc);
        if !rest.is_empty() {
            let j = self.signal(sm, &rest, desc.label.unwrap_or(0));
            desc.deps.push(nu_model::Dep::Unversioned(j.base));
        }
        self.g.add(desc)
    }

    /// A signal for later consumers over `deps`: a join at `sm` for
    /// several deps (fan-in), or a pin of the one producer
    /// ([`Graph::pin`]): versions make a remote dependency on data
    /// unambiguous, so a single dep needs no join.
    fn signal(&mut self, sm: usize, deps: &[Option<Range>], label: u64) -> Range {
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
                    self.signal(sm, &c, label)
                })
                .collect();
        }
        let deps: Vec<Option<Range>> = present.iter().map(|r| Some(*r)).collect();
        let r = self.pools[sm].alloc();
        let d = with_deps(Descriptor::join(r, vec![]), &deps, label);
        let ix = self.add(sm, d);
        self.g.pin_task(ix)
    }

    // ---- weight stream ----

    fn fetch(&mut self, sm: usize, blocks: &[(u64, u64)], l: u32, step: u32) {
        for &(src, bytes) in blocks {
            let resident = self.resident.get(&(sm, src)).copied();
            let c = self.m.sms[sm]
                .cluster
                .and_then(|cl| self.cmem.get(&(cl, src)).copied());
            self.streams[sm].queue.push_back(Block {
                src,
                from: c.unwrap_or(src),
                cmem: c.is_some(),
                bytes,
                resident,
            });
        }
        self.pump(sm, l, step);
    }

    fn pump(&mut self, sm: usize, l: u32, step: u32) {
        loop {
            let st = &mut self.streams[sm];
            let Some(&b) = st.queue.front() else {
                return;
            };
            if let Some(addr) = b.resident {
                st.queue.pop_front();
                st.ready.push_back(Src::Resident(addr));
                self.report.borrow_mut().resident_uses += 1;
                continue;
            }
            if st.in_flight >= 2 {
                return;
            }
            st.queue.pop_front();
            let slot = st.next;
            st.next ^= 1;
            st.in_flight += 1;
            st.ready.push_back(Src::Slot(slot));
            let dst = st.slots[slot];
            let war: Vec<Option<Range>> = st.last_reader[slot].iter().map(|r| Some(*r)).collect();
            let desc = self.m.dma(sm, b.from, dst, b.bytes, self.p.dma_chunk);
            let desc = with_deps(
                desc,
                &war,
                label::label(Kind::DmaW, step, l, sm as u32, (b.src >> 12) as u32),
            );
            self.add(sm, desc);
            let mut r = self.report.borrow_mut();
            if b.cmem {
                r.cmem_bytes += b.bytes;
            } else {
                r.hbm_bytes += b.bytes;
            }
        }
    }

    /// The next fetched block: its address and dep range (a slot) if
    /// streamed.
    fn take(&mut self, sm: usize) -> (Src, u64, Option<Range>) {
        match self.streams[sm].ready.pop_front().expect("fetched block") {
            Src::Slot(i) => {
                let r = self.streams[sm].slots[i];
                (Src::Slot(i), r.base, Some(r))
            }
            Src::Resident(a) => (Src::Resident(a), a, None),
        }
    }

    fn read_done(&mut self, sm: usize, src: Src, reader_out: Range, l: u32, step: u32) {
        if let Src::Slot(i) = src {
            let st = &mut self.streams[sm];
            st.last_reader[i] = vec![reader_out];
            st.in_flight -= 1;
        }
        self.pump(sm, l, step);
    }

    fn bufs(&self, sm: usize) -> &SmBufs {
        self.bufs[sm].as_ref().expect("worker SM")
    }

    // ---- home ----

    /// Pulls SM `sm`'s partial sum of `rows` rows to home and adds it into x.
    fn pull_and_add(&mut self, sm: usize, rows: u64, sig: Range, l: u32, step: u32) {
        let (home, d, dma_chunk, add_ops) =
            (self.p.home, self.p.d, self.p.dma_chunk, self.p.add_ops);
        let k = self.pulls % self.home.gbuf.len();
        let reuse = self.pulls >= self.home.gbuf.len();
        self.pulls += 1;
        let gb = self.home.gbuf[k];
        let x = self.home.x;
        let src = self.bufs(sm).p.base;
        let bytes = rows * d * 2;
        let dma = self.m.dma(home, src, gb, bytes, dma_chunk);
        let dma = with_deps(
            dma,
            &[Some(sig), reuse.then_some(x), self.home_gate],
            label::label(Kind::DmaAct, step, l, sm as u32, 0),
        );
        self.add(home, dma);
        self.prev_pull[sm] = Some(self.signal(
            home,
            &[Some(gb)],
            label::label(Kind::Join, step, l, sm as u32, 1),
        ));
        let add = self.m.vector(home, x, rows * d * add_ops, 2 * bytes, bytes);
        let add = with_deps(
            add,
            &[Some(x), Some(gb), self.home_gate],
            label::label(Kind::Vec, step, l, sm as u32, 0),
        );
        self.add(home, add);
    }

    fn rmsnorm(&mut self, dst: Range, l: u32, step: u32, unit: u32) {
        let p = &self.p;
        let bytes = p.tokens * p.d * 2;
        let desc = self
            .m
            .vector(p.home, dst, p.tokens * p.d * p.rms_ops, bytes, bytes);
        let desc = with_deps(
            desc,
            &[Some(self.home.x)],
            label::label(Kind::Vec, step, l, unit, 1),
        );
        self.add(self.p.home, desc);
    }

    // ---- attention ----

    fn attention(&mut self, l: u32, step: u32) {
        let p = self.p.clone();
        let h = &self.hbm;
        let (hpb, nkb) = (h.hpb, p.prefix.div_ceil(p.key_block));
        let t = p.tokens;
        self.rmsnorm(self.home.xn, l, step, 0);
        let sig_xn = self.signal(
            p.home,
            &[Some(self.home.xn)],
            label::label(Kind::Join, step, l, p.home as u32, 0),
        );
        let mut first = vec![true; self.m.sms.len()];
        let mut ends: Vec<(usize, Range)> = Vec::new();
        for head in 0..p.heads {
            let sm = self.head_sm(l, head);
            let lab = |kind, idx| label::label(kind, step, l, head as u32, idx);
            let pe = self.prev_end[sm];
            let (xc, pbuf, o, stats) = {
                let b = self.bufs(sm);
                (b.xc, b.p, b.o, b.stats)
            };
            if first[sm] {
                // The normed input, once per SM and layer.
                let dma = self
                    .m
                    .dma(sm, self.home.xn.base, xc, t * p.d * 2, p.dma_chunk);
                let dma = with_deps(dma, &[Some(sig_xn), pe], lab(Kind::DmaAct, 0));
                self.add(sm, dma);
            }
            // Q, K, V projections: one reader per weight block.
            let mut blocks = Vec::new();
            for j in 0..hpb {
                blocks.push((self.hbm.wq(l, head, j), self.hbm.bb));
            }
            for j in 0..hpb {
                blocks.push((self.hbm.wk(l, p.heads, j), self.hbm.bb));
            }
            for j in 0..hpb {
                blocks.push((self.hbm.wv(l, p.heads, j), self.hbm.bb));
            }
            if head > 0 {
                // MQA: every head after the first reloads the K/V projection.
                self.report.borrow_mut().mqa_dup_bytes += 2 * hpb * self.hbm.bb;
            }
            self.fetch(sm, &blocks, l, step);
            let outs: Vec<Range> = {
                let b = self.bufs(sm);
                b.q.iter().chain(&b.k).chain(&b.v).copied().collect()
            };
            for (j, &out) in outs.iter().enumerate() {
                let (src, addr, slot) = self.take(sm);
                let mm = self
                    .m
                    .matmul(sm, xc.base, addr, out, t, p.block_cols, p.d, false, p.tile);
                let mm = with_b_dtype(self.w_dtype, mm);
                let mm = with_deps(mm, &[Some(xc), slot, pe], lab(Kind::Qkv, j as u32));
                let ix = self.add(sm, mm);
                self.read_done(sm, src, self.g.dst(ix), l, step);
            }
            let jq = self.join_local(sm, &outs[0..hpb as usize], lab(Kind::Join, 10));
            let jk = self.join_local(
                sm,
                &outs[hpb as usize..2 * hpb as usize],
                lab(Kind::Join, 11),
            );
            let jv = self.join_local(sm, &outs[2 * hpb as usize..], lab(Kind::Join, 12));
            let (q0, k0, v0) = (
                outs[0].base,
                outs[hpb as usize].base,
                outs[2 * hpb as usize].base,
            );
            // Scores: QK^T per prefix key block, then the suffix.
            let s = self.bufs(sm).s.clone();
            let kb_bytes = |j: u64| p.key_block.min(p.prefix - j * p.key_block) * p.head_dim * 2;
            let kblocks: Vec<(u64, u64)> = (0..nkb)
                .map(|j| {
                    (
                        self.hbm.k_prefix(l) + j * p.key_block * p.head_dim * 2,
                        kb_bytes(j),
                    )
                })
                .collect();
            if head > 0 {
                self.report.borrow_mut().mqa_dup_bytes +=
                    kblocks.iter().map(|b| b.1).sum::<u64>() * 2;
            }
            self.fetch(sm, &kblocks, l, step);
            for (j, &sj) in s.iter().take(nkb as usize).enumerate() {
                let keys = p.key_block.min(p.prefix - j as u64 * p.key_block);
                let (src, addr, slot) = self.take(sm);
                let mm = self
                    .m
                    .matmul(sm, q0, addr, sj, t, keys, p.head_dim, false, p.tile);
                let mm = with_deps(mm, &[Some(jq), slot, pe], lab(Kind::Attn, j as u32));
                let ix = self.add(sm, mm);
                self.read_done(sm, src, self.g.dst(ix), l, step);
            }
            let s_last = s[nkb as usize];
            let mm = self
                .m
                .matmul(sm, q0, k0, s_last, t, t, p.head_dim, false, p.tile);
            let mm = with_deps(mm, &[Some(jq), Some(jk), pe], lab(Kind::Attn, 100));
            self.add(sm, mm);
            // Softmax: row statistics over all slices, then normalize each
            // slice in place.
            let keys_total = p.prefix + t;
            let js = self.join_local(sm, &s, lab(Kind::Join, 13));
            let st = self
                .m
                .vector(sm, stats, t * keys_total * 2, t * keys_total * 2, t * 8);
            let st = with_deps(st, &[Some(js), pe], lab(Kind::Vec, 10));
            self.add(sm, st);
            for (j, &sj) in s.iter().enumerate() {
                let keys = if j as u64 == nkb {
                    t
                } else {
                    p.key_block.min(p.prefix - j as u64 * p.key_block)
                };
                let bytes = t * keys * 2;
                let nm = self
                    .m
                    .vector(sm, sj, t * keys * p.softmax_ops, bytes, bytes);
                let nm = with_deps(nm, &[Some(sj), Some(stats)], lab(Kind::Vec, 20 + j as u32));
                self.add(sm, nm);
            }
            // PV: accumulate over key blocks into o.
            let vblocks: Vec<(u64, u64)> = (0..nkb)
                .map(|j| {
                    (
                        self.hbm.v_prefix(l) + j * p.key_block * p.head_dim * 2,
                        kb_bytes(j),
                    )
                })
                .collect();
            self.fetch(sm, &vblocks, l, step);
            for (j, &sj) in s.iter().take(nkb as usize).enumerate() {
                let keys = p.key_block.min(p.prefix - j as u64 * p.key_block);
                let (src, addr, slot) = self.take(sm);
                let acc = j > 0;
                let mm = self
                    .m
                    .matmul(sm, sj.base, addr, o, t, p.head_dim, keys, acc, p.tile);
                let prev_o = acc.then_some(o);
                let mm = with_deps(
                    mm,
                    &[Some(sj), slot, prev_o, pe],
                    lab(Kind::Attn, 200 + j as u32),
                );
                let ix = self.add(sm, mm);
                self.read_done(sm, src, self.g.dst(ix), l, step);
            }
            let mm = self
                .m
                .matmul(sm, s_last.base, v0, o, t, p.head_dim, t, true, p.tile);
            let mm = with_deps(mm, &[Some(s_last), Some(jv), Some(o)], lab(Kind::Attn, 300));
            self.add(sm, mm);
            // Wo: this head's rows, accumulated into the partial sum.
            let wo: Vec<(u64, u64)> = (0..hpb)
                .map(|j| (self.hbm.wo(l, p.heads, head, j), self.hbm.bb))
                .collect();
            self.fetch(sm, &wo, l, step);
            for j in 0..hpb {
                let (src, addr, slot) = self.take(sm);
                let acc = !first[sm] || j > 0;
                let a = o.base + j * t * p.block_cols * 2;
                let mm = self
                    .m
                    .matmul(sm, a, addr, pbuf, t, p.d, p.block_cols, acc, p.tile);
                let mm = with_b_dtype(self.w_dtype, mm);
                let chain = if acc { Some(pbuf) } else { self.prev_pull[sm] };
                let mm = with_deps(mm, &[Some(o), slot, chain, pe], lab(Kind::Wo, j as u32));
                let ix = self.add(sm, mm);
                self.read_done(sm, src, self.g.dst(ix), l, step);
            }
            // End of this head at this SM.
            let end = self.signal(sm, &[Some(pbuf)], lab(Kind::Join, 14));
            self.prev_end[sm] = Some(end);
            first[sm] = false;
            if let Some(e) = ends.iter_mut().find(|e| e.0 == sm) {
                e.1 = end;
            } else {
                ends.push((sm, end));
            }
        }
        for (sm, end) in ends {
            self.pull_and_add(sm, t, end, l, step);
        }
    }

    /// A staging slot of `sm` for a hand-driven block (KV multicast): the
    /// generic stream must be idle.
    fn mc_slot(&mut self, sm: usize) -> (usize, Range) {
        let st = &mut self.streams[sm];
        assert!(st.in_flight == 0 && st.queue.is_empty(), "stream busy");
        let slot = st.next;
        st.next ^= 1;
        (slot, st.slots[slot])
    }

    /// One pass of multicast blocks over a cluster group: for block j the
    /// loader streams it from HBM into a slot and signals; each other member
    /// pulls it from the loader's slot into its own slot and signals back;
    /// `reader` adds the member's reader of the block and returns its
    /// output. The loader's next load into a slot waits for its reader and
    /// every puller's signal.
    fn mc_blocks(
        &mut self,
        group: &[(u64, usize)],
        blocks: &[(u64, u64)],
        l: u32,
        step: u32,
        reader: &mut dyn FnMut(&mut Self, usize, u64, usize, Range) -> Range,
    ) {
        let loader = group[0].1;
        for (j, &(src, bytes)) in blocks.iter().enumerate() {
            let (ls, lr) = self.mc_slot(loader);
            let war: Vec<Option<Range>> = self.streams[loader].last_reader[ls]
                .iter()
                .map(|r| Some(*r))
                .collect();
            let dma = self.m.dma(loader, src, lr, bytes, self.p.dma_chunk);
            let lab = label::label(Kind::DmaW, step, l, loader as u32, (src >> 12) as u32);
            self.add(loader, with_deps(dma, &war, lab));
            self.report.borrow_mut().hbm_bytes += bytes;
            let sig = self.signal(
                loader,
                &[Some(lr)],
                label::label(Kind::Join, step, l, loader as u32, 30),
            );
            let mut loader_war = Vec::new();
            for &(head, sm) in group {
                let slot_range = if sm == loader {
                    lr
                } else {
                    let (ms, mr) = self.mc_slot(sm);
                    let mut deps: Vec<Option<Range>> = vec![Some(sig)];
                    deps.extend(self.streams[sm].last_reader[ms].iter().map(|r| Some(*r)));
                    let pull = self.m.dma(sm, lr.base, mr, bytes, self.p.dma_chunk);
                    let lab = label::label(Kind::DmaAct, step, l, sm as u32, 30 + j as u32);
                    self.add(sm, with_deps(pull, &deps, lab));
                    let back = self.signal(
                        sm,
                        &[Some(mr)],
                        label::label(Kind::Join, step, l, sm as u32, 31),
                    );
                    loader_war.push(back);
                    self.streams[sm].last_reader[ms] = Vec::new();
                    mr
                };
                let out = reader(self, j, head, sm, slot_range);
                if sm == loader {
                    loader_war.push(out);
                } else {
                    let st = &mut self.streams[sm];
                    let ms = if st.slots[0] == slot_range { 0 } else { 1 };
                    st.last_reader[ms] = vec![out];
                }
            }
            self.streams[loader].last_reader[ls] = loader_war;
        }
    }

    /// Head-parallel attention with the prefix KV and the MQA K/V projection
    /// multicast within each cluster (`attn_kv = "multicast"`).
    fn attention_mc(&mut self, l: u32, step: u32) {
        let p = self.p.clone();
        let (hpb, nkb) = (self.hbm.hpb, p.prefix.div_ceil(p.key_block));
        let t = p.tokens;
        let unit = t * p.block_cols * 2;
        self.rmsnorm(self.home.xn, l, step, 0);
        let sig_xn = self.signal(
            p.home,
            &[Some(self.home.xn)],
            label::label(Kind::Join, step, l, p.home as u32, 0),
        );
        let n = self.attn_sms.len();
        let rounds = p.heads.div_ceil(n as u64);
        let mut first = vec![true; self.m.sms.len()];
        let mut ends: Vec<(usize, Range)> = Vec::new();
        let kb_bytes = |j: u64| p.key_block.min(p.prefix - j * p.key_block) * p.head_dim * 2;
        let keys_of = |j: u64| p.key_block.min(p.prefix - j * p.key_block);
        for r in 0..rounds {
            let members: Vec<(u64, usize)> = (r * n as u64..((r + 1) * n as u64).min(p.heads))
                .map(|h| (h, self.head_sm(l, h)))
                .collect();
            // Q projection per head; the normed input once per SM.
            let mut jq = std::collections::HashMap::new();
            for &(head, sm) in &members {
                let lab = |kind, idx| label::label(kind, step, l, head as u32, idx);
                let pe = self.prev_end[sm];
                let xc = self.bufs(sm).xc;
                if first[sm] {
                    let dma = self
                        .m
                        .dma(sm, self.home.xn.base, xc, t * p.d * 2, p.dma_chunk);
                    self.add(
                        sm,
                        with_deps(dma, &[Some(sig_xn), pe], lab(Kind::DmaAct, 0)),
                    );
                }
                let blocks: Vec<(u64, u64)> = (0..hpb)
                    .map(|j| (self.hbm.wq(l, head, j), self.hbm.bb))
                    .collect();
                self.fetch(sm, &blocks, l, step);
                let qs = self.bufs(sm).q.clone();
                for (j, &out) in qs.iter().enumerate() {
                    let (src, addr, slot) = self.take(sm);
                    let mm =
                        self.m
                            .matmul(sm, xc.base, addr, out, t, p.block_cols, p.d, false, p.tile);
                    let mm = with_b_dtype(self.w_dtype, mm);
                    let ix = self.add(
                        sm,
                        with_deps(mm, &[Some(xc), slot, pe], lab(Kind::Qkv, j as u32)),
                    );
                    self.read_done(sm, src, self.g.dst(ix), l, step);
                }
                jq.insert(sm, self.join_local(sm, &qs, lab(Kind::Join, 10)));
            }
            // Group the round's heads by cluster; the first is the loader.
            let mut groups: Vec<Vec<(u64, usize)>> = Vec::new();
            for &(head, sm) in &members {
                let c = self.m.sms[sm].cluster;
                match groups.iter_mut().find(|g| self.m.sms[g[0].1].cluster == c) {
                    Some(g) => g.push((head, sm)),
                    None => groups.push(vec![(head, sm)]),
                }
            }
            // The loader is the group's first member.
            if p.rotate_loader {
                for g in &mut groups {
                    let k = (l as usize + r as usize) % g.len();
                    g.rotate_left(k);
                }
            }
            for (gi, group) in groups.iter().enumerate() {
                // Loads beyond one copy per layer count as MQA duplication,
                // as in the per-head mode.
                let dup = r > 0 || gi > 0;
                let loader = group[0].1;
                // The MQA K/V projection once per cluster and round.
                let pe_l = self.prev_end[loader];
                let xl = self.bufs(loader).xc;
                let mut blocks = Vec::new();
                for j in 0..hpb {
                    blocks.push((self.hbm.wk(l, p.heads, j), self.hbm.bb));
                }
                for j in 0..hpb {
                    blocks.push((self.hbm.wv(l, p.heads, j), self.hbm.bb));
                }
                if dup {
                    self.report.borrow_mut().mqa_dup_bytes += 2 * hpb * self.hbm.bb;
                }
                self.fetch(loader, &blocks, l, step);
                let kv: Vec<Range> = {
                    let b = self.bufs(loader);
                    b.k.iter().chain(&b.v).copied().collect()
                };
                for (j, &out) in kv.iter().enumerate() {
                    let (src, addr, slot) = self.take(loader);
                    let mm = self.m.matmul(
                        loader,
                        xl.base,
                        addr,
                        out,
                        t,
                        p.block_cols,
                        p.d,
                        false,
                        p.tile,
                    );
                    let mm = with_b_dtype(self.w_dtype, mm);
                    let lab = label::label(Kind::Qkv, step, l, group[0].0 as u32, 10 + j as u32);
                    let ix = self.add(loader, with_deps(mm, &[Some(xl), slot, pe_l], lab));
                    self.read_done(loader, src, self.g.dst(ix), l, step);
                }
                let deps: Vec<Option<Range>> = kv.iter().map(|r| Some(*r)).collect();
                let sig_kv = self.signal(
                    loader,
                    &deps,
                    label::label(Kind::Join, step, l, loader as u32, 32),
                );
                let mut joins = std::collections::HashMap::new();
                for &(head, sm) in group {
                    let lab = |kind, idx| label::label(kind, step, l, head as u32, idx);
                    if sm != loader {
                        let pe = self.prev_end[sm];
                        let mine: Vec<Range> = {
                            let b = self.bufs(sm);
                            b.k.iter().chain(&b.v).copied().collect()
                        };
                        for (i, (&src, &dst)) in kv.iter().zip(&mine).enumerate() {
                            let pull = self.m.dma(sm, src.base, dst, unit, p.dma_chunk);
                            self.add(
                                sm,
                                with_deps(
                                    pull,
                                    &[Some(sig_kv), pe],
                                    lab(Kind::DmaAct, 40 + i as u32),
                                ),
                            );
                        }
                    }
                    let (k, v) = {
                        let b = self.bufs(sm);
                        (b.k.clone(), b.v.clone())
                    };
                    let jk = self.join_local(sm, &k, lab(Kind::Join, 11));
                    let jv = self.join_local(sm, &v, lab(Kind::Join, 12));
                    joins.insert(sm, (jk, jv, k[0].base, v[0].base));
                }
                if dup {
                    self.report.borrow_mut().mqa_dup_bytes +=
                        (0..nkb).map(kb_bytes).sum::<u64>() * 2;
                }
                // Scores over the multicast K blocks.
                let kblocks: Vec<(u64, u64)> = (0..nkb)
                    .map(|j| {
                        (
                            self.hbm.k_prefix(l) + j * p.key_block * p.head_dim * 2,
                            kb_bytes(j),
                        )
                    })
                    .collect();
                let jqc = jq.clone();
                let mut qk = |d: &mut Self, j: usize, head: u64, sm: usize, slot: Range| -> Range {
                    let (q0, sj, pe) = (d.bufs(sm).q[0].base, d.bufs(sm).s[j], d.prev_end[sm]);
                    let mm = d.m.matmul(
                        sm,
                        q0,
                        slot.base,
                        sj,
                        t,
                        keys_of(j as u64),
                        p.head_dim,
                        false,
                        p.tile,
                    );
                    let lab = label::label(Kind::Attn, step, l, head as u32, j as u32);
                    let ix = d.add(sm, with_deps(mm, &[Some(jqc[&sm]), Some(slot), pe], lab));
                    d.g.dst(ix)
                };
                self.mc_blocks(group, &kblocks, l, step, &mut qk);
                // Suffix scores and the softmax, per member.
                for &(head, sm) in group {
                    let lab = |kind, idx| label::label(kind, step, l, head as u32, idx);
                    let (jk, _, k0, _) = joins[&sm];
                    let (s, stats, q0, pe) = {
                        let b = self.bufs(sm);
                        (b.s.clone(), b.stats, b.q[0].base, self.prev_end[sm])
                    };
                    let s_last = s[nkb as usize];
                    let mm = self
                        .m
                        .matmul(sm, q0, k0, s_last, t, t, p.head_dim, false, p.tile);
                    self.add(
                        sm,
                        with_deps(mm, &[Some(jq[&sm]), Some(jk), pe], lab(Kind::Attn, 100)),
                    );
                    let keys_total = p.prefix + t;
                    let js = self.join_local(sm, &s, lab(Kind::Join, 13));
                    let st =
                        self.m
                            .vector(sm, stats, t * keys_total * 2, t * keys_total * 2, t * 8);
                    self.add(sm, with_deps(st, &[Some(js), pe], lab(Kind::Vec, 10)));
                    for (j, &sj) in s.iter().enumerate() {
                        let keys = if j as u64 == nkb {
                            t
                        } else {
                            keys_of(j as u64)
                        };
                        let bytes = t * keys * 2;
                        let nm = self
                            .m
                            .vector(sm, sj, t * keys * p.softmax_ops, bytes, bytes);
                        self.add(
                            sm,
                            with_deps(nm, &[Some(sj), Some(stats)], lab(Kind::Vec, 20 + j as u32)),
                        );
                    }
                }
                // PV over the multicast V blocks.
                let vblocks: Vec<(u64, u64)> = (0..nkb)
                    .map(|j| {
                        (
                            self.hbm.v_prefix(l) + j * p.key_block * p.head_dim * 2,
                            kb_bytes(j),
                        )
                    })
                    .collect();
                let mut pv = |d: &mut Self, j: usize, head: u64, sm: usize, slot: Range| -> Range {
                    let (sj, o, pe) = (d.bufs(sm).s[j], d.bufs(sm).o, d.prev_end[sm]);
                    let acc = j > 0;
                    let mm = d.m.matmul(
                        sm,
                        sj.base,
                        slot.base,
                        o,
                        t,
                        p.head_dim,
                        keys_of(j as u64),
                        acc,
                        p.tile,
                    );
                    let lab = label::label(Kind::Attn, step, l, head as u32, 200 + j as u32);
                    let ix = d.add(
                        sm,
                        with_deps(mm, &[Some(sj), Some(slot), acc.then_some(o), pe], lab),
                    );
                    d.g.dst(ix)
                };
                self.mc_blocks(group, &vblocks, l, step, &mut pv);
                // Suffix PV, Wo and the end of the head, per member.
                for &(head, sm) in group {
                    let lab = |kind, idx| label::label(kind, step, l, head as u32, idx);
                    let (_, jv, _, v0) = joins[&sm];
                    let (s_last, o, pbuf, pe) = {
                        let b = self.bufs(sm);
                        (b.s[nkb as usize], b.o, b.p, self.prev_end[sm])
                    };
                    let mm = self
                        .m
                        .matmul(sm, s_last.base, v0, o, t, p.head_dim, t, true, p.tile);
                    self.add(
                        sm,
                        with_deps(mm, &[Some(s_last), Some(jv), Some(o)], lab(Kind::Attn, 300)),
                    );
                    let wo: Vec<(u64, u64)> = (0..hpb)
                        .map(|j| (self.hbm.wo(l, p.heads, head, j), self.hbm.bb))
                        .collect();
                    self.fetch(sm, &wo, l, step);
                    for j in 0..hpb {
                        let (src, addr, slot) = self.take(sm);
                        let acc = !first[sm] || j > 0;
                        let a = o.base + j * t * p.block_cols * 2;
                        let mm =
                            self.m
                                .matmul(sm, a, addr, pbuf, t, p.d, p.block_cols, acc, p.tile);
                        let mm = with_b_dtype(self.w_dtype, mm);
                        let chain = if acc { Some(pbuf) } else { self.prev_pull[sm] };
                        let ix = self.add(
                            sm,
                            with_deps(mm, &[Some(o), slot, chain, pe], lab(Kind::Wo, j as u32)),
                        );
                        self.read_done(sm, src, self.g.dst(ix), l, step);
                    }
                    let end = self.signal(sm, &[Some(pbuf)], lab(Kind::Join, 14));
                    self.prev_end[sm] = Some(end);
                    first[sm] = false;
                    match ends.iter_mut().find(|e| e.0 == sm) {
                        Some(e) => e.1 = end,
                        None => ends.push((sm, end)),
                    }
                }
            }
        }
        for (sm, end) in ends {
            self.pull_and_add(sm, t, end, l, step);
        }
    }

    fn join_local(&mut self, sm: usize, parts: &[Range], label: u64) -> Range {
        let deps: Vec<Option<Range>> = parts.iter().map(|r| Some(*r)).collect();
        self.signal(sm, &deps, label)
    }

    // ---- FFN ----

    /// One FFN (expert `x`, `experts` = shared) over `rows` tokens on
    /// `sms`, input `src` at home (signal `sig`), pulled back and added.
    fn ffn(&mut self, x: u64, sms: &[usize], rows: u64, sig: Range, l: u32, step: u32) {
        let p = self.p.clone();
        let nb = self.hbm.nb;
        let mut ends = Vec::new();
        for (si, &sm) in sms.iter().enumerate() {
            let hidden: Vec<u64> = (0..nb)
                .filter(|&b| self.ffn_slot(l, b, sms.len()) == si)
                .collect();
            if hidden.is_empty() {
                continue;
            }
            let lab = |kind, idx| label::label(kind, step, l, x as u32, idx);
            let pe = self.prev_end[sm];
            let (xc, pbuf, gg, uu, hh) = {
                let b = self.bufs(sm);
                (b.xc, b.p, b.g, b.u, b.h)
            };
            let dma = self
                .m
                .dma(sm, self.home.xn2.base, xc, rows * p.d * 2, p.dma_chunk);
            let dma = with_deps(dma, &[Some(sig), pe], lab(Kind::DmaAct, sm as u32));
            self.add(sm, dma);
            let mut blocks = Vec::new();
            for &b in &hidden {
                for mat in 0..3 {
                    blocks.push((self.hbm.ffn(l, p.heads, x, mat, b), self.hbm.bb));
                }
            }
            self.fetch(sm, &blocks, l, step);
            for (i, &b) in hidden.iter().enumerate() {
                let (gi, ui, hi) = (gg[i % 2], uu[i % 2], hh[i % 2]);
                let idx = b as u32;
                // Gate and up: column blocks. The previous GeGLU into this
                // H slot read these G and U slots.
                let war = (i >= 2).then_some(hi);
                for (out, kind) in [(gi, Kind::FfnGate), (ui, Kind::FfnUp)] {
                    let (src, addr, slot) = self.take(sm);
                    let mm = self.m.matmul(
                        sm,
                        xc.base,
                        addr,
                        out,
                        rows,
                        p.block_cols,
                        p.d,
                        false,
                        p.tile,
                    );
                    let mm = with_b_dtype(self.w_dtype, mm);
                    let mm = with_deps(mm, &[Some(xc), slot, war, pe], lab(kind, idx));
                    let ix = self.add(sm, mm);
                    self.read_done(sm, src, self.g.dst(ix), l, step);
                }
                // GeGLU: the previous down matmul read this H slot.
                let bytes = rows * p.block_cols * 2;
                let prev_down = (i >= 2).then_some(pbuf);
                let ge = self
                    .m
                    .vector(sm, hi, rows * p.block_cols * p.geglu_ops, 2 * bytes, bytes);
                let ge = with_deps(
                    ge,
                    &[Some(gi), Some(ui), prev_down, pe],
                    lab(Kind::Vec, idx),
                );
                self.add(sm, ge);
                // Down: row block, accumulated into the partial sum.
                let (src, addr, slot) = self.take(sm);
                let acc = i > 0;
                let mm = self.m.matmul(
                    sm,
                    hi.base,
                    addr,
                    pbuf,
                    rows,
                    p.d,
                    p.block_cols,
                    acc,
                    p.tile,
                );
                let mm = with_b_dtype(self.w_dtype, mm);
                let chain = if acc { Some(pbuf) } else { self.prev_pull[sm] };
                let mm = with_deps(mm, &[Some(hi), slot, chain, pe], lab(Kind::FfnDown, idx));
                let ix = self.add(sm, mm);
                self.read_done(sm, src, self.g.dst(ix), l, step);
            }
            let end = self.signal(sm, &[Some(pbuf)], lab(Kind::Join, sm as u32));
            self.prev_end[sm] = Some(end);
            ends.push((sm, end));
        }
        for (sm, end) in ends {
            self.pull_and_add(sm, rows, end, l, step);
        }
    }

    fn moe(&mut self, l: u32, step: u32, ids: &[u8]) {
        let p = self.p.clone();
        let mut counts = vec![0u32; p.experts as usize];
        for &e in ids.iter().take(p.tokens as usize) {
            counts[(e as usize).min(p.experts as usize - 1)] += 1;
        }
        {
            let mut r = self.report.borrow_mut();
            while r.tokens_per_expert.len() <= step as usize {
                r.tokens_per_expert.push(Vec::new());
            }
            r.tokens_per_expert[step as usize].push(counts.clone());
        }
        let sig = self.signal(
            p.home,
            &[Some(self.home.xn2)],
            label::label(Kind::Join, step, l, p.home as u32, 2),
        );
        self.take_carry(l, step);
        if p.shared_expert && !p.shared_before_barrier {
            let shared = self.shared_sms.clone();
            self.ffn(p.experts, &shared, p.tokens, sig, l, step);
        }
        for e in 0..p.experts {
            let n = counts[e as usize] as u64;
            if n == 0 {
                continue;
            }
            let sms = self.expert_sms[e as usize].clone();
            self.ffn(e, &sms, n, sig, l, step);
        }
    }

    /// The shared expert before the router barrier, with carry signals: one
    /// per SM over its last shared-expert work (phase end, stream readers,
    /// the home pull of its partial sum) and one at home over x. Each is
    /// carried with one consumer: the gate of the next window.
    fn shared_before(&mut self, l: u32, step: u32) {
        let p = self.p.clone();
        let sig = self.signal(
            p.home,
            &[Some(self.home.xn2)],
            label::label(Kind::Join, step, l, p.home as u32, 3),
        );
        let shared = self.shared_sms.clone();
        self.ffn(p.experts, &shared, p.tokens, sig, l, step);
        let mut carried = Vec::new();
        for &sm in &shared {
            let st = &self.streams[sm];
            let mut deps = vec![self.prev_end[sm], self.prev_pull[sm]];
            deps.extend(st.last_reader.iter().flatten().map(|r| Some(*r)));
            let r = self.pools[sm].alloc();
            let d = with_deps(
                Descriptor::join(r, vec![]),
                &deps,
                label::label(Kind::Join, step, l, sm as u32, 20),
            );
            let ix = self.add(sm, d);
            self.g.carry(ix);
            carried.push((sm, r));
        }
        let r = self.pools[p.home].alloc();
        let d = with_deps(
            Descriptor::join(r, vec![]),
            &[Some(self.home.x)],
            label::label(Kind::Join, step, l, p.home as u32, 20),
        );
        let ix = self.add(p.home, d);
        self.g.carry(ix);
        carried.push((p.home, r));
        self.carried = carried;
    }

    /// Consumes the carry of the previous window: a gate per SM that its
    /// next work depends on, and a home gate over all of them.
    fn take_carry(&mut self, l: u32, step: u32) {
        let home = self.p.home;
        let carried = std::mem::take(&mut self.carried);
        let mut home_deps = Vec::new();
        for (sm, k) in carried {
            if sm == home {
                home_deps.push(Some(k));
                continue;
            }
            let g = self.signal(
                sm,
                &[Some(k)],
                label::label(Kind::Join, step, l, sm as u32, 21),
            );
            self.prev_end[sm] = Some(g);
            self.streams[sm].last_reader = [vec![g], vec![g]];
            home_deps.push(Some(g));
        }
        if !home_deps.is_empty() {
            let g = self.signal(
                home,
                &home_deps,
                label::label(Kind::Join, step, l, home as u32, 21),
            );
            self.home_gate = Some(g);
        }
    }

    /// Router of layer `l`: rmsnorm, router matmul (labeled; its output is
    /// the trace record), the shared expert if it goes before the barrier,
    /// then the barrier and the load.
    async fn route(&mut self, ctx: &NuCtx, l: u32, step: u32) -> Vec<u8> {
        let p = self.p.clone();
        self.rmsnorm(self.home.xn2, l, step, 1);
        let rout = self.home.rout;
        let mm = self.m.matmul(
            p.home,
            self.home.xn2.base,
            self.home.wr.base,
            rout,
            p.tokens,
            p.experts,
            p.d,
            false,
            p.tile,
        );
        let mm = with_deps(mm, &[Some(self.home.xn2)], label::router(step, l));
        self.add(p.home, mm);
        if p.shared_expert && p.shared_before_barrier {
            self.shared_before(l, step);
        }
        self.flush(ctx).await;
        if let Some(k) = self.g.key(&rout) {
            ctx.wait_key(k).await;
        }
        ctx.load(Addr(rout.base), p.tokens).await
    }

    async fn flush(&mut self, ctx: &NuCtx) {
        self.g.flush(ctx).await;
        let mut r = self.report.borrow_mut();
        r.windows += 1;
        r.tasks = self.g.invoked;
        r.joins = self.g.joins;
        r.dropped_deps = self.g.dropped;
        drop(r);
        // Barrier: everything invoked so far is done or waited for.
        self.home_gate = None;
        for st in &mut self.streams {
            st.last_reader = [Vec::new(), Vec::new()];
        }
        self.prev_end.iter_mut().for_each(|e| *e = None);
        self.prev_pull.iter_mut().for_each(|e| *e = None);
        self.pulls = 0;
    }

    /// Where `run_linear` writes each output block: (block, SMEM range).
    /// At most two blocks per worker.
    pub fn linear_outputs(&self, n_out: u64) -> Vec<(u64, Range)> {
        let w = &self.shared_sms;
        let mut per = vec![0usize; self.m.sms.len()];
        (0..n_out / self.p.block_cols)
            .map(|b| {
                let sm = w[b as usize % w.len()];
                let i = per[sm];
                per[sm] += 1;
                assert!(i < 2, "at most two linear blocks per worker");
                (b, self.bufs(sm).g[i])
            })
            .collect()
    }

    /// HBM address of block `b` of the `run_linear` weight (d x block_cols,
    /// row-major, blocks contiguous).
    pub fn linear_block(&self, b: u64) -> u64 {
        self.hbm.ffn(0, self.p.heads, self.p.experts, 0, b)
    }

    /// The residual `x` in the home SM's SMEM (`tokens x d`, bf16).
    pub fn home_x(&self) -> Range {
        self.home.x
    }

    /// A test building block: one column-parallel linear layer
    /// `y = x W` of the home residual `x` (tokens x d) with W of `n_out`
    /// columns, blocks round-robin over `shared_sms`. Every block: a DMA of
    /// its weight into a staging slot and one matmul into its output slice.
    pub async fn run_linear(mut self, ctx: NuCtx, n_out: u64) {
        let p = self.p.clone();
        let t = p.tokens;
        let outs = self.linear_outputs(n_out);
        let sig_x = self.signal(
            p.home,
            &[Some(self.home.x)],
            label::label(Kind::Join, 0, 0, 0, 0),
        );
        let mut ends = Vec::new();
        for &sm in &self.shared_sms.clone() {
            let mine: Vec<(u64, Range)> = outs
                .iter()
                .filter(|(b, _)| self.shared_sms[*b as usize % self.shared_sms.len()] == sm)
                .copied()
                .collect();
            if mine.is_empty() {
                continue;
            }
            let xc = self.bufs(sm).xc;
            let dma = self
                .m
                .dma(sm, self.home.x.base, xc, t * p.d * 2, p.dma_chunk);
            let dma = with_deps(
                dma,
                &[Some(sig_x)],
                label::label(Kind::DmaAct, 0, 0, sm as u32, 0),
            );
            self.add(sm, dma);
            let blocks: Vec<(u64, u64)> = mine
                .iter()
                .map(|(b, _)| (self.linear_block(*b), self.hbm.bb))
                .collect();
            self.fetch(sm, &blocks, 0, 0);
            let mut deps = Vec::new();
            for &(b, out) in &mine {
                let (src, addr, slot) = self.take(sm);
                let mm = self
                    .m
                    .matmul(sm, xc.base, addr, out, t, p.block_cols, p.d, false, p.tile);
                let mm = with_deps(
                    mm,
                    &[Some(xc), slot],
                    label::label(Kind::FfnGate, 0, 0, 0, b as u32),
                );
                let ix = self.add(sm, mm);
                self.read_done(sm, src, self.g.dst(ix), 0, 0);
                deps.push(Some(out));
            }
            ends.push(self.signal(sm, &deps, label::label(Kind::Join, 0, 0, sm as u32, 9)));
        }
        self.flush(&ctx).await;
        for e in ends {
            if let Some(k) = self.g.key(&e) {
                ctx.wait_key(k).await;
            }
        }
        self.report.borrow_mut().step_end.push(ctx.now());
    }

    /// A test building block: one router decision and its MoE dispatch
    /// (layer 0 of step 0), then the residual.
    pub async fn run_dispatch(mut self, ctx: NuCtx) {
        let ids = self.route(&ctx, 0, 0).await;
        self.moe(0, 0, &ids);
        let x = self.home.x;
        self.flush(&ctx).await;
        if let Some(k) = self.g.key(&x) {
            ctx.wait_key(k).await;
        }
        self.report.borrow_mut().step_end.push(ctx.now());
    }

    /// The whole run.
    pub async fn run(mut self, ctx: NuCtx) {
        let p = self.p.clone();
        for step in 0..p.steps {
            let mut ids: Option<Vec<u8>> = None;
            for l in 0..p.layers {
                if let Some(ids) = ids.take() {
                    self.moe(l - 1, step, &ids);
                }
                if p.attn_kv == "multicast" {
                    self.attention_mc(l, step);
                } else {
                    self.attention(l, step);
                }
                ids = Some(self.route(&ctx, l, step).await);
            }
            if let Some(ids) = ids.take() {
                self.moe(p.layers - 1, step, &ids);
            }
            let x = self.home.x;
            self.flush(&ctx).await;
            if let Some(k) = self.g.key(&x) {
                ctx.wait_key(k).await;
            }
            self.report.borrow_mut().step_end.push(ctx.now());
        }
    }
}

/// Per layer and routed expert: the fraction of steps in which the router
/// output of `records` sends at least one token to the expert.
pub fn expert_prob(
    records: &std::collections::BTreeMap<u64, Vec<u8>>,
    p: &Params,
) -> Vec<Vec<f64>> {
    (0..p.layers)
        .map(|l| {
            let mut hits = vec![0u32; p.experts as usize];
            let mut steps = 0u32;
            for s in 0..p.steps {
                let Some(ids) = records.get(&label::router(s, l)) else {
                    continue;
                };
                steps += 1;
                for (e, h) in hits.iter_mut().enumerate() {
                    if ids.iter().any(|&x| x as usize == e) {
                        *h += 1;
                    }
                }
            }
            hits.iter()
                .map(|&h| {
                    if steps == 0 {
                        1.0
                    } else {
                        h as f64 / steps as f64
                    }
                })
                .collect()
        })
        .collect()
}

/// The action expert's HBM map: weights from the HBM base (layer-major,
/// `layer_stride` bytes per layer), then the prefix KV cache.
pub fn hbm_map(m: &Machine, p: &Params) -> Result<(u64, u64, KvLayout), String> {
    let w = parse_dtype(&p.weight_dtype)?;
    let bb = m.round(w.bytes_for(p.d * p.block_cols));
    let hpb = p.head_dim / p.block_cols;
    let nb = p.ffn / p.block_cols;
    let layer_stride = ((2 * p.heads + 2) * hpb + (p.experts + 1) * 3 * nb) * bb;
    let base = m.hbm.base;
    let kv = KvLayout::new(
        base + p.layers as u64 * layer_stride,
        p.layers,
        p.prefix,
        p.head_dim,
        m.granule,
    );
    if kv.end() > m.hbm.bound {
        return Err(format!(
            "weights and KV need {} B of HBM, have {}",
            kv.end() - base,
            m.hbm.len()
        ));
    }
    Ok((base, layer_stride, kv))
}
