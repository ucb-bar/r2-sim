//! The program's view of a built configuration: the SMs with their PEs,
//! SMEM and sync-only space, and the HBM slices. An SM is a node with PEs
//! named `atlas`, `muon` and `dma` and a scratch block named `smem`; an
//! HBM slice is a block named `hbm` (one per gpc under NUMA, or one at the
//! device).
//!
//! Placement rule for HBM (architecture Section 8.2, NUMA): a tensor goes
//! to the slice of the gpc whose SMs read it most ([`Machine::slice_for`]);
//! a tie goes to the lower slice, and a slice without room passes the
//! tensor to the next one.

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::{Addr, NeutrinoId, Range};
use sim_core::{Args, Built, Descriptor, Dtype, OpcodeId, PeId};

use crate::versions::VersionState;

#[derive(Debug, Clone)]
pub(crate) struct Sm {
    pub(crate) nu: NeutrinoId,
    /// The SM's parent node (its cluster).
    pub(crate) cluster: Option<NeutrinoId>,
    pub(crate) atlas: PeId,
    pub(crate) muon: PeId,
    pub(crate) dma: PeId,
    pub(crate) smem: Range,
    /// The banks of a banked SMEM (`smem.b0` ...), in address order; empty
    /// for one block.
    pub(crate) banks: Vec<Range>,
    /// The bank group that buffers of no operand role try first
    /// ([`Layout`]).
    pub(crate) whole: Role,
    /// The SM's rates and Atlas knobs, for cost models.
    pub(crate) rates: SmRates,
    /// The cluster's memory tier (a block named `cmem` at the parent).
    pub(crate) cmem: Option<Range>,
    pub(crate) own: Range,
    pub(crate) sync: Range,
}

/// The port rates and Atlas knobs of one SM, for cost models. A rate is
/// 0 when the configuration has no such port or link.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SmRates {
    /// Link from the parent node into the SM, bytes per cycle.
    pub link_down: f64,
    /// Link from the SM to its parent node, bytes per cycle.
    pub link_up: f64,
    /// SMEM read port, bytes per cycle (the sum over the banks of a
    /// banked SMEM).
    pub smem: f64,
    /// Atlas array, MACs per cycle (its highest rate).
    pub mac: f64,
    /// Atlas MACs per cycle by operand type, in [`Dtype`] order (bf16,
    /// f32, fp8, mxfp4); 0: `mac`.
    pub mac_by_type: [f64; 4],
    /// The Atlas keeps the A row panel across a row's output tiles (H1).
    pub row_panel: bool,
    /// Atlas accumulator memory, bytes (H3).
    pub acc_buffer: u64,
    /// The SM Neutrino's execution table capacity (executing and done tasks).
    pub execution: u64,
    /// The SM's cluster's link to its parent, each way (0 without one).
    pub cluster_down: f64,
    /// The SM's cluster's link to its parent, upward (see `cluster_down`).
    pub cluster_up: f64,
}

impl SmRates {
    /// Atlas MACs per cycle on operands of types `a` and `b`: the slower
    /// type's rate (architecture Section 5.5).
    pub fn mac_for(&self, a: Dtype, b: Dtype) -> f64 {
        let r = |d: Dtype| match self.mac_by_type[d as usize] {
            x if x > 0.0 => x,
            _ => self.mac,
        };
        r(a).min(r(b))
    }
}

/// A node that owns a shared memory block (the device with HBM, a cluster
/// with cmem): it tracks every write into the block.
#[derive(Debug, Clone)]
pub(crate) struct MemNode {
    pub(crate) nu: NeutrinoId,
    pub(crate) mem: Range,
    /// Port rate of the block, bytes per cycle.
    pub(crate) rate: f64,
    pub(crate) own: Range,
    pub(crate) sync: Range,
    /// The node's copy engine, if any (writes into its memory).
    pub(crate) dma: Option<PeId>,
    /// The tasks the node can track for one requester (the driver) at
    /// once: its scoreboard less its issue headroom.
    pub(crate) grant: Option<u64>,
}

/// Bytes one gpc's SMs moved to or from HBM: from their own gpc's slice
/// (`local`) or from another slice (`remote`).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct GpcBytes {
    /// The gpc's dotted name.
    pub gpc: String,
    /// Bytes to or from the gpc's own slice.
    pub local: u64,
    /// Bytes to or from other slices.
    pub remote: u64,
}

/// Bump allocators over the HBM slices: [`SliceBump::take`] tries the
/// preferred slice, then the others in order.
#[derive(Debug, Clone)]
pub(crate) struct SliceBump {
    pub(crate) next: Vec<u64>,
    bound: Vec<u64>,
    granule: u64,
}

impl SliceBump {
    /// From `next` (per slice) to the slices' bounds.
    pub(crate) fn new(m: &Machine, next: Vec<u64>) -> Self {
        Self {
            next,
            bound: m.hbm.iter().map(|h| h.mem.bound).collect(),
            granule: m.granule,
        }
    }

    /// From the start of every slice.
    pub(crate) fn start(m: &Machine) -> Self {
        Self::new(m, m.hbm.iter().map(|h| h.mem.base).collect())
    }

    /// `bytes` (granule-rounded) in slice `pref`, or the next slice with
    /// room.
    pub(crate) fn take(&mut self, pref: usize, bytes: u64) -> Result<u64, String> {
        let bytes = bytes.div_ceil(self.granule) * self.granule;
        let n = self.next.len();
        for k in 0..n {
            let s = (pref + k) % n;
            if self.next[s] + bytes <= self.bound[s] {
                let a = self.next[s];
                self.next[s] += bytes;
                return Ok(a);
            }
        }
        Err(format!(
            "HBM: no slice has {bytes} B free (next {:x?}, bounds {:x?})",
            self.next, self.bound
        ))
    }
}

/// The program's view of a built configuration: the SMs, the HBM slices
/// and cmem owner nodes, the granule and the task opcodes.
#[derive(Debug, Clone)]
pub struct Machine {
    pub(crate) sms: Vec<Sm>,
    /// HBM slices with their owner nodes, in configuration order.
    pub(crate) hbm: Vec<MemNode>,
    /// The parents of the SMs (gpcs), in configuration order, and their
    /// dotted names.
    pub(crate) gpcs: Vec<(NeutrinoId, String)>,
    /// Per gpc (as `gpcs`): its own space, and its subtree (its space and
    /// its SMs').
    pub(crate) gpc_spaces: Vec<(Range, Range)>,
    /// The sync-only space of the device (the root), if any.
    pub(crate) root_sync: Option<Range>,
    /// Clusters with a cmem block, in configuration order.
    pub(crate) cmem_nodes: Vec<MemNode>,
    pub(crate) granule: u64,
    pub(crate) dma_op: OpcodeId,
    pub(crate) mm_op: OpcodeId,
    pub(crate) vec_op: OpcodeId,
    /// The program's version bookkeeping, shared by its drivers.
    pub(crate) versions: Rc<RefCell<VersionState>>,
}

impl Machine {
    /// Collects the machine from `b`; `granule` is the allocation granule,
    /// bytes. Errors when an SM has no sync-only space, when there is no
    /// `hbm` memory, or when a `dma`, `atlas.matmul` or `muon.vector` profile
    /// is missing.
    pub fn from_built(b: &Built, granule: u64) -> Result<Self, String> {
        let topo = b.sim.topology();
        let res = b.sim.resources();
        let mut sms = Vec::new();
        let mut hbm = Vec::new();
        let mut cmem_nodes = Vec::new();
        // The port rate of the memory named `name`: the sum over the
        // banks of a banked memory.
        let rate_of = |name: &str| -> f64 {
            b.sim
                .memories()
                .iter()
                .filter(|m| m.memory_name() == name)
                .map(|m| res.flow(m.port).rate().to_f64())
                .sum()
        };
        let rate_named = |name: String| {
            res.by_name(&name)
                .map_or(0.0, |id| res.flow(id).rate().to_f64())
        };
        for i in 0..topo.len() {
            let nu = NeutrinoId(i as u32);
            let prefix = topo.path_name(nu);
            let mem_node = |mem: Range, name: &str| -> Option<MemNode> {
                Some(MemNode {
                    grant: Some(u64::from(b.sim.scoreboard_room(nu))),
                    nu,
                    mem,
                    rate: rate_of(name),
                    own: b.own_space(nu),
                    sync: b.sync_only(nu)?,
                    dma: b.try_pe(&format!("{prefix}.dma")),
                })
            };
            if let Some(h) = b.try_memory(&format!("{prefix}.hbm")) {
                hbm.push(
                    mem_node(h, &format!("{prefix}.hbm"))
                        .ok_or_else(|| format!("{prefix} owns hbm but has no sync space"))?,
                );
            }
            if let Some(c) = b.try_memory(&format!("{prefix}.cmem")) {
                cmem_nodes.extend(mem_node(c, &format!("{prefix}.cmem")));
            }
            let pe = |n: &str| b.try_pe(&format!("{prefix}.{n}"));
            let (Some(atlas), Some(muon), Some(dma), Some(smem)) = (
                pe("atlas"),
                pe("muon"),
                pe("dma"),
                b.try_memory(&format!("{prefix}.smem")),
            ) else {
                continue;
            };
            let sync = b
                .sync_only(nu)
                .ok_or_else(|| format!("SM {prefix} has no sync_only space"))?;
            let parent = topo.node(nu).parent;
            let ppath = parent.map(|c| topo.path_name(c));
            let cmem = ppath
                .as_ref()
                .and_then(|cp| b.try_memory(&format!("{cp}.cmem")));
            // The data link of a node to its parent each way (down: parent
            // to child), found by its buses: one end on a bus of the
            // parent. 0 when unlimited or absent.
            let link = |n: NeutrinoId| -> (f64, f64) {
                let Some(p) = topo.node(n).parent else {
                    return (0.0, 0.0);
                };
                let ppath = topo.path_name(p);
                let on_parent = |bus: sim_core::BusId| {
                    topo.bus(bus).name.rsplit_once('.').map(|x| x.0) == Some(ppath.as_str())
                };
                let rate = |l: sim_core::Lane| l.map_or(0.0, |r| res.flow(r).rate().to_f64());
                for (info, l) in b.layout.links.iter().zip(topo.fabric_links()) {
                    if info.node != n || info.control || info.cache.is_some() {
                        continue;
                    }
                    if on_parent(l.b) {
                        return (rate(l.ba), rate(l.ab));
                    }
                    if on_parent(l.a) {
                        return (rate(l.ab), rate(l.ba));
                    }
                }
                (0.0, 0.0)
            };
            let (link_down, link_up) = link(nu);
            // The cluster's link to its parent.
            let (cluster_down, cluster_up) = parent.map_or((0.0, 0.0), link);
            let (acc_buffer, row_panel, mut mac_by_type) = match &b.sim.pe_config(atlas).params {
                sim_core::PeParams::Atlas {
                    acc_buffer,
                    row_panel,
                    ..
                } => (*acc_buffer, *row_panel, [0.0; 4]),
                _ => (0, false, [0.0; 4]),
            };
            if let sim_core::PeParams::Atlas { rates, .. } = &b.sim.pe_config(atlas).params {
                for (d, r) in rates {
                    mac_by_type[*d as usize] = r.to_f64();
                }
            }
            let rates = SmRates {
                cluster_down,
                cluster_up,
                link_down,
                link_up,
                smem: rate_of(&format!("{prefix}.smem")),
                mac: rate_named(format!("{prefix}.atlas.array")),
                mac_by_type,
                execution: b.sim.execution_capacity(nu) as u64,
                acc_buffer,
                row_panel,
            };
            let mut banks: Vec<Range> = b
                .sim
                .memories()
                .iter()
                .filter(|m| m.banked && m.memory_name() == format!("{prefix}.smem"))
                .map(|m| m.range)
                .collect();
            banks.sort_by_key(|r| r.base);
            sms.push(Sm {
                nu,
                cluster: parent,
                rates,
                cmem,
                atlas,
                muon,
                dma,
                smem,
                banks,
                whole: Role::Act,
                own: b.own_space(nu),
                sync,
            });
        }
        let mut gpcs: Vec<(NeutrinoId, String)> = Vec::new();
        for s in &sms {
            if let Some(c) = s.cluster {
                if !gpcs.iter().any(|g| g.0 == c) {
                    gpcs.push((c, topo.path_name(c)));
                }
            }
        }
        if hbm.is_empty() {
            return Err("no memory named hbm".into());
        }
        let gpc_spaces = gpcs
            .iter()
            .map(|g| {
                let own = b.own_space(g.0);
                let bound = sms
                    .iter()
                    .filter(|s| s.cluster == Some(g.0))
                    .map(|s| s.own.bound)
                    .fold(own.bound, u64::max);
                (own, Range::new(own.base, bound))
            })
            .collect();
        let reg = b.sim.profiles();
        let owners = (0..topo.len())
            .map(|i| {
                let nu = NeutrinoId(i as u32);
                (nu, b.own_space(nu), b.sim.version_capacity(nu))
            })
            .collect();
        Ok(Self {
            versions: Rc::new(RefCell::new(VersionState::new(owners))),
            sms,
            hbm,
            gpcs,
            gpc_spaces,
            root_sync: b.sync_only(NeutrinoId(0)),
            cmem_nodes,
            granule,
            dma_op: reg.lookup("dma").ok_or("no dma profile")?,
            mm_op: reg.lookup("atlas.matmul").ok_or("no matmul profile")?,
            vec_op: reg.lookup("muon.vector").ok_or("no vector profile")?,
        })
    }

    pub(crate) fn round(&self, bytes: u64) -> u64 {
        bytes.div_ceil(self.granule) * self.granule
    }

    /// The same machine with its own version bookkeeping (another
    /// driver's).
    pub(crate) fn with_own_versions(&self) -> Self {
        let mut m = self.clone();
        m.versions = Rc::new(RefCell::new(self.versions.borrow().fresh()));
        m
    }

    /// The HBM bandwidth of all slices, bytes per cycle.
    pub(crate) fn hbm_rate(&self) -> f64 {
        self.hbm.iter().map(|h| h.rate).sum()
    }

    /// The HBM slice that holds `addr`.
    pub(crate) fn slice_of(&self, addr: u64) -> Option<usize> {
        self.hbm
            .iter()
            .position(|h| h.mem.covers(&Range::from_len(addr, 1)))
    }

    /// The slice of the gpc whose SMs among `readers` are the most; a tie
    /// goes to the lower slice; slice 0 when no reader's gpc owns one.
    pub(crate) fn slice_for(&self, readers: impl IntoIterator<Item = usize>) -> usize {
        let mut count = vec![0usize; self.hbm.len()];
        for sm in readers {
            if let Some(i) = self.local_slice(sm) {
                count[i] += 1;
            }
        }
        let best = count.iter().copied().max().unwrap_or(0);
        count.iter().position(|&c| c == best).unwrap_or(0)
    }

    /// The slice that SM `sm`'s gpc owns.
    pub(crate) fn local_slice(&self, sm: usize) -> Option<usize> {
        let g = self.sms[sm].cluster?;
        self.hbm.iter().position(|h| h.nu == g)
    }

    /// Per gpc, no bytes yet.
    pub(crate) fn gpc_bytes(&self) -> Vec<GpcBytes> {
        self.gpcs
            .iter()
            .map(|g| GpcBytes {
                gpc: g.1.clone(),
                ..GpcBytes::default()
            })
            .collect()
    }

    /// Adds `bytes` of SM `sm` at HBM address `addr` to `split`: local
    /// when the slice is the SM's gpc's.
    pub(crate) fn count_hbm(&self, split: &mut [GpcBytes], sm: usize, addr: u64, bytes: u64) {
        let Some(c) = self.sms[sm].cluster else {
            return;
        };
        let Some(g) = self.gpcs.iter().position(|g| g.0 == c) else {
            return;
        };
        if self.slice_of(addr).is_some_and(|s| self.hbm[s].nu == c) {
            split[g].local += bytes;
        } else {
            split[g].remote += bytes;
        }
    }

    pub(crate) fn dma(
        &self,
        sm: usize,
        src: u64,
        dst: Range,
        bytes: u64,
        chunk: u64,
    ) -> Descriptor {
        Descriptor {
            opcode: self.dma_op,
            pe: Some(self.sms[sm].dma),
            dst,
            versioned: true,
            untracked: false,
            inferred_version: None,
            deps: Vec::new(),
            label: None,
            args: Args::Dma {
                src: Addr(src),
                dst: Addr(dst.base),
                bytes,
                tile_bytes: chunk.min(bytes).max(1),
            },
        }
    }

    /// `C = A x B` (or `C += A x B`), bf16, row-major, C at `dst.base`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul(
        &self,
        sm: usize,
        a: u64,
        b: u64,
        dst: Range,
        m: u64,
        n: u64,
        k: u64,
        accumulate: bool,
        tile: [u64; 2],
    ) -> Descriptor {
        Descriptor {
            opcode: self.mm_op,
            pe: Some(self.sms[sm].atlas),
            dst,
            versioned: true,
            untracked: false,
            inferred_version: None,
            deps: Vec::new(),
            label: None,
            args: Args::Matmul {
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
                in_dtype: Dtype::Bf16,
                b_dtype: Dtype::Bf16,
                out_dtype: Dtype::Bf16,
                accumulate,
                keep: false,
                tile,
            },
        }
    }

    /// `src` for a read of `bytes` in SM `sm`'s SMEM, moved down so that
    /// the read stays in the bank of its first byte (a banked SMEM only).
    pub(crate) fn in_bank(&self, sm: usize, src: u64, bytes: u64) -> u64 {
        match self.sms[sm].banks.iter().find(|b| b.contains(Addr(src))) {
            Some(b) if src + bytes > b.bound => b.bound.saturating_sub(bytes).max(b.base),
            _ => src,
        }
    }

    /// [`Machine::matmul`] on operands that may be split over a bank pair
    /// (Args::Matmul, architecture Section 5.5): A from `a`, B from `b`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_split(
        &self,
        sm: usize,
        a: &Split,
        b: &Split,
        dst: Range,
        m: u64,
        n: u64,
        k: u64,
        accumulate: bool,
        tile: [u64; 2],
    ) -> Descriptor {
        let d = self.matmul(sm, a.lo.base, b.lo.base, dst, m, n, k, accumulate, tile);
        with_b_split(b, with_a_split(a, d))
    }

    pub(crate) fn vector(
        &self,
        sm: usize,
        dst: Range,
        ops: u64,
        bytes_in: u64,
        bytes_out: u64,
    ) -> Descriptor {
        Descriptor {
            opcode: self.vec_op,
            pe: Some(self.sms[sm].muon),
            dst,
            versioned: true,
            untracked: false,
            inferred_version: None,
            deps: Vec::new(),
            label: None,
            args: Args::Vector {
                ops,
                bytes_in,
                bytes_out,
                src: None,
            },
        }
    }
}

/// The weight element type named `s` (`weight_dtype`).
pub(crate) fn parse_dtype(s: &str) -> Result<Dtype, String> {
    match s {
        "bf16" => Ok(Dtype::Bf16),
        "fp8" => Ok(Dtype::Fp8),
        "mxfp4" => Ok(Dtype::Mxfp4),
        _ => Err(format!("unknown weight_dtype {s:?} (bf16, fp8, mxfp4)")),
    }
}

/// The worker SMs of a driver on SM `home` of `n`: `workers`, or every SM
/// but `home` when it is empty.
pub(crate) fn worker_list(n: usize, home: usize, workers: &[usize]) -> Result<Vec<usize>, String> {
    if home >= n {
        return Err(format!("home SM {home} of {n}"));
    }
    let w: Vec<usize> = if workers.is_empty() {
        (0..n).filter(|&i| i != home).collect()
    } else {
        workers.to_vec()
    };
    if w.is_empty() || w.iter().any(|&x| x >= n) {
        return Err(format!("bad worker list {w:?} for {n} SMs"));
    }
    Ok(w)
}

/// Sets the high half of operand A of a matmul from `a` (Args::Matmul).
pub(crate) fn with_a_split(a: &Split, mut d: Descriptor) -> Descriptor {
    if let Args::Matmul { a2, .. } = &mut d.args {
        *a2 = a.hi.map(|r| Addr(r.base));
    }
    d
}

/// Sets the high half of operand B of a matmul from `b`, and its leading
/// dimension to the low half matrix's columns (Args::Matmul).
pub(crate) fn with_b_split(b: &Split, mut d: Descriptor) -> Descriptor {
    if let Args::Matmul {
        b2, ldb, n, tile, ..
    } = &mut d.args
    {
        *b2 = b.hi.map(|r| Addr(r.base));
        if b.hi.is_some() {
            *ldb = low_rows(*n, tile[1]);
        }
    }
    d
}

/// Sets the B operand type of a matmul (a weight block) to `dtype`.
pub(crate) fn with_b_dtype(dtype: Dtype, mut d: Descriptor) -> Descriptor {
    if let Args::Matmul { b_dtype, .. } = &mut d.args {
        *b_dtype = dtype;
    }
    d
}

/// A ring of signal ranges in one node's sync-only space. Joins are
/// versioned (Section 3.2, Versions), so a range is reused at once: a
/// consumer names the join's version, not the range alone.
#[derive(Debug, Clone)]
pub(crate) struct SyncPool {
    base: u64,
    next: u64,
    granule: u64,
}

impl SyncPool {
    /// Ranges in the ring: each takes one version table entry at the owner.
    pub(crate) const RING: u64 = 32;

    pub(crate) fn new(space: Range, granule: u64) -> Self {
        assert!(
            space.len() >= Self::RING * granule,
            "sync-only space smaller than the signal ring"
        );
        Self {
            base: space.base,
            next: 0,
            granule,
        }
    }

    pub(crate) fn alloc(&mut self) -> Range {
        let r = Range::from_len(
            self.base + (self.next % Self::RING) * self.granule,
            self.granule,
        );
        self.next += 1;
        r
    }
}

/// Rows of `rows` (from row 0) that are in the low halves of their tiles
/// of height `t` (Args::Matmul split operands).
pub(crate) fn low_rows(rows: u64, t: u64) -> u64 {
    let h = t / 2;
    rows / t * h + (rows % t).min(h)
}

/// The operand role of an SMEM buffer: activations (Atlas operand A, Muon
/// inputs) or weights (operand B); see [`Layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Activations.
    Act,
    /// Weights.
    W,
}

/// A buffer of `units` equal units for an Atlas operand: one range, or on
/// a banked SMEM a low half and a high half in the bank pair of its role
/// (Args::Matmul split operands). A unit of `unit` bytes puts the share
/// `low / total` of its bytes in the low half (rows in low tile halves
/// over rows; 1 / 2 for weight columns); each half unit is
/// granule-rounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Split {
    pub(crate) lo: Range,
    pub(crate) hi: Option<Range>,
    unit: u64,
    unit_lo: u64,
    unit_hi: u64,
    low: u64,
    total: u64,
}

impl Split {
    /// One range, one unit.
    pub(crate) fn whole(r: Range) -> Self {
        Self {
            lo: r,
            hi: None,
            unit: r.len(),
            unit_lo: r.len(),
            unit_hi: 0,
            low: 1,
            total: 1,
        }
    }

    /// Unit `i`.
    pub(crate) fn at(&self, i: u64) -> Self {
        Self {
            lo: match self.hi {
                None => Range::from_len(self.lo.base + i * self.unit, self.unit),
                Some(_) => Range::from_len(self.lo.base + i * self.unit_lo, self.unit_lo),
            },
            hi: self
                .hi
                .map(|h| Range::from_len(h.base + i * self.unit_hi, self.unit_hi)),
            ..*self
        }
    }

    /// A low half `lo` and a high half `hi` of one unit, `low / total` of
    /// its bytes in the low half.
    pub(crate) fn pair(lo: Range, hi: Range, low: u64, total: u64) -> Self {
        Self {
            lo,
            hi: Some(hi),
            unit: lo.len() + hi.len(),
            unit_lo: lo.len(),
            unit_hi: hi.len(),
            low,
            total,
        }
    }

    /// The ranges, as dependencies.
    pub(crate) fn deps(&self) -> [Option<Range>; 2] {
        [Some(self.lo), self.hi]
    }

    /// `bytes` of a unit's data split as the halves hold it: (low, high);
    /// all low for one range.
    pub(crate) fn halves(&self, bytes: u64) -> (u64, u64) {
        match self.hi {
            None => (bytes, 0),
            Some(_) => {
                let lo = bytes * self.low / self.total;
                (lo, bytes - lo)
            }
        }
    }
}

/// Bump allocator over one SM's SMEM, for the fixed buffer layout
/// (architecture Section 5.5, banked SMEM). On a banked SMEM each role has
/// a bank group: with two banks, activations bank 0 and weights bank 1;
/// with four, banks 0 and 1 and banks 2 and 3. An operand buffer goes to
/// its role's group: whole in the emptier bank of the group, or with four
/// banks split in halves over the pair ([`Layout::take_split`]). Other
/// buffers ([`Layout::take`]) go to the emptier bank of the group of
/// `whole`, or of the other group when that has no room.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    space: Range,
    next: u64,
    granule: u64,
    /// Per bank of a banked SMEM: (bank, next free address).
    banks: Vec<(Range, u64)>,
    /// The group that [`Layout::take`] tries first.
    whole: Role,
}

impl Layout {
    pub(crate) fn new(space: Range, granule: u64) -> Self {
        Self {
            space,
            next: space.base,
            granule,
            banks: Vec::new(),
            whole: Role::Act,
        }
    }

    /// The layout of SM `sm`'s SMEM: banked when it has two or four banks.
    pub(crate) fn of_sm(sm: &Sm, granule: u64) -> Self {
        Self::with_banks(sm.smem, &sm.banks, granule, sm.whole)
    }

    /// The layout of `space`, banked over `banks` when there are two or
    /// four; other buffers try the group of `whole` first.
    pub(crate) fn with_banks(space: Range, banks: &[Range], granule: u64, whole: Role) -> Self {
        let mut l = Self::new(space, granule);
        if banks.len() == 2 || banks.len() == 4 {
            l.banks = banks.iter().map(|b| (*b, b.base)).collect();
        }
        l.whole = whole;
        l
    }

    /// True on a banked SMEM.
    pub(crate) fn banked(&self) -> bool {
        !self.banks.is_empty()
    }

    fn round(&self, bytes: u64) -> u64 {
        (bytes.div_ceil(self.granule) * self.granule).max(self.granule)
    }

    /// The banks of `role`'s group.
    fn group(&self, role: Role) -> std::ops::Range<usize> {
        let half = self.banks.len() / 2;
        match role {
            Role::Act => 0..half,
            Role::W => half..2 * half,
        }
    }

    fn take_in(&mut self, i: usize, bytes: u64) -> Result<Range, String> {
        let len = self.round(bytes);
        let (bank, next) = self.banks[i];
        let r = Range::from_len(next, len);
        if !bank.covers(&r) {
            return Err(format!(
                "SMEM bank {i} needs more than {} B (at {:#x} + {len})",
                bank.len(),
                next - bank.base
            ));
        }
        self.banks[i].1 += len;
        Ok(r)
    }

    /// The emptier bank of `role`'s group (ties: the lower) and its room.
    fn emptiest(&self, role: Role) -> (usize, u64) {
        let room = |i: usize| self.banks[i].0.bound - self.banks[i].1;
        self.group(role)
            .map(|i| (i, room(i)))
            .fold((usize::MAX, 0), |a, x| {
                if a.0 == usize::MAX || x.1 > a.1 {
                    x
                } else {
                    a
                }
            })
    }

    /// A buffer of no operand role: the emptier bank of the group of
    /// `whole`, else of the other group.
    pub(crate) fn take(&mut self, bytes: u64) -> Result<Range, String> {
        if self.banked() {
            let len = self.round(bytes);
            let other = match self.whole {
                Role::Act => Role::W,
                Role::W => Role::Act,
            };
            for role in [self.whole, other] {
                let (i, room) = self.emptiest(role);
                if room >= len {
                    return self.take_in(i, bytes);
                }
            }
            return self.take_in(self.emptiest(self.whole).0, bytes);
        }
        let len = bytes.div_ceil(self.granule) * self.granule;
        let r = Range::from_len(self.next, len.max(self.granule));
        if !self.space.covers(&r) {
            return Err(format!(
                "SMEM layout needs more than {} B (at {:#x} + {len})",
                self.space.len(),
                self.next - self.space.base
            ));
        }
        self.next += r.len();
        Ok(r)
    }

    /// A whole buffer of `role` (an operand that is not split, such as a
    /// matmul output a later matmul reads): the emptier bank of the role's
    /// group, else of the other group when the role's has no room (with
    /// bf16 weights the weight bank of the action expert's SMs is full).
    pub(crate) fn take_role(&mut self, role: Role, bytes: u64) -> Result<Range, String> {
        if !self.banked() {
            return self.take(bytes);
        }
        let len = self.round(bytes);
        let other = match role {
            Role::Act => Role::W,
            Role::W => Role::Act,
        };
        for r in [role, other] {
            let (i, room) = self.emptiest(r);
            if room >= len {
                return self.take_in(i, bytes);
            }
        }
        self.take_in(self.emptiest(role).0, bytes)
    }

    /// `units` units of `unit` bytes for role `role`: on four banks split
    /// over the role's pair, `low / total` of each unit in the low half
    /// (see [`Split`]); otherwise one range of `units * unit` bytes (on two
    /// banks in the role's bank).
    pub(crate) fn take_split(
        &mut self,
        role: Role,
        units: u64,
        unit: u64,
        low: u64,
        total: u64,
    ) -> Result<Split, String> {
        if self.banks.len() != 4 {
            // One range: whole in the role's bank on two banks.
            let r = self.take_role(role, units * unit)?;
            return Ok(Split {
                lo: r,
                hi: None,
                unit,
                unit_lo: unit,
                unit_hi: 0,
                low: 1,
                total: 1,
            });
        }
        let ul = unit * low / total;
        let (unit_lo, unit_hi) = (self.round(ul), self.round(unit - ul));
        let pair = match role {
            Role::Act => 0,
            Role::W => 2,
        };
        let lo = self.take_in(pair, units * unit_lo)?;
        let hi = self.take_in(pair + 1, units * unit_hi)?;
        Ok(Split {
            lo,
            hi: Some(hi),
            unit,
            unit_lo,
            unit_hi,
            low,
            total,
        })
    }
}
