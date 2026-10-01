//! The program's view of a built configuration: the SMs with their PEs,
//! SMEM and sync-only space, and the HBM block. An SM is a node with PEs
//! named `atlas`, `muon` and `dma` and a scratch block named `smem`.

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
    /// SMEM port, bytes per cycle.
    pub smem: f64,
    /// Atlas array, MACs per cycle.
    pub mac: f64,
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

/// The program's view of a built configuration: the SMs, the HBM and
/// cmem owner nodes, the granule and the task opcodes.
#[derive(Debug, Clone)]
pub struct Machine {
    pub(crate) sms: Vec<Sm>,
    pub(crate) hbm: Range,
    /// The owner of HBM.
    pub(crate) hbm_node: Option<MemNode>,
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
        let mut hbm = None;
        let mut hbm_node = None;
        let mut cmem_nodes = Vec::new();
        let rate_of = |r: Range| -> f64 {
            b.sim
                .memories()
                .iter()
                .find(|m| m.range == r)
                .map_or(0.0, |m| res.flow(m.port).rate().to_f64())
        };
        let rate_named = |name: String| {
            res.by_name(&name)
                .map_or(0.0, |id| res.flow(id).rate().to_f64())
        };
        for i in 0..topo.len() {
            let nu = NeutrinoId(i as u32);
            let prefix = topo.path_name(nu);
            let mem_node = |mem: Range| -> Option<MemNode> {
                Some(MemNode {
                    grant: Some(u64::from(b.sim.scoreboard_room(nu))),
                    nu,
                    mem,
                    rate: rate_of(mem),
                    own: b.own_space(nu),
                    sync: b.sync_only(nu)?,
                    dma: b.try_pe(&format!("{prefix}.dma")),
                })
            };
            if let Some(h) = b.try_memory(&format!("{prefix}.hbm")) {
                hbm = Some(h);
                hbm_node = mem_node(h);
            }
            if let Some(c) = b.try_memory(&format!("{prefix}.cmem")) {
                cmem_nodes.extend(mem_node(c));
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
            // The bridge of a node to its parent each way: down is parent to
            // child.
            let link = |n: NeutrinoId| {
                topo.node(n).parent.map_or((0.0, 0.0), |_| {
                    let c = topo.path_name(n);
                    (
                        rate_named(format!("{c}.bridge.down")),
                        rate_named(format!("{c}.bridge.up")),
                    )
                })
            };
            let (link_down, link_up) = link(nu);
            // The cluster's link to its parent.
            let (cluster_down, cluster_up) = parent.map_or((0.0, 0.0), link);
            let (acc_buffer, row_panel) = match b.sim.pe_config(atlas).params {
                sim_core::PeParams::Atlas {
                    acc_buffer,
                    row_panel,
                    ..
                } => (acc_buffer, row_panel),
                _ => (0, false),
            };
            let rates = SmRates {
                cluster_down,
                cluster_up,
                link_down,
                link_up,
                smem: rate_named(format!("{prefix}.smem.port")),
                mac: rate_named(format!("{prefix}.atlas.array")),
                execution: b.sim.execution_capacity(nu) as u64,
                acc_buffer,
                row_panel,
            };
            sms.push(Sm {
                nu,
                cluster: parent,
                rates,
                cmem,
                atlas,
                muon,
                dma,
                smem,
                own: b.own_space(nu),
                sync,
            });
        }
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
            hbm: hbm.ok_or("no memory named hbm")?,
            hbm_node,
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
            inferred_version: None,
            deps: Vec::new(),
            label: None,
            args: Args::Matmul {
                a: Addr(a),
                b: Addr(b),
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
            inferred_version: None,
            deps: Vec::new(),
            label: None,
            args: Args::Vector {
                ops,
                bytes_in,
                bytes_out,
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

/// Bump allocator over one SM's SMEM, for the fixed buffer layout.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    space: Range,
    next: u64,
    granule: u64,
}

impl Layout {
    pub(crate) fn new(space: Range, granule: u64) -> Self {
        Self {
            space,
            next: space.base,
            granule,
        }
    }

    pub(crate) fn take(&mut self, bytes: u64) -> Result<Range, String> {
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
}
