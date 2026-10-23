//! Micro-benchmarks: targeted tests that isolate a few parameters each, to
//! compare the model with measurements on a card (the RTX PRO 6000
//! configurations, r2sim spec Section 13.3).
//!
//! One mode per run (`mode`):
//!
//! * `stream`: each of `sms` SMs moves `bytes` between its SMEM and
//!   `mem` in transfers of `chunk` bytes, at most `inflight` at once per
//!   SM. `dir = read` moves `mem` into the SM's SMEM, `write` the SM's
//!   SMEM into `mem`. A write into the device's memories (`l2`, `dram`) is
//!   a task of the device's copy engine (`root.dma`), as every write into
//!   another node's memory is a pull by its owner's engine.
//! * `latency`: one transfer of `bytes` from `mem` into the first SM's
//!   SMEM.
//! * `atlas`: one matmul (`m`, `n`, `k`, `dtype`) per tensor unit of each
//!   SM (one per sub-partition when the SM has `atlas0`, `atlas1`, ...),
//!   operands in SMEM; the rate is the SM's over all its units.
//! * `muon`: one vector task (`lane_ops`, `bytes` read, `bytes_out`
//!   written) per SM.
//! * `pingpong`: two SMs (`pair = same_gpc | other_gpc`); `rounds` round
//!   trips of a dependency: a task on A writes a flag, a task on B depends
//!   on it and writes the next flag, and so on.
//! * `barrier`: `sms` SMs each run an empty task, then a join on the
//!   device depends on all of them.
//! * `invoke`: the driver invokes `tasks` empty tasks on one SM (`targets =
//!   one`) or round-robin over `sms` SMs (`targets = many`).
//! * `untracked`: each of `sms` SMs copies `bytes` from its SMEM into the
//!   DRAM of its gpc (`<gpc>.hbm`) in untracked DMA tasks of `chunk` bytes
//!   on its own engine (no pull by the gpc's engine, no Invoke message),
//!   fences (`wait 0`), and then signals the device with a forwarded join
//!   into the device's sync-only space. The device waits for the signals.
//!
//! The flags and empty tasks of `pingpong`, `barrier` and `invoke` are
//! joins in the SMs' sync-only space (a join writes no data and must be
//! there, 2026-10-20).
//!
//! `mem`: `l2` and `dram` (the linear ranges of the memories named by `l2`
//! and `dram`), `smem` (the SM's own SMEM), `rsmem_gpc` (another SM's SMEM
//! in the same gpc), `rsmem_far` (an SM's SMEM in another gpc). SMs are
//! taken `packed` (gpc0's first) or `spread` (round-robin over the gpcs).
//! The driver: `root` (one program on the device invokes every task, its
//! invokes forwarded to the SMs), `sm` (one program on the first SM;
//! single-SM modes), or `sms` (the device starts a program on each gpc
//! that starts one on each of its SMs, and each SM invokes its own tasks,
//! as a card's SMs issue their own copies). The default is `sms` for
//! `stream` and `root` for the other modes.
//!
//! Every task writes a versioned range, so a task retires when the next
//! version of its range is invoked; the program infers the versions. A
//! transfer writes one of `inflight` slots of `chunk` bytes per SM, so
//! writes into `mem` reuse `inflight x chunk` bytes per SM; reads take
//! `bytes` of contiguous source per SM.
//!
//! [`result`] turns the run's task timeline and resource totals into the
//! measured quantities of the mode (the `micro` object of `--stats`).

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use nu_model::{Addr, Dep, NeutrinoId, Range};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sim_core::{Args, Built, Descriptor, Dtype, NuCtx, PeId, Stats};

use crate::machine::parse_dtype;

/// The alignment of every range a task writes (the Neutrino granule).
const GRANULE: u64 = 512;

/// Parameters of a micro-benchmark run (TOML, or `--param key=value`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Params {
    pub mode: String,
    pub sms: usize,
    pub placement: String,
    pub mem: String,
    pub dir: String,
    pub bytes: u64,
    pub chunk: u64,
    pub inflight: usize,
    pub m: u64,
    pub n: u64,
    /// The k of a matmul (`atlas`, default 1024) or of a chain (`tmem`,
    /// default 2048).
    pub k: Option<u64>,
    pub dtype: String,
    /// `tmem`: the type of B when it differs from `dtype` (for example
    /// mxfp4 with bf16 A).
    pub b_dtype: String,
    /// `tmem`: the k of each matmul task of a chain.
    pub kc: u64,
    /// `tmem`: the A and B buffers in the tmem.
    pub stages: usize,
    /// `tmem`: the 128 x 128 output tiles, each a chain over k.
    pub tiles: usize,
    /// Output tile of a matmul `[rows, columns]`.
    pub tile: [u64; 2],
    pub lane_ops: u64,
    pub bytes_out: u64,
    pub pair: String,
    pub rounds: usize,
    pub tasks: usize,
    pub targets: String,
    pub driver: String,
    /// The memories `mem = l2` and `mem = dram` name.
    pub l2: String,
    pub dram: String,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            mode: "stream".into(),
            sms: 1,
            placement: "packed".into(),
            mem: "l2".into(),
            dir: "read".into(),
            bytes: 1 << 20,
            chunk: 16 << 10,
            inflight: 4,
            m: 128,
            n: 128,
            k: None,
            dtype: "bf16".into(),
            b_dtype: String::new(),
            kc: 128,
            stages: 2,
            tiles: 4,
            tile: [128, 128],
            lane_ops: 1 << 16,
            bytes_out: 0,
            pair: "same_gpc".into(),
            rounds: 16,
            tasks: 64,
            targets: "one".into(),
            driver: String::new(),
            l2: "root.l2".into(),
            dram: "root.hbm".into(),
        }
    }
}

impl Params {
    /// `k`, or `default` when it is not given.
    fn k_or(&self, default: u64) -> u64 {
        self.k.unwrap_or(default)
    }
}

/// One SM of the machine.
#[derive(Debug, Clone)]
pub(crate) struct Sm {
    pub(crate) path: String,
    pub(crate) nu: NeutrinoId,
    pub(crate) gpc: usize,
    pub(crate) dma: PeId,
    /// The tensor units: `atlas`, or `atlas0`, `atlas1`, ... (one per
    /// sub-partition).
    pub(crate) atlas: Vec<PeId>,
    pub(crate) muon: Option<PeId>,
    pub(crate) smem: Range,
    /// Its sync-only range (joins and signals; none without one).
    pub(crate) sync: Option<Range>,
}

/// What a task of the run is for.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Role {
    /// A transfer of `bytes` for SM `sm`.
    Data,
    /// A flag or empty task on SM `sm` (`usize::MAX`: the device).
    Flag,
    /// The join of `barrier`.
    Join,
    /// `tmem`: a matmul of a chain.
    Matmul,
    /// `tmem`: the copy of a tile's C from the tmem to the SMEM.
    Copy,
}

#[derive(Debug, Clone, Copy)]
struct Meta {
    role: Role,
    /// Index into the run's SMs.
    sm: usize,
    bytes: u64,
}

/// What the program records, read by [`result`] after the run.
#[derive(Debug, Default)]
pub struct Report {
    params: Option<Params>,
    sms: Vec<Sm>,
    /// By task label (the index of the task in the run).
    metas: Vec<Meta>,
    driver_start: u64,
    driver_end: u64,
    /// The driver's cycle after each invoke call returned.
    invoked_at: Vec<u64>,
    /// `tmem`: the chains' MACs and the Atlas rate (MACs per cycle) of
    /// their operand types.
    macs: u64,
    rate: f64,
}

/// One task of the program: its descriptor, the window it counts in (at
/// most `limit` open per window) and the version it completes.
struct Item {
    desc: Descriptor,
    window: usize,
    done: (u64, u64),
    /// A fence (`wait 0`) before the invoke.
    fence: bool,
    /// The driver waits for `done` at the end. An untracked task has no
    /// key, and a signal is the device's to wait for.
    wait: bool,
}

/// A gpc of a `driver = sms` run: its Neutrino, its sync range (the
/// `done` of its SMs' spawns) and its SMs (index into the run's SMs,
/// Neutrino).
type Gpc = (NeutrinoId, Range, Vec<(usize, NeutrinoId)>);

/// A built run: the tasks and how the driver issues them.
pub struct Run {
    items: Vec<Item>,
    /// Open tasks per window before the driver waits for the oldest
    /// (`usize::MAX`: no limit).
    limit: usize,
    /// The driver waits for these at the end (all tasks when empty).
    last: Vec<(u64, u64)>,
    pub report: Rc<RefCell<Report>>,
    /// The Neutrino that runs the driver.
    pub driver: NeutrinoId,
    /// `driver = sms`: the gpcs (each with its sync range for the spawns'
    /// `done`) and their SMs (index into the run's SMs, Neutrino); the
    /// device's sync range.
    tree: Option<(Range, Vec<Gpc>)>,
    /// The device waits for these signals (unversioned joins of the SMs)
    /// after it starts the gpcs.
    signals: Vec<u64>,
}

/// The SMs of `b`: nodes with an SMEM block and a DMA engine, ordered by
/// gpc and then by SM.
pub(crate) fn sms_of(b: &Built) -> Vec<Sm> {
    let topo = b.sim.topology();
    let mut out = Vec::new();
    for i in 0..topo.len() {
        let nu = NeutrinoId(i as u32);
        let path = topo.path_name(nu);
        let (Some(smem), Some(dma)) = (
            b.try_memory(&format!("{path}.smem")),
            b.try_pe(&format!("{path}.dma")),
        ) else {
            continue;
        };
        let gpc = topo.node(nu).parent.map_or(0, |p| p.0 as usize);
        out.push(Sm {
            atlas: match b.try_pe(&format!("{path}.atlas")) {
                Some(a) => vec![a],
                None => (0..)
                    .map_while(|i| b.try_pe(&format!("{path}.atlas{i}")))
                    .collect(),
            },
            muon: b.try_pe(&format!("{path}.muon")),
            path,
            nu,
            gpc,
            dma,
            smem,
            sync: b.sync_only(nu),
        });
    }
    out.sort_by_key(|s| (s.gpc, s.nu));
    out
}

/// `count` SMs, packed (in order) or spread round-robin over the gpcs.
pub(crate) fn pick(all: &[Sm], count: usize, placement: &str) -> Result<Vec<Sm>, String> {
    if count == 0 || count > all.len() {
        return Err(format!("sms = {count}: the machine has {} SMs", all.len()));
    }
    match placement {
        "packed" => Ok(all[..count].to_vec()),
        "spread" => {
            let mut gpcs: Vec<usize> = all.iter().map(|s| s.gpc).collect();
            gpcs.dedup();
            let by: Vec<Vec<&Sm>> = gpcs
                .iter()
                .map(|g| all.iter().filter(|s| s.gpc == *g).collect())
                .collect();
            let mut out = Vec::new();
            let mut i = 0;
            while out.len() < count {
                let g = &by[i % by.len()];
                if let Some(s) = g.get(i / by.len()) {
                    out.push((*s).clone());
                }
                i += 1;
            }
            Ok(out)
        }
        p => Err(format!("unknown placement {p:?} (packed, spread)")),
    }
}

/// The base of `s`'s sync-only range: joins (flags) go there (a join
/// writes no data; Neutrino spec Section 2.4).
fn sync_of(s: &Sm) -> Result<u64, String> {
    s.sync
        .map(|r| r.base)
        .ok_or_else(|| format!("{}: no sync-only range for its flags", s.path))
}

/// The range a task writes for `bytes` at `base`: at least one granule.
fn dst_range(base: u64, bytes: u64) -> Range {
    Range::from_len(base, bytes.max(1).div_ceil(GRANULE) * GRANULE)
}

/// Builds the run of `p` on `b`.
pub fn build(b: &Built, p: Params) -> Result<Run, String> {
    let all = sms_of(b);
    if all.is_empty() {
        return Err("micro: no SM (a node with smem and dma)".into());
    }
    let profiles = b.sim.profiles();
    let op = |name: &str| {
        profiles
            .lookup(name)
            .ok_or_else(|| format!("micro: no {name} profile"))
    };
    let mut versions: HashMap<u64, u64> = HashMap::new();
    let mut items: Vec<Item> = Vec::new();
    let mut metas: Vec<Meta> = Vec::new();
    // A task on a versioned range: its version is the next of the range.
    let mut push = |items: &mut Vec<Item>,
                    metas: &mut Vec<Meta>,
                    mut desc: Descriptor,
                    window: usize,
                    meta: Meta| {
        let base = desc.dst.base;
        let v = versions.entry(base).or_insert(0);
        *v += 1;
        desc.versioned = true;
        desc.inferred_version = Some(*v);
        desc.label = Some(items.len() as u64);
        items.push(Item {
            desc,
            window,
            done: (base, *v),
            fence: false,
            wait: true,
        });
        metas.push(meta);
    };
    let region = |name: &str| {
        b.try_memory(name)
            .ok_or_else(|| format!("micro: no memory {name:?}"))
    };
    let root_dma = b.try_pe("root.dma");
    let half = |s: &Sm| s.smem.len() / 2;
    let mut limit = usize::MAX;
    let mut last = Vec::new();
    let sms;
    let mut p = p;
    if p.driver.is_empty() {
        p.driver = match p.mode.as_str() {
            "stream" | "untracked" => "sms",
            "tmem" => "sm",
            _ => "root",
        }
        .into();
    }
    let per_sm = matches!(
        p.mode.as_str(),
        "stream" | "latency" | "atlas" | "muon" | "untracked" | "tmem"
    );
    let (mut macs, mut rate) = (0u64, 0f64);
    let mut signals = Vec::new();
    let driver = match p.driver.as_str() {
        "root" | "sms" if p.driver == "root" || per_sm => NeutrinoId(0),
        "sm" => {
            if p.sms != 1 && p.mode != "latency" {
                return Err("driver = sm runs one SM's tasks (sms = 1)".into());
            }
            pick(&all, 1, &p.placement)?[0].nu
        }
        d => {
            return Err(format!(
                "driver {d:?} for mode {:?} (root, sm, sms)",
                p.mode
            ))
        }
    };
    match p.mode.as_str() {
        "stream" | "latency" => {
            let (count, bytes, chunk, inflight) = if p.mode == "latency" {
                (1, p.bytes, p.bytes, 1)
            } else {
                (p.sms, p.bytes, p.chunk, p.inflight)
            };
            if chunk == 0 || bytes == 0 || inflight == 0 {
                return Err("bytes, chunk and inflight are positive".into());
            }
            sms = pick(&all, count, &p.placement)?;
            let slot = chunk.div_ceil(GRANULE) * GRANULE;
            let dma = op("dma")?;
            // The slots: all of the SMEM against a device memory, its first
            // half when the other half is the far side (smem, rsmem_*).
            let device = p.mem == "l2" || p.mem == "dram";
            for (j, s) in sms.iter().enumerate() {
                let room = if device { s.smem.len() } else { half(s) };
                if slot * inflight as u64 > room {
                    return Err(format!(
                        "{}: {inflight} slots of {chunk} B exceed {room} B of its SMEM",
                        s.path
                    ));
                }
                // The peer of a remote SMEM: the next SM of the gpc, or the
                // same SM of the next gpc.
                let peer = || -> Result<&Sm, String> {
                    let same: Vec<&Sm> = all.iter().filter(|x| x.gpc == s.gpc).collect();
                    let k = same.iter().position(|x| x.nu == s.nu).expect("own SM");
                    match p.mem.as_str() {
                        "rsmem_gpc" if same.len() > 1 => Ok(same[(k + 1) % same.len()]),
                        "rsmem_far" => {
                            let mut gpcs: Vec<usize> = all.iter().map(|x| x.gpc).collect();
                            gpcs.dedup();
                            let gi = gpcs.iter().position(|g| *g == s.gpc).expect("gpc");
                            let other = gpcs[(gi + 1) % gpcs.len()];
                            let far: Vec<&Sm> = all.iter().filter(|x| x.gpc == other).collect();
                            if other == s.gpc {
                                return Err("rsmem_far needs two gpcs".into());
                            }
                            Ok(far[k % far.len()])
                        }
                        m => Err(format!("{m}: no peer SM")),
                    }
                };
                let n = bytes.div_ceil(chunk);
                for i in 0..n {
                    let len = chunk.min(bytes - i * chunk);
                    let own_slot = s.smem.base + (i % inflight as u64) * slot;
                    // The far side: (address, its owner's DMA engine).
                    let (far_addr, far_dma, far_slot) = match p.mem.as_str() {
                        "l2" | "dram" => {
                            let r = region(if p.mem == "l2" { &p.l2 } else { &p.dram })?;
                            let span = r.len() / slot * slot;
                            let read = (j as u64 * bytes + i * chunk) % span;
                            let write = ((j * inflight) as u64 * slot
                                + (i % inflight as u64) * slot)
                                % span;
                            (r.base + read, root_dma, r.base + write)
                        }
                        "smem" => {
                            let off = half(s) + (i % inflight as u64) * slot;
                            (
                                s.smem.base + half(s) + (i * chunk) % half(s),
                                Some(s.dma),
                                s.smem.base + off,
                            )
                        }
                        "rsmem_gpc" | "rsmem_far" => {
                            let q = peer()?;
                            let off = half(q) + (i % inflight as u64) * slot;
                            (
                                q.smem.base + half(q) + (i * chunk) % half(q),
                                Some(q.dma),
                                q.smem.base + off,
                            )
                        }
                        m => {
                            return Err(format!(
                                "unknown mem {m:?} (l2, dram, smem, rsmem_gpc, rsmem_far)"
                            ))
                        }
                    };
                    let (src, dst, pe) = match p.dir.as_str() {
                        "read" => (far_addr, own_slot, s.dma),
                        "write" => {
                            let Some(e) = far_dma else {
                                return Err(
                                    "a write into the device's memory needs a copy engine root.dma \
                                     (a dma on the device, master sbus)"
                                        .into(),
                                );
                            };
                            (own_slot, far_slot, e)
                        }
                        d => return Err(format!("unknown dir {d:?} (read, write)")),
                    };
                    let desc = Descriptor {
                        opcode: dma,
                        pe: Some(pe),
                        dst: dst_range(dst, len),
                        versioned: true,
                        untracked: false,
                        dst_mask: nu_model::FULL_MASK,
                        inferred_version: None,
                        pc: None,
                        deps: Vec::new(),
                        label: None,
                        args: Args::Dma {
                            src: Addr(src),
                            dst: Addr(dst),
                            bytes: len,
                            tile_bytes: len,
                        },
                    };
                    push(
                        &mut items,
                        &mut metas,
                        desc,
                        j,
                        Meta {
                            role: Role::Data,
                            sm: j,
                            bytes: len,
                        },
                    );
                }
            }
            // Issue round-robin over the SMs (transfer i of every SM, then
            // i + 1), so that the driver's wait on one SM's window does not
            // hold back the others. Each range keeps its version order.
            let per = bytes.div_ceil(chunk) as usize;
            let mut order: Vec<usize> = (0..items.len()).collect();
            order.sort_by_key(|&x| (x % per, x / per));
            let mut slots: Vec<Option<(Item, Meta)>> =
                items.drain(..).zip(metas.drain(..)).map(Some).collect();
            for (l, x) in order.into_iter().enumerate() {
                let (mut it, m) = slots[x].take().expect("each once");
                it.desc.label = Some(l as u64);
                items.push(it);
                metas.push(m);
            }
            limit = inflight;
        }
        "atlas" | "muon" => {
            sms = pick(&all, p.sms, &p.placement)?;
            for (j, s) in sms.iter().enumerate() {
                // atlas: one matmul per tensor unit, each with its own C in
                // the SMEM; muon: one vector task.
                let mut descs = Vec::new();
                if p.mode == "atlas" {
                    let d = parse_dtype(&p.dtype)?;
                    if s.atlas.is_empty() {
                        return Err(format!("{}: no atlas", s.path));
                    }
                    let c = p.m * p.n * Dtype::Bf16.bytes();
                    let slot = c.div_ceil(GRANULE) * GRANULE;
                    if slot * s.atlas.len() as u64 > s.smem.len() {
                        return Err(format!(
                            "{}: C of {c} B for each of {} tensor units exceeds its SMEM",
                            s.path,
                            s.atlas.len()
                        ));
                    }
                    for (u, &unit) in s.atlas.iter().enumerate() {
                        descs.push(Descriptor {
                            opcode: op("atlas.matmul")?,
                            pe: Some(unit),
                            dst: dst_range(s.smem.base + u as u64 * slot, c),
                            versioned: true,
                            untracked: false,
                            dst_mask: nu_model::FULL_MASK,
                            inferred_version: None,
                            pc: None,
                            deps: Vec::new(),
                            label: None,
                            args: Args::Matmul {
                                a: Addr(s.smem.base),
                                b: Addr(s.smem.base),
                                a2: None,
                                b2: None,
                                lda: p.k_or(1024),
                                ldb: p.n,
                                ldc: p.n,
                                m: p.m,
                                n: p.n,
                                k: p.k_or(1024),
                                in_dtype: d,
                                b_dtype: d,
                                out_dtype: Dtype::Bf16,
                                accumulate: false,
                                keep: false,
                                tile: p.tile,
                            },
                        });
                    }
                } else {
                    descs.push(Descriptor {
                        opcode: op("muon.vector")?,
                        pe: Some(s.muon.ok_or_else(|| format!("{}: no muon", s.path))?),
                        dst: dst_range(s.smem.base, p.bytes_out),
                        versioned: true,
                        untracked: false,
                        dst_mask: nu_model::FULL_MASK,
                        inferred_version: None,
                        pc: None,
                        deps: Vec::new(),
                        label: None,
                        args: Args::Vector {
                            ops: p.lane_ops,
                            bytes_in: p.bytes,
                            bytes_out: p.bytes_out,
                            src: None,
                        },
                    });
                }
                let bytes = if p.mode == "muon" {
                    p.bytes + p.bytes_out
                } else {
                    0
                };
                for desc in descs {
                    push(
                        &mut items,
                        &mut metas,
                        desc,
                        j,
                        Meta {
                            role: Role::Data,
                            sm: j,
                            bytes,
                        },
                    );
                }
            }
        }
        "pingpong" => {
            let a = all[0].clone();
            let b2 = match p.pair.as_str() {
                "same_gpc" => all.iter().find(|s| s.gpc == a.gpc && s.nu != a.nu),
                "other_gpc" => all.iter().find(|s| s.gpc != a.gpc),
                x => return Err(format!("unknown pair {x:?} (same_gpc, other_gpc)")),
            }
            .ok_or("pingpong: no second SM for this pair")?
            .clone();
            sms = vec![a, b2];
            let mut prev: Option<(u64, u64)> = None;
            for step in 0..2 * p.rounds.max(1) {
                let j = step % 2;
                let flag = sync_of(&sms[j])?;
                let deps: Vec<Dep> = prev
                    .iter()
                    .map(|&(addr, version)| Dep::Versioned { addr, version })
                    .collect();
                let mut desc = Descriptor::join(dst_range(flag, GRANULE), deps);
                desc.pe = None;
                let v = versions_peek(&items, flag) + 1;
                push(
                    &mut items,
                    &mut metas,
                    desc,
                    0,
                    Meta {
                        role: Role::Flag,
                        sm: j,
                        bytes: 0,
                    },
                );
                prev = Some((flag, v));
            }
            last = prev.into_iter().collect();
        }
        "barrier" => {
            sms = pick(&all, p.sms, &p.placement)?;
            let mut deps = Vec::new();
            for (j, s) in sms.iter().enumerate() {
                let flag = sync_of(s)?;
                let v = versions_peek(&items, flag) + 1;
                push(
                    &mut items,
                    &mut metas,
                    Descriptor::join(dst_range(flag, GRANULE), Vec::new()),
                    0,
                    Meta {
                        role: Role::Flag,
                        sm: j,
                        bytes: 0,
                    },
                );
                deps.push(Dep::Versioned {
                    addr: flag,
                    version: v,
                });
            }
            let sync = b
                .sync_only(NeutrinoId(0))
                .ok_or("barrier: the device has no sync space")?;
            let mut join = Descriptor::join(dst_range(sync.base, GRANULE), deps);
            join.versioned = false;
            let label = items.len() as u64;
            join.label = Some(label);
            metas.push(Meta {
                role: Role::Join,
                sm: usize::MAX,
                bytes: 0,
            });
            items.push(Item {
                desc: join,
                window: 0,
                done: (sync.base, 0),
                fence: false,
                wait: true,
            });
            last = vec![(sync.base, 0)];
        }
        "invoke" => {
            let targets = match p.targets.as_str() {
                "one" => 1,
                "many" => p.sms,
                t => return Err(format!("unknown targets {t:?} (one, many)")),
            };
            sms = pick(&all, targets, &p.placement)?;
            for i in 0..p.tasks {
                let j = i % sms.len();
                let flag = sync_of(&sms[j])?;
                push(
                    &mut items,
                    &mut metas,
                    Descriptor::join(dst_range(flag, GRANULE), Vec::new()),
                    0,
                    Meta {
                        role: Role::Flag,
                        sm: j,
                        bytes: 0,
                    },
                );
            }
        }
        "untracked" => {
            if p.chunk == 0 || p.bytes == 0 {
                return Err("bytes and chunk are positive".into());
            }
            sms = pick(&all, p.sms, &p.placement)?;
            let dma = op("dma")?;
            let topo = b.sim.topology();
            let root_sync = b
                .sync_only(NeutrinoId(0))
                .ok_or("untracked: the device has no sync space")?;
            let slot = p.chunk.div_ceil(GRANULE) * GRANULE;
            for (j, s) in sms.iter().enumerate() {
                let gpc = topo.node(s.nu).parent.ok_or("untracked: an SM without a gpc")?;
                let dram = region(&format!("{}.hbm", topo.path_name(gpc)))?;
                let span = dram.len() / slot * slot;
                if slot > s.smem.len() || span == 0 {
                    return Err(format!("untracked: {} B chunks do not fit", p.chunk));
                }
                // Each SM writes its own part of the gpc's DRAM.
                let at = |i: u64| (j as u64 * p.bytes.div_ceil(slot) + i) * slot % span;
                for i in 0..p.bytes.div_ceil(p.chunk) {
                    let len = p.chunk.min(p.bytes - i * p.chunk);
                    let src = s.smem.base + i * slot % (s.smem.len() / slot * slot);
                    let dst = dram.base + at(i);
                    let desc = Descriptor {
                        opcode: dma,
                        pe: Some(s.dma),
                        dst: dst_range(dst, len),
                        versioned: false,
                        untracked: true,
                        dst_mask: nu_model::FULL_MASK,
                        inferred_version: None,
                        pc: None,
                        deps: Vec::new(),
                        label: Some(items.len() as u64),
                        args: Args::Dma {
                            src: Addr(src),
                            dst: Addr(dst),
                            bytes: len,
                            tile_bytes: len,
                        },
                    };
                    items.push(Item {
                        desc,
                        window: j,
                        done: (0, 0),
                        fence: false,
                        wait: false,
                    });
                    metas.push(Meta {
                        role: Role::Data,
                        sm: j,
                        bytes: len,
                    });
                }
                // The signal: after a fence, a forwarded join into the
                // device's sync space (one granule per SM, after the
                // gpcs' spawn dones).
                let signal = root_sync.base + (1024 + j as u64) * GRANULE;
                if signal + GRANULE > root_sync.bound {
                    return Err("untracked: the device's sync space is too small".into());
                }
                let mut join = Descriptor::join(dst_range(signal, GRANULE), Vec::new());
                join.label = Some(items.len() as u64);
                items.push(Item {
                    desc: join,
                    window: j,
                    done: (signal, 0),
                    fence: true,
                    wait: false,
                });
                metas.push(Meta {
                    role: Role::Flag,
                    sm: j,
                    bytes: 0,
                });
                signals.push(signal);
            }
        }
        "tmem" => {
            let s = pick(&all, 1, &p.placement)?.remove(0);
            sms = vec![s.clone()];
            let atlas = *s.atlas.first().ok_or_else(|| format!("{}: no atlas", s.path))?;
            let a_t = parse_dtype(&p.dtype)?;
            let b_t = if p.b_dtype.is_empty() {
                a_t
            } else {
                parse_dtype(&p.b_dtype)?
            };
            let (k, kc) = (p.k_or(2048), p.kc);
            if kc == 0 || !k.is_multiple_of(kc) || p.stages == 0 || p.tiles == 0 {
                return Err("tmem: kc divides k, and stages and tiles are positive".into());
            }
            let banks_of = |name: &str| -> Vec<Range> {
                let mut v: Vec<Range> = b
                    .sim
                    .memories()
                    .iter()
                    .filter(|m| m.name.starts_with(&format!("{}.{name}.b", s.path)))
                    .map(|m| m.range)
                    .collect();
                v.sort_by_key(|r| r.base);
                v
            };
            let tb = banks_of("tmem");
            if tb.len() != 4 {
                return Err(format!("tmem: {} needs a tmem of four banks", s.path));
            }
            let sb = banks_of("smem");
            let (s0, s1) = match sb.as_slice() {
                [x, y, ..] => (*x, *y),
                _ => (s.smem, s.smem),
            };
            // The tmem layout (output stationary, 2026-10-21): the A stages
            // only in the A banks (0, then 2) and the B stages only in the B
            // banks (1, then 3), from each bank's base; two C buffers in the
            // output type (one tile drains while the next accumulates) in
            // what is left (banks 2, 3, 0, 1 in turn).
            const T: u64 = 128;
            let a_bytes = a_t.bytes_for(T * kc);
            let b_bytes = b_t.bytes_for(kc * T);
            let a_slot = a_bytes.div_ceil(GRANULE) * GRANULE;
            let b_slot = b_bytes.div_ceil(GRANULE) * GRANULE;
            let c_bytes = T * T * Dtype::Bf16.bytes();
            let mut used = [0u64; 4];
            let mut place = |what: &str, banks: &[usize], slot: u64, n: usize| {
                let mut at = Vec::with_capacity(n);
                for _ in 0..n {
                    let b = banks
                        .iter()
                        .copied()
                        .find(|&b| used[b] + slot <= tb[b].len())
                        .ok_or_else(|| {
                            format!(
                                "tmem: {what} ({n} of {slot} B) do not fit in banks {banks:?} \
                                 (k = {k}, kc = {kc}, stages = {})",
                                p.stages
                            )
                        })?;
                    at.push(tb[b].base + used[b]);
                    used[b] += slot;
                }
                Ok::<_, String>(at)
            };
            let a_at = place("the A stages", &[0, 2], a_slot, p.stages)?;
            let b_at = place("the B stages", &[1, 3], b_slot, p.stages)?;
            let c_bufs = place("the C buffers", &[2, 3, 0, 1], c_bytes, 2)?;
            let fit = |what: &str, need: u64, room: u64| {
                if need > room {
                    Err(format!("tmem: {what} of {need} B does not fit in {room} B"))
                } else {
                    Ok(())
                }
            };
            // The sources: one fixed range in SMEM bank 0 (A) and bank 1
            // (B), the same for every chunk (time, not data); each C
            // buffer's copy goes to SMEM bank 0 after the A source.
            let out_bytes = T * T * Dtype::Bf16.bytes();
            let outs: Vec<u64> = (0..c_bufs.len() as u64)
                .map(|i| s0.base + a_slot + i * out_bytes.div_ceil(GRANULE) * GRANULE)
                .collect();
            fit(
                "the A source and the C copies",
                outs[outs.len() - 1] + out_bytes - s0.base,
                s0.len(),
            )?;
            fit("the B source", b_slot, s1.len())?;
            let (dma, mm) = (op("dma")?, op("atlas.matmul")?);
            let key = |items: &[Item], base: u64| Dep::Versioned {
                addr: base,
                version: versions_peek(items, base),
            };
            let copy = |src: u64, dst: u64, bytes: u64, deps: Vec<Dep>| Descriptor {
                opcode: dma,
                pe: Some(s.dma),
                dst: dst_range(dst, bytes),
                versioned: true,
                untracked: false,
                dst_mask: nu_model::FULL_MASK,
                inferred_version: None,
                pc: None,
                deps,
                label: None,
                args: Args::Dma {
                    src: Addr(src),
                    dst: Addr(dst),
                    bytes,
                    tile_bytes: bytes,
                },
            };
            // Per stage: the matmul that last read it; per C buffer: the
            // copy that last read it.
            let mut stage_reader: Vec<Option<Dep>> = vec![None; p.stages];
            let mut c_reader: Vec<Option<Dep>> = vec![None; c_bufs.len()];
            let mut i = 0usize;
            for t in 0..p.tiles {
                let cb = t % c_bufs.len();
                let c = c_bufs[cb];
                let mut prev: Option<Dep> = None;
                for x in 0..k / kc {
                    let st = i % p.stages;
                    let (a_at, b_at) = (a_at[st], b_at[st]);
                    let war: Vec<Dep> = stage_reader[st].into_iter().collect();
                    let role = |r| Meta {
                        role: r,
                        sm: 0,
                        bytes: 0,
                    };
                    push(
                        &mut items,
                        &mut metas,
                        copy(s0.base, a_at, a_bytes, war.clone()),
                        0,
                        Meta {
                            bytes: a_bytes,
                            ..role(Role::Data)
                        },
                    );
                    let ka = key(&items, a_at);
                    push(
                        &mut items,
                        &mut metas,
                        copy(s1.base, b_at, b_bytes, war),
                        0,
                        Meta {
                            bytes: b_bytes,
                            ..role(Role::Data)
                        },
                    );
                    let kb = key(&items, b_at);
                    let mut deps = vec![ka, kb];
                    deps.extend(prev);
                    if x == 0 {
                        deps.extend(c_reader[cb]);
                    }
                    let last = x + 1 == k / kc;
                    let desc = Descriptor {
                        opcode: mm,
                        pe: Some(atlas),
                        dst: dst_range(c, c_bytes),
                        versioned: true,
                        untracked: false,
                        dst_mask: nu_model::FULL_MASK,
                        inferred_version: None,
                        pc: None,
                        deps,
                        label: None,
                        args: Args::Matmul {
                            a: Addr(a_at),
                            b: Addr(b_at),
                            a2: None,
                            b2: None,
                            lda: kc,
                            ldb: T,
                            ldc: T,
                            m: T,
                            n: T,
                            k: kc,
                            in_dtype: a_t,
                            b_dtype: b_t,
                            out_dtype: Dtype::Bf16,
                            accumulate: x > 0,
                            keep: !last,
                            tile: [T, T],
                        },
                    };
                    push(&mut items, &mut metas, desc, 0, role(Role::Matmul));
                    let km = key(&items, c);
                    stage_reader[st] = Some(km);
                    prev = Some(km);
                    i += 1;
                    if last {
                        push(
                            &mut items,
                            &mut metas,
                            copy(c, outs[cb], out_bytes, vec![km]),
                            0,
                            Meta {
                                bytes: out_bytes,
                                ..role(Role::Copy)
                            },
                        );
                        c_reader[cb] = Some(key(&items, outs[cb]));
                    }
                }
            }
            macs = p.tiles as u64 * T * T * k;
            rate = match &b.sim.pe_config(atlas).params {
                sim_core::PeParams::Atlas { rates, .. } => {
                    let r = |d: Dtype| {
                        rates
                            .iter()
                            .find(|x| x.0 == d)
                            .map_or(0.0, |x| x.1.num as f64 / x.1.den as f64)
                    };
                    r(a_t).min(r(b_t))
                }
                _ => 0.0,
            };
        }
        m => {
            return Err(format!(
                "unknown mode {m:?} (stream, latency, atlas, muon, pingpong, barrier, invoke, untracked, tmem)"
            ))
        }
    }
    // driver = sms: the spawn tree over the gpcs of the run's SMs.
    let tree = if p.driver == "sms" {
        let topo = b.sim.topology();
        let root_sync = b
            .sync_only(NeutrinoId(0))
            .ok_or("driver = sms: the device has no sync space")?;
        let mut gpcs: Vec<Gpc> = Vec::new();
        for (j, s) in sms.iter().enumerate() {
            let g = topo
                .node(s.nu)
                .parent
                .ok_or("driver = sms: an SM without a gpc")?;
            if topo.node(g).parent != Some(NeutrinoId(0)) {
                return Err("driver = sms: an SM's parent is not a child of the device".into());
            }
            match gpcs.iter_mut().find(|x| x.0 == g) {
                Some(x) => x.2.push((j, s.nu)),
                None => {
                    let sync = b.sync_only(g).ok_or_else(|| {
                        format!("driver = sms: {} has no sync space", topo.path_name(g))
                    })?;
                    gpcs.push((g, sync, vec![(j, s.nu)]));
                }
            }
        }
        Some((root_sync, gpcs))
    } else {
        None
    };
    let report = Rc::new(RefCell::new(Report {
        params: Some(p),
        sms,
        metas,
        macs,
        rate,
        ..Report::default()
    }));
    Ok(Run {
        items,
        limit,
        last,
        report,
        driver,
        tree,
        signals,
    })
}

/// The latest version of `base` among `items` (0: none).
fn versions_peek(items: &[Item], base: u64) -> u64 {
    items
        .iter()
        .rev()
        .find(|i| i.done.0 == base)
        .map_or(0, |i| i.done.1)
}

/// Invokes `items` in order, at most `limit` open per window, then waits
/// for `last` (the latest version of every range when empty); returns the
/// cycle after each invoke call.
async fn issue(ctx: &NuCtx, items: Vec<Item>, limit: usize, last: Vec<(u64, u64)>) -> Vec<u64> {
    let mut open: HashMap<usize, VecDeque<(u64, u64)>> = HashMap::new();
    let mut invoked_at = Vec::with_capacity(items.len());
    let mut latest: HashMap<u64, u64> = HashMap::new();
    for it in items {
        if it.fence {
            ctx.fence().await;
        }
        if it.desc.untracked {
            ctx.invoke_untracked(it.desc).await;
            invoked_at.push(ctx.now());
            continue;
        }
        let q = open.entry(it.window).or_default();
        if q.len() >= limit {
            let (a, v) = q.pop_front().expect("open task");
            ctx.wait(a, v).await;
        }
        ctx.invoke(it.desc).await;
        invoked_at.push(ctx.now());
        if it.wait {
            open.entry(it.window).or_default().push_back(it.done);
            latest.insert(it.done.0, it.done.1);
        }
    }
    let waits = if last.is_empty() {
        let mut w: Vec<(u64, u64)> = latest.into_iter().collect();
        w.sort_unstable();
        w
    } else {
        last
    };
    for (a, v) in waits {
        if v == 0 {
            ctx.wait_unversioned(a).await;
        } else {
            ctx.wait(a, v).await;
        }
    }
    invoked_at
}

impl Run {
    /// The driver program: one program issues every task, or (`driver =
    /// sms`) the device starts the gpcs' programs, which start the SMs'.
    pub async fn run(self, ctx: NuCtx) {
        let start = ctx.now();
        let invoked_at = match self.tree {
            None => issue(&ctx, self.items, self.limit, self.last).await,
            Some((root_sync, gpcs)) => {
                // Each SM's tasks (its window), in order.
                let mut per: HashMap<usize, Vec<Item>> = HashMap::new();
                for it in self.items {
                    per.entry(it.window).or_default().push(it);
                }
                let limit = self.limit;
                let mut gpc_dones = Vec::new();
                for (k, (g, sync, sms)) in gpcs.into_iter().enumerate() {
                    let lists: Vec<(NeutrinoId, Vec<Item>)> = sms
                        .into_iter()
                        .map(|(j, nu)| (nu, per.remove(&j).unwrap_or_default()))
                        .collect();
                    let done = Range::from_len(root_sync.base + k as u64 * GRANULE, GRANULE);
                    let gpc = move |gctx: NuCtx| async move {
                        let mut dones = Vec::new();
                        for (i, (nu, items)) in lists.into_iter().enumerate() {
                            let d = Range::from_len(sync.base + i as u64 * GRANULE, GRANULE);
                            let sm = move |sctx: NuCtx| async move {
                                issue(&sctx, items, limit, Vec::new()).await;
                            };
                            gctx.spawn_named(nu, "micro_sm", d, vec![], sm).await;
                            dones.push(d.base);
                        }
                        for d in dones {
                            gctx.wait_unversioned(d).await;
                        }
                    };
                    ctx.spawn_named(g, "micro_gpc", done, vec![], gpc).await;
                    gpc_dones.push(done.base);
                }
                for a in self.signals {
                    ctx.wait_unversioned(a).await;
                }
                for d in gpc_dones {
                    ctx.wait_unversioned(d).await;
                }
                Vec::new()
            }
        };
        let mut r = self.report.borrow_mut();
        r.driver_start = start;
        r.driver_end = ctx.now();
        r.invoked_at = invoked_at;
    }
}

/// An SM's tasks in `result`: first chunk start, last chunk end, task
/// count, and each task's invoke to completion.
type Span = (u64, u64, u64, Vec<u64>);

/// The measured quantities of the run (the `micro` object of `--stats`);
/// `stats` needs the task timeline.
pub fn result(r: &Report, stats: &Stats) -> Value {
    let p = r.params.clone().unwrap_or_default();
    let by_label: HashMap<u64, &sim_core::TaskTimeline> = stats
        .tasks
        .iter()
        .filter_map(|t| t.label.map(|l| (l, t)))
        .collect();
    let task = |i: usize| by_label.get(&(i as u64)).copied();
    let first_start = |t: &sim_core::TaskTimeline| t.chunks.first().map(|c| c.start);
    let last_end = |t: &sim_core::TaskTimeline| t.chunks.last().map(|c| c.end);
    let driver_cycles = r.driver_end - r.driver_start;
    let res: HashMap<&str, &sim_core::ResourceStats> = stats
        .resources
        .iter()
        .map(|x| (x.name.as_str(), x))
        .collect();
    let mut culprits: Vec<(&str, u64)> = stats
        .resources
        .iter()
        .filter(|x| x.culprit_cycles > 0)
        .map(|x| (x.name.as_str(), x.culprit_cycles))
        .collect();
    culprits.sort_by_key(|c| std::cmp::Reverse(c.1));
    culprits.truncate(5);
    let sm_paths: Vec<&str> = r.sms.iter().map(|s| s.path.as_str()).collect();
    let mut out = json!({
        "mode": p.mode,
        "params": p,
        "sms": sm_paths,
        "driver_cycles": driver_cycles,
        "culprits": culprits,
    });
    let data: Vec<(usize, &Meta)> = r
        .metas
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::Data)
        .collect();
    match p.mode.as_str() {
        "stream" => {
            let (mut lo, mut hi, mut total) = (u64::MAX, 0u64, 0u64);
            let mut per: Vec<(u64, u64, u64)> = vec![(u64::MAX, 0, 0); r.sms.len()];
            for &(i, m) in &data {
                let Some(t) = task(i) else { continue };
                let (Some(s), Some(e)) = (first_start(t), last_end(t)) else {
                    continue;
                };
                lo = lo.min(s);
                hi = hi.max(e);
                total += m.bytes;
                let q = &mut per[m.sm];
                *q = (q.0.min(s), q.1.max(e), q.2 + m.bytes);
            }
            let cycles = hi.saturating_sub(lo);
            let per_sm: Vec<f64> = per
                .iter()
                .map(|&(s, e, b)| b as f64 / e.saturating_sub(s).max(1) as f64)
                .collect();
            out["bytes"] = json!(total);
            out["cycles"] = json!(cycles);
            out["bytes_per_cycle"] = json!(total as f64 / cycles.max(1) as f64);
            out["bytes_per_cycle_per_sm"] = json!(per_sm);
            out["bytes_per_cycle_per_sm_mean"] =
                json!(per_sm.iter().sum::<f64>() / per_sm.len().max(1) as f64);
            out["driver_bytes_per_cycle"] = json!(total as f64 / driver_cycles.max(1) as f64);
        }
        "latency" => {
            if let Some(t) = data.first().and_then(|&(i, _)| task(i)) {
                out["bytes"] = json!(p.bytes);
                out["invoke_to_complete"] = json!(t.complete.map(|c| c - t.invoke));
                out["issue_to_last_chunk_end"] =
                    json!(t.issue.zip(last_end(t)).map(|(i, e)| e - i));
                out["driver_cycles"] = json!(driver_cycles);
            }
        }
        "atlas" | "muon" => {
            // Per SM: the span from its first chunk start to its last chunk
            // end over its tasks (atlas: one per tensor unit).
            let mut per: Vec<Option<Span>> = vec![None; r.sms.len()];
            for &(i, m) in &data {
                let Some(t) = task(i) else { continue };
                let (Some(s), Some(e)) = (first_start(t), last_end(t)) else {
                    continue;
                };
                let ic = t.complete.map(|c| c - t.invoke);
                let q = per[m.sm].get_or_insert((u64::MAX, 0, 0, Vec::new()));
                q.0 = q.0.min(s);
                q.1 = q.1.max(e);
                q.2 += 1;
                q.3.extend(ic);
            }
            let mut rows = Vec::new();
            for (j, x) in per.iter().enumerate() {
                let Some((s, e, n, ics)) = x else { continue };
                let cycles = e - s;
                let sm = &r.sms[j];
                let port = res.get(format!("{}.smem.port", sm.path).as_str());
                let share = port.map(|x| {
                    // The port's lane-cycles over the SM's window.
                    x.busy_slot_cycles / (x.slots.max(1e-9) * cycles.max(1) as f64)
                });
                let mut row = json!({
                    "sm": sm.path,
                    "tasks": n,
                    "cycles": cycles,
                    "invoke_to_complete": ics.iter().max(),
                    "smem_port_share": share,
                });
                if p.mode == "atlas" {
                    row["mac_per_cycle"] =
                        json!((n * p.m * p.n * p.k_or(1024)) as f64 / cycles.max(1) as f64);
                }
                rows.push(row);
            }
            let mean = |k: &str| {
                rows.iter().filter_map(|x| x[k].as_f64()).sum::<f64>() / rows.len().max(1) as f64
            };
            if p.mode == "atlas" {
                out["mac_per_cycle_per_sm"] = json!(mean("mac_per_cycle"));
            }
            out["cycles_mean"] = json!(mean("cycles"));
            out["smem_port_share_mean"] = json!(mean("smem_port_share"));
            out["per_sm"] = Value::Array(rows);
        }
        "pingpong" => {
            let steps: Vec<&sim_core::TaskTimeline> = (0..r.metas.len()).filter_map(task).collect();
            // One way: from a flag's completion to the next task's issue
            // (its dependency resolved at the other SM), and the mean step.
            let hops: Vec<u64> = steps
                .windows(2)
                .filter_map(|w| Some(w[1].issue? - w[0].complete?))
                .collect();
            let span = steps
                .last()
                .and_then(|l| l.complete)
                .zip(steps.first().and_then(|f| f.complete))
                .map(|(l, f)| l - f);
            out["rounds"] = json!(p.rounds);
            out["one_way_cycles"] =
                json!(hops.iter().sum::<u64>() as f64 / hops.len().max(1) as f64);
            out["step_cycles"] = json!(span.map(|s| s as f64 / (steps.len().max(2) - 1) as f64));
        }
        "barrier" => {
            let flags: Vec<&sim_core::TaskTimeline> = r
                .metas
                .iter()
                .enumerate()
                .filter(|(_, m)| m.role == Role::Flag)
                .filter_map(|(i, _)| task(i))
                .collect();
            let last = flags.iter().filter_map(|t| t.complete).max();
            let join = r
                .metas
                .iter()
                .position(|m| m.role == Role::Join)
                .and_then(task);
            out["n"] = json!(flags.len());
            out["last_task_end"] = json!(last);
            if let (Some(l), Some(j)) = (last, join) {
                out["to_join_ready"] = json!(j.ready.map(|x| x.saturating_sub(l)));
                out["to_join_complete"] = json!(j.complete.map(|x| x.saturating_sub(l)));
            }
        }
        "invoke" => {
            let entries: Vec<u64> = (0..r.metas.len())
                .filter_map(task)
                .map(|t| t.invoke)
                .collect();
            let n = entries.len() as u64;
            let (lo, hi) = (entries.iter().min(), entries.iter().max());
            out["tasks"] = json!(n);
            out["driver_cycles_per_task"] = json!(driver_cycles as f64 / n.max(1) as f64);
            out["entry_cycles_per_task"] = json!(lo
                .zip(hi)
                .map(|(l, h)| (h - l) as f64 / (n.max(2) - 1) as f64));
            let calls = &r.invoked_at;
            if calls.len() > 1 {
                out["call_cycles_per_task"] =
                    json!((calls[calls.len() - 1] - calls[0]) as f64 / (calls.len() - 1) as f64);
            }
        }
        "untracked" => {
            // Per SM: bytes, the first chunk start, the last chunk end
            // and the signal's completion at the device.
            let mut per = Vec::new();
            for j in 0..r.sms.len() {
                let mine = |role: Role| {
                    r.metas
                        .iter()
                        .enumerate()
                        .filter(move |(_, m)| m.role == role && m.sm == j)
                };
                let dma: Vec<&sim_core::TaskTimeline> =
                    mine(Role::Data).filter_map(|(i, _)| task(i)).collect();
                let bytes: u64 = mine(Role::Data).map(|(_, m)| m.bytes).sum();
                let start = dma.iter().filter_map(|t| first_start(t)).min();
                let end = dma.iter().filter_map(|t| last_end(t)).max();
                let signal = mine(Role::Flag)
                    .filter_map(|(i, _)| task(i))
                    .find_map(|t| t.complete);
                per.push(json!({
                    "sm": r.sms[j].path,
                    "bytes": bytes,
                    "first_start": start,
                    "last_end": end,
                    "signal_complete": signal,
                    "end_to_signal": end.zip(signal).map(|(e, s)| s.saturating_sub(e)),
                }));
            }
            let total: u64 = data.iter().map(|(_, m)| m.bytes).sum();
            out["bytes"] = json!(total);
            out["bytes_per_cycle"] = json!(total as f64 / driver_cycles.max(1) as f64);
            out["invoke_messages"] =
                json!(stats.messages_by_kind.get("Invoke").copied().unwrap_or(0));
            out["per_sm"] = Value::Array(per);
        }
        "tmem" => {
            let end = stats.end_time;
            out["end_cycles"] = json!(end);
            out["macs"] = json!(r.macs);
            out["atlas_rate"] = json!(r.rate);
            out["atlas_utilization"] = json!(r.macs as f64 / (r.rate * end.max(1) as f64));
            // The busiest port of the tmem banks.
            let sm = r.sms.first().map_or("", |x| x.path.as_str());
            let prefix = format!("{sm}.tmem.b");
            let util = |x: &sim_core::ResourceStats| {
                x.busy_slot_cycles / (x.slots.max(f64::MIN_POSITIVE) * end.max(1) as f64)
            };
            let busiest = stats
                .resources
                .iter()
                .filter(|x| x.name.starts_with(&prefix))
                .max_by(|x, y| util(x).total_cmp(&util(y)));
            if let Some(x) = busiest {
                out["tmem_busiest"] = json!(x.name);
                out["tmem_busiest_utilization"] = json!(util(x));
            }
            // Per tile: from the completion of its last matmul to the end of
            // its C copy.
            let mut tails = Vec::new();
            let mut last_mm: Option<u64> = None;
            for (i, m) in r.metas.iter().enumerate() {
                match m.role {
                    Role::Matmul => last_mm = task(i).and_then(|t| t.complete),
                    Role::Copy => {
                        let end = task(i).and_then(last_end);
                        if let (Some(a), Some(e)) = (last_mm, end) {
                            tails.push(e.saturating_sub(a));
                        }
                    }
                    _ => {}
                }
            }
            out["tile_tail_cycles"] = json!(tails);
        }
        _ => {}
    }
    out
}
