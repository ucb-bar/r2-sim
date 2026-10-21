//! The full pi0 forward pass (5b item 4): the vision encoder, the VLM
//! prefill and the action expert's denoise loop, run by one driver on the
//! home SM, one stage after the other.
//!
//! One HBM map over the slices: the action expert's weights and the
//! prefix KV cache (as [`adamoe::hbm_map`]), then the VLM weights, then
//! the vision weights, each slice in that order (the placement rule is in
//! [`crate::machine`]).
//! Prefill activations: VLM tensors, then vision tensors, in the
//! activation memory (cmem of the first cmem cluster when present, else
//! HBM). The vision projector writes the VLM input rows; the VLM writes the
//! KV the action expert reads. The action expert's cmem placement starts
//! after the prefill activations in that cluster.

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::{NeutrinoId, Range};
use serde::{Deserialize, Serialize};
use sim_core::NuCtx;

use crate::adamoe;
use crate::machine::Machine;
use crate::prefill::{self, Place};
use crate::versions::{VersionState, VresetPolicy};
use crate::ProgramFn;

/// Program parameters (a TOML file, every key optional).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Params {
    /// Weight type of all three stages: "bf16", "fp8" or "mxfp4".
    pub weight_dtype: String,
    /// The VLM's last layer: "no_ffn" (default: its FFN output feeds
    /// nothing), "kv" (only its K/V projection) or "full".
    pub vlm_last_layer: String,
    /// Prefill activation memory: "auto", "hbm" or "cmem".
    pub act_mem: String,
    /// A copy of the prefill activations in every cmem cluster.
    pub act_copies: bool,
    /// One accumulator per SM in the prefill GEMMs.
    pub single_accumulator: bool,
    /// When the driver resets (`vreset`) versioned ranges: "lru" (only when an
    /// owner's version table would overflow) or "barrier" (every range a
    /// window used, after its barrier).
    pub vreset_policy: String,
    /// Router activation probabilities for the action expert's cmem
    /// placement (the CLI fills it from the trace).
    pub expert_prob: Vec<Vec<f64>>,
    /// Vision encoder parameters. `model`, `weight_dtype`, `act_mem`,
    /// `act_copies`, `single_accumulator`, `smem_cost` and `action` come
    /// from this struct.
    pub vision: prefill::Params,
    /// VLM prefill parameters. `model`, `weight_dtype`, `act_mem`,
    /// `act_copies`, `single_accumulator`, `smem_cost`, `action` and
    /// `last_layer` come from this struct.
    pub vlm: prefill::Params,
    /// Action expert parameters. `weight_dtype`, `cmem_reserve` and a
    /// non-empty `expert_prob` come from this struct.
    pub action: adamoe::Params,
    /// "hierarchical" (the default): the device program starts, per
    /// stage, one driver per gpc on the gpc's Neutrino, each invoking the
    /// tasks of its gpc (Radiance 2 spec Section 6.2); "single": one driver on the home SM
    /// runs the three stages. "gpc" is a deprecated name of
    /// "hierarchical".
    pub drivers: String,
    /// With per-gpc drivers: who invokes a write into a gpc's slice or
    /// cmem: "owner" (the gpc that owns it) or "writer" (the gpc of the
    /// writing SM; a cross-gpc invoke when they differ).
    pub slice_writes: String,
    /// On a banked SMEM, the bank group that buffers of no operand role
    /// (accumulators, residual, statistics) try first: "act" (the
    /// activation banks) or "weights" (r2sim spec Section 12.3).
    pub smem_whole: String,
    /// The SMEM term of the prefill grid cost model: "role" (the default)
    /// or "port" (`prefill::Params::smem_cost`); copied to both prefill
    /// stages.
    pub smem_cost: String,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            weight_dtype: "bf16".into(),
            vlm_last_layer: "no_ffn".into(),
            act_mem: "auto".into(),
            act_copies: false,
            single_accumulator: true,
            vreset_policy: "lru".into(),
            expert_prob: Vec::new(),
            vision: prefill::Params {
                model: "vision".into(),
                ..prefill::Params::default()
            },
            vlm: prefill::Params::default(),
            action: adamoe::Params::default(),
            drivers: "hierarchical".into(),
            slice_writes: "owner".into(),
            smem_whole: "act".into(),
            smem_cost: "role".into(),
        }
    }
}

/// One stage of the pass.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Stage {
    /// Stage name: "vision", "vlm", "action" or "total".
    pub name: String,
    /// Start cycle.
    pub start: u64,
    /// End cycle.
    pub end: u64,
    /// Cycles (end - start).
    pub cycles: u64,
    /// HBM bytes of the stage (its report's `hbm_bytes`).
    pub hbm_bytes: u64,
    /// cmem bytes of the stage (its report's `cmem_bytes`).
    pub cmem_bytes: u64,
    /// The HBM bytes per gpc of the SM that moved them: from or to its own
    /// gpc's slice (`local`) or another slice (`remote`).
    pub hbm_by_gpc: Vec<crate::machine::GpcBytes>,
    /// Every weight byte the stage uses, once per use (the action expert
    /// uses its weights once per denoise step), whether it is in HBM or
    /// cmem.
    pub w_once_bytes: u64,
    /// w_once_bytes / HBM rate.
    pub weights_once_bound: u64,
    /// hbm_bytes / HBM rate: the stage's own traffic.
    pub own_traffic_bound: u64,
    /// Prefill stages: the sum of their GEMM phases' rooflines (attention
    /// not included). Action: its weights-once bound (its Atlas and SMEM
    /// floors are far lower at 51 tokens).
    pub roofline: u64,
}

/// One driver's tasks in one stage.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DriverStat {
    /// The Neutrino that ran the driver.
    pub nu: u32,
    /// "vision", "vlm" or "action".
    pub stage: String,
    /// Tasks it invoked.
    pub invoked: u64,
    /// Tasks it built (planned), its own and the other drivers'.
    pub planned: u64,
    /// Windows it ran.
    pub windows: u64,
}

/// What the forward pass measured, per stage, for the CLI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    /// The vision, VLM and action stages, in run order.
    pub stages: Vec<Stage>,
    /// The whole pass as one stage: byte sums, and the sum of the stage
    /// rooflines.
    pub total: Stage,
    /// HBM rate of all slices, bytes per cycle.
    pub hbm_rate: f64,
    /// Bytes of each cmem cluster kept for prefill activations.
    pub cmem_reserve: Vec<u64>,
    /// The vision encoder's prefill report.
    pub vision: Option<prefill::Report>,
    /// The VLM's prefill report.
    pub vlm: Option<prefill::Report>,
    /// Releases of data ranges the driver issued (local and remote; the
    /// least recently used, or all at the barrier in "barrier" mode).
    pub vreset_data: u64,
    /// Releases of one-off signals after their window's barrier.
    pub vreset_signal: u64,
    /// Per driver and stage: tasks invoked and planned, windows.
    pub drivers: Vec<DriverStat>,
    /// The parameters used.
    pub params: Option<Params>,
}

/// The three stage drivers of one program, and the version bookkeeping
/// they share.
struct Stages {
    /// The Neutrino the program runs on: the home SM, or a gpc.
    nu: NeutrinoId,
    vision: prefill::Driver,
    vlm: prefill::Driver,
    action: adamoe::Driver,
    versions: Rc<RefCell<VersionState>>,
}

/// The forward-pass driver: runs the vision, VLM and action-expert
/// drivers on the home SM, one after the other; or, with per-gpc drivers,
/// the device program that starts one driver per gpc and stage.
pub struct Driver {
    /// One entry (the home SM), or one per gpc.
    stages: Vec<Stages>,
    /// Per-gpc drivers: the device Neutrino, its sync space for the spawn
    /// tasks, and the own spaces of the gpcs (reset at every Start).
    root: Option<(NeutrinoId, Range, Vec<Range>)>,
    home: NeutrinoId,
    hbm_rate: f64,
    /// Action weight bytes per denoise step, and steps.
    action_w: (u64, u32),
    /// The per-stage report.
    pub report: Rc<RefCell<Report>>,
    /// The action expert's report.
    pub action_report: Rc<RefCell<adamoe::Report>>,
}

/// The three stage drivers on `m` in one HBM map, the cmem reserve and the
/// action weights per step.
#[allow(clippy::type_complexity)]
fn build(
    m: &Machine,
    p: &Params,
) -> Result<
    (
        prefill::Driver,
        prefill::Driver,
        adamoe::Driver,
        Vec<u64>,
        (u64, u32),
    ),
    String,
> {
    m.versions
        .borrow_mut()
        .set_mode(VresetPolicy::parse(&p.vreset_policy)?);
    let whole = match p.smem_whole.as_str() {
        "act" => crate::machine::Role::Act,
        "weights" => crate::machine::Role::W,
        w => return Err(format!("unknown smem_whole {w:?} (act, weights)")),
    };
    let mut m = m.clone();
    for s in &mut m.sms {
        s.whole = whole;
    }
    let m = &m;
    let mut vp = p.vision.clone();
    let mut lp = p.vlm.clone();
    let mut ap = p.action.clone();
    // Before the copies below: the prefill drivers compute the action
    // expert's HBM map (and so the KV cache address) from it.
    ap.weight_dtype = p.weight_dtype.clone();
    if !p.expert_prob.is_empty() {
        ap.expert_prob = p.expert_prob.clone();
    }
    vp.model = "vision".into();
    lp.model = "vlm".into();
    for x in [&mut vp, &mut lp] {
        x.weight_dtype = p.weight_dtype.clone();
        x.act_mem = p.act_mem.clone();
        x.act_copies = p.act_copies;
        x.single_accumulator = p.single_accumulator;
        x.smem_cost = p.smem_cost.clone();
        x.action = ap.clone();
    }
    lp.last_layer = p.vlm_last_layer.clone();
    let vlm = prefill::Driver::new_at(m.clone(), lp, Place::default())?;
    let vision = prefill::Driver::new_at(
        m.clone(),
        vp,
        Place {
            hbm_next: Some(vlm.hbm_end()),
            act_base: Some(vlm.act_end()),
            out_x: Some(vlm.x_in()),
        },
    )?;
    // The action expert's cmem placement starts after the prefill
    // activations, in every cmem block that holds a copy of them.
    let act = vision.act_mem();
    let used = vision.act_end().saturating_sub(act.base);
    let copies = vision.act_copy_mems();
    let reserve: Vec<u64> = m
        .cmem_nodes
        .iter()
        .map(|c| if copies.contains(&c.mem) { used } else { 0 })
        .collect();
    ap.cmem_reserve = reserve.clone();
    let action_w = (adamoe::hbm_map(m, &ap)?.weight_bytes, ap.steps);
    let action = adamoe::Driver::new(m.clone(), ap)?;
    if vlm.kv() != action.kv() {
        return Err(format!(
            "the VLM writes the KV cache at {:?}, the action expert reads {:?}",
            vlm.kv(),
            action.kv()
        ));
    }
    let home = action.home_nu();
    if vlm.home_nu() != home || vision.home_nu() != home {
        return Err("the three stages need the same home SM".into());
    }
    Ok((vision, vlm, action, reserve, action_w))
}

impl Driver {
    /// Builds the three stage drivers in one HBM map (one set per gpc with
    /// `drivers = "hierarchical"`, each with its own version bookkeeping). Errors
    /// when the VLM writes the KV cache at a layout other than the one the
    /// action expert reads, or when the stages have different home SMs.
    pub fn new(m: Machine, p: Params) -> Result<Self, String> {
        let hbm_rate = m.hbm_rate();
        let mut stages = Vec::new();
        let (root, home, reserve, action_w) = match p.drivers.as_str() {
            "single" => {
                let (vision, vlm, action, reserve, action_w) = build(&m, &p)?;
                let home = action.home_nu();
                stages.push(Stages {
                    nu: home,
                    vision,
                    vlm,
                    action,
                    versions: m.versions.clone(),
                });
                (None, home, reserve, action_w)
            }
            "hierarchical" | "gpc" => {
                if p.drivers == "gpc" {
                    eprintln!("forward: drivers = \"gpc\" is deprecated; use \"hierarchical\"");
                }
                let root_sync = m
                    .root_sync
                    .ok_or("per-gpc drivers need a sync space at the device")?;
                let mut reserve = Vec::new();
                let mut action_w = (0, 0);
                for g in 0..m.gpcs.len() {
                    let mg = m.with_own_versions();
                    let (mut vision, mut vlm, mut action, r, w) = build(&mg, &p)?;
                    vision.for_gpc(g)?;
                    vlm.for_gpc(g)?;
                    action.for_gpc(g);
                    match p.slice_writes.as_str() {
                        "owner" => {}
                        "writer" => {
                            let spaces: Vec<Range> = m.gpc_spaces.iter().map(|x| x.0).collect();
                            vision.writer_invokes(spaces.clone());
                            vlm.writer_invokes(spaces.clone());
                            action.writer_invokes(spaces);
                        }
                        w => return Err(format!("unknown slice_writes {w:?} (owner, writer)")),
                    }
                    (reserve, action_w) = (r, w);
                    stages.push(Stages {
                        nu: m.gpcs[g].0,
                        vision,
                        vlm,
                        action,
                        versions: mg.versions.clone(),
                    });
                }
                let spaces = m.gpc_spaces.iter().map(|x| x.0).collect();
                (
                    Some((NeutrinoId(0), root_sync, spaces)),
                    NeutrinoId(0),
                    reserve,
                    action_w,
                )
            }
            d => return Err(format!("unknown drivers {d:?} (hierarchical, single)")),
        };
        if stages.is_empty() {
            return Err("per-gpc drivers need gpcs".into());
        }
        let report = Rc::new(RefCell::new(Report {
            hbm_rate,
            cmem_reserve: reserve,
            params: Some(p),
            ..Report::default()
        }));
        let action_report = stages[0].action.report.clone();
        Ok(Self {
            stages,
            root,
            home,
            hbm_rate,
            action_w,
            report,
            action_report,
        })
    }

    /// The Neutrino that runs the program: the home SM, or the device with
    /// per-gpc drivers.
    pub fn home_nu(&self) -> NeutrinoId {
        self.home
    }

    /// The Neutrinos whose programs invoke the tasks.
    pub fn drivers(&self) -> Vec<NeutrinoId> {
        self.stages.iter().map(|s| s.nu).collect()
    }

    /// The home SM.
    pub fn home_sm(&self) -> NeutrinoId {
        self.stages[0].action.home_nu()
    }

    /// Runs the vision, VLM and action stages in order and fills `report`.
    pub async fn run(self, ctx: NuCtx) {
        let Self {
            stages,
            root,
            hbm_rate,
            action_w,
            report,
            action_report,
            ..
        } = self;
        let (vr, lr) = (
            stages[0].vision.report.clone(),
            stages[0].vlm.report.clone(),
        );
        let versions = stages[0].versions.clone();
        let reports: Vec<_> = stages
            .iter()
            .map(|s| {
                (
                    s.nu,
                    s.vision.report.clone(),
                    s.vlm.report.clone(),
                    s.action.report.clone(),
                )
            })
            .collect();
        let mut t = [ctx.now(); 4];
        match root {
            None => {
                let Stages {
                    vision,
                    vlm,
                    action,
                    ..
                } = stages.into_iter().next().expect("one driver");
                vision.run(ctx.clone()).await;
                ctx.mark("vision end").await;
                t[1] = ctx.now();
                vlm.run(ctx.clone()).await;
                ctx.mark("vlm end").await;
                t[2] = ctx.now();
                action.run(ctx.clone()).await;
                ctx.mark("action end").await;
                t[3] = ctx.now();
            }
            Some((_, sync, spaces)) => {
                // Per stage, a Start per gpc with that gpc's driver; the
                // stage ends when every gpc's program returns. Each Start
                // resets the gpc's version table, and so each driver's view
                // of the gpcs' ranges.
                let n = stages.len() as u64;
                let mut programs: Vec<[Option<ProgramFn>; 3]> = Vec::new();
                let mut nus = Vec::new();
                for s in stages {
                    let Stages {
                        nu,
                        vision,
                        vlm,
                        action,
                        versions,
                    } = s;
                    nus.push(nu);
                    let reset = |v: Rc<RefCell<VersionState>>, sp: Vec<Range>| {
                        move || v.borrow_mut().reset_spaces(&sp)
                    };
                    let (r0, r1, r2) = (
                        reset(versions.clone(), spaces.clone()),
                        reset(versions.clone(), spaces.clone()),
                        reset(versions, spaces.clone()),
                    );
                    programs.push([
                        Some(Box::new(move |c: NuCtx| {
                            Box::pin(async move {
                                r0();
                                vision.run(c).await
                            })
                        })),
                        Some(Box::new(move |c: NuCtx| {
                            Box::pin(async move {
                                r1();
                                vlm.run(c).await
                            })
                        })),
                        Some(Box::new(move |c: NuCtx| {
                            Box::pin(async move {
                                r2();
                                action.run(c).await
                            })
                        })),
                    ]);
                }
                for (stage, name) in ["vision", "vlm", "action"].iter().enumerate() {
                    let mut handles = Vec::new();
                    for (g, p) in programs.iter_mut().enumerate() {
                        let prog = p[stage].take().expect("a stage program");
                        let done =
                            Range::from_len(sync.base + (stage as u64 * n + g as u64) * 512, 512);
                        handles.push(ctx.spawn_named(nus[g], name, done, vec![], prog).await);
                    }
                    for h in handles {
                        ctx.wait_key(h.key()).await;
                    }
                    ctx.mark(&format!("{name} end")).await;
                    t[stage + 1] = ctx.now();
                }
            }
        }
        let [t0, t1, t2, t3] = t;
        let roof = |r: &prefill::Report| -> u64 {
            r.phases
                .iter()
                .filter_map(|p| p.roofline.map(|x| x.bound))
                .sum::<f64>()
                .ceil() as u64
        };
        let stage =
            |name: &str, start: u64, end: u64, hbm: u64, cmem: u64, once: u64, rl: Option<u64>| {
                let weights_once_bound = (once as f64 / hbm_rate).ceil() as u64;
                Stage {
                    roofline: rl.unwrap_or(weights_once_bound),
                    name: name.into(),
                    start,
                    end,
                    cycles: end - start,
                    hbm_bytes: hbm,
                    cmem_bytes: cmem,
                    w_once_bytes: once,
                    weights_once_bound,
                    own_traffic_bound: (hbm as f64 / hbm_rate).ceil() as u64,
                    hbm_by_gpc: Vec::new(),
                }
            };
        let (v, l, a) = (
            vr.borrow().clone(),
            lr.borrow().clone(),
            action_report.borrow().clone(),
        );
        let splits = [
            v.hbm_by_gpc.clone(),
            l.hbm_by_gpc.clone(),
            a.hbm_by_gpc.clone(),
        ];
        let mut stages = vec![
            stage(
                "vision",
                t0,
                t1,
                v.hbm_bytes,
                v.cmem_bytes,
                v.w_once,
                Some(roof(&v)),
            ),
            stage(
                "vlm",
                t1,
                t2,
                l.hbm_bytes,
                l.cmem_bytes,
                l.w_once,
                Some(roof(&l)),
            ),
            stage(
                "action",
                t2,
                t3,
                a.hbm_bytes,
                a.cmem_bytes,
                action_w.0 * action_w.1 as u64,
                None,
            ),
        ];
        let mut all = splits[0].clone();
        for (st, sp) in stages.iter_mut().zip(&splits) {
            st.hbm_by_gpc = sp.clone();
        }
        for sp in &splits[1..] {
            for (t, x) in all.iter_mut().zip(sp) {
                t.local += x.local;
                t.remote += x.remote;
            }
        }
        let sum = |f: fn(&Stage) -> u64| stages.iter().map(f).sum::<u64>();
        let mut total = stage(
            "total",
            t0,
            t3,
            sum(|s| s.hbm_bytes),
            sum(|s| s.cmem_bytes),
            sum(|s| s.w_once_bytes),
            Some(sum(|s| s.roofline)),
        );
        total.hbm_by_gpc = all;
        let mut r = report.borrow_mut();
        for (nu, v, l, a) in &reports {
            for (stage, invoked, planned, windows) in [
                (
                    "vision",
                    v.borrow().tasks,
                    v.borrow().planned,
                    v.borrow().windows,
                ),
                (
                    "vlm",
                    l.borrow().tasks,
                    l.borrow().planned,
                    l.borrow().windows,
                ),
                (
                    "action",
                    a.borrow().tasks,
                    a.borrow().planned,
                    a.borrow().windows,
                ),
            ] {
                r.drivers.push(DriverStat {
                    nu: nu.0,
                    stage: stage.into(),
                    invoked,
                    planned,
                    windows,
                });
            }
        }
        r.stages = stages;
        r.total = total;
        r.vision = Some(v);
        r.vlm = Some(l);
        r.vreset_data = versions.borrow().vreset_data;
        r.vreset_signal = versions.borrow().vreset_signal;
    }
}
