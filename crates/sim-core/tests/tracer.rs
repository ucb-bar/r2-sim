//! The observation points (architecture Section 7): a traced run ends at
//! the same cycle as an untraced run, and the hooks see the tasks, the
//! charges, the remote dependencies and the programs. Configs and timings
//! as in `remote_invoke.rs` (two_sm.yaml) and `hierarchy2.rs`
//! (two_gpc.yaml, two gpcs of two SMs).

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::{Addr, Dep, NeutrinoId, Range};
use sim_core::{
    build_config_file, Args, Built, Charge, Cycle, Descriptor, Dtype, NuCtx, ProfileRegistry, Sim,
    Stats, TaskCall, TaskEntered, Tracer, Who,
};

const MAT: u64 = 64 * 64 * 2;

fn built(file: &str) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/configs")
        .join(file);
    build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).expect("config")
}

fn matmul(b: &Built, pe: &str, c: Range, deps: Vec<Dep>) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe(pe)),
        dst: c,
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: Some(1),
        deps,
        label: None,
        args: Args::Matmul {
            a: Addr(c.base),
            b: Addr(c.base),
            a2: None,
            b2: None,
            lda: 64,
            ldb: 64,
            ldc: 64,
            m: 64,
            n: 64,
            k: 64,
            in_dtype: Dtype::Bf16,
            b_dtype: Dtype::Bf16,
            out_dtype: Dtype::Bf16,
            accumulate: false,
            keep: false,
            tile: [64, 64],
        },
    }
}

/// What the recording tracer saw.
#[derive(Debug, Default)]
struct Seen {
    /// (gid, nu, call, invoke).
    invokes: Vec<(u64, NeutrinoId, TaskCall, Cycle)>,
    retires: usize,
    /// Cycles per resource from the charges.
    cycles: Vec<f64>,
    message_charges: usize,
    /// Opened remote requests: (requester, request, gid).
    remote: Vec<(NeutrinoId, u32, u64)>,
    /// Satisfied remote requests: (requester, request, time).
    remote_done: Vec<(NeutrinoId, u32, Cycle)>,
    /// (nu, name, spawn, start).
    starts: Vec<(NeutrinoId, &'static str, Option<u64>, Cycle)>,
    returns: Vec<(NeutrinoId, Cycle)>,
    finished: bool,
}

struct Recorder(Rc<RefCell<Seen>>);

impl Tracer for Recorder {
    fn begin(&mut self, sim: &Sim) {
        self.0.borrow_mut().cycles = vec![0.0; sim.resources().len()];
    }
    fn task_invoke(&mut self, t: &TaskEntered) {
        let e = (t.gid, t.nu, t.call, t.time);
        self.0.borrow_mut().invokes.push(e);
    }
    fn task_retire(&mut self, _gid: u64, _time: Cycle) {
        self.0.borrow_mut().retires += 1;
    }
    fn charge(&mut self, c: &Charge, who: Who) {
        let mut s = self.0.borrow_mut();
        s.cycles[c.resource.0 as usize] += c.cycles;
        if matches!(who, Who::Message { .. }) {
            s.message_charges += 1;
        }
    }
    fn dep_remote_open(&mut self, nu: NeutrinoId, req: u32, gid: u64, _key: Dep) {
        self.0.borrow_mut().remote.push((nu, req, gid));
    }
    fn dep_remote_done(&mut self, nu: NeutrinoId, req: u32, time: Cycle) {
        self.0.borrow_mut().remote_done.push((nu, req, time));
    }
    fn program_start(&mut self, nu: NeutrinoId, name: &'static str, spawn: Option<u64>, t: Cycle) {
        self.0.borrow_mut().starts.push((nu, name, spawn, t));
    }
    fn program_return(&mut self, nu: NeutrinoId, time: Cycle) {
        self.0.borrow_mut().returns.push((nu, time));
    }
    fn finish(self: Box<Self>, _stats: &mut Stats) {
        self.0.borrow_mut().finished = true;
    }
}

/// Runs `setup`'s simulation untraced (no collector, no tracer) and
/// traced (collector and recorder): both stats and what was seen.
fn both(setup: impl Fn() -> Sim) -> (Stats, Stats, Seen) {
    let mut plain = setup();
    plain.set_timeline(false);
    let plain = plain.run();
    let seen = Rc::new(RefCell::new(Seen::default()));
    let mut traced = setup();
    traced.set_log(true);
    traced.set_marks(true);
    traced.set_tracer(Box::new(Recorder(seen.clone())));
    let traced = traced.run();
    let seen = Rc::try_unwrap(seen).expect("tracer dropped").into_inner();
    (plain, traced, seen)
}

/// The charges add up to each flow resource's busy slot-cycles.
fn check_cycles(s: &Stats, seen: &Seen) {
    for (i, r) in s.resources.iter().enumerate() {
        let (got, want) = (seen.cycles[i], r.busy_slot_cycles);
        assert!(
            (got - want).abs() <= 1e-6 * want.max(1.0),
            "resource {}: {got} vs {want}",
            r.name
        );
    }
}

fn two_sm_run() -> Sim {
    // remote_invoke.rs, forwarded_invoke_with_a_dep_back_on_the_invoker.
    let b = built("two_sm.yaml");
    let sm0 = b.nu("root.gpc0.sm0");
    let a = Range::from_len(b.memory("root.gpc0.sm0.smem").base, MAT);
    let c = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let p0 = matmul(&b, "root.gpc0.sm0.atlas", a, vec![]);
    let v1 = Dep::Versioned {
        addr: a.base,
        version: 1,
    };
    let m = matmul(&b, "root.gpc0.sm1.atlas", c, vec![v1]);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(p0).await;
        let h = ctx.invoke(m).await;
        ctx.wait(h.addr, 1).await;
    });
    sim
}

#[test]
fn a_traced_two_sm_run_ends_at_the_untraced_cycle() {
    let (plain, traced, seen) = both(two_sm_run);
    assert_eq!(plain.fatal, None);
    assert_eq!((plain.end_time, traced.end_time), (452, 452));
    assert_eq!(plain.events, traced.events);
    assert!(plain.tasks.is_empty(), "no collector");
    assert_eq!(traced.tasks.len(), 2);
    assert!(traced.resources.iter().all(|r| r.log.is_some()));
    assert!(seen.finished);
    let b = built("two_sm.yaml");
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    // P0: called at 0, local. M: called at 8 (after P0's c_invoke),
    // queued at 16 after its own, sent at once, in sm1's scoreboard at 48.
    let call = |call, queued, sent, seq| TaskCall {
        src: sm0,
        call,
        queued,
        sent,
        seq,
        n: seq + 1,
    };
    assert_eq!(seen.invokes[0], (0, sm0, call(0, None, None, 0), 8));
    assert_eq!(
        seen.invokes[1],
        (1, sm1, call(8, Some(16), Some(16), 1), 48)
    );
    assert_eq!(seen.retires, 2);
    // M's remote dep on A: Registered, then the Resolve at 230.
    assert_eq!(seen.remote.len(), 1);
    let (nu, req, gid) = seen.remote[0];
    assert_eq!((nu, gid), (sm1, 1));
    assert_eq!(seen.remote_done, vec![(sm1, req, 230)]);
    assert_eq!(seen.starts, vec![(sm0, "main", None, 0)]);
    assert_eq!(seen.returns, vec![(sm0, 452)]);
    assert!(seen.message_charges > 0);
    check_cycles(&traced, &seen);
}

fn two_gpc_run() -> Sim {
    // hierarchy2.rs, a_device_program_starts_two_gpc_drivers.
    let b = built("two_gpc.yaml");
    let (root, g0, g1) = (b.nu("root"), b.nu("root.gpc0"), b.nu("root.gpc1"));
    let sync = b.sync_only(root).unwrap();
    let m1r = Range::from_len(b.memory("root.gpc1.sm0.smem").base, MAT);
    let m1 = matmul(&b, "root.gpc1.sm0.atlas", m1r, vec![]);
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        let p0 = move |c: NuCtx| async move {
            c.wait(m1r.base, 1).await;
        };
        let p1 = move |c: NuCtx| async move {
            c.invoke(m1).await;
            c.wait(m1r.base, 1).await;
        };
        let h0 = ctx
            .spawn_named(g0, "p0", Range::from_len(sync.base, 512), vec![], p0)
            .await;
        let h1 = ctx
            .spawn_child(g1, Range::from_len(sync.base + 512, 512), vec![], p1)
            .await;
        ctx.wait_key(h0.key()).await;
        ctx.wait_key(h1.key()).await;
    });
    sim
}

#[test]
fn a_traced_two_gpc_run_ends_at_the_untraced_cycle() {
    let (plain, traced, seen) = both(two_gpc_run);
    assert_eq!(plain.fatal, None);
    assert_eq!((plain.end_time, traced.end_time), (375, 375));
    assert_eq!(plain.events, traced.events);
    let b = built("two_gpc.yaml");
    let (root, g0, g1) = (b.nu("root"), b.nu("root.gpc0"), b.nu("root.gpc1"));
    // The spawn tasks are gids 0 and 1; M1 is gid 2.
    assert_eq!(
        seen.starts,
        vec![
            (root, "main", None, 0),
            (g0, "p0", Some(0), 39),
            (g1, "child", Some(1), 55)
        ]
    );
    let mut r = seen.returns.clone();
    r.sort_by_key(|x| x.1);
    assert_eq!(r, vec![(g1, 290), (g0, 346), (root, 375)]);
    assert_eq!(seen.invokes.len(), 3);
    assert_eq!(seen.retires, 3);
    check_cycles(&traced, &seen);
}
