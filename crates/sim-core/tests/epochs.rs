//! Epochs in the version table: a program start with epoch e, or the
//! instruction `epoch e`, clears the version table and sets the epoch, and
//! a base without an entry reads as `e << EPOCH_SHIFT`. Two gpcs run steps
//! out of phase without a barrier: each names keys of the previous step by
//! their epoch.
//! Config `tests/configs/two_gpc.yaml`.

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::{Dep, DepError, NeutrinoId, Range, EPOCH_SHIFT};
use sim_core::{
    build_config_file, Built, Descriptor, FatalError, NuCtx, PeId, ProfileRegistry, ProgramName,
    Stats,
};

fn two_gpc() -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_gpc.yaml");
    build_config_file(&path, &[] as &[&str], ProfileRegistry::with_builtins()).expect("config")
}

/// Version `count` of epoch `epoch`.
fn ev(epoch: u64, count: u64) -> u64 {
    epoch << EPOCH_SHIFT | count
}

const STEPS: u64 = 3;
/// Bytes of the long copy of the slow gpc per step: about 16,400 cycles
/// on the 64 B/cycle data crossbar.
const LONG: u64 = 1 << 20;
/// Scalar cycles of the slow gpc's program before its invokes.
const SLOW: u64 = 300;

/// The label of the step task of gpc `g` in step `k`.
fn label(g: u64, k: u64) -> u64 {
    10 * g + k
}

type Log = Rc<RefCell<Vec<(u64, u64, u64)>>>;

/// The step keys X of the two gpcs, and the HBM and copy engine of the
/// slow gpc (gpc1).
#[derive(Clone, Copy)]
struct Parts {
    x: [Range; 2],
    hbm: Range,
    dma: PeId,
}

fn parts(b: &Built) -> ([NeutrinoId; 2], Parts) {
    let g = [b.nu("root.gpc0"), b.nu("root.gpc1")];
    let x = g.map(|n| Range::from_len(b.sync_only(n).unwrap().base, 512));
    let hbm = b.memory("root.gpc1.hbm");
    let dma = b.pe("root.gpc1.dma");
    (g, Parts { x, hbm, dma })
}

/// Step `s` of gpc `i` in epoch `epoch`: writes version 1 of its step key
/// X in that epoch, a join that depends on its own X and the other gpc's
/// X of the step before, named by their epoch `prev` (none in the first
/// step). gpc1 is slow: it first does `SLOW` cycles of scalar work, and
/// its X also waits for a long copy into its HBM. Logs (gpc, step, cycle)
/// at the start; returns the key of the step task.
async fn step(
    c: &NuCtx,
    p: Parts,
    log: &Log,
    i: usize,
    s: u64,
    epoch: u64,
    prev: Option<u64>,
) -> Dep {
    log.borrow_mut().push((i as u64, s, c.now()));
    let (me, other) = (p.x[i], p.x[1 - i]);
    let mut deps = Vec::new();
    if let Some(e) = prev {
        for r in [me, other] {
            deps.push(Dep::Versioned {
                addr: r.base,
                version: ev(e, 1),
            });
        }
    }
    if i == 1 {
        c.cycles(SLOW).await;
        let src = Range::from_len(p.hbm.base, LONG);
        let dst = Range::from_len(p.hbm.base + (1 + (s - 1) % STEPS) * LONG, LONG);
        let h = c
            .invoke(Descriptor::dma(p.dma, src.base, dst, LONG).versioned())
            .await;
        deps.push(h.key());
    }
    let h = c
        .invoke(
            Descriptor::join(me, deps)
                .versioned()
                .with_label(label(i as u64, s)),
        )
        .await;
    assert_eq!(h.version, ev(epoch, 1));
    h.key()
}

/// The root starts step programs 1 to 3 on gpc0 and gpc1, each after the
/// gpc's previous program returned (the dependency of each spawn on the
/// done range of the one before), in epoch `epoch(k)`; each program runs
/// [`step`] k. No program waits for its tasks.
fn run(epoch: fn(u64) -> u64) -> (Stats, Log) {
    let b = two_gpc();
    let root = b.nu("root");
    let sync = b.sync_only(root).unwrap();
    let (g, p) = parts(&b);
    let log: Log = Rc::default();
    let l = log.clone();
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        let mut prev: [Option<Dep>; 2] = [None, None];
        let mut all = Vec::new();
        for k in 1..=STEPS {
            for i in 0..2 {
                let l = l.clone();
                let program = move |c: NuCtx| async move {
                    let before = (k > 1).then(|| epoch(k - 1));
                    step(&c, p, &l, i, k, epoch(k), before).await;
                };
                let done = Range::from_len(sync.base + 512 * (2 * k + i as u64), 512);
                let h = ctx
                    .spawn_in_epoch(
                        g[i],
                        ProgramName::Fixed("step"),
                        done,
                        prev[i].into_iter().collect(),
                        None,
                        epoch(k) as u16,
                        program,
                    )
                    .await;
                prev[i] = Some(h.key());
                all.push(h.key());
            }
        }
        for key in all {
            ctx.wait_key(key).await;
        }
    });
    (sim.run(), log)
}

/// One persistent program per gpc (started in epoch 0): at step k it
/// executes `epoch epoch(k)` and runs [`step`] k, without waiting for its
/// tasks.
fn run_persistent(epoch: fn(u64) -> u64) -> (Stats, Log) {
    let b = two_gpc();
    let root = b.nu("root");
    let sync = b.sync_only(root).unwrap();
    let (g, p) = parts(&b);
    let log: Log = Rc::default();
    let l = log.clone();
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        let mut all = Vec::new();
        for (i, gpc) in g.into_iter().enumerate() {
            let l = l.clone();
            let program = move |c: NuCtx| async move {
                for k in 1..=STEPS {
                    c.set_epoch(epoch(k)).await;
                    assert_eq!(c.epoch(), epoch(k));
                    let before = (k > 1).then(|| epoch(k - 1));
                    step(&c, p, &l, i, k, epoch(k), before).await;
                }
            };
            let done = Range::from_len(sync.base + 512 * i as u64, 512);
            let name = ProgramName::Fixed("steps");
            let h = ctx
                .spawn_in_epoch(gpc, name, done, vec![], None, 0, program)
                .await;
            all.push(h.key());
        }
        for key in all {
            ctx.wait_key(key).await;
        }
    });
    (sim.run(), log)
}

/// The completion of the step task of gpc `g` in step `k`.
fn done(s: &Stats, g: u64, k: u64) -> u64 {
    let t = s.tasks.iter().find(|t| t.label == Some(label(g, k)));
    t.and_then(|t| t.complete).expect("step task completed")
}

/// Checks that the steps of a run ran out of phase and completed in the
/// order of the steps.
fn out_of_phase((s, log): (Stats, Log)) {
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    // The fast gpc runs ahead: its program of step 3 starts before the
    // slow gpc's program of step 2, so it names a key of an epoch that
    // gpc1 has not begun (a pending entry at gpc1, adopted in that epoch).
    // The slow gpc's step tasks of earlier epochs are still in flight when
    // later epochs name them.
    let start = |g, k| {
        log.borrow()
            .iter()
            .find(|e| (e.0, e.1) == (g, k))
            .unwrap()
            .2
    };
    assert!(start(0, STEPS) < start(1, 2));
    assert!(start(1, STEPS) < done(&s, 1, 1));
    // Every step task completes after both step tasks of the step before.
    for k in 2..=STEPS {
        for g in 0..2 {
            assert!(done(&s, g, k) > done(&s, g, k - 1), "gpc{g} step {k}");
            assert!(done(&s, g, k) > done(&s, 1 - g, k - 1), "gpc{g} step {k}");
        }
    }
    // The slow gpc's long copies set the pace of both.
    let mut order: Vec<(u64, u64, u64)> = (1..=STEPS)
        .flat_map(|k| (0..2).map(move |g| (g, k)))
        .map(|(g, k)| (done(&s, g, k), g, k))
        .collect();
    order.sort();
    let order: Vec<(u64, u64)> = order.into_iter().map(|(_, g, k)| (g, k)).collect();
    assert_eq!(order, vec![(0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)]);
}

#[test]
fn two_gpcs_out_of_phase_name_the_keys_of_the_previous_epoch() {
    out_of_phase(run(|k| k));
}

#[test]
fn the_out_of_phase_steps_across_the_wrap_of_the_epochs() {
    // Steps 1 to 3 in epochs 255, 0 and 1: epoch 255 is before epoch 0,
    // and epoch 1 after it.
    out_of_phase(run(|k| (254 + k) % 256));
}

#[test]
fn persistent_programs_set_the_epoch_of_each_step() {
    out_of_phase(run_persistent(|k| k));
    out_of_phase(run_persistent(|k| (254 + k) % 256));
}

#[test]
fn the_same_programs_in_epoch_0_fail() {
    // In epoch 0 the key of the previous step is (X, 1), the version that
    // the step's own write gets: the write would wait on itself.
    let (s, _) = run(|_| 0);
    let f = s.fatal.expect("a check error");
    assert!(
        matches!(f.error, FatalError::Dep(DepError::SelfDependency { .. })),
        "{f:?}"
    );
}

#[test]
fn a_spawn_with_an_epoch_above_255_is_an_error() {
    let b = two_gpc();
    let (root, g0) = (b.nu("root"), b.nu("root.gpc0"));
    let done = Range::from_len(b.sync_only(root).unwrap().base, 512);
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        let child = |_c: NuCtx| async move {};
        let name = ProgramName::Fixed("child");
        ctx.spawn_in_epoch(g0, name, done, vec![], None, 256, child)
            .await;
    });
    let f = sim.run().fatal.expect("an error at the spawn");
    assert!(
        matches!(&f.error, FatalError::Invalid(m) if m == "spawn with epoch 256, above 255"),
        "{f:?}"
    );
}

#[test]
fn a_start_that_ends_an_epoch_with_a_pending_key_is_an_error() {
    // In epoch 1, gpc0 invokes a join that depends on (X, 1 << 24 | 2) of
    // gpc1, which nothing writes in epoch 1: a pending entry at gpc1. The
    // next start of gpc1 (epoch 2, after gpc0's program) finds it.
    let b = two_gpc();
    let (root, g0, g1) = (b.nu("root"), b.nu("root.gpc0"), b.nu("root.gpc1"));
    let sync = b.sync_only(root).unwrap();
    let x = Range::from_len(b.sync_only(g1).unwrap().base, 512);
    let y = Range::from_len(b.sync_only(g0).unwrap().base, 512);
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        let key = Dep::Versioned {
            addr: x.base,
            version: ev(1, 2),
        };
        // The program returns well after its Await reached gpc1.
        let waiter = move |c: NuCtx| async move {
            c.invoke(Descriptor::join(y, vec![key])).await;
            c.cycles(1000).await;
        };
        let empty = |_c: NuCtx| async move {};
        let d = |i: u64| Range::from_len(sync.base + 512 * i, 512);
        let name = || ProgramName::Fixed("p");
        let h1 = ctx
            .spawn_in_epoch(g1, name(), d(0), vec![], None, 1, empty)
            .await;
        let h0 = ctx
            .spawn_in_epoch(g0, name(), d(1), vec![h1.key()], None, 1, waiter)
            .await;
        ctx.spawn_in_epoch(g1, name(), d(2), vec![h0.key()], None, 2, empty)
            .await;
    });
    let s = sim.run();
    let f = s.fatal.expect("an error at the start");
    assert_eq!(f.nu, g1);
    assert_eq!(
        f.error,
        FatalError::Dep(DepError::StalePending {
            addr: x.base,
            version: ev(1, 2),
            epoch: 2
        })
    );
}

/// Cycles between the starts of gpc0's and gpc1's programs of lifetime 2.
const DELAY: u64 = 2000;

/// Two lifetimes of steps 1 to 3 (epochs 1 to 3): steps 1 to 3, then 4 to
/// 6. Each gpc runs one program per lifetime, started in epoch 1, that
/// executes `epoch k` at step k > 1 and runs [`step`]. The last step waits
/// for its task and, with `reset`, executes `epoch 0` before the return.
/// The root waits for both programs of lifetime 1, starts gpc0's program
/// of lifetime 2, and starts gpc1's `DELAY` cycles later.
fn run_lifetimes(reset: bool) -> (Stats, Log) {
    let b = two_gpc();
    let root = b.nu("root");
    let sync = b.sync_only(root).unwrap();
    let (g, p) = parts(&b);
    let log: Log = Rc::default();
    let l = log.clone();
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        for life in 0..2 {
            let mut done = Vec::new();
            for (i, gpc) in g.into_iter().enumerate() {
                let l = l.clone();
                let program = move |c: NuCtx| async move {
                    for k in 1..=STEPS {
                        if k > 1 {
                            c.set_epoch(k).await;
                        }
                        let s = STEPS * life + k;
                        let key = step(&c, p, &l, i, s, k, (k > 1).then(|| k - 1)).await;
                        if k == STEPS {
                            c.wait_key(key).await;
                            if reset {
                                c.set_epoch(0).await;
                            }
                        }
                    }
                };
                if life == 1 && i == 1 {
                    ctx.cycles(DELAY).await;
                }
                let range = Range::from_len(sync.base + 512 * (2 * life + i as u64), 512);
                let name = ProgramName::Fixed("lifetime");
                let h = ctx
                    .spawn_in_epoch(gpc, name, range, vec![], None, 1, program)
                    .await;
                done.push(h.key());
            }
            for key in done {
                ctx.wait_key(key).await;
            }
        }
    });
    (sim.run(), log)
}

#[test]
fn the_last_program_of_a_lifetime_returns_its_neutrino_to_epoch_0() {
    let (s, log) = run_lifetimes(true);
    assert_eq!(s.fatal, None);
    let start = |g, k| {
        log.borrow()
            .iter()
            .find(|e| (e.0, e.1) == (g, k))
            .unwrap()
            .2
    };
    // gpc0 names gpc1's key (X, 1:1) of lifetime 2 before gpc1 starts
    // lifetime 2: gpc1 is in epoch 0, so the key is pending there.
    assert!(start(0, 5) < start(1, 4));
    for s_ in [2, 3, 5, 6] {
        for g in 0..2 {
            assert!(done(&s, g, s_) > done(&s, g, s_ - 1), "gpc{g} step {s_}");
            assert!(
                done(&s, g, s_) > done(&s, 1 - g, s_ - 1),
                "gpc{g} step {s_}"
            );
        }
    }
    for g in 0..2 {
        assert!(done(&s, g, 4) > done(&s, 0, 3).max(done(&s, 1, 3)));
    }
}

#[test]
fn without_epoch_0_a_new_lifetime_reads_the_keys_of_the_old_one() {
    // The control: gpc1 is still in epoch 3 of lifetime 1 when gpc0 names
    // its key (X, 1:1) of lifetime 2. Epoch 1 is before epoch 3 and
    // lifetime 1 produced (X, 1:1), so the key is done: gpc0's step 5
    // completes before gpc1's step 4.
    let (s, log) = run_lifetimes(false);
    assert_eq!(s.fatal, None);
    let start = |g, k| {
        log.borrow()
            .iter()
            .find(|e| (e.0, e.1) == (g, k))
            .unwrap()
            .2
    };
    assert!(start(0, 5) < start(1, 4));
    assert!(done(&s, 0, 5) < done(&s, 1, 4), "the wrong order");
}

/// Runs `program` on the root of `two_gpc.yaml` with X and Y, two ranges
/// of the root's sync space; the stats and X.
fn on_root<F, Fut>(program: F) -> (Stats, Range)
where
    F: FnOnce(NuCtx, Range, Range) -> Fut + 'static,
    Fut: std::future::Future<Output = ()> + 'static,
{
    let b = two_gpc();
    let root = b.nu("root");
    let sync = b.sync_only(root).unwrap();
    let (x, y) = (
        Range::from_len(sync.base, 512),
        Range::from_len(sync.base + 512, 512),
    );
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| program(ctx, x, y));
    (sim.run(), x)
}

#[test]
fn the_instruction_epoch_changes_the_epoch_as_a_start() {
    // Epoch 0 writes X twice. After `epoch 1`, X's next version follows
    // the epoch base of epoch 1, (X, 2) is done and (X, 3) was never
    // produced.
    let (s, x) = on_root(|ctx, x, _| async move {
        for v in 1..=2 {
            let h = ctx.invoke(Descriptor::join(x, vec![]).versioned()).await;
            assert_eq!(h.version, v);
        }
        ctx.set_epoch(1).await;
        assert_eq!(ctx.epoch(), 1);
        let h = ctx.invoke(Descriptor::join(x, vec![]).versioned()).await;
        assert_eq!(h.version, ev(1, 1));
        ctx.wait(x.base, 2).await;
        ctx.wait(x.base, 3).await;
    });
    let f = s.fatal.expect("an error at the wait");
    assert_eq!(
        f.error,
        FatalError::Dep(DepError::Unproduced {
            addr: x.base,
            version: 3,
            highest: Some(2)
        })
    );
}

#[test]
fn the_instruction_epoch_with_a_stale_pending_key_is_an_error() {
    // A join waits on (X, 5), which epoch 0 never writes: `epoch 1` ends
    // epoch 0 with that pending entry.
    let (s, x) = on_root(|ctx, x, y| async move {
        let key = Dep::Versioned {
            addr: x.base,
            version: 5,
        };
        ctx.invoke(Descriptor::join(y, vec![key])).await;
        ctx.set_epoch(1).await;
        unreachable!("the run stops at the instruction");
    });
    let f = s.fatal.expect("an error at the instruction");
    assert_eq!(
        f.error,
        FatalError::Dep(DepError::StalePending {
            addr: x.base,
            version: 5,
            epoch: 1
        })
    );
}

#[test]
fn the_instruction_epoch_above_255_is_an_error() {
    let (s, _) = on_root(|ctx, _, _| async move {
        ctx.set_epoch(256).await;
    });
    let f = s.fatal.expect("an error at the instruction");
    assert!(
        matches!(&f.error, FatalError::Invalid(m) if m == "epoch 256, above 255"),
        "{f:?}"
    );
}
