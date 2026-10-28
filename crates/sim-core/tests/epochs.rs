//! Epochs in the version table: a program start with epoch e clears the
//! version table and sets the epoch, and a base without an entry reads as
//! `e << EPOCH_SHIFT`. Two gpcs run step programs out of phase without a
//! barrier: each names keys of the previous step by their epoch.
//! Config `tests/configs/two_gpc.yaml`.

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::{Dep, DepError, Range, EPOCH_SHIFT};
use sim_core::{
    build_config_file, Built, Descriptor, FatalError, NuCtx, ProfileRegistry, ProgramName, Stats,
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

/// The root starts step programs 1 to 3 on gpc0 and gpc1, each after the
/// gpc's previous program returned (the dependency of each spawn on the
/// done range of the one before), in epoch `epoch(k)`. In step k each gpc
/// writes version 1 of its step key X in the epoch of step k, a join that
/// depends on its own X and the other gpc's X of step k - 1 (named by the
/// epoch of step k - 1). gpc1 is slow: its program first does `SLOW`
/// cycles of scalar work, and its X also waits for a long copy into its
/// HBM. No program waits for its tasks. The log has (gpc, step,
/// start cycle) per program.
fn run(epoch: fn(u64) -> u64) -> (Stats, Log) {
    let b = two_gpc();
    let root = b.nu("root");
    let g = [b.nu("root.gpc0"), b.nu("root.gpc1")];
    let sync = b.sync_only(root).unwrap();
    let x: Vec<Range> = g
        .iter()
        .map(|&n| Range::from_len(b.sync_only(n).unwrap().base, 512))
        .collect();
    let hbm = b.memory("root.gpc1.hbm");
    let dma = b.pe("root.gpc1.dma");
    let log: Log = Rc::default();
    let l = log.clone();
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        let mut prev: [Option<Dep>; 2] = [None, None];
        let mut all = Vec::new();
        for k in 1..=STEPS {
            for i in 0..2 {
                let (me, other) = (x[i], x[1 - i]);
                let l = l.clone();
                let program = move |c: NuCtx| async move {
                    l.borrow_mut().push((i as u64, k, c.now()));
                    let mut deps = Vec::new();
                    if k > 1 {
                        let v = ev(epoch(k - 1), 1);
                        deps.push(Dep::Versioned {
                            addr: me.base,
                            version: v,
                        });
                        deps.push(Dep::Versioned {
                            addr: other.base,
                            version: v,
                        });
                    }
                    if i == 1 {
                        c.cycles(SLOW).await;
                        let src = Range::from_len(hbm.base, LONG);
                        let dst = Range::from_len(hbm.base + k * LONG, LONG);
                        let h = c
                            .invoke(Descriptor::dma(dma, src.base, dst, LONG).versioned())
                            .await;
                        deps.push(h.key());
                    }
                    let h = c
                        .invoke(
                            Descriptor::join(me, deps)
                                .versioned()
                                .with_label(label(i as u64, k)),
                        )
                        .await;
                    assert_eq!(h.version, ev(epoch(k), 1));
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

/// The completion of the step task of gpc `g` in step `k`.
fn done(s: &Stats, g: u64, k: u64) -> u64 {
    let t = s.tasks.iter().find(|t| t.label == Some(label(g, k)));
    t.and_then(|t| t.complete).expect("step task completed")
}

/// Runs the step programs in the epochs `epoch(k)` and checks that they
/// run out of phase and complete in the order of the steps.
fn out_of_phase(epoch: fn(u64) -> u64) {
    let (s, log) = run(epoch);
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
    out_of_phase(|k| k);
}

#[test]
fn the_out_of_phase_steps_across_the_wrap_of_the_epochs() {
    // Steps 1 to 3 in epochs 255, 0 and 1: epoch 255 is before epoch 0,
    // and epoch 1 after it.
    out_of_phase(|k| (254 + k) % 256);
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
