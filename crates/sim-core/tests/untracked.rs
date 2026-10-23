//! Untracked tasks (2026-10-20): the invoker runs them, no lookup finds
//! them, and only the fence (`wait 0`) and the program's return wait for
//! them. Config: `configs/two_sm.yaml` (sm0 and sm1 under gpc0; c_wait 2,
//! c_complete 4, c_resolve 0).

use std::cell::Cell;
use std::rc::Rc;

use nu_model::{Addr, Dep, NeutrinoId, Range};
use sim_core::{
    build_config_file, Args, Built, Descriptor, FatalError, NuCtx, OpcodeId, ProfileRegistry,
    Stats, TaskTimeline,
};

fn two_sm(sets: &[&str]) -> Result<Built, sim_core::ConfigError> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_sm.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins())
}

fn built() -> Built {
    two_sm(&[]).expect("config")
}

/// A DMA of `bytes` from `src` to `dst` on the engine `pe`, labelled
/// `label`.
fn dma(b: &Built, pe: &str, src: u64, dst: u64, bytes: u64, label: u64) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe(pe)),
        dst: Range::from_len(dst, bytes),
        versioned: false,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        pc: None,
        deps: vec![],
        label: Some(label),
        args: Args::Dma {
            src: Addr(src),
            dst: Addr(dst),
            bytes,
            tile_bytes: bytes,
        },
    }
}

fn task(s: &Stats, label: u64) -> &TaskTimeline {
    s.tasks
        .iter()
        .find(|t| t.label == Some(label))
        .unwrap_or_else(|| panic!("no task {label}"))
}

/// Runs `program` on `nu` with the task timeline on.
fn run_on(b: Built, nu: NeutrinoId, program: impl sim_core::NeutrinoProgram + 'static) -> Stats {
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(nu, program);
    sim.run()
}

#[test]
fn an_untracked_dma_into_another_nodes_memory_runs_on_the_invokers_engine() {
    // sm0 copies 4 KiB of its SMEM into the SMEM of sm1. A tracked task
    // would be forwarded to sm1 and run on sm1's engine; the untracked one
    // enters sm0's tables and runs on sm0's engine, with no Invoke.
    let b = built();
    let sm0 = b.nu("root.gpc0.sm0");
    let (s0, s1) = (
        b.memory("root.gpc0.sm0.smem").base,
        b.memory("root.gpc0.sm1.smem").base,
    );
    let d = dma(&b, "root.gpc0.sm0.dma", s0, s1, 4096, 1);
    let pe = b.pe("root.gpc0.sm0.dma");
    let s = run_on(b, sm0, move |ctx: NuCtx| async move {
        ctx.invoke_untracked(d).await;
    });
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let t = task(&s, 1);
    assert_eq!((t.nu, t.pe, t.version), (sm0, Some(pe), None));
    // It retires at its completion.
    assert!(t.complete.is_some());
    assert_eq!(t.retire, t.complete);
    assert_eq!(s.messages_by_kind.get("Invoke"), None);
}

#[test]
fn a_fence_waits_for_the_untracked_tasks() {
    let b = built();
    let sm0 = b.nu("root.gpc0.sm0");
    let (s0, s1) = (
        b.memory("root.gpc0.sm0.smem").base,
        b.memory("root.gpc0.sm1.smem").base,
    );
    let (d1, d2) = (
        dma(&b, "root.gpc0.sm0.dma", s0, s1, 4096, 1),
        dma(&b, "root.gpc0.sm0.dma", s0 + 4096, s1 + 4096, 8192, 2),
    );
    let after = Rc::new(Cell::new(0));
    let a = after.clone();
    let s = run_on(b, sm0, move |ctx: NuCtx| async move {
        ctx.invoke_untracked(d1).await;
        ctx.invoke_untracked(d2).await;
        ctx.fence().await;
        a.set(ctx.now());
        // With no untracked task in flight a fence answers at once
        // (c_wait 2).
        let t = ctx.now();
        ctx.fence().await;
        assert_eq!(ctx.now(), t + 2);
    });
    assert_eq!(s.fatal, None);
    let last = task(&s, 1)
        .complete
        .unwrap()
        .max(task(&s, 2).complete.unwrap());
    // c_resolve 0: the program continues when the last one completes.
    assert_eq!(after.get(), last);
}

#[test]
fn a_return_waits_for_the_untracked_tasks() {
    // gpc0 starts a program on sm0 that invokes an untracked DMA into
    // sm1's SMEM and returns at once. The return, and so the Return
    // message and the completion of the spawn, come after the DMA.
    let b = built();
    let (gpc0, sm0) = (b.nu("root.gpc0"), b.nu("root.gpc0.sm0"));
    let (s0, s1) = (
        b.memory("root.gpc0.sm0.smem").base,
        b.memory("root.gpc0.sm1.smem").base,
    );
    let done = Range::from_len(b.sync_only(gpc0).unwrap().base, 512);
    let d = dma(&b, "root.gpc0.sm0.dma", s0, s1, 16384, 1);
    let s = run_on(b, gpc0, move |ctx: NuCtx| async move {
        let child = move |c: NuCtx| async move {
            c.invoke_untracked(d).await;
        };
        ctx.spawn_child(sm0, done, vec![], child).await;
        ctx.wait_unversioned(done.base).await;
    });
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let dma_end = task(&s, 1).complete.unwrap();
    let spawn = s
        .tasks
        .iter()
        .find(|t| t.opcode == OpcodeId::SPAWN)
        .unwrap();
    // The Return travels after the DMA's completion: sm1's link up, the
    // crossbar, and nothing on the way is faster than 10 cycles.
    assert!(spawn.complete.unwrap() > dma_end + 10);
    assert_eq!(s.messages_by_kind.get("Return"), Some(&1));
}

#[test]
fn a_lookup_does_not_find_an_untracked_task() {
    let b = built();
    let sm0 = b.nu("root.gpc0.sm0");
    let s0 = b.memory("root.gpc0.sm0.smem").base;
    let (src, dst) = (s0, s0 + 0x40000);
    // P writes dst tracked; U writes it untracked and is long. A wait on dst
    // binds P (done) and not U.
    let p = dma(&b, "root.gpc0.sm0.dma", src, dst, 512, 1);
    let u = dma(&b, "root.gpc0.sm0.dma", src, dst, 65536, 2);
    // C depends on dst2 before anything writes it (a pending entry). U2
    // writes dst2 untracked and does not adopt the pending entry; P2, a
    // later tracked writer, does.
    let dst2 = s0 + 0x60000;
    let mut c = dma(&b, "root.gpc0.sm0.dma", src, s0 + 0x70000, 512, 3);
    c.deps = vec![Dep::Unversioned(dst2)];
    let u2 = dma(&b, "root.gpc0.sm0.dma", src, dst2, 8192, 4);
    let p2 = dma(&b, "root.gpc0.sm0.dma", src, dst2, 512, 5);
    let waited = Rc::new(Cell::new(0));
    let w = waited.clone();
    let s = run_on(b, sm0, move |ctx: NuCtx| async move {
        ctx.invoke(p).await;
        ctx.wait_unversioned(dst).await;
        ctx.invoke_untracked(u).await;
        ctx.wait_unversioned(dst).await;
        w.set(ctx.now());
        let hc = ctx.invoke(c).await;
        ctx.invoke_untracked(u2).await;
        ctx.fence().await;
        ctx.invoke(p2).await;
        ctx.wait_unversioned(hc.addr).await;
    });
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    assert!(waited.get() < task(&s, 2).complete.unwrap());
    let ready = task(&s, 3).ready.unwrap();
    assert!(ready >= task(&s, 5).complete.unwrap());
    assert!(ready > task(&s, 4).complete.unwrap());
}

/// The fatal error of `program` on sm0, as text.
fn invalid(b: Built, program: impl sim_core::NeutrinoProgram + 'static) -> String {
    let sm0 = b.nu("root.gpc0.sm0");
    let s = run_on(b, sm0, program);
    match s.fatal.expect("fatal").error {
        FatalError::Invalid(m) => m,
        e => panic!("not Invalid: {e:?}"),
    }
}

#[test]
fn the_invalid_untracked_tasks() {
    let b = built();
    let s0 = b.memory("root.gpc0.sm0.smem").base;
    let sync = b.sync_only(b.nu("root.gpc0.sm0")).unwrap().base;
    let d = dma(&b, "root.gpc0.sm0.dma", s0, s0 + 0x1000, 512, 1);
    // A join, a spawn, a versioned task, a task without a PE, a PE of
    // another node.
    let join = Descriptor::join(Range::from_len(sync, 512), vec![]);
    let sm1 = b.nu("root.gpc0.sm1");
    let spawn = Descriptor {
        opcode: OpcodeId::SPAWN,
        pe: None,
        args: Args::Spawn { child: sm1 },
        ..join.clone()
    };
    let versioned = d.clone().versioned();
    let no_pe = Descriptor {
        pe: None,
        ..d.clone()
    };
    let other_pe = Descriptor {
        pe: Some(b.pe("root.gpc0.sm1.dma")),
        ..d.clone()
    };
    let cases = [
        (join, "a join cannot be untracked"),
        (spawn, "a spawn cannot be untracked"),
        (versioned, "a versioned task cannot be untracked"),
        (no_pe, "a task without a PE cannot be untracked"),
        (other_pe, "untracked invoke on"),
    ];
    for (desc, want) in cases {
        let m = invalid(built(), move |ctx: NuCtx| async move {
            ctx.invoke_untracked(desc).await;
        });
        assert!(m.contains(want), "{m}");
    }
    // The key 0 names no producer: not in deps, and a wait on it is the
    // fence (unversioned) only.
    let mut dep0 = d.clone();
    dep0.deps = vec![Dep::Unversioned(0)];
    let m = invalid(built(), move |ctx: NuCtx| async move {
        ctx.invoke(dep0).await;
    });
    assert!(m.contains("the key 0 in deps"), "{m}");
    let m = invalid(b, move |ctx: NuCtx| async move {
        ctx.wait(0, 1).await;
    });
    assert!(m.contains("the key 0 is the fence"), "{m}");
}

#[test]
fn the_loader_rejects_an_owned_range_that_holds_address_0() {
    let e = two_sm(&["root.hbm.addr=[[0x0, 0x800000]]"])
        .err()
        .expect("rejected")
        .0;
    assert!(e.contains("holds address 0, the null address"), "{e}");
}

#[test]
fn an_untracked_dma_writes_another_nodes_memory_in_the_functional_plane() {
    // Numeric fidelity: sm0 stores a pattern in its SMEM, copies it into
    // sm1's SMEM with an untracked DMA, fences, and loads it back from
    // sm1's SMEM.
    let b = two_sm(&["root.fidelity=numeric"]).expect("config");
    let sm0 = b.nu("root.gpc0.sm0");
    let (s0, s1) = (
        b.memory("root.gpc0.sm0.smem").base,
        b.memory("root.gpc0.sm1.smem").base,
    );
    let d = dma(&b, "root.gpc0.sm0.dma", s0, s1 + 0x1000, 512, 1);
    let mut sim = b.sim;
    sim.set_functional(Box::new(func::Numeric::new()));
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let pattern: Vec<u8> = (0..512).map(|i| (i * 7 % 251) as u8).collect();
        ctx.store(Addr(s0), pattern.clone()).await;
        ctx.invoke_untracked(d).await;
        ctx.fence().await;
        assert_eq!(ctx.load(Addr(s1 + 0x1000), 512).await, pattern);
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
}
