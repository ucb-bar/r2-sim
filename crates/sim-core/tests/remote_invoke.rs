//! Forwarded invoke (Neutrino spec Section 9.4). Config and path
//! timings as in `hierarchy.rs`: a message of b 64 B beats between the SMs
//! arrives 20 + 6b cycles after it is sent when the path is free; a 64^3
//! bf16 matmul takes 192 cycles on smem.

use nu_model::{Addr, Dep, Range};
use sim_core::{
    build_config_file, Args, Built, Descriptor, Dtype, FatalError, NuCtx, ProfileRegistry,
};

const MAT: u64 = 64 * 64 * 2;

fn two_sm(sets: &[&str]) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_sm.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins()).expect("config")
}

/// two_sm.yaml with the sm1 Neutrino overrides `sm1` ("key=value").
fn built_with(sm1: &[&str]) -> Built {
    let sets: Vec<String> = sm1
        .iter()
        .map(|kv| format!("root.gpc0.sm1.neutrino.{kv}"))
        .collect();
    let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
    two_sm(&sets)
}

/// Version 1 of `r`: every test range is written once unless a test says
/// otherwise.
fn v1(r: Range) -> Dep {
    Dep::Versioned {
        addr: r.base,
        version: 1,
    }
}

/// A versioned matmul into `c`. A forwarded one needs its version
/// ([`Descriptor::with_version`]).
fn matmul(b: &Built, pe: &str, c: Range, deps: Vec<Dep>) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe(pe)),
        dst: c,
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: deps.clone(),
        label: None,
        args: Args::Matmul {
            a: Addr(deps.first().map_or(c.base, |d| d.addr())),
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

#[test]
fn forwarded_invoke_with_a_dep_back_on_the_invoker() {
    let b = built_with(&[]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let a = Range::from_len(b.memory("root.gpc0.sm0.smem").base, MAT);
    let c = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let p0 = matmul(&b, "root.gpc0.sm0.atlas", a, vec![]);
    let m = matmul(&b, "root.gpc0.sm1.atlas", c, vec![v1(a)]).with_inferred_version(1);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(p0).await;
        let h = ctx.invoke(m).await;
        assert_eq!((h.task, h.gid), (None, None), "no local task");
        assert_eq!(h.version, 0, "the owner assigns the version on arrival");
        assert_eq!(ctx.now(), 16, "the invoke returns once queued");
        ctx.wait(h.addr, 1).await;
        assert_eq!(ctx.now(), 452);
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let (tp, tm) = (&s.tasks[0], &s.tasks[1]);
    // P0 at sm0: [8,200), complete 204.
    assert_eq!(tp.complete, Some(204));
    // The Invoke (80 B, 2 beats) is sent at 16 and moves hop by hop: sm0's
    // up bridge [16,20) + 10, the crossbar ports [30,32) and [32,34),
    // sm1's down bridge [34,38) + 10: M enters sm1's scoreboard at 48 with
    // a remote dep on A (the 1-beat Await back to sm0 takes 26 cycles,
    // arrives 74, Registered).
    assert_eq!(tm.nu, sm1, "the task lives at the owner");
    assert_eq!(tm.invoke, 48);
    // Resolve(A) leaves sm0 at 204 and arrives 230: M runs [230,422) on
    // sm1's atlas, completes 426; Resolve(M) for sm0's wait arrives 452.
    assert_eq!(tm.ready, Some(230));
    assert_eq!((tm.chunks[0].start, tm.chunks[0].end), (230, 422));
    assert_eq!(tm.complete, Some(426));
    assert_eq!(s.end_time, 452);
    // Invoke, 2 Await, 2 Registered, 2 Resolve (no acknowledgement).
    assert_eq!(s.messages, 7);
    assert_eq!(s.rejects, 0);
}

#[test]
fn descriptor_assembly_precedes_each_invoke() {
    // forwarded_invoke_with_a_dep_back_on_the_invoker with the default
    // descriptor costs on sm0 (c_base 8, c_dep 3, c_word 1), charged on
    // sm0's scalar before each c_invoke 8:
    // * P0 (no deps; 2 + 2 + 16 + 52 = 72 B = 9 words): c_desc 8 + 9 = 17,
    //   [0, 17), then c_invoke [17, 25); P0 runs [25, 217), completes 221.
    // * M (one dep; 80 B = 10 words): c_desc 8 + 3 + 10 = 21, [25, 46),
    //   then c_invoke [46, 54): queued and sent at 54. Its path as before
    //   (32 cycles): M enters sm1's scoreboard at 86; its Await to sm0
    //   arrives 112, before P0 completes (Registered).
    // * A bridge (capacity 1/2 at the 64 B message beat) has its slot at
    //   odd cycles. Resolve(A) leaves sm0 at 221: sm0's up bridge
    //   [221, 222) + 10, the crossbar ports [232, 234), sm1's down bridge
    //   [234, 236) + 10: it arrives 246. M runs [246, 438), completes 442;
    //   Resolve(M) takes sm1's up bridge [442, 444) + 10, the ports
    //   [454, 456) and sm0's down bridge [456, 458) + 10: for sm0's wait it
    //   arrives 468.
    let b = two_sm(&[
        "root.gpc0.sm0.neutrino.c_base=8",
        "root.gpc0.sm0.neutrino.c_dep=3",
        "root.gpc0.sm0.neutrino.c_word=1",
    ]);
    let sm0 = b.nu("root.gpc0.sm0");
    let a = Range::from_len(b.memory("root.gpc0.sm0.smem").base, MAT);
    let c = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let p0 = matmul(&b, "root.gpc0.sm0.atlas", a, vec![]);
    let m = matmul(&b, "root.gpc0.sm1.atlas", c, vec![v1(a)]).with_inferred_version(1);
    assert_eq!((p0.wire_bytes(), m.wire_bytes()), (72, 80));
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(p0).await;
        assert_eq!(ctx.now(), 25);
        let h = ctx.invoke(m).await;
        assert_eq!(ctx.now(), 54, "the invoke returns once queued");
        ctx.wait(h.addr, 1).await;
        assert_eq!(ctx.now(), 468);
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let (tp, tm) = (&s.tasks[0], &s.tasks[1]);
    assert_eq!((tp.invoke, tp.complete), (25, Some(221)));
    assert_eq!(tm.invoke, 86);
    assert_eq!(tm.ready, Some(246));
    assert_eq!((tm.chunks[0].start, tm.chunks[0].end), (246, 438));
    assert_eq!(s.end_time, 468);
    // sm0's scalar: 17 + 8 + 21 + 8 + c_wait 2.
    let scalar = s
        .resources
        .iter()
        .find(|r| r.name == "root.gpc0.sm0.scalar")
        .unwrap();
    assert_eq!(scalar.busy_cycles, 56);
}

#[test]
fn a_full_scoreboard_holds_forwarded_invokes_in_the_inbound_table() {
    let b = built_with(&["scoreboard=1"]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let s1 = b.memory("root.gpc0.sm1.smem").base;
    let (x, y, c) = (
        Range::from_len(s1, MAT),
        Range::from_len(s1 + MAT, MAT),
        Range::from_len(s1 + 2 * MAT, MAT),
    );
    let (p1, p2) = (
        matmul(&b, "root.gpc0.sm1.atlas", x, vec![]),
        matmul(&b, "root.gpc0.sm1.atlas", y, vec![]),
    );
    let m = matmul(&b, "root.gpc0.sm1.atlas", c, vec![]).with_inferred_version(1);
    let mut sim = b.sim;
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        // P1 runs [8,200); P2 waits for the atlas in the scoreboard (full) until 200.
        ctx.invoke(p1).await;
        ctx.invoke(p2).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(m).await;
        ctx.wait(h.addr, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    // The Invoke (72 B, 2 beats) at 8 arrives 40: the scoreboard is full
    // (P2), so M waits in the inbound invoke table. P2 issues at 200 and
    // frees the entry: M enters then. sm0's wait Await for (C, 1) arrived
    // at 36 and created a pending entry at sm1, which M adopts.
    assert_eq!(s.rejects, 0);
    assert_eq!(s.inbound_invoke_peak[sm1.0 as usize], 1);
    let tm = &s.tasks[2];
    assert_eq!(tm.invoke, 200);
    // The atlas is busy with P2 until 392: M runs [392,584), completes 588;
    // Resolve arrives at sm0 at 614.
    assert_eq!(tm.issue, Some(392));
    assert_eq!(tm.complete, Some(588));
    assert_eq!(s.end_time, 614);
    assert!(s.open_pending.is_empty());
}

#[test]
fn forwarded_invoke_of_a_matmul_on_the_invokers_atlas_is_fatal() {
    // The descriptor's PE must be the owner's (Neutrino spec Section 3).
    let b = built_with(&[]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let c = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let m = matmul(&b, "root.gpc0.sm0.atlas", c, vec![]).with_inferred_version(1);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(m).await;
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    assert!(matches!(f.error, FatalError::Invalid(_)));
    // The Invoke (72 B, 2 beats) sent at 8 arrives at 40.
    assert_eq!((f.nu, f.time), (sm1, 40), "at the owner, on arrival");
}

#[test]
fn forwarded_invoke_of_a_dma_pulling_from_the_invoker() {
    // The canonical case: sm0 forwards a DMA that copies its smem into
    // sm1's smem; sm1 runs it on its engine and charges the path.
    let b = built_with(&[]);
    let sm0 = b.nu("root.gpc0.sm0");
    let (src, dst) = (
        b.memory("root.gpc0.sm0.smem").base,
        b.memory("root.gpc0.sm1.smem").base,
    );
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe("root.gpc0.sm1.dma")),
        dst: Range::from_len(dst, 8192),
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: Some(1),
        deps: vec![],
        label: None,
        args: Args::Dma {
            src: Addr(src),
            dst: Addr(dst),
            bytes: 8192,
            tile_bytes: 8192,
        },
    };
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait(h.addr, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let t = &s.tasks[0];
    // The Invoke (52 B, 1 beat) arrives at sm1 at 34 and issues; the
    // request's trip from sm1's engine to sm0's SMEM (sm1's bridge up and
    // sm0's bridge down, 10 each; 2026-10-21), sm0's SMEM latency 10 (the
    // pipeline fill) and the engine latency 100: the hops in route order
    // from 164, at the engine's 64 B beat narrowed by each hop: sm0's SMEM
    // port [164, 292); sm0's bridge up (32 B/cycle) [164, 420), latency
    // 10; the crossbar ports and sm1's bridge down narrowed to 32,
    // [174, 430), latency 10; sm1's SMEM port [184, 440). Complete 444;
    // Resolve arrives at sm0 at 470.
    assert_eq!(t.invoke, 34);
    assert_eq!((t.chunks[0].start, t.chunks[0].end), (164, 440));
    assert_eq!(t.complete, Some(444));
    assert_eq!(s.end_time, 470);
}

#[test]
fn a_forwarded_invoke_gets_the_owners_version_and_the_inference_matches() {
    // sm0 forwards two writes of X to sm1: the handles carry version 0,
    // the owner assigns 1 and 2 on arrival, and the inferences (1, 2) match
    // (checked by the owner in debug builds).
    let b = built_with(&[]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let x = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let atlas = "root.gpc0.sm1.atlas";
    let w1 = matmul(&b, atlas, x, vec![]).with_inferred_version(1);
    let w2 = matmul(&b, atlas, x, vec![v1(x)]).with_inferred_version(2);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h1 = ctx.invoke(w1).await;
        let h2 = ctx.invoke(w2).await;
        assert_eq!((h1.version, h2.version), (0, 0));
        ctx.wait(x.base, 2).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let versions: Vec<_> = s.tasks.iter().map(|t| (t.nu, t.version)).collect();
    assert_eq!(versions, vec![(sm1, Some(1)), (sm1, Some(2))]);
}

#[test]
#[cfg(debug_assertions)]
fn a_wrong_inference_is_fatal_in_debug_builds() {
    let b = built_with(&[]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let x = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let w = matmul(&b, "root.gpc0.sm1.atlas", x, vec![]).with_inferred_version(3);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(w).await;
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    assert_eq!(f.nu, sm1, "at the owner");
    let FatalError::Invalid(msg) = f.error else {
        panic!("{f:?}");
    };
    assert!(
        msg.contains("infers version 3, the owner assigns 1"),
        "{msg}"
    );
}

#[test]
fn an_await_from_a_third_neutrino_before_the_producers_invoke_binds() {
    // sm0 forwards C (a copy of X from sm1's SMEM into HBM, dep (X, 1)) to
    // the device, waits 200 cycles, then forwards P (the writer of X) to
    // sm1. The device's Await for (X, 1) reaches sm1 before P's Invoke: the
    // counter (none) is below 1, so a pending entry for (X, 1) waits, and
    // P adopts it.
    let b = built_with(&[]);
    let (sm0, sm1, dev) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"), b.nu("root"));
    let x = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let h = Range::from_len(b.memory("root.hbm").base, MAT);
    let c = Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe("root.dma")),
        dst: h,
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: Some(1),
        deps: vec![v1(x)],
        label: None,
        args: Args::Dma {
            src: Addr(x.base),
            dst: Addr(h.base),
            bytes: MAT,
            tile_bytes: MAT,
        },
    };
    let p = matmul(&b, "root.gpc0.sm1.atlas", x, vec![]).with_inferred_version(1);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(c).await;
        ctx.cycles(200).await;
        ctx.invoke(p).await;
        ctx.wait(h.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    let (tc, tp) = (&s.tasks[0], &s.tasks[1]);
    assert_eq!((tc.nu, tp.nu), (dev, sm1));
    // C (60 B, 1 beat) is sent at 8 and enters the device at 43 (sm0's up
    // bridge 2 + 10, two crossbar ports, the cluster bridge 1 + 20); its
    // Await for (X, 1) reaches sm1 at 78 (the cluster bridge 1 + 20, two
    // ports, sm1's down bridge 2 + 10) and waits on a pending entry. P
    // (72 B, 2 beats) is invoked at [208, 216) and enters sm1 at 248,
    // adopting the entry; it runs [248, 440) and completes at 444. Resolve
    // reaches the device at 479: C is ready. C's DMA: the request's trip
    // from the device's engine to sm1's SMEM (the cluster bridge down 20
    // and sm1's bridge down 10: 30; 2026-10-21), sm1's SMEM latency 10
    // (the pipeline fill) and the engine latency 100, its first hop (sm1's
    // SMEM port) at 619; 8 KiB at 32 B/cycle on sm1's bridge up from 619,
    // then the crossbar ports and the cluster bridge narrowed to 32 from
    // 629 and the HBM port from 649, to 905; complete 909; Resolve reaches
    // sm0's wait at 944.
    assert_eq!((tc.invoke, tp.invoke), (43, 248));
    assert_eq!((tp.complete, tc.ready), (Some(444), Some(479)));
    assert_eq!(
        (tc.chunks[0].start, tc.chunks[0].end, tc.complete),
        (619, 905, Some(909))
    );
    assert_eq!(s.end_time, 944);
}

#[test]
fn backpressure_holds_invokes_across_two_hops() {
    // sm1: one scoreboard entry, one inbound invoke table entry. P1 runs
    // [8,200); P2 holds the scoreboard entry until it issues at 200. sm0
    // forwards M1, M2 and M3 (72 B, 2 beats each) at 8, 16 and 24: M1 takes
    // the inbound entry, M2 waits in the last stage (after sm1's down
    // bridge) and M3 in the stage before it (after the crossbar port to
    // sm1); no invoke is rejected. Stages hold one message each, from the
    // start of the transmission into them until the message moves on.
    //
    // M1: sm0's up bridge [8,12), in stage 1 at 22; port [22,24), stage 2
    // at 24; port [24,26); sm1's down bridge [26,30) + 10: the inbound
    // table at 40.
    // M2: waits in sm0's source queue for stage 1 until 24 (M1 left it);
    // [24,28), stage 1 at 38; [38,40); [40,42); [42,46) + 10: stage 4 at
    // 56, where it waits for the inbound entry (held by M1 until 200).
    // M3: waits for stage 1 until M2's transmission out of it ends at 40;
    // [40,44), stage 1 at 54; [54,56), stage 2 at 56; [56,58) (M2 left
    // stage 3 at 46), stage 3 at 58, where it waits for stage 4 (M2).
    // At 200 P2 issues and M1 enters the scoreboard: M2 moves into the
    // inbound table, M3 takes stage 4 ([200,204) + 10, at 214) and waits
    // there for M2. At 392 M1 issues, M2 enters and M3 moves into the
    // table; at 584 M2 issues and M3 enters.
    let b = built_with(&["scoreboard=1", "inbound_invokes=1"]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let s1 = b.memory("root.gpc0.sm1.smem").base;
    let at = |i: u64| Range::from_len(s1 + i * MAT, MAT);
    let atlas = "root.gpc0.sm1.atlas";
    let (p1, p2) = (
        matmul(&b, atlas, at(0), vec![]),
        matmul(&b, atlas, at(1), vec![]),
    );
    let ms: Vec<Descriptor> = (2..5)
        .map(|i| matmul(&b, atlas, at(i), vec![]).with_inferred_version(1))
        .collect();
    let mut sim = b.sim;
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        ctx.invoke(p1).await;
        let h = ctx.invoke(p2).await;
        ctx.wait(h.addr, h.version).await;
    });
    let last = at(4);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        for m in ms {
            ctx.invoke(m).await;
        }
        ctx.wait(last.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    assert_eq!(s.rejects, 0);
    let (t1, t2, t3) = (&s.tasks[2], &s.tasks[3], &s.tasks[4]);
    assert_eq!((t1.invoke, t2.invoke, t3.invoke), (200, 392, 584));
    let hol = |name: &str| {
        s.stages
            .iter()
            .find(|x| x.stage == name)
            .map_or(0, |x| x.hol_cycles)
    };
    // Head-of-line cycles: the stage after sm1's down bridge M2 [56,200)
    // and M3 [214,392); the stage after the port to sm1 M3 [58,200); none
    // after the port from sm0; sm0's source queue M2 [16,24) and M3
    // [24,40).
    assert_eq!(hol("root.gpc0.sm1.smcbus_to_clcbus.ba.invoke"), 144 + 178);
    assert_eq!(hol("root.gpc0.clcbus.sm1.smcbus_to_clcbus.out.invoke"), 142);
    assert_eq!(hol("root.gpc0.clcbus.sm0.smcbus_to_clcbus.in.invoke"), 0);
    assert_eq!(hol("root.gpc0.sm0.source.invoke"), 8 + 16);
}

#[test]
fn the_deadlock_detector_fires_on_a_cycle() {
    // sm1: one scoreboard entry, one inbound invoke table entry. Its own
    // task T waits forever for (X5, 1), which nobody produces, and holds
    // the scoreboard entry; sm0's M1 waits in the inbound table for it and
    // M2 in the channel for M1. No event is left: the run ends with a
    // Deadlock that names the stage holding M2.
    let b = built_with(&["scoreboard=1", "inbound_invokes=1"]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let s1 = b.memory("root.gpc0.sm1.smem").base;
    let at = |i: u64| Range::from_len(s1 + i * MAT, MAT);
    let t = matmul(&b, "root.gpc0.sm1.atlas", at(0), vec![v1(at(5))]);
    let (m1, m2) = (
        matmul(&b, "root.gpc0.sm1.atlas", at(1), vec![]).with_inferred_version(1),
        matmul(&b, "root.gpc0.sm1.atlas", at(2), vec![]).with_inferred_version(1),
    );
    let mut sim = b.sim;
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        ctx.invoke(t).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(m1).await;
        ctx.invoke(m2).await;
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    let FatalError::Deadlock { tables, stages, .. } = f.error else {
        panic!("{f:?}");
    };
    let t1 = &tables[sm1.0 as usize];
    assert_eq!((t1.scoreboard, t1.inbound_invokes, t1.pending), (1, 1, 1));
    assert_eq!(tables[sm0.0 as usize].outbound_invokes, 0);
    assert_eq!(stages.len(), 1, "{stages:?}");
    assert_eq!(stages[0].stage, "root.gpc0.sm1.smcbus_to_clcbus.ba.invoke");
    assert_eq!(stages[0].messages, 1);
}

#[test]
fn a_join_with_20_deps_travels_in_3_beats() {
    // 2 + 2 + 16 + 8 x 20 = 180 B: 3 beats, 20 + 6 x 3 = 38 cycles. sm1
    // first writes version 1 of 20 ranges with local joins (done by 208);
    // sm0 forwards the join on them at 208, so it is ready on arrival.
    let b = built_with(&[]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    // Joins live in the owner's sync-only space.
    let s1 = b.sync_only(sm1).unwrap().base;
    let at = |i: u64| Range::from_len(s1 + i * 512, 512);
    let j = at(0);
    let signals: Vec<Descriptor> = (1..21)
        .map(|i| {
            let mut d = Descriptor::join(at(i), vec![]);
            d.versioned = true;
            d
        })
        .collect();
    let mut join =
        Descriptor::join(j, (1..21).map(|i| v1(at(i))).collect()).with_inferred_version(1);
    join.versioned = true;
    assert_eq!(join.wire_bytes(), 180);
    let mut sim = b.sim;
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        for d in signals {
            ctx.invoke(d).await;
        }
        ctx.wait(j.base, 1).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.cycles(200).await;
        ctx.invoke(join).await;
        ctx.wait(j.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let t = &s.tasks[20];
    assert_eq!((t.invoke, t.ready), (208 + 38, Some(246)));
    // The join, then the Await, AwaitAck and Resolve of sm0's wait.
    assert_eq!(s.beats, [3, 0, 1, 0]);
}

#[test]
fn a_version_reset_takes_effect_after_the_invokes_before_it() {
    // sm1's scoreboard holds one entry, taken by P2 until 200, so sm0's
    // forwarded M1 and M2 (versions 1 and 2 of C) wait in sm1's inbound
    // invoke table; sm0's VersionReset of C arrives behind them and waits
    // too (all messages of the invoke protocol from one requester take
    // effect in program order, 2026-10-20). It frees C's version entry
    // only after M1 and M2 have entered, so M3, invoked after M2 is done,
    // is version 1 again (an immediate free on arrival would have found no
    // entry and M3 would be version 3).
    let b = built_with(&["scoreboard=1"]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let s1 = b.memory("root.gpc0.sm1.smem").base;
    let (x, y, c) = (
        Range::from_len(s1, MAT),
        Range::from_len(s1 + MAT, MAT),
        Range::from_len(s1 + 2 * MAT, MAT),
    );
    let atlas = "root.gpc0.sm1.atlas";
    let (p1, p2) = (matmul(&b, atlas, x, vec![]), matmul(&b, atlas, y, vec![]));
    let ms: Vec<Descriptor> = (1..=3)
        .map(|v| matmul(&b, atlas, c, vec![]).with_inferred_version(if v == 3 { 1 } else { v }))
        .collect();
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        ctx.invoke(p1).await;
        ctx.invoke(p2).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let mut ms = ms.into_iter();
        ctx.invoke(ms.next().unwrap()).await;
        ctx.invoke(ms.next().unwrap()).await;
        ctx.vreset(c.base).await;
        ctx.wait(c.base, 2).await;
        ctx.invoke(ms.next().unwrap()).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let versions: Vec<u64> = s
        .tasks
        .iter()
        .filter(|t| t.nu == sm1)
        .skip(2)
        .map(|t| t.version.unwrap())
        .collect();
    assert_eq!(versions, vec![1, 2, 1]);
    // M1, M2 and the VersionReset were in the inbound table at once.
    assert_eq!(s.inbound_invoke_peak[sm1.0 as usize], 3);
}

/// The fatal error of a run of `program` on sm0 of two_sm.yaml.
fn fatal_on_sm0(
    b: Built,
    program: impl FnOnce(NuCtx) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()>>> + 'static,
) -> sim_core::Fatal {
    let sm0 = b.nu("root.gpc0.sm0");
    let mut sim = b.sim;
    sim.spawn(sm0, program);
    sim.run().fatal.expect("fatal")
}

#[test]
fn a_join_outside_the_sync_only_space_is_invalid() {
    // A join writes no data: its range must be in its owner's sync-only
    // space (2026-10-20), whether it is invoked locally or forwarded.
    let b = built_with(&[]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let local = Descriptor::join(
        Range::from_len(b.memory("root.gpc0.sm0.smem").base, 512),
        vec![],
    );
    let f = fatal_on_sm0(b, move |ctx| {
        Box::pin(async move {
            ctx.invoke(local).await;
        })
    });
    assert_eq!(f.nu, sm0);
    assert!(
        matches!(&f.error, FatalError::Invalid(m) if m.contains("sync-only")),
        "{f:?}"
    );
    let b = built_with(&[]);
    let mut remote = Descriptor::join(
        Range::from_len(b.memory("root.gpc0.sm1.smem").base, 512),
        vec![],
    );
    remote.versioned = true;
    let remote = remote.with_inferred_version(1);
    let f = fatal_on_sm0(b, move |ctx| {
        Box::pin(async move {
            ctx.invoke(remote).await;
        })
    });
    assert_eq!(f.nu, sm1, "at the owner, on admission");
    assert!(
        matches!(&f.error, FatalError::Invalid(m) if m.contains("sync-only")),
        "{f:?}"
    );
    // The same joins in the sync-only space run.
    let b = built_with(&[]);
    let j = Descriptor::join(Range::from_len(b.sync_only(sm0).unwrap().base, 512), vec![]);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(j).await;
        ctx.wait_unversioned(h.addr).await;
    });
    assert_eq!(sim.run().fatal, None);
}

#[test]
fn a_spawn_whose_done_range_is_outside_the_sync_only_space_is_invalid() {
    // The parent owns the done range, in its sync-only space; the root's
    // HBM and the child's sync-only space are both refused.
    for done in ["root.hbm", "child sync"] {
        let b = built_with(&[]);
        let (root, gpc0) = (b.nu("root"), b.nu("root.gpc0"));
        let r = match done {
            "root.hbm" => Range::from_len(b.memory("root.hbm").base, 512),
            _ => Range::from_len(b.sync_only(gpc0).unwrap().base, 512),
        };
        let mut sim = b.sim;
        sim.spawn(root, move |ctx: NuCtx| async move {
            ctx.spawn_child(gpc0, r, vec![], |_c: NuCtx| async move {})
                .await;
        });
        let f = sim.run().fatal.expect("fatal");
        assert_eq!(f.nu, root, "{done}");
        assert!(
            matches!(&f.error, FatalError::Invalid(m) if m.contains("sync-only")),
            "{done}: {f:?}"
        );
    }
}
