//! Step 4: configuration, fabric paths, spawn_child, remote dependencies,
//! remote loads. Config: `configs/two_sm.yaml`.
//!
//! Paths used below. sm0 -> sm1: sm0's bridge up (32 B/cycle, 10), the
//! gpc0 crossbar clsbus (an xbar: sm0's ingress port and sm1's egress
//! port, 64 B/cycle each), sm1's bridge down (32 B/cycle, 10): latency 20.
//! A 64 B beat takes 2 cycles on a bridge and 1 on a port. Control messages
//! go store and forward through one-message stages, so a message of b beats
//! arrives 2b + 10 + b + b + 2b + 10 = 20 + 6b cycles after it is sent when
//! the path is free (1 beat: 26; a 72 B matmul descriptor, 2 beats: 32).
//! Transfers
//! between disjoint port pairs of the crossbar do not contend. Costs: c_invoke 8, c_wait 2, c_complete 4, c_spawn 16.
//! A 64^3 bf16 matmul: 64 cycles on the array, 24576 B / 128 = 192 cycles
//! on smem.

use nu_model::{Addr, Dep, Range};
use sim_core::{
    build_config_file, Args, Blocked, Built, Descriptor, Dtype, FatalError, NuCtx, ProfileRegistry,
    ResourceId, Stats,
};

const MAT: u64 = 64 * 64 * 2;

fn two_sm(sets: &[&str]) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_sm.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins()).expect("config")
}

fn built() -> Built {
    two_sm(&[])
}

fn res(s: &Stats, name: &str) -> ResourceId {
    ResourceId(
        s.resources
            .iter()
            .position(|r| r.name == name)
            .unwrap_or_else(|| panic!("no resource {name}")) as u32,
    )
}

fn v1(range: Range) -> Dep {
    Dep::Versioned {
        addr: range.base,
        version: 1,
    }
}

/// A versioned matmul into `c` (a local invoke: the owner assigns the
/// version).
fn matmul(b: &Built, pe: &str, c: Range, deps: Vec<Dep>) -> Descriptor {
    let base = c.base;
    Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe(pe)),
        dst: c,
        versioned: true,
        inferred_version: None,
        deps,
        label: None,
        args: Args::Matmul {
            a: Addr(base),
            b: Addr(base),
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
fn config_builds_owners_and_paths() {
    let b = built();
    let (sm0, sm1, cl) = (
        b.nu("root.gpc0.sm0"),
        b.nu("root.gpc0.sm1"),
        b.nu("root.gpc0"),
    );
    assert_eq!(b.own_space(sm0), Range::from_len(0x500_0000, 1 << 24));
    assert_eq!(b.own_space(sm1), Range::from_len(0x600_0000, 1 << 24));
    assert_eq!(
        b.memory("root.gpc0.sm1.smem"),
        Range::from_len(0x600_0000, 512 << 10)
    );
    assert_eq!(b.sync_only(cl), Some(Range::from_len(0x4F0_0000, 1 << 20)));
    let topo = b.sim.topology();
    let p = topo.path(sm0, sm1);
    assert_eq!(p.latency, 20);
    let names: Vec<&str> = p
        .resources
        .iter()
        .map(|r| b.sim_resource_name(*r))
        .collect();
    assert_eq!(
        names,
        vec![
            "root.gpc0.sm0.smsbus_to_clsbus.ab",
            "root.gpc0.clsbus.sm0.smsbus_to_clsbus.in",
            "root.gpc0.clsbus.sm1.smsbus_to_clsbus.out",
            "root.gpc0.sm1.smsbus_to_clsbus.ba"
        ]
    );
    // From the cluster's Neutrino (an attachment of clsbus) to sm0: its
    // ingress port, sm0's egress port, sm0's bridge down.
    let p = topo.path(cl, sm0);
    assert_eq!(p.latency, 10);
    let names: Vec<&str> = p
        .resources
        .iter()
        .map(|r| b.sim_resource_name(*r))
        .collect();
    assert_eq!(
        names,
        vec![
            "root.gpc0.clsbus.neutrino.in",
            "root.gpc0.clsbus.sm0.smsbus_to_clsbus.out",
            "root.gpc0.sm0.smsbus_to_clsbus.ba"
        ]
    );
    // Up to the device: two links.
    assert_eq!(topo.path(sm1, b.nu("root")).latency, 30);
    assert!(!b.numeric);
}

/// SM1 produces X (a matmul); SM0 consumes X by a remote dep. `delay`
/// holds SM1 back so that the consumer comes first.
fn remote_dep(delay: u64) -> Stats {
    let b = built();
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let x = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let c = Range::from_len(b.memory("root.gpc0.sm0.smem").base, MAT);
    let p = matmul(&b, "root.gpc0.sm1.atlas", x, vec![]);
    let q = matmul(&b, "root.gpc0.sm0.atlas", c, vec![v1(x)]);
    let mut sim = b.sim;
    // Each producer program waits for its output: its return frees its
    // version table (Section 3.2), which a later remote lookup needs.
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        if delay > 0 {
            ctx.cycles(delay).await;
        }
        let h = ctx.invoke(p).await;
        ctx.wait(h.addr, h.version).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(q).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty(), "{:?}", s.unfinished);
    s
}

#[test]
fn remote_dep_producer_first() {
    let s = remote_dep(0);
    let (p, q) = (&s.tasks[0], &s.tasks[1]);
    // P: invoke [0,8), smem [8,200), complete [200,204).
    assert_eq!(p.complete, Some(204));
    // Q enters at 8 and sends Await. A 1-beat message between the SMs
    // moves hop by hop: sm0's up bridge 2 cycles + 10, the two crossbar
    // ports 1 cycle each, sm1's down bridge 2 cycles + 10: 26 cycles. The
    // Await arrives at 34, P executing: Registered. Resolve leaves at 204
    // and arrives at 230; Q issues at 230.
    assert_eq!(q.invoke, 8);
    assert_eq!(q.ready, Some(230));
    assert_eq!((q.chunks[0].start, q.chunks[0].end), (230, 422));
    assert_eq!(q.complete, Some(426));
    assert_eq!(s.end_time, 426);
    assert_eq!(s.messages, 3, "Await, AwaitAck(registered), Resolve");
}

#[test]
fn remote_dep_consumer_first_goes_through_the_owners_pending() {
    let s = remote_dep(100);
    let (q, p) = (&s.tasks[0], &s.tasks[1]);
    // Await arrives at SM1 at 34 and creates a pending entry. P is invoked at
    // 108 (cycles [0,100), invoke [100,108)), adopts it, runs [108,300),
    // completes at 304; Resolve arrives at 330 (26 cycles).
    assert_eq!(p.invoke, 108);
    assert_eq!(p.complete, Some(304));
    assert_eq!(q.ready, Some(330));
    assert_eq!(q.complete, Some(330 + 192 + 4));
    assert!(s.open_pending.is_empty());
}

#[test]
fn reject_and_retry() {
    // SM1 has room for one inbound ref. Two consumers on SM0 depend on X and
    // Y at SM1; Y's Await is rejected until X's ref frees.
    let b = two_sm(&["root.gpc0.sm1.neutrino.inbound_refs=1"]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let smem1 = b.memory("root.gpc0.sm1.smem").base;
    let smem0 = b.memory("root.gpc0.sm0.smem").base;
    let (x, y) = (
        Range::from_len(smem1, MAT),
        Range::from_len(smem1 + MAT, MAT),
    );
    let (cx, cy) = (
        Range::from_len(smem0, MAT),
        Range::from_len(smem0 + MAT, MAT),
    );
    let (px, py) = (
        matmul(&b, "root.gpc0.sm1.atlas", x, vec![]),
        matmul(&b, "root.gpc0.sm1.atlas", y, vec![]),
    );
    let (qx, qy) = (
        matmul(&b, "root.gpc0.sm0.atlas", cx, vec![v1(x)]),
        matmul(&b, "root.gpc0.sm0.atlas", cy, vec![v1(y)]),
    );
    let mut sim = b.sim;
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        ctx.invoke(px).await;
        let h = ctx.invoke(py).await;
        ctx.wait(h.addr, h.version).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(qx).await;
        ctx.invoke(qy).await;
        ctx.wait(cx.base, 1).await;
        ctx.wait(cy.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let (tpx, tpy, tqx, tqy) = (&s.tasks[0], &s.tasks[2], &s.tasks[1], &s.tasks[3]);
    // PX [8,200) done 204; PY waits for the atlas: [200,392), done 396.
    assert_eq!((tpx.complete, tpy.complete), (Some(204), Some(396)));
    // Await X at 8: sm0's up bridge [8,10), in its stage at 20, the ports
    // [20,21) and [21,22), sm1's down bridge [22,24), arrives 34:
    // Registered. Await Y at 16 waits for the one-message stage after
    // sm0's up bridge until X leaves it at 21: [21,23), arrives 47: Reject
    // (ack arrives 73), resend at 123 arrives 149: Reject (ack 175), resend
    // at 225 arrives 251: Registered on PY (X's ref freed at 204). PX's
    // Resolve leaves at 204 and arrives at 230.
    assert_eq!(tqx.ready, Some(230));
    // QX [230,422), done 426. Resolve Y leaves at 396 and arrives at 422,
    // when the atlas frees: QY issues at 422.
    assert_eq!(tqy.ready, Some(422));
    assert_eq!(tqy.issue, Some(422));
    assert_eq!(tqy.complete, Some(422 + 192 + 4));
    assert_eq!(s.end_time, 618);
    // Messages: 2 Await + 2 resend + 4 acks + 2 Resolve.
    assert_eq!(s.messages, 10);
}

#[test]
fn parent_spawns_two_children_and_waits() {
    let b = built();
    let cl = b.nu("root.gpc0");
    let so = b.sync_only(cl).unwrap();
    let (d0, d1) = (
        Range::from_len(so.base, 512),
        Range::from_len(so.base + 512, 512),
    );
    let mut sim = b.sim;
    sim.spawn(cl, move |ctx: NuCtx| async move {
        let kids = ctx.children().to_vec();
        ctx.spawn_child(kids[0], d0, vec![], |c: NuCtx| async move {
            c.cycles(100).await;
        })
        .await;
        ctx.spawn_child(kids[1], d1, vec![], |c: NuCtx| async move {
            c.cycles(100).await;
        })
        .await;
        ctx.wait_unversioned(d0.base).await;
        ctx.wait_unversioned(d1.base).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let (s0, s1) = (&s.tasks[0], &s.tasks[1]);
    // Spawn 0: c_spawn [0,16); Start down: the crossbar ports [16,17) and
    // [17,18), sm0's down bridge [18,20) + 10: arrives 30; child cycles
    // [30,130); Return up: sm0's up bridge [130,132) + 10, the ports
    // [142,143) and [143,144): arrives 144; complete [144,148).
    assert_eq!(s0.issue, Some(16));
    assert_eq!(s0.complete, Some(148));
    // Spawn 1: c_spawn [16,32); arrives 46; [46,146); Return 160; 164.
    assert_eq!(s1.complete, Some(164));
    assert_eq!(s.end_time, 164);
    assert_eq!(s.messages, 4);
}

#[test]
fn a_program_start_resets_the_version_table() {
    // The cluster spawns a program on sm0 that sets X's counter to 5 and
    // writes X (version 6), then, after it returned, a second program that
    // writes X again: the start of the second program cleared the table,
    // so the write gets version 1.
    let b = built();
    let cl = b.nu("root.gpc0");
    let kids = b.sim.topology().node(cl).children.clone();
    let so = b.sync_only(cl).unwrap();
    let (d0, d1) = (
        Range::from_len(so.base, 512),
        Range::from_len(so.base + 512, 512),
    );
    let x = Range::from_len(b.memory("root.gpc0.sm0.smem").base, MAT);
    let w1 = matmul(&b, "root.gpc0.sm0.atlas", x, vec![]);
    let w2 = w1.clone();
    let mut sim = b.sim;
    sim.spawn(cl, move |ctx: NuCtx| async move {
        ctx.spawn_child(kids[0], d0, vec![], move |c: NuCtx| async move {
            c.version(x.base, 5).await;
            let h = c.invoke(w1).await;
            assert_eq!(h.version, 6);
            c.wait(h.addr, h.version).await;
        })
        .await;
        ctx.wait_unversioned(d0.base).await;
        ctx.spawn_child(kids[0], d1, vec![], move |c: NuCtx| async move {
            let h = c.invoke(w2).await;
            assert_eq!(h.version, 1, "reset at the start");
            c.wait(h.addr, h.version).await;
        })
        .await;
        ctx.wait_unversioned(d1.base).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let versions: Vec<_> = s.tasks.iter().filter_map(|t| t.version).collect();
    assert_eq!(versions, vec![6, 1]);
}

#[test]
fn spawn_on_a_non_child_is_fatal() {
    let b = built();
    let (cl, dev) = (b.nu("root.gpc0"), b.nu("root"));
    let d = Range::from_len(b.sync_only(cl).unwrap().base, 512);
    let mut sim = b.sim;
    sim.spawn(cl, move |ctx: NuCtx| async move {
        ctx.spawn_child(dev, d, vec![], |_c: NuCtx| async move {})
            .await;
    });
    let s = sim.run();
    assert!(matches!(s.fatal.unwrap().error, FatalError::Invalid(_)));
}

#[test]
fn dma_between_sms_charges_both_bridges_and_two_crossbar_ports() {
    let b = built();
    let sm0 = b.nu("root.gpc0.sm0");
    let (src, dst) = (
        b.memory("root.gpc0.sm1.smem").base,
        b.memory("root.gpc0.sm0.smem").base,
    );
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe("root.gpc0.sm0.dma")),
        dst: Range::from_len(dst, 8192),
        versioned: true,
        inferred_version: None,
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
        ctx.wait(h.addr, h.version).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let t = &s.tasks[0];
    // Issue 8, sm1's SMEM latency 10 (the pipeline fill) and the engine
    // latency 100: the hops in route order from 118, at the engine's 64 B
    // beat narrowed by each hop (Section 5.4): sm1's SMEM port (128
    // B/cycle) at 64, one of its two slots, [118, 246); sm1's bridge up
    // (32 B/cycle, half a 64 B slot) [118, 374), latency 10; the crossbar
    // ports (64 B/cycle, one slot) narrowed to 32, half a slot, [128, 384);
    // sm0's bridge down [128, 384), latency 10; sm0's SMEM port narrowed to
    // 32, [138, 394). The bridges and the ports each hold half a slot for
    // 256 cycles: 128 slot-cycles.
    assert_eq!((t.chunks[0].start, t.chunks[0].end), (118, 394));
    assert_eq!(t.complete, Some(398));
    for name in [
        "root.gpc0.sm1.smsbus_to_clsbus.ab",
        "root.gpc0.clsbus.sm1.smsbus_to_clsbus.in",
        "root.gpc0.clsbus.sm0.smsbus_to_clsbus.out",
        "root.gpc0.sm0.smsbus_to_clsbus.ba",
    ] {
        assert_eq!(
            s.resources[res(&s, name).0 as usize].busy_cycles,
            128,
            "{name}"
        );
    }
    // The first hop with the longest window: sm1's bridge up, the
    // narrowest (the hops after it are narrowed to its rate).
    assert_eq!(
        t.chunks[0].culprit,
        Some(res(&s, "root.gpc0.sm1.smsbus_to_clsbus.ab"))
    );
}

#[test]
fn remote_load_charges_the_path_twice_in_latency() {
    // Numeric fidelity for the whole tree, inherited from the device.
    let b = two_sm(&["root.fidelity=numeric"]);
    assert!(b.numeric);
    let sm0 = b.nu("root.gpc0.sm0");
    let src = b.memory("root.gpc0.sm1.smem").base;
    let mut sim = b.sim;
    sim.set_functional(Box::new(func::Numeric::new()));
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let v = ctx.load(Addr(src), 64).await;
        assert_eq!(v, vec![0; 64]);
    });
    let s = sim.run();
    // c_load [0,1); latency 10 + 2 * 20; 64 B: 2 cycles on the links.
    assert_eq!(s.fatal, None);
    assert_eq!(s.end_time, 1 + 50 + 2);
}

#[test]
fn remote_store_pays_the_path_latency_and_no_block_latency() {
    // The path of `remote_load_charges_the_path_twice_in_latency`: a store
    // takes the route once (20) and the time of its bytes (64 B: 2 cycles
    // on the links); sm1's SMEM latency (10) is not paid by a write.
    let b = built();
    let sm0 = b.nu("root.gpc0.sm0");
    let dst = b.memory("root.gpc0.sm1.smem").base;
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.store(Addr(dst), vec![0u8; 64]).await;
    });
    let s = sim.run();
    // c_store [0,1); route latency 20; 2 cycles of bytes.
    assert_eq!(s.fatal, None);
    assert_eq!(s.end_time, 1 + 20 + 2);
}

#[test]
fn remote_load_race_check_reads_the_owners_tables() {
    let b = built();
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let x = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let p = matmul(&b, "root.gpc0.sm1.atlas", x, vec![]);
    let mut sim = b.sim;
    sim.set_functional(Box::new(func::Numeric::new()));
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        ctx.invoke(p).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        // P runs [8,200); the load reads X at 53.
        ctx.load(Addr(x.base), 64).await;
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    assert_eq!(
        f.error,
        FatalError::Race {
            range: Range::from_len(x.base, 64),
            writer: 0
        }
    );
    assert_eq!(f.time, 53);
}

#[test]
fn remote_wait_blocks_until_ready() {
    let b = built();
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let x = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let p = matmul(&b, "root.gpc0.sm1.atlas", x, vec![]);
    let mut sim = b.sim;
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        let h = ctx.invoke(p).await;
        ctx.wait(h.addr, h.version).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.wait(x.base, 1).await;
        // Await at 2 arrives 28 (Registered); Resolve leaves at 204 and
        // arrives 26 cycles later.
        assert_eq!(ctx.now(), 230);
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
}

#[test]
fn null_fidelity_rejects_a_load() {
    let b = built();
    let sm0 = b.nu("root.gpc0.sm0");
    let src = b.memory("root.gpc0.sm1.smem").base;
    let mut sim = b.sim;
    sim.set_functional(Box::new(func::Numeric::new()));
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.load(Addr(src), 64).await;
    });
    let s = sim.run();
    assert!(matches!(s.fatal.unwrap().error, FatalError::Func { .. }));
    assert!(!s.unfinished.contains(&(sm0, Blocked::Wait)));
}
