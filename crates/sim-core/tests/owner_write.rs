//! Writes into another node's memory (Neutrino spec Section 3): a DMA task invoked at
//! the memory's owner (forwarded there) runs on the owner's copy engine as
//! a pull from the source node. configs/two_sm.yaml: first co-charge rule,
//! control links at the data rate; a 64 B message between an SM and the
//! device: 2 cycles on the 32 B/cycle SM bridge lane, latency 10 + 20.

use nu_model::{Addr, Dep, Range};
use sim_core::{build_config_file, Args, Built, Descriptor, FatalError, NuCtx, ProfileRegistry};

const KB64: u64 = 64 << 10;

fn two_sm(sets: &[&str]) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_sm.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins()).expect("config")
}

fn dma(b: &Built, pe: &str, src: u64, dst: Range, tile: u64) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe(pe)),
        dst,
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        pc: None,
        call_path: Vec::new(),
        deps: vec![],
        label: None,
        args: Args::Dma {
            src: Addr(src),
            dst: Addr(dst.base),
            bytes: dst.len(),
            tile_bytes: tile,
        },
    }
}

fn v1(range: Range) -> Dep {
    Dep::Versioned {
        addr: range.base,
        version: 1,
    }
}

/// sm0 writes 64 KiB of its SMEM into HBM through the device's copy engine
/// and waits for it.
fn write_into_hbm(sets: &[&str]) -> sim_core::Stats {
    let b = two_sm(sets);
    let sm0 = b.nu("root.gpc0.sm0");
    let s0 = b.memory("root.gpc0.sm0.smem").base;
    let h = Range::from_len(b.memory("root.hbm").base, KB64);
    let write = dma(&b, "root.dma", s0, h, KB64).with_inferred_version(1);
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(write).await;
        ctx.wait(h.base, 1).await;
    });
    sim.run()
}

#[test]
fn a_write_into_hbm_is_a_pull_by_the_device() {
    let s = write_into_hbm(&[]);
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    // invoke [0,8); the Invoke (52 B, 1 beat) is sent at 8 and moves hop
    // by hop: sm0's up bridge 2 cycles + 10, the crossbar ports 1 cycle
    // each, the cluster bridge 1 cycle + 20: arrives 43. The device enters
    // and issues the task on its copy engine at 43.
    let t = &s.tasks[0];
    assert_eq!((t.invoke, t.ready, t.issue), (43, Some(43), Some(43)));
    // The request's trip from the device's engine to sm0's SMEM (gpc0's
    // bridge down 20 and sm0's bridge down 10: 30; 2026-10-21), sm0's SMEM
    // latency 10 (the pipeline fill) and the engine latency 100: the hops
    // in route order from 183 at the engine's 64 B beat, narrowed by each
    // hop (r2sim spec Section 5.6): sm0's SMEM port (one of its two slots) [183, 1207);
    // sm0's bridge up (32 B/cycle) [183, 2231), latency 10; the crossbar
    // ports and gpc0's bridge up narrowed to 32, [193, 2241), latency 20;
    // the HBM port [213, 2261). The chunk runs from its first hop's start
    // to its last hop's end; c_complete 4 at the device: 2265.
    assert_eq!((t.chunks[0].start, t.chunks[0].end), (183, 2261));
    assert_eq!(t.complete, Some(2265));
    // sm0's wait: Await sent 10 in the dependency channel (independent of
    // the invoke channel), registered at 45; Resolve sent 2265, arrives
    // 2300. No other message kinds (no acknowledgement of invokes).
    assert_eq!(s.end_time, 2300);
    let kinds: Vec<&str> = s.messages_by_kind.keys().map(String::as_str).collect();
    assert_eq!(kinds, vec!["Await", "AwaitAck", "Invoke", "Resolve"]);
}

#[test]
fn a_consumer_of_a_write_registers_at_the_owner() {
    // sm0 writes X into HBM; sm1 later copies X from HBM into its SMEM on
    // its own engine. The copy's dep on X is remote (HBM is the device's):
    // it registers at the device and is released by the device's Resolve.
    let b = two_sm(&[]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let s0 = b.memory("root.gpc0.sm0.smem").base;
    let s1 = Range::from_len(b.memory("root.gpc0.sm1.smem").base, KB64);
    let h = Range::from_len(b.memory("root.hbm").base, KB64);
    let write = dma(&b, "root.dma", s0, h, KB64).with_inferred_version(1);
    let mut get = dma(&b, "root.gpc0.sm1.dma", h.base, s1, KB64);
    get.deps = vec![v1(h)];
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(write).await;
    });
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        ctx.cycles(100).await;
        ctx.invoke(get).await;
        ctx.wait(s1.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    let (tw, tg) = (&s.tasks[0], &s.tasks[1]);
    assert_eq!(tg.nu, sm1);
    // The write completes at 2265 as above; Resolve to sm1 takes 35 cycles.
    assert_eq!(tw.complete, Some(2265));
    assert_eq!(tg.ready, Some(2300));
    assert_eq!(s.warnings.len(), 0);
}

#[test]
fn a_task_on_a_pe_of_another_neutrino_is_fatal() {
    // A DMA into HBM named on sm0's engine: the device owns the dst, and
    // the PE is not the device's.
    let b = two_sm(&[]);
    let sm0 = b.nu("root.gpc0.sm0");
    let s0 = b.memory("root.gpc0.sm0.smem").base;
    let h = Range::from_len(b.memory("root.hbm").base, KB64);
    let write = dma(&b, "root.gpc0.sm0.dma", s0, h, KB64).with_inferred_version(1);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(write).await;
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    assert!(matches!(f.error, FatalError::Invalid(ref m) if m.contains("another Neutrino")));
}

#[test]
fn message_latency_adds_to_every_protocol_message() {
    // As a_write_into_hbm_is_a_pull_by_the_device with msg_latency 10:
    // Invoke and Resolve are on the critical path, 2 x 10 cycles:
    // 2300 -> 2320.
    let s = write_into_hbm(&["global.msg_latency=10"]);
    assert_eq!(s.fatal, None);
    assert_eq!(s.end_time, 2320);
}
