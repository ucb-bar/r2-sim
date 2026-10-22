//! The pool for data in flight into a node through a link (r2sim
//! spec Section 5.6, Rule S5.11): Little's law, the source latency, the
//! direction, the wait for room, loads, and a link without a rate.

use nu_model::{Addr, Range};
use sim_core::{build_config, Args, Built, Descriptor, NuCtx, ProfileRegistry, Stats};

/// A root with an HBM of latency `lat` and one SM whose data link to the
/// root has `link` keys; both have a DMA engine of 64 transfers.
fn node(lat: u64, link: &str) -> Built {
    let text = format!(
        "global:
  neutrino_granule: 512
  address: {{fields: [{{name: sm, bits: 2}}], offset_bits: 24}}
device:
  name: root
  fidelity: numeric
  fabric:
    - {{name: sbus}}
    - {{name: cbus, control: true}}
  memory:
    - {{name: hbm, type: dram, addr: [[0x100000, 0x200000]], bw_bytes: 1024, latency: {lat}, slave: $this.sbus}}
    - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
  compute:
    - {{name: dma, type: dma, inflight: 64, beat_bytes: 1024, master: $this.sbus}}
  neutrino: {{c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}}
  children:
    - name: sm0
      fabric:
        - {{name: smsbus}}
        - {{name: smcbus, control: true}}
        - {{name: smsbus_to_sbus, topology: link, from: $this.smsbus, to: $parent.sbus, latency: 10{link}}}
        - {{name: smcbus_to_cbus, topology: link, from: $this.smcbus, to: $parent.cbus, bw_bytes: 128, control: true}}
      memory:
        - {{name: smem, type: scratch, addr: [[0x0, 0x40000]], bw_bytes: 1024, slave: [$this.smsbus]}}
        - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
      compute:
        - {{name: dma, type: dma, inflight: 64, beat_bytes: 1024, master: [$this.smsbus]}}
      neutrino: {{c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}}
"
    );
    build_config(&text, ProfileRegistry::with_builtins()).expect("config")
}

/// `n` DMAs of `bytes` each on `pe`'s engine, from `src + i x bytes` to
/// `dst + i x bytes`, invoked at once by `nu`; waits for all.
fn copies(b: Built, nu: &str, pe: &str, src: u64, dst: u64, bytes: u64, n: u64) -> Stats {
    let who = b.nu(nu);
    let op = b.sim.profiles().lookup("dma").unwrap();
    let pe = b.pe(pe);
    let ds: Vec<Descriptor> = (0..n)
        .map(|i| Descriptor {
            opcode: op,
            pe: Some(pe),
            dst: Range::from_len(dst + i * bytes, bytes),
            versioned: true,
            untracked: false,
            dst_mask: nu_model::FULL_MASK,
            inferred_version: None,
            deps: vec![],
            label: Some(i),
            args: Args::Dma {
                src: Addr(src + i * bytes),
                dst: Addr(dst + i * bytes),
                bytes,
                tile_bytes: bytes,
            },
        })
        .collect();
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(who, move |ctx: NuCtx| async move {
        let mut hs = Vec::new();
        for d in ds {
            let h = ctx.invoke(d).await;
            hs.push((h.addr, h.version));
        }
        for (a, v) in hs {
            ctx.wait(a, v).await;
        }
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty(), "{:?}", s.unfinished);
    s
}

/// 64 reads of 512 B from the HBM into sm0's SMEM by sm0's engine.
fn reads(lat: u64, link: &str) -> Stats {
    let b = node(lat, link);
    let (hbm, smem) = (b.memory("root.hbm").base, b.memory("root.sm0.smem").base);
    copies(b, "root.sm0", "root.sm0.dma", hbm, smem, 512, 64)
}

fn pool<'a>(s: &'a Stats, name: &str) -> &'a sim_core::ResourceStats {
    s.resources
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("{name}"))
}

const POOL: &str = "root.sm0.smsbus_to_sbus.inflight";

/// The end time of one read alone (issue to completion).
fn one_read(lat: u64, link: &str) -> u64 {
    let b = node(lat, link);
    let (hbm, smem) = (b.memory("root.hbm").base, b.memory("root.sm0.smem").base);
    copies(b, "root.sm0", "root.sm0.dma", hbm, smem, 512, 1).end_time
}

/// Little's law on a pooled run of 64 reads of 512 B: the bytes per cycle
/// against the pool's bytes over the mean hold H (held byte-cycles per
/// byte read).
fn littles_law(s: &Stats, bytes: u64) -> (f64, f64) {
    let p = pool(s, POOL);
    let h = p.held_cycles / (64.0 * 512.0);
    (64.0 * 512.0 / s.end_time as f64, bytes as f64 / h)
}

#[test]
fn a_pool_limits_the_reads_to_its_bytes_over_their_round_trip() {
    // Little's law: a pool of 4 reads (2 KiB) and holds of H cycles (201:
    // the HBM's 200 and one cycle on the link) carry 2048 / H bytes per
    // cycle.
    let s = reads(200, ", bw_bytes: 1024, inflight_bytes: 2048");
    let (rate, want) = littles_law(&s, 2048);
    assert!((rate / want - 1.0).abs() < 0.03, "{rate} vs {want}");
    // The mean hold: at least the HBM's 200 cycles and the cycle on the
    // link, plus a few cycles where reads of one batch share the HBM port.
    let h = pool(&s, POOL).held_cycles / (64.0 * 512.0);
    assert!((201.0..205.0).contains(&h), "{h}");
    // Without the pool all 64 overlap: one HBM latency and 64 cycles on
    // the HBM port, against 16 holds.
    let free = reads(200, ", bw_bytes: 1024");
    assert!(
        free.end_time * 4 < s.end_time,
        "{} vs {}",
        free.end_time,
        s.end_time
    );
    let p = pool(&s, POOL);
    assert_eq!((p.size, p.peak), (2048, 2048));
    assert!(p.delay_cycles > 0);
}

#[test]
fn a_shorter_source_latency_gives_more_bandwidth_with_the_same_pool() {
    let link = ", bw_bytes: 1024, inflight_bytes: 2048";
    let (slow, fast) = (reads(400, link).end_time, reads(100, link).end_time);
    let (h_slow, h_fast) = (one_read(400, link), one_read(100, link));
    assert!(fast < slow, "{fast} vs {slow}");
    // Both at the pool's rate: the ratio of the end times is the ratio of
    // the round trips.
    let (r, want) = (fast as f64 / slow as f64, h_fast as f64 / h_slow as f64);
    assert!((r - want).abs() < 0.05, "{r} vs {want}");
}

#[test]
fn data_leaving_the_node_does_not_use_the_pool() {
    // The root's engine copies sm0's SMEM out to the HBM: the data crosses
    // the link out of sm0, so a pool of one read changes nothing.
    let run = |link: &str| {
        let b = node(200, link);
        let (hbm, smem) = (b.memory("root.hbm").base, b.memory("root.sm0.smem").base);
        copies(b, "root", "root.dma", smem, hbm, 512, 64)
    };
    let (with, without) = (
        run(", bw_bytes: 1024, inflight_bytes: 512"),
        run(", bw_bytes: 1024"),
    );
    assert_eq!(with.end_time, without.end_time);
    let p = pool(&with, POOL);
    assert_eq!((p.peak, p.held_cycles, p.delay_cycles), (0, 0.0, 0));
}

#[test]
fn a_chunk_that_does_not_fit_waits_and_holds_from_when_room_frees() {
    // A pool of one read; two reads issued together at 0. The first
    // read's request crosses the link (latency 10) and reaches the HBM at
    // 10 (the request leg, 2026-10-21); its first hop (the HBM port)
    // starts after the HBM's 200 cycles, at 210, and 512 B take one cycle
    // on the HBM port and on the link (1024 B/cycle; the root bus is
    // unlimited, no latency between them). It holds the pool from its
    // request crossing the link to the end of its window on the link:
    // [10, 211). The second's request crosses the link when the room
    // frees, at 211: its first hop starts the source latency (200) and the
    // request route from the link to the HBM (0 cycles) later, at 411, and
    // its hold starts at 211.
    let s = reads_n(200, ", bw_bytes: 1024, inflight_bytes: 512", 2);
    let (a, b) = (&s.tasks[0], &s.tasks[1]);
    assert_eq!((a.issue, b.issue), (Some(0), Some(0)));
    assert_eq!(a.chunks[0].start, 210);
    assert_eq!(b.chunks[0].start, 211 + 200);
    let p = pool(&s, POOL);
    assert_eq!(p.peak, 512);
    // The wait: the second read's first hop at 411 instead of 210.
    assert_eq!(p.delay_cycles, 201);
    // Held byte-cycles: 512 x 201 for each read.
    assert_eq!(p.held_cycles, 2.0 * 512.0 * 201.0);
}

/// `n` reads of 512 B from the HBM into sm0 with a timeline.
fn reads_n(lat: u64, link: &str, n: u64) -> Stats {
    let b = node(lat, link);
    let (hbm, smem) = (b.memory("root.hbm").base, b.memory("root.sm0.smem").base);
    copies(b, "root.sm0", "root.sm0.dma", hbm, smem, 512, n)
}

#[test]
fn a_load_through_the_link_uses_the_pool() {
    // sm0's Neutrino loads 512 B from the HBM: the data enters sm0 through
    // the link, so the load holds the pool.
    let b = node(200, ", bw_bytes: 1024, inflight_bytes: 512");
    let sm0 = b.nu("root.sm0");
    let hbm = b.memory("root.hbm").base;
    let mut sim = b.sim;
    sim.set_functional(Box::new(func::Numeric::new()));
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.load(Addr(hbm), 512).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let p = pool(&s, POOL);
    assert_eq!(p.peak, 512);
    assert!(p.held_cycles > 0.0);
}

#[test]
fn a_link_without_a_rate_ends_the_hold_at_the_next_rated_hop() {
    // The same pool on an unlimited link: the route keeps no resource for
    // the link, and the hold ends with the window on sm0's SMEM port (the
    // first rated hop after the link), which starts the link's 10 cycles
    // after the HBM port: holds of 211 cycles. The reads still run at the
    // pool's rate.
    let s = reads(200, ", inflight_bytes: 2048");
    let (rate, want) = littles_law(&s, 2048);
    assert!((rate / want - 1.0).abs() < 0.03, "{rate} vs {want}");
    let h = pool(&s, POOL).held_cycles / (64.0 * 512.0);
    assert!((211.0..215.0).contains(&h), "{h}");
    assert_eq!(pool(&s, POOL).peak, 2048);
}

#[test]
fn inflight_bytes_is_a_key_of_a_data_link() {
    let text = |fabric: &str| {
        format!(
            "global:
  neutrino_granule: 512
  address: {{fields: [{{name: sm, bits: 2}}], offset_bits: 24}}
device:
  name: root
  fabric:
    - {{name: sbus{fabric}}}
    - {{name: cbus, control: true}}
  memory:
    - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
  neutrino: {{c_invoke: 0}}
"
        )
    };
    let e = build_config(
        &text(", inflight_bytes: 512"),
        ProfileRegistry::with_builtins(),
    )
    .err()
    .unwrap()
    .0;
    assert!(e.contains("inflight_bytes is a key of a data link"), "{e}");
    let e = build_config(
        &node_text_err(", inflight_bytes: 512, control: true"),
        ProfileRegistry::with_builtins(),
    )
    .err()
    .unwrap()
    .0;
    assert!(e.contains("inflight_bytes"), "{e}");
}

/// A node whose control link gets `extra` keys.
fn node_text_err(extra: &str) -> String {
    format!(
        "global:
  neutrino_granule: 512
  address: {{fields: [{{name: sm, bits: 2}}], offset_bits: 24}}
device:
  name: root
  fabric:
    - {{name: sbus}}
    - {{name: cbus, control: true}}
  memory:
    - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
  neutrino: {{c_invoke: 0}}
  children:
    - name: sm0
      fabric:
        - {{name: smsbus}}
        - {{name: smcbus, control: true}}
        - {{name: smsbus_to_sbus, topology: link, from: $this.smsbus, to: $parent.sbus}}
        - {{name: smcbus_to_cbus, topology: link, from: $this.smcbus, to: $parent.cbus{extra}}}
      memory:
        - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
      neutrino: {{c_invoke: 0}}
"
    )
}
