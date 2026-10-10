//! Read and write channels (architecture Section 3.5): with `wr_bytes` a
//! bus or link carries reads and writes on separate channels.

use nu_model::{Addr, Range};
use sim_core::{build_config, Args, Built, Descriptor, NuCtx, ProfileRegistry, Stats};

/// A root with a 1 KiB/cycle HBM on `sbus` (`sbus` as given) and one SM
/// behind the link `smsbus_to_sbus` (keys as given). Data from the root to
/// the SM crosses the link's `ba` direction and the egress port of the
/// link's attachment on `sbus`.
fn node(sbus: &str, link: &str, sets: &[&str]) -> Result<Built, sim_core::ConfigError> {
    let mut text = format!(
        "global:
  neutrino_granule: 512
  address: {{fields: [{{name: sm, bits: 2}}], offset_bits: 24}}
device:
  name: root
  fabric:
    - {{name: sbus{sbus}}}
    - {{name: cbus, control: true}}
  memory:
    - {{name: hbm, type: dram, addr: [[0x0, 0x10000]], bw_bytes: 1024, slave: $this.sbus}}
    - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
  neutrino: {{c_base: 0, c_dep: 0, c_word: 0, c_store: 0}}
  children:
    - name: sm0
      fabric:
        - {{name: smsbus}}
        - {{name: smcbus, control: true}}
        - {{name: lbus}}
        - {{name: smsbus_to_sbus, topology: link, from: $this.smsbus, to: $parent.sbus{link}}}
        - {{name: smcbus_to_cbus, topology: link, from: $this.smcbus, to: $parent.cbus, bw_bytes: 128, control: true}}
      memory:
        - {{name: smem, type: scratch, addr: [[0x0, 0x10000]], bw_bytes: 1024, slave: [$this.lbus, $this.smsbus]}}
        - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
      compute:
        - {{name: dma, type: dma, beat_bytes: 1024, master: [$this.lbus, $this.smsbus]}}
      neutrino: {{c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}}
"
    );
    if !sets.is_empty() {
        text.push_str("set:\n");
        for s in sets {
            text.push_str(&format!("  - \"{s}\"\n"));
        }
    }
    build_config(&text, ProfileRegistry::with_builtins())
}

/// The end of a 4 KiB DMA read from HBM into SM memory (a read: data from
/// the root to the SM), with a 4 KiB store by the root's program into the
/// SM's memory at the same time (a write, the same direction) or not.
fn run(b: Built, store: bool) -> (u64, Stats) {
    run_some(b, true, store)
}

/// [`run`] with the DMA (`dma`) and the store (`store`) each optional.
fn run_some(b: Built, dma: bool, store: bool) -> (u64, Stats) {
    let (root, sm) = (b.nu("root"), b.nu("root.sm0"));
    let (hbm, smem) = (b.memory("root.hbm").base, b.memory("root.sm0.smem").base);
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe("root.sm0.dma")),
        dst: Range::from_len(smem, 4096),
        versioned: false,
        inferred_version: None,
        deps: vec![],
        label: Some(0),
        args: Args::Dma {
            src: Addr(hbm),
            dst: Addr(smem),
            bytes: 4096,
            tile_bytes: 4096,
        },
    };
    let mut sim = b.sim;
    sim.set_timeline(true);
    if dma {
        sim.spawn(sm, move |ctx: NuCtx| async move {
            let h = ctx.invoke(d).await;
            ctx.wait_unversioned(h.addr).await;
        });
    }
    if store {
        sim.spawn(root, move |ctx: NuCtx| async move {
            ctx.store(Addr(smem + 0x8000), vec![0u8; 4096]).await;
        });
    }
    let s = sim.run();
    assert_eq!(s.fatal, None);
    (s.tasks.first().map_or(0, |t| t.chunks[0].end), s)
}

fn busy(s: &Stats, name: &str) -> u64 {
    s.resources
        .iter()
        .find(|r| r.name == name)
        .unwrap()
        .busy_cycles
}

/// A read and a write in the same direction, both requested at cycle 0
/// at 64 B/cycle (4096 B, 64 cycles on the shared element): on one channel
/// the second waits for the first; with a write channel of 64 B/cycle each
/// takes as long as alone; with one of 32 B/cycle the write takes 4096 /
/// 32 = 128 cycles on it. `cfg(wr)` gives the bus and link keys (`wr` 0:
/// no write channel); `rd` and `wr` name the shared element's channels.
fn check(cfg: impl Fn(u64) -> (String, String), rd: &str, wr: &str) {
    let build = |w| {
        let (sbus, link) = cfg(w);
        node(&sbus, &link, &[]).unwrap()
    };
    let read = run_some(build(64), true, false).1.end_time;
    let write = run_some(build(64), false, true).1.end_time;
    let (_, s) = run(build(0), true);
    assert_eq!(busy(&s, rd), 128);
    assert!(
        s.end_time >= read.max(write) + 60,
        "{} {read} {write}",
        s.end_time
    );
    let (_, s) = run(build(64), true);
    assert_eq!(
        (s.end_time, busy(&s, rd), busy(&s, wr)),
        (read.max(write), 64, 64)
    );
    let (_, s) = run(build(32), true);
    assert_eq!((busy(&s, rd), busy(&s, wr)), (64, 128));
    // A read alone does not touch the write channel.
    let (end, s) = run(build(32), false);
    assert_eq!((end, busy(&s, wr)), (64, 0));
}

/// `, wr_bytes: w` for `w` > 0.
fn wr(w: u64) -> String {
    if w == 0 {
        String::new()
    } else {
        format!(", wr_bytes: {w}")
    }
}

#[test]
fn a_link_with_wr_bytes_carries_reads_and_writes_apart() {
    check(
        |w| (String::new(), format!(", bw_bytes: 64{}", wr(w))),
        "root.sm0.smsbus_to_sbus.ba",
        "root.sm0.smsbus_to_sbus.ba.wr",
    );
}

#[test]
fn an_xbar_port_with_wr_bytes_carries_reads_and_writes_apart() {
    check(
        |w| {
            (
                format!(", topology: xbar, bw_bytes: 64{}", wr(w)),
                String::new(),
            )
        },
        "root.sbus.sm0.smsbus_to_sbus.out",
        "root.sbus.sm0.smsbus_to_sbus.out.wr",
    );
}

#[test]
fn a_bus_with_wr_bytes_carries_reads_and_writes_apart() {
    check(
        |w| {
            (
                format!(", topology: bus, bw_bytes: 64{}", wr(w)),
                String::new(),
            )
        },
        "root.sbus",
        "root.sbus.wr",
    );
}

#[test]
fn wr_bytes_is_checked_and_can_be_set() {
    let e = node(
        "",
        ", bw_bytes: 64",
        &["root.sm0.smcbus_to_cbus.wr_bytes=64"],
    )
    .err()
    .unwrap()
    .0;
    assert!(e.contains("wr_bytes on a control bus or link"), "{e}");
    let e = node(", wr_bytes: 64", ", bw_bytes: 64", &[])
        .err()
        .unwrap()
        .0;
    assert!(e.contains("wr_bytes needs bw_bytes"), "{e}");
    // --set on a link: the write channel appears in both directions.
    let b = node(
        "",
        ", bw_bytes: 64",
        &["root.sm0.smsbus_to_sbus.wr_bytes=32"],
    )
    .unwrap();
    let res = b.sim.resources();
    for d in ["ab", "ba"] {
        let id = res
            .by_name(&format!("root.sm0.smsbus_to_sbus.{d}.wr"))
            .unwrap();
        assert_eq!(res.flow(id).rate(), sim_core::Ratio::int(32));
    }
    // Without wr_bytes there is no write channel.
    let b = node("", ", bw_bytes: 64", &[]).unwrap();
    assert!(b
        .sim
        .resources()
        .by_name("root.sm0.smsbus_to_sbus.ba.wr")
        .is_none());
}

#[test]
#[should_panic(expected = "set_bus_wr on xb after its attachment xb.m")]
fn a_write_channel_after_an_attachment_is_an_error() {
    // An attachment made before the bus gets its write channel would have
    // no write ports and its writes would fall back to the read channel.
    let mut sim = sim_core::Sim::new(ProfileRegistry::with_builtins());
    let bus = sim.add_bus(
        "xb",
        sim_core::BusTopology::Xbar,
        Some(sim_core::Ratio::int(64)),
        0,
        false,
    );
    sim.attach(bus, "m", true);
    sim.set_bus_wr(bus, sim_core::Ratio::int(32));
}
