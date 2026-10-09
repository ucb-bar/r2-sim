//! Address sets of memory blocks (architecture Section 3.1): sliced
//! memories, the chunk clamp at a block boundary (Section 5.3) and the
//! write port (Section 8.2).

use nu_model::{Addr, AddressSet, Range};
use sim_core::{build_config, Args, Built, Descriptor, NuCtx, ProfileRegistry, Ratio, Stats};

/// A root with the memories `mems` (YAML list items) on `sbus` and one SM
/// with a 64 KiB SMEM, a DMA engine and the compute items `compute`.
fn node(mems: &str, sets: &[&str]) -> Result<Built, sim_core::ConfigError> {
    node_with(mems, "", sets)
}

fn node_with(mems: &str, compute: &str, sets: &[&str]) -> Result<Built, sim_core::ConfigError> {
    let mut text = format!(
        "global:
  neutrino_granule: 512
  address: {{fields: [{{name: sm, bits: 2}}], offset_bits: 24}}
device:
  name: root
  fabric: [{{name: sbus}}, {{name: cbus, control: true}}]
  memory:
{mems}
    - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
  compute:
    - {{name: dma, type: dma, inflight: 2, beat_bytes: 1024, master: $this.sbus}}
  neutrino: {{c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}}
  children:
    - name: sm0
      fabric:
        - {{name: smsbus}}
        - {{name: smcbus, control: true}}
        - {{name: lbus}}
        - {{name: smsbus_to_sbus, topology: link, from: $this.smsbus, to: $parent.sbus, bw_bytes: 1024}}
        - {{name: smcbus_to_cbus, topology: link, from: $this.smcbus, to: $parent.cbus, bw_bytes: 128, control: true}}
      memory:
        - {{name: smem, type: scratch, addr: [[0x0, 0x10000]], bw_bytes: 1024, slave: [$this.lbus, $this.smsbus]}}
        - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
      compute:
        - {{name: dma, type: dma, inflight: 2, beat_bytes: 1024, master: [$this.lbus, $this.smsbus]}}
{compute}
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

const SLICED: &str =
    "    - {name: hbm, type: dram, addr: [[0x0, 0x100000]], slices: 4, interleave: 512,
       bw_bytes: 64, wr_bytes: 32, latency: 5, slave: $this.sbus, slice: {s2: {bw_bytes: 128}}}";

/// Busy lane-cycles of the resource `name`.
fn busy(s: &Stats, name: &str) -> u64 {
    s.resources
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no resource {name}"))
        .busy_cycles
}

/// Runs one DMA of `bytes` from `src` to `dst` in one tile, invoked by
/// node `nu` on its DMA engine (the owner of `dst`).
fn dma(b: Built, nu: &str, src: u64, dst: u64, bytes: u64) -> Stats {
    let sm0 = b.nu(nu);
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe(&format!("{nu}.dma"))),
        dst: Range::from_len(dst, bytes),
        versioned: false,
        inferred_version: None,
        deps: vec![],
        label: Some(0),
        args: Args::Dma {
            src: Addr(src),
            dst: Addr(dst),
            bytes,
            tile_bytes: bytes,
        },
    };
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    s
}

#[test]
fn a_sliced_memory_is_one_block_per_slice() {
    let b = node(SLICED, &["root.hbm.s3.latency=7"]).unwrap();
    let blocks: Vec<_> = b
        .sim
        .memories()
        .iter()
        .filter(|m| m.name.starts_with("root.hbm"))
        .collect();
    let names: Vec<&str> = blocks.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        ["root.hbm.s0", "root.hbm.s1", "root.hbm.s2", "root.hbm.s3"]
    );
    // Slice i holds the 512 B granules whose bits 9..11 equal i.
    assert_eq!(blocks[1].sets, [AddressSet::new(0x200, 0xf_ffff & !0x600)]);
    assert!(blocks[1].holds(Addr(0xa00)) && !blocks[1].holds(Addr(0xc00)));
    assert_eq!(blocks[1].run(0x2f0), 0x110);
    assert!(blocks
        .iter()
        .all(|m| m.sliced && m.range == Range::new(0, 0x10_0000)));
    // Per-slice keys: the slice table of the file and a --set.
    let res = b.sim.resources();
    let rate = |m: &sim_core::MemBlock| res.flow(m.port).rate();
    assert_eq!(rate(blocks[0]), Ratio::int(64));
    assert_eq!(rate(blocks[2]), Ratio::int(128));
    assert_eq!(res.flow(blocks[0].wport.unwrap()).rate(), Ratio::int(32));
    assert_eq!(
        b.sim_resource_name(blocks[0].wport.unwrap()),
        "root.hbm.s0.wport"
    );
    assert_eq!((blocks[0].load_latency, blocks[3].load_latency), (5, 7));
    // The memory keeps its name for its whole range.
    assert_eq!(b.memory("root.hbm"), Range::new(0, 0x10_0000));
    // Errors: a slice that does not exist, a size that is not a power of
    // two, a granule that is not one.
    let bad = |m: &str| node(m, &[]).err().unwrap().0;
    let e = bad(
        "    - {name: hbm, type: dram, addr: [[0x0, 0x100000]], slices: 4, interleave: 512,
       bw_bytes: 64, slave: $this.sbus, slice: {s4: {bw_bytes: 1}}}",
    );
    assert!(e.contains("no slice \"s4\""), "{e}");
    let e = bad(
        "    - {name: hbm, type: dram, addr: [[0x0, 0xC0000]], slices: 4, interleave: 512,
       bw_bytes: 64, slave: $this.sbus}",
    );
    assert!(e.contains("power-of-two size"), "{e}");
    let e = bad(
        "    - {name: hbm, type: dram, addr: [[0x0, 0x100000]], slices: 4, interleave: 500,
       bw_bytes: 64, slave: $this.sbus}",
    );
    assert!(e.contains("powers of two"), "{e}");
}

#[test]
fn a_range_that_is_not_a_power_of_two_is_decomposed() {
    let b = node(
        "    - {name: hbm, type: dram, addr: [[0x0, 0xC0000]], bw_bytes: 64, slave: $this.sbus}",
        &[],
    )
    .unwrap();
    let m = b
        .sim
        .memories()
        .iter()
        .find(|m| m.name == "root.hbm")
        .unwrap();
    assert_eq!(
        m.sets,
        [
            AddressSet::block(0, 0x8_0000),
            AddressSet::block(0x8_0000, 0x4_0000)
        ]
    );
    // The run continues across the two sets of one block.
    assert_eq!(m.run(0x7_f000), 0x4_1000);
    assert!(m.covers(&Range::new(0x7_f000, 0x8_1000)));
}

#[test]
fn a_dma_chunk_stops_at_a_block_boundary() {
    // Two adjacent blocks; a 4 KiB transfer from 1 KiB below the boundary
    // is two chunks, 1 KiB on hbm and 3 KiB on hbm2, each on its own port
    // (64 B/cycle: 16 and 48 cycles).
    let b = node(
        "    - {name: hbm, type: dram, addr: [[0x0, 0x10000]], bw_bytes: 64, slave: $this.sbus}
    - {name: hbm2, type: dram, addr: [[0x10000, 0x20000]], bw_bytes: 64, slave: $this.sbus}",
        &[],
    )
    .unwrap();
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root.sm0", 0x10000 - 1024, smem, 4096);
    assert_eq!(s.tasks[0].chunks.len(), 2);
    assert_eq!(busy(&s, "root.hbm.port"), 16);
    assert_eq!(busy(&s, "root.hbm2.port"), 48);
}

#[test]
fn a_dma_into_a_sliced_memory_writes_one_granule_per_chunk() {
    // 4 KiB into 4 slices of 512 B: 8 chunks, two per slice, each on the
    // slice's write port (32 B/cycle: 2 x 16 cycles); the read ports stay
    // idle.
    let b = node(SLICED, &[]).unwrap();
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root", smem, 0x1000, 4096);
    assert_eq!(s.tasks[0].chunks.len(), 8);
    for i in 0..4 {
        assert_eq!(busy(&s, &format!("root.hbm.s{i}.wport")), 32);
        assert_eq!(busy(&s, &format!("root.hbm.s{i}.port")), 0);
    }
    // Reads use the read ports: 1 KiB per slice at 64 B/cycle, 16 cycles;
    // on s2 at 128 B/cycle in two 64 B lanes, 8 cycles x 2 lanes.
    let b = node(SLICED, &[]).unwrap();
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root.sm0", 0x1000, smem, 4096);
    assert_eq!(busy(&s, "root.hbm.s0.port"), 16);
    assert_eq!(busy(&s, "root.hbm.s2.port"), 16);
    assert_eq!(busy(&s, "root.hbm.s0.wport"), 0);
}

#[test]
fn a_store_across_slices_is_one_piece_per_slice() {
    let b = node(SLICED, &[]).unwrap();
    let sm0 = b.nu("root.sm0");
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.store(Addr(0x800), vec![0u8; 2048]).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    for i in 0..4 {
        assert_eq!(busy(&s, &format!("root.hbm.s{i}.wport")), 16);
    }
}

#[test]
fn a_compute_pe_does_not_use_a_sliced_memory() {
    // An Atlas reads the one SMEM on its bus: a sliced SMEM is a build
    // error.
    let atlas = "        - {name: atlas, type: tensor, rate: {bf16: 4096}, quantize: none, master: [$this.lbus]}";
    let slice_smem = ["root.sm0.smem.slices=4", "root.sm0.smem.interleave=512"];
    let e = node_with(SLICED, atlas, &slice_smem).err().unwrap().0;
    assert!(e.contains("root.sm0.smem is sliced"), "{e}");
    // A Muon task whose source is in a slice stops the run.
    let muon = "        - {name: muon, type: vector, rate: {lane_op: 64}, master: [$this.lbus, $this.smsbus]}";
    let b = node_with(SLICED, muon, &[]).unwrap();
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("muon.vector").unwrap(),
        pe: Some(b.pe("root.sm0.muon")),
        dst: Range::from_len(smem, 1024),
        versioned: false,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Vector {
            ops: 64,
            bytes_in: 1024,
            bytes_out: 1024,
            src: Some(Addr(0x1000)),
        },
    };
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
    });
    let s = sim.run();
    let f = format!("{:?}", s.fatal.expect("fatal"));
    assert!(f.contains("root.hbm.s0, a slice of a sliced memory"), "{f}");
}

#[test]
fn a_bus_with_block_bytes_cuts_chunks_at_its_grid() {
    // 4 KiB from HBM at 0x100 into SMEM in one tile; the root bus on the
    // read half has block_bytes 1024: the chunks end at multiples of 1 KiB
    // of the source address, 768 B first and 256 B last. The HBM port (64
    // B/cycle) is the slow hop: 12, 16, 16, 16 and 4 cycles.
    let hbm =
        "    - {name: hbm, type: dram, addr: [[0x0, 0x10000]], bw_bytes: 64, slave: $this.sbus}";
    let b = node(hbm, &["root.sbus.block_bytes=1024"]).unwrap();
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root.sm0", 0x100, smem, 4096);
    let spans: Vec<u64> = s.tasks[0].chunks.iter().map(|c| c.end - c.start).collect();
    assert_eq!(spans, [12, 16, 16, 16, 4]);
    assert_eq!(busy(&s, "root.hbm.port"), 64);
    // Without block_bytes the tile is one chunk.
    let b = node(hbm, &[]).unwrap();
    let s = dma(b, "root.sm0", 0x100, smem, 4096);
    assert_eq!(s.tasks[0].chunks.len(), 1);
}

#[test]
fn chunks_of_a_transfer_are_issued_at_the_beat_rate() {
    // 4 KiB from HBM (64 B/cycle) in two 2 KiB bursts, engine beat 256
    // B/cycle, latency 10 (Section 5.3, issue-driven chunks). Chunk 0 is
    // issued at the task's issue and takes the HBM port over [10, 42);
    // chunk 1 is issued 2048 / 256 = 8 cycles later, so it waits for the
    // port only: [42, 74). The latency is paid once.
    let hbm =
        "    - {name: hbm, type: dram, addr: [[0x0, 0x10000]], bw_bytes: 64, slave: $this.sbus}";
    let engine = [
        "root.sm0.dma.latency=10",
        "root.sm0.dma.burst_bytes=2048",
        "root.sm0.dma.beat_bytes=256",
    ];
    let spans = |extra: &[&str]| {
        let sets: Vec<&str> = engine.iter().chain(extra).copied().collect();
        let b = node(hbm, &sets).unwrap();
        let smem = b.memory("root.sm0.smem").base;
        let s = dma(b, "root.sm0", 0, smem, 4096);
        let t = &s.tasks[0];
        let at = t.issue.unwrap();
        let c: Vec<(u64, u64)> = t
            .chunks
            .iter()
            .map(|c| (c.start - at, c.end - at))
            .collect();
        (c, t.complete.unwrap() - at)
    };
    let (c, done) = spans(&[]);
    assert_eq!(c, [(10, 42), (42, 74)]);
    assert!(done >= 74);
    // With one chunk outstanding, chunk 1 waits for chunk 0's end and pays
    // the latency again: [52, 84).
    let (c, _) = spans(&["root.sm0.dma.outstanding=1"]);
    assert_eq!(c, [(10, 42), (52, 84)]);
}
