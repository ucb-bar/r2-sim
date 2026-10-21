//! Address sets of memory blocks and hash links (architecture Section
//! 3.1): banked memories, the address rewrite of a link, the chunk clamp at
//! a block boundary and at the hash granule (Section 5.3), and the write
//! port (Section 8.2).

use nu_model::{Addr, AddressSet, Range};
use sim_core::{build_config, Args, Built, Descriptor, NuCtx, ProfileRegistry, Ratio, Stats};

/// A root with the memories `mems` (YAML list items), the buses `sbus`
/// and `hbmbus` and the fabric items `fabric`, and one SM with a 64 KiB
/// SMEM, a DMA engine and the compute items `compute`.
fn node(mems: &str, sets: &[&str]) -> Result<Built, sim_core::ConfigError> {
    node_with("", mems, "", sets)
}

fn node_with(
    fabric: &str,
    mems: &str,
    compute: &str,
    sets: &[&str],
) -> Result<Built, sim_core::ConfigError> {
    let mut text = format!(
        "global:
  neutrino_granule: 512
  address: {{fields: [{{name: sm, bits: 2}}], offset_bits: 24}}
device:
  name: root
  fabric:
    - {{name: sbus}}
    - {{name: cbus, control: true}}
    - {{name: hbmbus}}
{fabric}
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

/// A hash link from `sbus` to `hbmbus`: the linear range [1 MiB, 2 MiB)
/// (not at 0: no node owns the null address) in 512 B granules over four
/// banks of 256 KiB at 4 MiB.
const LINK: &str =
    "    - {name: sbus_to_hbmbus, topology: link, from: $this.sbus, to: $this.hbmbus,
       hash: {range: [0x100000, 0x200000], to_range: [0x400000, 0x500000], index: [0x200, 0x400]}}";

/// The four banks behind `LINK`.
const BANKS: &str = "    - {name: hbm, type: dram, addr: [[0x400000, 0x500000]], banks: 4,
       bw_bytes: 64, wr_bytes: 32, latency: 5, slave: $this.hbmbus, bank: {b2: {bw_bytes: 128}}}";

fn hashed(sets: &[&str]) -> Built {
    node_with(LINK, BANKS, "", sets).unwrap()
}

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
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
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
fn a_banked_memory_is_one_block_per_bank() {
    let b = hashed(&["root.hbm.b3.latency=7"]);
    let blocks: Vec<_> = b
        .sim
        .memories()
        .iter()
        .filter(|m| m.name.starts_with("root.hbm"))
        .collect();
    let names: Vec<&str> = blocks.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        ["root.hbm.b0", "root.hbm.b1", "root.hbm.b2", "root.hbm.b3"]
    );
    // Equal contiguous blocks.
    assert_eq!(blocks[1].range, Range::new(0x44_0000, 0x48_0000));
    assert_eq!(blocks[1].sets, [AddressSet::block(0x44_0000, 0x4_0000)]);
    assert!(blocks
        .iter()
        .all(|m| m.banked && m.memory_name() == "root.hbm"));
    // Per-bank keys: the bank table of the file and a --set.
    let res = b.sim.resources();
    let rate = |m: &sim_core::MemBlock| res.flow(m.port).rate();
    assert_eq!(rate(blocks[0]), Ratio::int(64));
    assert_eq!(rate(blocks[2]), Ratio::int(128));
    assert_eq!(res.flow(blocks[0].wport.unwrap()).rate(), Ratio::int(32));
    assert_eq!(
        b.sim_resource_name(blocks[0].wport.unwrap()),
        "root.hbm.b0.wport"
    );
    assert_eq!((blocks[0].load_latency, blocks[3].load_latency), (5, 7));
    // Behind a hash link the memory's name is its linear range.
    assert_eq!(b.memory("root.hbm"), Range::new(0x10_0000, 0x20_0000));
    let h = b.sim.hashes().next().unwrap();
    assert_eq!((h.granule(), h.slice_bytes()), (512, 0x4_0000));
    assert_eq!(h.rewrite(0x10_0a00), 0x44_0000 + 0x200);
    // Without the link the banks are plain blocks at their range.
    let b = node(&BANKS.replace("hbmbus", "sbus"), &[]).unwrap();
    assert_eq!(b.memory("root.hbm"), Range::new(0x40_0000, 0x50_0000));
    // Errors: a bank that does not exist, banks that do not divide.
    let bad = |m: &str| node(m, &[]).err().unwrap().0;
    let e = bad(
        "    - {name: hbm, type: dram, addr: [[0x100000, 0x200000]], banks: 4,
       bw_bytes: 64, slave: $this.sbus, bank: {b4: {bw_bytes: 1}}}",
    );
    assert!(e.contains("no bank \"b4\""), "{e}");
    let e = bad(
        "    - {name: hbm, type: dram, addr: [[0x100000, 0x200000]], banks: 3,
       bw_bytes: 64, slave: $this.sbus}",
    );
    assert!(e.contains("3 equal banks"), "{e}");
}

#[test]
fn the_loader_checks_a_hash_link() {
    let bad = |fabric: &str, mems: &str| node_with(fabric, mems, "", &[]).err().unwrap().0;
    // Not a bijection: two masks that see only bit 9 among the drop bits.
    let link = LINK.replace(
        "index: [0x200, 0x400]",
        "index: [0x200, 0x8200], drop: [9, 10]",
    );
    let e = bad(&link, BANKS);
    assert!(e.contains("not invertible"), "{e}");
    // A memory in the linear range.
    let mems = format!("{BANKS}\n    - {{name: x, type: dram, addr: [[0x180000, 0x190000]], bw_bytes: 64, slave: $this.sbus}}");
    let e = bad(LINK, &mems);
    assert!(e.contains("overlaps the hash's linear range"), "{e}");
    // Banks that do not cover the physical range.
    let e = bad(LINK, &BANKS.replace("0x500000", "0x480000"));
    assert!(e.contains("cover"), "{e}");
    // A hash on a bus.
    let e = bad(
        "    - {name: x, hash: {range: [0, 0x1000], to_range: [0x1000, 0x2000], index: [0x100]}}",
        BANKS,
    );
    assert!(e.contains("hash is a key of a link"), "{e}");
}

#[test]
fn a_range_that_is_not_a_power_of_two_is_decomposed() {
    let b = node(
        "    - {name: hbm, type: dram, addr: [[0x100000, 0x1c0000]], bw_bytes: 64, slave: $this.sbus}",
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
            AddressSet::block(0x10_0000, 0x8_0000),
            AddressSet::block(0x18_0000, 0x4_0000)
        ]
    );
    // The run continues across the two sets of one block.
    assert_eq!(m.run(0x17_f000), 0x4_1000);
    assert!(m.covers(&Range::new(0x17_f000, 0x18_1000)));
}

#[test]
fn a_dma_chunk_stops_at_a_block_boundary() {
    // Two adjacent blocks; a 4 KiB transfer from 1 KiB below the boundary
    // is two chunks, 1 KiB on hbm and 3 KiB on hbm2, each on its own port
    // (64 B/cycle: 16 and 48 cycles).
    let b = node(
        "    - {name: hbm, type: dram, addr: [[0x100000, 0x110000]], bw_bytes: 64, slave: $this.sbus}
    - {name: hbm2, type: dram, addr: [[0x110000, 0x120000]], bw_bytes: 64, slave: $this.sbus}",
        &[],
    )
    .unwrap();
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root.sm0", 0x110000 - 1024, smem, 4096);
    assert_eq!(s.tasks[0].chunks.len(), 2);
    assert_eq!(busy(&s, "root.hbm.port"), 16);
    assert_eq!(busy(&s, "root.hbm2.port"), 48);
}

#[test]
fn a_dma_through_a_hash_link_moves_one_granule_per_chunk() {
    // 4 KiB into the linear range: 8 chunks of 512 B, two per bank, each on
    // the bank's write port (32 B/cycle in 64 B beats: half a slot, 2 x 16
    // cycles, 16 slot-cycles); the read ports stay idle.
    let b = hashed(&[]);
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root", smem, 0x10_1000, 4096);
    assert_eq!(s.tasks[0].chunks.len(), 8);
    for i in 0..4 {
        assert_eq!(busy(&s, &format!("root.hbm.b{i}.wport")), 16);
        assert_eq!(busy(&s, &format!("root.hbm.b{i}.port")), 0);
    }
    // Reads use the read ports: 1 KiB per bank at 64 B/cycle, 16 cycles;
    // on b2 at 128 B/cycle in two 64 B slots, 8 cycles x 2 slots.
    let b = hashed(&[]);
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root.sm0", 0x10_1000, smem, 4096);
    assert_eq!(busy(&s, "root.hbm.b0.port"), 16);
    assert_eq!(busy(&s, "root.hbm.b2.port"), 16);
    assert_eq!(busy(&s, "root.hbm.b0.wport"), 0);
    // An unaligned start: 256 B to the first granule's end, then 512 B.
    let b = hashed(&[]);
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root.sm0", 0x10_1100, smem, 1024);
    assert_eq!(s.tasks[0].chunks.len(), 3);
}

#[test]
fn a_store_through_a_hash_link_is_one_piece_per_granule() {
    let b = hashed(&[]);
    let sm0 = b.nu("root.sm0");
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.store(Addr(0x10_0800), vec![0u8; 2048]).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    // 512 B per bank on its half-slot write port: 16 cycles, 8
    // slot-cycles.
    for i in 0..4 {
        assert_eq!(busy(&s, &format!("root.hbm.b{i}.wport")), 8);
    }
}

/// The Atlas of `atlas_node` (a 4096 MAC/cycle bf16 array) on an SMEM of
/// four 16 KiB banks of 64 B/cycle per port.
const ATLAS: &str = "        - {name: atlas, type: tensor, rate: {bf16: 4096}, quantize: none, master: [$this.lbus]}";

fn atlas_node(fabric: &str, mems: &str) -> Built {
    let sets = [
        "root.sm0.smem.banks=4",
        "root.sm0.smem.bw_bytes=64",
        "root.sm0.smem.wr_bytes=64",
    ];
    node_with(fabric, mems, ATLAS, &sets).unwrap()
}

/// A 64 x 64 x 64 bf16 matmul (one 64 x 64 tile) with A at `a`
/// (`a2`), B at `b` (`b2`) and C at `c`, offsets in the SMEM.
fn atlas_mm(b: Built, a: u64, a2: Option<u64>, bb: u64, b2: Option<u64>, c: u64) -> Stats {
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let at = |o: u64| Addr(smem + o);
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe("root.sm0.atlas")),
        dst: Range::from_len(smem + c, 8192),
        versioned: false,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Matmul {
            a: at(a),
            b: at(bb),
            a2: a2.map(at),
            b2: b2.map(at),
            lda: 64,
            ldb: if b2.is_some() { 32 } else { 64 },
            ldc: 64,
            m: 64,
            n: 64,
            k: 64,
            in_dtype: sim_core::Dtype::Bf16,
            b_dtype: sim_core::Dtype::Bf16,
            out_dtype: sim_core::Dtype::Bf16,
            accumulate: false,
            keep: false,
            tile: [64, 64],
        },
    };
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    s
}

#[test]
fn an_atlas_chunk_charges_the_banks_of_its_operands() {
    // One SMEM block on lbus as four banks: the Atlas builds and charges
    // each operand range on the bank that holds it. A (8 KiB) in bank 0,
    // B in bank 2, C written in bank 1: 128 cycles on each of the three
    // ports at 64 B/cycle; bank 3 idle.
    let s = atlas_mm(atlas_node("", ""), 0, None, 0x8000, None, 0x4000);
    assert_eq!(busy(&s, "root.sm0.smem.b0.port"), 128);
    assert_eq!(busy(&s, "root.sm0.smem.b2.port"), 128);
    assert_eq!(busy(&s, "root.sm0.smem.b1.wport"), 128);
    assert_eq!(busy(&s, "root.sm0.smem.b1.port"), 0);
    assert_eq!(busy(&s, "root.sm0.smem.b3.port"), 0);
    // Split operands: rows 0 to 31 of A in bank 0 and 32 to 63 in bank 1,
    // columns of B over banks 2 and 3: 4 KiB and 64 cycles on each read
    // port, as long as the array (64^3 / 4096).
    let s = atlas_mm(
        atlas_node("", ""),
        0,
        Some(0x4000),
        0x8000,
        Some(0xC000),
        0x6000,
    );
    for i in 0..4 {
        assert_eq!(busy(&s, &format!("root.sm0.smem.b{i}.port")), 64, "b{i}");
    }
    assert_eq!(busy(&s, "root.sm0.smem.b1.wport"), 128);
    assert_eq!(busy(&s, "root.sm0.atlas.array"), 64);
}

#[test]
fn a_compute_pe_does_not_use_a_hashed_range() {
    // An Atlas operand in the linear range of a hash link stops the run.
    let b = atlas_node(LINK, BANKS);
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe("root.sm0.atlas")),
        dst: Range::from_len(smem, 8192),
        versioned: false,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Matmul {
            a: Addr(0x10_1000),
            b: Addr(smem),
            a2: None,
            b2: None,
            lda: 64,
            ldb: 64,
            ldc: 64,
            m: 64,
            n: 64,
            k: 64,
            in_dtype: sim_core::Dtype::Bf16,
            b_dtype: sim_core::Dtype::Bf16,
            out_dtype: sim_core::Dtype::Bf16,
            accumulate: false,
            keep: false,
            tile: [64, 64],
        },
    };
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
    });
    let s = sim.run();
    let f = format!("{:?}", s.fatal.expect("fatal"));
    assert!(f.contains("linear range"), "{f}");
    // A Muon task whose source is in the linear range stops the run.
    let muon = "        - {name: muon, type: vector, rate: {lane_op: 64}, master: [$this.lbus, $this.smsbus]}";
    let b = node_with(LINK, BANKS, muon, &[]).unwrap();
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("muon.vector").unwrap(),
        pe: Some(b.pe("root.sm0.muon")),
        dst: Range::from_len(smem, 1024),
        versioned: false,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Vector {
            ops: 64,
            bytes_in: 1024,
            bytes_out: 1024,
            src: Some(Addr(0x10_1000)),
        },
    };
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
    });
    let s = sim.run();
    let f = format!("{:?}", s.fatal.expect("fatal"));
    assert!(f.contains("linear range"), "{f}");
}

#[test]
fn a_muon_task_spreads_over_the_banks_of_its_range() {
    // 8 KiB read from 4 KiB below the end of bank 0: 4 KiB on bank 0 and
    // 4 KiB on bank 1, 64 cycles each at 64 B/cycle.
    let muon = "        - {name: muon, type: vector, rate: {lane_op: 64}, master: [$this.lbus, $this.smsbus]}";
    let sets = [
        "root.sm0.smem.banks=4",
        "root.sm0.smem.bw_bytes=64",
        "root.sm0.smem.wr_bytes=64",
    ];
    let b = node_with("", "", muon, &sets).unwrap();
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("muon.vector").unwrap(),
        pe: Some(b.pe("root.sm0.muon")),
        dst: Range::from_len(smem + 0x8000, 1024),
        versioned: false,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Vector {
            ops: 64,
            bytes_in: 8192,
            bytes_out: 1024,
            src: Some(Addr(smem + 0x3000)),
        },
    };
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert_eq!(busy(&s, "root.sm0.smem.b0.port"), 64);
    assert_eq!(busy(&s, "root.sm0.smem.b1.port"), 64);
    assert_eq!(busy(&s, "root.sm0.smem.b2.wport"), 16);
}

#[test]
fn a_bus_with_block_bytes_cuts_chunks_at_its_grid() {
    // 4 KiB from HBM at 0x100 (offset) into SMEM in one tile; the root bus on the
    // read half has block_bytes 1024: the chunks end at multiples of 1 KiB
    // of the source address, 768 B first and 256 B last. The HBM port (64
    // B/cycle) is the slow hop: 12, 16, 16, 16 and 4 cycles.
    let hbm =
        "    - {name: hbm, type: dram, addr: [[0x100000, 0x110000]], bw_bytes: 64, slave: $this.sbus}";
    let b = node(hbm, &["root.sbus.block_bytes=1024"]).unwrap();
    let smem = b.memory("root.sm0.smem").base;
    let s = dma(b, "root.sm0", 0x10_0100, smem, 4096);
    let spans: Vec<u64> = s.tasks[0].chunks.iter().map(|c| c.end - c.start).collect();
    assert_eq!(spans, [12, 16, 16, 16, 4]);
    assert_eq!(busy(&s, "root.hbm.port"), 64);
    // Without block_bytes the tile is one chunk.
    let b = node(hbm, &[]).unwrap();
    let s = dma(b, "root.sm0", 0x10_0100, smem, 4096);
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
        "    - {name: hbm, type: dram, addr: [[0x100000, 0x110000]], bw_bytes: 64, slave: $this.sbus}";
    let engine = [
        "root.sm0.dma.latency=10",
        "root.sm0.dma.burst_bytes=2048",
        "root.sm0.dma.beat_bytes=256",
    ];
    let spans = |extra: &[&str]| {
        let sets: Vec<&str> = engine.iter().chain(extra).copied().collect();
        let b = node(hbm, &sets).unwrap();
        let smem = b.memory("root.sm0.smem").base;
        let s = dma(b, "root.sm0", 0x10_0000, smem, 4096);
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
