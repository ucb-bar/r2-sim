//! The Muon profile from the kernel registry: the traffic of a task comes
//! from its kernel and argument block (`time` of kernels.yaml). A root with
//! an HBM (64 B/cycle) and one SM with a 64 KiB SMEM (1024 B/cycle) and a
//! Muon of 64 lanes; the SM reaches the HBM over its data link.

use nu_model::Range;
use sim_core::{build_config, kernels, Built, Descriptor, NuCtx, ProfileRegistry, Stats};

fn node(hbm_latency: u64) -> Built {
    node_with(hbm_latency, "")
}

/// The node with `muon` (YAML keys with a trailing comma) added to the
/// Muon's entry.
fn node_with(hbm_latency: u64, muon: &str) -> Built {
    let text = format!(
        "global:
  neutrino_granule: 512
  address: {{fields: [{{name: sm, bits: 2}}], offset_bits: 24}}
device:
  name: root
  fabric:
    - {{name: sbus}}
    - {{name: cbus, control: true}}
  memory:
    - {{name: hbm, type: dram, addr: [[0x100000, 0x200000]], bw_bytes: 64, latency: {hbm_latency}, slave: $this.sbus}}
    - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
  neutrino: {{c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}}
  children:
    - name: sm0
      fabric:
        - {{name: smsbus}}
        - {{name: smcbus, control: true}}
        - {{name: lbus}}
        - {{name: smsbus_to_sbus, topology: link, from: $this.smsbus, to: $parent.sbus, bw_bytes: 1024, latency: 10}}
        - {{name: smcbus_to_cbus, topology: link, from: $this.smcbus, to: $parent.cbus, bw_bytes: 128, control: true}}
      memory:
        - {{name: smem, type: scratch, addr: [[0x0, 0x10000]], bw_bytes: 1024, latency: 5, slave: [$this.lbus, $this.smsbus]}}
        - {{name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}}
      compute:
        - {{name: muon, type: vector, rate: {{lane_op: 64}}, {muon} master: [$this.lbus, $this.smsbus]}}
      neutrino: {{c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}}
"
    );
    build_config(&text, ProfileRegistry::with_builtins()).expect("config")
}

/// The argument block of kernel `id` with `values` by field name.
fn block(id: u32, values: &[(&str, u64)]) -> Vec<u8> {
    let k = kernels::by_id(id).unwrap();
    let mut b = vec![0u8; k.bytes as usize];
    for &(name, v) in values {
        let o = k.field(name).unwrap().offset as usize;
        match k.field(name).unwrap().ty {
            kernels::FieldType::U64 => b[o..o + 8].copy_from_slice(&v.to_le_bytes()),
            _ => b[o..o + 4].copy_from_slice(&(v as u32).to_le_bytes()),
        }
    }
    b
}

/// The bases of sm0's SMEM and of the HBM.
fn bases() -> (u64, u64) {
    let b = node(0);
    (b.memory("root.sm0.smem").base, b.memory("root.hbm").base)
}

/// Runs one Muon task of kernel `id` with `values` on sm0 whose
/// destination is [SMEM + `OUT`, +1024).
fn run(hbm_latency: u64, id: u32, values: &[(&str, u64)]) -> Stats {
    let b = node(hbm_latency);
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let d = Descriptor::muon(
        b.pe("root.sm0.muon"),
        Range::from_len(smem + OUT, 1024),
        id,
        block(id, values),
    )
    .unversioned();
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

const OUT: u64 = 0x8000;

fn bytes(s: &Stats, name: &str) -> u64 {
    s.resources
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no resource {name}"))
        .bytes
}

/// An epilogue of 4 x 128 with NORM and BIAS: C (1024 B) and the
/// statistics (32 B) in the SMEM, the bias and u (512 B each) in the HBM,
/// out (1024 B) in the SMEM; 4 x 128 x 3 = 1536 lane-ops.
fn epilogue(hbm_latency: u64, flags: u64) -> Stats {
    let (smem, hbm) = bases();
    run(
        hbm_latency,
        2,
        &[
            ("c", smem + 0x1000),
            ("out", smem + OUT),
            ("bias", hbm),
            ("u", hbm + 0x1000),
            ("stats", smem + 0x2000),
            ("rows", 4),
            ("cols", 128),
            ("ld_c", 128),
            ("ld_out", 128),
            ("k", 1152),
            ("flags", flags),
        ],
    )
}

const NORM: u64 = 1;
const BIAS: u64 = 2;

#[test]
fn the_entries_of_one_block_add_up() {
    let s = epilogue(50, NORM | BIAS);
    // The HBM: bias and u, 1024 B (16 cycles at 64 B/cycle); the link
    // carries them. The SMEM: C and the statistics read (1056 B), out
    // written (1024 B).
    assert_eq!(bytes(&s, "root.hbm.port"), 1024);
    assert_eq!(bytes(&s, "root.sm0.smsbus_to_sbus.ba"), 1024);
    assert_eq!(bytes(&s, "root.sm0.smem.port"), 1056 + 1024);
    // The lane-ops: 1536 at 64 lanes, 24 cycles.
    let v = s
        .resources
        .iter()
        .find(|r| r.name == "root.sm0.muon.vector")
        .unwrap();
    assert_eq!(v.busy_cycles, 24);
}

#[test]
fn the_latency_is_the_longest_path_of_the_entries() {
    // With BIAS the bias is read from the HBM: its path (the request's
    // trip over the link 10, the HBM latency, the link 10) is the longest,
    // and the chunk ends 100 cycles later when the HBM latency is 100 more.
    // From the issue to the chunk's end: the latency delays the chunk's start.
    let span = |s: &Stats| s.tasks[0].chunks[0].end - s.tasks[0].issue.unwrap();
    let (a, b) = (epilogue(50, BIAS), epilogue(150, BIAS));
    assert_eq!(span(&b) - span(&a), 100);
    // Without a flag every entry is in the SMEM: the HBM latency does not
    // count.
    let (a, b) = (epilogue(50, 0), epilogue(150, 0));
    assert_eq!(span(&a), span(&b));
    assert!(span(&a) < span(&epilogue(50, BIAS)));
}

#[test]
fn an_opaque_task_with_src_0_reads_its_destination_block() {
    let s = run(
        50,
        0,
        &[("ops", 640), ("bytes_in", 2048), ("bytes_out", 1024)],
    );
    // 2048 B read and 1024 B written on the SMEM (the block of OUT), none
    // on the HBM.
    assert_eq!(bytes(&s, "root.sm0.smem.port"), 2048 + 1024);
    assert_eq!(bytes(&s, "root.hbm.port"), 0);
    // With src in the HBM, the read goes there.
    let s = run(
        50,
        0,
        &[
            ("src", bases().1),
            ("ops", 640),
            ("bytes_in", 2048),
            ("bytes_out", 1024),
        ],
    );
    assert_eq!(bytes(&s, "root.hbm.port"), 2048);
    assert_eq!(bytes(&s, "root.sm0.smem.port"), 1024);
}

#[test]
fn an_entry_in_no_memory_block_stops_the_run_and_names_the_pc() {
    let b = node(50);
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let mut d = Descriptor::muon(
        b.pe("root.sm0.muon"),
        Range::from_len(smem + OUT, 1024),
        2,
        block(
            2,
            &[
                ("c", smem),
                ("out", smem + OUT),
                ("bias", 0x30_0000),
                ("rows", 1),
                ("cols", 128),
                ("flags", BIAS),
            ],
        ),
    )
    .unversioned();
    d.pc = Some(0x40);
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(d).await;
    });
    let f = sim.run().fatal.expect("an error");
    assert_eq!(
        f.error,
        sim_core::FatalError::Invalid(
            "the Muon task at pc 0x40: its read of 512 B at 0x300000 is in no memory block".into()
        )
    );
}

/// Runs one opaque task per (ops, latency) on sm0's Muon, all issued at
/// once (independent, each to its own 1 KiB of the SMEM, no memory
/// traffic), on the node with `muon`; per task, its chunk's end minus the
/// first issue.
fn opaque_ends(muon: &str, tasks: &[(u64, u64)]) -> Vec<u64> {
    let b = node_with(0, muon);
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let ds: Vec<Descriptor> = tasks
        .iter()
        .enumerate()
        .map(|(i, &(ops, latency))| {
            Descriptor::muon(
                b.pe("root.sm0.muon"),
                Range::from_len(smem + OUT + i as u64 * 1024, 1024),
                0,
                block(0, &[("ops", ops), ("latency", latency)]),
            )
            .unversioned()
        })
        .collect();
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let mut hs = Vec::new();
        for d in ds {
            hs.push(ctx.invoke(d).await);
        }
        for h in hs {
            ctx.wait_unversioned(h.addr).await;
        }
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let start = s.tasks.iter().filter_map(|t| t.issue).min().unwrap();
    s.tasks.iter().map(|t| t.chunks[0].end - start).collect()
}

/// The fixed latency of the tests, and the work of a task: 64 lane-ops
/// per cycle.
const L: u64 = 300;
const W: u64 = 40;

#[test]
fn the_latency_of_an_opaque_task_adds_its_cycles() {
    let with = opaque_ends("", &[(W * 64, L)]);
    let without = opaque_ends("", &[(W * 64, 0)]);
    assert_eq!(with[0] - without[0], L);
    // Ops 0: the task ends after its latency, plus the Muon's overhead.
    assert_eq!(opaque_ends("", &[(0, L)]), [L]);
    assert_eq!(opaque_ends("overhead: 25,", &[(0, L)]), [25 + L]);
}

#[test]
fn the_latencies_of_tasks_in_flight_overlap_and_their_work_does_not() {
    let two = [(W * 64, L), (W * 64, L)];
    assert_eq!(opaque_ends("inflight: 2,", &two), [L + W, L + 2 * W]);
    assert_eq!(opaque_ends("inflight: 1,", &two), [L + W, 2 * L + 2 * W]);
}
