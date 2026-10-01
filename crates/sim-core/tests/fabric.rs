//! The YAML loader and the fabric (architecture Section 8) on
//! tests/configs/two_sm.yaml and configs/2p.yaml: charging by bus
//! topology, the per-type tensor rate, includes, names and child-entry
//! overrides, NUMA slices, route selection by target address, and route
//! checks.

use nu_model::{Addr, Range};
use sim_core::{
    build_config, build_config_file, Args, Built, Descriptor, Dtype, NuCtx, ProfileRegistry,
};

const MAT: u64 = 64 * 64 * 2;

fn two_sm(sets: &[&str]) -> Result<Built, sim_core::ConfigError> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_sm.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins())
}

fn names(b: &Built, a: &str, c: &str) -> Vec<String> {
    b.sim
        .topology()
        .path(b.nu(a), b.nu(c))
        .resources
        .iter()
        .map(|r| b.sim_resource_name(*r).to_string())
        .collect()
}

#[test]
fn an_xbar_charges_ports_and_a_bus_one_resource() {
    // xbar (two_sm.yaml): a transfer charges the ingress port of its
    // source attachment and the egress port of its destination; the SM
    // bus is unlimited (no ports).
    let b = two_sm(&[]).unwrap();
    let (sm0, sm1, cl) = ("root.gpc0.sm0", "root.gpc0.sm1", "root.gpc0");
    assert_eq!(
        names(&b, sm0, cl),
        vec![
            "root.gpc0.sm0.bridge.up",
            "root.gpc0.clsbus.sm0.in",
            "root.gpc0.clsbus.neutrino.out"
        ]
    );
    assert_eq!(
        names(&b, sm1, sm0),
        vec![
            "root.gpc0.sm1.bridge.up",
            "root.gpc0.clsbus.sm1.in",
            "root.gpc0.clsbus.sm0.out",
            "root.gpc0.sm0.bridge.down"
        ]
    );
    // bus: one shared resource for every transfer on it.
    let b = two_sm(&["root.gpc0.clsbus.topology=bus"]).unwrap();
    assert_eq!(
        names(&b, sm1, sm0),
        vec![
            "root.gpc0.sm1.bridge.up",
            "root.gpc0.clsbus",
            "root.gpc0.sm0.bridge.down"
        ]
    );
    // ring: not implemented.
    let e = two_sm(&["root.gpc0.clsbus.topology=ring"]).err().unwrap();
    assert!(e.0.contains("not implemented"), "{e}");
    // Includes and $parent: each SM of two_sm_sm.yaml has its own names.
    assert!(b.try_pe("root.gpc0.sm1.atlas").is_some());
}

/// Array busy cycles of one 64^3 GEMM with operands of type `d` on sm0.
fn gemm_array_cycles(sets: &[&str], d: Dtype) -> u64 {
    let b = two_sm(sets).unwrap();
    let sm0 = b.nu("root.gpc0.sm0");
    let c = Range::from_len(b.memory("root.gpc0.sm0.smem").base, MAT);
    let mm = Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe("root.gpc0.sm0.atlas")),
        dst: c,
        versioned: true,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Matmul {
            a: Addr(c.base),
            b: Addr(c.base),
            lda: 64,
            ldb: 64,
            ldc: 64,
            m: 64,
            n: 64,
            k: 64,
            in_dtype: d,
            b_dtype: d,
            out_dtype: Dtype::Bf16,
            accumulate: false,
            keep: false,
            tile: [64, 64],
        },
    };
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(mm).await;
        ctx.wait(h.addr, h.version).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    s.resources
        .iter()
        .find(|r| r.name == "root.gpc0.sm0.atlas.array")
        .unwrap()
        .busy_cycles
}

#[test]
fn a_gemm_charges_the_rate_of_its_data_type() {
    // 64^3 = 262144 MACs, own co-charging (the array's own time). The
    // array runs at the highest rate (2048); a bf16 GEMM at 1024 charges
    // twice its MACs: 256 cycles; fp8: 128.
    let (own, rates) = (
        "global.cocharge=own",
        "root.gpc0.sm0.atlas.rate={bf16: 1024, fp8: 2048}",
    );
    assert_eq!(gemm_array_cycles(&[own, rates], Dtype::Bf16), 256);
    assert_eq!(gemm_array_cycles(&[own, rates], Dtype::Fp8), 128);
    // One rate: 262144 / 4096.
    assert_eq!(gemm_array_cycles(&[own], Dtype::Bf16), 64);
}

#[test]
fn overrides_address_elements_by_name() {
    let b = two_sm(&["root.gpc0.sm*.smem.bw_bytes=64"]).unwrap();
    let port = b
        .sim
        .memories()
        .iter()
        .find(|m| m.name == "root.gpc0.sm1.smem")
        .unwrap()
        .port;
    assert_eq!(b.sim.resources().flow(port).rate().num, 64);
    // A missing key is added inside a section, not into elements that a
    // wildcard selected.
    assert!(two_sm(&["root.gpc0.*.neutrino.c_resolve=5"]).is_ok());
    let e = two_sm(&["root.gpc0.nosuch.bw_bytes=1"]).err().unwrap();
    assert!(e.0.contains("nothing matches"), "{e}");
}

#[test]
fn two_shortest_routes_are_an_error() {
    // Two caches that join the same two buses: two routes of one hop.
    let e = build_config(
        "global:
  neutrino_granule: 512
  address: {offset_bits: 24}
device:
  name: root
  fabric: [{name: A}, {name: B}, {name: C, control: true}]
  memory:
    - {name: c1, type: cache, addr: [[0x0, 0x1000]], bw_bytes: 8, slave: $this.A, master: $this.B}
    - {name: c2, type: cache, addr: [[0x0, 0x1000]], bw_bytes: 8, slave: $this.A, master: $this.B}
  neutrino: {}
",
        ProfileRegistry::with_builtins(),
    )
    .err()
    .unwrap();
    assert!(e.0.contains("two routes"), "{e}");
}

#[test]
fn the_loader_rejects_tables_that_cannot_accept_an_invoke() {
    // Section 3.3: an owner needs one inbound invoke entry, and an
    // execution table above the issue headroom (2 on sm1, its 2 PEs).
    let e = two_sm(&["root.gpc0.sm1.neutrino.inbound_invokes=0"])
        .err()
        .unwrap();
    assert!(e.0.contains("inbound invokes 0"), "{e}");
    let e = two_sm(&["root.gpc0.sm1.neutrino.execution=2"])
        .err()
        .unwrap();
    assert!(
        e.0.contains("execution table 2 must exceed the issue headroom 2"),
        "{e}"
    );
    assert!(two_sm(&[
        "root.gpc0.sm1.neutrino.execution=3",
        "root.gpc0.sm1.neutrino.inbound_invokes=1",
        "root.gpc0.sm1.neutrino.scoreboard=1",
    ])
    .is_ok());
}

#[test]
fn the_2p_family_passes_the_loader_check() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with("2p") && name.ends_with(".yaml") {
            build_config_file::<&str>(&p, &[], ProfileRegistry::with_builtins())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            n += 1;
        }
    }
    assert_eq!(n, 8, "the 2p*.yaml files");
}

/// The 2p family's configs/2p.yaml.
fn two_p() -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p.yaml");
    build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).expect("2p.yaml")
}

/// The resource names on the route a DMA engine `pe` takes from the block
/// `from` (`None`: the engine) to the block `to`.
fn dma_route(b: &Built, pe: &str, from: &str) -> Vec<String> {
    let sim_core::PeParams::Dma { masters, .. } = &b.sim.pe_config(b.pe(pe)).params else {
        panic!("{pe} is not a DMA engine");
    };
    let block = b.sim.memories().iter().find(|m| m.name == from).unwrap();
    b.sim
        .topology()
        .best_path(&block.attach, masters)
        .expect("a route")
        .resources
        .iter()
        .map(|r| b.sim_resource_name(*r).to_string())
        .collect()
}

#[test]
fn every_gpc_owns_its_hbm_slice() {
    // NUMA (Section 8.2): each gpc owns an HBM slice in its own space; the
    // device has none.
    let b = two_p();
    assert!(b.try_memory("root.hbm").is_none());
    for g in ["root.gpc0", "root.gpc1"] {
        let nu = b.nu(g);
        let h = b.memory(&format!("{g}.hbm"));
        assert!(b.own_space(nu).covers(&h), "{g}.hbm in {g}'s space");
        let block = b.sim.memories().iter().find(|m| m.range == h).unwrap();
        assert_eq!(block.owner, nu);
    }
    // A read of gpc0's slice by root.gpc0.sm1's DMA stays in gpc0; a read
    // of gpc1's slice leaves gpc1 through its bridge, crosses the device
    // crossbar and enters gpc0 (the SM's data bus smsbus is unlimited).
    assert_eq!(
        dma_route(&b, "root.gpc0.sm1.dma", "root.gpc0.hbm"),
        vec![
            "root.gpc0.clsbus.hbm.in",
            "root.gpc0.clsbus.sm1.out",
            "root.gpc0.sm1.bridge.down"
        ]
    );
    assert_eq!(
        dma_route(&b, "root.gpc0.sm1.dma", "root.gpc1.hbm"),
        vec![
            "root.gpc1.clsbus.hbm.in",
            "root.gpc1.clsbus.bridge.out",
            "root.gpc1.bridge.up",
            "root.sbus.gpc1.in",
            "root.sbus.gpc0.out",
            "root.gpc0.bridge.down",
            "root.gpc0.clsbus.bridge.in",
            "root.gpc0.clsbus.sm1.out",
            "root.gpc0.sm1.bridge.down"
        ]
    );
}

#[test]
fn one_gpc_file_is_instantiated_twice() {
    // configs/2p.yaml lists gpc0 and gpc1 from gpc.yaml, which lists sm0 to
    // sm3: eight SMs, named by their parent, in distinct spaces.
    let b = two_p();
    let mut spaces = Vec::new();
    for g in 0..2 {
        for s in 0..4 {
            let name = format!("root.gpc{g}.sm{s}");
            let nu = b.nu(&name);
            assert!(b.own_space(nu).covers(&b.memory(&format!("{name}.smem"))));
            spaces.push(b.own_space(nu));
            assert!(b
                .sim
                .resources()
                .by_name(&format!("root.gpc{g}.clsbus.sm{s}.in"))
                .is_some());
        }
    }
    spaces.sort_by_key(|r| r.base);
    spaces.dedup();
    assert_eq!(spaces.len(), 8);
}

#[test]
fn a_child_entry_overrides_its_files_bridge() {
    // The bridge is in the child's file (two_sm_sm.yaml: 32 B/cycle); the
    // parent's entry for sm1 overrides its width only, so the parent buses
    // stay those of the file.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs");
    let text = std::fs::read_to_string(dir.join("two_sm.yaml"))
        .unwrap()
        .replace(
            "- {name: sm1, config: two_sm_sm.yaml}",
            "- {name: sm1, config: two_sm_sm.yaml, bridge: {bw_bytes: 16}}",
        );
    let b = sim_core::build_config_str::<&str>(&text, &dir, &[], ProfileRegistry::with_builtins())
        .unwrap();
    let rate = |n: &str| {
        let r = b.sim.resources();
        let x = r.flow(r.by_name(n).unwrap()).rate();
        x.num / x.den
    };
    assert_eq!(rate("root.gpc0.sm0.bridge.up"), 32);
    assert_eq!(rate("root.gpc0.sm1.bridge.up"), 16);
    assert_eq!(rate("root.gpc0.sm1.bridge.down.ctl"), 16);
    assert_eq!(
        names(&b, "root.gpc0.sm1", "root.gpc0"),
        vec![
            "root.gpc0.sm1.bridge.up",
            "root.gpc0.clsbus.sm1.in",
            "root.gpc0.clsbus.neutrino.out"
        ]
    );
}

#[test]
fn a_muon_reads_smem_over_lbus_and_a_remote_range_over_smsbus() {
    // One SM under the device. The Muon masters lbus and smsbus; SMEM is a
    // slave on both; HBM is at the device. V1 reads and writes SMEM: the
    // SMEM port only (2048 B at 128 B/cycle = 16 cycles) after the 10-cycle
    // overhead. V2 reads 1024 B of HBM and writes 1024 B of SMEM: the read
    // takes the route over smsbus and the SM's bridge down (16 B/cycle: 64
    // cycles, latency 20; the HBM port, 64 B/cycle, takes 16), so it
    // starts 10 + 20 cycles after its issue and lasts 64.
    let b = build_config(
        "global:
  neutrino_granule: 512
  address: {fields: [{name: sm, bits: 2}], offset_bits: 24}
device:
  name: root
  fabric: [{name: sbus}, {name: cbus, control: true}]
  memory:
    - {name: hbm, type: dram, addr: [[0x0, 0x100000]], bw_bytes: 64, slave: $this.sbus}
    - {name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}
  neutrino: {}
  children:
    - name: sm0
      bridge: {data: $parent.sbus, control: $parent.cbus, bw_bytes: 16, latency: 20}
      fabric: [{name: smsbus}, {name: smcbus, control: true}, {name: lbus}]
      memory:
        - {name: smem, type: scratch, addr: [[0x0, 0x10000]], bw_bytes: 128, slave: [$this.lbus, $this.smsbus]}
        - {name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}
      compute:
        - {name: muon, type: vector, rate: {lane_op: 64}, overhead: 10, master: [$this.lbus, $this.smsbus]}
      neutrino: {}
",
        ProfileRegistry::with_builtins(),
    )
    .unwrap();
    let sm0 = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem");
    let hbm = b.memory("root.hbm");
    let vec_op = b.sim.profiles().lookup("muon.vector").unwrap();
    let v = |i: u64, src: Option<Addr>| Descriptor {
        opcode: vec_op,
        pe: Some(b.pe("root.sm0.muon")),
        dst: Range::from_len(smem.base + i * 4096, 4096),
        versioned: false,
        inferred_version: None,
        deps: vec![],
        label: Some(i),
        args: Args::Vector {
            ops: 64,
            bytes_in: 1024,
            bytes_out: 1024,
            src,
        },
    };
    let (v1, v2) = (v(0, None), v(1, Some(Addr(hbm.base))));
    let mut sim = b.sim;
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h1 = ctx.invoke(v1).await;
        let h2 = ctx.invoke(v2).await;
        ctx.wait_unversioned(h1.addr).await;
        ctx.wait_unversioned(h2.addr).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let span = |l: u64| {
        let t = s.tasks.iter().find(|t| t.label == Some(l)).unwrap();
        let c = t.chunks[0];
        (c.start - t.issue.unwrap(), c.end - c.start)
    };
    assert_eq!(span(0), (10, 16));
    assert_eq!(span(1), (30, 64));
}
