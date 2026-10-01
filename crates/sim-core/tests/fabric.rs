//! The YAML loader and the fabric (architecture Section 8) on
//! configs/two_sm.yaml: charging by bus topology, the per-type tensor
//! rate, includes and names, overrides, and route checks.

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
    let (sm0, sm1, cl) = ("root.cluster0.sm0", "root.cluster0.sm1", "root.cluster0");
    assert_eq!(
        names(&b, sm0, cl),
        vec![
            "root.cluster0.sm0.bridge.up",
            "root.cluster0.CSBUS.sm0.in",
            "root.cluster0.CSBUS.neutrino.out"
        ]
    );
    assert_eq!(
        names(&b, sm1, sm0),
        vec![
            "root.cluster0.sm1.bridge.up",
            "root.cluster0.CSBUS.sm1.in",
            "root.cluster0.CSBUS.sm0.out",
            "root.cluster0.sm0.bridge.down"
        ]
    );
    // bus: one shared resource for every transfer on it.
    let b = two_sm(&["root.cluster0.CSBUS.topology=bus"]).unwrap();
    assert_eq!(
        names(&b, sm1, sm0),
        vec![
            "root.cluster0.sm1.bridge.up",
            "root.cluster0.CSBUS",
            "root.cluster0.sm0.bridge.down"
        ]
    );
    // ring: not implemented.
    let e = two_sm(&["root.cluster0.CSBUS.topology=ring"])
        .err()
        .unwrap();
    assert!(e.0.contains("not implemented"), "{e}");
    // Includes and $parent: each SM of two_sm_sm.yaml has its own names.
    assert!(b.try_pe("root.cluster0.sm1.atlas").is_some());
}

/// Array busy cycles of one 64^3 GEMM with operands of type `d` on sm0.
fn gemm_array_cycles(sets: &[&str], d: Dtype) -> u64 {
    let b = two_sm(sets).unwrap();
    let sm0 = b.nu("root.cluster0.sm0");
    let c = Range::from_len(b.memory("root.cluster0.sm0.smem").base, MAT);
    let mm = Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe("root.cluster0.sm0.atlas")),
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
        .find(|r| r.name == "root.cluster0.sm0.atlas.array")
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
        "root.cluster0.sm0.atlas.rate={bf16: 1024, fp8: 2048}",
    );
    assert_eq!(gemm_array_cycles(&[own, rates], Dtype::Bf16), 256);
    assert_eq!(gemm_array_cycles(&[own, rates], Dtype::Fp8), 128);
    // One rate: 262144 / 4096.
    assert_eq!(gemm_array_cycles(&[own], Dtype::Bf16), 64);
}

#[test]
fn overrides_address_elements_by_name() {
    let b = two_sm(&["root.cluster0.sm*.smem.bw_bytes=64"]).unwrap();
    let port = b
        .sim
        .memories()
        .iter()
        .find(|m| m.name == "root.cluster0.sm1.smem")
        .unwrap()
        .port;
    assert_eq!(b.sim.resources().flow(port).rate().num, 64);
    // A missing key is added inside a section, not into elements that a
    // wildcard selected.
    assert!(two_sm(&["root.cluster0.*.neutrino.c_resolve=5"]).is_ok());
    let e = two_sm(&["root.cluster0.nosuch.bw_bytes=1"]).err().unwrap();
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
  fabric: [{name: A}, {name: B}]
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
    let e = two_sm(&["root.cluster0.sm1.neutrino.inbound_invokes=0"])
        .err()
        .unwrap();
    assert!(e.0.contains("inbound invokes 0"), "{e}");
    let e = two_sm(&["root.cluster0.sm1.neutrino.execution=2"])
        .err()
        .unwrap();
    assert!(
        e.0.contains("execution table 2 must exceed the issue headroom 2"),
        "{e}"
    );
    assert!(two_sm(&[
        "root.cluster0.sm1.neutrino.execution=3",
        "root.cluster0.sm1.neutrino.inbound_invokes=1",
        "root.cluster0.sm1.neutrino.scoreboard=1",
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
