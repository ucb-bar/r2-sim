//! The micro-benchmark program on tests/configs/tiny.yaml (one gpc of
//! three SMs, a 64 B/cycle HBM at the device and its copy engine).

use serde_json::Value;
use sim_core::{build_config_file, ProfileRegistry};

/// Runs `micro` with the parameters `params` on tiny.yaml; its result.
fn micro(params: &[&str]) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/tiny.yaml");
    let mut b =
        build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).expect("config");
    let mut ov: Vec<String> = vec!["l2=\"root.hbm\"".into(), "dram=\"root.hbm\"".into()];
    ov.extend(params.iter().map(|s| s.to_string()));
    let l = nu_prog::launch("micro", &b, "", &ov).expect("launch");
    b.sim.set_timeline(true);
    let mut sim = b.sim;
    sim.spawn_named(l.nu, l.name, l.program);
    let s = sim.run();
    assert_eq!(s.fatal, None, "{:?}", s.fatal);
    assert!(s.unfinished.is_empty(), "{:?}", s.unfinished);
    nu_prog::micro::result(&l.micro.expect("micro report").borrow(), &s)
}

fn f(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or_else(|| panic!("{k} in {v}"))
}

#[test]
fn a_stream_reads_and_writes_the_hbm() {
    // Two SMs read 8 KiB each from the HBM in 4 KiB transfers: 16 KiB
    // moved, at most the HBM's 64 B/cycle.
    let r = micro(&[
        "mode=\"stream\"",
        "sms=2",
        "mem=\"dram\"",
        "bytes=8192",
        "chunk=4096",
        "inflight=2",
    ]);
    assert_eq!(r["bytes"], 16384);
    let bw = f(&r, "bytes_per_cycle");
    assert!(bw > 16.0 && bw <= 64.0, "{bw}");
    // Writes into the HBM are the device's copy engine's pulls.
    let r = micro(&[
        "mode=\"stream\"",
        "sms=2",
        "mem=\"dram\"",
        "dir=\"write\"",
        "bytes=8192",
        "chunk=4096",
    ]);
    assert_eq!(r["bytes"], 16384);
    assert!(f(&r, "bytes_per_cycle") <= 64.0);
    // The same through the device's single driver.
    let r = micro(&[
        "mode=\"stream\"",
        "sms=2",
        "mem=\"dram\"",
        "bytes=8192",
        "chunk=4096",
        "driver=\"root\"",
    ]);
    assert_eq!(r["bytes"], 16384);
}

#[test]
fn a_stream_between_smems() {
    for mem in ["smem", "rsmem_gpc"] {
        let m = format!("mem=\"{mem}\"");
        let r = micro(&["mode=\"stream\"", "sms=1", &m, "bytes=16384", "chunk=4096"]);
        assert_eq!(r["bytes"], 16384, "{mem}");
        assert!(f(&r, "cycles") > 0.0);
    }
}

#[test]
fn latency_atlas_and_muon() {
    let r = micro(&["mode=\"latency\"", "mem=\"dram\"", "bytes=4096"]);
    let (all, data) = (
        f(&r, "invoke_to_complete"),
        f(&r, "issue_to_last_chunk_end"),
    );
    // 4 KiB at 64 B/cycle is at least 64 cycles of data.
    assert!(data >= 64.0 && all >= data, "{r}");
    // A 64 x 64 x 64 bf16 matmul on a 4096 MAC/cycle array.
    let r = micro(&["mode=\"atlas\"", "m=64", "n=64", "k=64", "tile=[64, 64]"]);
    assert!(f(&r, "mac_per_cycle_per_sm") <= 4096.0);
    // 6400 lane-ops on a 64-lane Muon: at least 100 cycles.
    let r = micro(&["mode=\"muon\"", "lane_ops=6400", "bytes=1024"]);
    assert!(f(&r, "cycles_mean") >= 100.0, "{r}");
}

#[test]
fn control_modes() {
    let r = micro(&["mode=\"pingpong\"", "pair=\"same_gpc\"", "rounds=4"]);
    assert!(
        f(&r, "one_way_cycles") > 0.0 && f(&r, "step_cycles") > 0.0,
        "{r}"
    );
    let r = micro(&["mode=\"barrier\"", "sms=3"]);
    assert_eq!(r["n"], 3);
    assert!(r["to_join_ready"].as_u64().is_some(), "{r}");
    // Each invoke call costs at least c_invoke (8 cycles) on the driver.
    let r = micro(&["mode=\"invoke\"", "tasks=8"]);
    assert_eq!(r["tasks"], 8);
    assert!(f(&r, "call_cycles_per_task") >= 8.0, "{r}");
    let r = micro(&["mode=\"invoke\"", "tasks=9", "targets=\"many\"", "sms=3"]);
    assert_eq!(r["tasks"], 9);
}

#[test]
fn bad_parameters_are_errors() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/tiny.yaml");
    let b = build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).unwrap();
    let err = |p: &[&str]| {
        let ov: Vec<String> = p.iter().map(|s| s.to_string()).collect();
        nu_prog::launch("micro", &b, "", &ov)
            .err()
            .expect("an error")
    };
    assert!(err(&["mode=\"nope\""]).contains("unknown mode"));
    assert!(err(&["sms=9"]).contains("the machine has 3 SMs"));
    assert!(err(&["mem=\"l3\""]).contains("unknown mem"));
}

/// Launches `micro` with `params` on configs/2p-tmem.yaml.
fn tmem_launch(params: &[&str]) -> Result<Value, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p-tmem.yaml");
    let mut b =
        build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).expect("config");
    let ov: Vec<String> = params.iter().map(|s| s.to_string()).collect();
    let l = nu_prog::launch("micro", &b, "", &ov)?;
    b.sim.set_timeline(true);
    let mut sim = b.sim;
    sim.spawn_named(l.nu, l.name, l.program);
    let s = sim.run();
    assert_eq!(s.fatal, None, "{:?}", s.fatal);
    assert!(s.unfinished.is_empty(), "{:?}", s.unfinished);
    Ok(nu_prog::micro::result(
        &l.micro.expect("micro report").borrow(),
        &s,
    ))
}

#[test]
fn tmem_chains_on_one_sm() {
    // Two tiles of a 128 x 128 x 512 bf16 GEMM in chains of 4 matmuls of
    // kc 128 on 2p-tmem.yaml (tmem model, 2026-10-21): 16,777,216 MACs at
    // 8192 per cycle, 2048 cycles at best.
    let r = tmem_launch(&["mode=\"tmem\"", "k=512", "tiles=2"]).unwrap();
    assert_eq!(r["macs"], 2 * 128 * 128 * 512);
    let u = f(&r, "atlas_utilization");
    assert!(u > 0.1 && u <= 1.0, "{u}");
    assert!(f(&r, "end_cycles") >= 2048.0);
    assert_eq!(r["tile_tail_cycles"].as_array().unwrap().len(), 2);
    assert!(r["tmem_busiest"].as_str().unwrap().contains(".tmem.b"));
    // A layout that does not fit stops with what does not fit.
    let e = tmem_launch(&["mode=\"tmem\"", "kc=1024"]).unwrap_err();
    assert!(
        e.contains("the A stages") && e.contains("do not fit"),
        "{e}"
    );
}
