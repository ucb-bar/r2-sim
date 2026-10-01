//! The AdaMoE program end to end on configs/2p.yaml with a synthetic trace.

use func::{Records, Trace};
use nu_prog::label;
use sim_core::{build_config_file, ProfileRegistry, Stats};

fn synthetic(steps: u32, layers: u32, tokens: usize, experts: u8, seed: u64) -> Records {
    let mut x = seed;
    let mut r = Records::new();
    for s in 0..steps {
        for l in 0..layers {
            let ids: Vec<u8> = (0..tokens)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((x >> 33) % experts as u64) as u8
                })
                .collect();
            r.insert(label::router(s, l), ids);
        }
    }
    r
}

fn run(overrides: &[&str], quantize: &str) -> (Stats, nu_prog::adamoe::Report) {
    run_flow(overrides, quantize, None)
}

/// Runs on configs/2p.yaml; `flow` overrides its flow control.
fn run_flow(
    overrides: &[&str],
    quantize: &str,
    flow: Option<sim_core::Flow>,
) -> (Stats, nu_prog::adamoe::Report) {
    let set = format!("root.*.*.atlas.quantize={quantize}");
    run_config("2p.yaml", &[set.as_str()], overrides, flow)
}

/// Runs on `configs/<file>` with the configuration overrides `sets`.
fn run_config(
    file: &str,
    sets: &[&str],
    overrides: &[&str],
    flow: Option<sim_core::Flow>,
) -> (Stats, nu_prog::adamoe::Report) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs")
        .join(file);
    let mut b = build_config_file(&path, sets, ProfileRegistry::with_builtins()).expect("config");
    let ov: Vec<String> = overrides.iter().map(|s| s.to_string()).collect();
    let l = nu_prog::launch(
        "adamoe",
        &b,
        512,
        include_str!("../../../configs/adamoe.toml"),
        &ov,
    )
    .expect("launch");
    b.sim.set_timeline(false);
    if let Some(f) = flow {
        b.sim.set_flow(f);
    }
    b.sim
        .set_functional(Box::new(Trace::new(synthetic(10, 18, 51, 4, 7))));
    let mut sim = b.sim;
    sim.spawn(l.nu, l.program);
    let s = sim.run();
    let r = l.report.borrow().clone();
    (s, r)
}

#[test]
fn two_steps_two_layers() {
    for q in ["tile", "none"] {
        check(q);
    }
}

fn check(quantize: &str) {
    let (s, r) = run(&["steps=2", "layers=2"], quantize);
    assert_eq!(s.fatal, None, "{:?}", s.fatal);
    assert!(s.unfinished.is_empty(), "{:?}", s.unfinished);
    assert!(s.open_pending.is_empty(), "{:?}", s.open_pending);
    assert!(
        s.warnings.is_empty(),
        "{:?}",
        &s.warnings[..s.warnings.len().min(5)]
    );
    assert_eq!(r.step_end.len(), 2);
    // HBM bound: 273 B/cycle on configs/2p.yaml.
    let bound = r.hbm_bytes.div_ceil(273);
    assert!(
        s.end_time >= bound,
        "end {} below the HBM bound {bound}",
        s.end_time
    );
    eprintln!(
        "quantize {quantize}: end / HBM bound = {:.3}",
        s.end_time as f64 / bound as f64
    );
    eprintln!(
        "end {} steps {:?} hbm {} tasks {} events {} warnings {} msgs {} rejects {} resends {}",
        s.end_time,
        r.step_end,
        r.hbm_bytes,
        r.tasks,
        s.events,
        s.warnings.len(),
        s.messages,
        s.rejects,
        s.resends
    );
}

#[test]
fn ticket_flow_answers_every_request() {
    let (s, r) = run_flow(
        &["steps=2", "layers=2"],
        "tile",
        Some(sim_core::Flow::Ticket),
    );
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    // Every Invoke and VersionReset (first send or replay) gets one answer.
    let k = |n: &str| s.messages_by_kind.get(n).copied().unwrap_or(0);
    assert_eq!(
        k("Invoke") + k("VersionReset"),
        k("InvokeAck") + k("RetryAck") + k("Reject")
    );
    assert_eq!(k("RetryAck"), s.tickets.iter().sum::<u64>());
    assert!(s.end_time >= r.hbm_bytes.div_ceil(273));
}

#[test]
fn kv_multicast_loads_less_from_hbm() {
    let (a, ra) = run(&["steps=1", "layers=2", "attn_kv=\"per_head\""], "tile");
    let (b, rb) = run(&["steps=1", "layers=2", "attn_kv=\"multicast\""], "tile");
    for s in [&a, &b] {
        assert_eq!(s.fatal, None);
        assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    }
    // Duplication = bytes beyond one copy per layer: per head, 7 extra
    // copies of the K/V projection weights and the prefix KV; multicast, 2
    // (cluster 1 in round 0, cluster 0 in round 1). The difference is the
    // whole HBM saving.
    assert!(rb.hbm_bytes < ra.hbm_bytes);
    assert_eq!(
        ra.hbm_bytes - rb.hbm_bytes,
        ra.mqa_dup_bytes - rb.mqa_dup_bytes
    );
}

#[test]
fn weight_dtype_scales_weight_bytes_only() {
    // A d x 128 weight block: bf16 256 KiB, fp8 128 KiB, mxfp4 64 KiB plus
    // 4 KiB of scales. The KV cache stays bf16, so the HBM differences
    // between types are the same block count times the block differences.
    let runs: Vec<_> = ["bf16", "fp8", "mxfp4"]
        .iter()
        .map(|w| {
            let o = format!("weight_dtype=\"{w}\"");
            run(&["steps=1", "layers=2", o.as_str()], "tile")
        })
        .collect();
    for (s, _) in &runs {
        assert_eq!(s.fatal, None);
        assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    }
    let h: Vec<u64> = runs.iter().map(|(_, r)| r.hbm_bytes).collect();
    let (d8, d4) = (h[0] - h[1], h[0] - h[2]);
    assert_eq!(d8 * (262_144 - 69_632), d4 * (262_144 - 131_072));
    assert!(runs[2].0.end_time < runs[0].0.end_time);
}

#[test]
fn cmem_moves_weight_traffic_off_hbm() {
    // 2 layers of mxfp4 weights (69 MiB) fit 1.5 GiB of cmem: every weight
    // block comes from cmem; only the prefix KV still streams from HBM.
    // 2 layers of bf16 weights (258 MiB) overfill 64 MiB per cluster: the
    // placement fills each cluster's cmem and stops.
    let mx = ["steps=1", "layers=2", "weight_dtype=\"mxfp4\""];
    let bf = ["steps=1", "layers=2"];

    let (s0, r0) = run(&mx, "tile");
    let (s1, r1) = run_config("2p-cmem-large.yaml", &[], &mx, None);
    let (s2, r2) = run(&bf, "tile");
    let (s3, r3) = run_config("2p-cmem64.yaml", &[], &bf, None);
    for s in [&s0, &s1, &s2, &s3] {
        assert_eq!(s.fatal, None);
        assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    }
    assert_eq!((r0.cmem_bytes, r2.cmem_bytes), (0, 0));
    assert_eq!(r1.hbm_bytes + r1.cmem_bytes, r0.hbm_bytes);
    assert_eq!(r3.hbm_bytes + r3.cmem_bytes, r2.hbm_bytes);
    let placed = |r: &nu_prog::adamoe::Report| r.cmem_placed.values().sum::<u64>();
    let weights: u64 = r1.weight_bytes.values().sum();
    assert!(
        placed(&r1) >= weights,
        "every block in at least one cluster"
    );
    assert_eq!(placed(&r3), 2 * (64 << 20));
    assert!(r3.cmem_bytes > 0 && r3.hbm_bytes > r1.hbm_bytes);
    assert!(s1.end_time < s0.end_time && s3.end_time < s2.end_time);
}
