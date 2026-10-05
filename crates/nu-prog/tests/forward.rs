//! The forward pass driver: its joint map (5b item 4) on
//! configs/2p-cmem64.yaml, and per-gpc drivers on configs/2p.yaml.

use nu_prog::forward::{Driver, Params};
use nu_prog::Machine;
use sim_core::{build_config_file, ProfileRegistry};

fn reserve(copies: bool) -> (Vec<u64>, u64) {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p-cmem64.yaml");
    let b = build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).unwrap();
    let m = Machine::from_built(&b, 512).unwrap();
    let p = Params {
        weight_dtype: "mxfp4".into(),
        act_copies: copies,
        ..Params::default()
    };
    let d = Driver::new(m, p).unwrap();
    let r = d.report.borrow().cmem_reserve.clone();
    let placed = d.action_report.borrow().cmem_placed.values().sum();
    (r, placed)
}

#[test]
fn prefill_activations_and_action_weights_share_cmem_without_overlap() {
    // Prefill activations (VLM 40 MB, vision 18 MB) take the start of
    // gpc0's cmem; with copies, of every cluster's. The action expert's
    // weights fill the rest.
    let (r, placed) = reserve(false);
    assert!(r[0] > 50 << 20 && r[0] < 64 << 20);
    assert_eq!(r[1], 0);
    assert!(placed <= (64 << 20) - r[0] + (64 << 20));
    let (r, placed) = reserve(true);
    assert_eq!(r[1], r[0]);
    assert!(placed <= 2 * ((64 << 20) - r[0]));
}

/// The forward pass on configs/2p.yaml with one driver per gpc, small
/// stages (one vision and one VLM layer, one step of two action layers).
fn per_gpc(slice_writes: &str) -> (sim_core::Stats, nu_prog::forward::Report) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p.yaml");
    let mut b = build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).unwrap();
    let mut p = Params {
        weight_dtype: "mxfp4".into(),
        drivers: "hierarchical".into(),
        slice_writes: slice_writes.into(),
        ..Params::default()
    };
    p.vision.layers = 1;
    p.vlm.layers = 1;
    p.action.steps = 1;
    p.action.layers = 2;
    let records = nu_prog::synthetic_trace(&p.action, 7, 4.0, 0.9);
    let m = Machine::from_built(&b, 512).unwrap();
    let d = Driver::new(m, p).unwrap();
    let report = d.report.clone();
    let nu = d.home_nu();
    b.sim.set_timeline(false);
    b.sim.set_functional(Box::new(func::Trace::new(records)));
    let mut sim = b.sim;
    sim.spawn(nu, move |ctx| d.run(ctx));
    let s = sim.run();
    let r = report.borrow().clone();
    (s, r)
}

#[test]
fn per_gpc_drivers_split_the_tasks_and_agree_on_versions() {
    // Each gpc driver builds every task and invokes its gpc's share. The
    // owners check every inferred version in a debug build: a mismatch
    // would stop the run. Writes into the other gpc's slice: invoked by
    // the owner ("owner") or by the writer ("writer").
    for sw in ["owner", "writer"] {
        let (s, r) = per_gpc(sw);
        assert_eq!(s.fatal, None, "{sw}: {:?}", s.fatal);
        assert!(s.unfinished.is_empty() && s.open_pending.is_empty(), "{sw}");
        assert_eq!(s.messages_by_kind.get("Start"), Some(&6), "{sw}");
        for stage in ["vision", "vlm", "action"] {
            let ds: Vec<_> = r.drivers.iter().filter(|d| d.stage == stage).collect();
            assert_eq!(ds.len(), 2, "{sw} {stage}");
            assert_eq!(ds[0].planned, ds[1].planned, "{sw} {stage}: same plan");
            assert!(ds.iter().all(|d| d.invoked > 0), "{sw} {stage}");
            assert_eq!(
                ds[0].invoked + ds[1].invoked,
                ds[0].planned,
                "{sw} {stage}: every task invoked once"
            );
        }
    }
}
