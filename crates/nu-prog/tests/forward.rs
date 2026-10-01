//! The forward pass driver's joint map (5b item 4) on configs/2p-cmem64.yaml.

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
