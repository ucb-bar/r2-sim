//! `--program hello` on configs/2p.yaml: two loads into an SM's SMEM, a
//! matmul there, and the copy of C back to the gpc's DRAM.

use sim_core::{build_config_file, ProfileRegistry};

#[test]
fn hello_runs_four_tasks_on_2p() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p.yaml");
    let b = build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).unwrap();
    let l = nu_prog::launch("hello", &b, 512, "", &[]).unwrap();
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn_named(l.nu, l.name, l.program);
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    // Two DMAs on the SM's engine, the matmul, the copy back.
    assert_eq!(s.invoked, 4);
    assert_eq!(s.end_time, 1949);
}
