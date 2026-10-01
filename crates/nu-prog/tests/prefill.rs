//! Prefill builders (5b item 3) on tests/configs/tiny.yaml with a small VLM
//! shape: 64 tokens, d 256, 2 heads x 128 (one KV head), FFN 256, one
//! layer, activations in HBM.

use func::Trace;
use nu_prog::adamoe;
use nu_prog::kv::KvLayout;
use nu_prog::prefill::{choose_grid, CostEnv, Driver, GemmShape, Order, Params, Report, NEAR};
use nu_prog::Machine;
use sim_core::{build_config_file, Built, Dtype, ProfileRegistry, Stats};

fn tiny() -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/tiny.yaml");
    build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).unwrap()
}

fn action() -> adamoe::Params {
    adamoe::Params {
        layers: 1,
        tokens: 16,
        d: 256,
        ffn: 128,
        heads: 1,
        head_dim: 128,
        prefix: 64,
        experts: 1,
        ..adamoe::Params::default()
    }
}

fn params() -> Params {
    Params {
        model: "vlm".into(),
        layers: 1,
        d: 256,
        heads: 2,
        kv_heads: 1,
        head_dim: 128,
        ffn: 256,
        images: 1,
        tokens_per_image: 64,
        text_tokens: 0,
        quantize: false,
        action: action(),
        ..Params::default()
    }
}

fn run() -> (Stats, Report, KvLayout) {
    run_p(params())
}

fn run_p(p: Params) -> (Stats, Report, KvLayout) {
    let b = tiny();
    let m = Machine::from_built(&b, 512).unwrap();
    let (_, _, kv) = adamoe::hbm_map(&m, &action()).unwrap();
    let d = Driver::new(m, p).unwrap();
    let report = d.report.clone();
    let nu = d.home_nu();
    let mut sim = b.sim;
    sim.set_functional(Box::new(Trace::new(Default::default())));
    sim.spawn(nu, move |ctx| d.run(ctx));
    let s = sim.run();
    let r = report.borrow().clone();
    (s, r, kv)
}

fn clean(s: &Stats) {
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    assert!(
        s.warnings.is_empty(),
        "{:?}",
        &s.warnings[..s.warnings.len().min(3)]
    );
}

#[test]
fn vlm_kv_lands_where_the_action_expert_reads_it() {
    let (s, r, kv) = run();
    clean(&s);
    let mut w: Vec<(u64, u64)> = r
        .kv_writes
        .iter()
        .filter(|x| x.0 == 0)
        .map(|x| (x.1, x.2))
        .collect();
    w.sort();
    // The writes tile K and V of layer 0 exactly, without gaps or overlap.
    let (k, v) = (kv.k(0), kv.v(0));
    let mut at = k.base;
    for &(base, len) in &w {
        if base == v.base {
            assert_eq!(at, k.bound, "K fully written before V");
            at = v.base;
        }
        assert_eq!(base, at);
        at += len;
    }
    assert_eq!(at, v.bound);
    assert_eq!(w.iter().map(|x| x.1).sum::<u64>(), 2 * kv.bytes());
}

#[test]
fn writes_into_hbm_run_on_the_owners_copy_engine() {
    // Every write into HBM is a task of the device on its copy engine: no
    // other Neutrino's PE, and no issue or completion messages (VersionReset
    // messages free version table entries).
    let (s, r, _) = run();
    clean(&s);
    assert!(r.writes > 0);
    let kinds: Vec<&str> = s.messages_by_kind.keys().map(String::as_str).collect();
    assert!(
        kinds.iter().all(|k| [
            "Await",
            "AwaitAck",
            "Resolve",
            "Invoke",
            "InvokeAck",
            "RetryAck",
            "Reject",
            "PCrdGrant",
            "VersionReset"
        ]
        .contains(k)),
        "{kinds:?}"
    );
}

#[test]
fn the_chosen_grid_is_within_the_margin_and_reads_the_least_hbm() {
    // The VLM gate/up GEMM on a 7-worker, 1 MiB SMEM machine at the 2p
    // rates.
    let s = GemmShape {
        n: 816,
        k: 2048,
        c: 16384,
        w: Dtype::Bf16,
        gated: true,
        residual: false,
        cw_fixed: None,
        norm: true,
        single_acc: false,
    };
    let e = CostEnv {
        workers: 7,
        budget: 1 << 20,
        granule: 512,
        kcs: vec![128, 256, 512, 1024],
        sm: nu_prog::machine::SmRates {
            link_down: 128.0,
            link_up: 128.0,
            smem: 256.0,
            mac: 16384.0,
            row_panel: false,
            acc_buffer: 0,
            execution: 256,
            cluster_down: 0.0,
            cluster_up: 0.0,
        },
        hbm_rate: 273.0,
        act_rate: 273.0,
        act_in_hbm: true,
        quantize: true,
        pass_overhead: 150.0,
        cw: 0,
        worker_cluster: Vec::new(),
        cluster_links: Vec::new(),
        act_copies: 1,
        order: Order::Auto,
        owner_grant: None,
    };
    // The chosen grid's estimate is within NEAR of every feasible grid's,
    // and no grid with an estimate as low reads fewer HBM bytes.
    let best = choose_grid(&s, &e, None).unwrap();
    for tg in 1..=7 {
        for cg in 1..=7 / tg {
            if let Ok(c) = choose_grid(&s, &e, Some((tg, cg))) {
                assert!(best.cost <= c.cost * (1.0 + NEAR), "{best:?} vs {c:?}");
                if c.cost <= best.cost {
                    assert!(best.hbm_bytes <= c.hbm_bytes, "{best:?} vs {c:?}");
                }
            }
        }
    }
}

#[test]
fn weight_stationary_runs_clean_and_writes_the_same_kv() {
    // 64 tokens, one token tile; every GEMM that fits takes the
    // weight-stationary order.
    let (so, ro, kv) = run_p(Params {
        order: "os".into(),
        ..params()
    });
    let (sw, rw, _) = run_p(Params {
        order: "ws".into(),
        ..params()
    });
    clean(&so);
    clean(&sw);
    assert!(rw.phases.iter().filter_map(|p| p.grid).all(|g| g.ws));
    let kvw = |r: &Report| r.kv_writes.iter().map(|x| x.2).sum::<u64>();
    assert_eq!(kvw(&ro), 2 * kv.bytes());
    assert_eq!(kvw(&rw), 2 * kv.bytes());
}

#[test]
fn an_unknown_order_is_an_error() {
    let b = tiny();
    let m = Machine::from_built(&b, 512).unwrap();
    let p = Params {
        order: "weight_stationary".into(),
        ..params()
    };
    let e = Driver::new(m, p).err().expect("error");
    assert!(e.contains("unknown order"), "{e}");
}
