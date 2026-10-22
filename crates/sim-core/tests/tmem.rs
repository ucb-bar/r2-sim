//! The tensor memory of an SM (tmem model, 2026-10-21) on
//! configs/2p-tmem.yaml: four banks of 64 KiB, each with a read and a
//! write port of 128 B/cycle, latency 5. The Atlas reads A, B and C from
//! the tmem and writes C into it; the partial C of a chain is fp32.

use nu_model::{Addr, Range};
use sim_core::{
    build_config_file, Args, Built, Descriptor, Dtype, FatalError, NuCtx, ProfileRegistry, Stats,
};

const T: u64 = 128;
const BANK: u64 = 64 << 10;

fn tmem(sets: &[&str]) -> Result<Built, sim_core::ConfigError> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p-tmem.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins())
}

/// The base of sm0's tmem bank `i`.
fn bank(b: &Built, i: u64) -> u64 {
    b.memory("root.gpc0.sm0.tmem").base + i * BANK
}

/// A 128 x 128 x 128 bf16 matmul on sm0 with A at `a`, B at `b` and C
/// at `c`.
fn mm(b: &Built, (a, bb, c): (u64, u64, u64), accumulate: bool, keep: bool) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe("root.gpc0.sm0.atlas")),
        dst: Range::from_len(c, T * T * 4),
        versioned: false,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
        label: Some(1),
        args: Args::Matmul {
            a: Addr(a),
            b: Addr(bb),
            a2: None,
            b2: None,
            lda: T,
            ldb: T,
            ldc: T,
            m: T,
            n: T,
            k: T,
            in_dtype: Dtype::Bf16,
            b_dtype: Dtype::Bf16,
            out_dtype: Dtype::Bf16,
            accumulate,
            keep,
            tile: [T, T],
        },
    }
}

fn run(b: Built, d: Descriptor) -> Stats {
    let sm0 = b.nu("root.gpc0.sm0");
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
    });
    sim.run()
}

/// Bytes charged on resource `name`.
fn bytes(s: &Stats, name: &str) -> u64 {
    s.resources
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no resource {name}"))
        .bytes
}

#[test]
fn an_atlas_task_charges_the_tmem_banks_its_ranges_overlap() {
    // A in bank 0, B in bank 1, C across the boundary of banks 2 and 3.
    // The last task of a chain: C in bf16 (32 KiB), half in each bank;
    // no C read. The SMEM takes nothing.
    let b = tmem(&[]).unwrap();
    let c = bank(&b, 3) - T * T;
    let d = mm(&b, (bank(&b, 0), bank(&b, 1), c), false, false);
    let s = run(b, d);
    assert_eq!(s.fatal, None);
    let p = "root.gpc0.sm0.tmem";
    assert_eq!(bytes(&s, &format!("{p}.b0.port")), T * T * 2);
    assert_eq!(bytes(&s, &format!("{p}.b1.port")), T * T * 2);
    assert_eq!(bytes(&s, &format!("{p}.b2.wport")), T * T);
    assert_eq!(bytes(&s, &format!("{p}.b3.wport")), T * T);
    assert_eq!(bytes(&s, &format!("{p}.b2.port")), 0);
    assert_eq!(bytes(&s, "root.gpc0.sm0.smem.b0.port"), 0);
}

#[test]
fn the_partial_c_of_a_chain_is_fp32() {
    // A middle task of a chain: it reads and writes C as fp32 (64 KiB),
    // half in each of banks 2 and 3.
    let b = tmem(&[]).unwrap();
    let c = bank(&b, 3) - T * T * 2;
    let d = mm(&b, (bank(&b, 0), bank(&b, 1), c), true, true);
    let s = run(b, d);
    assert_eq!(s.fatal, None);
    let p = "root.gpc0.sm0.tmem";
    for i in [2, 3] {
        assert_eq!(
            bytes(&s, &format!("{p}.b{i}.port")),
            T * T * 2,
            "bank {i} read"
        );
        assert_eq!(
            bytes(&s, &format!("{p}.b{i}.wport")),
            T * T * 2,
            "bank {i} write"
        );
    }
}

#[test]
fn an_atlas_task_pays_the_tmem_latency_once() {
    // The first chunk starts 5 cycles (the tmem latency) after the issue.
    let b = tmem(&[]).unwrap();
    let c = bank(&b, 2);
    let d = mm(&b, (bank(&b, 0), bank(&b, 1), c), false, false);
    let s = run(b, d);
    let t = s.tasks.iter().find(|t| t.label == Some(1)).unwrap();
    assert_eq!(t.chunks[0].start - t.issue.unwrap(), 5);
}

#[test]
fn the_tmem_access_rules_stop_the_run() {
    // (a) An Atlas task with A in the SMEM of an SM with a tmem.
    let b = tmem(&[]).unwrap();
    let smem = b.memory("root.gpc0.sm0.smem").base;
    let d = mm(&b, (smem, bank(&b, 1), bank(&b, 2)), false, false);
    match run(b, d).fatal.expect("fatal").error {
        FatalError::Invalid(m) => assert!(m.contains("A") && m.contains("not in its tmem"), "{m}"),
        e => panic!("{e:?}"),
    }
    // (b) A DMA from the tmem to the gpc's HBM; to the SMEM of the SM is
    // fine.
    let dma = |b: &Built, to: u64| Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe("root.gpc0.sm0.dma")),
        dst: Range::from_len(to, 4096),
        versioned: false,
        untracked: true,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Dma {
            src: Addr(bank(b, 2)),
            dst: Addr(to),
            bytes: 4096,
            tile_bytes: 4096,
        },
    };
    for (to, ok) in [
        (b_smem(), true),
        (tmem(&[]).unwrap().memory("root.gpc0.hbm").base, false),
    ] {
        let b = tmem(&[]).unwrap();
        let d = dma(&b, to);
        let sm0 = b.nu("root.gpc0.sm0");
        let mut sim = b.sim;
        sim.spawn(sm0, move |ctx: NuCtx| async move {
            ctx.invoke_untracked(d).await;
            ctx.fence().await;
        });
        let s = sim.run();
        match (ok, s.fatal) {
            (true, None) => {}
            (false, Some(f)) => match f.error {
                FatalError::Invalid(m) => assert!(m.contains("not the SMEM"), "{m}"),
                e => panic!("{e:?}"),
            },
            (ok, f) => panic!("to {to:#x}: ok {ok}, fatal {f:?}"),
        }
    }
}

/// sm0's SMEM base.
fn b_smem() -> u64 {
    tmem(&[]).unwrap().memory("root.gpc0.sm0.smem").base
}

#[test]
fn an_accumulator_buffer_on_an_sm_with_a_tmem_is_an_error() {
    let e = tmem(&["root.gpc0.sm0.atlas.acc_buffer=65536"])
        .err()
        .expect("rejected")
        .0;
    assert!(
        e.contains("accumulator buffer") && e.contains("tmem"),
        "{e}"
    );
    let mut b = tmem(&[]).unwrap();
    let e = b.sim.set_atlas_acc_buffer(1024).unwrap_err();
    assert!(e.contains("accumulator buffer"), "{e}");
}
