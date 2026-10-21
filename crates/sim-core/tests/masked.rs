//! Masked destination ranges in the simulator (2026-10-21): the ownership
//! check covers the whole span, the functional plane checks the written
//! bytes against the held granules, and the race check sees only held
//! granules. Config: `configs/two_sm.yaml` (granule 512 B; sm0's SMEM
//! [0, 512 KiB) of its space).

use nu_model::{Addr, Range};
use sim_core::{
    build_config_file, Args, Built, Descriptor, FatalError, NuCtx, ProfileRegistry, Stats,
};

const G: u64 = 512;

fn two_sm(sets: &[&str]) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_sm.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins()).expect("config")
}

/// A DMA on sm0's engine of `bytes` from `src` to `to`, writing `dst` with
/// mask `mask`.
fn dma(b: &Built, src: u64, to: u64, bytes: u64, dst: Range, mask: u32) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe("root.gpc0.sm0.dma")),
        dst,
        versioned: false,
        untracked: false,
        dst_mask: mask,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Dma {
            src: Addr(src),
            dst: Addr(to),
            bytes,
            tile_bytes: bytes,
        },
    }
}

fn run_sm0(b: Built, d: Descriptor) -> Stats {
    let sm0 = b.nu("root.gpc0.sm0");
    let mut sim = b.sim;
    sim.set_functional(Box::new(func::Numeric::new()));
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
    });
    sim.run()
}

#[test]
fn the_ownership_check_covers_the_whole_span() {
    // The task holds only the last granule of sm0's SMEM (mask 0), but its
    // span of 4 granules runs past the SMEM, where nothing is owned.
    let b = two_sm(&["root.fidelity=numeric"]);
    let smem = b.memory("root.gpc0.sm0.smem");
    let dst = Range::from_len(smem.bound - G, 4 * G);
    let d = dma(&b, smem.base, smem.bound - G, G, dst, 0);
    let s = run_sm0(b, d);
    assert_eq!(s.fatal.expect("fatal").error, FatalError::Unowned(dst));
}

#[test]
fn the_functional_plane_checks_the_held_granules() {
    // dst holds granules 0 and 2 of a span of 4 (mask 0b10) at 64 KiB in
    // sm0's SMEM. Writing granule 2 is inside; writing granule 1, in the
    // span but in a gap, is outside.
    let b = two_sm(&["root.fidelity=numeric"]);
    let base = b.memory("root.gpc0.sm0.smem").base;
    let dst = Range::from_len(base + 0x10000, 4 * G);
    let ok = dma(&b, base, dst.base + 2 * G, G, dst, 0b10);
    let s = run_sm0(b, ok);
    assert_eq!(s.fatal, None);
    let b = two_sm(&["root.fidelity=numeric"]);
    let bad = dma(&b, base, dst.base + G + 8, 16, dst, 0b10);
    let s = run_sm0(b, bad);
    match s.fatal.expect("fatal").error {
        FatalError::Func { msg, .. } => {
            assert!(
                msg.contains("outside its dst") && msg.contains("mask 0x2"),
                "{msg}"
            )
        }
        e => panic!("not a functional error: {e:?}"),
    }
}

#[test]
fn a_load_in_a_gap_of_an_executing_writer_is_no_race() {
    // A long DMA writes granules 0 and 2 (mask 0b10). A load of granule 1
    // while it executes is no race; a load of granule 2 is.
    for (off, race) in [(G, false), (2 * G, true)] {
        let b = two_sm(&["root.fidelity=numeric"]);
        let sm0 = b.nu("root.gpc0.sm0");
        let base = b.memory("root.gpc0.sm0.smem").base;
        let dst = Range::from_len(base + 0x10000, 4 * G);
        let d = dma(&b, base + 0x20000, dst.base, G, dst, 0b10);
        let mut sim = b.sim;
        sim.set_functional(Box::new(func::Numeric::new()));
        sim.spawn(sm0, move |ctx: NuCtx| async move {
            ctx.invoke(d).await;
            ctx.load(Addr(dst.base + off), 8).await;
        });
        let s = sim.run();
        let raced = matches!(s.fatal.map(|f| f.error), Some(FatalError::Race { .. }));
        assert_eq!(raced, race, "load at granule {}", off / G);
    }
}
