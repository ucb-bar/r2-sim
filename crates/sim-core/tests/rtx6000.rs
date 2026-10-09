//! The RTX PRO 6000 configurations (architecture Section 8.3): they build,
//! the L2 hash link maps the granules verified on the card, and a DMA
//! from the linear GDDR7 range is cut at the hash granule.

use std::collections::VecDeque;

use nu_model::{Addr, Range};
use sim_core::{build_config_file, Args, Built, Descriptor, NuCtx, ProfileRegistry};

fn config(name: &str) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs")
        .join(name);
    build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins())
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn the_rtx6000_configurations_build() {
    for (name, gpcs, l2, channels) in [
        ("rtx6000.yaml", 12, 64, 16),
        ("rtx6000-half.yaml", 6, 32, 8),
    ] {
        let b = config(name);
        let sim = &b.sim;
        let topo = sim.topology();
        let count = |p: &str| {
            sim.memories()
                .iter()
                .filter(|m| m.name.starts_with(p))
                .count()
        };
        println!(
            "{name}: {} nodes, {} PEs, {} memory blocks, {} buses, {} links, {} attachments, {} resources, {} warnings",
            topo.len(),
            sim.pe_count(),
            sim.memories().len(),
            topo.bus_count(),
            topo.fabric_links().len(),
            topo.attachment_count(),
            sim.resources().len(),
            b.warnings.len()
        );
        for w in &b.warnings {
            println!("  warning: {w}");
        }
        assert_eq!(topo.len(), 1 + gpcs + 16 * gpcs);
        assert_eq!((count("root.l2.b"), count("root.hbm.b")), (l2, channels));
        let hashes: Vec<_> = sim.hashes().collect();
        assert_eq!(hashes.len(), 2);
        for h in &hashes {
            println!(
                "  hash {} -> {}: {} masks, granule {} B, slice {:#x} B",
                h.range,
                h.to_range,
                h.index.len(),
                h.granule(),
                h.slice_bytes()
            );
        }
        // Programs address the linear ranges: the banked memories' names
        // map there.
        assert_eq!(b.memory("root.l2").base, 0);
        assert_eq!(b.memory("root.hbm").base, 0x8_0000_0000);
    }
}

/// Granules of page 0 on the card (scratch/probe-rtx6000/labels.txt, every
/// 127th line from line 5): offset and slice index in the config's bit
/// order.
const GRANULES: &[(u64, u64)] = &[
    (0x500, 3),
    (0x8400, 34),
    (0x10300, 31),
    (0x18200, 62),
    (0x20100, 58),
    (0x28000, 27),
    (0x2ff00, 29),
    (0x37e00, 34),
    (0x3fd00, 3),
    (0x47c00, 13),
    (0x4fb00, 46),
    (0x57a00, 17),
    (0x5f900, 48),
    (0x67800, 52),
    (0x6f700, 19),
    (0x77600, 44),
    (0x7f500, 13),
    (0x87400, 44),
    (0x8f300, 15),
    (0x97200, 48),
    (0x9f100, 17),
    (0xa7000, 21),
    (0xaef00, 58),
    (0xb6e00, 5),
    (0xbed00, 36),
    (0xc6c00, 42),
    (0xceb00, 9),
    (0xd6a00, 54),
    (0xde900, 23),
    (0xe6800, 19),
    (0xee700, 52),
    (0xf6600, 11),
    (0xfe500, 42),
    (0x106400, 52),
    (0x10e300, 23),
    (0x116200, 40),
    (0x11e100, 9),
    (0x126000, 13),
    (0x12df00, 50),
    (0x135e00, 13),
    (0x13dd00, 44),
    (0x145c00, 34),
    (0x14db00, 1),
    (0x155a00, 62),
    (0x15d900, 31),
    (0x165800, 27),
    (0x16d700, 60),
    (0x175600, 3),
    (0x17d500, 34),
    (0x185400, 3),
    (0x18d300, 32),
    (0x195200, 31),
    (0x19d100, 62),
    (0x1a5000, 58),
    (0x1acf00, 21),
    (0x1b4e00, 42),
    (0x1bcd00, 11),
    (0x1c4c00, 5),
    (0x1ccb00, 38),
    (0x1d4a00, 25),
    (0x1dc900, 56),
    (0x1e4800, 60),
    (0x1ec700, 27),
    (0x1f4600, 36),
];

#[test]
fn the_l2_hash_maps_the_granules_of_the_card() {
    let b = config("rtx6000.yaml");
    let h = b.sim.hashes().find(|h| h.range.base == 0).unwrap().clone();
    assert_eq!((h.granule(), h.slice_bytes()), (256, 2 << 20));
    for &(a, slice) in GRANULES {
        assert_eq!(h.slice(a), slice, "granule {a:#x}");
        // Physically in the slice's bank.
        assert_eq!((h.rewrite(a) - h.to_range.base) / h.slice_bytes(), slice);
    }
    // Bit 9 does not take part: the two halves of a 1 KiB pair share a
    // slice; every granule of the page.
    for a in (0..2u64 << 20).step_by(256) {
        assert_eq!(h.slice(a), h.slice(a ^ 0x200), "granule {a:#x}");
    }
}

#[test]
fn a_dma_from_the_linear_gddr7_is_cut_at_the_hash_granule() {
    // rtx6000-half.yaml: sm0 of gpc0 copies 4 MiB from the linear GDDR7
    // range into its smem in 64 KiB transfers, two buffers in turn. Each
    // chunk is one 256 B granule (the channel stripe), so the transfers
    // touch all 8 channels and no L2 slice. The SM link (32 B/cycle) is
    // the bottleneck: 4 MiB / 32 = 131072 cycles plus a pipeline fill per
    // pair of transfers.
    let b = config("rtx6000-half.yaml");
    let sm = b.nu("root.gpc0.sm0");
    let (hbm, smem) = (
        b.memory("root.hbm").base,
        b.memory("root.gpc0.sm0.smem").base,
    );
    const T: u64 = 64 << 10;
    let descs: Vec<Descriptor> = (0..64)
        .map(|i| {
            let dst = smem + (i % 2) * T;
            Descriptor {
                opcode: b.sim.profiles().lookup("dma").unwrap(),
                pe: Some(b.pe("root.gpc0.sm0.dma")),
                dst: Range::from_len(dst, T),
                versioned: false,
                inferred_version: None,
                deps: vec![],
                label: Some(i),
                args: Args::Dma {
                    src: Addr(hbm + i * T),
                    dst: Addr(dst),
                    bytes: T,
                    tile_bytes: T,
                },
            }
        })
        .collect();
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn(sm, move |ctx: NuCtx| async move {
        let mut open = VecDeque::new();
        for d in descs {
            if open.len() == 2 {
                ctx.wait_unversioned(open.pop_front().unwrap()).await;
            }
            open.push_back(ctx.invoke(d).await.addr);
        }
        for a in open {
            ctx.wait_unversioned(a).await;
        }
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let chunks: usize = s.tasks.iter().map(|t| t.chunks.len()).sum();
    let busy = |p: &str| {
        s.resources
            .iter()
            .filter(|r| r.name.starts_with(p) && r.busy_cycles > 0)
            .count()
    };
    let (channels, slices) = (busy("root.hbm.b"), busy("root.l2.b"));
    println!(
        "4 MiB in {} transfers: {chunks} chunks, {channels} channels, {slices} L2 slices, {} cycles (link bound 131072)",
        s.tasks.len(),
        s.end_time
    );
    assert_eq!(chunks, (4 << 20) / 256);
    assert_eq!((channels, slices), (8, 0));
    // The two transfers in flight interleave chunk by chunk and end
    // together, so each of the 32 pairs starts on an empty pipeline (the
    // engine's 40 cycles and the links' 60, about 96 to the SM link):
    // 131072 + 32 x 96 = 134144.
    assert!(
        s.end_time >= 131_072 && s.end_time <= 131_072 + 32 * 100,
        "{}",
        s.end_time
    );
}
