//! The memory store of the functional plane: one mapping per block, a
//! lookup by global address, zero for bytes that no op wrote, and no
//! effect for a write into the sync-only space.

use nu_model::{Dst, NeutrinoId, Range, FULL_MASK};
use sim_core::{MemBlock, ResourceId, Store};

/// A block `[base, base + size)` named `name`.
fn block(name: &str, base: u64, size: u64) -> MemBlock {
    MemBlock::plain(
        name.into(),
        NeutrinoId(0),
        Range::from_len(base, size),
        ResourceId(0),
        0,
    )
}

/// Three 1 MiB blocks (listed out of order, one of them twice), and the
/// sync-only space at the start of the last.
fn store() -> Store {
    let mib = 1 << 20;
    Store::new(
        &[
            block("hi", 4 * mib, mib),
            block("lo", mib, mib),
            block("mid", 2 * mib, mib),
            block("hi", 4 * mib, mib),
        ],
        &[Range::from_len(4 * mib, 0x1000)],
        512,
    )
    .unwrap()
}

#[test]
fn the_blocks_are_sorted_by_base_and_found_by_address() {
    let mut s = store();
    let names: Vec<&str> = s.blocks().iter().map(|b| b.name.as_str()).collect();
    assert_eq!(
        names,
        ["lo", "mid", "hi"],
        "sorted, and a block listed twice is one"
    );
    s.write(0x10_0010, &[1, 2, 3]).unwrap();
    s.write(0x40_2000, &[4, 5]).unwrap();
    assert_eq!(s.read(0x10_0010, 3).unwrap(), [1, 2, 3]);
    assert_eq!(s.read(0x40_2000, 2).unwrap(), [4, 5]);
    // The last byte of a block.
    s.write(0x1f_ffff, &[9]).unwrap();
    assert_eq!(s.read(0x1f_ffff, 1).unwrap(), [9]);
}

#[test]
fn a_byte_that_no_op_wrote_reads_as_zero() {
    let s = store();
    assert_eq!(s.read(0x18_0000, 64).unwrap(), vec![0; 64]);
}

#[test]
fn a_range_across_the_end_of_a_block_or_outside_all_blocks_is_an_error() {
    let mut s = store();
    let e = s.read(0x1f_fff0, 32).unwrap_err();
    assert!(e.0.contains("not inside one memory block"), "{}", e.0);
    assert!(s.write(0x1f_fff0, &[0; 32]).is_err());
    assert!(s.read(0x8_0000, 4).is_err(), "below the first block");
    assert!(s.read(0x60_0000, 4).is_err(), "past the last block");
}

#[test]
fn a_write_into_the_sync_only_space_has_no_effect() {
    let mut s = store();
    s.write(0x40_0000, &[1, 2, 3]).unwrap();
    assert_eq!(s.read(0x40_0000, 3).unwrap(), [0, 0, 0]);
}

#[test]
fn a_dma_between_two_blocks_copies() {
    let mut s = store();
    let dst = |r: Range| Dst {
        span: r,
        mask: FULL_MASK,
    };
    s.write(0x10_0000, &[7; 16]).unwrap();
    s.dma(
        &dst(Range::from_len(0x20_0000, 512)),
        0x10_0000,
        0x20_0000,
        16,
        None,
    )
    .unwrap();
    assert_eq!(s.read(0x20_0000, 16).unwrap(), vec![7; 16]);
    let into = Range::from_len(0x10_1000, 512);
    s.dma(&dst(into), 0x20_0000, 0x10_1000, 16, None).unwrap();
    assert_eq!(s.read(0x10_1000, 16).unwrap(), vec![7; 16]);
}

#[test]
fn a_dma_outside_its_dst_is_an_error() {
    let mut s = store();
    let dst = Dst {
        span: Range::from_len(0x10_1000, 512),
        mask: FULL_MASK,
    };
    let e = s.dma(&dst, 0x10_0000, 0x10_1200, 16, None).unwrap_err();
    assert!(e.0.contains("outside its dst"), "{}", e.0);
}

#[test]
fn a_dma_whose_source_and_destination_overlap_is_a_memmove() {
    let dst = |r: Range| Dst {
        span: r,
        mask: FULL_MASK,
    };
    let data: Vec<u8> = (0..64).collect();
    // Forward: the destination starts inside the source.
    let mut s = store();
    s.write(0x10_0000, &data).unwrap();
    s.dma(
        &dst(Range::from_len(0x10_0000, 512)),
        0x10_0000,
        0x10_0010,
        64,
        None,
    )
    .unwrap();
    assert_eq!(s.read(0x10_0010, 64).unwrap(), data);
    assert_eq!(s.read(0x10_0000, 16).unwrap(), data[..16]);
    // Backward: the source starts inside the destination.
    let mut s = store();
    s.write(0x10_0010, &data).unwrap();
    s.dma(
        &dst(Range::from_len(0x10_0000, 512)),
        0x10_0010,
        0x10_0000,
        64,
        None,
    )
    .unwrap();
    assert_eq!(s.read(0x10_0000, 64).unwrap(), data);
    assert_eq!(s.read(0x10_0040, 16).unwrap(), data[48..]);
}
