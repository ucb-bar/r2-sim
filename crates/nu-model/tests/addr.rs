use nu_model::{Addr, AddressSet, NeutrinoId, OwnerMap, Range};

#[test]
fn overlap_is_half_open() {
    let a = Range::new(0x1000, 0x2000);
    assert!(a.overlaps(&Range::new(0x1800, 0x3000)));
    assert!(a.overlaps(&Range::new(0x0000, 0x1200)));
    assert!(a.overlaps(&Range::new(0x1200, 0x1400)));
    assert!(!a.overlaps(&Range::new(0x2000, 0x3000)));
    assert!(!a.overlaps(&Range::new(0x0000, 0x1000)));
    assert!(!a.overlaps(&Range::new(0x1800, 0x1800)));
}

#[test]
fn alignment_uses_granule() {
    assert!(Range::new(0x0, 0x200).is_aligned(512));
    assert!(!Range::new(0x0, 0x201).is_aligned(512));
    assert!(!Range::new(0x100, 0x400).is_aligned(512));
}

#[test]
fn owner_map_routes_by_prefix() {
    let mut m = OwnerMap::default();
    m.insert_range(&Range::new(0x1_0000_0000, 0x2_0000_0000), NeutrinoId(1));
    m.insert_range(&Range::new(0x0_0000_0000, 0x1_0000_0000), NeutrinoId(0));
    m.insert_range(&Range::new(0x2_0000_0000, 0x3_0000_0000), NeutrinoId(2));
    assert_eq!(m.owner_of(Addr(0x0_1234)), Some(NeutrinoId(0)));
    assert_eq!(m.owner_of(Addr(0x1_0000_0000)), Some(NeutrinoId(1)));
    assert_eq!(m.owner_of(Addr(0x2_ffff_ffff)), Some(NeutrinoId(2)));
    assert_eq!(m.owner_of(Addr(0x3_0000_0000)), None);
    assert_eq!(
        m.owner_of_range(&Range::new(0x1_0000_1000, 0x1_0000_2000)),
        Some(NeutrinoId(1))
    );
    assert_eq!(
        m.owner_of_range(&Range::new(0x0_ffff_f000, 0x1_0000_1000)),
        None
    );
}

#[test]
#[should_panic]
fn owner_map_rejects_overlap() {
    let mut m = OwnerMap::default();
    m.insert_range(&Range::new(0, 0x1000), NeutrinoId(0));
    m.insert_range(&Range::new(0x800, 0x2000), NeutrinoId(1));
}

#[test]
fn address_set_contains_overlaps_and_run() {
    let b = AddressSet::block(0x1000, 0x1000);
    assert!(b.contains(Addr(0x1000)) && b.contains(Addr(0x1fff)));
    assert!(!b.contains(Addr(0xfff)) && !b.contains(Addr(0x2000)));
    assert_eq!(b.run(0x1000), 0x1000);
    assert_eq!(b.run(0x1800), 0x800);
    assert_eq!(b.run(0x2000), 0);
    assert!(b.covers(&Range::new(0x1400, 0x2000)));
    assert!(!b.covers(&Range::new(0x1400, 0x2001)));
    assert!(b.is_contiguous());
    // Slice 1 of 4 slices of 512 B over [0, 64 KiB).
    let s1 = AddressSet::new(1 << 9, 0xffff & !(3 << 9));
    assert!(!s1.is_contiguous());
    assert!(s1.contains(Addr(0x200)) && s1.contains(Addr(0xa00)));
    assert!(!s1.contains(Addr(0x400)) && !s1.contains(Addr(0x1_0200)));
    assert_eq!(s1.run(0x200), 512);
    assert_eq!(s1.run(0x2f0), 0x110);
    assert_eq!(s1.size(), 16 * 1024);
    assert!(s1.covers(&Range::new(0xa00, 0xc00)));
    assert!(!s1.covers(&Range::new(0xa00, 0xc01)));
    let s2 = AddressSet::new(2 << 9, 0xffff & !(3 << 9));
    assert!(!s1.overlaps(&s2));
    assert!(s1.overlaps(&AddressSet::block(0, 0x1_0000)));
    assert!(!s1.overlaps(&AddressSet::block(0x1_0000, 0x1_0000)));
}

#[test]
fn a_range_decomposes_into_the_fewest_aligned_blocks() {
    let sets = AddressSet::decompose(&Range::new(0, 0xF000_0000));
    let sizes: Vec<u64> = sets.iter().map(|s| s.size()).collect();
    assert_eq!(sizes, [0x8000_0000, 0x4000_0000, 0x2000_0000, 0x1000_0000]);
    let sets = AddressSet::decompose(&Range::new(0x1000, 0x5000));
    assert_eq!(
        sets,
        [
            AddressSet::block(0x1000, 0x1000),
            AddressSet::block(0x2000, 0x2000),
            AddressSet::block(0x4000, 0x1000)
        ]
    );
    assert!(AddressSet::decompose(&Range::new(0x100_0000, 0x200_0000)).len() == 1);
    assert!(AddressSet::decompose(&Range::new(7, 7)).is_empty());
}

#[test]
fn an_owner_range_across_its_decomposed_sets_has_one_owner() {
    let mut m = OwnerMap::default();
    m.insert_range(&Range::new(0, 0xF000_0000), NeutrinoId(0));
    m.insert_range(&Range::new(0xF000_0000, 0x1_0000_0000), NeutrinoId(1));
    // Crosses the boundary of 0x8000_0000 between two sets of owner 0.
    assert_eq!(
        m.owner_of_range(&Range::new(0x7fff_f000, 0x8000_1000)),
        Some(NeutrinoId(0))
    );
    assert_eq!(
        m.owner_of_range(&Range::new(0xefff_f000, 0xf000_1000)),
        None
    );
    // Slices of one owner.
    let mut m = OwnerMap::default();
    for i in 0..2u64 {
        m.insert(
            AddressSet::new(i << 9, 0xfff & !(1 << 9)),
            NeutrinoId(i as u32),
        );
    }
    assert_eq!(m.owner_of(Addr(0x600)), Some(NeutrinoId(1)));
    assert_eq!(
        m.owner_of_range(&Range::new(0x400, 0x600)),
        Some(NeutrinoId(0))
    );
    assert_eq!(m.owner_of_range(&Range::new(0x400, 0x601)), None);
}
