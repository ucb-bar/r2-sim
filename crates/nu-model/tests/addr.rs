use nu_model::{Addr, NeutrinoId, OwnerMap, Range};

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
    m.insert(Range::new(0x1_0000_0000, 0x2_0000_0000), NeutrinoId(1));
    m.insert(Range::new(0x0_0000_0000, 0x1_0000_0000), NeutrinoId(0));
    m.insert(Range::new(0x2_0000_0000, 0x3_0000_0000), NeutrinoId(2));
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
    m.insert(Range::new(0, 0x1000), NeutrinoId(0));
    m.insert(Range::new(0x800, 0x2000), NeutrinoId(1));
}
