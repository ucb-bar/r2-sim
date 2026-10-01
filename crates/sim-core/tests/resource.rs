use sim_core::{CoCharge, Demand, Grant, Kind, Ratio, ResourceId, Resources};

fn reserve(r: &mut Resources, d: &[Demand], t: u64) -> Grant {
    r.reserve(d, t).unwrap()
}

#[test]
fn fractional_carry_keeps_the_rate() {
    // Rate 2/3 unit per cycle: one unit takes 1.5 cycles.
    let mut r = Resources::new(false);
    let f = r.add_flow("f", Kind::Bandwidth, Ratio::new(2, 3));
    let ends: Vec<u64> = (0..4)
        .map(|_| reserve(&mut r, &[Demand::new(f, 1)], 0).end)
        .collect();
    assert_eq!(ends, vec![2, 3, 5, 6]);
    // 100 more units at the rate: exactly 150 cycles in total from 6.
    let mut end = 0;
    for _ in 0..100 {
        end = reserve(&mut r, &[Demand::new(f, 1)], 0).end;
    }
    assert_eq!(end, 156);
}

#[test]
fn utilization_is_exclusive() {
    let mut r = Resources::new(false);
    let f = r.add_flow("muon", Kind::VectorOps, Ratio::int(256));
    let mut d = Demand::new(f, 256 * 100);
    d.utilization = Ratio::new(1, 2);
    assert_eq!(reserve(&mut r, &[d], 0).end, 200);
}

#[test]
fn gap_filling_is_causal() {
    let mut r = Resources::new(false);
    let f = r.add_flow("f", Kind::Bandwidth, Ratio::ONE);
    assert_eq!(reserve(&mut r, &[Demand::new(f, 10)], 20).start, 20);
    // A 10-cycle chunk fits in the hole before 20.
    let g = reserve(&mut r, &[Demand::new(f, 10)], 5);
    assert_eq!((g.start, g.end), (5, 15));
    // An 8-cycle chunk does not fit in [15, 20); it goes after 30.
    let g = reserve(&mut r, &[Demand::new(f, 8)], 12);
    assert_eq!((g.start, g.end), (30, 38));
    // Never before the requested time, even if the calendar is free.
    assert_eq!(reserve(&mut r, &[Demand::new(f, 1)], 100).start, 100);
}

#[test]
fn co_charging_fixpoint_and_culprit() {
    let mut r = Resources::new(true);
    r.set_cocharge(CoCharge::Longest);
    let a = r.add_flow("a", Kind::TensorOps, Ratio::ONE);
    let b = r.add_flow("b", Kind::Bandwidth, Ratio::ONE);
    // b is busy over [12, 20). a needs 10, b needs 2.
    reserve(&mut r, &[Demand::new(b, 8)], 12);
    // Own starts are 0 and 0; D = 10 still fits on b in [0, 12).
    let g = reserve(&mut r, &[Demand::new(a, 10), Demand::new(b, 2)], 0);
    assert_eq!((g.start, g.end, g.culprit), (0, 10, a));
    // Now a needs 5 and b needs 5 at t = 5: b has [10, 12) free only, so
    // D = 5 moves t to 20 on b; a is free there.
    let g = reserve(&mut r, &[Demand::new(a, 5), Demand::new(b, 5)], 5);
    assert_eq!((g.start, g.end), (20, 25));
    // Tie on duration: the culprit is the resource that delayed the start.
    assert_eq!(g.culprit, b);
    let s = r.stats();
    assert_eq!(s[a.0 as usize].busy_cycles, 15);
    assert_eq!(s[b.0 as usize].busy_cycles, 8 + 10 + 5);
}

#[test]
fn occupancy_is_fifo() {
    let mut r = Resources::new(false);
    let s = r.add_occupancy("slots", Kind::Slots, 2);
    assert!(r.try_acquire(s, 2));
    assert!(!r.try_acquire(s, 1));
    r.enqueue(s, 7);
    // A newcomer does not bypass the queued waiter.
    let waiters = r.release(s, 1);
    assert_eq!(waiters, vec![7]);
    assert!(r.try_acquire(s, 1));
    assert!(!r.fits(s, 1));
    let _: ResourceId = s;
}

#[test]
fn carry_only_on_resources_charged_their_own_duration() {
    let mut r = Resources::new(false);
    r.set_cocharge(CoCharge::Longest);
    let a = r.add_flow("a", Kind::TensorOps, Ratio::ONE);
    let b = r.add_flow("b", Kind::Bandwidth, Ratio::new(2, 3));
    // b needs 1.5 cycles (2 with a carry of 0.5), but is charged 10 with a.
    let g = reserve(&mut r, &[Demand::new(a, 10), Demand::new(b, 1)], 0);
    assert_eq!((g.start, g.end, g.culprit), (0, 10, a));
    // No carry on b: its next unit takes 2 cycles, not 1.
    let g = reserve(&mut r, &[Demand::new(b, 1)], 100);
    assert_eq!((g.start, g.end), (100, 102));
    // Carry kept where the own duration is the charged one.
    let g = reserve(&mut r, &[Demand::new(b, 1)], 100);
    assert_eq!((g.start, g.end), (102, 103));
}

#[test]
fn own_rule_charges_each_resource_its_own_duration() {
    // A fast shared resource (HBM, 4/cycle) on the path of slow links
    // (1/cycle each): chunk D = 8, but HBM is charged 2, so a second flow on
    // another link uses HBM at once.
    let mut r = Resources::new(false);
    let hbm = r.add_flow("hbm", Kind::Bandwidth, Ratio::int(4));
    let l1 = r.add_flow("l1", Kind::Bandwidth, Ratio::ONE);
    let l2 = r.add_flow("l2", Kind::Bandwidth, Ratio::ONE);
    let g1 = reserve(&mut r, &[Demand::new(hbm, 8), Demand::new(l1, 8)], 0);
    let g2 = reserve(&mut r, &[Demand::new(hbm, 8), Demand::new(l2, 8)], 0);
    assert_eq!((g1.start, g1.end, g1.culprit), (0, 8, l1));
    assert_eq!(
        (g2.start, g2.end),
        (2, 10),
        "HBM free after its own 2 cycles"
    );
    let s = r.stats();
    assert_eq!(s[hbm.0 as usize].busy_cycles, 4);
    // The second chunk queued on HBM for 2 cycles.
    assert_eq!(s[hbm.0 as usize].delay_cycles, 2);
    // The first rule holds HBM for the whole chunk.
    let mut r = Resources::new(false);
    r.set_cocharge(CoCharge::Longest);
    let hbm = r.add_flow("hbm", Kind::Bandwidth, Ratio::int(4));
    let l1 = r.add_flow("l1", Kind::Bandwidth, Ratio::ONE);
    let l2 = r.add_flow("l2", Kind::Bandwidth, Ratio::ONE);
    reserve(&mut r, &[Demand::new(hbm, 8), Demand::new(l1, 8)], 0);
    let g2 = reserve(&mut r, &[Demand::new(hbm, 8), Demand::new(l2, 8)], 0);
    assert_eq!((g2.start, g2.end), (8, 16));
}
