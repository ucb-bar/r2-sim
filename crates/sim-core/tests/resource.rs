use sim_core::{CoCharge, Demand, Dir, Grant, Kind, PathHop, Ratio, ResourceId, Resources};

fn reserve(r: &mut Resources, d: &[Demand], t: u64) -> Grant {
    r.reserve(d, t).unwrap()
}

#[test]
fn transfers_take_whole_cycles_without_carry() {
    // A link moves whole beats: two half-width messages take two cycles.
    let mut r = Resources::new();
    let f = r.add_flow("ctl", Kind::Bandwidth, Ratio::int(128));
    let ends: Vec<u64> = (0..2)
        .map(|_| reserve(&mut r, &[Demand::write(f, 64)], 0).end)
        .collect();
    assert_eq!(ends, vec![1, 2]);
    // 64 KiB on a 273 B/cycle port: ceil(65536 / 273) = 241 cycles, and
    // the next chunk starts fresh: 241 again.
    let p = r.add_flow("port", Kind::Bandwidth, Ratio::int(273));
    let g = reserve(&mut r, &[Demand::read(p, 65536)], 0);
    assert_eq!((g.start, g.end), (0, 241));
    let g = reserve(&mut r, &[Demand::read(p, 65536)], 0);
    assert_eq!((g.start, g.end), (241, 482));
    // A rational rate: one unit at 2/3 per cycle takes 2 cycles each time.
    let q = r.add_flow("q", Kind::Bandwidth, Ratio::new(2, 3));
    let ends: Vec<u64> = (0..3)
        .map(|_| reserve(&mut r, &[Demand::work(q, 1)], 0).end)
        .collect();
    assert_eq!(ends, vec![2, 4, 6]);
}

#[test]
fn a_small_transfer_takes_one_cycle() {
    // 1 B on a 273 B/cycle port and 1 lane-op on a half-used 256-wide
    // vector unit: one whole cycle each, never zero.
    let mut r = Resources::new();
    let p = r.add_flow("port", Kind::Bandwidth, Ratio::int(273));
    let mut v = Demand::work(r.add_flow("muon", Kind::VectorOps, Ratio::int(256)), 1);
    v.utilization = Ratio::new(1, 2);
    let g = reserve(&mut r, &[Demand::read(p, 1)], 5);
    assert_eq!((g.start, g.end), (5, 6));
    let g = reserve(&mut r, &[Demand::read(p, 1)], 5);
    assert_eq!((g.start, g.end), (6, 7));
    let g = reserve(&mut r, &[v], 0);
    assert_eq!((g.start, g.end), (0, 1));
}

#[test]
fn utilization_is_exclusive() {
    let mut r = Resources::new();
    let f = r.add_flow("muon", Kind::VectorOps, Ratio::int(256));
    let mut d = Demand::work(f, 256 * 100);
    d.utilization = Ratio::new(1, 2);
    assert_eq!(reserve(&mut r, &[d], 0).end, 200);
}

#[test]
fn gap_filling_is_causal() {
    let mut r = Resources::new();
    let f = r.add_flow("f", Kind::Bandwidth, Ratio::ONE);
    assert_eq!(reserve(&mut r, &[Demand::work(f, 10)], 20).start, 20);
    // A 10-cycle chunk fits in the hole before 20.
    let g = reserve(&mut r, &[Demand::work(f, 10)], 5);
    assert_eq!((g.start, g.end), (5, 15));
    // An 8-cycle chunk does not fit in [15, 20); it goes after 30.
    let g = reserve(&mut r, &[Demand::work(f, 8)], 12);
    assert_eq!((g.start, g.end), (30, 38));
    // Never before the requested time, even if the calendar is free.
    assert_eq!(reserve(&mut r, &[Demand::work(f, 1)], 100).start, 100);
}

#[test]
fn co_charging_fixpoint_and_culprit() {
    let mut r = Resources::new();
    r.set_cocharge(CoCharge::Longest);
    let a = r.add_flow("a", Kind::TensorOps, Ratio::ONE);
    let b = r.add_flow("b", Kind::Bandwidth, Ratio::ONE);
    // b is busy over [12, 20). a needs 10, b needs 2.
    reserve(&mut r, &[Demand::work(b, 8)], 12);
    // Own starts are 0 and 0; D = 10 still fits on b in [0, 12).
    let g = reserve(&mut r, &[Demand::work(a, 10), Demand::work(b, 2)], 0);
    assert_eq!((g.start, g.end, g.culprit), (0, 10, a));
    // Now a needs 5 and b needs 5 at t = 5: b has [10, 12) free only, so
    // D = 5 moves t to 20 on b; a is free there.
    let g = reserve(&mut r, &[Demand::work(a, 5), Demand::work(b, 5)], 5);
    assert_eq!((g.start, g.end), (20, 25));
    // Tie on duration: the culprit is the resource that delayed the start.
    assert_eq!(g.culprit, b);
    let s = r.stats();
    assert_eq!(s[a.0 as usize].busy_cycles, 15);
    assert_eq!(s[b.0 as usize].busy_cycles, 8 + 10 + 5);
}

#[test]
fn occupancy_is_fifo() {
    let mut r = Resources::new();
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
fn own_rule_charges_each_resource_its_own_duration() {
    // A fast shared resource (HBM, 4/cycle) on the path of slow links
    // (1/cycle each): chunk D = 8, but HBM is charged 2, so a second flow on
    // another link uses HBM at once.
    let mut r = Resources::new();
    let hbm = r.add_flow("hbm", Kind::Bandwidth, Ratio::int(4));
    let l1 = r.add_flow("l1", Kind::Bandwidth, Ratio::ONE);
    let l2 = r.add_flow("l2", Kind::Bandwidth, Ratio::ONE);
    let g1 = reserve(&mut r, &[Demand::work(hbm, 8), Demand::work(l1, 8)], 0);
    let g2 = reserve(&mut r, &[Demand::work(hbm, 8), Demand::work(l2, 8)], 0);
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
    let mut r = Resources::new();
    r.set_cocharge(CoCharge::Longest);
    let hbm = r.add_flow("hbm", Kind::Bandwidth, Ratio::int(4));
    let l1 = r.add_flow("l1", Kind::Bandwidth, Ratio::ONE);
    let l2 = r.add_flow("l2", Kind::Bandwidth, Ratio::ONE);
    reserve(&mut r, &[Demand::work(hbm, 8), Demand::work(l1, 8)], 0);
    let g2 = reserve(&mut r, &[Demand::work(hbm, 8), Demand::work(l2, 8)], 0);
    assert_eq!((g2.start, g2.end), (8, 16));
}

fn hop(resource: ResourceId, amount: u64, latency: u64) -> PathHop {
    PathHop {
        resource,
        amount,
        dir: Dir::Read,
        latency,
    }
}

#[test]
fn a_transfer_takes_its_hops_in_route_order() {
    // Route 256 -> 128 -> 512 B/cycle, 64 B beats (4, 2 and 8 lanes),
    // latencies 5 and 7 after hops 0 and 1; 4096 B, a wide master.
    let mut r = Resources::new();
    let h = [256, 128, 512].map(|bw| {
        let id = r.add_flow("h", Kind::Bandwidth, Ratio::int(bw));
        r.set_beat(id, 64);
        id
    });
    let path = [hop(h[0], 4096, 5), hop(h[1], 4096, 7), hop(h[2], 4096, 0)];
    let wide = Ratio::int(1024);
    // Free path: hop 0 at 256 B/cycle, 16 cycles on its 4 lanes, [10, 26);
    // hop 1 at 128, 32 cycles on its 2 lanes from 10 + 5, [15, 47); hop 2
    // narrowed to 128: 2 of its 8 lanes for 32 cycles from 15 + 7, [22,
    // 54). The chunk ends at the last hop's end. The service culprit is
    // the first hop with the longest window: hop 1, the narrowest (hop 2 is
    // narrowed to its rate).
    let g = r.reserve_path(&path, wide, 10).unwrap();
    assert_eq!((g.start, g.end, g.culprit), (10, 54, h[1]));
    let s = r.stats();
    assert_eq!(s[h[0].0 as usize].busy_cycles, 16 * 4);
    assert_eq!(s[h[1].0 as usize].busy_cycles, 32 * 2);
    assert_eq!(s[h[2].0 as usize].busy_cycles, 32 * 2);
    assert_eq!(s[h[2].0 as usize].lanes, 8);
}

#[test]
fn the_backward_pass_moves_a_hop_up_to_its_successor() {
    let mut r = Resources::new();
    let h = [256, 128, 512].map(|bw| {
        let id = r.add_flow("h", Kind::Bandwidth, Ratio::int(bw));
        r.set_beat(id, 64);
        id
    });
    let wide = Ratio::int(1024);
    // Another flow holds both lanes of hop 1 over [15, 115) (12800 B at 128).
    let g = r.reserve_path(&[hop(h[1], 12800, 0)], wide, 15).unwrap();
    assert_eq!((g.start, g.end), (15, 115));
    // Forward: hop 0 [10, 26); hop 1 waits to [115, 147); hop 2 [122, 154).
    // Backward: hop 1 starts 115 - 26 - 5 = 84 cycles after hop 0's end,
    // more than one skid (latency 5 + 1): hop 0 moves to end at 115 - 5 -
    // 1 = 109, [93, 109). The queue culprit is hop 1 (100 cycles).
    let path = [hop(h[0], 4096, 5), hop(h[1], 4096, 7), hop(h[2], 4096, 0)];
    let g = r.reserve_path(&path, wide, 10).unwrap();
    assert_eq!((g.start, g.end), (93, 154));
    assert_eq!(g.queue_culprit, Some(h[1]));
    // Hop 0 is free again over [10, 26): a 16-cycle 4-lane window fits.
    let g = r.reserve_path(&[hop(h[0], 4096, 0)], wide, 10).unwrap();
    assert_eq!((g.start, g.end), (10, 26));
}

#[test]
fn narrow_flows_share_the_lanes_of_a_wide_port() {
    // A 512 B/cycle port of 64 B beats has 8 lanes. A 128 B/cycle flow
    // (its master's beat rate) holds 2 of them for 4096 / 128 = 32 cycles:
    // four run together, a fifth waits.
    let mut r = Resources::new();
    let p = r.add_flow("port", Kind::Bandwidth, Ratio::int(512));
    r.set_beat(p, 64);
    let narrow = Ratio::int(128);
    let ends: Vec<(u64, u64)> = (0..5)
        .map(|_| {
            let g = r.reserve_path(&[hop(p, 4096, 0)], narrow, 0).unwrap();
            (g.start, g.end)
        })
        .collect();
    assert_eq!(ends, vec![(0, 32), (0, 32), (0, 32), (0, 32), (32, 64)]);
    assert_eq!(r.stats()[p.0 as usize].busy_cycles, 5 * 32 * 2);
    // With one 512 B beat the port has one lane: each flow holds it all
    // for 32 cycles, so two flows serialize.
    let mut r = Resources::new();
    let p = r.add_flow("port", Kind::Bandwidth, Ratio::int(512));
    r.set_beat(p, 512);
    let ends: Vec<(u64, u64)> = (0..2)
        .map(|_| {
            let g = r.reserve_path(&[hop(p, 4096, 0)], narrow, 0).unwrap();
            (g.start, g.end)
        })
        .collect();
    assert_eq!(ends, vec![(0, 32), (32, 64)]);
}

#[test]
fn an_array_bound_chunk_leaves_port_lanes_to_a_dma() {
    // SMEM port 256 B/cycle of 64 B beats: 4 lanes. A DMA at its 64 B beat
    // holds one lane for 6400 / 64 = 100 cycles, [0, 100).
    let mut r = Resources::new();
    let port = r.add_flow("smem.port", Kind::Bandwidth, Ratio::int(256));
    r.set_beat(port, 64);
    let array = r.add_flow("array", Kind::TensorOps, Ratio::int(10));
    let dma = r
        .reserve_path(&[hop(port, 6400, 0)], Ratio::int(64), 0)
        .unwrap();
    assert_eq!((dma.start, dma.end), (0, 100));
    // An array-bound matmul: 1000 MAC at 10 per cycle, D = 100; its 12800
    // B on the port need ceil(12800 / (100 x 64)) = 2 lanes for D. Two of
    // the three free lanes: it runs beside the DMA.
    let mm = r
        .reserve(&[Demand::work(array, 1000), Demand::read(port, 12800)], 0)
        .unwrap();
    assert_eq!((mm.start, mm.end, mm.culprit), (0, 100, array));
    assert_eq!(r.stats()[port.0 as usize].busy_cycles, 100 + 2 * 100);
    // A port-bound matmul (51200 B: 200 cycles at the full port, D = 200)
    // needs all 4 lanes and waits for the DMA's lane: [100, 300).
    let mm = r
        .reserve(&[Demand::work(array, 1000), Demand::read(port, 51200)], 0)
        .unwrap();
    assert_eq!((mm.start, mm.end, mm.culprit), (100, 300, port));
}

/// An SMEM port of 256 B/cycle in 4 lanes of 64 B, an array of 1 op per
/// cycle, and a DMA chunk holding `dma_lanes` lanes (64 B/cycle each) of
/// the port for `dma_cycles` from cycle 0.
fn port_with_dma(dma_lanes: u64, dma_cycles: u64) -> (Resources, ResourceId, ResourceId) {
    let mut r = Resources::new();
    let port = r.add_flow("smem.port", Kind::Bandwidth, Ratio::int(256));
    r.set_beat(port, 64);
    let array = r.add_flow("array", Kind::TensorOps, Ratio::ONE);
    if dma_lanes > 0 {
        let bytes = dma_lanes * 64 * dma_cycles;
        let g = r
            .reserve_path(&[hop(port, bytes, 0)], Ratio::int(dma_lanes * 64), 0)
            .unwrap();
        assert_eq!((g.start, g.end), (0, dma_cycles));
    }
    (r, port, array)
}

#[test]
fn a_delayed_compute_chunk_starts_on_the_free_lanes() {
    // A port-bound Atlas tile: 41,472 B in 128 array cycles. With every
    // lane it takes 41472 / 256 = 162 cycles; beside a DMA that holds 2 of
    // the 4 lanes for 256 cycles it would wait: [256, 418). On the 2 free
    // lanes it runs at 128 B/cycle, 324 cycles, from 0: it ends sooner.
    let tile = |array| [Demand::work(array, 128), Demand::read(ResourceId(0), 41472)];
    let (mut r, port, array) = port_with_dma(2, 256);
    let mut d = tile(array);
    d[1].resource = port;
    let g = r.reserve(&d, 0).unwrap();
    assert_eq!((g.start, g.end), (0, 324));
    // It holds the 2 free lanes for 324 cycles: 2 x 256 for the DMA plus 2
    // x 324.
    assert_eq!(r.stats()[port.0 as usize].busy_cycles, 2 * 256 + 2 * 324);
    assert_eq!(g.queue_culprit, None);
}

#[test]
fn a_chunk_that_does_not_wait_keeps_every_lane() {
    let (mut r, port, array) = port_with_dma(0, 0);
    let g = r
        .reserve(&[Demand::work(array, 128), Demand::read(port, 41472)], 0)
        .unwrap();
    assert_eq!((g.start, g.end, g.culprit), (0, 162, port));
    assert_eq!(r.stats()[port.0 as usize].busy_cycles, 4 * 162);
}

#[test]
fn the_free_lanes_are_refused_when_they_end_later() {
    // A DMA holds 3 of the 4 lanes for 50 cycles. Full lanes: [50, 212).
    // On the one free lane: 41472 / 64 = 648 cycles, ending at 648: refused.
    let (mut r, port, array) = port_with_dma(3, 50);
    let g = r
        .reserve(&[Demand::work(array, 128), Demand::read(port, 41472)], 0)
        .unwrap();
    assert_eq!((g.start, g.end), (50, 212));
    assert_eq!(r.stats()[port.0 as usize].busy_cycles, 3 * 50 + 4 * 162);
}
