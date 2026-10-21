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
        pool: None,
    }
}

#[test]
fn a_transfer_takes_its_hops_in_route_order() {
    // Route 256 -> 128 -> 512 B/cycle, 64 B beats (4, 2 and 8 slots),
    // latencies 5 and 7 after hops 0 and 1; 4096 B, a wide master.
    let mut r = Resources::new();
    let h = [256, 128, 512].map(|bw| {
        let id = r.add_flow("h", Kind::Bandwidth, Ratio::int(bw));
        r.set_beat(id, 64);
        id
    });
    let path = [hop(h[0], 4096, 5), hop(h[1], 4096, 7), hop(h[2], 4096, 0)];
    let wide = 1024;
    // Free path: hop 0 at 256 B/cycle, 16 cycles on its 4 slots, [10, 26);
    // hop 1 at 128, 32 cycles on its 2 slots from 10 + 5, [15, 47); hop 2
    // narrowed to 128: 2 of its 8 slots for 32 cycles from 15 + 7, [22,
    // 54). The chunk ends at the last hop's end. The service culprit is
    // the first hop with the longest window: hop 1, the narrowest (hop 2 is
    // narrowed to its rate).
    let g = r.reserve_path(&path, wide, 10).unwrap();
    assert_eq!((g.start, g.end, g.culprit), (10, 54, h[1]));
    let s = r.stats();
    assert_eq!(s[h[0].0 as usize].busy_cycles, 16 * 4);
    assert_eq!(s[h[1].0 as usize].busy_cycles, 32 * 2);
    assert_eq!(s[h[2].0 as usize].busy_cycles, 32 * 2);
    assert_eq!(s[h[2].0 as usize].slots, 8.0);
}

#[test]
fn a_hop_keeps_its_forward_window_before_a_busy_hop() {
    let mut r = Resources::new();
    let h = [256, 128, 512].map(|bw| {
        let id = r.add_flow("h", Kind::Bandwidth, Ratio::int(bw));
        r.set_beat(id, 64);
        id
    });
    let wide = 1024;
    // Another flow holds both slots of hop 1 over [15, 115) (12800 B at 128).
    let g = r.reserve_path(&[hop(h[1], 12800, 0)], wide, 15).unwrap();
    assert_eq!((g.start, g.end), (15, 115));
    // Hop 0 [10, 26); hop 1 waits to [115, 147); hop 2 [122, 154). Hop 0
    // keeps its window: the data waits between hops 0 and 1. The queue
    // culprit is hop 1 (100 cycles).
    let path = [hop(h[0], 4096, 5), hop(h[1], 4096, 7), hop(h[2], 4096, 0)];
    let g = r.reserve_path(&path, wide, 10).unwrap();
    assert_eq!((g.start, g.end), (10, 154));
    assert_eq!(g.queue_culprit, Some(h[1]));
    // Hop 0 is busy over [10, 26): the next 4-slot window starts at 26.
    let g = r.reserve_path(&[hop(h[0], 4096, 0)], wide, 10).unwrap();
    assert_eq!((g.start, g.end), (26, 42));
}

#[test]
fn narrow_flows_share_the_slots_of_a_wide_port() {
    // A 512 B/cycle port of 64 B beats has 8 slots. A 128 B/cycle flow
    // (its master's beat rate) holds 2 of them for 4096 / 128 = 32 cycles:
    // four run together, a fifth waits.
    let mut r = Resources::new();
    let p = r.add_flow("port", Kind::Bandwidth, Ratio::int(512));
    r.set_beat(p, 64);
    let narrow = 128;
    let ends: Vec<(u64, u64)> = (0..5)
        .map(|_| {
            let g = r.reserve_path(&[hop(p, 4096, 0)], narrow, 0).unwrap();
            (g.start, g.end)
        })
        .collect();
    assert_eq!(ends, vec![(0, 32), (0, 32), (0, 32), (0, 32), (32, 64)]);
    assert_eq!(r.stats()[p.0 as usize].busy_cycles, 5 * 32 * 2);
    // With one 512 B beat the port has one slot: each flow holds it all
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
fn an_array_bound_chunk_leaves_port_slots_to_a_dma() {
    // SMEM port 256 B/cycle of 64 B beats: 4 slots. A DMA at its 64 B beat
    // holds one slot for 6400 / 64 = 100 cycles, [0, 100).
    let mut r = Resources::new();
    let port = r.add_flow("smem.port", Kind::Bandwidth, Ratio::int(256));
    r.set_beat(port, 64);
    let array = r.add_flow("array", Kind::TensorOps, Ratio::int(10));
    let dma = r.reserve_path(&[hop(port, 6400, 0)], 64, 0).unwrap();
    assert_eq!((dma.start, dma.end), (0, 100));
    // An array-bound matmul: 1000 MAC at 10 per cycle, D = 100; its 12800
    // B on the port need ceil(12800 / (100 x 64)) = 2 slots for D. Two of
    // the three free slots: it runs beside the DMA.
    let mm = r
        .reserve(&[Demand::work(array, 1000), Demand::read(port, 12800)], 0)
        .unwrap();
    assert_eq!((mm.start, mm.end, mm.culprit), (0, 100, array));
    assert_eq!(r.stats()[port.0 as usize].busy_cycles, 100 + 2 * 100);
    // A port-bound matmul (51200 B: 200 cycles at the full port, D = 200)
    // needs all 4 slots and waits for the DMA's slot: [100, 300).
    let mm = r
        .reserve(&[Demand::work(array, 1000), Demand::read(port, 51200)], 0)
        .unwrap();
    assert_eq!((mm.start, mm.end, mm.culprit), (100, 300, port));
}

/// An SMEM port of 256 B/cycle in 4 slots of 64 B, an array of 1 op per
/// cycle, and a DMA chunk holding `dma_slots` slots (64 B/cycle each) of
/// the port for `dma_cycles` from cycle 0.
fn port_with_dma(dma_slots: u64, dma_cycles: u64) -> (Resources, ResourceId, ResourceId) {
    let mut r = Resources::new();
    let port = r.add_flow("smem.port", Kind::Bandwidth, Ratio::int(256));
    r.set_beat(port, 64);
    let array = r.add_flow("array", Kind::TensorOps, Ratio::ONE);
    if dma_slots > 0 {
        let bytes = dma_slots * 64 * dma_cycles;
        let g = r
            .reserve_path(&[hop(port, bytes, 0)], dma_slots * 64, 0)
            .unwrap();
        assert_eq!((g.start, g.end), (0, dma_cycles));
    }
    (r, port, array)
}

#[test]
fn a_delayed_compute_chunk_starts_on_the_free_slots() {
    // A port-bound Atlas tile: 41,472 B in 128 array cycles. With every
    // slot it takes 41472 / 256 = 162 cycles; beside a DMA that holds 2 of
    // the 4 slots for 256 cycles it would wait: [256, 418). On the 2 free
    // slots it runs at 128 B/cycle, 324 cycles, from 0: it ends sooner.
    let tile = |array| [Demand::work(array, 128), Demand::read(ResourceId(0), 41472)];
    let (mut r, port, array) = port_with_dma(2, 256);
    let mut d = tile(array);
    d[1].resource = port;
    let g = r.reserve(&d, 0).unwrap();
    assert_eq!((g.start, g.end), (0, 324));
    // It holds the 2 free slots for 324 cycles: 2 x 256 for the DMA plus 2
    // x 324.
    assert_eq!(r.stats()[port.0 as usize].busy_cycles, 2 * 256 + 2 * 324);
    assert_eq!(g.queue_culprit, None);
}

#[test]
fn a_chunk_that_does_not_wait_keeps_every_slot() {
    let (mut r, port, array) = port_with_dma(0, 0);
    let g = r
        .reserve(&[Demand::work(array, 128), Demand::read(port, 41472)], 0)
        .unwrap();
    assert_eq!((g.start, g.end, g.culprit), (0, 162, port));
    assert_eq!(r.stats()[port.0 as usize].busy_cycles, 4 * 162);
}

#[test]
fn the_free_slots_are_refused_when_they_end_later() {
    // A DMA holds 3 of the 4 slots for 50 cycles. Full slots: [50, 212).
    // On the one free slot: 41472 / 64 = 648 cycles, ending at 648: refused.
    let (mut r, port, array) = port_with_dma(3, 50);
    let g = r
        .reserve(&[Demand::work(array, 128), Demand::read(port, 41472)], 0)
        .unwrap();
    assert_eq!((g.start, g.end), (50, 212));
    assert_eq!(r.stats()[port.0 as usize].busy_cycles, 3 * 50 + 4 * 162);
}

/// A data resource of `bw` B/cycle in beats of `beat` bytes.
fn data(r: &mut Resources, name: &str, bw: u64, beat: u64) -> ResourceId {
    let id = r.add_flow(name, Kind::Bandwidth, Ratio::int(bw));
    r.set_beat(id, beat);
    id
}

#[test]
fn a_narrow_master_fills_a_one_slot_wide_link() {
    // A 4 B master on a link of one 32 B slot: every 4 B beat takes the
    // slot (beats never merge), so 4096 B take 1024 cycles, the link is
    // full, and its byte utilization is 4096 / (32 x 1024) = 1/8.
    let mut r = Resources::new();
    let link = data(&mut r, "link", 32, 32);
    let g = r.reserve_path(&[hop(link, 4096, 0)], 4, 0).unwrap();
    assert_eq!((g.start, g.end), (0, 1024));
    let s = &r.stats()[link.0 as usize];
    assert_eq!((s.slots, s.busy_slot_cycles, s.bytes), (1.0, 1024.0, 4096));
    assert_eq!(s.bytes as f64 / (s.rate * 1024.0), 0.125);
    // A second 4 B flow waits for the slot.
    let g = r.reserve_path(&[hop(link, 4096, 0)], 4, 0).unwrap();
    assert_eq!((g.start, g.end), (1024, 2048));
}

#[test]
fn a_narrowed_flow_holds_half_a_slot_and_another_uses_the_other_half() {
    // Each flow comes from its own port of 16 B/cycle in 32 B beats (half
    // a slot) onto a link of one 32 B slot. Narrowed to 16 B/cycle, a flow
    // needs 16 / 32 = half of the link's slot, so the two run together:
    // 4096 B, 128 beats at half a slot, 256 cycles each.
    let mut r = Resources::new();
    let (p0, p1) = (data(&mut r, "p0", 16, 32), data(&mut r, "p1", 16, 32));
    let link = data(&mut r, "link", 32, 32);
    let a = r
        .reserve_path(&[hop(p0, 4096, 0), hop(link, 4096, 0)], 32, 0)
        .unwrap();
    let b = r
        .reserve_path(&[hop(p1, 4096, 0), hop(link, 4096, 0)], 32, 0)
        .unwrap();
    assert_eq!((a.start, a.end), (0, 256));
    assert_eq!((b.start, b.end), (0, 256));
    // Together they fill the link: 2 x 0.5 slot x 256 cycles.
    assert_eq!(r.stats()[link.0 as usize].busy_slot_cycles, 256.0);
}

#[test]
fn a_clock_ratio_scales_the_slots_but_not_the_beat_rate() {
    // 64 B per own cycle in 32 B beats at clock 7/8: 1.75 slots per core
    // cycle. A flow of one 32 B beat per cycle still moves a whole beat in
    // every core cycle: 4096 B in 128 cycles.
    let mut r = Resources::new();
    let port = data(&mut r, "port", 64, 32);
    r.set_clock(port, Ratio::new(7, 8));
    assert_eq!(r.flow(port).capacity(), Ratio::new(7, 4));
    assert_eq!(r.stats()[port.0 as usize].slots, 1.75);
    let g = r.reserve_path(&[hop(port, 4096, 0)], 32, 0).unwrap();
    assert_eq!((g.start, g.end), (0, 128));
    // A second flow takes the 0.75 slot left: 128 beats in ceil(128 /
    // 0.75) = 171 cycles from 0 (window B), sooner than one slot after the
    // first flow, [128, 256) (window A).
    let g = r.reserve_path(&[hop(port, 4096, 0)], 32, 0).unwrap();
    assert_eq!((g.start, g.end), (0, 171));
}

#[test]
fn a_hop_takes_the_window_that_ends_first() {
    // A port of two 64 B slots; another flow holds one slot over [0, 100).
    let setup = || {
        let mut r = Resources::new();
        let port = data(&mut r, "port", 128, 64);
        let g = r.reserve_path(&[hop(port, 6400, 0)], 64, 0).unwrap();
        assert_eq!((g.start, g.end), (0, 100));
        (r, port)
    };
    // 6400 B needing both slots: window A from 100 for 50 cycles ends at
    // 150; window B on the free slot from 0 for 100 cycles ends at 100: B.
    let (mut r, port) = setup();
    let g = r.reserve_path(&[hop(port, 6400, 0)], 128, 0).unwrap();
    assert_eq!((g.start, g.end), (0, 100));
    // 25,600 B: window A [100, 300); window B 400 cycles on one slot, to
    // 400: A.
    let (mut r, port) = setup();
    let g = r.reserve_path(&[hop(port, 25_600, 0)], 128, 0).unwrap();
    assert_eq!((g.start, g.end), (100, 300));
}

#[test]
fn a_chunk_on_few_slots_does_not_speed_up_when_the_other_flow_ends() {
    // A port of two 64 B slots; another flow holds one over [0, 100). A
    // 10,000 B chunk needs both slots (157 beats): window A [100, 179),
    // window B on one slot [0, 157): B. Exact sharing would take both
    // slots at 100 and end at 129; the chunk keeps its one slot to 157
    // (no re-planning), so a flow arriving at 100 finds the other slot
    // free at once.
    let mut r = Resources::new();
    let port = data(&mut r, "port", 128, 64);
    r.reserve_path(&[hop(port, 6400, 0)], 64, 0).unwrap();
    let g = r.reserve_path(&[hop(port, 10_000, 0)], 128, 0).unwrap();
    assert_eq!((g.start, g.end), (0, 157));
    let g = r.reserve_path(&[hop(port, 640, 0)], 64, 100).unwrap();
    assert_eq!((g.start, g.end), (100, 110));
}

#[test]
fn a_compute_chunk_holds_a_fraction_of_the_port_beside_a_dma() {
    // A port of four 64 B slots; a DMA holds one over [0, 100). An
    // array-bound chunk (1000 MAC at 10 per cycle, D = 100) reads 9600 B,
    // 150 beats: it holds 150 / 100 = 1.5 slots for D (no rounding to
    // whole slots), beside the DMA.
    let mut r = Resources::new();
    let port = data(&mut r, "port", 256, 64);
    let array = r.add_flow("array", Kind::TensorOps, Ratio::int(10));
    r.reserve_path(&[hop(port, 6400, 0)], 64, 0).unwrap();
    let g = r
        .reserve(&[Demand::work(array, 1000), Demand::read(port, 9600)], 0)
        .unwrap();
    assert_eq!((g.start, g.end, g.culprit), (0, 100, array));
    assert_eq!(r.stats()[port.0 as usize].busy_slot_cycles, 100.0 + 150.0);
    // 2.5 of the 4 slots are in use over [0, 100): a 1.5-slot flow (96
    // B/cycle in 64 B beats) fits beside them.
    let g = r.reserve_path(&[hop(port, 9600, 0)], 96, 0).unwrap();
    assert_eq!((g.start, g.end), (0, 100));
}

/// The end of a chunk of `beats` beats that holds the whole capacity of
/// `port` from `t0` (a wide master).
fn whole_end(r: &mut Resources, port: ResourceId, beat: u64, beats: u64, t0: u64) -> u64 {
    let g = r
        .reserve_path(&[hop(port, beats * beat, 0)], 1024, t0)
        .unwrap();
    assert_eq!(g.start, t0);
    g.end
}

#[test]
fn a_fractional_capacity_has_whole_slots_by_cycle() {
    // r2sim spec Section 5.2: k(t) = floor((t + 1) C) - floor(t C) slots at cycle t.
    // A chunk that holds the whole capacity from 0 ends at the first cycle
    // whose slots hold its beats. Each case runs on a new resource.
    let ends = |bw: u64, beat: u64, n: u64| -> Vec<u64> {
        (1..=n)
            .map(|beats| {
                let mut r = Resources::new();
                let port = data(&mut r, "port", bw, beat);
                whole_end(&mut r, port, beat, beats, 0)
            })
            .collect()
    };
    // C = 3/2: 1, 2, 1, 2 slots in cycles 0, 1, 2, 3.
    assert_eq!(ends(3, 2, 6), vec![1, 2, 2, 3, 4, 4]);
    // C = 3/10: a slot in cycles 3, 6, 9, 13, 16, 19: 3 in every 10.
    assert_eq!(ends(3, 10, 6), vec![4, 7, 10, 14, 17, 20]);
    // C = 7/8: a slot in every cycle except 0, 8, 16, ..: 7 in every 8.
    assert_eq!(ends(7, 8, 8), vec![2, 3, 4, 5, 6, 7, 8, 10]);
    // From cycle 1 under C = 3/2, cycle 1 holds 2 beats.
    let mut r = Resources::new();
    let port = data(&mut r, "port", 3, 2);
    assert_eq!(whole_end(&mut r, port, 2, 2, 1), 2);
}

#[test]
fn eight_beats_on_a_whole_7_8_port_take_9_or_10_cycles() {
    // A chunk that holds the whole 7/8 port: 8 beats need 8 slots, and the
    // window from t0 holds them in 10 cycles if it contains two slot-less
    // cycles (t0 = 0 mod 8) and in 9 cycles otherwise.
    for t0 in 0..16 {
        let mut r = Resources::new();
        let port = data(&mut r, "port", 7, 8);
        let d = whole_end(&mut r, port, 8, 8, t0) - t0;
        assert_eq!(d, if t0 % 8 == 0 { 10 } else { 9 }, "t0 {t0}");
    }
    // A stream of such chunks, each after the one before: 7 chunks (56
    // beats) end at 56 x 8/7 = 64, 8/7 cycles per beat on average.
    let mut r = Resources::new();
    let port = data(&mut r, "port", 7, 8);
    let mut ends = Vec::new();
    for _ in 0..7 {
        ends.push(r.reserve_path(&[hop(port, 64, 0)], 1024, 0).unwrap().end);
    }
    assert_eq!(ends, vec![10, 19, 28, 37, 46, 55, 64]);
}

#[test]
fn an_integer_capacity_keeps_its_windows() {
    // C = 2: 10 beats on both slots take ceil(10 / 2) = 5 cycles, and on
    // one slot (a 64 B master) 10 cycles, from any start.
    for t0 in 0..4 {
        let mut r = Resources::new();
        let port = data(&mut r, "port", 128, 64);
        assert_eq!(whole_end(&mut r, port, 64, 10, t0), t0 + 5);
        let g = r.reserve_path(&[hop(port, 640, 0)], 64, t0 + 5).unwrap();
        assert_eq!((g.start, g.end), (t0 + 5, t0 + 15));
    }
}

#[test]
fn two_chunks_split_each_cycles_slots_by_usage() {
    // A port of C = 3/2 (3 B/cycle in 2 B beats). Two compute chunks of
    // D = 4 (4 work units at 1 per cycle, each on its own array) read 4
    // and 2 beats: usages 1 and 0.5. Each moves usage / C of the slots of
    // its window: N = ceil(beats x C / usage) = 6 stream slots for both,
    // and floor(4 x 3/2) = 6 slots in [0, 4): both end at 4, and the 6
    // slots carry 4 + 2 beats.
    let mut r = Resources::new();
    let port = data(&mut r, "port", 3, 2);
    let a0 = r.add_flow("a0", Kind::TensorOps, Ratio::int(1));
    let a1 = r.add_flow("a1", Kind::TensorOps, Ratio::int(1));
    let g = r
        .reserve(&[Demand::work(a0, 4), Demand::read(port, 8)], 0)
        .unwrap();
    assert_eq!((g.start, g.end), (0, 4));
    let g = r
        .reserve(&[Demand::work(a1, 4), Demand::read(port, 4)], 0)
        .unwrap();
    assert_eq!((g.start, g.end), (0, 4));
    // Usage x span: 1 x 4 + 0.5 x 4.
    assert_eq!(r.stats()[port.0 as usize].busy_slot_cycles, 6.0);
}

#[test]
fn a_message_on_a_hop_below_one_slot_moves_in_its_slot_cycles() {
    // A 4 B/cycle hop in 64 B beats: C = 1/16, one slot in each cycle
    // 15 mod 16. A 1-beat message holds the whole capacity up to the next
    // slot cycle; a 2-beat message to the second.
    let mut r = Resources::new();
    let h = data(&mut r, "h", 4, 64);
    let g = r.reserve_message(hop(h, 64, 0), 64, 3);
    assert_eq!((g.start, g.end), (3, 16));
    let g = r.reserve_message(hop(h, 128, 0), 64, 16);
    assert_eq!((g.start, g.end), (16, 48));
    let g = r.reserve_message(hop(h, 64, 0), 64, 47);
    assert_eq!((g.start, g.end), (48, 64));
}
