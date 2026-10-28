//! The single-Neutrino matmul example of Neutrino spec Section 10.2, end to end in
//! timing-only mode, and lifecycle tests around it. Rates and costs are in
//! `common/mod.rs`.

mod common;

use common::{node, smem, sync, v1, Knobs};
use nu_model::{Dep, DepError, Range, Warning};
use sim_core::{Blocked, BlockedProgram, FatalError, SimWarning, Stats};

const KB8: u64 = 8192;

#[test]
fn matmul_end_to_end() {
    let n = node(Knobs::default());
    let (a, b, c) = (smem(0, KB8), smem(KB8, KB8), smem(2 * KB8, KB8));
    let (da, db) = (n.dma(0, 0, KB8, KB8), n.dma(KB8, KB8, KB8, KB8));
    let mm = n.matmul(c, vec![v1(a), v1(b)], 64, 64);
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(da).await;
        ctx.invoke(db).await;
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
        assert_eq!(ctx.now(), 920);
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    assert_eq!(s.end_time, 920);

    // Hand computation. Scalar: invoke A [0,8), invoke B [8,16),
    // invoke C [16,24), wait [24,26).
    let (ta, tb, tc) = (&s.tasks[0], &s.tasks[1], &s.tasks[2]);
    // DMA A: issue 8, the dram block's latency 100 (the pipeline fill,
    // r2sim spec Section 5.9) and the engine latency 100; hops in route
    // order from 208: dram port (64 B/cycle) [208, 336), link (32)
    // [208, 464), smem port narrowed to the link's 32 [208, 464); culprit
    // link (the first longest window).
    assert_eq!((ta.invoke, ta.issue), (8, Some(8)));
    assert_eq!((ta.chunks[0].start, ta.chunks[0].end), (208, 464));
    assert_eq!(ta.chunks[0].culprit, Some(n.link));
    // c_complete [464, 468).
    assert_eq!(ta.complete, Some(468));
    // DMA B: issue 16, earliest 216: dram port busy until 336, [336, 464);
    // the link is busy until 464, [464, 720), and the smem port [464, 720).
    // The chunk starts at its first hop.
    assert_eq!(tb.issue, Some(16));
    assert_eq!((tb.chunks[0].start, tb.chunks[0].end), (336, 720));
    assert_eq!(tb.complete, Some(724));
    // Matmul: ready at B's completion. 64^3 MAC / 4096 = 64 cycles on the
    // array. On smem: operands (64*64 + 64*64) * 2 B plus the C writeback
    // 64*64 * 2 B = 24576 B / 128 = 192 cycles: culprit smem.
    assert_eq!((tc.invoke, tc.ready, tc.issue), (24, Some(724), Some(724)));
    assert_eq!((tc.chunks[0].start, tc.chunks[0].end), (724, 916));
    assert_eq!(tc.chunks[0].culprit, Some(n.smem_port));
    // c_complete [916, 920); the wait wakes at 920.
    assert_eq!(tc.complete, Some(920));
    // Versioned tasks retire at their own completion.
    let retires: Vec<Option<u64>> = s.tasks.iter().map(|t| t.retire).collect();
    assert_eq!(retires, vec![Some(468), Some(724), Some(920)]);
    assert_eq!(s.func_ops, 3);
    assert_eq!(s.resources[n.link.0 as usize].busy_cycles, 512);
}

fn two_dmas(k: Knobs, bytes: u64, tile: u64) -> (Stats, sim_core::ResourceId) {
    let n = node(k);
    let (da, db) = (n.dma(0, 0, bytes, tile), n.dma(bytes, bytes, bytes, tile));
    let (a, b) = (smem(0, bytes), smem(bytes, bytes));
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(da).await;
        ctx.invoke(db).await;
        ctx.wait(a.base, 1).await;
        ctx.wait(b.base, 1).await;
    });
    (sim.run(), n.link)
}

#[test]
fn two_dmas_on_one_link_interleave_by_chunk() {
    // 4 chunks of 2 KiB each, 64 cycles on the link, no engine latency;
    // each transfer waits the dram block's latency (100) once.
    let k = Knobs {
        dma_latency: 0,
        ..Knobs::default()
    };
    let (s, _) = two_dmas(k, KB8, 2048);
    let ends = |i: usize| -> Vec<u64> { s.tasks[i].chunks.iter().map(|c| c.end).collect() };
    // A issues at 8 and starts at 108, B at 16 and 116. They alternate on
    // the link.
    assert_eq!(ends(0), vec![172, 300, 428, 556]);
    assert_eq!(ends(1), vec![236, 364, 492, 620]);
    // FIFO (one chunk per DMA) would end A at 108 + 256 = 364.
    let (s, _) = two_dmas(
        Knobs {
            dma_latency: 0,
            ..Knobs::default()
        },
        KB8,
        KB8,
    );
    assert_eq!(s.tasks[0].chunks[0].end, 364);
    assert_eq!(s.tasks[1].chunks[0].end, 620);
}

fn one_dma_last_end(outstanding: u32) -> u64 {
    let n = node(Knobs {
        dma_outstanding: outstanding,
        ..Knobs::default()
    });
    let d = n.dma(0, 0, KB8, 2048);
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(d).await;
    });
    let s = sim.run();
    s.tasks[0].chunks.iter().map(|c| c.end).max().unwrap()
}

#[test]
fn max_outstanding_overlaps_latency() {
    // 4 chunks, 64 cycles each on the link, engine latency 100, issue at
    // 8; the dram block's latency (100) once, before the first chunk.
    // N = 1: every chunk pays the engine latency in series:
    // 8 + 100 + 4 * (100 + 64).
    assert_eq!(one_dma_last_end(1), 108 + 4 * 164);
    // N = 2: chunks 0 and 1 at 208 and 272; chunk 2 is requested at 272
    // and starts at 372, chunk 3 at 436.
    assert_eq!(one_dma_last_end(2), 500);
    // N = 4: all requested at 8, serialized from 208 at the link rate.
    assert_eq!(one_dma_last_end(4), 208 + 4 * 64);
}

#[test]
fn full_scoreboard_stalls_invoke_until_issue() {
    let n = node(Knobs {
        scoreboard: 1,
        dma_latency: 0,
        ..Knobs::default()
    });
    let (a, c) = (smem(0, KB8), smem(2 * KB8, KB8));
    let da = n.dma(0, 0, KB8, KB8);
    let mm = n.matmul(c, vec![v1(a)], 64, 64);
    let db = n.dma(KB8, KB8, KB8, KB8);
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(da).await;
        ctx.invoke(mm).await;
        // The scoreboard holds the matmul: this invoke stalls.
        ctx.invoke(db).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let (ta, tc, tb) = (&s.tasks[0], &s.tasks[1], &s.tasks[2]);
    // A: the dram latency 100, [108, 364), c_complete [364, 368). The
    // matmul issues at 368 and frees the scoreboard slot; the stalled
    // invoke enters at the same time.
    assert_eq!(ta.complete, Some(368));
    assert_eq!(tc.issue, Some(368));
    assert_eq!(tb.invoke, 368);
    assert!(s.unfinished.is_empty());
}

#[test]
fn join_fans_in() {
    let n = node(Knobs {
        dma_latency: 0,
        ..Knobs::default()
    });
    let (a, b, c) = (smem(0, KB8), smem(KB8, KB8), smem(2 * KB8, KB8));
    let j = sync(0);
    let (da, db) = (n.dma(0, 0, KB8, KB8), n.dma(KB8, KB8, KB8, KB8));
    let join = sim_core::Descriptor::join(j, vec![v1(a), v1(b)]);
    let mut mm = n.matmul(c, vec![], 64, 64);
    mm.deps = vec![Dep::Unversioned(j.base)];
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(da).await;
        ctx.invoke(db).await;
        ctx.invoke(join).await;
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let (tb, tj, tc) = (&s.tasks[1], &s.tasks[2], &s.tasks[3]);
    // The join issues at B's completion and completes after c_complete.
    assert_eq!(tj.issue, tb.complete);
    assert_eq!(tj.complete, Some(tb.complete.unwrap() + 4));
    assert!(tj.chunks.is_empty());
    assert_eq!(tc.ready, tj.complete);
    assert_eq!(s.func_ops, 3, "the join emits no functional op");
}

#[test]
fn same_base_different_bound_is_fatal() {
    // Two live versioned ranges at one base address with different bounds
    // are a software error (Neutrino spec Section 7.2, Rule N7.3).
    let n = node(Knobs::default());
    let (d1, d2) = (n.dma(0, 0, KB8, KB8), n.dma(0, 0, 4096, 4096));
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(d1).await;
        ctx.invoke(d2).await;
        unreachable!("the run stops at the second invoke");
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    // invoke [0, 8), invoke [8, 16): the error at 16.
    assert_eq!((f.time, f.nu), (16, n.nu));
    assert_eq!(
        f.error,
        FatalError::Dep(DepError::BaseConflict {
            base: smem(0, 1).base,
            bound: smem(0, KB8).bound,
            new_bound: smem(0, 4096).bound,
            mask: nu_model::FULL_MASK,
            new_mask: nu_model::FULL_MASK,
        })
    );
}

#[test]
fn wait_on_a_range_nobody_produces_is_reported() {
    let n = node(Knobs::default());
    let x = smem(0, 512);
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.wait_unversioned(x.base).await;
    });
    let s = sim.run();
    assert_eq!(s.unfinished, vec![(n.nu, Blocked::Wait)]);
    // A run that ends with a program that has not returned is a deadlock;
    // the report names the wait and its key.
    let f = s.fatal.as_ref().expect("a deadlock");
    let FatalError::Deadlock { programs, .. } = &f.error else {
        panic!("{f:?}");
    };
    assert_eq!(
        programs,
        &vec![BlockedProgram {
            nu: n.nu,
            blocked: Blocked::Wait,
            call: "wait",
            key: Some(Dep::Unversioned(x.base)),
            owner: Some(n.nu),
        }]
    );
    assert_eq!(s.open_pending.len(), 1);
    assert_eq!(s.open_pending[0].pending.addr, x.base);
    assert_eq!(s.open_pending[0].last_writer, None, "nobody wrote X");
}

#[test]
fn unordered_writers_warn() {
    let n = node(Knobs::default());
    // Two unversioned writers of one range with no order between them.
    let (mut d1, mut d2) = (n.dma(0, 0, KB8, KB8), n.dma(0, 0, 4096, 4096));
    (d1.versioned, d2.versioned) = (false, false);
    let mut mm = n.matmul(smem(2 * KB8, KB8), vec![], 64, 64);
    mm.deps = vec![Dep::Unversioned(smem(0, 1024).base)];
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(d1).await;
        ctx.invoke(d2).await;
        let h = ctx.invoke(mm).await;
        ctx.wait(h.addr, h.version).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert_eq!(s.warnings.len(), 1);
    assert_eq!(s.warnings[0].gid, Some(2));
    assert!(matches!(
        s.warnings[0].warning,
        SimWarning::Dep(Warning::MultipleCovers { count: 2, .. })
    ));
}

#[test]
fn an_unversioned_signal_is_found_after_its_producer_completed() {
    let n = node(Knobs::default());
    let (j, c) = (sync(0), smem(2 * KB8, KB8));
    let join = sim_core::Descriptor::join(j, vec![]);
    let mut mm = n.matmul(c, vec![], 64, 64);
    mm.deps = vec![Dep::Unversioned(j.base)];
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(join).await;
        ctx.wait_unversioned(j.base).await;
        // J is done and unversioned: its entry stays for this consumer.
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    assert!(s.open_pending.is_empty());
    // Join: invoke [0, 8), issue 8, c_complete [8, 12). wait(J) [8, 10),
    // woken at 12. Matmul: invoke [12, 20), ready at 20 (J done); SMEM
    // 24576 B / 128 = 192 cycles [20, 212), c_complete [212, 216).
    // wait(C) [20, 22), woken at 216; the program returns at 216.
    let (tj, tc) = (&s.tasks[0], &s.tasks[1]);
    assert_eq!(tj.complete, Some(12));
    assert_eq!((tc.invoke, tc.ready), (20, Some(20)));
    assert_eq!(tc.complete, Some(216));
    // The matmul (versioned) retires at completion; the join at the
    // program return.
    assert_eq!((tc.retire, tj.retire), (Some(216), Some(216)));
}

#[test]
fn vreset_all_clears_the_callers_table() {
    let n = node(Knobs {
        dma_latency: 0,
        ..Knobs::default()
    });
    let (a, b) = (smem(0, KB8), smem(KB8, KB8));
    let (da, db) = (n.dma(0, 0, KB8, KB8), n.dma(0, KB8, KB8, KB8));
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.version(a.base, 41).await;
        ctx.version(b.base, 7).await;
        ctx.vreset_all().await;
        let h = ctx.invoke(da).await;
        assert_eq!(h.version, 1, "the counter of A was cleared");
        let h2 = ctx.invoke(db).await;
        assert_eq!(h2.version, 1);
        // version [0, 1), [1, 2); vreset_all (c_version) [2, 3); the
        // invokes [3, 11) and [11, 19).
        assert_eq!(ctx.now(), 19);
        ctx.wait(h.addr, h.version).await;
        ctx.wait(h2.addr, h2.version).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    assert_eq!(s.version_peak, vec![2]);
}

#[test]
fn version_and_vreset_reset_the_counter() {
    let n = node(Knobs {
        dma_latency: 0,
        ..Knobs::default()
    });
    let a = smem(0, KB8);
    let (d1, d2, d3) = (
        n.dma(0, 0, KB8, KB8),
        n.dma(0, 0, KB8, KB8),
        n.dma(0, 0, KB8, KB8),
    );
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.version(a.base, 41).await;
        let h = ctx.invoke(d1).await;
        assert_eq!(h.version, 42);
        ctx.wait(h.addr, h.version).await;
        ctx.vreset(a.base).await;
        let h = ctx.invoke(d2).await;
        assert_eq!(h.version, 1, "a released range starts again");
        ctx.wait(h.addr, h.version).await;
        let h = ctx.invoke(d3).await;
        assert_eq!(h.version, 2);
        ctx.wait(h.addr, h.version).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    assert_eq!(s.version_peak, vec![1]);
}

#[test]
fn a_full_version_table_is_fatal_for_a_local_invoke() {
    // Version capacity 16: the 17th versioned range stops the run.
    let n = node(Knobs::default());
    let ds: Vec<_> = (0..17).map(|i| n.dma(0, i * 512, 512, 512)).collect();
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        for d in ds {
            ctx.invoke(d).await;
        }
        unreachable!("the 17th range has no version table entry");
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    assert_eq!(f.error, FatalError::Dep(DepError::VersionsFull));
    // 17 invokes of 8 cycles; the 17th waits for a scoreboard entry (16):
    // the first DMA issues at 8, waits the dram latency and the engine
    // latency (208), moves 512 B on the 32 B/cycle link [208, 224) and
    // completes at 228, when the 17th enters and finds no version entry.
    assert_eq!(f.time, 228);
    assert_eq!(s.version_peak, vec![16]);
}

#[test]
fn program_return_keeps_the_version_table() {
    // The version table resets at a program start (hierarchy.rs), not at a
    // return.
    let n = node(Knobs::default());
    let da = n.dma(0, 0, KB8, KB8);
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        let h = ctx.invoke(da).await;
        ctx.wait(h.addr, h.version).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert_eq!(s.version_peak, vec![1]);
    assert_eq!(s.tables_at_end[0].versions, 1, "kept at the return");
}

#[test]
fn full_pending_is_fatal_for_a_local_caller() {
    let n = node(Knobs {
        pending: 1,
        ..Knobs::default()
    });
    // Two consumers of ranges that nobody has invoked yet.
    let c1 = n.matmul(smem(2 * KB8, KB8), vec![v1(smem(0, 512))], 64, 64);
    let c2 = n.matmul(smem(3 * KB8, KB8), vec![v1(smem(512, 512))], 64, 64);
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(c1).await;
        ctx.invoke(c2).await;
        unreachable!("the second invoke needs a second pending table slot");
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    assert_eq!((f.time, f.nu), (16, n.nu));
    assert_eq!(f.error, FatalError::Dep(DepError::PendingFull));
}

#[test]
fn timeline_can_be_turned_off() {
    let n = node(Knobs::default());
    let (a, c) = (smem(0, KB8), smem(2 * KB8, KB8));
    let da = n.dma(0, 0, KB8, KB8);
    let mm = n.matmul(c, vec![v1(a)], 64, 64);
    let mut sim = n.sim;
    sim.set_timeline(false);
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(da).await;
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.tasks.is_empty());
    assert_eq!(s.invoked, 2);
}

fn range_dep_on_versioned_data(check: bool) -> Stats {
    // A consumer that names versioned data by its range (without a
    // version): the unversioned lookup sees no producer, so it waits on a
    // pending entry over a range that a completed task wrote.
    let n = node(Knobs::default());
    let (a, c) = (smem(0, KB8), smem(2 * KB8, KB8));
    let da = n.dma(0, 0, KB8, KB8);
    let mut mm = n.matmul(c, vec![], 64, 64);
    mm.deps = vec![Dep::Unversioned(a.base)];
    let mut sim = n.sim;
    sim.set_zero_time(true);
    sim.set_late_consumer_check(check);
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        let h = ctx.invoke(da).await;
        ctx.wait(h.addr, h.version).await;
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
    });
    sim.run()
}

#[test]
fn late_consumer_is_diagnosed() {
    let s = range_dep_on_versioned_data(true);
    assert_eq!(s.unfinished, vec![(s.tasks[0].nu, Blocked::Wait)]);
    assert!(matches!(
        s.fatal.as_ref().map(|f| &f.error),
        Some(FatalError::Deadlock { .. })
    ));
    // The end-of-run report names the completed writer.
    // (The wait on (C, 1) waits on the matmul itself, not a pending entry.)
    assert_eq!(s.open_pending.len(), 1);
    assert_eq!(s.open_pending[0].pending.addr, smem(0, KB8).base);
    assert_eq!(s.open_pending[0].last_writer, Some(Some(0)));
    let late: Vec<_> = s
        .warnings
        .iter()
        .filter(|w| matches!(w.warning, SimWarning::LateConsumer { .. }))
        .collect();
    assert_eq!(late.len(), 1);
    assert_eq!(late[0].gid, Some(1), "the late consumer");
    assert_eq!(
        late[0].warning,
        SimWarning::LateConsumer {
            range: Range::from_len(smem(0, KB8).base, 1),
            writer: Some(0)
        }
    );
    // The unversioned key names the base of a version table entry.
    let unversioned: Vec<_> = s
        .warnings
        .iter()
        .filter(|w| matches!(w.warning, SimWarning::Dep(_)))
        .map(|w| w.warning)
        .collect();
    assert_eq!(
        unversioned,
        vec![SimWarning::Dep(Warning::UnversionedWait {
            addr: smem(0, KB8).base,
            versioned: v1(smem(0, KB8))
        })]
    );
    // Off (the default): no LateConsumer warning; the deadlock, the report
    // and the UnversionedWait warning stay.
    let s = range_dep_on_versioned_data(false);
    assert_eq!(s.warnings.len(), 1);
    assert_eq!(s.unfinished.len(), 1);
    assert!(s.fatal.is_some());
}

#[test]
fn a_versioned_consumer_after_its_producer_completed_resolves() {
    // The zero-time producer-first program that deadlocked under the
    // invocation-order rule: the producers retire at completion, and the
    // consumer names (A, 1) and (B, 1), which the version table resolves.
    let n = node(Knobs::default());
    let (a, b, c) = (smem(0, KB8), smem(KB8, KB8), smem(2 * KB8, KB8));
    let (da, db) = (n.dma(0, 0, KB8, KB8), n.dma(KB8, KB8, KB8, KB8));
    let mm = n.matmul(c, vec![v1(a), v1(b)], 64, 64);
    let mut sim = n.sim;
    sim.set_zero_time(true);
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(da).await;
        ctx.invoke(db).await;
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert!(s.unfinished.is_empty());
    assert!(s.open_pending.is_empty());
}

#[test]
fn consumer_first_raises_no_late_consumer_warning() {
    let n = node(Knobs::default());
    let (a, c) = (smem(0, KB8), smem(2 * KB8, KB8));
    let da = n.dma(0, 0, KB8, KB8);
    let mm = n.matmul(c, vec![v1(a)], 64, 64);
    let mut sim = n.sim;
    sim.set_zero_time(true);
    sim.set_late_consumer_check(true);
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(mm).await;
        ctx.invoke(da).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert!(s.unfinished.is_empty());
    assert!(s.warnings.is_empty());
}

#[test]
fn quantized_b_operand_scales_smem_bytes_not_macs() {
    // 64^3 MAC / 4096 = 64 cycles on the array. SMEM: A 64*64*2 = 8192 B,
    // B mxfp4 4096 / 2 + 4096 / 32 = 2176 B, C 8192 B: 18560 B / 128 = 145
    // cycles, culprit smem.
    let n = node(Knobs::default());
    let c = smem(0, KB8);
    let mut mm = n.matmul(c, vec![], 64, 64);
    if let sim_core::Args::Matmul { b_dtype, .. } = &mut mm.args {
        *b_dtype = sim_core::Dtype::Mxfp4;
    }
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let ch = &s.tasks[0].chunks[0];
    assert_eq!(ch.end - ch.start, 145);
    assert_eq!(ch.culprit, Some(n.smem_port));
}

#[test]
fn row_panel_reads_a_once_per_row() {
    // m 64, n 128, k 64, tile 64: two output tiles in one row. SMEM per
    // tile without reuse: A 8192 + B 8192 + C 8192 B = 192 cycles at
    // 128 B/cycle; with a row panel the second tile skips A: 128 cycles.
    let n = node(Knobs::default());
    let c = smem(0, 2 * KB8);
    let mut mm = n.matmul(c, vec![], 64, 64);
    if let sim_core::Args::Matmul {
        n: nn, ldb, ldc, ..
    } = &mut mm.args
    {
        *nn = 128;
        *ldb = 128;
        *ldc = 128;
    }
    let run = |panel: bool| {
        let n = node(Knobs::default());
        let mut sim = n.sim;
        sim.set_atlas_row_panel(panel);
        let mm = mm.clone();
        sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
            ctx.invoke(mm).await;
            ctx.wait(c.base, 1).await;
        });
        let s = sim.run();
        assert_eq!(s.fatal, None);
        s.tasks[0]
            .chunks
            .iter()
            .map(|ch| ch.end - ch.start)
            .collect::<Vec<_>>()
    };
    assert_eq!(run(false), vec![192, 192]);
    assert_eq!(run(true), vec![192, 128]);
}

#[test]
fn an_accumulator_buffer_keeps_c_across_a_chain() {
    // Two chained 64^3 tasks on one 64 x 64 C (8 KiB). Without a buffer:
    // A 8192 + B 8192 + C out 8192 (+ C in 8192 for the second) B at
    // 128 B/cycle = 192 and 256 cycles. With a buffer of 8 KiB, the first
    // (keep) skips C out and the second skips C in: 128 and 192 cycles.
    let run = |buffer: u64| {
        let n = node(Knobs::default());
        let c = smem(0, KB8);
        let mut first = n.matmul(c, vec![], 64, 64);
        if let sim_core::Args::Matmul { keep, .. } = &mut first.args {
            *keep = true;
        }
        let mut second = n.matmul(c, vec![v1(c)], 64, 64);
        if let sim_core::Args::Matmul { accumulate, .. } = &mut second.args {
            *accumulate = true;
        }
        let mut sim = n.sim;
        sim.set_atlas_acc_buffer(buffer);
        sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
            ctx.invoke(first).await;
            ctx.invoke(second).await;
            ctx.wait(c.base, 2).await;
        });
        let s = sim.run();
        assert_eq!(s.fatal, None);
        s.tasks
            .iter()
            .map(|t| t.chunks[0].end - t.chunks[0].start)
            .collect::<Vec<_>>()
    };
    assert_eq!(run(0), vec![192, 256]);
    assert_eq!(run(KB8), vec![128, 192]);
}

#[test]
fn resolution_latency_delays_ready_and_wait_wake() {
    // As matmul_end_to_end with c_resolve = 100: the matmul's last dep
    // (DMA B) resolves at 724, the matmul is ready at 824 and ends at 1020;
    // the program's pending wait on C wakes 100 later, at 1120. The DMAs
    // have no deps and issue at once.
    let n = node(Knobs {
        resolve: 100,
        ..Knobs::default()
    });
    let (a, b, c) = (smem(0, KB8), smem(KB8, KB8), smem(2 * KB8, KB8));
    let (da, db) = (n.dma(0, 0, KB8, KB8), n.dma(KB8, KB8, KB8, KB8));
    let mm = n.matmul(c, vec![v1(a), v1(b)], 64, 64);
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(da).await;
        ctx.invoke(db).await;
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
        assert_eq!(ctx.now(), 1120);
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let tc = &s.tasks[2];
    assert_eq!((tc.ready, tc.complete), (Some(824), Some(1020)));
    assert_eq!(s.end_time, 1120);
}

#[test]
fn zero_time_mode_ignores_the_resolution_latency() {
    // As resolution_latency_delays_ready_and_wait_wake in zero-time mode,
    // with DMAs of many chunks so that the wait on C is pending when C
    // completes: neither the ready nor the wait wake takes time.
    let n = node(Knobs {
        resolve: 100,
        ..Knobs::default()
    });
    let (a, b, c) = (smem(0, KB8), smem(KB8, KB8), smem(2 * KB8, KB8));
    let (da, db) = (n.dma(0, 0, KB8, 64), n.dma(KB8, KB8, KB8, 64));
    let mm = n.matmul(c, vec![v1(a), v1(b)], 64, 64);
    let mut sim = n.sim;
    sim.set_zero_time(true);
    sim.spawn(n.nu, move |ctx: sim_core::NuCtx| async move {
        ctx.invoke(da).await;
        ctx.invoke(db).await;
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert_eq!(s.end_time, 0);
}
