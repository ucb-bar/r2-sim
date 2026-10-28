use nu_model::{
    epoch_order, handle_await, version_order, AwaitMsg, Dep, DepConfig, DepError, DepTables,
    NeutrinoId, Phase, Range, RequestId, TaskId, TaskSpec, WaitStatus, Warning, COUNT_MASK,
    EPOCH_SHIFT,
};
use std::cmp::Ordering::{Equal, Greater, Less};

const G: u64 = 512;

fn cfg(scoreboard: usize, execution: usize, pending: usize) -> DepConfig {
    DepConfig {
        scoreboard_capacity: scoreboard,
        execution_capacity: execution,
        pending_capacity: pending,
        version_capacity: 4,
        inbound_ref_capacity: 4,
        issue_headroom: 0,
        granule: G,
    }
}

fn tables() -> DepTables {
    DepTables::new(cfg(16, 16, 4))
}

fn r(base_g: u64, len_g: u64) -> Range {
    Range::from_len(base_g * G, len_g * G)
}

/// The address of granule `g`.
fn at(g: u64) -> u64 {
    g * G
}

/// An unversioned task (a signal) whose deps are addresses.
fn t(dst: Range, deps: &[u64]) -> TaskSpec {
    TaskSpec::new(dst, deps.iter().map(|a| Dep::Unversioned(*a)).collect())
}

/// A versioned task; the tables assign the version.
fn tv(dst: Range, deps: &[Dep]) -> TaskSpec {
    TaskSpec::versioned(dst, deps.to_vec())
}

/// Version `version` of the range at `range`'s base.
fn key(range: Range, version: u64) -> Dep {
    Dep::Versioned {
        addr: range.base,
        version,
    }
}

/// Invokes a task with no deps and issues it.
fn run(tb: &mut DepTables, spec: TaskSpec) -> TaskId {
    let inv = tb.invoke(spec, 0).unwrap();
    assert!(inv.ready);
    tb.issue(inv.task).unwrap();
    inv.task
}

// ---- unversioned addresses (signals) ----

#[test]
fn unversioned_raw_chain_keeps_done_entries() {
    // Two producers write A and B; the consumer reads both and writes C.
    // Unversioned done entries stay in the execution table (Neutrino spec Section 6.3)
    // until a new producer of the range or the program return.
    let mut tb = tables();
    let (a, b, c) = (r(0, 4), r(4, 4), r(8, 4));
    let pa = run(&mut tb, t(a, &[]));
    let pb = run(&mut tb, t(b, &[]));
    let cc = tb.invoke(t(c, &[a.base, b.base]), 0).unwrap();
    assert!(!cc.ready);
    assert_eq!(tb.depcount(cc.task), Some(2));

    let fx = tb.complete(pa);
    assert!(fx.ready.is_empty());
    assert!(fx.retired.is_empty(), "done unversioned entry stays");
    assert_eq!(tb.phase(pa), Some(Phase::Done));
    assert_eq!(tb.complete(pb).ready, vec![cc.task]);

    tb.issue(cc.task).unwrap();
    tb.complete(cc.task);
    assert_eq!(
        (tb.scoreboard_len(), tb.execution_len(), tb.pending_len()),
        (0, 3, 0)
    );
    let mut retired = tb.program_returned().retired;
    retired.sort();
    assert_eq!(retired, vec![pa, pb, cc.task]);
    assert_eq!(tb.execution_len(), 0);
}

#[test]
fn unversioned_signal_is_found_after_its_producer_completed() {
    let mut tb = tables();
    let s = r(0, 1);
    let p = run(&mut tb, t(s, &[]));
    tb.complete(p);
    let late = tb.invoke(t(r(8, 1), &[s.base]), 0).unwrap();
    assert!(late.ready, "the done producer resolves the late consumer");
    assert_eq!(
        tb.wait(Dep::Unversioned(s.base)).unwrap().0,
        WaitStatus::Done
    );
    assert_eq!(tb.pending_len(), 0);
}

#[test]
fn a_new_producer_supersedes_a_done_unversioned_entry() {
    let mut tb = tables();
    let s = r(0, 1);
    let p1 = run(&mut tb, t(s, &[]));
    tb.complete(p1);
    // The next producer of the recycled signal retires the old entry.
    let p2 = tb.invoke(t(s, &[]), 0).unwrap();
    assert_eq!(p2.effects.retired, vec![p1]);
    // A consumer now binds to the new producer.
    let c = tb.invoke(t(r(8, 1), &[s.base]), 0).unwrap();
    assert_eq!(tb.depcount(c.task), Some(1));
    tb.issue(p2.task).unwrap();
    assert_eq!(tb.complete(p2.task).ready, vec![c.task]);
}

#[test]
fn a_superseded_executing_entry_retires_at_completion() {
    let mut tb = tables();
    let s = r(0, 1);
    let p1 = run(&mut tb, t(s, &[]));
    let p2 = tb.invoke(t(s, &[]), 0).unwrap();
    assert!(p2.effects.retired.is_empty(), "p1 still executes");
    assert_eq!(tb.complete(p1).retired, vec![p1]);
}

#[test]
fn an_address_inside_a_producer_resolves() {
    let mut tb = tables();
    let p = run(&mut tb, t(r(0, 8), &[]));
    let c = tb.invoke(t(r(20, 1), &[at(3)]), 0).unwrap();
    assert_eq!(tb.depcount(c.task), Some(1));
    assert_eq!(tb.complete(p).ready, vec![c.task]);
    // One past the end is not inside: it waits on a pending entry.
    let adj = tb.invoke(t(r(21, 1), &[at(8)]), 0).unwrap();
    assert!(!adj.ready);
    assert_eq!(tb.pending_len(), 1);
}

#[test]
fn multiple_executing_covers_count_all_and_warn() {
    let mut tb = tables();
    let outer = run(&mut tb, t(r(0, 8), &[]));
    let inner = run(&mut tb, t(r(0, 4), &[]));
    let c = tb.invoke(t(r(20, 1), &[at(1)]), 0).unwrap();
    assert_eq!(tb.depcount(c.task), Some(2));
    assert_eq!(
        c.effects.warnings,
        vec![Warning::MultipleCovers {
            addr: at(1),
            count: 2
        }]
    );
    assert!(tb.complete(inner).ready.is_empty());
    assert_eq!(tb.complete(outer).ready, vec![c.task]);
}

#[test]
fn capacities() {
    let mut tb = DepTables::new(cfg(2, 4, 1));
    let a = tb.invoke(tv(r(0, 1), &[]), 0).unwrap().task;
    let b = tb.invoke(tv(r(1, 1), &[]), 0).unwrap().task;
    assert_eq!(
        tb.invoke(tv(r(2, 1), &[]), 0),
        Err(DepError::ScoreboardFull)
    );
    tb.issue(a).unwrap();
    assert_eq!(
        tb.complete(a).retired,
        vec![a],
        "versioned: retires at once"
    );
    tb.issue(b).unwrap();
    tb.complete(b);
    assert_eq!(tb.execution_len(), 0);

    // Issue headroom (Neutrino spec Section 6.2): scoreboard plus execution table entries
    // stay below the execution capacity (4) less the headroom (0 here).
    // Three done signals hold three execution entries.
    for i in 0..3 {
        let p = run(&mut tb, t(r(20 + i, 1), &[]));
        tb.complete(p);
    }
    assert_eq!(tb.execution_len(), 3);
    let c = tb.invoke(t(r(30, 1), &[]), 0).unwrap().task;
    assert_eq!(tb.invoke(t(r(31, 1), &[]), 0), Err(DepError::NoHeadroom));
    tb.issue(c).unwrap();
    // The program return frees the done signals.
    assert_eq!(tb.program_returned().retired.len(), 3);
    assert!(tb.invoke(t(r(31, 1), &[]), 0).is_ok());

    // Pending table capacity: one unresolved dep fits, a second one does
    // not.
    let mut tb = DepTables::new(cfg(16, 16, 1));
    tb.invoke(t(r(10, 1), &[at(5)]), 0).unwrap();
    assert_eq!(
        tb.invoke(t(r(11, 1), &[at(6)]), 0),
        Err(DepError::PendingFull)
    );
    assert!(matches!(
        tb.wait(Dep::Unversioned(at(8))),
        Err(DepError::PendingFull)
    ));
    // A dep on the open pending address needs no new slot.
    tb.invoke(t(r(12, 1), &[at(5)]), 0).unwrap();
    let bad = Range::new(0x200, 0x300);
    assert_eq!(tb.invoke(t(bad, &[]), 0), Err(DepError::Misaligned(bad)));
}

#[test]
fn consumer_invoked_before_producer_waits_on_pending_and_is_adopted() {
    let mut tb = tables();
    let c = tb.invoke(t(r(8, 4), &[at(2)]), 0).unwrap();
    assert!(!c.ready);
    assert_eq!(tb.pending_len(), 1);
    let (ws, _) = tb.wait(Dep::Unversioned(at(2))).unwrap();
    let WaitStatus::Pending(wid) = ws else {
        panic!("a pending entry is pending");
    };
    // The producer's range contains the address: it adopts the entry.
    let p = tb.invoke(t(r(0, 8), &[]), 0).unwrap();
    assert!(p.ready);
    assert_eq!(p.effects.adopted.len(), 1);
    assert_eq!(tb.pending_len(), 0);
    tb.issue(p.task).unwrap();
    let fx = tb.complete(p.task);
    assert_eq!(fx.ready, vec![c.task]);
    assert_eq!(fx.waits_done, vec![wid]);
}

#[test]
fn self_dependency_is_an_error() {
    let mut tb = tables();
    // No producer of the address: the dep would create a pending entry
    // that the task itself adopts.
    assert!(matches!(
        tb.invoke(t(r(0, 8), &[at(2)]), 0),
        Err(DepError::SelfDependency { .. })
    ));
    assert_eq!(tb.pending_len(), 0);
    // A versioned task that names its own version.
    assert!(matches!(
        tb.invoke(tv(r(0, 4), &[key(r(0, 4), 1)]), 0),
        Err(DepError::SelfDependency { .. })
    ));
    // Reading the previous version of the own range is fine (accumulation).
    let p = run(&mut tb, tv(r(0, 4), &[]));
    let acc = tb.invoke(tv(r(0, 4), &[key(r(0, 4), 1)]), 0).unwrap();
    assert_eq!(acc.version, Some(2));
    assert_eq!(tb.depcount(acc.task), Some(1));
    assert_eq!(tb.complete(p).ready, vec![acc.task]);
}

#[test]
fn join_task_fans_in() {
    // Sync-only range J: the join depends on three versioned producers; the
    // consumer depends on J.
    let mut tb = tables();
    let j = Range::from_len(0x1000_0000, G);
    let ps: Vec<TaskId> = (0..3).map(|i| run(&mut tb, tv(r(i * 4, 4), &[]))).collect();
    let deps: Vec<Dep> = (0..3).map(|i| key(r(i * 4, 4), 1)).collect();
    let join = tb.invoke(TaskSpec::new(j, deps), 0).unwrap();
    let c = tb.invoke(t(r(20, 1), &[j.base]), 0).unwrap();
    assert_eq!(tb.depcount(join.task), Some(3));
    assert_eq!(tb.depcount(c.task), Some(1));
    for &p in &ps[..2] {
        let fx = tb.complete(p);
        assert!(fx.ready.is_empty());
        assert_eq!(fx.retired, vec![p]);
    }
    assert_eq!(tb.complete(ps[2]).ready, vec![join.task]);
    tb.issue(join.task).unwrap();
    let fx = tb.complete(join.task);
    assert_eq!(fx.ready, vec![c.task]);
    assert!(fx.retired.is_empty(), "the join is unversioned: it stays");
}

#[test]
fn wait_counts_every_executing_cover() {
    let mut tb = tables();
    let outer = run(&mut tb, t(r(0, 8), &[]));
    let inner = run(&mut tb, t(r(0, 4), &[]));
    let (ws, fx) = tb.wait(Dep::Unversioned(at(1))).unwrap();
    let WaitStatus::Pending(wid) = ws else {
        panic!("two pending covers");
    };
    assert_eq!(fx.warnings.len(), 1);
    assert!(tb.complete(inner).waits_done.is_empty());
    assert_eq!(tb.complete(outer).waits_done, vec![wid]);
    assert_eq!(tb.pending_waits(), 0);
}

#[test]
fn external_dependencies_count_and_resolve() {
    let mut tb = tables();
    let local = run(&mut tb, t(r(0, 1), &[]));
    let c = tb.invoke(t(r(4, 1), &[at(0)]), 2).unwrap();
    assert_eq!(tb.depcount(c.task), Some(3));
    assert!(!tb.external_resolved(c.task));
    assert!(tb.complete(local).ready.is_empty());
    assert!(tb.external_resolved(c.task));
}

#[test]
fn two_deps_on_one_new_pending_share_it() {
    let mut tb = tables();
    let c = tb.invoke(t(r(20, 1), &[at(1), at(1)]), 0).unwrap();
    assert_eq!(tb.pending_len(), 1);
    assert_eq!(tb.depcount(c.task), Some(2));
    let p = run(&mut tb, t(r(0, 4), &[]));
    assert_eq!(tb.pending_len(), 0, "P adopted the one entry");
    assert_eq!(tb.complete(p).ready, vec![c.task]);
}

// ---- versions ----

#[test]
fn versions_count_up_and_a_done_version_resolves() {
    let mut tb = tables();
    let a = r(0, 4);
    let p1 = tb.invoke(tv(a, &[]), 0).unwrap();
    assert_eq!(p1.version, Some(1));
    tb.issue(p1.task).unwrap();
    // A consumer of (a, 1) while it runs finds it in the execution table.
    let w = tb
        .invoke(TaskSpec::new(r(9, 1), vec![key(a, 1)]), 0)
        .unwrap();
    assert_eq!(tb.depcount(w.task), Some(1));
    // Completion retires the versioned producer at once.
    let fx = tb.complete(p1.task);
    assert_eq!(fx.ready, vec![w.task]);
    assert_eq!(fx.retired, vec![p1.task]);
    // The counter (1) is at least 1: done.
    let late = tb
        .invoke(TaskSpec::new(r(10, 1), vec![key(a, 1)]), 0)
        .unwrap();
    assert!(late.ready);
    assert_eq!(tb.version_counter(a.base), Some(1));
    let p2 = tb.invoke(tv(a, &[key(a, 1)]), 0).unwrap();
    assert_eq!(p2.version, Some(2));
    assert!(p2.ready, "version 1 is done");
}

#[test]
fn a_versioned_consumer_invoked_before_its_producer_binds_correctly() {
    // Version 1 of A is done. A consumer of (a, 2) is invoked before the
    // writer of version 2: it must not bind to the done version 1.
    let mut tb = tables();
    let a = r(0, 4);
    let p1 = run(&mut tb, tv(a, &[]));
    tb.complete(p1);
    let c = tb
        .invoke(TaskSpec::new(r(8, 1), vec![key(a, 2)]), 0)
        .unwrap();
    assert!(!c.ready, "counter 1 is below 2: a pending entry for (a, 2)");
    assert_eq!(tb.pending_len(), 1);
    let p2 = tb.invoke(tv(a, &[key(a, 1)]), 0).unwrap();
    assert_eq!(p2.version, Some(2));
    assert_eq!(p2.effects.adopted.len(), 1, "the writer of v2 adopts it");
    tb.issue(p2.task).unwrap();
    assert_eq!(tb.complete(p2.task).ready, vec![c.task]);
}

#[test]
fn unversioned_lookups_ignore_versioned_tasks() {
    let mut tb = tables();
    let a = r(0, 4);
    run(&mut tb, tv(a, &[]));
    // An unversioned dep on a versioned range sees no unversioned producer.
    let c = tb.invoke(t(r(8, 1), &[a.base]), 0).unwrap();
    assert!(!c.ready);
    assert_eq!(tb.pending_len(), 1);
}

#[test]
fn version_vreset_and_reset_all() {
    let mut tb = tables();
    let a = r(0, 4);
    tb.set_version(a.base, 10).unwrap();
    let p = tb.invoke(tv(a, &[]), 0).unwrap();
    assert_eq!(p.version, Some(11));
    // VersionReset while the producer is in flight: its consumers still find it
    // in the tables.
    tb.vreset(a.base);
    assert_eq!(tb.versions_len(), 0);
    let c = tb
        .invoke(TaskSpec::new(r(8, 1), vec![key(a, 11)]), 0)
        .unwrap();
    assert_eq!(tb.depcount(c.task), Some(1));
    tb.issue(p.task).unwrap();
    assert_eq!(tb.complete(p.task).ready, vec![c.task]);
    // A released range starts again at 1; an absent entry is a no-op.
    tb.vreset(a.base);
    assert_eq!(tb.invoke(tv(a, &[]), 0).unwrap().version, Some(1));
    tb.set_version(r(8, 4).base, 3).unwrap();
    assert_eq!(tb.versions_len(), 2);
    tb.vreset_all();
    assert_eq!(tb.versions_len(), 0);
}

#[test]
fn a_reused_version_in_flight_is_an_error() {
    // Released while version 1 is in flight, then invoked again: the new
    // task would also be (a, 1).
    let mut tb = tables();
    let a = r(0, 4);
    run(&mut tb, tv(a, &[]));
    tb.vreset(a.base);
    assert_eq!(
        tb.invoke(tv(a, &[]), 0).unwrap_err(),
        DepError::VersionInFlight {
            addr: a.base,
            version: 1
        }
    );
}

#[test]
fn same_base_different_bound_is_an_error() {
    let mut tb = tables();
    run(&mut tb, tv(r(0, 4), &[]));
    assert_eq!(
        tb.invoke(tv(r(0, 2), &[]), 0).unwrap_err(),
        DepError::BaseConflict {
            base: 0,
            bound: 4 * G,
            new_bound: 2 * G,
            mask: nu_model::FULL_MASK,
            new_mask: nu_model::FULL_MASK,
        }
    );
}

#[test]
fn a_full_version_table_refuses_a_new_range() {
    let mut tb = tables(); // version capacity 4
    for i in 0..4 {
        run(&mut tb, tv(r(i * 4, 4), &[]));
    }
    assert_eq!(tb.versions_len(), 4);
    assert_eq!(
        tb.invoke(tv(r(16, 4), &[]), 0).unwrap_err(),
        DepError::VersionsFull
    );
    assert_eq!(
        tb.set_version(r(16, 4).base, 1),
        Err(DepError::VersionsFull)
    );
    // A known range needs no new entry.
    assert!(tb.invoke(tv(r(0, 4), &[]), 0).is_ok());
    assert_eq!(tb.version_peak(), 4);
}

#[test]
fn program_return_keeps_the_version_table() {
    // The version table resets at a program start (Neutrino spec Section 7.7), not at a
    // return.
    let mut tb = tables();
    let a = r(0, 4);
    let p = run(&mut tb, tv(a, &[]));
    let s = run(&mut tb, t(r(8, 1), &[]));
    tb.complete(s);
    let fx = tb.program_returned();
    assert_eq!(fx.retired, vec![s]);
    assert_eq!(tb.versions_len(), 1);
    assert_eq!(tb.complete(p).retired, vec![p]);
    assert_eq!(tb.invoke(tv(a, &[]), 0).unwrap().version, Some(2));
}

// ---- masked destination ranges (2026-10-21) ----

/// An unversioned task on `dst` with mask `mask`.
fn tm(dst: Range, mask: u32, deps: &[u64]) -> TaskSpec {
    t(dst, deps).with_mask(mask)
}

#[test]
fn a_lookup_of_a_granule_in_a_gap_does_not_bind() {
    // U holds granules 0 and 2 of its span of 4 (mask 0b10). A wait on
    // granule 1 finds no producer (a new pending entry); a wait on
    // granule 2 waits for U.
    let mut tb = tables();
    let u = run(&mut tb, tm(r(0, 4), 0b10, &[]));
    let (gap, fx) = tb.wait(Dep::Unversioned(at(1) + 7)).unwrap();
    assert_eq!(fx.pending_created.len(), 1);
    let (held, _) = tb.wait(Dep::Unversioned(at(2))).unwrap();
    let (WaitStatus::Pending(gap), WaitStatus::Pending(held)) = (gap, held) else {
        panic!("both wait")
    };
    let fx = tb.complete(u);
    assert_eq!(fx.waits_done, vec![held]);
    assert!(!fx.waits_done.contains(&gap));
}

#[test]
fn the_warp_handles_of_the_spec_are_independent_chains() {
    // The warp-tile example: four handles of 128 granules with mask 0x7e
    // at c, c + 1, c + 128 and c + 129. Versioned: each v2 waits for its
    // own v1 only. Unversioned: a key of warp 1 binds warp 1 only.
    let mut tb = DepTables::new(cfg(16, 16, 8));
    let c = 1024;
    let h: Vec<Range> = [0, 1, 128, 129].iter().map(|o| r(c + o, 128)).collect();
    let v1: Vec<TaskId> = h
        .iter()
        .map(|d| run(&mut tb, tv(*d, &[]).with_mask(0x7e)))
        .collect();
    let v2: Vec<TaskId> = h
        .iter()
        .map(|d| {
            let inv = tb.invoke(tv(*d, &[key(*d, 1)]).with_mask(0x7e), 0).unwrap();
            assert_eq!(inv.version, Some(2));
            assert!(!inv.ready);
            inv.task
        })
        .collect();
    for w in [2, 0, 3, 1] {
        assert_eq!(tb.complete(v1[w]).ready, vec![v2[w]], "warp {w}");
    }
    let mut tb = DepTables::new(cfg(16, 16, 8));
    let u: Vec<TaskId> = h.iter().map(|d| run(&mut tb, tm(*d, 0x7e, &[]))).collect();
    // Granule c + 3 is warp 1's (c + 1 + 2), in a gap of warp 0's span.
    let (st, _) = tb.wait(Dep::Unversioned(at(c + 3))).unwrap();
    let WaitStatus::Pending(id) = st else {
        panic!("waits for warp 1")
    };
    // Warp 1 alone completes the wait.
    assert_eq!(tb.complete(u[1]).waits_done, vec![id]);
    for w in [0, 2, 3] {
        assert!(tb.complete(u[w]).waits_done.is_empty(), "warp {w}");
    }
}

#[test]
fn adoption_and_supersede_follow_the_mask() {
    let mut tb = tables();
    // A consumer of granule 1 before any producer: a pending entry. A
    // producer that holds granules 0 and 2 does not adopt it; one that
    // holds granule 1 does.
    let c = tb.invoke(t(r(8, 1), &[at(1)]), 0).unwrap();
    assert!(!c.ready);
    let even = tb.invoke(tm(r(0, 4), 0b10, &[]), 0).unwrap();
    assert!(even.effects.adopted.is_empty());
    let odd = tb.invoke(tm(r(1, 2), 0b0, &[]), 0).unwrap();
    assert_eq!(odd.effects.adopted.len(), 1);
    tb.issue(odd.task).unwrap();
    assert_eq!(tb.complete(odd.task).ready, vec![c.task]);
    // Supersede: a done plain producer of 4 granules stays when the new
    // producer holds only some of them, and retires when it holds all.
    let mut tb = tables();
    let p = run(&mut tb, t(r(0, 4), &[]));
    tb.complete(p);
    let m = tb.invoke(tm(r(0, 4), 0b10, &[]), 0).unwrap();
    assert!(m.effects.retired.is_empty());
    assert_eq!(tb.phase(p), Some(Phase::Done));
    let all = tb.invoke(tm(r(0, 4), 0b11, &[]), 0).unwrap();
    assert_eq!(all.effects.retired, vec![p]);
}

#[test]
fn one_base_one_bound_and_one_mask() {
    let mut tb = tables();
    run(&mut tb, tv(r(0, 4), &[]));
    assert_eq!(
        tb.invoke(tv(r(0, 4), &[]).with_mask(0b10), 0).unwrap_err(),
        DepError::BaseConflict {
            base: 0,
            bound: 4 * G,
            new_bound: 4 * G,
            mask: nu_model::FULL_MASK,
            new_mask: 0b10,
        }
    );
    // A mask that holds the whole span is the plain range.
    tb.invoke(tv(r(0, 4), &[]).with_mask(0b11), 0).unwrap();
}

#[test]
fn a_range_without_a_granule_is_empty() {
    let mut tb = tables();
    let e = Range::from_len(at(3), 0);
    assert_eq!(tb.invoke(t(e, &[]), 0), Err(DepError::Empty(e)));
}

// ---- epochs ----

/// Version `count` of epoch `epoch`.
fn ev(epoch: u64, count: u64) -> u64 {
    epoch << EPOCH_SHIFT | count
}

#[test]
fn a_key_of_an_earlier_epoch_whose_producer_retired_is_done() {
    let mut tb = tables();
    let a = r(0, 4);
    let p = run(&mut tb, tv(a, &[]));
    tb.complete(p);
    tb.start(1).unwrap();
    assert_eq!(tb.epoch(), 1);
    assert_eq!(
        tb.version_counter(a.base),
        None,
        "the start clears the table"
    );
    assert_eq!(tb.wait(key(a, 1)).unwrap().0, WaitStatus::Done);
    // The first version of a base in epoch 1 follows the epoch base.
    let inv = tb.invoke(tv(a, &[]), 0).unwrap();
    assert_eq!(inv.version, Some(ev(1, 1)));
}

#[test]
fn a_key_of_an_earlier_epoch_waits_for_its_producer_in_flight() {
    // P writes (A, 1) in epoch 0 and is still executing at the start of
    // epoch 1. N writes (A, 1 << 24 | 1), the same count in epoch 1. A
    // consumer of (A, 1) waits for P, not for N.
    let mut tb = tables();
    let a = r(0, 4);
    let p = run(&mut tb, tv(a, &[]));
    tb.start(1).unwrap();
    let n = run(&mut tb, tv(a, &[]));
    let c = tb
        .invoke(TaskSpec::new(r(8, 1), vec![key(a, 1)]), 0)
        .unwrap();
    assert!(!c.ready);
    assert!(tb.complete(n).ready.is_empty(), "N is not the producer");
    assert_eq!(tb.complete(p).ready, vec![c.task]);
}

#[test]
fn a_key_of_a_later_epoch_is_adopted_by_the_write_of_that_epoch() {
    // In epoch 1 a consumer names (A, 2 << 24 | 1): pending. The write of
    // count 1 in epoch 1 does not adopt it; the write of count 1 in
    // epoch 2 does.
    let mut tb = tables();
    let a = r(0, 4);
    tb.start(1).unwrap();
    let c = tb
        .invoke(TaskSpec::new(r(8, 1), vec![key(a, ev(2, 1))]), 0)
        .unwrap();
    assert!(!c.ready);
    assert_eq!(tb.pending_len(), 1);
    let w1 = tb.invoke(tv(a, &[]), 0).unwrap();
    assert_eq!(w1.version, Some(ev(1, 1)));
    assert!(w1.effects.adopted.is_empty());
    tb.issue(w1.task).unwrap();
    assert!(tb.complete(w1.task).ready.is_empty());
    tb.start(2).unwrap();
    let w2 = tb.invoke(tv(a, &[]), 0).unwrap();
    assert_eq!(w2.version, Some(ev(2, 1)));
    assert_eq!(w2.effects.adopted.len(), 1);
    tb.issue(w2.task).unwrap();
    assert_eq!(tb.complete(w2.task).ready, vec![c.task]);
}

#[test]
fn a_version_that_would_enter_the_next_epoch_is_an_error() {
    let mut tb = tables();
    let a = r(0, 4);
    tb.set_version(a.base, COUNT_MASK).unwrap();
    assert_eq!(
        tb.invoke(tv(a, &[]), 0).unwrap_err(),
        DepError::VersionOverflow {
            addr: a.base,
            counter: COUNT_MASK
        }
    );
    tb.start(1).unwrap();
    tb.set_version(a.base, ev(1, COUNT_MASK - 1)).unwrap();
    assert_eq!(
        tb.invoke(tv(a, &[]), 0).unwrap().version,
        Some(ev(1, COUNT_MASK))
    );
}

#[test]
fn a_pending_key_of_an_earlier_epoch_at_a_start_is_an_error() {
    // (A, 1) is pending in epoch 0; a start of epoch 1 ends epoch 0
    // without its producer. A pending key of the new epoch is legal.
    let mut tb = tables();
    let a = r(0, 4);
    assert!(matches!(
        tb.wait(key(a, 1)).unwrap().0,
        WaitStatus::Pending(_)
    ));
    assert_eq!(
        tb.start(1).unwrap_err(),
        DepError::StalePending {
            addr: a.base,
            version: 1,
            epoch: 1
        }
    );
    assert_eq!(tb.epoch(), 0, "a refused start changes nothing");
    tb.start(0).unwrap();
    let mut tb = tables();
    assert!(matches!(
        tb.wait(key(a, ev(1, 1))).unwrap().0,
        WaitStatus::Pending(_)
    ));
    tb.start(1).unwrap();
}

#[test]
fn a_key_that_its_epoch_never_produced_is_an_error() {
    // Epoch 0 writes A twice. In epoch 1, (A, 2) is done; (A, 3) and
    // (B, 1) were never produced: a wait, a dependency and an Await
    // report it.
    let mut tb = tables();
    let (a, b) = (r(0, 4), r(4, 4));
    for _ in 0..2 {
        let p = run(&mut tb, tv(a, &[]));
        tb.complete(p);
    }
    tb.start(1).unwrap();
    assert_eq!(tb.wait(key(a, 2)).unwrap().0, WaitStatus::Done);
    let never_a = DepError::Unproduced {
        addr: a.base,
        version: 3,
        highest: Some(2),
    };
    assert_eq!(tb.wait(key(a, 3)).unwrap_err(), never_a);
    assert_eq!(
        tb.invoke(TaskSpec::new(r(8, 1), vec![key(b, 1)]), 0)
            .unwrap_err(),
        DepError::Unproduced {
            addr: b.base,
            version: 1,
            highest: None
        }
    );
    let msg = AwaitMsg {
        from: NeutrinoId(1),
        to: NeutrinoId(0),
        request: RequestId(0),
        key: key(a, 3),
    };
    assert_eq!(handle_await(&mut tb, &msg).unwrap_err(), never_a);
}

#[test]
fn a_start_that_reuses_epoch_numbers_drops_their_records() {
    // C is written in epoch 1. After a start of epoch 2 its key is done.
    // A new lifetime starts again at epoch 1: the record of epoch 1 is
    // dropped, so after epoch 2 the same key was never produced.
    let mut tb = tables();
    let c = r(0, 4);
    tb.start(1).unwrap();
    let p = run(&mut tb, tv(c, &[]));
    tb.complete(p);
    tb.start(2).unwrap();
    assert_eq!(tb.wait(key(c, ev(1, 1))).unwrap().0, WaitStatus::Done);
    tb.start(1).unwrap();
    tb.start(2).unwrap();
    assert!(matches!(
        tb.wait(key(c, ev(1, 1))).unwrap_err(),
        DepError::Unproduced { highest: None, .. }
    ));
}

#[test]
fn vreset_keeps_the_epoch() {
    let mut tb = tables();
    let a = r(0, 4);
    tb.start(3).unwrap();
    run(&mut tb, tv(a, &[]));
    tb.vreset(a.base);
    tb.vreset_all();
    assert_eq!(tb.epoch(), 3);
}

#[test]
fn versions_order_by_the_serial_order_of_their_epochs() {
    // d = 0: the counts.
    assert_eq!(version_order(ev(7, 2), ev(7, 3)), Some(Less));
    assert_eq!(version_order(ev(7, 3), ev(7, 3)), Some(Equal));
    assert_eq!(version_order(ev(7, 4), ev(7, 3)), Some(Greater));
    // The epochs decide, whatever the counts.
    let order = |d: u64| version_order(ev((5 + d) % 256, 1), ev(5, 9));
    assert_eq!(order(1), Some(Greater));
    assert_eq!(order(127), Some(Greater));
    assert_eq!(order(128), None);
    assert_eq!(order(129), Some(Less));
    assert_eq!(order(255), Some(Less));
    // Across the wrap.
    assert_eq!(epoch_order(0, 255), Some(Greater));
    assert_eq!(epoch_order(255, 0), Some(Less));
    assert_eq!(epoch_order(200, 72), None);
}

#[test]
fn lookups_across_the_wrap_of_the_epochs() {
    // P writes (A, 255:1) in epoch 255 and is still executing in epoch 1.
    let mut tb = tables();
    let a = r(0, 4);
    tb.start(255).unwrap();
    let p = run(&mut tb, tv(a, &[]));
    assert_eq!(tb.version_counter(a.base), Some(ev(255, 1)));
    tb.start(0).unwrap();
    tb.start(1).unwrap();
    let c = tb
        .invoke(TaskSpec::new(r(8, 1), vec![key(a, ev(255, 1))]), 0)
        .unwrap();
    assert!(!c.ready, "the key waits for its producer in flight");
    assert_eq!(tb.complete(p).ready, vec![c.task]);
    assert_eq!(tb.wait(key(a, ev(255, 1))).unwrap().0, WaitStatus::Done);
    // A key of epoch 2 is after the epoch base of epoch 1: pending.
    assert!(matches!(
        tb.wait(key(a, ev(2, 1))).unwrap().0,
        WaitStatus::Pending(_)
    ));
    // Epoch 129 is 128 from epoch 1: no order.
    assert_eq!(
        tb.wait(key(a, ev(129, 1))).unwrap_err(),
        DepError::EpochWindow {
            addr: a.base,
            version: ev(129, 1),
            counter: ev(1, 0)
        }
    );
}

#[test]
fn the_records_of_the_highest_versions_across_the_wrap() {
    // Epoch 255 writes B once. In epoch 0, (B, 255:1) is done and
    // (B, 255:2) was never produced.
    let mut tb = tables();
    let b = r(4, 4);
    tb.start(254).unwrap();
    tb.start(255).unwrap();
    let p = run(&mut tb, tv(b, &[]));
    tb.complete(p);
    tb.start(0).unwrap();
    assert_eq!(tb.wait(key(b, ev(255, 1))).unwrap().0, WaitStatus::Done);
    assert_eq!(
        tb.wait(key(b, ev(255, 2))).unwrap_err(),
        DepError::Unproduced {
            addr: b.base,
            version: ev(255, 2),
            highest: Some(ev(255, 1))
        }
    );
    // Epoch 0 writes X. A start of epoch 128 (128 after epoch 0) drops
    // the record of epoch 0: in epoch 1 the key of epoch 0 was never
    // produced. Without that start it is done.
    let x = r(0, 4);
    let produced_then_started = |mid: &[u64]| {
        let mut tb = tables();
        let p = run(&mut tb, tv(x, &[]));
        tb.complete(p);
        for &e in mid {
            tb.start(e).unwrap();
        }
        tb.start(1).unwrap();
        tb.wait(key(x, ev(0, 1))).map(|w| w.0)
    };
    assert_eq!(produced_then_started(&[]), Ok(WaitStatus::Done));
    assert_eq!(
        produced_then_started(&[128]),
        Err(DepError::Unproduced {
            addr: x.base,
            version: ev(0, 1),
            highest: None
        })
    );
}

#[test]
fn stale_pending_keys_across_the_wrap() {
    // A pending key of epoch 255 at a start of epoch 1: epoch 255 is
    // before epoch 1.
    let mut tb = tables();
    let a = r(0, 4);
    tb.start(255).unwrap();
    assert!(matches!(
        tb.wait(key(a, ev(255, 1))).unwrap().0,
        WaitStatus::Pending(_)
    ));
    assert_eq!(
        tb.start(1).unwrap_err(),
        DepError::StalePending {
            addr: a.base,
            version: ev(255, 1),
            epoch: 1
        }
    );
    // A pending key of epoch 1 at a start of epoch 0: epoch 1 is after it.
    let mut tb = tables();
    assert!(matches!(
        tb.wait(key(a, ev(1, 1))).unwrap().0,
        WaitStatus::Pending(_)
    ));
    tb.start(0).unwrap();
}

#[test]
fn an_unversioned_wait_on_a_versioned_range_warns() {
    let a = r(0, 4);
    let unversioned_wait = |tb: &mut DepTables, addr: u64| {
        let (status, fx) = tb.wait(Dep::Unversioned(addr)).unwrap();
        assert!(matches!(status, WaitStatus::Pending(_)));
        fx.warnings
    };
    // The base of a version table entry.
    let mut tb = tables();
    let p = run(&mut tb, tv(a, &[]));
    tb.complete(p);
    assert_eq!(
        unversioned_wait(&mut tb, a.base),
        vec![Warning::UnversionedWait {
            addr: a.base,
            versioned: key(a, 1)
        }]
    );
    // An address that a versioned task in the tables holds.
    let mut tb = tables();
    run(&mut tb, tv(a, &[]));
    tb.vreset(a.base);
    assert_eq!(
        unversioned_wait(&mut tb, at(2)),
        vec![Warning::UnversionedWait {
            addr: at(2),
            versioned: key(a, 1)
        }]
    );
    // A versioned invoke whose dst holds an unversioned pending address.
    let mut tb = tables();
    assert!(unversioned_wait(&mut tb, at(1)).is_empty());
    let inv = tb.invoke(tv(a, &[]), 0).unwrap();
    assert_eq!(
        inv.effects.warnings,
        vec![Warning::UnversionedWait {
            addr: at(1),
            versioned: key(a, 1)
        }]
    );
    // An unversioned producer adopts the key: no warning.
    let mut tb = tables();
    unversioned_wait(&mut tb, at(1));
    assert!(tb.invoke(t(a, &[]), 0).unwrap().effects.warnings.is_empty());
}
