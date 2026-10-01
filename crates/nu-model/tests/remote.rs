use nu_model::{
    handle_await, AckAction, AwaitAck, AwaitMsg, Dep, DepConfig, DepTables, NeutrinoId,
    OutboundDeps, Purpose, Range, TaskSpec,
};

const G: u64 = 512;
const A: NeutrinoId = NeutrinoId(0);
const B: NeutrinoId = NeutrinoId(1);

fn r(base_g: u64, len_g: u64) -> Range {
    Range::from_len(base_g * G, len_g * G)
}

fn tables(pending: usize, inbound_refs: usize) -> DepTables {
    DepTables::new(DepConfig {
        scoreboard_capacity: 16,
        execution_capacity: 16,
        pending_capacity: pending,
        version_capacity: 4,
        inbound_ref_capacity: inbound_refs,
        issue_headroom: 0,
        granule: G,
    })
}

fn send_await(b: &mut DepTables, msg: &AwaitMsg) -> AwaitAck {
    handle_await(b, msg).unwrap().0
}

#[test]
fn registered_then_ready() {
    let mut a_reqs = OutboundDeps::new(A, 4);
    let mut a_tb = tables(4, 4);
    let mut b_tb = tables(4, 4);

    // B has an executing producer of X.
    let x = r(0, 4);
    let p = b_tb.invoke(TaskSpec::new(x, vec![]), 0).unwrap().task;
    b_tb.issue(p).unwrap();

    // A invokes a consumer with one external dep and opens the request.
    let c = a_tb.invoke(TaskSpec::new(r(100, 1), vec![]), 1).unwrap();
    assert!(!c.ready);
    let msg = a_reqs
        .open(B, Dep::Unversioned(x.base), Purpose::TaskDep(c.task))
        .unwrap();
    assert_eq!(
        (msg.from, msg.to, msg.key),
        (A, B, Dep::Unversioned(x.base))
    );

    let ack = send_await(&mut b_tb, &msg);
    assert_eq!(ack, AwaitAck::Registered);
    assert_eq!(b_tb.inbound_refs_len(), 1);
    assert_eq!(a_reqs.on_ack(msg.request, ack), AckAction::Wait);

    // Completion at B reports the inbound ref; the ref frees.
    let fx = b_tb.complete(p);
    assert_eq!(fx.resolves.len(), 1);
    assert_eq!(fx.resolves[0].requester, A);
    assert_eq!(fx.resolves[0].request, msg.request);
    assert_eq!(b_tb.inbound_refs_len(), 0);
    assert!(fx.retired.is_empty(), "an unversioned done entry stays");

    match a_reqs.on_resolve(msg.request) {
        Purpose::TaskDep(t) => {
            assert_eq!(t, c.task);
            assert!(a_tb.external_resolved(t));
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(a_reqs.open_count(), 0);
}

#[test]
fn done_producer_replies_done() {
    let mut a_reqs = OutboundDeps::new(A, 4);
    let mut b_tb = tables(4, 1);
    let x = r(0, 4);
    let p = b_tb.invoke(TaskSpec::new(x, vec![]), 0).unwrap().task;
    b_tb.issue(p).unwrap();
    b_tb.complete(p);

    let msg = a_reqs
        .open(B, Dep::Unversioned(x.base), Purpose::Wait(7))
        .unwrap();
    assert_eq!(send_await(&mut b_tb, &msg), AwaitAck::Done);
    assert_eq!(b_tb.inbound_refs_len(), 0);
    assert_eq!(
        a_reqs.on_ack(msg.request, AwaitAck::Done),
        AckAction::Satisfied(Purpose::Wait(7))
    );
}

#[test]
fn reject_and_retry() {
    let mut a_reqs = OutboundDeps::new(A, 4);
    let mut b_tb = tables(4, 1);
    let (x, y) = (r(0, 4), r(4, 4));
    let px = b_tb.invoke(TaskSpec::new(x, vec![]), 0).unwrap().task;
    let py = b_tb.invoke(TaskSpec::new(y, vec![]), 0).unwrap().task;

    let m1 = a_reqs
        .open(B, Dep::Unversioned(x.base), Purpose::Wait(1))
        .unwrap();
    let m2 = a_reqs
        .open(B, Dep::Unversioned(y.base), Purpose::Wait(2))
        .unwrap();
    assert_eq!(send_await(&mut b_tb, &m1), AwaitAck::Registered);
    assert_eq!(
        a_reqs.on_ack(m1.request, AwaitAck::Registered),
        AckAction::Wait
    );

    // The inbound ref table is full: reject, and the requester retries.
    assert_eq!(send_await(&mut b_tb, &m2), AwaitAck::Reject);
    let AckAction::Retry { msg, attempt } = a_reqs.on_ack(m2.request, AwaitAck::Reject) else {
        panic!("expected retry");
    };
    assert_eq!((msg, attempt), (m2, 1));

    b_tb.issue(px).unwrap();
    assert_eq!(b_tb.complete(px).resolves.len(), 1);
    assert_eq!(send_await(&mut b_tb, &msg), AwaitAck::Registered);
    b_tb.issue(py).unwrap();
    let fx = b_tb.complete(py);
    assert_eq!(fx.resolves[0].request, m2.request);
}

#[test]
fn full_pending_rejects() {
    let mut a_reqs = OutboundDeps::new(A, 4);
    let mut b_tb = tables(1, 4);
    b_tb.wait(Dep::Unversioned(r(0, 4).base)).unwrap();
    let msg = a_reqs
        .open(B, Dep::Unversioned(r(8, 4).base), Purpose::Wait(0))
        .unwrap();
    assert_eq!(send_await(&mut b_tb, &msg), AwaitAck::Reject);
}

#[test]
fn request_before_producer_invoke_registers_on_pending() {
    let mut a_reqs = OutboundDeps::new(A, 4);
    let mut b_tb = tables(4, 4);
    let x = r(0, 4);
    let msg = a_reqs
        .open(B, Dep::Unversioned(x.base), Purpose::Wait(3))
        .unwrap();
    assert_eq!(send_await(&mut b_tb, &msg), AwaitAck::Registered);
    assert_eq!(b_tb.pending_len(), 1);

    let p = b_tb.invoke(TaskSpec::new(x, vec![]), 0).unwrap();
    assert!(p.ready);
    assert_eq!(p.effects.adopted.len(), 1);
    b_tb.issue(p.task).unwrap();
    let fx = b_tb.complete(p.task);
    assert_eq!(fx.resolves.len(), 1);
    assert_eq!(fx.resolves[0].request, msg.request);
}

#[test]
fn an_await_on_an_address_inside_a_producer_registers_on_it() {
    let mut a_reqs = OutboundDeps::new(A, 4);
    let mut b_tb = tables(4, 4);
    let p = b_tb.invoke(TaskSpec::new(r(0, 4), vec![]), 0).unwrap().task;
    let msg = a_reqs
        .open(B, Dep::Unversioned(r(2, 1).base), Purpose::Wait(0))
        .unwrap();
    assert_eq!(send_await(&mut b_tb, &msg), AwaitAck::Registered);
    assert_eq!(
        b_tb.pending_len(),
        0,
        "on the producer, not a pending entry"
    );
    b_tb.issue(p).unwrap();
    assert_eq!(b_tb.complete(p).resolves[0].request, msg.request);
}

#[test]
fn request_queue_capacity() {
    let mut a_reqs = OutboundDeps::new(A, 1);
    a_reqs
        .open(B, Dep::Unversioned(r(0, 1).base), Purpose::Wait(0))
        .unwrap();
    assert!(a_reqs
        .open(B, Dep::Unversioned(r(1, 1).base), Purpose::Wait(1))
        .is_err());
}

#[test]
fn a_versioned_await_resolves_by_version() {
    // B owns X. Version 1 is done, version 2 is executing, version 3 is not
    // invoked yet.
    let mut a_reqs = OutboundDeps::new(A, 4);
    let mut b_tb = tables(4, 4);
    let x = r(0, 4);
    let key = |version| Dep::Versioned {
        addr: x.base,
        version,
    };
    let p1 = b_tb.invoke(TaskSpec::versioned(x, vec![]), 0).unwrap().task;
    b_tb.issue(p1).unwrap();
    assert_eq!(b_tb.complete(p1).retired, vec![p1]);
    let p2 = b_tb
        .invoke(TaskSpec::versioned(x, vec![key(1)]), 0)
        .unwrap()
        .task;
    b_tb.issue(p2).unwrap();

    let m1 = a_reqs.open(B, key(1), Purpose::Wait(1)).unwrap();
    assert_eq!(send_await(&mut b_tb, &m1), AwaitAck::Done);
    let m2 = a_reqs.open(B, key(2), Purpose::Wait(2)).unwrap();
    assert_eq!(send_await(&mut b_tb, &m2), AwaitAck::Registered);
    let m3 = a_reqs.open(B, key(3), Purpose::Wait(3)).unwrap();
    assert_eq!(send_await(&mut b_tb, &m3), AwaitAck::Registered);
    assert_eq!(b_tb.pending_len(), 1, "(X, 3) waits in the pending table");

    assert_eq!(b_tb.complete(p2).resolves[0].request, m2.request);
    let p3 = b_tb
        .invoke(TaskSpec::versioned(x, vec![key(2)]), 0)
        .unwrap()
        .task;
    b_tb.issue(p3).unwrap();
    assert_eq!(b_tb.complete(p3).resolves[0].request, m3.request);
}
