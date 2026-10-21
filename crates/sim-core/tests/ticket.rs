//! Ticket flow (Section 3.3) on configs/two_sm.yaml. Path timings as in
//! `hierarchy.rs`: a message of b beats between the SMs arrives 20 + 6b
//! cycles after it is sent when the path is free (a 72 B matmul
//! descriptor: 2 beats, 32 cycles; `InvokeAck`, `RetryAck`, `Reject` and
//! `PCrdGrant`: 1 beat, 26 cycles: sm1's up bridge 2 + 10, the two crossbar
//! ports 1 + 1, sm0's down bridge 2 + 10); a 64^3 bf16 matmul takes 192
//! cycles on smem.

use nu_model::{Addr, Dep, Range};
use sim_core::{build_config_file, Args, Built, Descriptor, Dtype, NuCtx, ProfileRegistry};

const MAT: u64 = 64 * 64 * 2;

/// two_sm.yaml with `flow` and the overrides `sets`.
fn built_with(flow: &str, sets: &[&str]) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_sm.yaml");
    let flow = format!("global.flow={flow}");
    let mut all = vec![flow.as_str()];
    all.extend_from_slice(sets);
    build_config_file(&path, &all, ProfileRegistry::with_builtins()).expect("config")
}

/// two_sm.yaml with sm1's scoreboard and inbound invoke table of one entry,
/// and `flow`.
fn built(flow: &str) -> Built {
    built_with(
        flow,
        &[
            "root.gpc0.sm1.neutrino.scoreboard=1",
            "root.gpc0.sm1.neutrino.inbound_invokes=1",
        ],
    )
}

/// A versioned 64^3 matmul into `c` on `pe`, labeled `label`.
fn matmul(b: &Built, pe: &str, c: Range, deps: Vec<Dep>, label: u64) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe(pe)),
        dst: c,
        versioned: true,
        untracked: false,
        inferred_version: None,
        deps,
        label: Some(label),
        args: Args::Matmul {
            a: Addr(c.base),
            b: Addr(c.base),
            a2: None,
            b2: None,
            lda: 64,
            ldb: 64,
            ldc: 64,
            m: 64,
            n: 64,
            k: 64,
            in_dtype: Dtype::Bf16,
            b_dtype: Dtype::Bf16,
            out_dtype: Dtype::Bf16,
            accumulate: false,
            keep: false,
            tile: [64, 64],
        },
    }
}

/// sm1's own tasks: P1 runs [8, 200) and P2 holds the one scoreboard entry
/// until it issues at 200.
type Program = Box<dyn FnOnce(NuCtx) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>>;

fn sm1_program(b: &Built, at: impl Fn(u64) -> Range) -> Program {
    let atlas = "root.gpc0.sm1.atlas";
    let (p1, p2) = (
        matmul(b, atlas, at(0), vec![], 100),
        matmul(b, atlas, at(1), vec![], 101),
    );
    Box::new(move |ctx: NuCtx| {
        Box::pin(async move {
            ctx.invoke(p1).await;
            let h = ctx.invoke(p2).await;
            ctx.wait(h.addr, h.version).await;
        })
    })
}

#[test]
fn one_ticket_on_a_full_inbound_table() {
    // sm0 forwards M1 and M2 (2 beats each) at 8 and 16. M1 takes sm1's
    // inbound entry at 40; its InvokeAck reaches sm0 at 66. M2 reaches sm1
    // at 56 (as in backpressure_holds_invokes_across_two_hops): the table
    // is full, so sm1 sets sm0's bit and answers RetryAck, at sm0 at 82
    // (sm1's up bridge [56, 58) + 10, the ports [68, 70), sm0's down bridge
    // [70, 72) + 10). At 200 P2 issues, M1 enters the scoreboard and its
    // inbound entry frees: sm1 reserves it and sends a PCrdGrant, at sm0 at
    // 226. sm0 replays M2 with the ticket at 226; it arrives at 258, takes
    // the reserved entry, and its InvokeAck reaches sm0 at 284. M1 issues
    // at 392 (the atlas is free); M2 enters the scoreboard then and issues
    // at 584. sm0 holds both requests from their sends to their acks.
    let b = built("ticket");
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let s1 = b.memory("root.gpc0.sm1.smem").base;
    let at = move |i: u64| Range::from_len(s1 + i * MAT, MAT);
    let atlas = "root.gpc0.sm1.atlas";
    let prog1 = sm1_program(&b, at);
    let ms: Vec<Descriptor> = (2..4)
        .map(|i| matmul(&b, atlas, at(i), vec![], i).with_inferred_version(1))
        .collect();
    let last = at(3);
    let mut sim = b.sim;
    sim.spawn(sm1, prog1);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        for m in ms {
            ctx.invoke(m).await;
        }
        ctx.wait(last.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let (m1, m2) = (&s.tasks[2], &s.tasks[3]);
    assert_eq!((m1.label, m2.label), (Some(2), Some(3)));
    assert_eq!((m1.invoke, m2.invoke), (200, 392));
    assert_eq!(m2.issue, Some(584));
    let k = &s.messages_by_kind;
    assert_eq!(k.get("Invoke"), Some(&3), "M1, M2 and M2's replay");
    assert_eq!(
        (k.get("RetryAck"), k.get("PCrdGrant")),
        (Some(&1), Some(&1))
    );
    assert_eq!((k.get("InvokeAck"), k.get("Reject")), (Some(&2), None));
    let i0 = sm0.0 as usize;
    assert_eq!(s.tickets[i0], 1);
    assert_eq!(s.ticket_rejects[i0], 0);
    assert_eq!(s.invoke_acks[i0], 2);
    // The ticket from the RetryAck (82) to the replay (226).
    assert_eq!(s.ticket_wait[i0], 226 - 82);
    assert_eq!(s.ticket_bitmap_peak[sm1.0 as usize], 1);
    assert_eq!(s.outbound_invoke_peak[i0], 2);
    assert_eq!(s.resends, 1);
}

#[test]
fn grants_go_round_robin_between_two_ticket_holders() {
    // sm0 forwards M1, M2, M3 at 8, 16, 24; gpc0 forwards N1 and N2
    // after 30 cycles, so they reach sm1 after M1 (at 40) took the one
    // inbound entry. M2 sets sm0's bit and N1 gpc0's; M3 and N2 are
    // rejected behind them and replayed later. Each freed entry goes to the next set
    // bit after the last one granted: gpc0 (id 1), sm0 (id 2),
    // gpc0, sm0. So sm1 accepts M1, N1, M2, N2, M3: each requester's
    // invokes in its program order, the two interleaved.
    let b = built("ticket");
    let (c0, sm0, sm1) = (
        b.nu("root.gpc0"),
        b.nu("root.gpc0.sm0"),
        b.nu("root.gpc0.sm1"),
    );
    assert!(c0.0 < sm0.0);
    let s1 = b.memory("root.gpc0.sm1.smem").base;
    let at = move |i: u64| Range::from_len(s1 + i * MAT, MAT);
    let atlas = "root.gpc0.sm1.atlas";
    let prog1 = sm1_program(&b, at);
    let ms: Vec<Descriptor> = (2..5)
        .map(|i| matmul(&b, atlas, at(i), vec![], i).with_inferred_version(1))
        .collect();
    let ns: Vec<Descriptor> = (5..7)
        .map(|i| matmul(&b, atlas, at(i), vec![], i).with_inferred_version(1))
        .collect();
    let (m_last, n_last) = (at(4), at(6));
    let mut sim = b.sim;
    sim.spawn(sm1, prog1);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        for m in ms {
            ctx.invoke(m).await;
        }
        ctx.wait(m_last.base, 1).await;
    });
    sim.spawn(c0, move |ctx: NuCtx| async move {
        ctx.cycles(30).await;
        for n in ns {
            ctx.invoke(n).await;
        }
        ctx.wait(n_last.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let mut order: Vec<(u64, u64)> = s
        .tasks
        .iter()
        .filter(|t| t.label.is_some_and(|l| (2..100).contains(&l)))
        .map(|t| (t.invoke, t.label.unwrap()))
        .collect();
    order.sort();
    let labels: Vec<u64> = order.iter().map(|x| x.1).collect();
    assert_eq!(labels, vec![2, 5, 3, 6, 4]);
    let (i0, ic) = (sm0.0 as usize, c0.0 as usize);
    assert_eq!((s.tickets[i0], s.tickets[ic]), (1, 1));
    assert_eq!((s.ticket_rejects[i0], s.ticket_rejects[ic]), (1, 1));
    assert_eq!(s.ticket_bitmap_peak[sm1.0 as usize], 2);
    assert_eq!(s.messages_by_kind.get("PCrdGrant"), Some(&4));
}

#[test]
fn a_requester_sends_no_invoke_while_it_holds_a_ticket() {
    // As one_ticket_on_a_full_inbound_table, and at 124 sm0 forwards J, a
    // join on gpc0's sync memory (a free owner). Under ready/valid J
    // leaves at once. Under ticket flow sm0 holds a ticket from 82 to 226,
    // so J waits in its outbound invoke table: M2's replay takes sm0's up
    // bridge [226, 230) and the stage after it until its transmission
    // through the port from sm0 ends at 242; J follows it. J therefore
    // reaches gpc0 242 - 124 = 118 cycles later than under ready/valid.
    let run = |flow: &str| {
        let b = built(flow);
        let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
        let c0 = b.nu("root.gpc0");
        let s1 = b.memory("root.gpc0.sm1.smem").base;
        let at = move |i: u64| Range::from_len(s1 + i * MAT, MAT);
        let atlas = "root.gpc0.sm1.atlas";
        let prog1 = sm1_program(&b, at);
        let ms: Vec<Descriptor> = (2..4)
            .map(|i| matmul(&b, atlas, at(i), vec![], i).with_inferred_version(1))
            .collect();
        let js = Range::from_len(b.sync_only(c0).expect("sync").base, 512);
        let mut j = Descriptor::join(js, vec![]).with_inferred_version(1);
        j.versioned = true;
        j.label = Some(50);
        let m_last = at(3);
        let mut sim = b.sim;
        sim.spawn(sm1, prog1);
        sim.spawn(sm0, move |ctx: NuCtx| async move {
            for m in ms {
                ctx.invoke(m).await;
            }
            ctx.cycles(100).await;
            ctx.invoke(j).await;
            ctx.wait(js.base, 1).await;
            ctx.wait(m_last.base, 1).await;
        });
        let s = sim.run();
        assert_eq!(s.fatal, None, "{flow}");
        assert!(s.unfinished.is_empty(), "{flow}");
        let jt = s.tasks.iter().find(|t| t.label == Some(50)).unwrap();
        assert_eq!(jt.nu, c0);
        (jt.invoke, s.ticket_wait[sm0.0 as usize])
    };
    let (rv, _) = run("readyvalid");
    let (tk, wait) = run("ticket");
    assert_eq!(wait, 226 - 82);
    assert_eq!(tk - rv, 242 - 124);
}

#[test]
fn a_request_in_flight_behind_the_ticket_is_rejected_and_replayed_in_order() {
    // sm0 forwards M1, M2, M3 at 8, 16, 24. M1 takes the inbound entry at
    // 40; M2 at 56 gets the RetryAck (at sm0 at 82). M3 was sent before
    // sm0 knew: it reaches sm1 at 72 (sm0's up bridge [40, 44), stage at
    // 54; ports [54, 58); sm1's down bridge [58, 62) + 10) and is rejected
    // (the Reject reaches sm0 at 98). At 82 sm0 marks M2 and M3 for replay.
    // The first grant (sent at 200, at sm0 at 226) replays M2 (accepted at
    // 258); M2 then holds the inbound entry until M1 issues at 392 and M2
    // enters the scoreboard; the second grant (at sm0 at 418) replays M3.
    // M3 enters the scoreboard when M2 issues at 584.
    let b = built("ticket");
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let s1 = b.memory("root.gpc0.sm1.smem").base;
    let at = move |i: u64| Range::from_len(s1 + i * MAT, MAT);
    let atlas = "root.gpc0.sm1.atlas";
    let prog1 = sm1_program(&b, at);
    let ms: Vec<Descriptor> = (2..5)
        .map(|i| matmul(&b, atlas, at(i), vec![], i).with_inferred_version(1))
        .collect();
    let last = at(4);
    let mut sim = b.sim;
    sim.spawn(sm1, prog1);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        for m in ms {
            ctx.invoke(m).await;
        }
        ctx.wait(last.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let inv = |l: u64| s.tasks.iter().find(|t| t.label == Some(l)).unwrap().invoke;
    assert_eq!((inv(2), inv(3), inv(4)), (200, 392, 584));
    let i0 = sm0.0 as usize;
    assert_eq!((s.tickets[i0], s.ticket_rejects[i0]), (1, 1));
    assert_eq!(s.invoke_acks[i0], 3);
    assert_eq!(s.resends, 2);
    assert_eq!(s.messages_by_kind.get("PCrdGrant"), Some(&2));
    assert_eq!(s.messages_by_kind.get("Invoke"), Some(&5));
    // The ticket from the RetryAck (82) to the last replay (418).
    assert_eq!(s.ticket_wait[i0], 418 - 82);
    assert_eq!(s.ticket_bitmap_peak[sm1.0 as usize], 1);
}

#[test]
fn an_outbound_entry_frees_on_its_invoke_ack() {
    // sm0 has one outbound invoke entry and sm1 has room. sm0 forwards M1
    // at 8 and M2 at 16. Under ready/valid M1's entry frees when M1 leaves
    // sm0's source queue, and M2 reaches sm1 at 56 (it waits for the stage
    // after sm0's up bridge, as in backpressure_holds_invokes_across_two_hops).
    // Under ticket flow sm0 holds M1 until its InvokeAck arrives at 66: the
    // program stalls from 16 to 66, and M2 arrives at 66 + 32 = 98.
    let run = |flow: &str| {
        let b = built_with(flow, &["root.gpc0.sm0.neutrino.outbound_invokes=1"]);
        let sm0 = b.nu("root.gpc0.sm0");
        let s1 = b.memory("root.gpc0.sm1.smem").base;
        let at = move |i: u64| Range::from_len(s1 + i * MAT, MAT);
        let atlas = "root.gpc0.sm1.atlas";
        let ms: Vec<Descriptor> = (2..4)
            .map(|i| matmul(&b, atlas, at(i), vec![], i).with_inferred_version(1))
            .collect();
        let last = at(3);
        let mut sim = b.sim;
        sim.spawn(sm0, move |ctx: NuCtx| async move {
            for m in ms {
                ctx.invoke(m).await;
            }
            ctx.wait(last.base, 1).await;
        });
        let s = sim.run();
        assert_eq!(s.fatal, None, "{flow}");
        let inv = |l: u64| s.tasks.iter().find(|t| t.label == Some(l)).unwrap().invoke;
        let i0 = sm0.0 as usize;
        (
            (inv(2), inv(3)),
            s.outbound_stall[i0],
            s.outbound_invoke_peak[i0],
        )
    };
    assert_eq!(run("readyvalid").0, (40, 56));
    let (t, stall, peak) = run("ticket");
    assert_eq!(t, (40, 98));
    assert_eq!(stall, 66 - 16);
    assert_eq!(peak, 1);
}

#[test]
fn a_version_reset_at_a_full_inbound_table_is_replayed_in_order() {
    // M1 (version 1 of C) takes sm1's one inbound entry. sm0's
    // VersionReset of C then finds the table full: sm1 answers RetryAck
    // and sets sm0's bit (a VersionReset takes an entry like an Invoke,
    // 2026-10-20). When M1 enters the scoreboard (200) the freed entry is
    // granted, sm0 replays the VersionReset, and it frees C's version
    // entry: M2, invoked after M1 is done, is version 1 again.
    let b = built("ticket");
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let s1 = b.memory("root.gpc0.sm1.smem").base;
    let at = move |i: u64| Range::from_len(s1 + i * MAT, MAT);
    let atlas = "root.gpc0.sm1.atlas";
    let prog1 = sm1_program(&b, at);
    let c = at(2);
    let (m1, m2) = (
        matmul(&b, atlas, c, vec![], 2).with_inferred_version(1),
        matmul(&b, atlas, c, vec![], 3).with_inferred_version(1),
    );
    let mut sim = b.sim;
    sim.spawn(sm1, prog1);
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        ctx.invoke(m1).await;
        ctx.vreset(c.base).await;
        ctx.wait(c.base, 1).await;
        ctx.invoke(m2).await;
        ctx.wait(c.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let v = |l: u64| s.tasks.iter().find(|t| t.label == Some(l)).unwrap().version;
    assert_eq!((v(2), v(3)), (Some(1), Some(1)));
    let k = &s.messages_by_kind;
    assert_eq!(
        (k.get("RetryAck"), k.get("PCrdGrant")),
        (Some(&1), Some(&1))
    );
    assert_eq!(k.get("VersionReset"), Some(&2), "the reset and its replay");
    assert_eq!(s.tickets[sm0.0 as usize], 1);
}
