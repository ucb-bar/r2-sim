//! Per-gpc drivers (Radiance 2 spec Section 6.2): a device program that
//! starts a program on each gpc, cross-gpc waits, the version table reset
//! at Start, and two gpc programs as requesters of one owner.
//! Config `tests/configs/two_gpc.yaml`; control runs at the data rate.
//!
//! Message times used below (1 beat unless noted):
//! * root -> gpc (Start): the gpc's bridge down (1 cycle + 20), its control
//!   crossbar's bridge port and neutrino port (1 + 1): 23 cycles.
//! * The SM bridges (32 B per cycle) have capacity 1/2 at the 64 B message
//!   beat: one slot in each odd cycle (whole slots by cycle). A beat that
//!   enters at an odd cycle takes 1 cycle, at an even cycle 2.
//! * gpc -> its SM: the crossbar's neutrino port and SM port (1 + 1), the
//!   SM's bridge down (1 or 2, + 10): 13 or 14 cycles; a 72 B matmul
//!   descriptor (2 beats): 2 + 2 + (3 or 4) + 10 = 17 or 18.
//! * SM -> its gpc: (1 or 2) + 10 + 1 + 1 = 13 or 14. gpc -> root
//!   (Return): 1 + 1 + 1 + 20 = 23.
//! * Costs: c_spawn 16, c_invoke 8, c_wait 2, c_complete 4; a 64^3 bf16
//!   matmul takes 192 cycles on smem.

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::{Addr, Dep, Range};
use sim_core::{build_config_file, Args, Built, Descriptor, Dtype, NuCtx, ProfileRegistry};

const MAT: u64 = 64 * 64 * 2;

fn two_gpc(sets: &[&str]) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/two_gpc.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins()).expect("config")
}

fn matmul(b: &Built, pe: &str, c: Range, label: u64) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe(pe)),
        dst: c,
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
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

type Log = Rc<RefCell<Vec<(&'static str, u64)>>>;

#[test]
fn a_device_program_starts_two_gpc_drivers() {
    // The root spawns P0 on gpc0 (Start sent at 16, after c_spawn) and P1
    // on gpc1 (sent at 32): they start at 16 + 23 = 39 and 32 + 23 = 55.
    // * P1 forwards M1 to gpc1.sm0 at 63 (after c_invoke): ports [63, 67),
    //   sm0's bridge down [67, 70) (slots 67 and 69) + 10: it enters the
    //   scoreboard at 80 and runs [80, 272), done at 276. P1's Await (sent
    //   at 65 after c_wait) waits for the SM bridge behind M1 ([70, 72) +
    //   10: at 82). The Resolve leaves at 276, takes sm0's bridge up
    //   [276, 278) + 10 and the ports [288, 290): it reaches gpc1 at 290;
    //   P1 returns, the Return reaches the root at 313, and spawn task 1
    //   completes at 317.
    // * P0 waits for (M1, 1) across the gpcs: its Await (sent at 41) takes
    //   gpc0's crossbar ports [41, 43), gpc0's bridge up [43, 44) + 20,
    //   gpc1's bridge down [64, 65) + 20, gpc1's ports [85, 87) and sm0's
    //   bridge down [87, 88) + 10: at 98. Its Resolve follows P1's: the
    //   stage after sm0's bridge up holds P1's until 289, so [289, 290) +
    //   10, sm0's port [300, 301), gpc1's bridge port [301, 302), gpc1's
    //   bridge up [302, 303) + 20, gpc0's bridge down [323, 324) + 20,
    //   gpc0's ports [344, 346): P0 resumes at 346. Its Return (ports
    //   [346, 348), bridge up [348, 349) + 20) reaches the root at 369;
    //   spawn task 0 completes at 373, the root resumes, and its wait on
    //   the done spawn task 1 ends at 375.
    let b = two_gpc(&[]);
    let (root, g0, g1) = (b.nu("root"), b.nu("root.gpc0"), b.nu("root.gpc1"));
    let sync = b.sync_only(root).unwrap();
    let m1r = Range::from_len(b.memory("root.gpc1.sm0.smem").base, MAT);
    let m1 = matmul(&b, "root.gpc1.sm0.atlas", m1r, 1).with_inferred_version(1);
    let log: Log = Rc::default();
    let (l0, l1, lr) = (log.clone(), log.clone(), log.clone());
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        let p0 = move |c: NuCtx| async move {
            l0.borrow_mut().push(("p0 start", c.now()));
            c.wait(m1r.base, 1).await;
            l0.borrow_mut().push(("p0 resume", c.now()));
        };
        let p1 = move |c: NuCtx| async move {
            l1.borrow_mut().push(("p1 start", c.now()));
            c.invoke(m1).await;
            c.wait(m1r.base, 1).await;
            l1.borrow_mut().push(("p1 resume", c.now()));
        };
        let h0 = ctx
            .spawn_child(g0, Range::from_len(sync.base, 512), vec![], p0)
            .await;
        let h1 = ctx
            .spawn_child(g1, Range::from_len(sync.base + 512, 512), vec![], p1)
            .await;
        ctx.wait_key(h0.key()).await;
        lr.borrow_mut().push(("root h0", ctx.now()));
        ctx.wait_key(h1.key()).await;
        lr.borrow_mut().push(("root h1", ctx.now()));
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let mut got = log.borrow().clone();
    got.sort_by_key(|x| x.1);
    assert_eq!(
        got,
        vec![
            ("p0 start", 39),
            ("p1 start", 55),
            ("p1 resume", 290),
            ("p0 resume", 346),
            ("root h0", 373),
            ("root h1", 375)
        ]
    );
    let t = s.tasks.iter().find(|t| t.label == Some(1)).unwrap();
    assert_eq!((t.invoke, t.issue, t.complete), (80, Some(80), Some(276)));
    let k = &s.messages_by_kind;
    assert_eq!((k.get("Start"), k.get("Return")), (Some(&2), Some(&2)));
}

#[test]
fn a_start_resets_the_gpc_version_table_only() {
    // The root starts two programs on gpc0, one after the other. Each
    // writes version 1 of X (a join on gpc0's own sync space: the table
    // starts again at each Start) and the next version of Y (a join on
    // gpc0.sm0's sync space, whose table is not reset): 1, then 2.
    let b = two_gpc(&[]);
    let (root, g0) = (b.nu("root"), b.nu("root.gpc0"));
    let sync = b.sync_only(root).unwrap();
    let x = Range::from_len(b.sync_only(g0).unwrap().base, 512);
    let ysm = b.nu("root.gpc0.sm0");
    let y = Range::from_len(b.sync_only(ysm).unwrap().base, 512);
    let log: Log = Rc::default();
    let program = |log: Log, yv: u64| {
        move |c: NuCtx| async move {
            let mut jx = Descriptor::join(x, vec![]);
            jx.versioned = true;
            let jy = Descriptor::join(y, vec![]).with_inferred_version(yv);
            let hx = c.invoke(jx).await;
            log.borrow_mut().push(("x", hx.version));
            c.invoke(jy).await;
            c.wait(x.base, hx.version).await;
            c.wait(y.base, yv).await;
        }
    };
    let (pa, pb) = (program(log.clone(), 1), program(log.clone(), 2));
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        let h = ctx
            .spawn_child(g0, Range::from_len(sync.base, 512), vec![], pa)
            .await;
        ctx.wait_key(h.key()).await;
        let h = ctx
            .spawn_child(g0, Range::from_len(sync.base + 512, 512), vec![], pb)
            .await;
        ctx.wait_key(h.key()).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None, "the owner checks the inferred versions of Y");
    assert!(s.unfinished.is_empty());
    assert_eq!(*log.borrow(), vec![("x", 1), ("x", 1)]);
    let ys: Vec<Option<u64>> = s
        .tasks
        .iter()
        .filter(|t| t.nu == ysm)
        .map(|t| t.version)
        .collect();
    assert_eq!(ys, vec![Some(1), Some(2)]);
}

#[test]
fn two_gpc_programs_take_turns_at_a_full_owner() {
    // Ticket flow; gpc0.sm0 has one scoreboard entry and one inbound
    // invoke entry, held by its own long task L (it waits for (Z, 1), which
    // Q at gpc0.sm1 writes after 2000 cycles). P0 (gpc0) forwards a1, a2,
    // a3 and P1 (gpc1) b1, b2 to gpc0.sm0. a1 takes the inbound entry (gpc0
    // is one hop away); a2 gets gpc0's ticket and a3 is rejected behind
    // it; b1 gets gpc1's ticket and b2 is rejected. The bitmap holds two
    // bits, and each freed entry goes to the next bit after the last one
    // granted: gpc0 (a2), gpc1 (b1), gpc0 (a3), gpc1 (b2).
    let b = two_gpc(&[
        "root.gpc0.sm0.neutrino.scoreboard=1",
        "root.gpc0.sm0.neutrino.inbound_invokes=1",
        "global.flow=ticket",
    ]);
    let (root, g0, g1) = (b.nu("root"), b.nu("root.gpc0"), b.nu("root.gpc1"));
    let sm0 = b.nu("root.gpc0.sm0");
    let sm1 = b.nu("root.gpc0.sm1");
    let sync = b.sync_only(root).unwrap();
    let s0 = b.memory("root.gpc0.sm0.smem").base;
    let at = move |i: u64| Range::from_len(s0 + i * MAT, MAT);
    let z = Range::from_len(b.memory("root.gpc0.sm1.smem").base, MAT);
    let atlas = "root.gpc0.sm0.atlas";
    let mut l = matmul(&b, atlas, at(0), 100);
    l.deps = vec![Dep::Versioned {
        addr: z.base,
        version: 1,
    }];
    let q = matmul(&b, "root.gpc0.sm1.atlas", z, 101);
    let mine = |i: u64, label: u64| matmul(&b, atlas, at(i), label).with_inferred_version(1);
    let (a1, a2, a3) = (mine(1, 1), mine(2, 2), mine(3, 3));
    let (b1, b2) = (mine(4, 11), mine(5, 12));
    let last = at(5);
    let mut sim = b.sim;
    sim.spawn(sm0, move |c: NuCtx| async move {
        c.invoke(l).await;
    });
    sim.spawn(sm1, move |c: NuCtx| async move {
        c.cycles(2000).await;
        c.invoke(q).await;
        c.wait(z.base, 1).await;
    });
    sim.spawn(root, move |ctx: NuCtx| async move {
        let p0 = move |c: NuCtx| async move {
            c.invoke(a1).await;
            c.invoke(a2).await;
            c.invoke(a3).await;
            c.wait(at(3).base, 1).await;
        };
        let p1 = move |c: NuCtx| async move {
            c.invoke(b1).await;
            c.invoke(b2).await;
            c.wait(last.base, 1).await;
        };
        let h0 = ctx
            .spawn_child(g0, Range::from_len(sync.base, 512), vec![], p0)
            .await;
        let h1 = ctx
            .spawn_child(g1, Range::from_len(sync.base + 512, 512), vec![], p1)
            .await;
        ctx.wait_key(h0.key()).await;
        ctx.wait_key(h1.key()).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None, "{:?}", s.fatal);
    assert!(s.unfinished.is_empty());
    assert_eq!(s.ticket_bitmap_peak[sm0.0 as usize], 2);
    let mut order: Vec<(u64, u64)> = s
        .tasks
        .iter()
        .filter(|t| t.nu == sm0 && t.label.is_some_and(|l| l < 100))
        .map(|t| (t.invoke, t.label.unwrap()))
        .collect();
    order.sort();
    let labels: Vec<u64> = order.iter().map(|x| x.1).collect();
    assert_eq!(labels, vec![1, 2, 11, 3, 12]);
}
