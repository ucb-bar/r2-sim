//! Builder tests on tests/configs/tiny.yaml with small shapes, and one test
//! of head-of-line blocking of control messages on the same config. Timings
//! are reported first; the exact values are asserted in a second pass.

use func::{Numeric, Records, Trace};
use nu_prog::adamoe::{Driver, Params};
use nu_prog::{label, Machine};
use sim_core::{build_config_file, Args, Built, Descriptor, Dtype, NuCtx, ProfileRegistry, Stats};

fn tiny_with(sets: &[&str]) -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/configs/tiny.yaml");
    build_config_file(&path, sets, ProfileRegistry::with_builtins()).unwrap()
}

fn tiny() -> Built {
    tiny_with(&[])
}

fn params() -> Params {
    Params {
        steps: 1,
        layers: 1,
        tokens: 16,
        d: 256,
        ffn: 128,
        heads: 1,
        head_dim: 128,
        prefix: 128,
        experts: 2,
        shared_expert: false,
        expert_sms: vec![vec![1], vec![2]],
        block_cols: 128,
        key_block: 128,
        dma_chunk: 1 << 20,
        gather_bufs: 2,
        tile: [128, 128],
        ..Params::default()
    }
}

fn linear(b: Built, n_out: u64) -> (Stats, u64) {
    let m = Machine::from_built(&b, 512).unwrap();
    let d = Driver::new(m, params()).unwrap();
    let nu = d.home_nu();
    let report = d.report.clone();
    let mut sim = b.sim;
    sim.spawn(nu, move |ctx| d.run_linear(ctx, n_out));
    let s = sim.run();
    let end = report.borrow().step_end[0];
    (s, end)
}

#[test]
fn one_linear_layer() {
    let (s, end) = linear(tiny(), 256);
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    // Hand computation (tiny.yaml; own co-charging; control links at 1/8
    // of the data rate: a 4 B/cycle SM bridge lane has capacity 1/16 at the
    // 64 B message beat, with its slot in the cycles 15 mod 16, and an
    // 8 B/cycle crossbar port lane capacity 1/8, with its slot in the
    // cycles 7 mod 8 (whole slots by cycle): a beat waits for the next
    // slot cycle; each bridge adds 10 cycles of latency; the crossbar
    // clsbus is an xbar, charged per port). Control messages go store and
    // forward through one-message stages (after sm0's up bridge, after the
    // port from sm0, after the port to the target, then the target's
    // inbound table): a 1-beat message between SMs sent at 8 takes the up
    // bridge [8, 16) + 10, the ports [26, 32) and [32, 40) and the down
    // bridge [40, 48) + 10: it arrives at 58. Each SM has one output
    // block, so its end signal is a pin of the matmul (no join), and x has
    // no producer in the window (its dep drops).
    // * The driver invokes 6 tasks at 8, 16, .., 48. The stage after sm0's
    //   up bridge holds one message until its transmission through the port
    //   from sm0 ends, so each forward starts its up-bridge transmission
    //   when the previous one leaves that stage: sm1 enters its tasks at
    //   58, 90, 170 (up bridge from 8, 32, 64; the third is a 2-beat
    //   matmul descriptor (88 B) with the slots 79 and 95, ports to 136,
    //   down bridge [136, 160)), sm2 at 170, 202, 282 (from 120, 144, 176).
    // * DMAs (r2sim spec Section 5.6, data-path rule): hops in route order after the
    //   engine latency 50, the request's trip to the source block (the hop
    //   latencies from the engine, 2026-10-21: 20 from sm1 or sm2 to sm0's
    //   SMEM over the two SM bridges, 10 to the HBM over the SM's bridge)
    //   and the source block's latency (SMEM 10, HBM 100; r2sim spec Section 5.9),
    //   at the engine's 64 B beat narrowed by each hop.
    //   x to sm1 (8 KiB): sm0's SMEM port [138, 266) on one of its two
    //   slots, sm0's bridge up (32 B/cycle) [138, 394), sm1's bridge down
    //   and SMEM port to 414. x to sm2: its SMEM hop [250, 378) on the
    //   other slot; sm0's bridge is busy until 394, so the data waits and
    //   crosses it [394, 650): chunk [250, 670), end at sm2's SMEM port.
    // * W0 (64 KiB from HBM to sm1): HBM (64 B/cycle) [250, 1274), not held
    //   back by the slower bridge (no backpressure); sm1's bridge down after
    //   the x DMA, [404, 2452), its SMEM port to 2462. W1 (to sm2): HBM
    //   after W0, [1274, 2298); sm2's bridge [1274, 3322), SMEM to 3332.
    // * Matmuls (608 cycles on SMEM) after the weights: sm1 [2466, 3074),
    //   done 3078; sm2 [3336, 3944), done 3948.
    // * The driver waits on (out1, 1): Resolve leaves sm1 at 3078 and
    //   reaches it at 3134; the Await for (out2, 1) registers at sm2;
    //   Resolve leaves at 3948: up bridge [3948, 3952) + 10 (its slot is
    //   at the cycles 15 mod 16), ports [3962, 3968) and [3968, 3976), down
    //   bridge [3976, 3984) + 10: it arrives at 3994.
    // * sm0's source invoke queue: the head waits from 16 to 32, 32 to 64,
    //   64 to 120, 120 to 144 and 144 to 176 (160 cycles).
    assert_eq!(end, 3994);
    assert_eq!(
        (s.tasks[3].chunks[0].start, s.tasks[3].chunks[0].end),
        (250, 670),
        "sm2's x DMA waits for sm0's bridge behind sm1's"
    );
    assert_eq!(
        s.tasks[1].chunks[0],
        sim_core::ChunkRec {
            start: 250,
            end: 2462,
            culprit: s.tasks[1].chunks[0].culprit
        }
    );
    assert_eq!(
        s.tasks[4].chunks[0].start, 1274,
        "W1 queues on HBM only for W0's HBM window"
    );
    // 6 Invoke, 2 Await, 2 AwaitAck, 2 Resolve; the two matmul invokes
    // take 2 beats.
    assert_eq!(s.messages, 12);
    assert_eq!(s.beats, [10, 2, 0, 0]);
    let hol = |name: &str| {
        s.stages
            .iter()
            .find(|x| x.stage == name)
            .map_or(0, |x| x.hol_cycles)
    };
    assert_eq!(hol("root.gpc0.sm0.source.invoke"), 160);
}

#[test]
fn one_moe_dispatch() {
    let b = tiny();
    let m = Machine::from_built(&b, 512).unwrap();
    let d = Driver::new(m, params()).unwrap();
    let nu = d.home_nu();
    let report = d.report.clone();
    // Tokens 0..3 to expert 0, 3..5 to expert 1; the rest to expert 0 as
    // well would change n, so give the other tokens expert 0 too: n = (14, 2).
    let mut ids = vec![0u8; 16];
    ids[3] = 1;
    ids[4] = 1;
    let mut r = Records::new();
    r.insert(label::router(0, 0), ids);
    let mut sim = b.sim;
    sim.set_functional(Box::new(Trace::new(r)));
    sim.spawn(nu, move |ctx| d.run_dispatch(ctx));
    let s = sim.run();
    assert_eq!(s.fatal, None, "{:?}", s.fatal);
    assert!(s.unfinished.is_empty() && s.open_pending.is_empty());
    let r = report.borrow();
    assert_eq!(r.tokens_per_expert, vec![vec![vec![14, 2]]]);
    // Critical path (by hand; control messages by the slot cycles of
    // one_linear_layer; DMAs by the data-path rule, hops in route order at
    // the engine's 64 B beat after the engine latency 50, the request's
    // trip to the source block (2026-10-21: 10 from an SM to the HBM, 20
    // between two SMs) and the source block's latency, HBM 100 and SMEM 10;
    // a Muon task after its overhead and its read's SMEM latency 10,
    // 2026-10-21): rmsnorm [28, 284);
    // router [288, 361); the load of its output [376, 377); the MoE
    // window's forwards from 385: the first takes the up bridge (1/16 slot,
    // whose slot is at the cycles 15 mod 16) [385, 400) + 10, the ports
    // [410, 416) and [416, 424) and the down bridge [424, 432) + 10 and
    // reaches sm1 at 442; the second (expert 0's gate weight DMA) reaches
    // it at 474. HBM (64 B/cycle) is not held back by the 32 B/cycle SM
    // bridges, so it serves the weight DMAs in invoke order, 1024 cycles
    // each: expert 0's gate [634, 1658) and up [1658, 2682), expert 1's
    // gate [2682, 3706) and up [3706, 4730).
    // Expert 1 on sm2 (2 tokens): its gate weight crosses sm2's bridge
    // [2682, 4730) and reaches SMEM at 4740. The up weight's DMA then holds
    // 0.5 SMEM slot (32 B/cycle) until 6788, so the gate matmul starts on
    // the 1.5 free slots (r2sim spec Section 5.7, starting on the free slots): its
    // 1048 slot-cycles take 699 cycles, [4744, 5443). The down weight's DMA
    // waits for sm2's bridge behind the up weight: [6778, 8826), SMEM to
    // 8836. The up matmul (after the up weight, 6792) also runs on the 1.5
    // slots beside the down weight's DMA: [6792, 7491). The vector op is
    // bound by its Muon, so its SMEM demand takes one slot, [7515, 7547).
    // Down matmul (it reads the down weight) 2 x 264 cycles [8840, 9368),
    // done 9372. Home depends on the down matmul's output directly (a pin,
    // no end join): Resolve takes sm2's up bridge [9372, 9376) + 10 (its
    // slot is at the cycles 15 mod 16), the ports [9386, 9392) and
    // [9392, 9400) and sm0's down bridge [9400, 9408) + 10 and reaches home
    // at 9418; pull (request 20 over the two SM bridges, SMEM latency 10,
    // engine latency 50) [9498, 9550); add [9574, 9598), done 9602; the
    // driver resumes at 9602. Against 9556 before the Muon paid its read's
    // latency: +10 (rmsnorm), +6 (the forwards meet the up bridge's slot
    // at 399 instead of 383) and +10 (add); then +20 for the pull's
    // request (2026-10-21; the weights' request of 10 is absorbed by the
    // Resolve's wait for its slot).
    assert_eq!(r.step_end[0], 9602);
    // 16 Invoke, 2 Await, 2 AwaitAck, 2 Resolve.
    assert_eq!(s.messages, 22);
}

#[test]
fn zero_time_linear_matches_host() {
    // HBM must hold data for this check: numeric fidelity from the device.
    let b = tiny_with(&["root.fidelity=numeric"]);
    let m = Machine::from_built(&b, 512).unwrap();
    let p = params();
    let d = Driver::new(m, p.clone()).unwrap();
    let nu = d.home_nu();
    let outs = d.linear_outputs(256);
    let x = d.home_x();
    // bf16 inputs from a small LCG.
    let mut seed = 1u32;
    let mut rnd = |n: usize| -> Vec<u16> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                func::f32_to_bf16(((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5)
            })
            .collect()
    };
    let (t, dd, bc) = (p.tokens as usize, p.d as usize, p.block_cols as usize);
    let xs = rnd(t * dd);
    let ws: Vec<Vec<u16>> = (0..2).map(|_| rnd(dd * bc)).collect();
    let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut sim = b.sim;
    sim.set_functional(Box::new(Numeric::new()));
    sim.set_zero_time(true);
    sim.init_memory(nu_model::Addr(x.base), bytes(&xs));
    for (b_ix, w) in ws.iter().enumerate() {
        sim.init_memory(nu_model::Addr(d.linear_block(b_ix as u64)), bytes(w));
    }
    sim.spawn(nu, move |ctx| d.run_linear(ctx, 256));
    let (s, backend) = sim.run_with_backend();
    assert_eq!(s.fatal, None, "{:?}", s.fatal);
    let mut backend = backend.unwrap();
    for (b_ix, out) in outs {
        let got = backend
            .read(nu_model::Addr(out.base), (t * bc * 2) as u64)
            .unwrap();
        let w = &ws[b_ix as usize];
        let mut want = Vec::with_capacity(t * bc);
        for i in 0..t {
            for j in 0..bc {
                let mut acc = 0.0f32;
                for k in 0..dd {
                    acc += func::bf16_to_f32(xs[i * dd + k]) * func::bf16_to_f32(w[k * bc + j]);
                }
                want.push(func::f32_to_bf16(acc));
            }
        }
        assert_eq!(got, bytes(&want), "block {b_ix}");
    }
}

/// A versioned `n`^3 bf16 matmul into `c` on `pe`.
fn matmul(b: &Built, pe: &str, c: nu_model::Range, n: u64, deps: Vec<nu_model::Dep>) -> Descriptor {
    Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe(pe)),
        dst: c,
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        pc: None,
        deps,
        label: None,
        args: Args::Matmul {
            a: nu_model::Addr(c.base),
            b: nu_model::Addr(c.base),
            a2: None,
            b2: None,
            lda: n,
            ldb: n,
            ldc: n,
            m: n,
            n,
            k: n,
            in_dtype: Dtype::Bf16,
            b_dtype: Dtype::Bf16,
            out_dtype: Dtype::Bf16,
            accumulate: false,
            keep: false,
            tile: [64, 64],
        },
    }
}

#[test]
fn an_invoke_to_a_full_owner_delays_an_invoke_to_a_free_owner() {
    // Head-of-line blocking (Neutrino spec Section 9.2): sm1 has one scoreboard entry and
    // one inbound invoke entry; P1 (128^3) runs and P2 holds the scoreboard
    // entry until P1 is done, at T = 1292. sm0 forwards M1..M4 to sm1 at
    // 8, .., 32 and M5 to sm2 at 40 (72 B matmul descriptors, 2 beats).
    // Stages after sm0's up bridge (S1), the port from sm0 (S2), the port
    // to the target (S3) and the target's down bridge (S4) hold one
    // message each. A beat waits for the next slot cycle: 15 mod 16 on a
    // bridge (+ 10 latency), 7 mod 8 on a port (whole slots by cycle).
    // * M1: [8,32) S1 at 42; [42,56); [56,72); [72,96) + 10: sm1's inbound
    //   entry at 106.
    // * M2: [56,80) S1 at 90; [90,104); [104,120); [120,144) + 10: S4 at
    //   154, waits for the inbound entry.
    // * M3: [104,128) S1 at 138; [138,152); [152,168): S3 at 168, waits for
    //   S4. M4: [152,176) S1 at 186; [186,200): S2 at 200, waits for S3.
    // * M5 (to sm2): [200,224) S1 at 234, waits for S2, which M4 holds.
    // At T M1 enters sm1's scoreboard and M2 its inbound table; M3 moves
    // [1292, 1312) (slots 1295 and 1311), M4 [1312, 1328), and M5
    // [1328, 1344) into S2, then [1344, 1360) and sm2's down bridge
    // [1360, 1392) + 10: sm2 at 1402. On a free path M5 would reach sm2 at
    // 40 + 98 = 138.
    let b = tiny_with(&[
        "root.gpc0.sm1.neutrino.scoreboard=1",
        "root.gpc0.sm1.neutrino.inbound_invokes=1",
    ]);
    let (sm0, sm1) = (b.nu("root.gpc0.sm0"), b.nu("root.gpc0.sm1"));
    let b2 = b.nu("root.gpc0.sm2");
    let base1 = b.memory("root.gpc0.sm1.smem").base;
    let base2 = b.memory("root.gpc0.sm2.smem").base;
    let small = 64 * 64 * 2;
    let big = nu_model::Range::from_len(base1, 128 * 128 * 2);
    let at = |i: u64| nu_model::Range::from_len(base1 + big.len() + i * small, small);
    let atlas1 = "root.gpc0.sm1.atlas";
    let p1 = matmul(&b, atlas1, big, 128, vec![]);
    let p2 = matmul(
        &b,
        atlas1,
        at(0),
        64,
        vec![nu_model::Dep::Versioned {
            addr: big.base,
            version: 1,
        }],
    );
    let mut ms: Vec<Descriptor> = (1..5)
        .map(|i| matmul(&b, atlas1, at(i), 64, vec![]).with_inferred_version(1))
        .collect();
    let r5 = nu_model::Range::from_len(base2, small);
    ms.push(matmul(&b, "root.gpc0.sm2.atlas", r5, 64, vec![]).with_inferred_version(1));
    let last = at(4);
    let mut sim = b.sim;
    sim.spawn(sm1, move |ctx: NuCtx| async move {
        ctx.invoke(p1).await;
        let h = ctx.invoke(p2).await;
        ctx.wait(h.addr, h.version).await;
    });
    sim.spawn(sm0, move |ctx: NuCtx| async move {
        for m in ms {
            ctx.invoke(m).await;
        }
        ctx.wait(r5.base, 1).await;
        ctx.wait(last.base, 1).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None, "{:?}", s.fatal);
    assert!(s.unfinished.is_empty());
    let t = s.tasks[2].invoke;
    assert_eq!(t, 1292, "P1 is still running when M5 reaches S1");
    let m5 = s.tasks.iter().find(|x| x.nu == b2).unwrap();
    assert_eq!(m5.invoke, 1402);
    let hol = |name: &str| {
        s.stages
            .iter()
            .find(|x| x.stage == name)
            .map_or(0, |x| x.hol_cycles)
    };
    // Head-of-line cycles on M5's path: S1 M5 [234, 1328); S2 M4
    // [200, 1312); sm0's source queue M2 [16,56), M3 [56,104), M4
    // [104,152), M5 [152,200). None after the port to sm2.
    assert_eq!(hol("root.gpc0.sm0.smcbus_to_clcbus.ab.invoke"), 1328 - 234);
    assert_eq!(
        hol("root.gpc0.clcbus.sm0.smcbus_to_clcbus.in.invoke"),
        1312 - 200
    );
    assert_eq!(hol("root.gpc0.sm0.source.invoke"), 40 + 48 + 48 + 48);
    assert_eq!(hol("root.gpc0.clcbus.sm2.smcbus_to_clcbus.out.invoke"), 0);
}
