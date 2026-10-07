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
    // Hand computation (tiny.yaml; own co-charging; control lanes at 1/8
    // of the data rate: a 64 B beat takes 16 cycles on a 4 B/cycle SM
    // bridge lane and 8 on an 8 B/cycle crossbar port lane; each bridge
    // adds 10 cycles of latency; the crossbar clsbus is an xbar, charged per
    // port). Control messages go store and forward through one-message
    // stages (after sm0's up bridge, after the port from sm0, after the port
    // to the target, then the target's inbound table): a 1-beat message
    // between SMs takes 16 + 10 + 8 + 8 + 16 + 10 = 68 cycles, a 2-beat
    // matmul descriptor (88 B) 32 + 10 + 16 + 16 + 32 + 10 = 116. Each SM
    // has one output block, so its end signal is a pin of the matmul
    // (no join), and x has no producer in the window (its dep drops).
    // * The driver invokes 6 tasks at 8, 16, .., 48. The stage after sm0's
    //   up bridge holds one message until its transmission through the port
    //   from sm0 ends, so each forward starts its up-bridge transmission
    //   when the previous one leaves that stage: sm1 enters its tasks at
    //   76, 110, 192 (up bridge from 8, 42, 76), sm2 at 202, 236, 318 (from
    //   134, 168, 202).
    // * DMAs (Section 5.4, data-path rule): hops in route order from the
    //   engine latency 50, at the engine's 64 B beat narrowed by each hop.
    //   x to sm1 (8 KiB): sm0's SMEM port [126, 254) on one of its two
    //   lanes, sm0's bridge up (32 B/cycle) [126, 382), sm1's bridge down
    //   and SMEM port to 402. x to sm2: sm0's bridge is busy until 382,
    //   [382, 638); the backward pass moves its SMEM hop to [253, 381), one
    //   skid before: chunk [253, 658), end at sm2's SMEM port.
    // * W0 (64 KiB from HBM to sm1): HBM (64 B/cycle) [160, 1184), not held
    //   back by the slower bridge (no backpressure); sm1's bridge down after
    //   the x DMA, [392, 2440), its SMEM port to 2450. W1 (to sm2): HBM
    //   after W0, [1184, 2208); sm2's bridge [1184, 3232), SMEM to 3242.
    // * Matmuls (608 cycles on SMEM) after the weights: sm1 [2454, 3062),
    //   done 3066; sm2 [3246, 3854), done 3858.
    // * The driver waits on (out1, 1): Resolve reaches it at 3134; the
    //   Await for (out2, 1) registers at sm2; Resolve leaves at 3858 and
    //   arrives at 3926.
    // * sm0's source invoke queue: the head waits from 16 to 42, 42 to 76,
    //   76 to 134, 134 to 168 and 168 to 202 (186 cycles).
    assert_eq!(end, 3926);
    assert_eq!(
        (s.tasks[3].chunks[0].start, s.tasks[3].chunks[0].end),
        (253, 658),
        "sm2's x DMA waits for sm0's bridge behind sm1's"
    );
    assert_eq!(
        s.tasks[1].chunks[0],
        sim_core::ChunkRec {
            start: 160,
            end: 2450,
            culprit: s.tasks[1].chunks[0].culprit
        }
    );
    assert_eq!(
        s.tasks[4].chunks[0].start, 1184,
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
    assert_eq!(hol("root.gpc0.sm0.source.invoke"), 186);
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
    // Critical path (by hand; a 1-beat message between SMs takes 68
    // cycles, see one_linear_layer; DMAs by the data-path rule, hops in
    // route order at the engine's 64 B beat): rmsnorm [18, 274); router
    // [278, 351); the load of its output [366, 367); the MoE window's
    // forwards from 375. HBM (64 B/cycle) is not held back by the 32
    // B/cycle SM bridges, so it serves the weight DMAs in invoke order,
    // 1024 cycles each: expert 0's gate [527, 1551) and up [1551, 2575),
    // expert 1's gate [2575, 3599) and up [3599, 4623). Expert 1 on sm2 (2
    // tokens): its gate weight reaches SMEM at 4633; the gate matmul needs
    // both SMEM lanes and waits for the up weight's DMA, which holds one
    // lane at 32 B/cycle until 6681: [6681, 7205); up matmul [7205, 7729);
    // down weight from 7259 (latency 50 after the gate matmul's
    // completion; its SMEM hop after the up matmul, [7729, 9777)), done
    // 9781. The vector op is bound by its Muon, so its SMEM demand takes
    // one of the two lanes and runs beside that DMA, [7743, 7775). Down
    // matmul (it reads the down weight) 2 x 264 cycles [9781, 10309), done
    // 10313. Home depends on the down matmul's output directly (a pin, no
    // end join): Resolve reaches home at 10381; pull [10431, 10483); add
    // [10497, 10521), done 10525; the driver resumes at 10525.
    assert_eq!(r.step_end[0], 10525);
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
        inferred_version: None,
        deps,
        label: None,
        args: Args::Matmul {
            a: nu_model::Addr(c.base),
            b: nu_model::Addr(c.base),
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
    // Head-of-line blocking (Section 3.3): sm1 has one scoreboard entry and
    // one inbound invoke entry; P1 (128^3) runs and P2 holds the scoreboard
    // entry until P1 is done, at T. sm0 forwards M1..M4 to sm1 at 8, .., 32
    // and M5 to sm2 at 40 (72 B matmul descriptors, 2 beats). Stages after
    // sm0's up bridge (S1), the port from sm0 (S2), the port to the target
    // (S3) and the target's down bridge (S4) hold one message each; the
    // transmissions take 32 (+ 10), 16, 16 and 32 (+ 10) cycles.
    // * M1: [8,40) S1 at 50; [50,66); [66,82); [82,114) + 10: sm1's inbound
    //   entry at 124.
    // * M2: [66,98) S1 at 108; [108,124); [124,140); [140,172) + 10: S4 at
    //   182, waits for the inbound entry.
    // * M3: [124,156) S1 at 166; [166,182); [182,198): S3 at 198, waits for
    //   S4. M4: [182,214) S1 at 224; [224,240): S2 at 240, waits for S3.
    // * M5 (to sm2): [240,272) S1 at 282, waits for S2, which M4 holds.
    // At T M1 enters sm1's scoreboard and M2 its inbound table; M3 moves
    // [T, T+32), M4 [T+32, T+48), and M5 [T+48, T+64) into S2, then
    // [T+64, T+80) and sm2's down bridge [T+80, T+112) + 10: sm2 at T+122.
    // On a free path M5 would reach sm2 at 40 + 116 = 156.
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
    assert!(t > 282, "P1 is still running when M5 reaches S1");
    let m5 = s.tasks.iter().find(|x| x.nu == b2).unwrap();
    assert_eq!(m5.invoke, t + 122);
    let hol = |name: &str| {
        s.stages
            .iter()
            .find(|x| x.stage == name)
            .map_or(0, |x| x.hol_cycles)
    };
    // Head-of-line cycles on M5's path: S1 M5 [282, T+48); S2 M4
    // [240, T+32); sm0's source queue M2 [16,66), M3 [66,124), M4
    // [124,182), M5 [182,240). None after the port to sm2.
    assert_eq!(
        hol("root.gpc0.sm0.smcbus_to_clcbus.ab.invoke"),
        t + 48 - 282
    );
    assert_eq!(
        hol("root.gpc0.clcbus.sm0.smcbus_to_clcbus.in.invoke"),
        t + 32 - 240
    );
    assert_eq!(hol("root.gpc0.sm0.source.invoke"), 50 + 58 + 58 + 58);
    assert_eq!(hol("root.gpc0.clcbus.sm2.smcbus_to_clcbus.out.invoke"), 0);
}
