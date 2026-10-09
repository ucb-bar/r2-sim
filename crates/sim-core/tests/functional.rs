//! The functional plane with the Numeric backend: functional check mode,
//! the Option 2 sync protocol, stores, races, and worker errors.

mod common;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use common::{node, smem, v1, Knobs, DRAM, SMEM};
use func::{bf16_to_f32, f32_to_bf16, Numeric};
use nu_model::{Addr, Range};
use sim_core::{
    Args, Descriptor, FatalError, FuncError, FuncOp, FunctionalBackend, MemBlock, NuCtx, OpKind,
    PeConfig, PeParams, RaceCheck, Ratio, SimWarning,
};

const E: u64 = 64;
const MAT: u64 = E * E * 2;

/// Deterministic bf16 values in [-2, 2).
fn matrix(seed: u32) -> Vec<u16> {
    let mut x = seed;
    (0..E * E)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32_to_bf16(((x >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0)
        })
        .collect()
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Host reference with the backend's f32 accumulation order.
fn reference(a: &[u16], b: &[u16]) -> Vec<u8> {
    let n = E as usize;
    let mut c = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..n {
                acc += bf16_to_f32(a[i * n + p]) * bf16_to_f32(b[p * n + j]);
            }
            c.push(f32_to_bf16(acc));
        }
    }
    bytes(&c)
}

/// The matmul example with A and B preloaded in DRAM; returns the end time,
/// the loaded C, C read from the final backend state, and the reference.
///
/// The consumer is invoked first: its deps (A, 1) and (B, 1) wait on
/// pending entries that the DMAs adopt.
fn matmul_numeric(zero_time: bool) -> (u64, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (am, bm) = (matrix(1), matrix(2));
    let n = node(Knobs::default());
    let (a, b, c) = (smem(0, MAT), smem(MAT, MAT), smem(2 * MAT, MAT));
    let (da, db) = (n.dma(0, 0, MAT, MAT), n.dma(MAT, MAT, MAT, MAT));
    let mm = n.matmul(c, vec![v1(a), v1(b)], E, E);
    let mut sim = n.sim;
    sim.set_functional(Box::new(Numeric::new()));
    sim.set_zero_time(zero_time);
    sim.init_memory(Addr(DRAM), bytes(&am));
    sim.init_memory(Addr(DRAM + MAT), bytes(&bm));
    let out = Rc::new(RefCell::new(Vec::new()));
    let out2 = out.clone();
    sim.spawn(n.nu, move |ctx: NuCtx| async move {
        ctx.invoke(mm).await;
        ctx.invoke(da).await;
        ctx.invoke(db).await;
        ctx.wait(c.base, 1).await;
        *out2.borrow_mut() = ctx.load(Addr(c.base), MAT).await;
    });
    let (s, backend) = sim.run_with_backend();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    assert!(s.open_pending.is_empty());
    let final_c = backend.unwrap().read(Addr(c.base), MAT).unwrap();
    let loaded = out.borrow().clone();
    (s.end_time, loaded, final_c, reference(&am, &bm))
}

#[test]
fn zero_time_functional_check_is_bit_exact() {
    let (end, loaded, final_c, want) = matmul_numeric(true);
    assert_eq!(end, 0);
    assert_eq!(loaded, want);
    assert_eq!(final_c, want);
}

#[test]
fn timing_mode_with_numeric_gives_the_same_c() {
    let (end, loaded, final_c, want) = matmul_numeric(false);
    // Scalar: invoke C [0,8), A [8,16), B [16,24), wait [24,26). DMA A at
    // 16, hops in route order from 116: DRAM port [116,244) at 64 B/cycle,
    // the link (32 B/cycle) [116,372), the SMEM port narrowed to 32
    // [116,372); complete 376. DMA B at 24: DRAM port [244,372), link busy
    // until 372, [372,628), SMEM port [372,628); complete 632. Matmul
    // [632,824), complete 828. Load: 1 cycle issue cost, 10 cycles smem
    // latency, then 8 KiB on the port at the Neutrino's 64 B beat: 128
    // cycles.
    assert_eq!(end, 828 + 1 + 10 + 128);
    assert_eq!(loaded, want);
    assert_eq!(final_c, want);
}

#[test]
fn store_then_load_round_trips_through_the_worker() {
    let n = node(Knobs::default());
    let mut sim = n.sim;
    sim.set_functional(Box::new(Numeric::new()));
    let data: Vec<u8> = (0..=255u8).collect();
    let got = Rc::new(RefCell::new(Vec::new()));
    let got2 = got.clone();
    let d2 = data.clone();
    sim.spawn(n.nu, move |ctx: NuCtx| async move {
        ctx.store(Addr(SMEM + 4096), d2).await;
        *got2.borrow_mut() = ctx.load(Addr(SMEM + 4096), 256).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert_eq!(*got.borrow(), data);
    assert_eq!(s.func_ops, 1);
    assert_eq!(s.loads.len(), 1);
    assert_eq!(s.loads[0].seq_needed, 1, "the load waits for the store");
    assert!(s.loads[0].applied >= 1);
}

#[test]
fn load_with_the_null_backend_is_fatal() {
    let n = node(Knobs::default());
    let mut sim = n.sim;
    sim.spawn(n.nu, move |ctx: NuCtx| async move {
        ctx.load(Addr(SMEM), 64).await;
    });
    let s = sim.run();
    assert!(matches!(
        s.fatal.unwrap().error,
        FatalError::Func { seq: 0, .. }
    ));
}

fn racy(check: RaceCheck) -> sim_core::Stats {
    let n = node(Knobs::default());
    let x = smem(0, MAT);
    let (d1, d2) = (n.dma(0, 0, MAT, MAT), n.dma(MAT, 0, MAT, MAT));
    let mut sim = n.sim;
    sim.set_functional(Box::new(Numeric::new()));
    sim.set_race_check(check);
    sim.spawn(n.nu, move |ctx: NuCtx| async move {
        ctx.invoke(d1).await;
        ctx.wait(x.base, 1).await;
        // A second writer of X, not ordered against the load.
        ctx.invoke(d2).await;
        ctx.load(Addr(x.base), 64).await;
    });
    sim.run()
}

#[test]
fn racy_load_is_fatal_and_names_the_writer() {
    let s = racy(RaceCheck::Fatal);
    let f = s.fatal.expect("fatal");
    assert_eq!(f.gid, Some(1));
    assert_eq!(
        f.error,
        FatalError::Race {
            range: Range::from_len(SMEM, 64),
            writer: 1
        }
    );
}

#[test]
fn racy_load_can_be_a_warning() {
    let s = racy(RaceCheck::Warn);
    assert_eq!(s.fatal, None);
    assert!(s
        .warnings
        .iter()
        .any(|w| matches!(w.warning, SimWarning::Race { writer: 1, .. })));
    assert_eq!(s.loads.len(), 1);
}

/// Numeric, with a delay on every matmul op.
struct Slow(Numeric);

impl FunctionalBackend for Slow {
    fn configure(&mut self, m: &[MemBlock], s: &[Range]) {
        self.0.configure(m, s)
    }
    fn has_data(&self) -> bool {
        true
    }
    fn apply(&mut self, op: &FuncOp) -> Result<(), FuncError> {
        if let OpKind::Task(d) = &op.kind {
            if matches!(d.args, Args::Matmul { .. }) {
                std::thread::sleep(Duration::from_millis(30));
            }
        }
        self.0.apply(op)
    }
    fn read(&mut self, a: Addr, l: u64) -> Result<Vec<u8>, FuncError> {
        self.0.read(a, l)
    }
    fn write(&mut self, a: Addr, b: &[u8]) -> Result<(), FuncError> {
        self.0.write(a, b)
    }
}

#[test]
fn load_does_not_wait_for_unrelated_ops() {
    let n = node(Knobs::default());
    let x = smem(0, MAT);
    let dx = n.dma(0, 0, MAT, MAT);
    let mut mms: Vec<Descriptor> = (0..8)
        .map(|i| n.matmul(smem((4 + i) * MAT, MAT), vec![], E, E))
        .collect();
    let mut sim = n.sim;
    // A second tensor PE that runs 8 matmuls at once.
    let wide = sim.add_pe(
        n.nu,
        PeConfig {
            name: "atlas.wide".into(),
            max_inflight: 8,
            max_outstanding: 1,
            params: PeParams::Atlas {
                masters: Vec::new(),
                rates: Vec::new(),
                row_panel: false,
                acc_buffer: 0,
                array: n.array,
                smem_port: n.smem_port,
                smem_wport: None,
                utilization: Ratio::ONE,
                quantize: true,
            },
        },
    );
    for d in &mut mms {
        d.pe = Some(wide);
    }
    sim.set_functional(Box::new(Slow(Numeric::new())));
    let pattern: Vec<u8> = (0..MAT).map(|i| (i % 251) as u8).collect();
    sim.init_memory(Addr(DRAM), pattern.clone());
    let got = Rc::new(RefCell::new(Vec::new()));
    let got2 = got.clone();
    sim.spawn(n.nu, move |ctx: NuCtx| async move {
        ctx.invoke(dx).await;
        for d in mms {
            ctx.invoke(d).await;
        }
        ctx.wait(x.base, 1).await;
        *got2.borrow_mut() = ctx.load(Addr(x.base), MAT).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert_eq!(*got.borrow(), pattern);
    let l = s.loads[0];
    assert_eq!(l.seq_needed, 1, "X's producer is the first op");
    assert_eq!(l.emitted, 9);
    assert!(
        l.applied >= l.seq_needed && l.applied < l.emitted,
        "served at applied {} of {} emitted",
        l.applied,
        l.emitted
    );
}

#[test]
fn worker_error_stops_the_run_with_the_task() {
    let n = node(Knobs::default());
    let c = smem(2 * MAT, MAT);
    let mut mm = n.matmul(c, vec![], E, E);
    // A leading dimension that makes C overrun its dst.
    if let Args::Matmul { ldc, .. } = &mut mm.args {
        *ldc = 2 * E;
    }
    let mut sim = n.sim;
    sim.set_functional(Box::new(Numeric::new()));
    sim.spawn(n.nu, move |ctx: NuCtx| async move {
        ctx.invoke(mm).await;
        ctx.wait(c.base, 1).await;
        ctx.load(Addr(c.base), 64).await;
        unreachable!("the load reports the worker error");
    });
    let s = sim.run();
    let f = s.fatal.expect("fatal");
    assert_eq!(
        (f.gid, f.time),
        (Some(0), 8),
        "the op's own gid and issue time"
    );
    assert!(matches!(f.error, FatalError::Func { seq: 1, .. }));
}
