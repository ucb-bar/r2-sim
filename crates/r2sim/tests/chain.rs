//! A chained matmul end to end on the Numeric backend and on the Python
//! plane: C (128 x 128) = A (128 x 1152) B (1152 x 128) in bf16 as 9 tasks
//! of k = 128 on the Atlas of root.gpc0.sm0 (configs/2p-smem.yaml; A in
//! SMEM bank 0, B and C in bank 1), the first without `accumulate`, all
//! but the last with `keep`. The data are small positive integers whose
//! partial sums exceed 256: every fp32 sum is exact, and a rounding of a
//! partial to bf16 would change C. Both backends must give the exact
//! product rounded to bf16 once.

use std::path::Path;

use func::{bf16_to_f32, f32_to_bf16};
use nu_model::{Addr, Dep, Range};
use sim_core::{build_config_file, Descriptor, Dtype, FunctionalBackend, NuCtx, ProfileRegistry};

const M: usize = 128;
const N: usize = 128;
const K: usize = 1152;
const CHUNK: usize = 128;

fn a(i: usize, p: usize) -> i64 {
    ((i * 7 + p * 3) % 9) as i64 + 1
}

fn b(p: usize, j: usize) -> i64 {
    ((p * 5 + j * 11) % 9) as i64 + 1
}

fn bf16_bytes(v: impl Iterator<Item = i64>) -> Vec<u8> {
    v.flat_map(|x| f32_to_bf16(x as f32).to_le_bytes())
        .collect()
}

/// C read back after the chain on `backend`.
fn chain(backend: Box<dyn FunctionalBackend>) -> Vec<u16> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p-smem.yaml");
    let b_ = build_config_file(&path, &[] as &[&str], ProfileRegistry::with_builtins()).unwrap();
    let sm = b_.nu("root.gpc0.sm0");
    let atlas = b_.pe("root.gpc0.sm0.atlas");
    let smem = b_.memory("root.gpc0.sm0.smem");
    let (pa, pb) = (smem.base, smem.base + 0x80000);
    let c = Range::from_len(smem.base + 0xd0000, (M * N * 2) as u64);
    let mut sim = b_.sim;
    sim.set_functional(backend);
    sim.init_memory(Addr(pa), bf16_bytes((0..M * K).map(|x| a(x / K, x % K))));
    sim.init_memory(Addr(pb), bf16_bytes((0..K * N).map(|x| b(x / N, x % N))));
    sim.spawn(sm, move |ctx: NuCtx| async move {
        let chunks = K / CHUNK;
        let mut prev: Option<Dep> = None;
        for i in 0..chunks {
            let d = Descriptor::matmul(
                atlas,
                pa + (i * CHUNK * 2) as u64,
                pb + (i * CHUNK * N * 2) as u64,
                c,
                M as u64,
                N as u64,
                CHUNK as u64,
                Dtype::Bf16,
            )
            .with_ld(K as u64, N as u64, N as u64)
            .with_acc(i > 0, i + 1 < chunks)
            .with_deps(prev.into_iter().collect());
            prev = Some(ctx.invoke(d).await.key());
        }
        ctx.wait_key(prev.unwrap()).await;
    });
    let (stats, backend) = sim.run_with_backend();
    assert_eq!(stats.fatal, None);
    let bytes = backend.unwrap().read(Addr(c.base), c.len()).unwrap();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| u16::from_le_bytes(*x))
        .collect()
}

/// The exact product rounded to bf16 once, and the product with each
/// partial of 128 rounded to bf16 (what a chain that wrote C after every
/// task would give).
fn expected() -> (Vec<u16>, Vec<u16>) {
    let mut once = Vec::new();
    let mut each = Vec::new();
    for i in 0..M {
        for j in 0..N {
            let sum: i64 = (0..K).map(|p| a(i, p) * b(p, j)).sum();
            assert!(sum.abs() < 1 << 24, "an exact fp32 sum");
            once.push(f32_to_bf16(sum as f32));
            let mut c = 0.0f32;
            for q in 0..K / CHUNK {
                let part: i64 = (q * CHUNK..(q + 1) * CHUNK)
                    .map(|p| a(i, p) * b(p, j))
                    .sum();
                c = bf16_to_f32(f32_to_bf16(c + part as f32));
            }
            each.push(f32_to_bf16(c));
        }
    }
    (once, each)
}

#[test]
fn a_chain_rounds_c_to_bf16_once_on_both_backends() {
    let (once, each) = expected();
    // 2,814 of the 16,384 elements of C differ.
    let differ = once.iter().zip(&each).filter(|(x, y)| x != y).count();
    assert!(differ > 1000, "the data tell a per-task rounding: {differ}");
    assert_eq!(chain(Box::new(func::Numeric::new())), once, "numeric");
    let python = r2sim::spawn_process(None).expect("the Python plane");
    assert_eq!(chain(Box::new(python)), once, "python");
}
