use func::{bf16_to_f32, f32_to_bf16, Numeric};
use nu_model::{Addr, Range};
use sim_core::{FunctionalBackend, MemBlock, ResourceId, Store};

#[test]
fn bf16_rounds_to_nearest_even() {
    assert_eq!(f32_to_bf16(1.0), 0x3f80);
    assert_eq!(bf16_to_f32(0x3f80), 1.0);
    // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even (1.0).
    assert_eq!(f32_to_bf16(1.0 + 1.0 / 256.0), 0x3f80);
    // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: to even.
    assert_eq!(f32_to_bf16(1.0 + 3.0 / 256.0), 0x3f82);
    // Just above halfway rounds up.
    assert_eq!(f32_to_bf16(1.0 + 1.0 / 256.0 + 1.0 / 65536.0), 0x3f81);
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
}

/// The Numeric backend and a store of one block, [1 MiB, 3 MiB), with the
/// sync-only space [2 MiB, 2 MiB + 64 KiB).
fn numeric() -> (Numeric, Store) {
    let block = MemBlock::plain(
        "m".into(),
        nu_model::NeutrinoId(0),
        Range::new(0x10_0000, 0x30_0000),
        ResourceId(0),
        0,
    );
    let store = Store::new(&[block], &[Range::new(0x20_0000, 0x21_0000)], 512).unwrap();
    (Numeric::new(), store)
}

/// C = A x B for A 4 x 2 and B 2 x 4 (tile 2 x 2): once with each operand
/// one matrix, once with A split by rows and B split by columns into half
/// matrices (Args::Matmul): the same C.
#[test]
fn a_split_operand_reads_its_halves() {
    use sim_core::{Args, Descriptor, Dtype, FuncOp, OpKind, OpcodeId};
    let bf = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|&x| f32_to_bf16(x).to_le_bytes())
            .collect()
    };
    let a: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let b: Vec<f32> = (1..=8).map(|x| (x * 10) as f32).collect();
    let (a, b) = (&a, &b);
    let run = |split: bool| -> Vec<u8> {
        let (mut n, mut m) = numeric();
        let (pa, pa2, pb, pb2, pc) = (0x10_0000, 0x11_0000, 0x12_0000, 0x13_0000, 0x14_0000);
        if split {
            // Rows 0 and 2 (low halves of the two row tiles) at pa, 1 and 3
            // at pa2; columns 0 and 2 of B at pb, 1 and 3 at pb2.
            let rows = |r: &[usize]| -> Vec<f32> {
                r.iter()
                    .flat_map(|&i| a[2 * i..2 * i + 2].to_vec())
                    .collect()
            };
            m.write(pa, &bf(&rows(&[0, 2]))).unwrap();
            m.write(pa2, &bf(&rows(&[1, 3]))).unwrap();
            let cols = |c: &[usize]| -> Vec<f32> {
                (0..2)
                    .flat_map(|p| c.iter().map(move |&j| b[4 * p + j]))
                    .collect()
            };
            m.write(pb, &bf(&cols(&[0, 2]))).unwrap();
            m.write(pb2, &bf(&cols(&[1, 3]))).unwrap();
        } else {
            m.write(pa, &bf(a)).unwrap();
            m.write(pb, &bf(b)).unwrap();
        }
        let desc = Descriptor {
            opcode: OpcodeId(2),
            pe: None,
            dst: Range::from_len(pc, 32),
            versioned: false,
            untracked: false,
            dst_mask: nu_model::FULL_MASK,
            inferred_version: None,
            pc: None,
            call_path: Vec::new(),
            bytes: Vec::new(),
            deps: vec![],
            label: None,
            args: Args::Matmul {
                a: Addr(pa),
                b: Addr(pb),
                a2: split.then_some(Addr(pa2)),
                b2: split.then_some(Addr(pb2)),
                lda: 2,
                ldb: if split { 2 } else { 4 },
                ldc: 4,
                m: 4,
                n: 4,
                k: 2,
                in_dtype: Dtype::Bf16,
                b_dtype: Dtype::Bf16,
                out_dtype: Dtype::Bf16,
                accumulate: false,
                keep: false,
                tile: [2, 2],
                transpose_b: false,
            },
        };
        let op = FuncOp {
            seq: 1,
            time: 0,
            nu: nu_model::NeutrinoId(0),
            gid: None,
            backend: 0,
            kind: OpKind::Task(desc),
        };
        n.apply(&op, &mut m).unwrap();
        m.read(pc, 32).unwrap()
    };
    let one = run(false);
    assert_eq!(
        &one[0..2],
        &f32_to_bf16(1.0 * 10.0 + 2.0 * 50.0).to_le_bytes()
    );
    assert_eq!(run(true), one);
}

/// A bf16 matmul op on the Atlas `pe`: C (m x n at `c`) = A (m x k at `a`)
/// B (k x n at `b`), row-major.
#[allow(clippy::too_many_arguments)]
fn matmul(
    pe: u32,
    a: u64,
    b: u64,
    c: u64,
    (m, n, k): (u64, u64, u64),
    accumulate: bool,
    keep: bool,
) -> sim_core::FuncOp {
    use sim_core::{Descriptor, Dtype, FuncOp, OpKind, PeId};
    let desc = Descriptor::matmul(
        PeId(pe),
        a,
        b,
        Range::from_len(c, m * n * 2),
        m,
        n,
        k,
        Dtype::Bf16,
    )
    .with_acc(accumulate, keep);
    FuncOp {
        seq: 1,
        time: 0,
        nu: nu_model::NeutrinoId(0),
        gid: None,
        backend: 0,
        kind: OpKind::Task(desc),
    }
}

fn put_bf16(n: &mut Store, addr: u64, v: &[f32]) {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|&x| f32_to_bf16(x).to_le_bytes())
        .collect();
    n.write(addr, &bytes).unwrap();
}

fn get_bf16(n: &mut Store, addr: u64, len: usize) -> Vec<u16> {
    let bytes = n.read(addr, 2 * len as u64).unwrap();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| u16::from_le_bytes(*x))
        .collect()
}

#[test]
fn the_accumulator_keeps_the_partial_in_fp32() {
    // A chain of three tasks on one C: the first two keep C in the
    // accumulator (memory unchanged), the last writes C once in bf16, so C
    // is rounded to bf16 once, not after each task. A fourth task that
    // accumulates starts from C in memory: the last task dropped the
    // partial.
    let (m, n, k) = (16, 16, 16);
    let (pa, pb, pc) = (0x10_0000, 0x11_0000, 0x12_0000);
    let a: Vec<f32> = (0..m * k).map(|i| 1.0 + (i % 7) as f32 / 8.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| 1.0 + (i % 5) as f32 / 8.0).collect();
    let (mut nb, mut mem) = numeric();
    put_bf16(&mut mem, pa, &a);
    put_bf16(&mut mem, pb, &b);
    let dims = (m as u64, n as u64, k as u64);
    nb.apply(&matmul(0, pa, pb, pc, dims, false, true), &mut mem)
        .unwrap();
    nb.apply(&matmul(0, pa, pb, pc, dims, true, true), &mut mem)
        .unwrap();
    assert_eq!(
        get_bf16(&mut mem, pc, m * n),
        vec![0; m * n],
        "C stays in the accumulator"
    );
    nb.apply(&matmul(0, pa, pb, pc, dims, true, false), &mut mem)
        .unwrap();
    // The product is exact in fp32 (multiples of 1/64, at most 16 * 4).
    let p: Vec<f32> = (0..m * n)
        .map(|x| (0..k).map(|q| a[x / n * k + q] * b[q * n + x % n]).sum())
        .collect();
    let once: Vec<u16> = p.iter().map(|&x| f32_to_bf16(x + x + x)).collect();
    let r = |x: f32| bf16_to_f32(f32_to_bf16(x));
    let each: Vec<u16> = p.iter().map(|&x| f32_to_bf16(r(r(x) + x) + x)).collect();
    assert_ne!(once, each, "the data tell a rounding of the partial");
    assert_eq!(get_bf16(&mut mem, pc, m * n), once);
    nb.apply(&matmul(0, pa, pb, pc, dims, true, false), &mut mem)
        .unwrap();
    let from_memory: Vec<u16> = once
        .iter()
        .zip(&p)
        .map(|(&c, &x)| f32_to_bf16(bf16_to_f32(c) + x))
        .collect();
    assert_eq!(get_bf16(&mut mem, pc, m * n), from_memory);
}

#[test]
fn the_accumulator_is_per_atlas() {
    // A partial kept on PE 0 is not seen by an accumulating task on PE 1:
    // PE 1 starts from C in memory (zero).
    let (m, n, k) = (8, 8, 8);
    let (pa, pb, pc) = (0x10_0000, 0x11_0000, 0x12_0000);
    let (mut nb, mut mem) = numeric();
    put_bf16(&mut mem, pa, &[1.0; 64]);
    put_bf16(&mut mem, pb, &[1.0; 64]);
    nb.apply(&matmul(0, pa, pb, pc, (m, n, k), false, true), &mut mem)
        .unwrap();
    nb.apply(&matmul(1, pa, pb, pc, (m, n, k), true, false), &mut mem)
        .unwrap();
    assert_eq!(get_bf16(&mut mem, pc, 64), vec![f32_to_bf16(8.0); 64]);
}

#[test]
fn a_matmul_with_b_transposed() {
    // B is stored as n x k (leading dimension k = 72) and the task uses
    // its transpose. Integer data: the fp32 sums are exact.
    let (m, n, k) = (16u64, 32u64, 72u64);
    let (pa, pb, pc) = (0x10_0000, 0x11_0000, 0x12_0000);
    let a: Vec<f32> = (0..m * k)
        .map(|x| ((x / k * 3 + x % k) % 5) as f32)
        .collect();
    let bt: Vec<f32> = (0..n * k)
        .map(|x| ((x / k * 7 + x % k * 2) % 7) as f32)
        .collect();
    let (mut nb, mut mem) = numeric();
    put_bf16(&mut mem, pa, &a);
    put_bf16(&mut mem, pb, &bt);
    let mut op = matmul(0, pa, pb, pc, (m, n, k), false, false);
    if let sim_core::OpKind::Task(d) = &mut op.kind {
        if let sim_core::Args::Matmul {
            transpose_b, ldb, ..
        } = &mut d.args
        {
            (*transpose_b, *ldb) = (true, k);
        }
    }
    nb.apply(&op, &mut mem).unwrap();
    let want: Vec<u16> = (0..m * n)
        .map(|x| {
            let (i, j) = (x / n, x % n);
            f32_to_bf16(
                (0..k)
                    .map(|p| a[(i * k + p) as usize] * bt[(j * k + p) as usize])
                    .sum(),
            )
        })
        .collect();
    assert_eq!(get_bf16(&mut mem, pc, (m * n) as usize), want);
}

/// A Muon task with kernel `kernel` on `backend`.
fn muon(
    backend: &mut dyn FunctionalBackend,
    mem: &mut Store,
    kernel: u32,
) -> Result<(), sim_core::FuncError> {
    use sim_core::{Descriptor, FuncOp, OpKind, PeId};
    let mut d = Descriptor::vector(PeId(0), Range::from_len(0x10_0000, 512), 1, 64, 64);
    if let sim_core::Args::Vector {
        kernel: k, block, ..
    } = &mut d.args
    {
        (*k, *block) = (kernel, vec![1, 2, 3]);
    }
    let op = FuncOp {
        seq: 1,
        time: 0,
        nu: nu_model::NeutrinoId(0),
        gid: None,
        backend: 0,
        kind: OpKind::Task(d),
    };
    backend.apply(&op, mem)
}

#[test]
fn a_muon_kernel_is_an_error_on_numeric_and_writes_nothing_on_trace() {
    let (mut nb, mut mem) = numeric();
    muon(&mut nb, &mut mem, 0).unwrap();
    let e = muon(&mut nb, &mut mem, 7).unwrap_err();
    assert_eq!(
        e.0,
        "Muon kernel 7: the numeric backend has no Muon kernels"
    );
    let mut t = func::Trace::new();
    muon(&mut t, &mut mem, 0).unwrap();
    muon(&mut t, &mut mem, 7).unwrap();
    assert_eq!(mem.read(0x10_0000, 512).unwrap(), vec![0; 512]);
}
