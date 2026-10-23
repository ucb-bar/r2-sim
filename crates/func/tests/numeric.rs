use func::{bf16_to_f32, f32_to_bf16, Numeric};
use nu_model::{Addr, Range};
use sim_core::{FunctionalBackend, MemBlock, ResourceId};

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

fn numeric() -> Numeric {
    let mut n = Numeric::new();
    n.configure(
        &[MemBlock::plain(
            "m".into(),
            nu_model::NeutrinoId(0),
            Range::new(0x10_0000, 0x30_0000),
            ResourceId(0),
            0,
        )],
        &[Range::new(0x20_0000, 0x21_0000)],
        512,
    );
    n
}

#[test]
fn sparse_pages_cross_boundaries_and_read_zero() {
    let mut n = numeric();
    let data: Vec<u8> = (0..200u8).collect();
    // Crosses a 64 KiB page boundary.
    n.write(Addr(0x10_ffa0), &data).unwrap();
    assert_eq!(n.read(Addr(0x10_ffa0), 200).unwrap(), data);
    assert_eq!(n.read(Addr(0x18_0000), 4).unwrap(), vec![0; 4]);
    assert!(n.read(Addr(0x2f_fff0), 32).is_err(), "outside the block");
}

#[test]
fn writes_to_sync_only_are_ignored() {
    let mut n = numeric();
    n.write(Addr(0x20_0000), &[1, 2, 3]).unwrap();
    assert_eq!(n.read(Addr(0x20_0000), 3).unwrap(), vec![0; 3]);
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
        let mut n = numeric();
        let (pa, pa2, pb, pb2, pc) = (0x10_0000, 0x11_0000, 0x12_0000, 0x13_0000, 0x14_0000);
        if split {
            // Rows 0 and 2 (low halves of the two row tiles) at pa, 1 and 3
            // at pa2; columns 0 and 2 of B at pb, 1 and 3 at pb2.
            let rows = |r: &[usize]| -> Vec<f32> {
                r.iter()
                    .flat_map(|&i| a[2 * i..2 * i + 2].to_vec())
                    .collect()
            };
            n.write(Addr(pa), &bf(&rows(&[0, 2]))).unwrap();
            n.write(Addr(pa2), &bf(&rows(&[1, 3]))).unwrap();
            let cols = |c: &[usize]| -> Vec<f32> {
                (0..2)
                    .flat_map(|p| c.iter().map(move |&j| b[4 * p + j]))
                    .collect()
            };
            n.write(Addr(pb), &bf(&cols(&[0, 2]))).unwrap();
            n.write(Addr(pb2), &bf(&cols(&[1, 3]))).unwrap();
        } else {
            n.write(Addr(pa), &bf(a)).unwrap();
            n.write(Addr(pb), &bf(b)).unwrap();
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
            },
        };
        let op = FuncOp {
            seq: 1,
            time: 0,
            nu: nu_model::NeutrinoId(0),
            gid: None,
            kind: OpKind::Task(desc),
        };
        n.apply(&op).unwrap();
        n.read(Addr(pc), 32).unwrap()
    };
    let one = run(false);
    assert_eq!(
        &one[0..2],
        &f32_to_bf16(1.0 * 10.0 + 2.0 * 50.0).to_le_bytes()
    );
    assert_eq!(run(true), one);
}
