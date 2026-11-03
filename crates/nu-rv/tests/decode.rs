//! The descriptor decoder against the layout of ABI version 3 (a 24-byte
//! header, the keys, the arguments of the opcode), on configs/2p-smem.yaml,
//! and the four tasks of the fixture probe2.

mod common;

use common::{two_p, vector_args, words, Desc};
use nu_model::{Addr, Dep, NeutrinoId, Range};
use nu_rv::{decode, Abi, Decoded};
use sim_core::{Args, Dtype, OpcodeId, Rows};

const AT: usize = 0x100;
const PC: u64 = 0x40;

/// Decodes `d` from offset [`AT`] of the root's data scratch.
fn dec(abi: &Abi, d: &Desc) -> Result<Decoded, String> {
    dec_with(abi, d, &[])
}

/// [`dec`] with `bytes` at offset `BLOCK` of the data scratch.
fn dec_with(abi: &Abi, d: &Desc, bytes: &[u8]) -> Result<Decoded, String> {
    let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
    let mut s = vec![0u8; at.len() as usize];
    let b = d.bytes();
    s[AT..AT + b.len()].copy_from_slice(&b);
    s[BLOCK..BLOCK + bytes.len()].copy_from_slice(bytes);
    decode(&s, at, at.base + AT as u64, NeutrinoId(0), PC, abi)
}

/// The offset of an argument block in the root's data scratch.
const BLOCK: usize = 0x200;

fn task(r: Result<Decoded, String>) -> sim_core::Descriptor {
    match r {
        Ok(Decoded::Task(d)) => d,
        r => panic!("not a task: {r:?}"),
    }
}

/// sm0 of gpc0: its prefix, and a PE index by name.
fn sm0(b: &sim_core::Built, abi: &Abi, pe: &str) -> (u64, u16) {
    let nu = b.nu("root.gpc0.sm0");
    let id = b.pe(&format!("root.gpc0.sm0.{pe}"));
    let i = abi.node(nu).pes.iter().position(|&p| p == id).unwrap();
    (abi.node(nu).own.base, i as u16)
}

#[test]
fn the_pe_order_is_by_increasing_pe_id() {
    let b = two_p();
    let abi = Abi::new(&b);
    let nu = b.nu("root.gpc0.sm0");
    let pes = &abi.node(nu).pes;
    assert!(pes.windows(2).all(|w| w[0].0 < w[1].0));
    assert_eq!(
        pes.len(),
        b.pe_paths()
            .iter()
            .filter(|(p, _)| p.starts_with("root.gpc0.sm0."))
            .count()
    );
}

#[test]
fn the_key_bits_are_the_address_bits_less_the_granule_shift() {
    // 2p-smem.yaml: the highest own space is gpc1's sm3, (2 << 36) | (4 << 33)
    // plus 8 GiB: 38 bits; granule 512 B.
    let abi = Abi::new(&two_p());
    assert_eq!((abi.granule_shift, abi.key_bits), (9, 29));
    assert_eq!(
        abi.dep((3 << 29) | (0x10_0000_0000 >> 9)),
        Dep::Versioned {
            addr: 0x10_0000_0000,
            version: 3
        }
    );
    assert_eq!(
        abi.dep(0x10_0000_0000 >> 9),
        Dep::Unversioned(0x10_0000_0000)
    );
}

#[test]
fn dma() {
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "dma");
    let key = (1u64 << abi.key_bits) | (0x10_0000_0000 >> 9);
    let d = task(dec(
        &abi,
        &common::dma(pe, 0x10_0000_0000, sm, 0x8000, vec![key]),
    ));
    assert_eq!(d.opcode, OpcodeId::DMA);
    assert_eq!(d.pe, Some(b.pe("root.gpc0.sm0.dma")));
    assert_eq!(d.dst, Range::from_len(sm, 0x8000));
    assert_eq!(
        (d.versioned, d.untracked, d.dst_mask),
        (true, false, u32::MAX)
    );
    assert_eq!(
        d.deps,
        vec![Dep::Versioned {
            addr: 0x10_0000_0000,
            version: 1
        }]
    );
    assert_eq!(
        d.args,
        Args::Dma {
            src: Addr(0x10_0000_0000),
            dst: Addr(sm),
            bytes: 0x8000,
            rows: None,
        }
    );
    assert_eq!(d.pc, Some(PC));
}

/// The arguments of a matmul (ABI version 3, 40 bytes): a, b, the
/// offsets a2 and b2, lda ldb ldc m n k, types and flags.
fn mm_args(
    a: u64,
    b: u64,
    (a2, b2): (i32, i32),
    dims: [u16; 6],
    types: u16,
    flags: u16,
) -> Vec<u8> {
    let mut v = words(&[a, b]);
    v.extend(a2.to_le_bytes());
    v.extend(b2.to_le_bytes());
    for d in dims {
        v.extend(d.to_le_bytes());
    }
    v.extend(types.to_le_bytes());
    v.extend(flags.to_le_bytes());
    v
}

#[test]
fn matmul() {
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "atlas");
    // A split A (its high half 0x20000 above it), lda ldb ldc m n k = 128
    // 64 64 128 64 128, types f32 A and B with bf16 C, accumulate and keep.
    let args = mm_args(
        sm,
        sm + 0x8000,
        (0x20000, 0),
        [128, 64, 64, 128, 64, 128],
        1 | 1 << 4,
        3,
    );
    assert_eq!(args.len(), 40);
    let d = Desc {
        opcode: 3,
        pe,
        flags: 1,
        base: (sm + 0x10000) >> 9,
        bound: 64,
        mask: 0x3e,
        deps: vec![],
        args,
    };
    let d = task(dec(&abi, &d));
    assert_eq!(d.opcode, OpcodeId::MATMUL);
    assert_eq!(d.pe, Some(b.pe("root.gpc0.sm0.atlas")));
    assert_eq!(d.dst, Range::from_len(sm + 0x10000, 64 << 9));
    assert_eq!(d.dst_mask, 0x3e);
    assert_eq!(
        d.args,
        Args::Matmul {
            a: Addr(sm),
            b: Addr(sm + 0x8000),
            a2: Some(Addr(sm + 0x20000)),
            b2: None,
            lda: 128,
            ldb: 64,
            ldc: 64,
            m: 128,
            n: 64,
            k: 128,
            in_dtype: Dtype::F32,
            b_dtype: Dtype::F32,
            out_dtype: Dtype::Bf16,
            accumulate: true,
            keep: true,
            transpose_b: false,
        }
    );
    // A negative offset: the high half below the low one; an unknown type.
    let mut e = Desc {
        opcode: 3,
        pe,
        flags: 1,
        base: (sm + 0x10000) >> 9,
        bound: 64,
        mask: u32::MAX,
        deps: vec![],
        args: mm_args(sm + 0x8000, sm, (-0x8000, 0x100), [64; 6], 0, 0),
    };
    let Args::Matmul { a2, b2, .. } = task(dec(&abi, &e)).args else {
        panic!()
    };
    assert_eq!((a2, b2), (Some(Addr(sm)), Some(Addr(sm + 0x100))));
    e.args = mm_args(sm, sm, (0, 0), [64; 6], 5 << 8, 0);
    let err = dec(&abi, &e).unwrap_err();
    assert!(err.contains("unknown matmul element type 5"), "{err}");
}

#[test]
fn vector_untracked_on_the_invoker() {
    // An untracked task runs at the invoking node: its PE index is the
    // invoker's (the root has none, so index 0 is past its PEs); flags
    // bit 1. The opaque kernel with its 24-byte block.
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "muon");
    let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
    let block = sim_core::kernels::opaque_block(0, 100, 512, 256);
    let mut d = Desc {
        opcode: 4,
        pe,
        flags: 0,
        base: sm >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args: vector_args(0, 24, at.base + BLOCK as u64),
    };
    let t = task(dec_with(&abi, &d, &block));
    assert_eq!(t.opcode, OpcodeId::VECTOR);
    assert_eq!(t.pe, Some(b.pe("root.gpc0.sm0.muon")));
    assert!(!t.versioned);
    assert_eq!(
        t.args,
        Args::Vector {
            kernel: 0,
            block: block.clone()
        }
    );
    d.flags = 2;
    let e = dec_with(&abi, &d, &block).unwrap_err();
    assert!(e.contains("PE") && e.contains("root,"), "{e}");
}

#[test]
fn join() {
    let abi = Abi::new(&two_p());
    let d = Desc {
        opcode: 0,
        pe: 7,
        flags: 0,
        base: 0x1_F000_0000 >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args: vec![],
    };
    let t = task(dec(&abi, &d));
    assert_eq!((t.opcode, t.pe, t.args), (OpcodeId::JOIN, None, Args::Join));
}

#[test]
fn spawn_and_imem_copy() {
    let abi = Abi::new(&two_p());
    let done = 0x1_F000_0000_u64;
    let mut args = Vec::new();
    args.extend(1u16.to_le_bytes());
    // The epoch of the started program.
    args.extend(3u16.to_le_bytes());
    args.extend(0x2au32.to_le_bytes());
    args.extend(0x1_F100_0100_u64.to_le_bytes());
    let d = Desc {
        opcode: 1,
        pe: 0,
        flags: 0,
        base: done >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args,
    };
    assert_eq!(
        dec(&abi, &d).unwrap(),
        Decoded::Spawn {
            done: Range::from_len(done, 512),
            deps: vec![],
            child: 1,
            epoch: 3,
            entry: 0x2a,
            args: 0x1_F100_0100
        }
    );
    let mut args = words(&[0x200]);
    args.extend(2u16.to_le_bytes());
    args.extend(0u16.to_le_bytes());
    args.extend(0x40u32.to_le_bytes());
    args.extend(0x180u64.to_le_bytes());
    let mut d = Desc {
        opcode: 5,
        pe: 0xffff,
        flags: 0,
        base: done >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args,
    };
    assert_eq!(
        dec(&abi, &d).unwrap(),
        Decoded::ImemCopy {
            record: Range::from_len(done, 512),
            deps: vec![],
            child: 2,
            src: 0x200,
            offset: 0x40,
            bytes: 0x180
        }
    );
    d.pe = 0;
    assert!(dec(&abi, &d).unwrap_err().contains("instruction DMA"));
}

#[test]
fn the_errors_of_the_decoder() {
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, _) = sm0(&b, &abi, "dma");
    // An unknown opcode.
    let mut d = common::dma(0, 0, sm, 512, vec![]);
    d.opcode = 6;
    assert!(dec(&abi, &d).unwrap_err().contains("unknown opcode 6"));
    // A PE index past the owner's PEs.
    let d = common::dma(40, 0, sm, 512, vec![]);
    let e = dec(&abi, &d).unwrap_err();
    assert!(e.contains("PE 40 of root.gpc0.sm0"), "{e}");
    // A descriptor that runs past the data scratch: its keys and
    // arguments end beyond it.
    let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
    let s = vec![0u8; at.len() as usize];
    let mut s2 = s.clone();
    let end = at.len() as usize - 24;
    s2[end..end + 8].copy_from_slice(&[2, 0, 0, 0, 0, 0, 0, 0]);
    let e = decode(&s2, at, at.base + end as u64, NeutrinoId(0), PC, &abi).unwrap_err();
    assert!(e.contains("outside the own data scratch"), "{e}");
    // A descriptor address outside the data scratch.
    let e = decode(&s, at, 0x10_0000_0000, NeutrinoId(0), PC, &abi).unwrap_err();
    assert!(e.contains("outside the own data scratch"), "{e}");
}

#[test]
fn a_2d_dma_has_its_rows_and_strides() {
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "dma");
    let mut d = common::dma(pe, 0x10_0000_0000, sm, 0x8000, vec![]);
    d.args = common::dma_args(0x10_0000_0000, sm, 144, 128, 256, 144);
    let t = task(dec(&abi, &d));
    assert_eq!(
        t.args,
        Args::Dma {
            src: Addr(0x10_0000_0000),
            dst: Addr(sm),
            bytes: 128 * 144,
            rows: Some(Rows {
                count: 128,
                bytes: 144,
                src_stride: 256,
                dst_stride: 144,
            }),
        }
    );
    // The descriptor's bytes as read: the header and the 32 bytes of the
    // arguments.
    assert_eq!(t.bytes, d.bytes());
    assert_eq!(t.bytes.len(), 24 + 32);
    // Rows 0 is an error.
    d.args = common::dma_args(0x10_0000_0000, sm, 144, 0, 256, 144);
    let e = dec(&abi, &d).unwrap_err();
    assert!(
        e.contains("a DMA of 0 rows (rows must be 1 or more)"),
        "{e}"
    );
}

#[test]
fn a_matmul_with_b_transposed() {
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "atlas");
    // 128 x 128 x 72 bf16, B stored n x k (ldb 72).
    let args = |b2: i32| mm_args(sm, sm + 0x8000, (0, b2), [72, 72, 128, 128, 128, 72], 0, 4);
    let mut d = Desc {
        opcode: 3,
        pe,
        flags: 1,
        base: (sm + 0x10000) >> 9,
        bound: 64,
        mask: u32::MAX,
        deps: vec![],
        args: args(0),
    };
    let t = task(dec(&abi, &d));
    let Args::Matmul {
        transpose_b,
        ldb,
        k,
        ..
    } = t.args
    else {
        panic!("{:?}", t.args);
    };
    assert_eq!((transpose_b, ldb, k), (true, 72, 72));
    // Not with a split B.
    d.args = args(0x40000);
    let e = dec(&abi, &d).unwrap_err();
    assert!(e.contains("a matmul with transpose_b and b2"), "{e}");
}

/// A versioned vector task on sm0's Muon with `kernel`, `len` and the block
/// at `at`.
fn vector(abi: &Abi, b: &sim_core::Built, kernel: u16, len: u16, at: u64) -> Desc {
    let (sm, pe) = sm0(b, abi, "muon");
    Desc {
        opcode: 4,
        pe,
        flags: 1,
        base: sm >> 9,
        bound: 64,
        mask: u32::MAX,
        deps: vec![],
        args: vector_args(kernel, len, at),
    }
}

#[test]
fn a_muon_task_carries_its_kernel_and_argument_block() {
    let b = two_p();
    let abi = Abi::new(&b);
    let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
    // softmax_apply (id 4): a block of 40 bytes.
    let block: Vec<u8> = (0..40).collect();
    let d = vector(&abi, &b, 4, 40, at.base + BLOCK as u64);
    let t = task(dec_with(&abi, &d, &block));
    assert_eq!(
        t.args,
        Args::Vector {
            kernel: 4,
            block: block.clone()
        }
    );
    // The block travels with a forwarded invoke: the kernel and the size
    // (4 bytes) and the block.
    assert_eq!(t.wire_bytes(), 24 + 4 + 40);
    // A block outside the data scratch.
    let d = vector(&abi, &b, 4, 40, at.bound - 8);
    let e = dec_with(&abi, &d, &block).unwrap_err();
    assert!(
        e.contains("the argument block") && e.contains("outside the own data scratch"),
        "{e}"
    );
}

#[test]
fn a_muon_kernel_must_be_in_the_registry_with_its_block_size() {
    let b = two_p();
    let abi = Abi::new(&b);
    let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
    let block = vec![0u8; 80];
    let desc = |kernel, bytes| vector(&abi, &b, kernel, bytes, at.base + BLOCK as u64);
    let e = dec_with(&abi, &desc(7, 24), &block).unwrap_err();
    assert!(e.ends_with("kernel 7 is not in the kernel registry"), "{e}");
    let e = dec_with(&abi, &desc(2, 72), &block).unwrap_err();
    assert!(
        e.ends_with("kernel 2 (epilogue) has an argument block of 72 bytes, not 80"),
        "{e}"
    );
    // Kernel 0 has its 24-byte block too.
    let e = dec_with(&abi, &desc(0, 0), &block).unwrap_err();
    assert!(
        e.ends_with("kernel 0 (opaque) has an argument block of 0 bytes, not 24"),
        "{e}"
    );
    task(dec_with(&abi, &desc(2, 80), &block));
    task(dec_with(&abi, &desc(0, 24), &block));
}

#[test]
fn the_wire_bytes_of_each_opcode() {
    // The header (24), 8 per dependency, and the arguments: DMA 32,
    // matmul 40, vector 4 and its block, spawn 16, IMEM copy 24, join 0.
    use sim_core::{Descriptor, PeId};
    let r = Range::from_len(0x10_0000, 512);
    let dma = Descriptor::dma(PeId(0), 0, r, 512);
    assert_eq!(dma.wire_bytes(), 56);
    assert_eq!(
        dma.clone()
            .with_deps(vec![Dep::Unversioned(0)])
            .wire_bytes(),
        64
    );
    let mm = Descriptor::matmul(PeId(0), 0, 0, r, 16, 16, 16, Dtype::Bf16);
    assert_eq!(mm.wire_bytes(), 64);
    assert_eq!(
        Descriptor::vector(PeId(0), r, 1, 2, 3).wire_bytes(),
        24 + 4 + 24
    );
    assert_eq!(
        Descriptor::muon(PeId(0), r, 2, vec![0; 80]).wire_bytes(),
        24 + 4 + 80
    );
    assert_eq!(Descriptor::join(r, vec![]).wire_bytes(), 24);
    let mut spawn = Descriptor::join(r, vec![]);
    spawn.args = Args::Spawn {
        child: NeutrinoId(1),
    };
    assert_eq!(spawn.wire_bytes(), 40);
    spawn.args = Args::ImemCopy {
        child: NeutrinoId(1),
        src: 0,
        offset: 0,
        bytes: 64,
    };
    assert_eq!(spawn.wire_bytes(), 48);
}

#[test]
fn the_four_tasks_of_probe2() {
    // A 2D copy, a matmul with B transposed, the epilogue (BIAS) and an
    // opaque task that reads out (ABI version 3 image of the toolchain).
    let image = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rv/probe2.bin"),
    )
    .unwrap();
    let r = common::run(&image, &[]);
    assert_eq!(r.stats.fatal, None);
    let smem = r.tasks[0].dst.base;
    let bytes = 128 * 128 * 2;
    let (c, out) = (smem + 2 * bytes, smem + 3 * bytes);
    let ops: Vec<OpcodeId> = r.tasks.iter().map(|t| t.opcode).collect();
    assert_eq!(
        ops,
        [
            OpcodeId::DMA,
            OpcodeId::MATMUL,
            OpcodeId::VECTOR,
            OpcodeId::VECTOR
        ]
    );
    let Args::Dma { bytes: n, rows, .. } = &r.tasks[0].args else {
        panic!()
    };
    assert_eq!(
        (
            *n,
            rows.map(|r| (r.count, r.bytes, r.src_stride, r.dst_stride))
        ),
        (128 * 144, Some((128, 144, 256, 144)))
    );
    let Args::Matmul {
        transpose_b,
        ldb,
        k,
        ..
    } = r.tasks[1].args
    else {
        panic!()
    };
    assert_eq!((transpose_b, ldb, k), (true, 72, 72));
    let Args::Vector { kernel, block } = &r.tasks[2].args else {
        panic!()
    };
    let ep = sim_core::kernels::by_id(2).unwrap();
    let field = |b: &[u8], name: &str| {
        let o = ep.field(name).unwrap().offset as usize;
        u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    };
    assert_eq!((*kernel, block.len()), (2, 80));
    assert_eq!((field(block, "c"), field(block, "out")), (c, out));
    let Args::Vector { kernel, block } = &r.tasks[3].args else {
        panic!()
    };
    assert_eq!(*kernel, 0);
    assert_eq!(
        block,
        &sim_core::kernels::opaque_block(out, 128 * 128, bytes, 0)
    );
}
