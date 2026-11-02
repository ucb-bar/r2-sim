//! The descriptor decoder against the layout of ABI version 2 (a 24-byte
//! header, the keys, the arguments of the opcode), on configs/2p-smem.yaml.

mod common;

use common::{two_p, words, Desc};
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

/// The arguments of a vector task (ABI version 2).
fn vector_args(
    ops: u64,
    bytes_in: u64,
    bytes_out: u64,
    src: u64,
    kernel: u32,
    len: u32,
    at: u64,
) -> Vec<u8> {
    let mut a = words(&[ops, bytes_in, bytes_out, src]);
    a.extend(kernel.to_le_bytes());
    a.extend(len.to_le_bytes());
    a.extend(at.to_le_bytes());
    a
}

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
            tile_bytes: 0x8000,
            rows: None,
        }
    );
    assert_eq!(d.pc, Some(PC));
}

#[test]
fn matmul() {
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "atlas");
    let mut args = words(&[sm, sm + 0x8000, sm + 0x20000, 0]);
    for w in [128u32, 64, 64, 128, 64, 128] {
        args.extend(w.to_le_bytes()); // lda ldb ldc m n k
    }
    args.extend((1u32 | 1 << 8).to_le_bytes()); // types: f32, f32, bf16 (0)
    args.extend(3u32.to_le_bytes()); // accumulate, keep
    args.extend(128u16.to_le_bytes());
    args.extend(64u16.to_le_bytes());
    args.extend(0u32.to_le_bytes());
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
            tile: [128, 64],
            transpose_b: false,
        }
    );
}

#[test]
fn vector_untracked_on_the_invoker() {
    // An untracked task runs at the invoking node: its PE index is the
    // invoker's (the root has none, so index 0 is past its PEs); flags
    // bit 1.
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "muon");
    let mut d = Desc {
        opcode: 4,
        pe,
        flags: 0,
        base: sm >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args: vector_args(100, 512, 256, 0, 0, 0, 0),
    };
    let t = task(dec(&abi, &d));
    assert_eq!(t.opcode, OpcodeId::VECTOR);
    assert_eq!(t.pe, Some(b.pe("root.gpc0.sm0.muon")));
    assert!(!t.versioned);
    assert_eq!(
        t.args,
        Args::Vector {
            ops: 100,
            bytes_in: 512,
            bytes_out: 256,
            src: None,
            kernel: 0,
            block: Vec::new(),
        }
    );
    d.flags = 2;
    let e = dec(&abi, &d).unwrap_err();
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
    d.args = common::dma_args(0x10_0000_0000, sm, 128 * 144, 128, 144, 256, 144);
    let t = task(dec(&abi, &d));
    assert_eq!(
        t.args,
        Args::Dma {
            src: Addr(0x10_0000_0000),
            dst: Addr(sm),
            bytes: 128 * 144,
            tile_bytes: 128 * 144,
            rows: Some(Rows {
                count: 128,
                bytes: 144,
                src_stride: 256,
                dst_stride: 144,
            }),
        }
    );
    // The descriptor's bytes as read: the header and the 48 bytes of the
    // arguments.
    assert_eq!(t.bytes, d.bytes());
    assert_eq!(t.bytes.len(), 24 + 48);
    // Rows 0, and rows x row bytes other than the bytes, are errors.
    for (rows, row_bytes) in [(0, 144), (127, 144)] {
        d.args = common::dma_args(0x10_0000_0000, sm, 128 * 144, rows, row_bytes, 256, 144);
        let e = dec(&abi, &d).unwrap_err();
        assert!(
            e.contains(&format!(
                "a DMA of {rows} rows of {row_bytes} bytes for 18432 bytes"
            )),
            "{e}"
        );
    }
}

/// The arguments of a 128 x 128 x 72 bf16 matmul with `flags` and `b2`.
fn matmul_args(sm: u64, b2: u64, flags: u32) -> Vec<u8> {
    let mut a = words(&[sm, sm + 0x8000, 0, b2]);
    for w in [72u32, 72, 128, 128, 128, 72, 0, flags] {
        a.extend(w.to_le_bytes()); // lda ldb ldc m n k types flags
    }
    a.extend(128u16.to_le_bytes());
    a.extend(128u16.to_le_bytes());
    a.extend(0u32.to_le_bytes());
    a
}

#[test]
fn a_matmul_with_b_transposed() {
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "atlas");
    let mut d = Desc {
        opcode: 3,
        pe,
        flags: 1,
        base: (sm + 0x10000) >> 9,
        bound: 64,
        mask: u32::MAX,
        deps: vec![],
        args: matmul_args(sm, 0, 4),
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
    // Not with b2.
    d.args = matmul_args(sm, sm + 0x40000, 4);
    let e = dec(&abi, &d).unwrap_err();
    assert!(e.contains("a matmul with transpose_b and b2"), "{e}");
}

#[test]
fn a_muon_task_carries_its_kernel_and_argument_block() {
    let b = two_p();
    let abi = Abi::new(&b);
    let (sm, pe) = sm0(&b, &abi, "muon");
    let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
    let block: Vec<u8> = (0..24).collect();
    let mut d = Desc {
        opcode: 4,
        pe,
        flags: 1,
        base: sm >> 9,
        bound: 64,
        mask: u32::MAX,
        deps: vec![],
        args: vector_args(100, 512, 256, 0, 7, 24, at.base + BLOCK as u64),
    };
    let t = task(dec_with(&abi, &d, &block));
    let Args::Vector {
        kernel, block: got, ..
    } = &t.args
    else {
        panic!("{:?}", t.args);
    };
    assert_eq!((*kernel, got), (7, &block));
    // The block travels with a forwarded invoke: 32 bytes of arguments
    // plus the block.
    assert_eq!(t.wire_bytes(), 24 + 32 + 24);
    // A block outside the data scratch.
    d.args = vector_args(100, 512, 256, 0, 7, 24, at.bound - 8);
    let e = dec_with(&abi, &d, &block).unwrap_err();
    assert!(
        e.contains("the argument block") && e.contains("outside the own data scratch"),
        "{e}"
    );
}
