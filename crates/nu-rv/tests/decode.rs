//! The descriptor decoder against the layout of ABI version 1 (a 24-byte
//! header, the keys, the arguments of the opcode), on configs/2p.yaml.

mod common;

use common::{two_p, words, Desc};
use nu_model::{Addr, Dep, NeutrinoId, Range};
use nu_rv::{decode, Abi, Decoded};
use sim_core::{Args, Dtype, OpcodeId};

const AT: usize = 0x100;
const PC: u64 = 0x40;

/// Decodes `d` from offset [`AT`] of the root's data scratch.
fn dec(abi: &Abi, d: &Desc) -> Result<Decoded, String> {
    let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
    let mut s = vec![0u8; at.len() as usize];
    let b = d.bytes();
    s[AT..AT + b.len()].copy_from_slice(&b);
    decode(&s, at, at.base + AT as u64, NeutrinoId(0), PC, abi)
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
    // 2p.yaml: the highest own space is gpc1's sm3, (2 << 36) | (4 << 33)
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
            tile_bytes: 0x8000
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
        args: words(&[100, 512, 256, 0]),
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
            src: None
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
    args.extend(0u16.to_le_bytes());
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
