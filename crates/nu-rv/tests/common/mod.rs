//! Shared parts of the nu-rv tests: configs/2p.yaml, an encoder of the
//! few instructions the tests need, descriptor bytes, a tracer that keeps
//! the invoked descriptors, and a run of an RV root program.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::NeutrinoId;
use nu_rv::{Abi, RvProgram};
use sim_core::{build_config_file, Built, Descriptor, ProfileRegistry, Stats, TaskEntered, Tracer};

pub fn two_p() -> Built {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p.yaml");
    build_config_file(&path, &[] as &[&str], ProfileRegistry::with_builtins()).expect("config")
}

/// RV64 encodings.
pub mod rv {
    pub const ZERO: u32 = 0;
    pub const RA: u32 = 1;
    pub const SP: u32 = 2;
    pub const T0: u32 = 5;
    pub const T1: u32 = 6;
    pub const T2: u32 = 7;
    pub const A0: u32 = 10;
    pub const A1: u32 = 11;

    fn i(op: u32, f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32 & 0xfff) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
    }
    fn s(f3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
        let imm = imm as u32 & 0xfff;
        ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 0x1f) << 7) | 0x23
    }
    pub fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
        i(0x13, 0, rd, rs1, imm)
    }
    pub fn ld(rd: u32, rs1: u32, imm: i32) -> u32 {
        i(0x03, 3, rd, rs1, imm)
    }
    pub fn sd(rs2: u32, rs1: u32, imm: i32) -> u32 {
        s(3, rs1, rs2, imm)
    }
    pub fn add(rd: u32, rs1: u32, rs2: u32) -> u32 {
        (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33
    }
    /// `csrrs rd, csr, x0`.
    pub fn csrr(rd: u32, csr: u32) -> u32 {
        (csr << 20) | (2 << 12) | (rd << 7) | 0x73
    }
    pub fn jal(rd: u32, off: i32) -> u32 {
        let o = off as u32;
        (((o >> 20) & 1) << 31)
            | (((o >> 1) & 0x3ff) << 21)
            | (((o >> 11) & 1) << 20)
            | (((o >> 12) & 0xff) << 12)
            | (rd << 7)
            | 0x6f
    }
    pub fn jalr(rd: u32, rs1: u32, imm: i32) -> u32 {
        i(0x67, 0, rd, rs1, imm)
    }
    pub const ECALL: u32 = 0x73;
    pub const EBREAK: u32 = 0x0010_0073;
    fn nu(f3: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x0b
    }
    pub fn invoke(rd: u32, rs1: u32) -> u32 {
        nu(0, rd, rs1, 0)
    }
    pub fn wait(rs1: u32) -> u32 {
        nu(1, 0, rs1, 0)
    }
    pub fn version(rs1: u32, rs2: u32) -> u32 {
        nu(2, 0, rs1, rs2)
    }
    pub fn vreset(rs1: u32) -> u32 {
        nu(3, 0, rs1, 0)
    }
    pub fn vreset_all() -> u32 {
        nu(4, 0, 0, 0)
    }
    pub fn exit() -> u32 {
        nu(5, 0, 0, 0)
    }
}

/// The IMEM offset of the constants of a test program: `ld rd, CONST +
/// 8 i(x0)` loads constant `i`.
pub const CONST: i32 = 0x400;

/// Loads constant `i` into `rd`.
pub fn k(rd: u32, i: i32) -> u32 {
    rv::ld(rd, rv::ZERO, CONST + 8 * i)
}

/// An IMEM image: the trap word at 0, `code` from offset 4, the constants
/// at [`CONST`], and the root record (entry 4) after them.
pub fn image(code: &[u32], consts: &[u64]) -> Vec<u8> {
    let mut m = vec![0u8; 0x800];
    for (i, w) in code.iter().enumerate() {
        m[4 + 4 * i..8 + 4 * i].copy_from_slice(&w.to_le_bytes());
    }
    let c = CONST as usize;
    for (i, v) in consts.iter().enumerate() {
        m[c + 8 * i..c + 8 * i + 8].copy_from_slice(&v.to_le_bytes());
    }
    let r = 0x700;
    m[r..r + 8].copy_from_slice(&4u64.to_le_bytes());
    m[r + 8..r + 12].copy_from_slice(&1u32.to_le_bytes());
    m[r + 24..r + 28].copy_from_slice(b"test");
    m[r + 56..r + 60].copy_from_slice(b"root");
    m
}

/// A descriptor of ABI version 1.
pub struct Desc {
    pub opcode: u16,
    pub pe: u16,
    pub flags: u16,
    pub base: u64,
    pub bound: u32,
    pub mask: u32,
    pub deps: Vec<u64>,
    pub args: Vec<u8>,
}

impl Desc {
    pub fn bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend(self.opcode.to_le_bytes());
        v.extend(self.pe.to_le_bytes());
        v.extend(self.flags.to_le_bytes());
        v.extend((self.deps.len() as u16).to_le_bytes());
        v.extend(self.base.to_le_bytes());
        v.extend(self.bound.to_le_bytes());
        v.extend(self.mask.to_le_bytes());
        for d in &self.deps {
            v.extend(d.to_le_bytes());
        }
        v.extend(&self.args);
        v
    }
}

/// Little-endian bytes of `u64` words.
pub fn words(w: &[u64]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A DMA descriptor (versioned) of `bytes` from `src` to `to` on PE `pe`
/// of the owner of `to`, granule 512.
pub fn dma(pe: u16, src: u64, to: u64, bytes: u64, deps: Vec<u64>) -> Desc {
    Desc {
        opcode: 2,
        pe,
        flags: 1,
        base: to >> 9,
        bound: (bytes >> 9) as u32,
        mask: u32::MAX,
        deps,
        args: words(&[src, to, bytes, bytes]),
    }
}

/// The descriptors of the invoked tasks, in invoke order.
#[derive(Clone, Default)]
pub struct Tasks(pub Rc<RefCell<Vec<Descriptor>>>);

impl Tracer for Tasks {
    fn task_invoke(&mut self, t: &TaskEntered) {
        self.0.borrow_mut().push(t.desc.clone());
    }
}

/// A run of an RV root program on 2p.yaml.
pub struct Run {
    pub stats: Stats,
    pub tasks: Vec<Descriptor>,
    /// The root's data scratch after the run.
    pub scratch: Vec<u8>,
    pub abi: Abi,
}

/// Runs `imem` as the root program on `b` with the Numeric backend, after
/// `setup` (for example initial memory). `scratch` bytes go to the root's
/// data scratch at their offsets.
pub fn run_on(mut b: Built, imem: &[u8], scratch: &[(usize, Vec<u8>)], limit: u64) -> Run {
    let abi = Abi::new(&b);
    let root = NeutrinoId(0);
    let core = b.sim.core(root);
    {
        let mut m = core.mem.borrow_mut();
        m.imem()[..imem.len()].copy_from_slice(imem);
        for (o, v) in scratch {
            m.scratch()[*o..*o + v.len()].copy_from_slice(v);
        }
    }
    let tasks = Tasks::default();
    b.sim.set_tracer(Box::new(tasks.clone()));
    b.sim.set_functional(Box::new(func::Numeric::new()));
    b.sim.set_timeline(true);
    let p = RvProgram::root(Rc::new(abi.clone()), 0, limit);
    b.sim.spawn_named(root, "rv", move |ctx| p.run(ctx));
    let stats = b.sim.run();
    let scratch = core.mem.borrow_mut().scratch().to_vec();
    let tasks = tasks.0.borrow().clone();
    Run {
        stats,
        tasks,
        scratch,
        abi,
    }
}

pub fn run(imem: &[u8], scratch: &[(usize, Vec<u8>)]) -> Run {
    run_on(two_p(), imem, scratch, nu_rv::DEFAULT_LIMIT)
}

/// The fatal error message of a run.
pub fn fatal(r: &Run) -> String {
    match &r.stats.fatal {
        Some(f) => match &f.error {
            sim_core::FatalError::Invalid(m) => m.clone(),
            e => format!("{e:?}"),
        },
        None => panic!("the run did not stop on an error"),
    }
}
