//! The hierarchy of RV programs on configs/2p.yaml: IMEM_COPY on the
//! instruction DMA, SPAWN of RV children, and the three-level image
//! (tests/fixtures/rv/three-level.bin): a device program, a program on
//! each cluster, a program on each SM.

mod common;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use common::{fatal, image, k, run, rv, two_p, words, Desc};
use nu_model::NeutrinoId;
use nu_rv::{Abi, RvProgram};
use sim_core::{Args, Built, Cycle, Descriptor, OpcodeId, Stats, TaskEntered, Tracer, Who};

const THREE: &[u8] = include_bytes!("fixtures/rv/three-level.bin");
const SYNC: u64 = 0x1_F000_0000;
const SCRATCH: u64 = 0x1_F100_0000;

/// What the tracer saw.
#[derive(Default)]
struct Seen {
    /// (node, name, start, return).
    programs: Vec<(NeutrinoId, &'static str, Cycle, Option<Cycle>)>,
    /// gid -> (invoker, descriptor, complete).
    tasks: HashMap<u64, (NeutrinoId, Descriptor, Option<Cycle>)>,
}

#[derive(Clone, Default)]
struct Life(Rc<RefCell<Seen>>);

impl Tracer for Life {
    fn task_invoke(&mut self, t: &TaskEntered) {
        self.0
            .borrow_mut()
            .tasks
            .insert(t.gid, (t.nu, t.desc.clone(), None));
    }
    fn task_complete(&mut self, gid: u64, time: Cycle) {
        if let Some(t) = self.0.borrow_mut().tasks.get_mut(&gid) {
            t.2 = Some(time);
        }
    }
    fn program_start(&mut self, nu: NeutrinoId, name: &'static str, _: Option<u64>, t: Cycle) {
        self.0.borrow_mut().programs.push((nu, name, t, None));
    }
    fn program_return(&mut self, nu: NeutrinoId, time: Cycle) {
        let mut s = self.0.borrow_mut();
        let p = s
            .programs
            .iter_mut()
            .rev()
            .find(|p| p.0 == nu && p.3.is_none())
            .unwrap();
        p.3 = Some(time);
    }
    fn charge(&mut self, _: &sim_core::Charge, _: Who) {}
}

/// Runs `imem` as the root program of `b`; the stats and what the tracer
/// saw.
fn run_traced(mut b: Built, imem: &[u8]) -> (Stats, Seen, Abi) {
    let abi = Abi::new(&b);
    b.sim.core(NeutrinoId(0)).mem.borrow_mut().imem()[..imem.len()].copy_from_slice(imem);
    let life = Life::default();
    b.sim.set_tracer(Box::new(life.clone()));
    b.sim.set_functional(Box::new(func::Numeric::new()));
    let p = RvProgram::root(Rc::new(abi.clone()), 0, nu_rv::DEFAULT_LIMIT);
    b.sim
        .spawn_named(NeutrinoId(0), "device", move |ctx| p.run(ctx));
    let stats = b.sim.run();
    let seen = std::mem::take(&mut *life.0.borrow_mut());
    (stats, seen, abi)
}

#[test]
fn the_three_level_image() {
    let b = two_p();
    let path = |nu: NeutrinoId| b.sim.topology().path_name(nu);
    let paths: Vec<String> = (0..b.sim.topology().len())
        .map(|i| path(NeutrinoId(i as u32)))
        .collect();
    let (s, seen, abi) = run_traced(two_p(), THREE);
    assert_eq!(s.fatal, None);
    // 1 + 2 + 8 programs start and return.
    assert_eq!(seen.programs.len(), 11);
    assert!(seen.programs.iter().all(|p| p.3.is_some()));
    let named = |n: &str| seen.programs.iter().filter(|p| p.1 == n).count();
    assert_eq!(
        (named("device"), named("gpc_fanout"), named("sm_tile")),
        (1, 2, 8)
    );
    let tasks: Vec<_> = seen.tasks.values().collect();
    // 8 matmuls, one on each SM's Atlas, k = 256, C at smem + 0x10000.
    let mut sms = Vec::new();
    for (_, d, _) in tasks.iter().filter(|t| t.1.opcode == OpcodeId::MATMUL) {
        let Args::Matmul { k, .. } = d.args else {
            unreachable!()
        };
        assert_eq!(k, 256);
        let sm = abi.owner_of(d.dst.base).unwrap();
        assert_eq!(d.pe, Some(b.pe(&format!("{}.atlas", paths[sm.0 as usize]))));
        assert_eq!(d.dst.base, abi.node(sm).own.base + 0x10000);
        sms.push(sm);
    }
    sms.sort();
    sms.dedup();
    assert_eq!(sms.len(), 8);
    // The copies: two at the device (src 0xd8, 544 B; records at sync +0x0
    // and +0x400), eight at the clusters (src 0xf8, 224 B).
    let copies: Vec<_> = tasks
        .iter()
        .filter(|t| t.1.opcode == OpcodeId::IMEM_COPY)
        .collect();
    assert_eq!(copies.len(), 10);
    let mut device_records = Vec::new();
    for (nu, d, _) in &copies {
        let Args::ImemCopy {
            src, bytes, offset, ..
        } = d.args
        else {
            unreachable!()
        };
        match nu.0 {
            0 => {
                assert_eq!((src, bytes, offset), (0xd8, 544, 0));
                device_records.push(d.dst.base - SYNC);
            }
            _ => assert_eq!((src, bytes, offset), (0xf8, 224, 0)),
        }
    }
    device_records.sort();
    assert_eq!(device_records, [0, 0x400]);
    // Each spawn's program starts after its IMEM copy completes.
    for (nu, d, _) in tasks.iter().filter(|t| t.1.opcode == OpcodeId::SPAWN) {
        let Args::Spawn { child } = d.args else {
            unreachable!()
        };
        let copy = copies
            .iter()
            .find(|c| {
                c.0 == *nu && matches!(c.1.args, Args::ImemCopy { child: x, .. } if x == child)
            })
            .unwrap();
        let start = seen.programs.iter().find(|p| p.0 == child).unwrap().2;
        assert!(
            copy.2.unwrap() <= start,
            "{}: copy done {:?}, start {start}",
            paths[child.0 as usize],
            copy.2
        );
        if nu.0 == 0 {
            assert!([0x200, 0x600].contains(&(d.dst.base - SYNC)));
        }
    }
    // The IMEMs: gpc_fanout (544 B) in each cluster, sm_tile (224 B) in
    // each SM.
    let imem = |nu: NeutrinoId, n: usize| abi.node(nu).core.mem.borrow_mut().imem()[..n].to_vec();
    let root = abi.node(NeutrinoId(0)).core.mem.borrow_mut().imem()[0xd8..0xd8 + 544].to_vec();
    for g in ["root.gpc0", "root.gpc1"] {
        let gpc = b.nu(g);
        assert_eq!(imem(gpc, 544), root);
        let fan = abi.node(gpc).core.mem.borrow_mut().imem()[0xf8..0xf8 + 224].to_vec();
        for i in 0..4 {
            assert_eq!(imem(b.nu(&format!("{g}.sm{i}")), 224), fan);
        }
    }
    // The children read their arguments from their parents' data scratch:
    // the device's scratch port carries the clusters' loads, each
    // cluster's those of its SMs.
    for p in [
        "root.scratch.port",
        "root.gpc0.scratch.port",
        "root.gpc1.scratch.port",
    ] {
        let r = s
            .resources
            .iter()
            .find(|r| r.name == p)
            .unwrap_or_else(|| panic!("no {p}"));
        assert!(r.bytes > 0, "{p}");
    }
    println!("three-level: {} cycles", s.end_time);
    for (nu, name, start, ret) in &seen.programs {
        println!(
            "  {:<16} {name:<10} start {start:>5} return {:>5}",
            paths[nu.0 as usize],
            ret.unwrap()
        );
    }
}

/// An IMEM_COPY descriptor of the root: `bytes` from IMEM offset `src`
/// to offset `offset` of child `child`, with its record at sync + `rec`.
fn copy(child: u16, src: u64, offset: u32, bytes: u64, rec: u64) -> Desc {
    let mut args = words(&[src]);
    args.extend(child.to_le_bytes());
    args.extend(0u16.to_le_bytes());
    args.extend(offset.to_le_bytes());
    args.extend(bytes.to_le_bytes());
    Desc {
        opcode: 5,
        pe: 0xffff,
        flags: 0,
        base: rec >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args,
    }
}

/// The root invokes the descriptor at scratch + 0x100, waits for its
/// record (an unversioned key), and exits.
fn invoke_and_wait(record: u64) -> Vec<u8> {
    let code = [
        k(rv::A0, 0),
        rv::invoke(rv::A1, rv::A0),
        k(rv::T0, 1),
        rv::wait(rv::T0),
        rv::exit(),
    ];
    image(&code, &[SCRATCH + 0x100, record >> 9])
}

#[test]
fn an_imem_copy_moves_bytes_into_the_childs_imem_when_it_completes() {
    // 64 B of the root's constants (IMEM 0x400) to offset 0x100 of gpc0:
    // the instruction DMA takes 10 cycles, then 64 B at 8 B/cycle.
    let img = invoke_and_wait(SYNC);
    let d = copy(0, 0x400, 0x100, 64, SYNC);
    let r = run(&img, &[(0x100, d.bytes())]);
    assert_eq!(r.stats.fatal, None);
    assert_eq!(r.tasks.len(), 1);
    assert_eq!(r.tasks[0].opcode, OpcodeId::IMEM_COPY);
    let b = two_p();
    let gpc0 = r.abi.node(b.nu("root.gpc0")).core.mem.borrow_mut().imem()[0x100..0x140].to_vec();
    assert_eq!(gpc0, img[0x400..0x440]);
    // 64 B at 8 B/cycle on the instruction DMA, after its latency of 10.
    let idma = r
        .stats
        .resources
        .iter()
        .find(|x| x.name == "root.idma")
        .unwrap();
    assert_eq!(idma.busy_cycles, 8);
    let t = &r.stats.tasks[0];
    let (issue, complete) = (t.issue.unwrap(), t.complete.unwrap());
    assert!(complete - issue >= 18, "issue {issue}, complete {complete}");
}

#[test]
fn the_errors_of_imem_copy_and_spawn() {
    let cases = [
        (
            copy(5, 0x400, 0, 64, SYNC),
            "child index 5 past the 2 direct children",
        ),
        (
            copy(0, 0x7ff0, 0, 64, SYNC),
            "runs past the IMEM (32768 bytes) of root",
        ),
        (
            copy(1, 0x400, 0x7fe0, 64, SYNC),
            "runs past the IMEM (32768 bytes) of root.gpc1",
        ),
        (
            copy(0, 0x400, 0, 64, 0x10_0000_0000),
            "is not an unversioned range of the sync-only space of root",
        ),
    ];
    for (d, want) in cases {
        let e = fatal(&run(&invoke_and_wait(SYNC), &[(0x100, d.bytes())]));
        assert!(e.contains(want), "{e:?} does not contain {want:?}");
    }
    let mut args = 3u16.to_le_bytes().to_vec();
    args.extend(0u16.to_le_bytes());
    args.extend(0x2au32.to_le_bytes());
    args.extend(0u64.to_le_bytes());
    let spawn = Desc {
        opcode: 1,
        pe: 0,
        flags: 0,
        base: (SYNC + 0x200) >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args,
    };
    let e = fatal(&run(&invoke_and_wait(SYNC), &[(0x100, spawn.bytes())]));
    assert!(
        e.contains("root: pc 0x8: child index 3 past the 2 direct children"),
        "{e}"
    );
}

/// A child image of `bytes` bytes for node type `gpc`: at offset 4 the
/// start code (it jumps to the entry in the start mailbox at offset 0 of
/// the node's data scratch, prefix + 0x1f1000000 on 2p), `nu.exit` at
/// 0x40 and 0x50, and a record named `name` with entry 0x40 at `rec`.
fn child_image(bytes: usize, name: &str, rec: usize) -> Vec<u8> {
    let slli = |rd: u32, rs1: u32, sh: u32| (sh << 20) | (rs1 << 15) | (1 << 12) | (rd << 7) | 0x13;
    let start = [
        rv::csrr(rv::T0, 0xfc0),
        rv::addi(rv::T1, rv::ZERO, 0x1f1),
        slli(rv::T1, rv::T1, 24),
        rv::add(rv::T0, rv::T0, rv::T1),
        rv::ld(rv::T1, rv::T0, 0),
        rv::jalr(rv::ZERO, rv::T1, 0),
    ];
    let mut m = vec![0u8; bytes];
    for (i, w) in start.iter().enumerate() {
        m[4 + 4 * i..8 + 4 * i].copy_from_slice(&w.to_le_bytes());
    }
    for at in [0x40, 0x50] {
        m[at..at + 4].copy_from_slice(&rv::exit().to_le_bytes());
    }
    m[rec..rec + 8].copy_from_slice(&0x40u64.to_le_bytes());
    m[rec + 8..rec + 12].copy_from_slice(&1u32.to_le_bytes());
    m[rec + 24..rec + 24 + name.len()].copy_from_slice(name.as_bytes());
    m[rec + 56..rec + 59].copy_from_slice(b"gpc");
    m
}

/// A SPAWN of the root on gpc0 at `entry`, done at sync + `done`.
fn spawn(entry: u32, done: u64) -> Desc {
    let mut args = 0u16.to_le_bytes().to_vec();
    args.extend(0u16.to_le_bytes());
    args.extend(entry.to_le_bytes());
    args.extend(0u64.to_le_bytes());
    Desc {
        opcode: 1,
        pe: 0,
        flags: 0,
        base: (SYNC + done) >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args,
    }
}

#[test]
fn a_started_program_is_named_from_the_image_its_node_loaded() {
    // The root copies image A (0x300 B, record "big" at 0x200) into
    // gpc0's IMEM and starts it; then the smaller image B (0x100 B,
    // record "small" at 0x80, the same entry 0x40) and starts it. A's
    // record is still in the IMEM past B's end, but B's program is
    // "small". A start at entry 0x50, which no record of B names, is "rv".
    let (a, bimg) = (
        child_image(0x300, "big", 0x200),
        child_image(0x100, "small", 0x80),
    );
    let steps = [
        copy(0, 0x1000, 0, 0x300, SYNC),
        spawn(0x40, 0x200),
        copy(0, 0x1400, 0, 0x100, SYNC + 0x400),
        spawn(0x40, 0x600),
        spawn(0x50, 0x800),
    ];
    let waits = [0, 0x200, 0x400, 0x600, 0x800];
    let mut code = Vec::new();
    let mut consts = Vec::new();
    let mut scratch = Vec::new();
    for (i, (d, w)) in steps.iter().zip(waits).enumerate() {
        let at = 0x100 + 0x80 * i;
        scratch.push((at, d.bytes()));
        consts.push(SCRATCH + at as u64);
        consts.push((SYNC + w) >> 9);
        code.extend([
            k(rv::A0, 2 * i as i32),
            rv::invoke(rv::A1, rv::A0),
            k(rv::T0, 2 * i as i32 + 1),
            rv::wait(rv::T0),
        ]);
    }
    code.push(rv::exit());
    let mut img = image(&code, &consts);
    img.resize(0x1500, 0);
    img[0x1000..0x1300].copy_from_slice(&a);
    img[0x1400..0x1500].copy_from_slice(&bimg);
    let b = two_p();
    {
        let core = b.sim.core(NeutrinoId(0));
        let mut m = core.mem.borrow_mut();
        for (at, d) in &scratch {
            m.scratch()[*at..*at + d.len()].copy_from_slice(d);
        }
    }
    let gpc0 = b.nu("root.gpc0");
    let (s, seen, _) = run_traced(b, &img);
    assert_eq!(s.fatal, None);
    let names: Vec<&str> = seen
        .programs
        .iter()
        .filter(|p| p.0 == gpc0)
        .map(|p| p.1)
        .collect();
    assert_eq!(names, ["big", "small", "rv"]);
}
