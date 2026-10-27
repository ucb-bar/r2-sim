//! The interpreter on hand-encoded programs, as the root program of
//! configs/2p.yaml: the requests of the `nu.*` instructions, loads from
//! other nodes, the instruction costs, and each error.

mod common;

use common::{fatal, image, k, run, run_on, rv, two_p, Desc};
use nu_model::{Addr, Range};
use sim_core::OpcodeId;

/// The root's data scratch (2p.yaml: after the sync-only range).
const SCRATCH: u64 = 0x1_F100_0000;
/// The root's sync-only range.
const SYNC: u64 = 0x1_F000_0000;
/// gpc0's HBM and sm0's SMEM.
const HBM: u64 = 0x10_0000_0000;
const SMEM: u64 = 0x12_0000_0000;

fn u64_at(s: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(s[o..o + 8].try_into().unwrap())
}

#[test]
fn an_invoke_of_a_dma_built_with_stores_then_a_wait_and_a_load() {
    // The program stores the 7 words of a DMA descriptor (gpc0's HBM to
    // sm0's SMEM, on sm0's engine) at scratch + 0x100, invokes it, waits
    // for version 1 of the SMEM range, and loads the first word of the
    // SMEM (another node's memory) into scratch + 0x210.
    let b = two_p();
    let abi = nu_rv::Abi::new(&b);
    let pe = abi
        .node(b.nu("root.gpc0.sm0"))
        .pes
        .iter()
        .position(|&p| p == b.pe("root.gpc0.sm0.dma"))
        .unwrap();
    let d = common::dma(pe as u16, HBM, SMEM, 4096, vec![]).bytes();
    let key = (1u64 << abi.key_bits) | (SMEM >> 9);
    let mut consts = vec![SCRATCH + 0x100, key, SMEM];
    consts.extend(
        d.chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
    );
    let mut code = vec![k(rv::T0, 0)];
    for i in 0..7 {
        code.push(k(rv::T1, 3 + i));
        code.push(rv::sd(rv::T1, rv::T0, 8 * i));
    }
    code.extend([
        rv::invoke(rv::A0, rv::T0),
        rv::sd(rv::A0, rv::T0, 0x100),
        k(rv::T2, 1),
        rv::wait(rv::T2),
        k(rv::T1, 2),
        rv::ld(rv::A1, rv::T1, 0),
        rv::sd(rv::A1, rv::T0, 0x110),
        rv::exit(),
    ]);
    let mut b = b;
    b.sim
        .init_memory(Addr(HBM), 0x1122_3344_5566_7788u64.to_le_bytes().to_vec());
    let r = run_on(b, &image(&code, &consts), &[], nu_rv::DEFAULT_LIMIT);
    assert_eq!(r.stats.fatal, None);
    assert_eq!(r.tasks.len(), 1);
    let t = &r.tasks[0];
    assert_eq!(
        (t.opcode, t.dst),
        (OpcodeId::DMA, Range::from_len(SMEM, 4096))
    );
    assert_eq!(t.pc, Some(4 + 4 * 15));
    // A forwarded invoke returns 0; the load sees the DMA's data.
    assert_eq!(u64_at(&r.scratch, 0x200), 0);
    assert_eq!(u64_at(&r.scratch, 0x210), 0x1122_3344_5566_7788);
}

#[test]
fn version_vreset_and_a_local_invoke_returns_its_version() {
    // nu.version sets the counter of the root's sync granule to 5; a local
    // versioned join on it gets version 6 in rd; nu.vreset and
    // nu.vreset_all follow.
    let join = Desc {
        opcode: 0,
        pe: 0,
        flags: 1,
        base: SYNC >> 9,
        bound: 1,
        mask: u32::MAX,
        deps: vec![],
        args: vec![],
    };
    let consts = [SCRATCH + 0x100, SYNC >> 9, 5];
    let code = [
        k(rv::T0, 0),
        k(rv::T1, 1),
        k(rv::T2, 2),
        rv::version(rv::T1, rv::T2),
        rv::invoke(rv::A0, rv::T0),
        rv::sd(rv::A0, rv::T0, 0x100),
        rv::vreset(rv::T1),
        rv::vreset_all(),
        rv::exit(),
    ];
    let r = run(&image(&code, &consts), &[(0x100, join.bytes())]);
    assert_eq!(r.stats.fatal, None);
    assert_eq!(u64_at(&r.scratch, 0x200), 6);
    assert_eq!(r.tasks.len(), 1);
}

#[test]
fn a_load_from_another_nodes_data_scratch_takes_the_memory_path() {
    // gpc0's data scratch holds a word; the root loads it through the
    // route between the nodes and gpc0's scratch port.
    let b = two_p();
    let gpc0 = b.nu("root.gpc0");
    let at = b.sim.core(gpc0).scratch.unwrap();
    b.sim.core(gpc0).mem.borrow_mut().scratch()[0x40..0x48]
        .copy_from_slice(&0xabcdu64.to_le_bytes());
    let consts = [at.base + 0x40, SCRATCH];
    let code = [
        k(rv::T0, 0),
        rv::ld(rv::A0, rv::T0, 0),
        k(rv::T1, 1),
        rv::sd(rv::A0, rv::T1, 0x100),
        rv::exit(),
    ];
    let r = run_on(b, &image(&code, &consts), &[], nu_rv::DEFAULT_LIMIT);
    assert_eq!(r.stats.fatal, None);
    assert_eq!(u64_at(&r.scratch, 0x100), 0xabcd);
    let port = r
        .stats
        .resources
        .iter()
        .find(|x| x.name == "root.gpc0.scratch.port")
        .unwrap();
    assert_eq!(port.bytes, 8);
}

#[test]
fn the_instruction_classes_cost_their_cycles() {
    // 4 ALU (1), 2 loads (3), 1 store (1), a taken jump (2): 4 + 6 + 1 +
    // 2 = 13 scalar cycles, then the exit.
    let consts = [SCRATCH];
    let code = [
        rv::addi(rv::A0, rv::ZERO, 1),
        rv::addi(rv::A0, rv::A0, 1),
        rv::add(rv::A1, rv::A0, rv::A0),
        rv::addi(rv::A1, rv::A1, 1),
        k(rv::T0, 0),
        rv::ld(rv::T1, rv::ZERO, common::CONST),
        rv::sd(rv::A1, rv::T0, 0x100),
        rv::jal(rv::ZERO, 4),
        rv::exit(),
    ];
    let r = run(&image(&code, &consts), &[]);
    assert_eq!(r.stats.fatal, None);
    let s = r
        .stats
        .resources
        .iter()
        .find(|x| x.name == "root.scalar")
        .unwrap();
    assert_eq!(s.busy_cycles, 13);
    assert_eq!(u64_at(&r.scratch, 0x100), 5);
}

/// Runs `code` with `consts` (and `scratch` bytes) and returns the error.
fn error(code: &[u32], consts: &[u64], scratch: &[(usize, Vec<u8>)]) -> String {
    fatal(&run(&image(code, consts), scratch))
}

#[test]
fn the_errors_name_the_node_and_the_pc() {
    let cases: Vec<(Vec<u32>, Vec<u64>, &str)> = vec![
        (
            vec![rv::addi(0, 0, 0), 0xffff_ffff],
            vec![],
            "root: pc 0x8: illegal instruction",
        ),
        (
            vec![rv::csrr(rv::A0, 0x123)],
            vec![],
            "root: pc 0x4: unknown CSR 0x123",
        ),
        (vec![rv::ECALL], vec![], "root: pc 0x4: ecall or ebreak"),
        (vec![rv::EBREAK], vec![], "root: pc 0x4: ecall or ebreak"),
        (
            vec![k(rv::T0, 0), rv::jalr(0, rv::T0, 0)],
            vec![0x8000],
            "root: pc 0x8000: fetch outside the IMEM",
        ),
        (
            vec![rv::sd(rv::A0, rv::ZERO, 0x100)],
            vec![],
            "root: pc 0x4: store into the IMEM at 0x100",
        ),
        (
            vec![k(rv::A0, 0), rv::invoke(rv::A0, rv::A0)],
            vec![HBM],
            "root: pc 0x8: descriptor at 0x1000000000 outside the own data scratch",
        ),
    ];
    for (code, consts, want) in cases {
        let e = error(&code, &consts, &[]);
        assert!(e.contains(want), "{e:?} does not contain {want:?}");
    }
}

#[test]
fn bad_descriptors_stop_the_run() {
    let code = [k(rv::A0, 0), rv::invoke(rv::A0, rv::A0)];
    let consts = [SCRATCH + 0x100];
    let mut bad = common::dma(0, HBM, SMEM, 512, vec![]);
    bad.opcode = 9;
    let e = error(&code, &consts, &[(0x100, bad.bytes())]);
    assert!(
        e.contains("root: pc 0x8: descriptor at 0x1f1000100: unknown opcode 9"),
        "{e}"
    );
    let bad = common::dma(40, HBM, SMEM, 512, vec![]);
    let e = error(&code, &consts, &[(0x100, bad.bytes())]);
    assert!(e.contains("PE 40 of root.gpc0.sm0"), "{e}");
}

#[test]
fn the_instruction_limit_stops_the_run() {
    let r = run_on(two_p(), &image(&[rv::jal(0, 0)], &[]), &[], 1000);
    let e = fatal(&r);
    assert!(
        e.contains("root: pc 0x4: the instruction limit of 1000 is reached"),
        "{e}"
    );
}

#[test]
fn a_task_on_a_data_scratch_is_an_error() {
    // A DMA from the root's data scratch.
    let code = [k(rv::A0, 0), rv::invoke(rv::A0, rv::A0), rv::exit()];
    let consts = [SCRATCH + 0x100];
    let d = common::dma(0, SCRATCH, SMEM, 512, vec![]);
    let e = error(&code, &consts, &[(0x100, d.bytes())]);
    assert!(
        e.contains("src") && e.contains("data scratch of root"),
        "{e}"
    );
}

#[test]
fn the_root_program_is_the_one_record_of_its_node_type() {
    // No record: an error before the first instruction.
    let mut img = image(&[rv::exit()], &[]);
    img[0x700 + 56..0x700 + 60].copy_from_slice(b"gpc\0");
    let e = fatal(&run(&img, &[]));
    assert!(e.contains("0 program records of node type root"), "{e}");
}
