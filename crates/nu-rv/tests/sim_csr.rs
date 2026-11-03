//! The CSRs of r2-sim for RV programs (not the Neutrino ISA): a write of
//! CSR 0x8C0 is a phase mark named by its id, a write of CSR 0x8C1 empties
//! the cache of the DRAM; a read of either is an error. And the option
//! `global.rv_instruction_costs: zero`.

mod common;

use common::{fatal, image, k, run, run_on, rv, two_p};
use nu_model::NeutrinoId;
use sim_core::{build_config, build_config_file, ProfileRegistry};

const MARK: u32 = 0x8c0;
const FLUSH: u32 = 0x8c1;

#[test]
fn a_mark_is_a_phase_mark_named_by_its_id() {
    // 2 ALU (1 cycle each); the mark 5 at the end of its CSR write (ALU
    // class, 1 cycle): cycle 3; a load (3) and the mark 7: 3 + 3 + 1 = 7.
    let code = [
        rv::addi(rv::A0, rv::ZERO, 5),
        rv::addi(rv::A1, rv::ZERO, 7),
        rv::csrw(MARK, rv::A0),
        rv::ld(rv::T0, rv::ZERO, common::CONST),
        rv::csrw(MARK, rv::A1),
        rv::exit(),
    ];
    // The marks are collected with set_marks (r2sim's --phase-stats).
    let mut b = two_p();
    b.sim.set_marks(true);
    let r = run_on(b, &image(&code, &[0]), &[], nu_rv::DEFAULT_LIMIT);
    assert_eq!(r.stats.fatal, None);
    let marks: Vec<(&str, u64)> = r
        .stats
        .marks
        .iter()
        .map(|m| (m.name.as_str(), m.time))
        .collect();
    assert_eq!(marks, [("5", 3), ("7", 7)]);
}

#[test]
fn a_read_of_a_simulator_csr_is_an_error() {
    for code in [
        vec![rv::csrr(rv::A0, MARK)],
        vec![rv::csrr(rv::A0, FLUSH)],
        // csrrw with rd != x0 reads the old value.
        vec![(MARK << 20) | (rv::A0 << 15) | (1 << 12) | (rv::A1 << 7) | 0x73],
    ] {
        let e = fatal(&run(&image(&code, &[]), &[]));
        assert!(
            e.contains("root: pc 0x4: CSR 0x8c") && e.contains("is write-only"),
            "{e}"
        );
    }
}

/// A root with an L2 that caches its HBM (lines of 256 B) and a buffer.
const CACHED: &str = "global:
  neutrino_granule: 512
  address: {fields: [{name: sm, bits: 2}], offset_bits: 24}
device:
  name: root
  fabric:
    - {name: sbus}
    - {name: cbus, control: true}
  memory:
    - {name: l2, type: scratch, addr: [[0x100000, 0x102000]], banks: 2, bw_bytes: 64, latency: 10,
       slave: $this.sbus, cache: {of: $this.hbm, line_bytes: 256}}
    - {name: hbm, type: dram, addr: [[0x200000, 0x400000]], banks: 2, bw_bytes: 64, latency: 50,
       slave: $this.sbus}
    - {name: buf, type: scratch, addr: [[0x40000, 0x80000]], bw_bytes: 64, slave: $this.sbus}
    - {name: sync, type: sync, addr: [[0xE00000, 0xF00000]]}
  compute:
    - {name: dma, type: dma, latency: 4, master: [$this.sbus]}
  neutrino: {}
";

/// Two copies of the same 512 B of the HBM into the buffer, each waited
/// for, with a flush between them or not: the read misses of the cache.
fn misses(flush: bool) -> u64 {
    let b = build_config(CACHED, ProfileRegistry::with_builtins()).expect("config");
    let abi = nu_rv::Abi::new(&b);
    let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
    let (hbm, buf) = (b.memory("root.hbm").base, b.memory("root.buf").base);
    let d = common::dma(0, hbm, buf, 512, vec![]).bytes();
    let key = |v: u64| (v << abi.key_bits) | (buf >> 9);
    let consts = [at.base + 0x100, key(1), key(2)];
    let mut code = vec![
        k(rv::A0, 0),
        rv::invoke(rv::A1, rv::A0),
        k(rv::T0, 1),
        rv::wait(rv::T0),
    ];
    if flush {
        code.push(rv::csrw(FLUSH, rv::ZERO));
    }
    code.extend([
        rv::invoke(rv::A1, rv::A0),
        k(rv::T0, 2),
        rv::wait(rv::T0),
        rv::exit(),
    ]);
    let r = run_on(
        b,
        &image(&code, &consts),
        &[(0x100, d)],
        nu_rv::DEFAULT_LIMIT,
    );
    assert_eq!(r.stats.fatal, None);
    let c = &r.stats.caches[0];
    assert_eq!(c.flushes.len(), usize::from(flush));
    c.counts.read_misses
}

#[test]
fn a_flush_empties_the_cache_so_the_second_read_misses() {
    // 512 B are two lines: two misses for the first copy; the second
    // copy hits, unless the cache was flushed between them.
    assert_eq!(misses(false), 2);
    assert_eq!(misses(true), 4);
}

#[test]
fn zero_instruction_costs_leave_the_tasks_times() {
    // A DMA built with stores, invoked, then waited for: with
    // rv_instruction_costs zero the scalar core is busy only for the
    // Neutrino costs of the invoke and the wait (c_invoke 8 and c_wait 2
    // on 2p-smem's root), and the task takes as long from its issue to its
    // completion.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/2p-smem.yaml");
    let run_with = |sets: &[&str]| {
        let b = build_config_file(&path, sets, ProfileRegistry::with_builtins()).expect("config");
        let abi = nu_rv::Abi::new(&b);
        let at = abi.node(NeutrinoId(0)).scratch.clone().unwrap();
        let sm = b.memory("root.gpc0.sm0.smem").base;
        let hbm = b.memory("root.gpc0.hbm").base;
        let pe = abi
            .node(b.nu("root.gpc0.sm0"))
            .pes
            .iter()
            .position(|&p| p == b.pe("root.gpc0.sm0.dma"))
            .unwrap();
        let d = common::dma(pe as u16, hbm, sm, 4096, vec![]).bytes();
        let key = (1u64 << abi.key_bits) | (sm >> 9);
        let mut consts = vec![at.base + 0x100, key];
        consts.extend(
            d.chunks(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
        );
        let mut code = vec![k(rv::T0, 0)];
        for i in 0..(d.len() / 8) as i32 {
            code.push(k(rv::T1, 2 + i));
            code.push(rv::sd(rv::T1, rv::T0, 8 * i));
        }
        code.extend([
            rv::invoke(rv::A0, rv::T0),
            k(rv::T2, 1),
            rv::wait(rv::T2),
            rv::exit(),
        ]);
        let r = run_on(b, &image(&code, &consts), &[], nu_rv::DEFAULT_LIMIT);
        assert_eq!(r.stats.fatal, None);
        let scalar = r
            .stats
            .resources
            .iter()
            .find(|x| x.name == "root.scalar")
            .unwrap()
            .busy_cycles;
        let t = &r.stats.tasks[0];
        (scalar, t.complete.unwrap() - t.issue.unwrap())
    };
    let (busy, task) = run_with(&[]);
    let (busy0, task0) = run_with(&["global.rv_instruction_costs=zero"]);
    assert!(busy > 10);
    assert_eq!((busy0, task0), (10, task));
    // An unknown value is an error.
    let e = build_config_file(
        &path,
        &["global.rv_instruction_costs=none"],
        ProfileRegistry::with_builtins(),
    )
    .err()
    .unwrap();
    assert!(
        format!("{e:?}").contains("unknown rv_instruction_costs"),
        "{e:?}"
    );
}
