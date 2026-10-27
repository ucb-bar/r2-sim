//! The hello image (tests/fixtures/rv/hello.bin) on configs/2p-smem.yaml: the
//! device loads A and B from gpc0's DRAM into root.gpc0.sm0's SMEM,
//! multiplies them on that SM's Atlas, and copies C back to the DRAM.

mod common;

use common::two_p;
use nu_model::{Dep, Range};
use sim_core::{Descriptor, Dtype, Stats};

const HELLO: &[u8] = include_bytes!("fixtures/rv/hello.bin");

/// The comparable part of a task.
fn key(d: &Descriptor) -> impl PartialEq + std::fmt::Debug {
    (
        d.opcode,
        d.pe,
        d.dst,
        d.dst_mask,
        d.versioned,
        d.untracked,
        d.deps.clone(),
        d.args.clone(),
    )
}

fn scalar(s: &Stats) -> u64 {
    s.resources
        .iter()
        .find(|r| r.name == "root.scalar")
        .unwrap()
        .busy_cycles
}

#[test]
fn the_record_of_the_root() {
    let r = nu_rv::find_record(HELLO, "root");
    assert_eq!(r.len(), 1);
    assert_eq!(
        (r[0].at, r[0].entry, r[0].name.as_str()),
        (0x138, 0x2a, "hello")
    );
}

#[test]
fn hello_makes_its_four_tasks() {
    let b = two_p();
    let (dram, smem) = (b.memory("root.gpc0.hbm"), b.memory("root.gpc0.sm0.smem"));
    let (sm_dma, atlas, gpc_dma) = (
        b.pe("root.gpc0.sm0.dma"),
        b.pe("root.gpc0.sm0.atlas"),
        b.pe("root.gpc0.dma"),
    );
    // A, B and C: 128 x 128 bf16, 32 KiB apart.
    const N: u64 = 128;
    let bytes = N * N * 2;
    let at = |r: Range, i: u64| Range::from_len(r.base + i * bytes, bytes);
    let (a, bb, c, out) = (at(smem, 0), at(smem, 1), at(smem, 2), at(dram, 2));
    let v1 = |r: Range| Dep::Versioned {
        addr: r.base,
        version: 1,
    };
    // The loads are DMAs on the SM's engine (a pull by the owner of the
    // SMEM), the matmul waits for version 1 of A and B, and the copy back
    // is a DMA on the cluster's engine after version 1 of C.
    let load_a = Descriptor::dma(sm_dma, at(dram, 0).base, a, bytes).with_inferred_version(1);
    let load_b = Descriptor::dma(sm_dma, at(dram, 1).base, bb, bytes).with_inferred_version(1);
    let mm = Descriptor::matmul(atlas, a.base, bb.base, c, N, N, N, Dtype::Bf16)
        .with_deps(vec![v1(a), v1(bb)])
        .with_inferred_version(1);
    let back = Descriptor::dma(gpc_dma, c.base, out, bytes)
        .with_deps(vec![v1(c)])
        .with_inferred_version(1);
    assert_eq!(
        (a.base, c.base, out),
        (
            0x12_0000_0000,
            0x12_0001_0000,
            Range::from_len(0x10_0001_0000, 0x8000)
        )
    );

    let rv = common::run(HELLO, &[]);
    assert_eq!(rv.stats.fatal, None);
    assert_eq!(rv.tasks.len(), 4);
    // The toolchain's dry run: the two loads at 0x8a, the matmul at
    // 0xe2, the copy back at 0x122.
    let mut pcs: Vec<_> = rv.tasks.iter().map(|t| t.pc.unwrap()).collect();
    pcs.sort();
    assert_eq!(pcs, [0x8a, 0x8a, 0xe2, 0x122]);
    let task = |want: &Descriptor| {
        rv.tasks
            .iter()
            .find(|t| t.dst == want.dst)
            .unwrap_or_else(|| panic!("no task writes {:?}", want.dst))
    };
    for (want, pc) in [
        (&load_a, 0x8a),
        (&load_b, 0x8a),
        (&mm, 0xe2),
        (&back, 0x122),
    ] {
        let got = task(want);
        assert_eq!(key(got), key(want));
        assert_eq!(got.pc, Some(pc));
    }
    println!(
        "hello: {} cycles, root scalar {}",
        rv.stats.end_time,
        scalar(&rv.stats)
    );
}

/// Program names, by node.
#[derive(Clone, Default)]
struct Names(std::rc::Rc<std::cell::RefCell<Vec<&'static str>>>);

impl sim_core::Tracer for Names {
    fn program_start(
        &mut self,
        _: nu_model::NeutrinoId,
        name: &'static str,
        _: Option<u64>,
        _: u64,
    ) {
        self.0.borrow_mut().push(name);
    }
}

#[test]
fn the_root_program_is_named_by_its_record() {
    // `--program rv` names the root program "hello", from its record,
    // when it starts; without one record, "rv".
    for (image, want) in [(HELLO, "hello"), (&[0u8; 8][..], "rv")] {
        let mut b = two_p();
        b.sim.core(nu_model::NeutrinoId(0)).mem.borrow_mut().imem()[..image.len()]
            .copy_from_slice(image);
        let mut l = nu_rv::launch(&b, 0, nu_rv::DEFAULT_LIMIT).unwrap();
        let names = Names::default();
        b.sim.set_tracer(Box::new(names.clone()));
        b.sim
            .spawn_program(l.nu, l.start_name.take().unwrap(), l.program);
        b.sim.run();
        assert_eq!(names.0.borrow()[0], want);
    }
}
