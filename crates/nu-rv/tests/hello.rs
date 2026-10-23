//! The hello image (tests/fixtures/rv/hello.bin, the RV64 build of
//! `--program hello`) on configs/2p.yaml makes the same tasks as the Rust
//! program: the same PEs, destination ranges and dependencies.

mod common;

use common::{two_p, Tasks};
use nu_model::{Dep, Range};
use sim_core::{Descriptor, Stats};

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

fn rust_hello() -> (Stats, Vec<Descriptor>) {
    let mut b = two_p();
    let l = nu_prog::hello::launch(&b).unwrap();
    let tasks = Tasks::default();
    b.sim.set_tracer(Box::new(tasks.clone()));
    b.sim.spawn_named(l.nu, l.name, l.program);
    let s = b.sim.run();
    let t = tasks.0.borrow().clone();
    (s, t)
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
fn hello_makes_the_tasks_of_the_rust_program() {
    let (rs, rust) = rust_hello();
    let rv = common::run(HELLO, &[]);
    assert_eq!(rs.fatal, None);
    assert_eq!(rv.stats.fatal, None);
    assert_eq!(rust.len(), 4);
    assert_eq!(rv.tasks.len(), 4);
    for (a, b) in rust.iter().zip(&rv.tasks) {
        assert_eq!(key(a), key(b));
    }
    // The toolchain's dry run: the two loads at 0x8a, the matmul at
    // 0xe2, the copy back at 0x122. The tracer sees the tasks in the order
    // they enter their owners' tables: the copy back reaches gpc0 before
    // the matmul reaches sm0.
    let mut pcs: Vec<_> = rv.tasks.iter().map(|t| t.pc.unwrap()).collect();
    pcs.sort();
    assert_eq!(pcs, [0x8a, 0x8a, 0xe2, 0x122]);
    let at = |pc| rv.tasks.iter().find(|t| t.pc == Some(pc)).unwrap();
    let v1 = |addr| Dep::Versioned { addr, version: 1 };
    let mm = at(0xe2);
    assert_eq!(mm.dst.base, 0x12_0001_0000);
    assert_eq!(mm.deps, vec![v1(0x12_0000_0000), v1(0x12_0000_8000)]);
    let back = at(0x122);
    assert_eq!(back.dst, Range::from_len(0x10_0001_0000, 0x8000));
    assert_eq!(back.deps, vec![v1(0x12_0001_0000)]);
    println!(
        "hello: Rust {} cycles, root scalar {}; RV {} cycles, root scalar {}",
        rs.end_time,
        scalar(&rs),
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
