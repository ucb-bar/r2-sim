//! The dispatch of the functional plane: the backend lists of the FU
//! kinds, and the detection of backend switches (a matmul chain on two
//! backends, and a read of the output of a `null` task). The node of
//! `common`; the lists of the FU kinds and a test selection rule make the
//! switches (the fixed rule never splits a matmul chain).

mod common;

use common::{node, smem, ver, Knobs, Node, SMEM};
use nu_model::{Addr, Range};
use sim_core::{
    Args, Backends, Descriptor, FatalError, FunctionalBackend, NuCtx, NullBackend, SimWarning,
    Stats,
};

/// The matmul tiles of the tests.
const E: u64 = 64;
const MAT: u64 = E * E * 2;

fn names(n: &[&str]) -> Vec<String> {
    n.iter().map(|s| s.to_string()).collect()
}

fn make(name: &str) -> Result<Box<dyn FunctionalBackend>, String> {
    Ok(match name {
        "null" => Box::new(NullBackend),
        "builtin" => Box::new(func::Builtin::new()),
        n => return Err(format!("unknown backend {n:?}")),
    })
}

/// The backends of the lists (dma, atlas, muon).
fn lists(dma: &[&str], atlas: &[&str], muon: &[&str]) -> Result<Backends, String> {
    Backends::from_names([&names(dma), &names(atlas), &names(muon)], make)
}

#[test]
fn the_unit_lists_share_one_instance_per_backend() {
    let b = lists(&["builtin", "null"], &["builtin"], &["null"]).unwrap();
    let got: Vec<&str> = b.list.iter().map(|x| x.name()).collect();
    assert_eq!(got, ["builtin", "null"]);
    assert_eq!(
        (
            b.units.dma.clone(),
            b.units.atlas.clone(),
            b.units.muon.clone()
        ),
        (vec![0, 1], vec![0], vec![1])
    );
}

#[test]
fn a_bad_list_names_its_key() {
    let e = lists(&["builtin"], &["foo"], &["null"]).err().unwrap();
    assert_eq!(e, "functional.units.atlas: unknown backend \"foo\"");
    let e = lists(&["builtin"], &["builtin"], &[]).err().unwrap();
    assert_eq!(e, "functional.units.muon: an empty list");
}

/// A matmul of C (E x E bf16) with label `label`, `accumulate` and `keep`,
/// that reads A and B at SMEM offsets 0 and MAT (never written: zero) and
/// depends on `deps`.
fn matmul(
    n: &Node,
    c: Range,
    label: u64,
    acc: (bool, bool),
    deps: Vec<nu_model::Dep>,
) -> Descriptor {
    let mut d = n.matmul(c, deps, E).with_acc(acc.0, acc.1);
    if let Args::Matmul { a, b, .. } = &mut d.args {
        (*a, *b) = (Addr(SMEM), Addr(SMEM + MAT));
    }
    d.label = Some(label);
    d
}

/// A DMA of C into `to` (an SMEM offset) that depends on version `v` of C.
fn copy_c(n: &Node, c: Range, to: u64, v: u64) -> Descriptor {
    let mut d = n.dma(0, to, MAT).with_deps(vec![ver(c, v)]);
    if let Args::Dma { src, .. } = &mut d.args {
        *src = Addr(c.base);
    }
    d
}

/// Runs `tasks` in order on the node with `backends`; the matmul with
/// label 2 goes to the backend `second` (an index of the backends).
fn run(n: Node, backends: Backends, second: usize, tasks: Vec<Descriptor>) -> Stats {
    let mut sim = n.sim;
    sim.set_backends(backends);
    sim.set_selection(Box::new(
        move |d, b| if d.label == Some(2) { second } else { b },
    ));
    let nu = n.nu;
    sim.spawn(nu, move |ctx: NuCtx| async move {
        let mut last = None;
        for t in tasks {
            last = Some(ctx.invoke(t).await.key());
        }
        ctx.wait_key(last.unwrap()).await;
    });
    sim.run()
}

#[test]
fn a_matmul_chain_on_two_backends_stops_the_run() {
    // Atlas list [builtin, null]: the second task of the chain goes to
    // null, while builtin holds the partial sums.
    let n = node(Knobs::default());
    let c = smem(4 * MAT, MAT);
    let tasks = vec![
        matmul(&n, c, 1, (false, true), vec![]),
        matmul(&n, c, 2, (true, true), vec![ver(c, 1)]),
        matmul(&n, c, 3, (true, false), vec![ver(c, 2)]),
    ];
    let b = lists(&["builtin"], &["builtin", "null"], &["null"]).unwrap();
    let s = run(n, b, 1, tasks);
    let f = s.fatal.expect("an error at the switch");
    let FatalError::Invalid(msg) = &f.error else {
        panic!("{f:?}");
    };
    assert!(
        msg.contains("chain") && msg.contains("null") && msg.contains("builtin"),
        "{msg}"
    );
}

/// The `NullRead` warnings of a run.
fn null_reads(s: &Stats) -> Vec<&sim_core::WarningRec> {
    s.warnings
        .iter()
        .filter(|w| matches!(w.warning, SimWarning::NullRead { .. }))
        .collect()
}

#[test]
fn a_read_of_null_output_warns_once_per_range() {
    // Atlas list [null, builtin]: the matmul writes C on null; two DMAs
    // read C: one warning, at the first.
    let n = node(Knobs::default());
    let c = smem(4 * MAT, MAT);
    let tasks = vec![
        matmul(&n, c, 1, (false, false), vec![]),
        copy_c(&n, c, 6 * MAT, 1),
        copy_c(&n, c, 7 * MAT, 1),
    ];
    let b = lists(&["builtin"], &["null", "builtin"], &["null"]).unwrap();
    let s = run(n, b, 0, tasks);
    assert_eq!(s.fatal, None);
    let reads = null_reads(&s);
    assert_eq!(reads.len(), 1, "{reads:?}");
    assert_eq!(reads[0].gid, Some(1), "the first DMA");
    assert_eq!(
        reads[0].warning,
        SimWarning::NullRead {
            range: c,
            writer: Some(0)
        }
    );
}

#[test]
fn a_write_by_a_computing_backend_removes_the_mark() {
    // The matmul on null writes C, a matmul on builtin (the test
    // selection) writes C again, and a DMA reads C: no warning.
    let n = node(Knobs::default());
    let c = smem(4 * MAT, MAT);
    let tasks = vec![
        matmul(&n, c, 1, (false, false), vec![]),
        matmul(&n, c, 2, (false, false), vec![ver(c, 1)]),
        copy_c(&n, c, 6 * MAT, 2),
    ];
    let b = lists(&["builtin"], &["null", "builtin"], &["null"]).unwrap();
    let s = run(n, b, 0, tasks);
    assert_eq!(s.fatal, None);
    assert!(null_reads(&s).is_empty(), "{:?}", s.warnings);
}

#[test]
fn a_muon_task_that_reads_null_output_warns() {
    // Atlas list [null], Muon list [builtin]: the matmul writes C on null; a
    // Muon task reads [src, src + bytes_in) = C (one warning), another
    // reads its own dst when it has no src (none: nothing wrote it).
    let mut n = node(Knobs::default());
    let vector = n.sim.resources_mut().add_flow(
        "muon.vector",
        sim_core::resource::Kind::VectorOps,
        sim_core::Ratio::int(1024),
    );
    let muon = n.sim.add_pe(
        n.nu,
        sim_core::PeConfig {
            name: "muon0".into(),
            max_inflight: 1,
            max_outstanding: 1,
            params: sim_core::PeParams::Muon {
                vector,
                beat: 1024,
                masters: Vec::new(),
                utilization: sim_core::Ratio::ONE,
                overhead: 0,
            },
        },
    );
    let c = smem(4 * MAT, MAT);
    let out = smem(8 * MAT, MAT);
    let reader = Descriptor::vector(muon, out, 1024, MAT, MAT)
        .with_src(c.base)
        .with_deps(vec![ver(c, 1)]);
    let own = Descriptor::vector(muon, smem(10 * MAT, MAT), 1024, MAT, MAT);
    let tasks = vec![matmul(&n, c, 1, (false, false), vec![]), reader, own];
    let b = lists(&["builtin"], &["null"], &["builtin"]).unwrap();
    let s = run(n, b, 0, tasks);
    assert_eq!(s.fatal, None);
    let reads = null_reads(&s);
    assert_eq!(reads.len(), 1, "{reads:?}");
    assert_eq!(reads[0].gid, Some(1), "the Muon task with src C");
    assert_eq!(
        reads[0].warning,
        SimWarning::NullRead {
            range: c,
            writer: Some(0)
        }
    );
}
