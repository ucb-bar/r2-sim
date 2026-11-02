//! The dispatch of the functional plane: the backend lists of the FU
//! kinds, and the detection of fidelity switches (a matmul chain on two
//! backends, and a read of the output of a `null` task). The node of
//! `common`; a test selection rule forces the switches, which the fixed
//! rule never makes.

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
        "numeric" => Box::new(func::Numeric::new()),
        "trace" => Box::new(func::Trace::new()),
        n => return Err(format!("unknown backend {n:?}")),
    })
}

/// The backends of the lists (dma, atlas, muon).
fn lists(dma: &[&str], atlas: &[&str], muon: &[&str]) -> Result<Backends, String> {
    Backends::from_names([&names(dma), &names(atlas), &names(muon)], make)
}

#[test]
fn the_unit_lists_share_one_instance_per_backend() {
    let b = lists(&["numeric", "null"], &["numeric"], &["null"]).unwrap();
    let got: Vec<&str> = b.list.iter().map(|x| x.name()).collect();
    assert_eq!(got, ["numeric", "null"]);
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
    let e = lists(&["numeric"], &["foo"], &["null"]).err().unwrap();
    assert_eq!(e, "functional.units.atlas: unknown backend \"foo\"");
    let e = lists(&["numeric"], &["numeric"], &[]).err().unwrap();
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
    let mut d = n.matmul(c, deps, E, E).with_acc(acc.0, acc.1);
    if let Args::Matmul { a, b, .. } = &mut d.args {
        (*a, *b) = (Addr(SMEM), Addr(SMEM + MAT));
    }
    d.label = Some(label);
    d
}

/// A DMA of C into `to` (an SMEM offset) that depends on version `v` of C.
fn copy_c(n: &Node, c: Range, to: u64, v: u64) -> Descriptor {
    let mut d = n.dma(0, to, MAT, MAT).with_deps(vec![ver(c, v)]);
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
    // Atlas list [numeric, trace]: the second task of the chain goes to
    // trace, while numeric holds the partial sums.
    let n = node(Knobs::default());
    let c = smem(4 * MAT, MAT);
    let tasks = vec![
        matmul(&n, c, 1, (false, true), vec![]),
        matmul(&n, c, 2, (true, true), vec![ver(c, 1)]),
        matmul(&n, c, 3, (true, false), vec![ver(c, 2)]),
    ];
    let b = lists(&["numeric"], &["numeric", "trace"], &["null"]).unwrap();
    let s = run(n, b, 1, tasks);
    let f = s.fatal.expect("an error at the switch");
    let FatalError::Invalid(msg) = &f.error else {
        panic!("{f:?}");
    };
    assert!(
        msg.contains("chain") && msg.contains("trace") && msg.contains("numeric"),
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
    // Atlas list [null, numeric]: the matmul writes C on null; two DMAs
    // read C: one warning, at the first.
    let n = node(Knobs::default());
    let c = smem(4 * MAT, MAT);
    let tasks = vec![
        matmul(&n, c, 1, (false, false), vec![]),
        copy_c(&n, c, 6 * MAT, 1),
        copy_c(&n, c, 7 * MAT, 1),
    ];
    let b = lists(&["numeric"], &["null", "numeric"], &["null"]).unwrap();
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
    // The matmul on null writes C, a matmul on numeric (the test
    // selection) writes C again, and a DMA reads C: no warning.
    let n = node(Knobs::default());
    let c = smem(4 * MAT, MAT);
    let tasks = vec![
        matmul(&n, c, 1, (false, false), vec![]),
        matmul(&n, c, 2, (false, false), vec![ver(c, 1)]),
        copy_c(&n, c, 6 * MAT, 2),
    ];
    let b = lists(&["numeric"], &["null", "numeric"], &["null"]).unwrap();
    let s = run(n, b, 0, tasks);
    assert_eq!(s.fatal, None);
    assert!(null_reads(&s).is_empty(), "{:?}", s.warnings);
}
