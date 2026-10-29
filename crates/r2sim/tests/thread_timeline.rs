//! `--thread-timeline`: the wall-clock records of the timing thread and
//! the functional worker for the hello image, with the numeric backend and
//! with the Python plane.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

const R2SIM: &str = env!("CARGO_BIN_EXE_r2sim");

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs hello with `functional` and returns the timeline records.
fn timeline(functional: &str) -> Vec<Value> {
    let file = std::env::temp_dir().join(format!(
        "r2sim-thread-timeline-{}-{functional}.jsonl",
        std::process::id()
    ));
    let status = Command::new(R2SIM)
        .args(["run", "--config"])
        .arg(root().join("configs/2p-smem.yaml"))
        .args(["--program", "rv", "--functional", functional, "--image"])
        .arg(root().join("crates/nu-rv/tests/fixtures/rv/hello.toml"))
        .arg("--thread-timeline")
        .arg(&file)
        .output()
        .expect("run r2sim");
    assert!(status.status.success(), "{status:?}");
    let text = std::fs::read_to_string(&file).unwrap();
    std::fs::remove_file(&file).unwrap();
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn check(records: &[Value]) {
    let field = |r: &Value, k: &str| r[k].as_str().unwrap_or("").to_string();
    let starts: Vec<u64> = records
        .iter()
        .map(|r| r["start"].as_u64().unwrap())
        .collect();
    assert!(starts.windows(2).all(|w| w[0] <= w[1]), "ordered by start");
    assert!(records
        .iter()
        .all(|r| r["end"].as_u64().unwrap() >= r["start"].as_u64().unwrap()));
    // hello: two loads into the SMEM, the matmul, the copy back.
    let applies: Vec<&Value> = records
        .iter()
        .filter(|r| field(r, "thread") == "worker" && field(r, "state") == "apply")
        .collect();
    let ops: Vec<String> = applies.iter().map(|r| field(r, "op")).collect();
    assert_eq!(ops, ["dma", "dma", "matmul", "dma"]);
    let seqs: Vec<u64> = applies.iter().map(|r| r["seq"].as_u64().unwrap()).collect();
    assert_eq!(seqs, [1, 2, 3, 4]);
    assert_eq!(applies[0]["bytes"], 32768);
    assert_eq!(
        (applies[2]["m"].clone(), applies[2]["k"].clone()),
        (128.into(), 128.into())
    );
    // The timing thread runs the event loop, then waits for the worker to
    // apply the ops left, at the last cycle.
    let control: Vec<&Value> = records
        .iter()
        .filter(|r| field(r, "thread") == "control")
        .collect();
    assert_eq!(field(control[0], "state"), "run");
    let last = control.last().unwrap();
    assert_eq!(field(last, "state"), "finish_wait");
    assert_eq!(
        (last["cycle"].clone(), last["seq"].clone()),
        (1837.into(), 4.into())
    );
}

#[test]
fn the_numeric_backend_records_both_threads() {
    let records = timeline("numeric");
    check(&records);
    assert!(records.iter().all(|r| r.get("child_ns").is_none()));
}

#[test]
fn the_python_plane_reports_its_time_per_op() {
    let records = timeline("python");
    check(&records);
    let applies = records.iter().filter(|r| r["state"] == "apply");
    assert!(applies
        .into_iter()
        .all(|r| r["child_ns"].as_u64().is_some()));
}
