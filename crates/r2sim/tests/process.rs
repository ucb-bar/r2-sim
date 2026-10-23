//! The lifecycle of the process backend: the child dies with r2-sim on a
//! normal exit, on an error, on a panic, and when r2-sim is killed. The
//! children are `tests/func_child.py` (misbehaving on purpose) and the
//! Python plane.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const R2SIM: &str = env!("CARGO_BIN_EXE_r2sim");

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh directory for one test.
fn dir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("r2sim-process-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The command line of the test child in `mode`.
fn child(mode: &str) -> String {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/func_child.py");
    format!("python {} {mode}", script.display())
}

fn alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Waits up to `secs` for `f`.
fn within(secs: u64, f: impl Fn() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    f()
}

/// The pid the child wrote, once it is there.
fn child_pid(file: &Path) -> u32 {
    assert!(
        within(30, || std::fs::read_to_string(file)
            .is_ok_and(|s| !s.is_empty())),
        "the child wrote no pid"
    );
    std::fs::read_to_string(file)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

/// The process backend on the test child, which writes its pid to
/// `pidfile`.
fn spawn(cmd: &str, pidfile: &Path) -> func::Process {
    let words: Vec<String> = cmd.split_whitespace().map(String::from).collect();
    let env = [("R2SIM_CHILD_PID".to_string(), pidfile.display().to_string())];
    func::Process::spawn_with_env(&words, &env, "child: ").unwrap()
}

/// r2sim running hello on 2p.yaml with the process backend `cmd`.
fn r2sim(cmd: &str, pidfile: &Path) -> Command {
    let mut c = Command::new(R2SIM);
    c.args(["run", "--config"])
        .arg(root().join("configs/2p.yaml"))
        .args([
            "--program",
            "hello",
            "--functional",
            "python",
            "--functional-cmd",
            cmd,
        ])
        .env("R2SIM_CHILD_PID", pidfile)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    c
}

#[test]
fn killing_r2sim_kills_the_child() {
    // The child never answers CONFIGURE: r2sim waits on it. SIGKILL to
    // r2sim runs no Drop; the parent-death signal kills the child.
    let d = dir("kill");
    let pidfile = d.join("pid");
    let mut r = r2sim(&child("hang"), &pidfile).spawn().unwrap();
    let pid = child_pid(&pidfile);
    assert!(alive(pid));
    r.kill().unwrap();
    r.wait().unwrap();
    assert!(within(5, || !alive(pid)), "child {pid} outlived r2sim");
}

#[test]
fn a_child_error_stops_the_run_and_the_child_exits() {
    let d = dir("fail");
    let pidfile = d.join("pid");
    let out = r2sim(&child("fail"), &pidfile)
        .stdout(Stdio::piped())
        .output()
        .unwrap();
    let pid = child_pid(&pidfile);
    assert!(!out.status.success());
    // The summary on stdout names the op and the child's message.
    let summary = String::from_utf8_lossy(&out.stdout);
    assert!(
        summary.contains("FATAL") && summary.contains("refused by the test"),
        "{summary}"
    );
    assert!(within(5, || !alive(pid)), "child {pid} outlived r2sim");
}

#[test]
fn drop_kills_a_child_that_does_not_exit() {
    // The child ignores FINISH and end of file: Drop kills and reaps it.
    let d = dir("deaf");
    let pidfile = d.join("pid");
    let p = spawn(&child("deaf"), &pidfile);
    let pid = p.id();
    assert!(alive(pid));
    let t = Instant::now();
    drop(p);
    // Reaped: no zombie is left either.
    assert!(!alive(pid), "child {pid} not reaped");
    assert!(t.elapsed() < Duration::from_secs(5));
}

#[test]
fn a_panic_reaps_the_child() {
    let d = dir("panic");
    let pidfile = d.join("pid");
    let caught = std::panic::catch_unwind(|| {
        let p = spawn(&child("deaf"), &pidfile);
        std::fs::write(d.join("spawned"), p.id().to_string()).unwrap();
        panic!("a panic with a live child");
    })
    .expect_err("panicked");
    drop(caught);
    let pid: u32 = std::fs::read_to_string(d.join("spawned"))
        .unwrap()
        .parse()
        .unwrap();
    assert!(!alive(pid), "child {pid} not reaped after the panic");
}

#[test]
fn the_child_exits_after_a_normal_run() {
    let d = dir("ok");
    let pidfile = d.join("pid");
    let out = r2sim(&child("ok"), &pidfile).output().unwrap();
    let pid = child_pid(&pidfile);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!alive(pid), "child {pid} outlived r2sim");
}
