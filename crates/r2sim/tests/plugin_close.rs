//! A plug-in's `close` runs while the store's blocks are still mapped: the
//! test plug-in (crates/r2-backend-test) applies the DMAs of hello and, at
//! close, reads a byte of a block. The run ends cleanly. A plug-in that
//! fails at configure stops the run with a fatal error.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The test plug-in, which cargo builds next to this test.
fn test_lib() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let deps = exe.parent().unwrap();
    [deps, deps.parent().unwrap()]
        .iter()
        .map(|d| d.join("libr2_backend_test.so"))
        .find(|p| p.exists())
        .expect("libr2_backend_test.so")
}

#[test]
fn a_plugin_reads_the_store_at_close() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let log = std::env::temp_dir().join(format!("r2sim-plugin-close-{}.log", std::process::id()));
    let entry = format!(
        "functional.backends.t={{lib: {:?}, log: {:?}, close_read: true}}",
        test_lib(),
        log
    );
    let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
        .args(["run", "--config"])
        .arg(root.join("configs/2p-smem.yaml"))
        .args(["--program", "rv", "--functional", "numeric", "--image"])
        .arg(root.join("crates/nu-rv/tests/fixtures/rv/hello.toml"))
        .args(["--set", &entry, "--set", "functional.units.dma=[t]"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let text = std::fs::read_to_string(&log).unwrap();
    std::fs::remove_file(&log).unwrap();
    // The three DMAs of hello, then the close.
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1 + 3 + 1, "{text}");
    assert!(lines[4].starts_with("close "), "{text}");
}

#[test]
fn a_plugin_that_fails_at_configure_stops_the_run() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let entry = format!(
        "functional.backends.t={{lib: {:?}, fail: configure}}",
        test_lib()
    );
    let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
        .args(["run", "--config"])
        .arg(root.join("configs/2p-smem.yaml"))
        .args(["--program", "rv", "--functional", "numeric", "--image"])
        .arg(root.join("crates/nu-rv/tests/fixtures/rv/hello.toml"))
        .args(["--set", &entry, "--set", "functional.units.dma=[t]"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stdout.contains("FATAL: Fatal { time: 0,")
            && stdout.contains("configure: the test backend fails at configure"),
        "{stdout}"
    );
    assert!(
        stderr.contains("the run stopped on a fatal error"),
        "{stderr}"
    );
}
