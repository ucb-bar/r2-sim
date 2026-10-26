//! The ISA tests of the scalar core on nu-iss: the suites rv64ui (but
//! fence_i), rv64um, rv64uc and nu of the submodule isa-tests, built with
//! the clang of the pixi environment.

use std::path::Path;

use nu_isa_test::{build, run_iss, tests, Outcome};

#[test]
fn the_isa_tests_pass_on_nu_iss() {
    let isa = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../isa-tests");
    assert!(
        isa.join("third_party/riscv-tests/isa/rv64ui").is_dir(),
        "no ISA tests in {}: run `git submodule update --init --recursive isa-tests`",
        isa.display()
    );
    let clang = std::process::Command::new("clang")
        .arg("--version")
        .output();
    let version = clang.map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    assert!(
        version
            .as_deref()
            .is_ok_and(|v| v.contains("clang version")),
        "no clang: run the tests under `pixi run`"
    );
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("isa");
    std::fs::create_dir_all(&out).unwrap();
    let all = tests(&isa, &["rv64ui", "rv64um", "rv64uc", "nu"]).unwrap();
    let mut failed = Vec::new();
    for t in &all {
        let outcome = match build(&isa, "clang", t, &out, None) {
            Ok(elf) => run_iss(&elf),
            Err(e) => Outcome::Error(format!("build: {e}")),
        };
        match outcome {
            Outcome::Pass { .. } => {}
            Outcome::Fail { case, detail } => {
                failed.push(format!("{}-{}: case {case} ({detail})", t.suite, t.name))
            }
            Outcome::Error(e) => failed.push(format!("{}-{}: {e}", t.suite, t.name)),
        }
    }
    println!("{} ISA tests ran on nu-iss", all.len());
    assert!(failed.is_empty(), "failed:\n{}", failed.join("\n"));
}
