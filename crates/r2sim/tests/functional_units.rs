//! The backends of the FU kinds from the command line: the short forms of
//! `--functional`, the override `--set functional.units.<kind>=[...]`, and
//! the errors of a bad list. The RV hello image on configs/2p-smem.yaml
//! with A and B all ones: C = A B is 128 everywhere when the Atlas
//! computes it.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const R2SIM: &str = env!("CARGO_BIN_EXE_r2sim");

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh directory with the image file of hello: A and B (128 x 128
/// bf16 ones) and C named for `--image-out`.
fn dir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("r2sim-units-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let ones: Vec<u8> = (0..2 * 128 * 128)
        .flat_map(|_| 0x3f80u16.to_le_bytes())
        .collect();
    std::fs::write(d.join("ab.bin"), ones).unwrap();
    let hello = root().join("crates/nu-rv/tests/fixtures/rv/hello.bin");
    std::fs::write(
        d.join("image.toml"),
        format!(
            "[[entry]]\nimem = \"root\"\nbin = {hello:?}\n\n[[entry]]\nat = \"root.gpc0.hbm+0x0\"\nbin = \"ab.bin\"\n\n[[entry]]\nname = \"C\"\nat = \"root.gpc0.hbm+0x10000\"\ndtype = \"bf16\"\nshape = [128, 128]\n"
        ),
    )
    .unwrap();
    d
}

/// Runs hello in `d` with `args`, the thread timeline and `--image-out`.
fn run(d: &Path, args: &[&str]) -> Output {
    Command::new(R2SIM)
        .args(["run", "--config"])
        .arg(root().join("configs/2p-smem.yaml"))
        .args(["--program", "rv", "--image"])
        .arg(d.join("image.toml"))
        .arg("--image-out")
        .arg(d.join("out.safetensors"))
        .arg("--thread-timeline")
        .arg(d.join("timeline.jsonl"))
        .arg("--stats")
        .arg(d.join("stats.json"))
        .args(args)
        .output()
        .unwrap()
}

/// The (op, backend) of each apply record of the timeline.
fn applies(d: &Path) -> Vec<(String, String)> {
    std::fs::read_to_string(d.join("timeline.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter(|r| r["state"] == "apply")
        .map(|r| {
            let s = |k: &str| r[k].as_str().unwrap().to_string();
            (s("op"), s("backend"))
        })
        .collect()
}

/// The bf16 values of C in the output file.
fn c(d: &Path) -> Vec<u16> {
    let b = std::fs::read(d.join("out.safetensors")).unwrap();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let head: Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let [lo, hi] = [0, 1].map(|i| head["C"]["data_offsets"][i].as_u64().unwrap() as usize);
    b[8 + n + lo..8 + n + hi]
        .chunks(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}

fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
    v.iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

#[test]
fn the_short_forms_select_one_backend_per_kind() {
    let d = dir("short");
    let out = run(&d, &["--functional", "builtin"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let builtin = [
        ("dma", "builtin"),
        ("dma", "builtin"),
        ("matmul", "builtin"),
        ("dma", "builtin"),
    ];
    assert_eq!(applies(&d), pairs(&builtin));
    assert_eq!(c(&d), vec![0x4300; 128 * 128], "C = 128");
    // No flag and no section: every list null, so an image with entries in
    // memory has no backend to go to.
    let out = run(&d, &[]);
    assert!(!out.status.success());
    let e = String::from_utf8_lossy(&out.stderr);
    assert!(
        e.contains(
            "--image with entries in memory (or --image-out) needs a functional backend with data"
        ),
        "{e}"
    );
}

#[test]
fn trace_is_not_a_built_in_backend() {
    let d = dir("trace");
    let want = "trace is not a built-in backend; a trace backend is a plug-in";
    for args in [
        &["--functional", "trace"][..],
        &[
            "--functional",
            "builtin",
            "--set",
            "functional.units.dma=[trace]",
        ],
    ] {
        let out = run(&d, args);
        assert!(!out.status.success());
        let e = String::from_utf8_lossy(&out.stderr);
        assert!(e.contains(want), "{e}");
    }
}

#[test]
fn a_null_atlas_in_a_builtin_run_warns_at_the_read_of_c() {
    let d = dir("mixed");
    let out = run(
        &d,
        &[
            "--functional",
            "builtin",
            "--set",
            "functional.units.atlas=[null]",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mixed = [
        ("dma", "builtin"),
        ("dma", "builtin"),
        ("matmul", "null"),
        ("dma", "builtin"),
    ];
    assert_eq!(applies(&d), pairs(&mixed));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("warning NullRead") && stdout.contains("[0x1200010000, 0x1200018000)"),
        "{stdout}"
    );
    let stats: Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("stats.json")).unwrap()).unwrap();
    assert_eq!(stats["warnings"]["NullRead"], 1);
    assert_eq!(c(&d), vec![0; 128 * 128], "C reads as zero");
}

#[test]
fn a_bad_functional_section_is_an_error_that_names_the_key() {
    let d = dir("bad");
    let err = |args: &[&str]| {
        let out = run(&d, args);
        assert!(!out.status.success());
        String::from_utf8_lossy(&out.stderr).to_string()
    };
    let e = err(&["--set", "functional.units.atlas=[foo]"]);
    assert!(
        e.contains("functional.units.atlas: unknown backend \"foo\""),
        "{e}"
    );
    let missing = [
        "--set",
        "functional.backends.py={lib: /nonexistent/libpy.so}",
        "--set",
        "functional.units.atlas=[py]",
    ];
    let e = err(&missing);
    assert!(
        e.contains("functional.units.atlas: backend py (/nonexistent/libpy.so): "),
        "{e}"
    );
    let no_lib = [
        "--set",
        "functional.backends.py={provider: numpy}",
        "--set",
        "functional.units.atlas=[py]",
    ];
    let e = err(&no_lib);
    assert!(e.contains("backend \"py\" has no lib"), "{e}");
}
