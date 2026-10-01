//! The tracer export (--trace-out) on the tiny config: a one-layer VLM
//! prefill of 64 tokens.

use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

fn gunzip(p: PathBuf) -> String {
    let mut s = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(p).unwrap())
        .read_to_string(&mut s)
        .unwrap();
    s
}

#[test]
fn trace_out_writes_consistent_files() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("trace_out");
    let _ = std::fs::remove_dir_all(&tmp);
    let stats = tmp.join("run.json");
    std::fs::create_dir_all(&tmp).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
        .current_dir(&root)
        .args([
            "run",
            "--config",
            "crates/nu-prog/tests/configs/tiny.yaml",
            "--program",
            "prefill",
            "--param",
            "layers=1",
            "--param",
            "d=256",
            "--param",
            "heads=2",
            "--param",
            "kv_heads=1",
            "--param",
            "head_dim=128",
            "--param",
            "ffn=256",
            "--param",
            "images=1",
            "--param",
            "tokens_per_image=64",
            "--param",
            "text_tokens=0",
            "--param",
            "quantize=false",
            "--param",
            "action={layers=1,tokens=16,d=256,ffn=128,heads=1,head_dim=128,prefix=64,experts=1}",
            "--bin",
            "1000",
        ])
        .arg("--stats")
        .arg(&stats)
        .arg("--trace-out")
        .arg(&tmp)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&stats).unwrap()).unwrap();
    // One task row per task, header aside.
    let tasks = gunzip(tmp.join("tasks.csv.gz"));
    assert_eq!(
        tasks.lines().count() as u64 - 1,
        j["tasks"].as_u64().unwrap()
    );
    // Binned busy cycles add up to each resource's busy total.
    let res = gunzip(tmp.join("resources.csv.gz"));
    let mut sum: std::collections::HashMap<String, f64> = Default::default();
    for l in res.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        *sum.entry(f[0].to_string()).or_default() += f[4].parse::<f64>().unwrap() * 1000.0;
    }
    for r in j["resources"].as_array().unwrap() {
        let busy = r["busy_cycles"].as_u64().unwrap() as f64;
        let got = sum.get(r["name"].as_str().unwrap()).copied().unwrap_or(0.0);
        assert!(
            (got - busy).abs() <= 1.0 + busy * 1e-3,
            "{}: {got} vs {busy}",
            r["name"]
        );
    }
    // Every GEMM phase is in phases.csv; the tree has the root.
    let phases = std::fs::read_to_string(tmp.join("phases.csv")).unwrap();
    assert!(phases.lines().any(|l| l.starts_with("prefill,up,")));
    let tree: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.join("tree.json")).unwrap()).unwrap();
    assert_eq!(tree[0]["node"], "root");
}

#[test]
fn set_overrides_a_config_value_and_rejects_a_bad_path() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("set");
    std::fs::create_dir_all(&tmp).unwrap();
    let run = |set: &str| {
        let stats = tmp.join("run.json");
        let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
            .current_dir(&root)
            .args([
                "run",
                "--config",
                "configs/2p.yaml",
                "--program",
                "adamoe",
                "--param",
                "steps=1",
                "--param",
                "layers=1",
                "--synthetic",
                "7",
                "--set",
                set,
            ])
            .arg("--stats")
            .arg(&stats)
            .output()
            .unwrap();
        (out, stats)
    };
    // A faster gpc0 slice shows in the JSON's HBM rate (546 + gpc1's 512).
    let (out, stats) = run("root.gpc0.hbm.bw_bytes=546");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&stats).unwrap()).unwrap();
    assert_eq!(j["hbm_rate_bytes_per_cycle"], 1058.0);
    let (out, _) = run("root.nosuch*.smem.bw_bytes=1");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("nothing matches"));
}
