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

/// The arguments of a one-layer VLM prefill of 64 tokens on the tiny
/// config.
const TINY: &[&str] = &[
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
];

/// Runs the tiny prefill with `extra` arguments; returns the --stats JSON.
fn run_tiny(dir: &std::path::Path, extra: &[&str]) -> serde_json::Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let stats = dir.join("run.json");
    let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
        .current_dir(&root)
        .args(TINY)
        .args(extra)
        .arg("--stats")
        .arg(&stats)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_str(&std::fs::read_to_string(&stats).unwrap()).unwrap()
}

#[test]
fn trace_raw_writes_whole_records_that_match_the_stats() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("trace_raw");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let (plain, packed) = (tmp.join("plain"), tmp.join("zstd"));
    let j = run_tiny(&tmp, &["--trace-raw", plain.to_str().unwrap()]);
    run_tiny(
        &tmp,
        &["--trace-raw", packed.to_str().unwrap(), "--trace-zstd"],
    );
    let read = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    };
    let (manifest, summary) = (
        read(plain.join("manifest.json")),
        read(plain.join("summary.json")),
    );
    assert_eq!(summary["end_time"], j["end_time"]);
    // Each node gives its Neutrino's table sizes (tiny_sm.yaml for an SM):
    let sm1 = manifest["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["path"] == "root.gpc0.sm1")
        .unwrap();
    // Set in the file; versions and inbound_invokes take their defaults.
    for (k, v) in [
        ("scoreboard", 64),
        ("execution", 64),
        ("pending", 16),
        ("inbound_refs", 16),
        ("outbound_deps", 16),
        ("outbound_invokes", 16),
    ] {
        assert_eq!(sm1["neutrino"][k], v, "{k}");
    }
    for k in ["versions", "inbound_invokes"] {
        assert!(sm1["neutrino"][k].as_u64().unwrap() > 0, "{k}");
    }
    for (stem, size) in [
        ("tasks", 104),
        ("chunks", 32),
        ("deps", 24),
        ("charges", 32),
        ("programs", 32),
        ("marks", 16),
    ] {
        let layout = manifest["records"][stem].as_array().unwrap();
        let bytes: u64 = layout
            .iter()
            .map(|f| match f[1].as_str().unwrap() {
                "u8" => 1,
                "u16" => 2,
                "u32" => 4,
                _ => 8,
            })
            .sum();
        assert_eq!(bytes, size, "{stem} layout");
        let data = std::fs::read(plain.join(format!("{stem}.bin"))).unwrap();
        let count = summary["counts"][stem].as_u64().unwrap();
        assert_eq!(data.len() as u64, count * size, "{stem}");
        // The zstd dump holds the same records.
        let z = std::fs::File::open(packed.join(format!("{stem}.bin.zst"))).unwrap();
        assert_eq!(zstd::decode_all(z).unwrap(), data, "{stem} compressed");
    }
    // Every task has a record; per rated resource the charge cycles sum to
    // its busy cycles.
    assert_eq!(summary["counts"]["tasks"], j["tasks"]);
    let charges = std::fs::read(plain.join("charges.bin")).unwrap();
    let mut cycles: std::collections::HashMap<u16, u64> = Default::default();
    for c in charges.chunks(32) {
        let r = u16::from_le_bytes([c[28], c[29]]);
        *cycles.entry(r).or_default() += u32::from_le_bytes([c[20], c[21], c[22], c[23]]) as u64;
    }
    let res = manifest["resources"].as_array().unwrap();
    for r in j["resources"].as_array().unwrap() {
        let m = res.iter().find(|m| m["name"] == r["name"]).unwrap();
        let id = m["id"].as_u64().unwrap() as u16;
        assert_eq!(
            cycles.get(&id).copied().unwrap_or(0),
            r["busy_cycles"].as_u64().unwrap(),
            "{}",
            r["name"]
        );
    }
}

#[test]
fn trace_raw_filters_keep_the_subtree_and_the_window() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("trace_raw_filter");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let (all, part) = (tmp.join("all"), tmp.join("part"));
    run_tiny(&tmp, &["--trace-raw", all.to_str().unwrap()]);
    run_tiny(
        &tmp,
        &[
            "--trace-raw",
            part.to_str().unwrap(),
            "--set",
            "trace.records=[tasks, deps]",
            "--set",
            "trace.nodes=[root.gpc0.sm1]",
            "--set",
            "trace.window=[2000, 6000]",
        ],
    );
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(part.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["filters"]["partial"], true);
    assert_eq!(
        manifest["filters"]["window"],
        serde_json::json!([2000, 6000])
    );
    assert!(!part.join("charges.bin").exists());
    let sm1 = manifest["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["path"] == "root.gpc0.sm1")
        .unwrap()["id"]
        .as_u64()
        .unwrap() as u16;
    // (gid, nu, invoke, retire) of each task record.
    let tasks = |dir: &PathBuf| -> Vec<(u32, u16, u64, u64)> {
        let b = std::fs::read(dir.join("tasks.bin")).unwrap();
        let u64_at = |r: &[u8], o: usize| u64::from_le_bytes(r[o..o + 8].try_into().unwrap());
        b.chunks(104)
            .map(|r| {
                let gid = u32::from_le_bytes(r[0..4].try_into().unwrap());
                (
                    gid,
                    u16::from_le_bytes([r[4], r[5]]),
                    u64_at(r, 48),
                    u64_at(r, 80),
                )
            })
            .collect()
    };
    // The kept tasks are the complete dump's tasks of sm1 that overlap the
    // window.
    let mut want: Vec<_> = tasks(&all)
        .into_iter()
        .filter(|t| t.1 == sm1 && t.2 < 6000 && t.3 > 2000)
        .collect();
    let mut got = tasks(&part);
    want.sort();
    got.sort();
    assert!(!got.is_empty());
    assert_eq!(got, want);
    // A dep is kept with its consumer.
    let deps = std::fs::read(part.join("deps.bin")).unwrap();
    let kept: std::collections::HashSet<u32> = got.iter().map(|t| t.0).collect();
    assert!(deps
        .chunks(24)
        .all(|d| kept.contains(&u32::from_le_bytes(d[0..4].try_into().unwrap()))));
}

#[test]
fn trace_out_writes_consistent_files() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("trace_out");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = tmp.to_str().unwrap();
    let j = run_tiny(&tmp, &["--bin", "1000", "--trace-out", dir]);
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
        // Busy lane-cycles over the resource's lanes.
        let busy =
            r["busy_cycles"].as_u64().unwrap() as f64 / r["lanes"].as_u64().unwrap().max(1) as f64;
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
