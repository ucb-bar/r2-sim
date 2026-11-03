//! The tracer export (--trace-out) and the raw trace dump (--trace-raw)
//! on the RV fixture three-level.bin on configs/2p-smem.yaml: the device
//! starts a program on each cluster, which starts a matmul program on each
//! of its four SMs.

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

/// The arguments of the three-level run.
const THREE: &[&str] = &[
    "run",
    "--config",
    "configs/2p-smem.yaml",
    "--program",
    "rv",
    "--image",
    "crates/nu-rv/tests/fixtures/rv/three-level.toml",
];

/// Runs three-level with `extra` arguments; returns the --stats JSON.
fn run_three(dir: &std::path::Path, extra: &[&str]) -> serde_json::Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let stats = dir.join("run.json");
    let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
        .current_dir(&root)
        .args(THREE)
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
    let j = run_three(&tmp, &["--trace-raw", plain.to_str().unwrap()]);
    run_three(
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
    // No program reports phases; the key stays, an empty list.
    assert_eq!(summary["phases"], serde_json::json!([]));
    // Each node gives its Neutrino's table sizes (sm-smem.yaml for an SM).
    let sm1 = manifest["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["path"] == "root.gpc0.sm1")
        .unwrap();
    for (k, v) in [
        ("scoreboard", 128),
        ("execution", 256),
        ("pending", 256),
        ("versions", 256),
        ("inbound_refs", 128),
        ("inbound_invokes", 64),
        ("outbound_deps", 128),
        ("outbound_invokes", 32),
    ] {
        assert_eq!(sm1["neutrino"][k], v, "{k}");
    }
    for (stem, size) in [
        ("tasks", 120),
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
    // Every task has a record; per rated resource the charges' slot-cycles
    // (usage / 65536 x span, trace-raw 1.7) match its busy slot-cycles
    // within the rounding of each record's usage (span / 65536).
    assert_eq!(summary["counts"]["tasks"], j["tasks"]);
    let charges = std::fs::read(plain.join("charges.bin")).unwrap();
    let mut sc: std::collections::HashMap<u16, (f64, f64)> = Default::default();
    for c in charges.chunks(32) {
        let r = u16::from_le_bytes([c[28], c[29]]);
        let span = u32::from_le_bytes([c[16], c[17], c[18], c[19]]) as f64;
        let usage = u32::from_le_bytes([c[20], c[21], c[22], c[23]]) as f64;
        let e = sc.entry(r).or_default();
        e.0 += usage / 65536.0 * span;
        e.1 += span / 65536.0;
    }
    let res = manifest["resources"].as_array().unwrap();
    for r in j["resources"].as_array().unwrap() {
        let m = res.iter().find(|m| m["name"] == r["name"]).unwrap();
        let id = m["id"].as_u64().unwrap() as u16;
        let (got, tol) = sc.get(&id).copied().unwrap_or((0.0, 0.0));
        let want = r["busy_slot_cycles"].as_f64().unwrap();
        assert!(
            (got - want).abs() <= tol + 1e-6,
            "{}: {got} vs {want}",
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
    run_three(&tmp, &["--trace-raw", all.to_str().unwrap()]);
    run_three(
        &tmp,
        &[
            "--trace-raw",
            part.to_str().unwrap(),
            "--set",
            "trace.records=[tasks, deps]",
            "--set",
            "trace.nodes=[root.gpc0.sm1]",
            "--set",
            "trace.window=[500, 1500]",
        ],
    );
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(part.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["filters"]["partial"], true);
    assert_eq!(
        manifest["filters"]["window"],
        serde_json::json!([500, 1500])
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
        b.chunks(120)
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
        .filter(|t| t.1 == sm1 && t.2 < 1500 && t.3 > 500)
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
    let j = run_three(&tmp, &["--bin", "1000", "--trace-out", dir]);
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
        // Busy slot-cycles over the resource's slots.
        let busy = r["busy_slot_cycles"].as_f64().unwrap() / r["slots"].as_f64().unwrap().max(1e-9);
        let got = sum.get(r["name"].as_str().unwrap()).copied().unwrap_or(0.0);
        assert!(
            (got - busy).abs() <= 1.0 + busy * 1e-3,
            "{}: {got} vs {busy}",
            r["name"]
        );
    }
    // No program reports phases: phases.csv has its header only. The
    // tree has the root.
    let phases = std::fs::read_to_string(tmp.join("phases.csv")).unwrap();
    assert_eq!(
        phases.lines().collect::<Vec<_>>(),
        ["stage,name,layer,start,end,estimate,roofline,bound_by"]
    );
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
                "configs/2p-smem.yaml",
                "--program",
                "rv",
                "--image",
                "crates/nu-rv/tests/fixtures/rv/hello.toml",
                "--set",
                set,
            ])
            .arg("--stats")
            .arg(&stats)
            .output()
            .unwrap();
        (out, stats)
    };
    // A faster gpc0 slice shows in the slots of its port: 546 B per cycle
    // over 64 B slots; gpc1 keeps 512 B, 8 slots.
    let (out, stats) = run("root.gpc0.hbm.bw_bytes=546");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&stats).unwrap()).unwrap();
    let slots = |name: &str| {
        j["resources"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == name)
            .unwrap()["slots"]
            .as_f64()
            .unwrap()
    };
    assert_eq!(
        (slots("root.gpc0.hbm.port"), slots("root.gpc1.hbm.port")),
        (546.0 / 64.0, 8.0)
    );
    let (out, _) = run("root.nosuch*.smem.bw_bytes=1");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("nothing matches"));
}

/// RV64 encodings of the few instructions of an untracked program.
mod rv {
    pub const T0: u32 = 5;
    pub const T1: u32 = 6;
    pub const A0: u32 = 10;
    fn i(op: u32, f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32 & 0xfff) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
    }
    /// `ld rd, CONST + 8 k(x0)`: constant `k`.
    pub fn k(rd: u32, k: i32) -> u32 {
        i(0x03, 3, rd, 0, super::CONST as i32 + 8 * k)
    }
    pub fn sd(rs2: u32, rs1: u32, imm: i32) -> u32 {
        let imm = imm as u32 & 0xfff;
        ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (3 << 12) | ((imm & 0x1f) << 7) | 0x23
    }
    fn nu(f3: u32, rd: u32, rs1: u32) -> u32 {
        (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x0b
    }
    pub fn invoke(rd: u32, rs1: u32) -> u32 {
        nu(0, rd, rs1)
    }
    /// `nu.wait x0`: the fence.
    pub fn fence() -> u32 {
        nu(1, 0, 0)
    }
    pub fn exit() -> u32 {
        nu(5, 0, 0)
    }
}

/// The IMEM offset of the constants.
const CONST: usize = 0x400;

/// An IMEM image: the trap word at 0, `code` from 4, the constants at
/// [`CONST`], and the root record (entry 4, ABI version 3) after them.
fn imem(code: &[u32], consts: &[u64]) -> Vec<u8> {
    let mut m = vec![0u8; 0x800];
    for (i, w) in code.iter().enumerate() {
        m[4 + 4 * i..8 + 4 * i].copy_from_slice(&w.to_le_bytes());
    }
    for (i, v) in consts.iter().enumerate() {
        m[CONST + 8 * i..CONST + 8 * i + 8].copy_from_slice(&v.to_le_bytes());
    }
    let r = 0x700;
    m[r..r + 8].copy_from_slice(&4u64.to_le_bytes());
    m[r + 8..r + 12].copy_from_slice(&sim_core::abi::VERSION.to_le_bytes());
    m[r + 24..r + 32].copy_from_slice(b"untrackd");
    m[r + 56..r + 60].copy_from_slice(b"root");
    m
}

/// A root with a DMA engine, an HBM, a buffer and a sync-only range (its
/// data scratch at the end of it).
const UNTRACKED_CONFIG: &str = "global:
  neutrino_granule: 512
  address: {fields: [{name: sm, bits: 2}], offset_bits: 24}
device:
  name: root
  fabric:
    - {name: sbus}
    - {name: cbus, control: true}
  memory:
    - {name: hbm, type: dram, addr: [[0x200000, 0x400000]], bw_bytes: 64, latency: 50, slave: $this.sbus}
    - {name: buf, type: scratch, addr: [[0x40000, 0x80000]], bw_bytes: 64, slave: $this.sbus}
    - {name: sync, type: sync, addr: [[0xE00000, 0xF00000]]}
  compute:
    - {name: dma, type: dma, latency: 4, master: [$this.sbus]}
  neutrino: {}
";

#[test]
fn trace_raw_flags_untracked_tasks() {
    // On a root with a DMA engine, an HBM and a buffer, the root copies four
    // 16 KiB pieces of the buffer into the HBM as untracked DMA tasks on its
    // own engine, fences, and exits. The DMA records have flags bit 0,
    // version, queued and sent null, and nu = src_nu (the invoking root).
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("trace_untracked");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let config = tmp.join("root.yaml");
    std::fs::write(&config, UNTRACKED_CONFIG).unwrap();
    let b = sim_core::build_config_file::<&str>(
        &config,
        &[],
        sim_core::ProfileRegistry::with_builtins(),
    )
    .unwrap();
    let abi = nu_rv::Abi::new(&b);
    let me = &abi.node(nu_model::NeutrinoId(0));
    let scratch = me.scratch.clone().unwrap().base;
    let pe = me.pes.iter().position(|&p| p == b.pe("root.dma")).unwrap() as u16;
    let (smem, hbm) = (b.memory("root.buf").base, b.memory("root.hbm").base);
    // A descriptor (ABI version 3): the header (opcode 2, untracked flag
    // 2, dst in 512 B granules) and the DMA's 32 bytes of arguments.
    let piece = 16384u64;
    let desc = |i: u64| {
        let (src, dst) = (smem + i * piece, hbm + i * piece);
        let mut v = Vec::new();
        v.extend(2u16.to_le_bytes());
        v.extend(pe.to_le_bytes());
        v.extend(2u16.to_le_bytes());
        v.extend(0u16.to_le_bytes());
        v.extend((dst >> 9).to_le_bytes());
        v.extend(((piece >> 9) as u32).to_le_bytes());
        v.extend(u32::MAX.to_le_bytes());
        v.extend(src.to_le_bytes());
        v.extend(dst.to_le_bytes());
        for w in [piece as u32, 1, 0, 0] {
            v.extend(w.to_le_bytes());
        }
        v
    };
    // The constants: the four descriptors' words, then their addresses in
    // the data scratch (64 B apart from scratch + 0x100).
    let mut consts: Vec<u64> = Vec::new();
    for i in 0..4 {
        consts.extend(
            desc(i)
                .chunks(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
        );
    }
    let words = consts.len() / 4;
    let at = consts.len() as i32;
    consts.extend((0..4).map(|i| scratch + 0x100 + 64 * i));
    let mut code = Vec::new();
    for i in 0..4 {
        code.push(rv::k(rv::T0, at + i as i32));
        for w in 0..words {
            code.push(rv::k(rv::T1, (i * words + w) as i32));
            code.push(rv::sd(rv::T1, rv::T0, 8 * w as i32));
        }
        code.push(rv::invoke(rv::A0, rv::T0));
    }
    code.extend([rv::fence(), rv::exit()]);
    std::fs::write(tmp.join("untracked.bin"), imem(&code, &consts)).unwrap();
    std::fs::write(
        tmp.join("image.toml"),
        "[[entry]]\nimem = \"root\"\nbin = \"untracked.bin\"\n",
    )
    .unwrap();
    let raw = tmp.join("raw");
    let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
        .args(["run", "--config"])
        .arg(&config)
        .args(["--program", "rv", "--image"])
        .arg(tmp.join("image.toml"))
        .arg("--trace-raw")
        .arg(&raw)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(raw.join("manifest.json")).unwrap()).unwrap();
    let layout = manifest["records"]["tasks"].as_array().unwrap();
    assert_eq!(layout[17], serde_json::json!(["flags", "u32"]));
    let b = std::fs::read(raw.join("tasks.bin")).unwrap();
    assert_eq!(b.len() % 120, 0);
    let u32_at = |r: &[u8], o: usize| u32::from_le_bytes(r[o..o + 4].try_into().unwrap());
    let u64_at = |r: &[u8], o: usize| u64::from_le_bytes(r[o..o + 8].try_into().unwrap());
    let (mut untracked, mut tracked) = (0, 0);
    for r in b.chunks(120) {
        if u32_at(r, 104) & 1 == 1 {
            untracked += 1;
            assert_eq!(u32_at(r, 12), u32::MAX, "version");
            assert_eq!((u64_at(r, 88), u64_at(r, 96)), (u64::MAX, u64::MAX));
            assert_eq!(r[4..6], r[6..8], "nu = src_nu");
        } else {
            tracked += 1;
        }
        assert_eq!(u32_at(r, 108), u32::MAX, "dst_mask: a plain range");
    }
    // The four untracked DMA tasks, and no other task.
    assert_eq!((untracked, tracked), (4, 0));
}
