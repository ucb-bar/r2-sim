//! The raw trace 1.10 of RV programs (trace-raw.md): the `pc` of each
//! task, the `image` of each program, the manifest's `images`, and the
//! instruction DMAs and data scratch ports in the manifest. A Rust
//! program's dump has none of them.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

const NULL32: u32 = u32::MAX;
const THREE: &str = "../nu-rv/tests/fixtures/rv/three-level.bin";

fn dir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("r2sim-trace-rv-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Runs r2sim on configs/2p-smem.yaml with `args` and a raw dump into
/// `d/raw`; the manifest.
fn run(d: &Path, args: &[&str]) -> Value {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
        .args(["run", "--config"])
        .arg(root.join("../../configs/2p-smem.yaml"))
        .args(args)
        .arg("--trace-raw")
        .arg(d.join("raw"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_str(&std::fs::read_to_string(d.join("raw/manifest.json")).unwrap()).unwrap()
}

fn u16_at(r: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([r[o], r[o + 1]])
}

fn u32_at(r: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(r[o..o + 4].try_into().unwrap())
}

fn records(d: &Path, stem: &str, size: usize) -> Vec<Vec<u8>> {
    let b = std::fs::read(d.join(format!("raw/{stem}.bin"))).unwrap();
    assert_eq!(b.len() % size, 0);
    b.chunks(size).map(|c| c.to_vec()).collect()
}

/// Every charge's resource and every task's PE is in the manifest.
fn ids_resolve(d: &Path, m: &Value) {
    let (nres, npe) = (
        m["resources"].as_array().unwrap().len(),
        m["pes"].as_array().unwrap().len(),
    );
    for c in records(d, "charges", 32) {
        assert!((u16_at(&c, 28) as usize) < nres);
    }
    for t in records(d, "tasks", 120) {
        let pe = u16_at(&t, 8);
        assert!(pe == u16::MAX || (pe as usize) < npe);
    }
}

#[test]
fn the_three_level_dump() {
    let d = dir("three");
    let image = d.join("image.toml");
    // The image file names the binary relative to itself.
    let bin = d.join("three-level.bin");
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join(THREE), &bin).unwrap();
    std::fs::write(
        &image,
        "[[entry]]\nimem = \"root\"\nbin = \"three-level.bin\"\n",
    )
    .unwrap();
    let m = run(
        &d,
        &[
            "--program",
            "rv",
            "--functional",
            "builtin",
            "--image",
            image.to_str().unwrap(),
        ],
    );
    assert_eq!(m["version"], serde_json::json!([1, 11]));
    // Three images in order of first use.
    let images = m["images"].as_array().unwrap();
    let want = [
        (
            848,
            "8cc9f73f886b153d768d792d934bc68dc338d30666db309596f1b14c8d34bda3",
            true,
        ),
        (
            560,
            "efa4858698d80443ce1869c1b0a10d07fd539ecca8c8475f766f8a5177b66572",
            false,
        ),
        (
            240,
            "e6b1388e9ac0dd37125398bbb9536a39ab9183dccef13912356cc30722e91e11",
            false,
        ),
    ];
    assert_eq!(images.len(), 3);
    for (img, (bytes, sha, file)) in images.iter().zip(want) {
        assert_eq!(
            (img["bytes"].as_u64(), img["sha256"].as_str()),
            (Some(bytes), Some(sha))
        );
        assert_eq!(img["file"].is_string(), file);
        assert_eq!(img["imem"].is_string(), file);
    }
    let progs = |i: usize| -> Vec<&str> {
        let mut v: Vec<&str> = images[i]["programs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        v.sort();
        v
    };
    assert_eq!(progs(0), ["device", "gpc_fanout", "sm_tile"]);
    assert_eq!(images[0]["imem"], "root");
    // The binary's canonical absolute path, which a viewer can open.
    let canonical = std::fs::canonicalize(&bin).unwrap();
    assert_eq!(images[0]["file"].as_str(), canonical.to_str());
    // Each program's image and name.
    let strings: Vec<String> = std::fs::read_to_string(d.join("raw/strings.txt"))
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    let nodes = m["nodes"].as_array().unwrap();
    let path = |nu: u16| nodes[nu as usize]["path"].as_str().unwrap().to_string();
    let mut seen = Vec::new();
    for p in records(&d, "programs", 32) {
        let (nu, name, image) = (u16_at(&p, 0), u32_at(&p, 4), u32_at(&p, 12));
        let name = strings[name as usize].clone();
        let want = match path(nu).matches('.').count() {
            0 => ("device", 0),
            1 => ("gpc_fanout", 1),
            _ => ("sm_tile", 2),
        };
        assert_eq!((name.as_str(), image), want, "{}", path(nu));
        seen.push(name);
    }
    assert_eq!(seen.len(), 11);
    // The pcs: the device's IMEM copies at 0x90 and spawns at 0xae, the
    // matmuls at 0x8c; the copies run on their node's instruction DMA.
    let opcodes: Vec<&str> = m["opcodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_str().unwrap())
        .collect();
    let pes = m["pes"].as_array().unwrap();
    for t in records(&d, "tasks", 120) {
        let (nu, pe, op, pc) = (
            u16_at(&t, 4),
            u16_at(&t, 8),
            opcodes[u16_at(&t, 10) as usize],
            u32_at(&t, 112),
        );
        assert_ne!(pc, NULL32);
        match (op, nu) {
            ("imem_copy", 0) => assert_eq!(pc, 0x90),
            ("spawn", 0) => assert_eq!(pc, 0xae),
            ("atlas.matmul", _) => assert_eq!(pc, 0x8c),
            _ => {}
        }
        if op == "imem_copy" {
            assert_eq!(pes[pe as usize]["type"], "idma");
            assert_eq!(
                pes[pe as usize]["path"].as_str().unwrap(),
                format!("{}.idma", path(nu))
            );
        }
    }
    // The instruction DMAs and the data scratch ports are in the manifest.
    let res = m["resources"].as_array().unwrap();
    let idma = res.iter().find(|r| r["name"] == "root.idma").unwrap();
    assert_eq!(idma["role"]["type"], "pe");
    let port = res
        .iter()
        .find(|r| r["name"] == "root.scratch.port")
        .unwrap();
    let mem = &m["memories"][port["role"]["memory"].as_u64().unwrap() as usize];
    assert_eq!(
        (mem["path"].as_str(), mem["type"].as_str()),
        (Some("root.scratch"), Some("nu_scratch"))
    );
    ids_resolve(&d, &m);
}

#[test]
fn a_rust_program_has_no_pc_image_or_images() {
    // A micro-benchmark: two SMs copy into their gpc's DRAM.
    let d = dir("micro");
    let m = run(
        &d,
        &[
            "--program",
            "micro",
            "--param",
            "mode=untracked",
            "--param",
            "sms=2",
        ],
    );
    assert_eq!(m["version"], serde_json::json!([1, 11]));
    assert!(m.get("images").is_none());
    for t in records(&d, "tasks", 120) {
        assert_eq!(u32_at(&t, 112), NULL32);
    }
    for p in records(&d, "programs", 32) {
        assert_eq!(u32_at(&p, 12), NULL32);
    }
    let res = m["resources"].as_array().unwrap();
    assert!(!res.iter().any(|r| {
        let n = r["name"].as_str().unwrap();
        n.ends_with(".idma") || n.ends_with(".scratch.port")
    }));
    assert!(!m["pes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p["type"] == "idma"));
    ids_resolve(&d, &m);
}

#[test]
fn a_muon_task_of_an_rv_run_is_labelled_with_its_kernel_and_flags() {
    // probe2: a 2D copy, a matmul, the kernel epilogue with BIAS and an
    // opaque task.
    let d = dir("probe2");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::fs::copy(
        root.join("../nu-rv/tests/fixtures/rv/probe2.bin"),
        d.join("probe2.bin"),
    )
    .unwrap();
    let image = d.join("image.toml");
    std::fs::write(&image, "[[entry]]\nimem = \"root\"\nbin = \"probe2.bin\"\n").unwrap();
    let m = run(&d, &["--program", "rv", "--image", image.to_str().unwrap()]);
    let ops: Vec<&str> = m["opcodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_str().unwrap())
        .collect();
    let labels: Vec<(&str, u64)> = records(&d, "tasks", 120)
        .iter()
        .map(|t| {
            let label = u64::from_le_bytes(t[16..24].try_into().unwrap());
            (ops[u16_at(t, 10) as usize], label)
        })
        .collect();
    // epilogue (2) with BIAS (2) in the flags word, then the opaque task
    // (0); the others none.
    const BIAS: u64 = 2;
    assert_eq!(
        labels
            .iter()
            .filter(|l| l.0 == "muon.vector")
            .collect::<Vec<_>>(),
        [&("muon.vector", 2 | BIAS << 32), &("muon.vector", 0)]
    );
    assert!(labels
        .iter()
        .all(|l| l.0 == "muon.vector" || l.1 == u64::MAX));
    // The schema names the kernel and the flags: "epilogue+bias".
    let s = &m["label"];
    let f = |name: &str| {
        s["fields"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["name"] == name)
            .unwrap()
            .clone()
    };
    assert_eq!(
        f("kind"),
        serde_json::json!({"name": "kind", "shift": 0, "bits": 16, "enum":
            ["opaque", "row_stats", "epilogue", "softmax_stats", "softmax_apply"]})
    );
    assert_eq!(
        f("bias"),
        serde_json::json!({"name": "bias", "shift": 33, "bits": 1, "enum": ["", "+bias"]})
    );
    assert_eq!(
        f("gelu_tanh"),
        serde_json::json!({"name": "gelu_tanh", "shift": 35, "bits": 1, "enum": ["", "+gelu_tanh"]})
    );
    assert_eq!(s["format"], "{kind}{norm}{bias}{res}{gelu_tanh}{c_f32}");
    // The format applied to the label.
    let label = 2 | BIAS << 32;
    let mut text = s["format"].as_str().unwrap().to_string();
    for field in s["fields"].as_array().unwrap() {
        let v = (label >> field["shift"].as_u64().unwrap())
            & ((1 << field["bits"].as_u64().unwrap()) - 1);
        let name = field["enum"][v as usize].as_str().unwrap();
        text = text.replace(&format!("{{{}}}", field["name"].as_str().unwrap()), name);
    }
    assert_eq!(text, "epilogue+bias");
}
