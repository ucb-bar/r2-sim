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
            "numeric",
            "--image",
            image.to_str().unwrap(),
        ],
    );
    assert_eq!(m["version"], serde_json::json!([1, 10]));
    // Three images in order of first use.
    let images = m["images"].as_array().unwrap();
    let want = [
        (
            832,
            "0a287302d8def7759a80cc7ffbe833f450df0aa248ae5641fbc68e1a2298afa6",
            true,
        ),
        (
            544,
            "2a8c4b7c3dead947feb1d884684fecfb350b98d84ed79fe6886dd6650fecd73a",
            false,
        ),
        (
            224,
            "f320189323c4045ff0b98e25d8d80d4d50ed196e56f4adf15a40afa84d4361f5",
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
    // matmuls at 0x88; the copies run on their node's instruction DMA.
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
            ("atlas.matmul", _) => assert_eq!(pc, 0x88),
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
    assert_eq!(m["version"], serde_json::json!([1, 10]));
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
