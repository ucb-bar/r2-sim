//! The call paths of raw trace 1.11 on the encoder fixture (configs/2p.yaml):
//! a task made in the program function has `call_path` null; a task made
//! in a helper has a path of call instructions of its program's image
//! (JAL or JALR with a link register as rd, or C.JALR); the manifest lists
//! each path once, and every path is used.

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

const NULL32: u32 = u32::MAX;
const ENCODER: &str = "../nu-rv/tests/fixtures/rv/encoder.bin";
/// The images in encoder.bin by size: (bytes, offset in encoder.bin).
const IMAGES: [(u64, usize); 3] = [(14_608, 0), (14_288, 248), (10_496, 3_952)];

fn u32_at(r: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(r[o..o + 4].try_into().unwrap())
}

fn u64_at(r: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(r[o..o + 8].try_into().unwrap())
}

/// True when the instruction at `o` of `image` is a call: JAL or JALR
/// with rd x1 or x5, or C.JALR.
fn is_call(image: &[u8], o: usize) -> bool {
    let half = u16::from_le_bytes([image[o], image[o + 1]]);
    if half & 3 != 3 {
        // C.JALR: funct4 1001, rs1 != 0, rs2 = 0, op 10.
        return half >> 12 == 0b1001
            && (half >> 7) & 31 != 0
            && (half >> 2) & 31 == 0
            && half & 3 == 2;
    }
    let w = u32_at(image, o);
    let rd = (w >> 7) & 31;
    matches!(w & 0x7f, 0x6f | 0x67) && (rd == 1 || rd == 5)
}

#[test]
fn the_encoder_tasks_carry_their_call_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let d = std::env::temp_dir().join(format!("r2sim-call-paths-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_r2sim"))
        .args(["run", "--config"])
        .arg(root.join("../../configs/2p.yaml"))
        .args(["--program", "rv", "--image"])
        .arg(root.join("../nu-rv/tests/fixtures/rv/encoder.toml"))
        .arg("--trace-raw")
        .arg(d.join("raw"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let m: Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("raw/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(m["version"], serde_json::json!([1, 11]));
    let paths: Vec<Vec<u64>> = serde_json::from_value(m["call_paths"].clone()).unwrap();
    assert!(!paths.is_empty());
    assert!(paths.iter().all(|p| !p.is_empty()));
    assert_eq!(
        paths.iter().collect::<HashSet<_>>().len(),
        paths.len(),
        "each path once"
    );
    let images: Vec<u64> = m["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["bytes"].as_u64().unwrap())
        .collect();
    let encoder = std::fs::read(root.join(ENCODER)).unwrap();
    let image_bytes = |i: u32| -> &[u8] {
        let size = images[i as usize];
        let (_, at) = IMAGES
            .iter()
            .find(|(b, _)| *b == size)
            .expect("an image of encoder.bin");
        &encoder[*at..*at + size as usize]
    };
    // Programs by Neutrino: (start, image).
    let programs = std::fs::read(d.join("raw/programs.bin")).unwrap();
    let mut by_nu: Vec<Vec<(u64, u32)>> = Vec::new();
    for p in programs.chunks(32) {
        let nu = u16::from_le_bytes([p[0], p[1]]) as usize;
        by_nu.resize(by_nu.len().max(nu + 1), Vec::new());
        by_nu[nu].push((u64_at(p, 16), u32_at(p, 12)));
    }
    let tasks = std::fs::read(d.join("raw/tasks.bin")).unwrap();
    let (mut null, mut used) = (0, HashSet::new());
    for t in tasks.chunks(120) {
        let (pc, cp) = (u32_at(t, 112), u32_at(t, 116));
        assert_ne!(pc, NULL32, "every task of an RV program has a pc");
        if cp == NULL32 {
            null += 1;
            continue;
        }
        used.insert(cp);
        let (src, call) = (u16::from_le_bytes([t[6], t[7]]) as usize, u64_at(t, 40));
        let image = by_nu[src]
            .iter()
            .rev()
            .find(|(start, _)| *start <= call)
            .unwrap()
            .1;
        let bytes = image_bytes(image);
        for &o in &paths[cp as usize] {
            assert!(is_call(bytes, o as usize), "path element {o} is not a call");
        }
    }
    // The device program and the step programs make tasks in frame 1 too.
    assert!(null > 0);
    assert_eq!(used.len(), paths.len(), "every path is used");
    std::fs::remove_dir_all(&d).unwrap();
}
