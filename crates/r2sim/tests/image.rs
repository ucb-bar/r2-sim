//! The memory image (`--image`, `--image-out`) on configs/2p-smem.yaml with the
//! Numeric backend: every kind of entry loads and reads back, and an entry
//! outside a numeric memory block is an error.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

const R2SIM: &str = env!("CARGO_BIN_EXE_r2sim");

fn dir(test: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("r2sim-image-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Runs the RV hello fixture with the image `toml` (written into `d`);
/// the output. The fixture goes into the root's IMEM unless `toml`
/// loads the root's IMEM itself.
fn run(d: &Path, toml: &str) -> std::process::Output {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let hello = root.join("../nu-rv/tests/fixtures/rv/hello.bin");
    let mut text = toml.to_string();
    if !toml.contains("imem = \"root\"") {
        text += &format!("\n[[entry]]\nimem = \"root\"\nbin = {:?}\n", hello);
    }
    std::fs::write(d.join("image.toml"), text).unwrap();
    Command::new(R2SIM)
        .args(["run", "--config"])
        .arg(root.join("../../configs/2p-smem.yaml"))
        .args(["--program", "rv", "--functional", "numeric", "--image"])
        .arg(d.join("image.toml"))
        .arg("--image-out")
        .arg(d.join("out.safetensors"))
        .output()
        .unwrap()
}

/// A safetensors file: (header, data).
fn safetensors(p: &Path) -> (Value, Vec<u8>) {
    let f = std::fs::read(p).unwrap();
    let n = u64::from_le_bytes(f[..8].try_into().unwrap()) as usize;
    let header = serde_json::from_slice(&f[8..8 + n]).unwrap();
    (header, f[8 + n..].to_vec())
}

/// The bytes of tensor `name`.
fn tensor<'a>(h: &Value, data: &'a [u8], name: &str) -> &'a [u8] {
    let o = &h[name]["data_offsets"];
    &data[o[0].as_u64().unwrap() as usize..o[1].as_u64().unwrap() as usize]
}

/// A .npy file of little-endian f16 values (version 1).
fn npy_f16(vals: &[u16], shape: &str) -> Vec<u8> {
    let mut h = format!("{{'descr': '<f2', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut f = b"\x93NUMPY\x01\x00".to_vec();
    f.extend((h.len() as u16).to_le_bytes());
    f.extend(h.as_bytes());
    for v in vals {
        f.extend(v.to_le_bytes());
    }
    f
}

/// An ELF64 file with one PT_LOAD segment of `bytes` at physical address
/// `paddr`, zero-filled to `memsz`.
fn elf64(paddr: u64, bytes: &[u8], memsz: u64) -> Vec<u8> {
    let mut f = vec![0u8; 64 + 56];
    f[..4].copy_from_slice(b"\x7fELF");
    f[4] = 2; // 64-bit
    f[5] = 1; // little-endian
    f[0x20..0x28].copy_from_slice(&64u64.to_le_bytes()); // phoff
    f[0x36..0x38].copy_from_slice(&56u16.to_le_bytes()); // phentsize
    f[0x38..0x3a].copy_from_slice(&1u16.to_le_bytes()); // phnum
    let p = 64;
    f[p..p + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    f[p + 8..p + 16].copy_from_slice(&120u64.to_le_bytes()); // offset
    f[p + 16..p + 24].copy_from_slice(&0xdead_0000u64.to_le_bytes()); // vaddr
    f[p + 24..p + 32].copy_from_slice(&paddr.to_le_bytes());
    f[p + 32..p + 40].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    f[p + 40..p + 48].copy_from_slice(&memsz.to_le_bytes());
    f.extend(bytes);
    f
}

#[test]
fn every_kind_of_entry_loads_and_reads_back() {
    let d = dir("kinds");
    // f16 1.0, -2.0, 0.5, 65504 (the largest f16), converted to f32.
    std::fs::write(
        d.join("h.npy"),
        npy_f16(&[0x3c00, 0xc000, 0x3800, 0x7bff], "(2, 2)"),
    )
    .unwrap();
    std::fs::write(d.join("raw.bin"), b"raw bytes").unwrap();
    // gpc1's HBM is at 2 << 36.
    let hbm1 = 2u64 << 36;
    std::fs::write(d.join("k.elf"), elf64(hbm1 + 0x100, b"segment", 16)).unwrap();
    let out = run(
        &d,
        r#"
[[entry]]
name = "h"
at = "root.gpc1.hbm+0x40000"
npy = "h.npy"
dtype = "f32"

[[entry]]                     # across the SMEM's two banks
name = "raw"
at = "root.gpc1.sm0.smem+0x7fffc"
bin = "raw.bin"

[[entry]]
name = "k"
elf = "k.elf"

[[entry]]
name = "out"
at = "root.gpc1.hbm+0x50000"
dtype = "bf16"
shape = [3]
"#,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (h, data) = safetensors(&d.join("out.safetensors"));
    let f32s: Vec<f32> = tensor(&h, &data, "h")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(f32s, [1.0, -2.0, 0.5, 65504.0]);
    assert_eq!(h["h"]["dtype"], "F32");
    assert_eq!(h["h"]["shape"], serde_json::json!([2, 2]));
    assert_eq!(tensor(&h, &data, "raw"), b"raw bytes");
    let mut seg = b"segment".to_vec();
    seg.resize(16, 0);
    assert_eq!(tensor(&h, &data, "k"), &seg[..]);
    assert_eq!(tensor(&h, &data, "out"), &[0u8; 6]);
    let meta = &h["__metadata__"];
    let k: Value = serde_json::from_str(meta["k"].as_str().unwrap()).unwrap();
    assert_eq!(k["addr"], format!("{:#x}", hbm1 + 0x100));
    let o: Value = serde_json::from_str(meta["out"].as_str().unwrap()).unwrap();
    assert_eq!(o["shape"], serde_json::json!([3]));
}

#[test]
fn an_entry_outside_a_memory_block_is_an_error() {
    let d = dir("outside");
    std::fs::write(d.join("raw.bin"), vec![1u8; 64]).unwrap();
    // The last 32 bytes of the SMEM (1 MiB) and 32 bytes past it.
    let smem = 1u64 << 20;
    let out = run(
        &d,
        &format!(
            "[[entry]]\nat = \"root.gpc1.sm0.smem+{:#x}\"\nbin = \"raw.bin\"\n",
            smem - 32
        ),
    );
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("is in no memory block"), "{err}");
}

#[test]
fn a_bad_at_is_an_error() {
    let d = dir("at");
    let out = run(
        &d,
        "[[entry]]\nname = \"x\"\nat = \"root.gpc9.hbm\"\ndtype = \"f32\"\nshape = [1]\n",
    );
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no memory \"root.gpc9.hbm\""), "{err}");
}

#[test]
fn imem_and_data_scratch_entries_load_into_the_control_plane() {
    // A raw image into gpc0's IMEM and bytes into gpc0's data scratch;
    // both read back. The IMEM entry gives its node and offset.
    let d = dir("core");
    std::fs::write(d.join("code.bin"), b"\x13\x00\x00\x00code").unwrap();
    std::fs::write(d.join("args.bin"), b"arguments").unwrap();
    let out = run(
        &d,
        r#"
[[entry]]
name = "code"
imem = "root.gpc0"
bin = "code.bin"

[[entry]]
name = "args"
at = "root.gpc0.scratch+0x100"
bin = "args.bin"
"#,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (h, data) = safetensors(&d.join("out.safetensors"));
    assert_eq!(tensor(&h, &data, "code"), b"\x13\x00\x00\x00code");
    assert_eq!(tensor(&h, &data, "args"), b"arguments");
    let meta = &h["__metadata__"];
    let code: Value = serde_json::from_str(meta["code"].as_str().unwrap()).unwrap();
    assert_eq!(
        (code["imem"].as_str(), code["offset"].as_str()),
        (Some("root.gpc0"), Some("0x0"))
    );
    let args: Value = serde_json::from_str(meta["args"].as_str().unwrap()).unwrap();
    // gpc0's data scratch: its prefix 1 << 36 plus the end of its
    // sync-only range, 0x1f1000000.
    assert_eq!(args["addr"], "0x11f1000100");
}

#[test]
fn imem_and_data_scratch_entries_stay_inside() {
    let d = dir("core-bad");
    std::fs::write(d.join("big.bin"), vec![0u8; 40 << 10]).unwrap();
    let out = run(&d, "[[entry]]\nimem = \"root\"\nbin = \"big.bin\"\n");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() && err.contains("run past the IMEM"),
        "{err}"
    );
    let out = run(
        &d,
        "[[entry]]\nat = \"root.scratch+0x100\"\nbin = \"big.bin\"\n",
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() && err.contains("runs past the data scratch"),
        "{err}"
    );
}

#[test]
fn an_elf_given_as_bin_is_an_error() {
    // An ELF given as raw bytes would put its header where the code goes.
    let d = dir("elf-as-bin");
    std::fs::write(d.join("k.elf"), elf64(0, b"code", 4)).unwrap();
    let out = run(&d, "[[entry]]\nimem = \"root\"\nbin = \"k.elf\"\n");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert!(
        err.contains("bin is an ELF file; load it with `elf = \"k.elf\"`"),
        "{err}"
    );
}
