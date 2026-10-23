//! The memory image of a run (`--image FILE.toml`) and its readback
//! (`--image-out FILE.safetensors`).
//!
//! The image file lists entries (`[[entry]]`), each with an optional
//! `name`:
//!
//! ```toml
//! [[entry]]                      # a tensor from a .npy file
//! name = "A"
//! at = "root.gpc0.hbm+0x0"
//! npy = "a.npy"
//! dtype = "bf16"                 # optional: converts f32, f16 or bf16
//!
//! [[entry]]                      # a tensor from a .safetensors file
//! at = "root.gpc0.hbm+0x8000"
//! safetensors = "w.safetensors"
//! tensor = "B"
//!
//! [[entry]]                      # an ELF: its PT_LOAD segments at their
//! elf = "kernel.elf"             # physical addresses
//!
//! [[entry]]                      # raw bytes
//! at = "root.gpc0.sm0.smem+0x100"
//! bin = "data.bin"
//!
//! [[entry]]                      # a region: no data (an output)
//! name = "C"
//! at = "root.gpc0.hbm+0x10000"
//! dtype = "bf16"
//! shape = [128, 128]
//!
//! [[entry]]                      # an RV image into the IMEM of a node:
//! imem = "root"                  # an ELF (PT_LOAD segments at their
//! bin = "hello.bin"              # addresses, IMEM offsets) or raw bytes
//!                                # at offset 0
//!
//! [[entry]]                      # bytes in the data scratch of a node
//! at = "root.scratch+0x100"
//! bin = "args.bin"
//! ```
//!
//! `at` is a memory of the configuration by its dotted name, or
//! `<node path>.scratch` for the data scratch of a node, with an optional
//! `+offset` (decimal or `0x` hex). File paths are relative to the image
//! file. Every byte of an entry must lie in a memory block or hash range
//! of numeric fidelity, outside the sync-only space; or in the data
//! scratch or the IMEM of its node. The image loads through the functional
//! backend's host write before the run; IMEM and data scratch bytes go to
//! the control plane.
//!
//! After the run, `--image-out` reads every named entry back into one
//! safetensors file (an ELF entry gives one `U8` tensor per segment,
//! `<name>.<i>`); the metadata maps each tensor name to its global address
//! (`imem` and `offset` for an IMEM entry) and, for a region, its shape,
//! as JSON.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nu_model::{Addr, NeutrinoId, Range};
use serde::Deserialize;
use serde_json::{json, Value};
use sim_core::{Built, Core, FunctionalBackend, MemBlock};

/// An element type of an image entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    U8,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "f32" | "F32" | "<f4" | "float32" => Ok(Self::F32),
            "f16" | "F16" | "<f2" | "float16" => Ok(Self::F16),
            "bf16" | "BF16" | "bfloat16" => Ok(Self::Bf16),
            "u8" | "U8" | "|u1" | "uint8" => Ok(Self::U8),
            _ => Err(format!("unsupported dtype {s:?}")),
        }
    }

    fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::U8 => 1,
        }
    }

    /// The safetensors name.
    fn safetensors(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::U8 => "U8",
        }
    }
}

/// One entry of the image file, as written.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawEntry {
    name: Option<String>,
    at: Option<String>,
    npy: Option<PathBuf>,
    safetensors: Option<PathBuf>,
    tensor: Option<String>,
    elf: Option<PathBuf>,
    bin: Option<PathBuf>,
    imem: Option<String>,
    dtype: Option<String>,
    shape: Option<Vec<u64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawImage {
    #[serde(default)]
    entry: Vec<RawEntry>,
}

/// Where the bytes of an entry go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Global addresses, through the functional backend.
    Global,
    /// The IMEM of a node: addresses are IMEM offsets.
    Imem(NeutrinoId),
    /// The data scratch of a node: global addresses.
    Scratch(NeutrinoId),
}

/// One piece of data at an address of its entry's target.
pub struct Segment {
    pub addr: u64,
    pub bytes: Vec<u8>,
}

/// One entry, resolved: where it is and what it holds.
pub struct Entry {
    pub name: Option<String>,
    /// The element type and the shape (a tensor or a region); raw bytes
    /// and ELF segments are `U8` vectors.
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// True for a region (no data).
    pub region: bool,
    pub target: Target,
    /// The file of an ELF or raw entry: its canonical absolute path.
    pub file: Option<String>,
    /// The data (a region: one segment of zero-length data giving its
    /// address, see `ranges`).
    pub segments: Vec<Segment>,
    /// The global ranges the entry covers, one per segment.
    pub ranges: Vec<Range>,
}

/// A loaded image.
pub struct Image {
    pub entries: Vec<Entry>,
    /// The scalar cores, and paths, of the nodes of IMEM and data scratch
    /// entries.
    cores: BTreeMap<u32, (Core, String)>,
}

/// The node at dotted path `path`.
fn node(b: &Built, path: &str) -> Option<NeutrinoId> {
    let topo = b.sim.topology();
    (0..topo.len())
        .map(|i| NeutrinoId(i as u32))
        .find(|&nu| topo.path_name(nu) == path)
}

/// Resolves `at`: "<memory path>[+offset]" or "<node path>.scratch[+offset]",
/// to a global address and its target.
fn resolve(b: &Built, at: &str) -> Result<(u64, Target), String> {
    let (path, off) = match at.split_once('+') {
        Some((p, o)) => (p.trim(), parse_u64(o.trim())?),
        None => (at.trim(), 0),
    };
    if let Some(m) = b.try_memory(path) {
        return Ok((m.base + off, Target::Global));
    }
    if let Some(nu) = path.strip_suffix(".scratch").and_then(|n| node(b, n)) {
        let r = sim_core::abi::data_scratch(b, nu).map_err(|e| format!("at {at:?}: {e}"))?;
        return Ok((r.base + off, Target::Scratch(nu)));
    }
    Err(format!("at {at:?}: no memory {path:?}"))
}

fn parse_u64(s: &str) -> Result<u64, String> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(&h.replace('_', ""), 16),
        None => s.replace('_', "").parse(),
    }
    .map_err(|e| format!("offset {s:?}: {e}"))
}

/// The canonical absolute path of `p` (as given when it cannot be
/// resolved): the raw trace names an image's file by it.
fn absolute(p: &Path) -> String {
    std::fs::canonicalize(p)
        .unwrap_or_else(|_| p.to_path_buf())
        .display()
        .to_string()
}

fn read_file(base: &Path, p: &Path) -> Result<(PathBuf, Vec<u8>), String> {
    let full = base.join(p);
    let bytes = std::fs::read(&full).map_err(|e| format!("{}: {e}", full.display()))?;
    Ok((full, bytes))
}

/// Reads a .npy file (version 1 to 3, C order): (dtype, shape, data).
fn npy(path: &Path, f: &[u8]) -> Result<(Dtype, Vec<u64>, Vec<u8>), String> {
    let bad = |m: &str| format!("{}: {m}", path.display());
    if f.len() < 10 || &f[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy file"));
    }
    let (hlen, start) = match f[6] {
        1 => (u16::from_le_bytes([f[8], f[9]]) as usize, 10),
        2 | 3 if f.len() >= 12 => (
            u32::from_le_bytes(f[8..12].try_into().expect("4 bytes")) as usize,
            12,
        ),
        _ => return Err(bad("unknown .npy version")),
    };
    let header = std::str::from_utf8(f.get(start..start + hlen).ok_or_else(|| bad("short"))?)
        .map_err(|_| bad("header is not text"))?;
    let field = |k: &str| -> Result<&str, String> {
        let i = header
            .find(&format!("'{k}'"))
            .ok_or_else(|| bad(&format!("no {k} in the header")))?;
        Ok(header[i + k.len() + 2..]
            .trim_start()
            .trim_start_matches(':')
            .trim_start())
    };
    let descr = field("descr")?;
    let descr = descr
        .strip_prefix('\'')
        .and_then(|d| d.split('\'').next())
        .ok_or_else(|| bad("descr"))?;
    let dtype = Dtype::parse(descr).map_err(|e| bad(&e))?;
    if field("fortran_order")?.starts_with("True") {
        return Err(bad("Fortran order"));
    }
    let shape = field("shape")?;
    let shape = shape
        .strip_prefix('(')
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| bad("shape"))?;
    let shape: Vec<u64> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|_| bad("shape")))
        .collect::<Result<_, _>>()?;
    let n: u64 = shape.iter().product::<u64>() * dtype.bytes();
    let data = f
        .get(start + hlen..start + hlen + n as usize)
        .ok_or_else(|| bad("short data"))?;
    Ok((dtype, shape, data.to_vec()))
}

/// Reads tensor `name` of a safetensors file: (dtype, shape, data).
fn safetensors(path: &Path, f: &[u8], name: &str) -> Result<(Dtype, Vec<u64>, Vec<u8>), String> {
    let bad = |m: &str| format!("{}: {m}", path.display());
    let n = u64::from_le_bytes(
        f.get(..8)
            .ok_or_else(|| bad("short"))?
            .try_into()
            .expect("8"),
    );
    let header: Value = serde_json::from_slice(
        f.get(8..8 + n as usize)
            .ok_or_else(|| bad("short header"))?,
    )
    .map_err(|e| bad(&e.to_string()))?;
    let t = header
        .get(name)
        .ok_or_else(|| bad(&format!("no tensor {name:?}")))?;
    let dtype = Dtype::parse(t["dtype"].as_str().unwrap_or("")).map_err(|e| bad(&e))?;
    let shape: Vec<u64> = t["shape"]
        .as_array()
        .ok_or_else(|| bad("shape"))?
        .iter()
        .map(|x| x.as_u64().ok_or_else(|| bad("shape")))
        .collect::<Result<_, _>>()?;
    let off = t["data_offsets"].as_array().ok_or_else(|| bad("offsets"))?;
    let (a, z) = (
        off[0].as_u64().unwrap_or(0) as usize,
        off[1].as_u64().unwrap_or(0) as usize,
    );
    let base = 8 + n as usize;
    let data = f.get(base + a..base + z).ok_or_else(|| bad("short data"))?;
    Ok((dtype, shape, data.to_vec()))
}

/// The PT_LOAD segments of an ELF file (32 or 64 bit, little-endian), at
/// their physical addresses, zero-filled to their memory size.
fn elf(path: &Path, f: &[u8]) -> Result<Vec<Segment>, String> {
    let bad = |m: &str| format!("{}: {m}", path.display());
    if f.len() < 0x34 || &f[..4] != b"\x7fELF" {
        return Err(bad("not an ELF file"));
    }
    if f[5] != 1 {
        return Err(bad("not little-endian"));
    }
    let u16_at = |o: usize| u16::from_le_bytes([f[o], f[o + 1]]) as usize;
    let u32_at = |o: usize| -> Result<u64, String> {
        Ok(u32::from_le_bytes(
            f.get(o..o + 4)
                .ok_or_else(|| bad("short"))?
                .try_into()
                .expect("4"),
        ) as u64)
    };
    let u64_at = |o: usize| -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            f.get(o..o + 8)
                .ok_or_else(|| bad("short"))?
                .try_into()
                .expect("8"),
        ))
    };
    let wide = match f[4] {
        1 => false,
        2 => true,
        _ => return Err(bad("unknown ELF class")),
    };
    let (phoff, phentsize, phnum) = if wide {
        (u64_at(0x20)? as usize, u16_at(0x36), u16_at(0x38))
    } else {
        (u32_at(0x1C)? as usize, u16_at(0x2A), u16_at(0x2C))
    };
    let mut out = Vec::new();
    for i in 0..phnum {
        let p = phoff + i * phentsize;
        if u32_at(p)? != 1 {
            continue; // not PT_LOAD
        }
        let (offset, paddr, filesz, memsz) = if wide {
            (
                u64_at(p + 8)?,
                u64_at(p + 24)?,
                u64_at(p + 32)?,
                u64_at(p + 40)?,
            )
        } else {
            (
                u32_at(p + 4)?,
                u32_at(p + 12)?,
                u32_at(p + 16)?,
                u32_at(p + 20)?,
            )
        };
        if memsz == 0 {
            continue;
        }
        let mut bytes = f
            .get(offset as usize..(offset + filesz) as usize)
            .ok_or_else(|| bad("segment past the end of the file"))?
            .to_vec();
        bytes.resize(memsz as usize, 0);
        out.push(Segment { addr: paddr, bytes });
    }
    Ok(out)
}

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1f) as u32,
        (h & 0x3ff) as u32,
    );
    let bits = match (e, m) {
        (0, 0) => s << 31,
        (0, _) => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => (s << 31) | (0xff << 23) | (m << 13),
        _ => (s << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Converts little-endian elements from `from` to `to` (f32, f16 or bf16
/// to f32 or bf16; bf16 rounds to nearest even).
fn convert(data: &[u8], from: Dtype, to: Dtype) -> Result<Vec<u8>, String> {
    if from == to {
        return Ok(data.to_vec());
    }
    let vals: Vec<f32> = match from {
        Dtype::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        Dtype::Bf16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| func::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        Dtype::U8 => return Err("dtype converts only f32, f16 and bf16".into()),
    };
    Ok(match to {
        Dtype::F32 => vals.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Dtype::Bf16 => vals
            .iter()
            .flat_map(|&x| func::f32_to_bf16(x).to_le_bytes())
            .collect(),
        _ => return Err(format!("dtype converts to f32 or bf16, not {to:?}")),
    })
}

/// Splits `r` into pieces, each inside one functional block. Every byte
/// must be in a numeric block and outside the sync-only space.
fn pieces(blocks: &[MemBlock], sync_only: &[Range], r: Range) -> Result<Vec<Range>, String> {
    if let Some(s) = sync_only.iter().find(|s| s.overlaps(&r)) {
        return Err(format!("{r} overlaps the sync-only range {s}"));
    }
    let mut out = Vec::new();
    let mut a = r.base;
    while a < r.bound {
        let b = blocks
            .iter()
            .filter(|b| b.range.base <= a && a < b.range.bound)
            .min_by_key(|b| b.range.len())
            .ok_or_else(|| format!("{r}: byte {a:#x} is in no memory block"))?;
        if !b.numeric {
            return Err(format!("{r}: byte {a:#x} is in a null-fidelity block"));
        }
        let z = b.range.bound.min(r.bound);
        out.push(Range { base: a, bound: z });
        a = z;
    }
    Ok(out)
}

impl Image {
    /// Reads the image file and resolves its entries on `b`.
    pub fn load(path: &Path, b: &Built) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let raw: RawImage =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let blocks = b.sim.functional_blocks();
        let sync_only = b.sim.sync_only_ranges();
        let mut entries = Vec::new();
        let mut cores = BTreeMap::new();
        for (i, e) in raw.entry.into_iter().enumerate() {
            let what = match &e.name {
                Some(n) => format!("entry {n:?}"),
                None => format!("entry {}", i + 1),
            };
            let entry = Self::entry(e, dir, b).map_err(|m| format!("{what}: {m}"))?;
            let mut ranges = Vec::new();
            for s in &entry.segments {
                let len = s.bytes.len() as u64;
                let len = if entry.region {
                    entry.shape.iter().product::<u64>() * entry.dtype.bytes()
                } else {
                    len
                };
                let r = Range::from_len(s.addr, len);
                match entry.target {
                    Target::Global => {
                        pieces(&blocks, sync_only, r).map_err(|m| format!("{what}: {m}"))?;
                    }
                    Target::Scratch(nu) => {
                        let s = sim_core::abi::data_scratch(b, nu)?;
                        if !s.covers(&r) {
                            return Err(format!(
                                "{what}: {r} runs past the data scratch {s} of {}",
                                b.sim.topology().path_name(nu)
                            ));
                        }
                    }
                    Target::Imem(nu) => {
                        let n = sim_core::abi::imem_bytes(b, nu);
                        if r.bound > n {
                            return Err(format!(
                                "{what}: IMEM offsets {r} run past the IMEM ({n} bytes) of {}",
                                b.sim.topology().path_name(nu)
                            ));
                        }
                    }
                }
                ranges.push(r);
            }
            if let Target::Imem(nu) | Target::Scratch(nu) = entry.target {
                cores.insert(nu.0, (b.sim.core(nu), b.sim.topology().path_name(nu)));
            }
            entries.push(Entry { ranges, ..entry });
        }
        Ok(Self { entries, cores })
    }

    /// The address of the first byte of entry `name`: global, or an IMEM
    /// offset for an IMEM entry.
    pub fn address_of(&self, name: &str) -> Option<u64> {
        let e = self
            .entries
            .iter()
            .find(|e| e.name.as_deref() == Some(name))?;
        e.ranges.first().map(|r| r.base)
    }

    fn entry(e: RawEntry, dir: &Path, b: &Built) -> Result<Entry, String> {
        let imem = match &e.imem {
            Some(p) => {
                if e.at.is_some() || !(e.elf.is_some() || e.bin.is_some()) {
                    return Err("imem takes an elf or a bin, and no at".into());
                }
                Some(node(b, p).ok_or_else(|| format!("imem: no node {p:?}"))?)
            }
            None => None,
        };
        let resolved = match (&e.at, imem) {
            (_, Some(nu)) => Some(Ok((0, Target::Imem(nu)))),
            (Some(at), None) => Some(resolve(b, at)),
            (None, None) => None,
        };
        let at = || -> Result<u64, String> { Ok(resolved.clone().ok_or("needs at")??.0) };
        let target = match &resolved {
            Some(Ok((_, t))) => *t,
            _ => Target::Global,
        };
        let want = e.dtype.as_deref().map(Dtype::parse).transpose()?;
        let sources = [
            e.npy.is_some(),
            e.safetensors.is_some(),
            e.elf.is_some(),
            e.bin.is_some(),
        ];
        if sources.iter().filter(|&&x| x).count() > 1 {
            return Err("give one of npy, safetensors, elf and bin".into());
        }
        if e.tensor.is_some() != e.safetensors.is_some() {
            return Err("safetensors and tensor go together".into());
        }
        let tensor = |dtype: Dtype, shape: Vec<u64>, data: Vec<u8>| -> Result<Entry, String> {
            if e.shape.is_some() {
                return Err("shape is for a region".into());
            }
            let to = want.unwrap_or(dtype);
            Ok(Entry {
                name: e.name.clone(),
                dtype: to,
                shape,
                region: false,
                target,
                file: None,
                segments: vec![Segment {
                    addr: at()?,
                    bytes: convert(&data, dtype, to)?,
                }],
                ranges: Vec::new(),
            })
        };
        if let Some(p) = &e.npy {
            let (full, f) = read_file(dir, p)?;
            let (dt, shape, data) = npy(&full, &f)?;
            return tensor(dt, shape, data);
        }
        if let Some(p) = &e.safetensors {
            let (full, f) = read_file(dir, p)?;
            let (dt, shape, data) = safetensors(&full, &f, e.tensor.as_deref().unwrap_or(""))?;
            return tensor(dt, shape, data);
        }
        if e.shape.is_some() && (e.elf.is_some() || e.bin.is_some()) {
            return Err("shape is for a region".into());
        }
        if want.is_some() && (e.elf.is_some() || e.bin.is_some()) {
            return Err("dtype is for a tensor or a region".into());
        }
        if let Some(p) = &e.elf {
            if e.at.is_some() {
                return Err("an ELF loads at its physical addresses: no at".into());
            }
            let target = imem.map_or(Target::Global, Target::Imem);
            let (full, f) = read_file(dir, p)?;
            return Ok(Entry {
                name: e.name.clone(),
                dtype: Dtype::U8,
                shape: Vec::new(),
                region: false,
                target,
                file: Some(absolute(&full)),
                segments: elf(&full, &f)?,
                ranges: Vec::new(),
            });
        }
        if let Some(p) = &e.bin {
            let (full, f) = read_file(dir, p)?;
            return Ok(Entry {
                name: e.name.clone(),
                dtype: Dtype::U8,
                shape: vec![f.len() as u64],
                region: false,
                target,
                file: Some(absolute(&full)),
                segments: vec![Segment {
                    addr: at()?,
                    bytes: f,
                }],
                ranges: Vec::new(),
            });
        }
        // A region.
        if e.name.is_none() {
            return Err("a region needs a name".into());
        }
        Ok(Entry {
            name: e.name.clone(),
            dtype: want.ok_or("a region needs dtype")?,
            shape: e.shape.clone().ok_or("a region needs shape")?,
            region: true,
            target,
            file: None,
            segments: vec![Segment {
                addr: at()?,
                bytes: Vec::new(),
            }],
            ranges: Vec::new(),
        })
    }

    /// Writes the image's data into the simulation (host writes before the
    /// run), one write per functional block.
    pub fn install(&self, b: &mut Built) -> Result<(), String> {
        let blocks = b.sim.functional_blocks();
        let sync_only = b.sim.sync_only_ranges().to_vec();
        for e in &self.entries {
            // An image at offset 0 of an IMEM: the IMEM's image, for the
            // raw trace.
            if let Target::Imem(nu) = e.target {
                if e.segments.iter().any(|s| s.addr == 0) {
                    let end = e
                        .segments
                        .iter()
                        .map(|s| s.addr + s.bytes.len() as u64)
                        .max();
                    let (core, path) = &self.cores[&nu.0];
                    core.mem.borrow_mut().image = Some(sim_core::ImemImage {
                        bytes: end.unwrap_or(0),
                        file: e.file.clone(),
                        node: Some(path.clone()),
                    });
                }
            }
            for s in e.segments.iter().filter(|s| !s.bytes.is_empty()) {
                if let Some((nu, at)) = self.core_at(e.target, s.addr) {
                    let mut mem = self.cores[&nu].0.mem.borrow_mut();
                    let m = match e.target {
                        Target::Imem(_) => mem.imem(),
                        _ => mem.scratch(),
                    };
                    m[at..at + s.bytes.len()].copy_from_slice(&s.bytes);
                    continue;
                }
                let r = Range::from_len(s.addr, s.bytes.len() as u64);
                for p in pieces(&blocks, &sync_only, r)? {
                    let (a, z) = ((p.base - s.addr) as usize, (p.bound - s.addr) as usize);
                    b.sim.init_memory(Addr(p.base), s.bytes[a..z].to_vec());
                }
            }
        }
        Ok(())
    }

    /// The node and the offset in its IMEM or data scratch of address
    /// `addr` of target `t`; `None` for a global target.
    fn core_at(&self, t: Target, addr: u64) -> Option<(u32, usize)> {
        match t {
            Target::Global => None,
            Target::Imem(nu) => Some((nu.0, addr as usize)),
            Target::Scratch(nu) => {
                let base = self.cores[&nu.0].0.scratch.expect("a data scratch").base;
                Some((nu.0, (addr - base) as usize))
            }
        }
    }

    /// Reads every named entry back from `backend` and writes them to the
    /// safetensors file `out`.
    pub fn write_out(
        &self,
        backend: &mut dyn FunctionalBackend,
        blocks: &[MemBlock],
        sync_only: &[Range],
        out: &Path,
    ) -> Result<(), String> {
        let mut tensors: Vec<(String, Dtype, Vec<u64>, Vec<u8>)> = Vec::new();
        let mut meta = BTreeMap::new();
        for e in &self.entries {
            let Some(name) = &e.name else { continue };
            let many = e.ranges.len() != 1;
            for (i, r) in e.ranges.iter().enumerate() {
                let mut data = Vec::with_capacity(r.len() as usize);
                if let Some((nu, at)) = self.core_at(e.target, r.base) {
                    let mut mem = self.cores[&nu].0.mem.borrow_mut();
                    let m = match e.target {
                        Target::Imem(_) => mem.imem(),
                        _ => mem.scratch(),
                    };
                    data.extend_from_slice(&m[at..at + r.len() as usize]);
                }
                let global = if e.target == Target::Global {
                    pieces(blocks, sync_only, *r)?
                } else {
                    Vec::new()
                };
                for p in global {
                    data.extend(
                        backend
                            .read(Addr(p.base), p.len())
                            .map_err(|x| format!("{name}: {}", x.0))?,
                    );
                }
                let (tname, shape) = if many {
                    (format!("{name}.{i}"), vec![r.len()])
                } else if e.dtype == Dtype::U8 && e.shape.is_empty() {
                    (name.clone(), vec![r.len()])
                } else {
                    (name.clone(), e.shape.clone())
                };
                let mut m = match e.target {
                    Target::Imem(nu) => json!({
                        "imem": self.cores[&nu.0].1,
                        "offset": format!("{:#x}", r.base),
                    }),
                    _ => json!({ "addr": format!("{:#x}", r.base) }),
                };
                if e.region {
                    m["shape"] = json!(e.shape);
                }
                meta.insert(tname.clone(), m.to_string());
                tensors.push((tname, e.dtype, shape, data));
            }
        }
        write_safetensors(out, &tensors, &meta).map_err(|e| format!("{}: {e}", out.display()))
    }
}

/// Writes a safetensors file.
fn write_safetensors(
    out: &Path,
    tensors: &[(String, Dtype, Vec<u64>, Vec<u8>)],
    meta: &BTreeMap<String, String>,
) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!(meta));
    let mut off = 0usize;
    for (name, dt, shape, data) in tensors {
        header.insert(
            name.clone(),
            json!({ "dtype": dt.safetensors(), "shape": shape, "data_offsets": [off, off + data.len()] }),
        );
        off += data.len();
    }
    let mut h = serde_json::to_vec(&Value::Object(header)).expect("JSON");
    // The data starts 8-byte aligned.
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut f = Vec::with_capacity(8 + h.len() + off);
    f.extend((h.len() as u64).to_le_bytes());
    f.extend(h);
    for (_, _, _, data) in tensors {
        f.extend(data);
    }
    std::fs::write(out, f)
}
