//! The Muon kernel registry: the kernels that a Muon task (opcode
//! `vector`) can run, from `kernels.yaml` of this crate.
//!
//! The registry is a YAML list of kernels: `id`, `name`, `summary`,
//! `semantics`, optional `flags` (name: bit of the field `flags`), `fields`
//! (name, type, meaning) and `time` (the traffic and the fixed latency of a
//! task from its block: [`time`]). The fields of an argument block are in
//! order, in C layout with no implicit padding: each field starts where the
//! one before it ends, at a multiple of its size. A block is a multiple of
//! 8 bytes.
//! Kernel 0 (`opaque`) is listed: it writes nothing, and its block gives
//! its time.
//!
//! r2-sim checks each Muon task against the registry at decode and passes
//! it to the plug-in backends (without `time`). The toolchain generates the
//! Rust structs of the argument blocks from the same file with [`parse`].

use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

pub mod time;

pub use time::{Time, Traffic};

/// The registry file of this crate.
pub const REGISTRY: &str = include_str!("../kernels.yaml");

/// The type of a field of an argument block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    U64,
    U32,
    F32,
}

impl FieldType {
    fn parse(ty: &str) -> Option<Self> {
        match ty {
            "u64" => Some(Self::U64),
            "u32" => Some(Self::U32),
            "f32" => Some(Self::F32),
            _ => None,
        }
    }

    /// The name in the registry (and in Rust).
    pub fn name(self) -> &'static str {
        match self {
            Self::U64 => "u64",
            Self::U32 => "u32",
            Self::F32 => "f32",
        }
    }

    /// The bytes of a field of this type.
    pub fn size(self) -> u64 {
        match self {
            Self::U64 => 8,
            Self::U32 | Self::F32 => 4,
        }
    }
}

/// A field of an argument block.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Field {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: FieldType,
    /// The byte offset in the block.
    pub offset: u64,
    #[serde(skip)]
    pub meaning: String,
}

/// One kernel of the registry. As JSON (the configuration of the plug-ins):
/// id, name, bytes, flags and fields (name, type, offset); not `time`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Kernel {
    pub id: u32,
    pub name: String,
    #[serde(skip)]
    pub summary: String,
    #[serde(skip)]
    pub semantics: String,
    /// The bytes of the argument block.
    pub bytes: u64,
    /// Named bits of the field `flags`.
    pub flags: BTreeMap<String, u32>,
    pub fields: Vec<Field>,
    /// The traffic of a task ([`Kernel::traffic`]).
    #[serde(skip)]
    pub time: Time,
}

impl Kernel {
    /// The field `name`.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// A kernel as the YAML holds it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    id: u32,
    name: String,
    summary: String,
    semantics: String,
    #[serde(default)]
    flags: BTreeMap<String, u32>,
    /// Name, type, meaning.
    fields: Vec<(String, String, String)>,
    time: time::TimeEntry,
}

/// The kernels of the registry `yaml`, checked: ids and names unique, the
/// types `u64`, `u32` and `f32`, no implicit padding, blocks of a multiple
/// of 8 bytes; each flag one bit, a flag name with one bit across all
/// kernels and a bit with one name; the checks of `time` ([`time`]).
pub fn parse(yaml: &str) -> Result<Vec<Kernel>, String> {
    let entries: Vec<Entry> =
        serde_yaml::from_str(yaml).map_err(|e| format!("kernel registry: {e}"))?;
    let mut ids = HashSet::new();
    let mut names = HashSet::new();
    // The bit of each flag name and the name of each bit, over all kernels.
    let mut bits: BTreeMap<String, u32> = BTreeMap::new();
    let mut flag_of: BTreeMap<u32, String> = BTreeMap::new();
    let mut kernels = Vec::with_capacity(entries.len());
    for e in entries {
        let at = |msg: String| format!("kernel registry: {} (id {}): {msg}", e.name, e.id);
        if !ids.insert(e.id) {
            return Err(at("the id is not unique".into()));
        }
        if !names.insert(e.name.clone()) {
            return Err(at("the name is not unique".into()));
        }
        for (flag, &bit) in &e.flags {
            if !bit.is_power_of_two() {
                return Err(at(format!("flag {flag} = {bit} is not one bit")));
            }
            if *bits.entry(flag.clone()).or_insert(bit) != bit {
                return Err(at(format!(
                    "flag {flag} = {bit}, another kernel has {flag} = {}",
                    bits[flag]
                )));
            }
            let other = flag_of.entry(bit).or_insert_with(|| flag.clone());
            if other != flag {
                return Err(at(format!(
                    "flag {flag} = {bit}, another kernel names bit {bit} {other}"
                )));
            }
        }
        let mut offset = 0;
        let mut fields = Vec::with_capacity(e.fields.len());
        for (name, ty, meaning) in &e.fields {
            let t = FieldType::parse(ty)
                .ok_or_else(|| at(format!("field {name}: unknown type {ty}")))?;
            if offset % t.size() != 0 {
                return Err(at(format!(
                    "field {name} at offset {offset} is not aligned to {} bytes (implicit padding)",
                    t.size()
                )));
            }
            fields.push(Field {
                name: name.clone(),
                ty: t,
                offset,
                meaning: meaning.clone(),
            });
            offset += t.size();
        }
        if offset % 8 != 0 {
            return Err(at(format!("{offset} bytes, not a multiple of 8")));
        }
        let mut kernel = Kernel {
            id: e.id,
            name: e.name.clone(),
            summary: e.summary.clone(),
            semantics: e.semantics.clone(),
            bytes: offset,
            flags: e.flags.clone(),
            fields,
            time: Time::default(),
        };
        kernel.time = time::compile(&e.time, &kernel).map_err(|m| at(format!("time: {m}")))?;
        kernels.push(kernel);
    }
    Ok(kernels)
}

/// The kernels of [`REGISTRY`], parsed once.
pub fn registry() -> &'static [Kernel] {
    static KERNELS: OnceLock<Vec<Kernel>> = OnceLock::new();
    KERNELS.get_or_init(|| parse(REGISTRY).expect("the registry of sim-core"))
}

/// The kernel of the registry with id `id`.
pub fn by_id(id: u32) -> Option<&'static Kernel> {
    registry().iter().find(|k| k.id == id)
}

/// The name of kernel `id` (`opaque` for 0), or `None` for an id that the
/// registry does not hold.
pub fn name(id: u32) -> Option<&'static str> {
    by_id(id).map(|k| k.name.as_str())
}

/// The flags of all kernels of the registry, by bit: (name, bit).
pub fn flags() -> Vec<(&'static str, u32)> {
    let mut f: Vec<(&str, u32)> = registry()
        .iter()
        .flat_map(|k| k.flags.iter().map(|(n, &b)| (n.as_str(), b)))
        .collect();
    f.sort_by_key(|&(_, b)| b);
    f.dedup();
    f
}

/// The trace label of a Muon task that runs kernel `id` with the argument
/// block `block`: the id in bits 0 to 15, and the block's field `flags` in
/// bits 32 to 63 when the kernel has flags.
pub fn label(id: u32, block: &[u8]) -> u64 {
    let flags = by_id(id)
        .filter(|k| !k.flags.is_empty())
        .and_then(|k| k.field("flags"))
        .and_then(|f| block.get(f.offset as usize..f.offset as usize + 4))
        .map_or(0, |b| u32::from_le_bytes(b.try_into().expect("4 bytes")));
    id as u64 | (flags as u64) << 32
}

/// The argument block of the opaque kernel (id 0) with latency 0: the task
/// reads `bytes_in` at `src` (0: its destination), does `ops` lane
/// operations and writes `bytes_out` to its destination. A value that does
/// not fit the block's u32 fields is a program error (panic).
pub fn opaque_block(src: u64, ops: u64, bytes_in: u64, bytes_out: u64) -> Vec<u8> {
    let u32_of = |name: &str, v: u64| {
        u32::try_from(v)
            .unwrap_or_else(|_| panic!("opaque kernel: {name} {v} does not fit 32 bits"))
            .to_le_bytes()
    };
    let mut b = Vec::with_capacity(24);
    b.extend(src.to_le_bytes());
    b.extend(u32_of("ops", ops));
    b.extend(u32_of("bytes_in", bytes_in));
    b.extend(u32_of("bytes_out", bytes_out));
    b.extend(0u32.to_le_bytes());
    b
}
