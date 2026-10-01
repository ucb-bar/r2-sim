//! The YAML hardware loader (architecture Section 8).
//!
//! A file has two sections, `global` and `device` (the root node). Every
//! node has the sections `fabric`, `memory`, `compute`, `neutrino` and
//! `children`, and every node but the root a `bridge`. A child entry has a
//! `name` and either its sections inline or `config: FILE`, a file
//! (relative to the including file) that holds them; other keys of the
//! entry override the file's (mappings merge key by key, anything else is
//! replaced). A file may instead be `base: FILE` plus `set: [PATH=VALUE,
//! ...]`: the base configuration with overrides. The loader flattens the
//! includes, applies the overrides, resolves names, then builds.
//!
//! Fabric: a bus is a data bus, or a control bus with `control: true`
//! (protocol messages). A node's `bridge: {data, control, bw_bytes,
//! control_bw_bytes, latency}` joins its first data bus to the parent data
//! bus `data` and its first control bus to the parent control bus
//! `control`, each with one resource per direction (`control_bw_bytes`
//! defaults to `bw_bytes`). `slave` (memories) and `master` (compute, and
//! the Neutrino) are a bus or a list of buses; the default Neutrino master
//! is the node's first control bus and every data bus. A transfer takes
//! the route with the fewest bridges among them.
//!
//! Every element has a dotted name from the root (`root.gpc0.sm1.atlas`);
//! `$this` and `$parent` in a bus reference name the node being defined
//! and its parent. An override PATH is such a name followed by keys inside
//! the element (`root.gpc0.sm1.atlas.rate.bf16`, `global.msg_latency`);
//! `*` matches any run of characters inside one component, and `[i]`
//! indexes a list. VALUE is YAML.
//!
//! Keys beyond Section 8.1:
//!
//! * `global`: `msg_latency` (extra cycles per remote-protocol message),
//!   `cocharge` (own | longest), `log` (interval logs),
//!   `livelock_cycles` (idle cycles with continuing resends that stop the
//!   run; default 10 x the largest `t_retry` plus the longest route
//!   latency), `flow` (readyvalid | ticket), `hop_queue` (messages one
//!   hop stage holds per protocol class, default 1: a pipeline register,
//!   not a buffer to size).
//! * buses and bridges: a bus or bridge without `bw_bytes` is unlimited.
//! * nodes: `fidelity` (null | numeric; inherited; the root defaults to
//!   null).
//! * `neutrino`: every key of [`NeutrinoConfig`] (`scoreboard`, `execution`, `pending`,
//!   `versions` (default 256), `inbound_refs`, `inbound_invokes`, `outbound_deps`, `outbound_invokes`,
//!   `issue_headroom` (default: the node's number of compute elements), `t_retry`,
//!   `c_invoke`, `c_wait`, `c_version`, `c_vreset`, `c_complete`,
//!   `c_load`, `c_store`, `c_spawn`, `c_resolve`, `scalar_rate`,
//!   `complete_rate`), and `master`.
//! * compute: `utilization` (tensor and vector), `inflight` and
//!   `outstanding` on every type (default 1).
//!
//! Addressing: an address is [field 1] .. [field n] [offset]. A child's
//! field value is its index among its parent's children plus one; value 0
//! at a level is the parent's own space. Scratch, dram and sync ranges are
//! offsets inside the node's own space; cache ranges are global. A node
//! owns its scratch, dram and sync ranges.
//!
//! Rates and sizes: YAML numbers, or strings with a rational ("2/3") or a
//! unit ("512 KiB").

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nu_model::{DepConfig, NeutrinoId, Range};
use serde::Deserialize;
use serde_yaml::{Mapping, Value};

use super::{MemBlock, NeutrinoConfig, PeConfig, PeParams};
use crate::profile::{Dtype, PeId, ProfileRegistry};
use crate::resource::{Kind, ResourceId};
use crate::sim::Sim;
use crate::time::Ratio;
use crate::topology::{BusId, BusTopology};

/// A configuration that cannot be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(msg.into()))
}

// ---- loading, includes and overrides ----

fn read(path: &Path) -> Result<Value, ConfigError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ConfigError(format!("{}: {e}", path.display())))?;
    serde_yaml::from_str(&text).map_err(|e| ConfigError(format!("{}: {e}", path.display())))
}

fn dir_of(path: &Path) -> PathBuf {
    path.parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

/// The expanded tree of a parsed file in `dir`, and its overrides.
fn expand(mut v: Value, dir: &Path) -> Result<(Value, Vec<String>), ConfigError> {
    let map = v
        .as_mapping_mut()
        .ok_or_else(|| ConfigError("a configuration is a mapping".into()))?;
    let sets = match map.remove("set") {
        None => Vec::new(),
        Some(s) => serde_yaml::from_value(s).map_err(|e| ConfigError(format!("set: {e}")))?,
    };
    if let Some(base) = map.remove("base") {
        let Some(base) = base.as_str() else {
            return err("base is a file name");
        };
        if !map.is_empty() {
            return err("a file with base has only base and set");
        }
        let path = dir.join(base);
        let (tree, mut base_sets) = expand(read(&path)?, &dir_of(&path))?;
        base_sets.extend(sets);
        return Ok((tree, base_sets));
    }
    if let Some(device) = map.get_mut("device") {
        expand_node(device, dir)?;
    }
    Ok((v, sets))
}

/// Replaces every child entry with `config` by the included sections plus
/// the entry's own keys, recursively.
fn expand_node(node: &mut Value, dir: &Path) -> Result<(), ConfigError> {
    let Some(children) = node.get_mut("children").and_then(Value::as_sequence_mut) else {
        return Ok(());
    };
    for c in children {
        let Some(entry) = c.as_mapping_mut() else {
            return err("a child entry is a mapping");
        };
        let child_dir = match entry.remove("config") {
            None => dir.to_path_buf(),
            Some(f) => {
                let Some(f) = f.as_str() else {
                    return err("config is a file name");
                };
                let path = dir.join(f);
                let Value::Mapping(mut inc) = read(&path)? else {
                    return err(format!("{}: not a mapping", path.display()));
                };
                if inc.contains_key("name") {
                    return err(format!(
                        "{}: the name comes from the parent's entry",
                        path.display()
                    ));
                }
                for (k, v) in std::mem::take(entry) {
                    merge(&mut inc, k, v);
                }
                *entry = inc;
                dir_of(&path)
            }
        };
        expand_node(c, &child_dir)?;
    }
    Ok(())
}

/// Sets key `k` of `m` to `v`; two mappings merge key by key.
fn merge(m: &mut Mapping, k: Value, v: Value) {
    match (m.get_mut(&k), v) {
        (Some(Value::Mapping(old)), Value::Mapping(new)) => {
            for (k2, v2) in new {
                merge(old, k2, v2);
            }
        }
        (_, v) => {
            m.insert(k, v);
        }
    }
}

/// One override path component: a name pattern and list indexes.
struct Seg {
    key: String,
    idx: Vec<usize>,
}

fn parse_path(path: &str) -> Result<Vec<Seg>, ConfigError> {
    path.split('.')
        .map(|part| {
            let (key, rest) = part.split_once('[').unwrap_or((part, ""));
            let mut idx = Vec::new();
            for i in rest.split('[').filter(|s| !s.is_empty()) {
                let n = i
                    .strip_suffix(']')
                    .and_then(|n| n.parse().ok())
                    .ok_or_else(|| ConfigError(format!("bad index in {path:?}")))?;
                idx.push(n);
            }
            if key.is_empty() {
                return err(format!("empty component in {path:?}"));
            }
            Ok(Seg {
                key: key.to_string(),
                idx,
            })
        })
        .collect()
}

/// `*` in `pat` matches any run of characters.
fn glob(pat: &str, s: &str) -> bool {
    let parts: Vec<&str> = pat.split('*').collect();
    if parts.len() == 1 {
        return pat == s;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !s.starts_with(first) || !s.ends_with(last) || s.len() < first.len() + last.len() {
        return false;
    }
    let mut rest = &s[first.len()..s.len() - last.len()];
    for p in &parts[1..parts.len() - 1] {
        match rest.find(p) {
            Some(i) => rest = &rest[i + p.len()..],
            None => return false,
        }
    }
    true
}

fn descend<'a>(v: &'a mut Value, idx: &[usize]) -> Option<&'a mut Value> {
    match idx.split_first() {
        None => Some(v),
        Some((&i, rest)) => descend(v.as_sequence_mut()?.get_mut(i)?, rest),
    }
}

/// Sets `v` at the keys `segs` below the mapping `t`; returns the number of
/// places set. A missing last key without `*` or `[i]` is added, directly
/// in `t` only when `insert` is true (`t` was named without `*`).
fn set_in_map(t: &mut Mapping, segs: &[Seg], v: &Value, insert: bool) -> usize {
    let (seg, rest) = (&segs[0], &segs[1..]);
    let keys: Vec<Value> = t
        .keys()
        .filter(|k| k.as_str().is_some_and(|k| glob(&seg.key, k)))
        .cloned()
        .collect();
    let mut n = 0;
    for k in keys {
        let Some(cur) = descend(t.get_mut(&k).expect("key"), &seg.idx) else {
            continue;
        };
        if rest.is_empty() {
            *cur = v.clone();
            n += 1;
        } else if let Some(sub) = cur.as_mapping_mut() {
            n += set_in_map(sub, rest, v, true);
        }
    }
    if insert && n == 0 && rest.is_empty() && seg.idx.is_empty() && !seg.key.contains('*') {
        t.insert(Value::String(seg.key.clone()), v.clone());
        n = 1;
    }
    n
}

fn name_of(v: &Value) -> Option<&str> {
    v.get("name").and_then(Value::as_str)
}

/// Sets `v` at `segs` below node `node`: a child, an element of `fabric`,
/// `memory` or `compute`, or a key of the node. `literal`: the node was
/// named without `*`.
fn set_in_node(node: &mut Value, segs: &[Seg], v: &Value, literal: bool) -> usize {
    let (seg, rest) = (&segs[0], &segs[1..]);
    let mut n = 0;
    let Some(map) = node.as_mapping_mut() else {
        return 0;
    };
    for list in ["children", "fabric", "memory", "compute"] {
        let Some(items) = map.get_mut(list).and_then(Value::as_sequence_mut) else {
            continue;
        };
        for item in items {
            if !name_of(item).is_some_and(|name| glob(&seg.key, name)) {
                continue;
            }
            if rest.is_empty() {
                continue;
            }
            let literal = !seg.key.contains('*');
            n += if list == "children" {
                set_in_node(item, rest, v, literal)
            } else {
                item.as_mapping_mut()
                    .map_or(0, |m| set_in_map(m, rest, v, literal))
            };
        }
    }
    if n == 0 && !["children", "fabric", "memory", "compute", "name"].contains(&seg.key.as_str()) {
        n += set_in_map(map, segs, v, literal);
    }
    n
}

/// Applies the override `s` ("PATH=VALUE") to the expanded tree.
fn apply_set(tree: &mut Value, s: &str) -> Result<(), ConfigError> {
    let (path, value) = s
        .split_once('=')
        .ok_or_else(|| ConfigError(format!("override {s:?} is not PATH=VALUE")))?;
    let value = value.trim();
    let v: Value = serde_yaml::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
    let segs = parse_path(path.trim())?;
    let n = if segs[0].key == "global" && segs.len() > 1 {
        tree.get_mut("global")
            .and_then(Value::as_mapping_mut)
            .map_or(0, |g| set_in_map(g, &segs[1..], &v, true))
    } else {
        match tree.get_mut("device") {
            Some(d)
                if name_of(d).is_some_and(|name| glob(&segs[0].key, name)) && segs.len() > 1 =>
            {
                set_in_node(d, &segs[1..], &v, !segs[0].key.contains('*'))
            }
            _ => 0,
        }
    };
    if n == 0 {
        return err(format!("override {s:?}: nothing matches {path:?}"));
    }
    Ok(())
}

// ---- the schema ----

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    global: Global,
    device: NodeSec,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Global {
    neutrino_granule: u64,
    #[allow(dead_code)]
    clock_ghz: Option<f64>,
    #[serde(default = "default_msg_bytes")]
    msg_bytes: u64,
    #[serde(default)]
    msg_latency: u64,
    /// "readyvalid" (default) or "ticket".
    flow: Option<String>,
    /// Messages a hop stage holds (a pipeline register; default 1).
    hop_queue: Option<usize>,
    cocharge: Option<String>,
    #[serde(default)]
    log: bool,
    livelock_cycles: Option<u64>,
    address: AddrSec,
}

fn default_msg_bytes() -> u64 {
    64
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AddrSec {
    #[serde(default)]
    fields: Vec<FieldSec>,
    offset_bits: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldSec {
    #[allow(dead_code)]
    name: String,
    bits: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeSec {
    name: String,
    #[serde(default)]
    fabric: Vec<BusSec>,
    #[serde(default)]
    memory: Vec<MemSec>,
    #[serde(default)]
    compute: Vec<CompSec>,
    neutrino: Option<NuSec>,
    #[serde(default)]
    children: Vec<NodeSec>,
    bridge: Option<BridgeSec>,
    fidelity: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BusSec {
    name: String,
    topology: Option<String>,
    bw_bytes: Option<Num>,
    #[serde(default)]
    latency: u64,
    /// A control bus (protocol messages).
    #[serde(default)]
    control: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BridgeSec {
    /// The parent data bus.
    data: String,
    /// The parent control bus.
    control: String,
    bw_bytes: Option<Num>,
    control_bw_bytes: Option<Num>,
    #[serde(default)]
    latency: u64,
}

/// A bus reference or a list of them.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum Buses {
    One(String),
    Many(Vec<String>),
}

impl Buses {
    fn list(&self) -> Vec<&str> {
        match self {
            Buses::One(s) => vec![s.as_str()],
            Buses::Many(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemSec {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    addr: Vec<[Num; 2]>,
    bw_bytes: Option<Num>,
    #[serde(default)]
    latency: u64,
    slave: Option<Buses>,
    master: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct NuSec {
    scoreboard: Option<usize>,
    execution: Option<usize>,
    pending: Option<usize>,
    versions: Option<usize>,
    inbound_refs: Option<usize>,
    outbound_deps: Option<usize>,
    outbound_invokes: Option<usize>,
    inbound_invokes: Option<usize>,
    issue_headroom: Option<usize>,
    t_retry: Option<u64>,
    c_invoke: Option<u64>,
    c_wait: Option<u64>,
    c_version: Option<u64>,
    c_vreset: Option<u64>,
    c_complete: Option<u64>,
    c_load: Option<u64>,
    c_store: Option<u64>,
    c_spawn: Option<u64>,
    c_resolve: Option<u64>,
    scalar_rate: Option<Num>,
    complete_rate: Option<Num>,
    master: Option<Buses>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompSec {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    rate: HashMap<String, Num>,
    quantize: Option<String>,
    operand_reuse: Option<String>,
    acc_buffer: Option<Num>,
    overhead: Option<u64>,
    utilization: Option<Num>,
    latency: Option<u64>,
    inflight: Option<u32>,
    outstanding: Option<u32>,
    master: Option<Buses>,
}

/// A number: a YAML integer or float, or a string with a unit or a ratio.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
enum Num {
    Int(u64),
    Float(f64),
    Str(String),
}

impl Num {
    fn bytes(&self) -> Result<u64, ConfigError> {
        match self {
            Num::Int(n) => Ok(*n),
            Num::Float(f) => err(format!("size {f} is not an integer")),
            Num::Str(s) => parse_size(s),
        }
    }

    fn rate(&self) -> Result<Ratio, ConfigError> {
        match self {
            Num::Int(n) if *n > 0 => Ok(Ratio::int(*n)),
            Num::Int(_) => err("rate 0"),
            Num::Float(f) => parse_rate(&f.to_string()),
            Num::Str(s) => parse_rate(s),
        }
    }
}

fn opt_rate(n: &Option<Num>) -> Result<Option<Ratio>, ConfigError> {
    n.as_ref().map(Num::rate).transpose()
}

/// "4096", "0x1000", "0x1_0000", "512 KiB", "1 GiB".
pub(crate) fn parse_size(s: &str) -> Result<u64, ConfigError> {
    let t = s.trim().replace('_', "");
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return u64::from_str_radix(h, 16).or_else(|_| err(format!("bad hex size {s:?}")));
    }
    let (num, unit) = match t.find(|c: char| !c.is_ascii_digit()) {
        Some(i) => (&t[..i], t[i..].trim()),
        None => (t.as_str(), ""),
    };
    let n: u64 = num.parse().or_else(|_| err(format!("bad size {s:?}")))?;
    let mul = match unit {
        "" | "B" => 1,
        "KiB" => 1 << 10,
        "MiB" => 1 << 20,
        "GiB" => 1 << 30,
        _ => return err(format!("bad size unit in {s:?}")),
    };
    Ok(n * mul)
}

/// "64", "64 B/cycle", "0.5", "2/3 lane-op/cycle".
pub(crate) fn parse_rate(s: &str) -> Result<Ratio, ConfigError> {
    let num = s.split_whitespace().next().unwrap_or("");
    let bad = || ConfigError(format!("bad rate {s:?}"));
    let r = if let Some((a, b)) = num.split_once('/') {
        Ratio::new(a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?)
    } else if let Some((i, f)) = num.split_once('.') {
        let den = 10u64.pow(f.len() as u32);
        let whole: u64 = if i.is_empty() {
            0
        } else {
            i.parse().map_err(|_| bad())?
        };
        let frac: u64 = f.parse().map_err(|_| bad())?;
        Ratio::new(whole * den + frac, den)
    } else {
        Ratio::int(num.parse().map_err(|_| bad())?)
    };
    if r.num == 0 {
        return Err(bad());
    }
    Ok(r)
}

// ---- the built simulation ----

/// A simulation built from a configuration, with name lookups by dotted
/// name from the root.
pub struct Built {
    /// The simulation, ready for programs.
    pub sim: Sim,
    /// True when some subtree has `numeric` fidelity; the caller then sets
    /// a Numeric backend.
    pub numeric: bool,
    nodes: HashMap<String, NeutrinoId>,
    pes: HashMap<String, PeId>,
    memories: HashMap<String, Range>,
    sync_only: HashMap<NeutrinoId, Range>,
    own: Vec<Range>,
}

impl Built {
    /// Every PE with its dotted name.
    pub fn pe_paths(&self) -> Vec<(String, PeId)> {
        let mut v: Vec<(String, PeId)> = self.pes.iter().map(|(k, &p)| (k.clone(), p)).collect();
        v.sort_by_key(|x| x.1 .0);
        v
    }

    /// A node by its dotted name ("root.gpc0.sm1").
    pub fn nu(&self, name: &str) -> NeutrinoId {
        *self
            .nodes
            .get(name)
            .unwrap_or_else(|| panic!("no node {name:?}"))
    }

    /// A PE by its dotted name ("root.gpc0.sm1.atlas").
    pub fn pe(&self, name: &str) -> PeId {
        *self
            .pes
            .get(name)
            .unwrap_or_else(|| panic!("no PE {name:?}"))
    }

    /// A PE by its dotted name, if it exists.
    pub fn try_pe(&self, name: &str) -> Option<PeId> {
        self.pes.get(name).copied()
    }

    /// A scratch or dram block by its dotted name, if it exists.
    pub fn try_memory(&self, name: &str) -> Option<Range> {
        self.memories.get(name).copied()
    }

    /// A scratch or dram block's range by its dotted name.
    pub fn memory(&self, name: &str) -> Range {
        *self
            .memories
            .get(name)
            .unwrap_or_else(|| panic!("no memory {name:?}"))
    }

    /// The sync-only range of node `nu`, if it has one.
    pub fn sync_only(&self, nu: NeutrinoId) -> Option<Range> {
        self.sync_only.get(&nu).copied()
    }

    /// The node's own address space (its prefix).
    pub fn own_space(&self, nu: NeutrinoId) -> Range {
        self.own[nu.0 as usize]
    }

    /// The name of a resource of the built simulation.
    pub fn sim_resource_name(&self, r: ResourceId) -> &str {
        &self.sim.resources().get(r).name
    }
}

/// Builds the configuration in the file `path`, with the overrides `sets`
/// applied after the file's own.
pub fn build_file<S: AsRef<str>>(
    path: &Path,
    sets: &[S],
    profiles: ProfileRegistry,
) -> Result<Built, ConfigError> {
    let (tree, own) = expand(read(path)?, &dir_of(path))?;
    build_tree(tree, &own, sets, profiles)
}

/// Builds the configuration `text`, whose includes are relative to `dir`,
/// with the overrides `sets`.
pub fn build_str<S: AsRef<str>>(
    text: &str,
    dir: &Path,
    sets: &[S],
    profiles: ProfileRegistry,
) -> Result<Built, ConfigError> {
    let v: Value = serde_yaml::from_str(text).map_err(|e| ConfigError(e.to_string()))?;
    let (tree, own) = expand(v, dir)?;
    build_tree(tree, &own, sets, profiles)
}

/// Builds the configuration `text` without includes or overrides.
pub fn build(text: &str, profiles: ProfileRegistry) -> Result<Built, ConfigError> {
    build_str::<&str>(text, Path::new("."), &[], profiles)
}

fn build_tree<S: AsRef<str>>(
    mut tree: Value,
    own_sets: &[String],
    sets: &[S],
    profiles: ProfileRegistry,
) -> Result<Built, ConfigError> {
    for s in own_sets {
        apply_set(&mut tree, s)?;
    }
    for s in sets {
        apply_set(&mut tree, s.as_ref())?;
    }
    let file: File = serde_yaml::from_value(tree).map_err(|e| ConfigError(e.to_string()))?;
    Builder::new(file, profiles)?.run()
}

/// Resolves `$this` and `$parent` in a reference made inside node `this`.
fn resolve(name: &str, this: &str, parent: Option<&str>) -> Result<String, ConfigError> {
    if let Some(rest) = name.strip_prefix("$this") {
        return Ok(format!("{this}{rest}"));
    }
    if let Some(rest) = name.strip_prefix("$parent") {
        let Some(p) = parent else {
            return err(format!("{this}: $parent at the root"));
        };
        return Ok(format!("{p}{rest}"));
    }
    Ok(name.to_string())
}

/// A memory block whose buses are resolved at the end.
struct PendingMem {
    block: MemBlock,
    slaves: Vec<String>,
}

/// A cache: a bridge from its slave bus to its master bus through its port.
struct PendingCache {
    name: String,
    port: ResourceId,
    slave: String,
    master: String,
    latency: u64,
}

/// A node still to build: its section, its parent (id and name), its
/// address field values and its parent's fidelity.
type Pending<'a> = (&'a NodeSec, Option<(NeutrinoId, String)>, Vec<u64>, bool);

struct Builder {
    file: File,
    total_bits: u32,
    built: Built,
    buses: HashMap<String, BusId>,
    mems: Vec<PendingMem>,
    caches: Vec<PendingCache>,
    /// Scratch memories with their slave bus name and port, for the SMEM
    /// port of compute elements.
    scratch_ports: Vec<(String, String, ResourceId)>,
    /// Per Neutrino, for the table check.
    table_sizes: Vec<TableSizes>,
}

impl Builder {
    fn new(file: File, profiles: ProfileRegistry) -> Result<Self, ConfigError> {
        let g = &file.global;
        let total_bits =
            g.address.offset_bits + g.address.fields.iter().map(|f| f.bits).sum::<u32>();
        if total_bits > 64 {
            return err("address fields and offset exceed 64 bits");
        }
        let mut sim = Sim::new(profiles, g.log);
        sim.set_msg_bytes(g.msg_bytes);
        sim.set_msg_latency(g.msg_latency);
        match g.cocharge.as_deref() {
            None | Some("own") => {}
            Some("longest") => sim
                .resources_mut()
                .set_cocharge(crate::resource::CoCharge::Longest),
            Some(c) => return err(format!("unknown cocharge {c:?}")),
        }
        match g.flow.as_deref() {
            None | Some("readyvalid") => {}
            Some("ticket") => sim.set_flow(crate::sim::Flow::Ticket),
            Some(f) => return err(format!("unknown flow {f:?} (readyvalid, ticket)")),
        }
        if let Some(n) = g.hop_queue {
            if n == 0 {
                return err("global.hop_queue must be at least 1");
            }
            sim.set_hop_queue(n);
        }
        if let Some(c) = g.livelock_cycles {
            sim.set_livelock_cycles(c);
        }
        Ok(Self {
            file,
            total_bits,
            built: Built {
                sim,
                numeric: false,
                nodes: HashMap::new(),
                pes: HashMap::new(),
                memories: HashMap::new(),
                sync_only: HashMap::new(),
                own: Vec::new(),
            },
            buses: HashMap::new(),
            mems: Vec::new(),
            caches: Vec::new(),
            scratch_ports: Vec::new(),
            table_sizes: Vec::new(),
        })
    }

    fn bus(&self, name: &str, what: &str) -> Result<BusId, ConfigError> {
        self.buses
            .get(name)
            .copied()
            .ok_or_else(|| ConfigError(format!("{what}: no bus {name:?}")))
    }

    fn run(mut self) -> Result<Built, ConfigError> {
        let root = std::mem::replace(
            &mut self.file.device,
            NodeSec {
                name: String::new(),
                fabric: Vec::new(),
                memory: Vec::new(),
                compute: Vec::new(),
                neutrino: None,
                children: Vec::new(),
                bridge: None,
                fidelity: None,
            },
        );
        if root.bridge.is_some() {
            return err("the root has no bridge");
        }
        // Parents before children, in configuration order.
        let mut stack: Vec<Pending> = vec![(&root, None, Vec::new(), false)];
        while let Some((sec, parent, fields, parent_numeric)) = stack.pop() {
            let (nu, path, numeric) = self.node(sec, parent.as_ref(), &fields, parent_numeric)?;
            let max = self
                .file
                .global
                .address
                .fields
                .get(fields.len())
                .map_or(0, |f| (1u64 << f.bits) - 1);
            if sec.children.len() as u64 > max {
                return err(format!("{path}: too many children for its address field"));
            }
            for (i, c) in sec.children.iter().enumerate().rev() {
                let mut f = fields.clone();
                f.push(i as u64 + 1);
                stack.push((c, Some((nu, path.clone())), f, numeric));
            }
        }
        for c in std::mem::take(&mut self.caches) {
            let (a, b) = (self.bus(&c.slave, &c.name)?, self.bus(&c.master, &c.name)?);
            let lane = Some(c.port);
            let short = c.name.rsplit('.').next().unwrap_or(&c.name).to_string();
            self.built
                .sim
                .add_bridge(a, &short, b, &short, lane, lane, c.latency);
        }
        for m in std::mem::take(&mut self.mems) {
            let mut block = m.block;
            let short = block.name.rsplit('.').next().unwrap_or("").to_string();
            for slave in &m.slaves {
                let bus = self.bus(slave, &block.name)?;
                block.attach.push(self.built.sim.attach(bus, &short, true));
            }
            self.built.sim.add_memory(block);
        }
        self.built.sim.finish_fabric().map_err(ConfigError)?;
        check_tables(&self.table_sizes)?;
        Ok(self.built)
    }

    /// Adds node `sec` under `parent`; returns its id, name and fidelity.
    fn node(
        &mut self,
        sec: &NodeSec,
        parent: Option<&(NeutrinoId, String)>,
        fields: &[u64],
        parent_numeric: bool,
    ) -> Result<(NeutrinoId, String, bool), ConfigError> {
        let af = &self.file.global.address;
        if fields.len() > af.fields.len() {
            return err(format!(
                "node {} is deeper than the address fields",
                sec.name
            ));
        }
        let path = match parent {
            None => sec.name.clone(),
            Some((_, p)) => format!("{p}.{}", sec.name),
        };
        let ppath = parent.map(|p| p.1.as_str());
        // Own space: the node's field values, zero below.
        let mut base = 0u64;
        let mut shift = self.total_bits;
        for (l, f) in af.fields.iter().enumerate() {
            shift -= f.bits;
            if let Some(v) = fields.get(l) {
                base |= v << shift;
            }
        }
        let own = Range::from_len(base, 1u64 << af.offset_bits);
        let numeric = match sec.fidelity.as_deref() {
            None => parent_numeric,
            Some("numeric") => true,
            Some("null") => false,
            Some(f) => return err(format!("{path}: unknown fidelity {f:?}")),
        };
        self.built.numeric |= numeric;
        // Owned ranges and the sync-only space.
        let mut owned = Vec::new();
        let mut sync = None;
        for m in &sec.memory {
            for [a, b] in &m.addr {
                let (a, b) = (a.bytes()?, b.bytes()?);
                if b <= a {
                    return err(format!("{path}.{}: empty range", m.name));
                }
                let local = matches!(m.kind.as_str(), "scratch" | "dram" | "sync");
                let r = if local {
                    Range::new(base + a, base + b)
                } else {
                    Range::new(a, b)
                };
                if local && !own.covers(&r) {
                    return err(format!("{path}.{}: {r} is outside its space {own}", m.name));
                }
                match m.kind.as_str() {
                    "scratch" | "dram" => owned.push(r),
                    "sync" => {
                        if sync.replace(r).is_some() {
                            return err(format!("{path}: more than one sync range"));
                        }
                        owned.push(r);
                    }
                    _ => {}
                }
            }
        }
        // Bridge resources first: data, then control.
        let bridge = match (parent, &sec.bridge) {
            (None, _) => None,
            (Some(_), None) => return err(format!("{path}: a child needs a bridge")),
            (Some(_), Some(br)) => {
                let data = self.bus(&resolve(&br.data, &path, ppath)?, &path)?;
                let control = self.bus(&resolve(&br.control, &path, ppath)?, &path)?;
                if self.built.sim.topology().bus(data).control
                    || !self.built.sim.topology().bus(control).control
                {
                    return err(format!(
                        "{path}: bridge data must be a data bus and control a control bus"
                    ));
                }
                let rate = opt_rate(&br.bw_bytes)?;
                let ctl_rate = opt_rate(&br.control_bw_bytes)?.or(rate);
                let r = &mut self.built.sim;
                let up = r.lane(&format!("{path}.bridge.up"), rate);
                let down = r.lane(&format!("{path}.bridge.down"), rate);
                let up_ctl = r.lane(&format!("{path}.bridge.up.ctl"), ctl_rate);
                let down_ctl = r.lane(&format!("{path}.bridge.down.ctl"), ctl_rate);
                Some((data, control, (up, down), (up_ctl, down_ctl), br.latency))
            }
        };
        let Some(nsec) = &sec.neutrino else {
            return err(format!(
                "{path}: no neutrino section (nodes without a Neutrino are not supported yet)"
            ));
        };
        let cfg = nu_config(
            &path,
            nsec,
            self.file.global.neutrino_granule,
            sec.compute.len(),
        )?;
        self.table_sizes.push(TableSizes {
            name: path.clone(),
            scoreboard: cfg.tables.scoreboard_capacity,
            execution: cfg.tables.execution_capacity,
            headroom: cfg.tables.issue_headroom,
            inbound: cfg.inbound_invokes,
            outbound: cfg.outbound_invokes,
        });
        let nu = match parent {
            None => self.built.sim.add_neutrino(cfg, &owned),
            Some((p, _)) => self.built.sim.add_child_neutrino(*p, cfg, &owned),
        };
        self.built.own.push(own);
        self.built.nodes.insert(path.clone(), nu);
        if let Some(r) = sync {
            self.built.sim.add_sync_only(r);
            self.built.sync_only.insert(nu, r);
        }
        // Buses.
        let mut own_buses = Vec::new();
        for b in &sec.fabric {
            let name = format!("{path}.{}", b.name);
            let topology = match b.topology.as_deref().unwrap_or("xbar") {
                "xbar" => BusTopology::Xbar,
                "bus" => BusTopology::Shared,
                "ring" => return err(format!("{name}: topology ring is not implemented")),
                t => return err(format!("{name}: unknown topology {t:?} (xbar, bus)")),
            };
            let id = self.built.sim.add_bus(
                &name,
                topology,
                opt_rate(&b.bw_bytes)?,
                b.latency,
                b.control,
            );
            if self.buses.insert(name.clone(), id).is_some() {
                return err(format!("{name}: two buses of this name"));
            }
            own_buses.push((name, id, b.control));
        }
        let first = |control: bool| own_buses.iter().find(|b| b.2 == control).map(|b| b.1);
        let (Some(hub_data), Some(hub_ctl)) = (first(false), first(true)) else {
            return err(format!("{path}: needs a data bus and a control bus"));
        };
        // The Neutrino: its control bus (messages) and its data buses (loads
        // and stores).
        let masters: Vec<String> = match &nsec.master {
            Some(m) => m
                .list()
                .into_iter()
                .map(|b| resolve(b, &path, ppath))
                .collect::<Result<_, _>>()?,
            None => {
                let mut v: Vec<String> = own_buses
                    .iter()
                    .filter(|b| b.2)
                    .take(1)
                    .map(|b| b.0.clone())
                    .collect();
                v.extend(own_buses.iter().filter(|b| !b.2).map(|b| b.0.clone()));
                v
            }
        };
        let mut nu_ctl = None;
        let mut nu_data = Vec::new();
        for m in &masters {
            let bus = self.bus(m, &format!("{path}.neutrino"))?;
            let att = self.built.sim.attach(bus, "neutrino", true);
            if self.built.sim.topology().bus(bus).control {
                if nu_ctl.replace(att).is_some() {
                    return err(format!("{path}.neutrino: masters two control buses"));
                }
            } else {
                nu_data.push(att);
            }
        }
        let Some(nu_ctl) = nu_ctl else {
            return err(format!("{path}.neutrino: masters no control bus"));
        };
        self.built.sim.set_hub(nu, nu_ctl, nu_data);
        if let Some((data, control, (up, down), (up_ctl, down_ctl), latency)) = bridge {
            let sim = &mut self.built.sim;
            sim.add_bridge(hub_data, "bridge", data, &sec.name, up, down, latency);
            sim.add_bridge(
                hub_ctl, "bridge", control, &sec.name, up_ctl, down_ctl, latency,
            );
        }
        self.memories(sec, &path, ppath, nu, own.base, numeric)?;
        self.compute(sec, &path, ppath, nu)?;
        Ok((nu, path, numeric))
    }

    fn memories(
        &mut self,
        sec: &NodeSec,
        path: &str,
        ppath: Option<&str>,
        nu: NeutrinoId,
        base: u64,
        numeric: bool,
    ) -> Result<(), ConfigError> {
        for m in &sec.memory {
            let name = format!("{path}.{}", m.name);
            if m.kind == "sync" {
                continue;
            }
            let Some(bw) = &m.bw_bytes else {
                return err(format!("{name}: needs bw_bytes"));
            };
            let port = self.built.sim.resources_mut().add_flow(
                &format!("{name}.port"),
                Kind::Bandwidth,
                bw.rate()?,
            );
            let Some(slave) = &m.slave else {
                return err(format!("{name}: needs slave"));
            };
            let slaves: Vec<String> = slave
                .list()
                .into_iter()
                .map(|b| resolve(b, path, ppath))
                .collect::<Result<_, _>>()?;
            match m.kind.as_str() {
                "scratch" | "dram" => {
                    if m.master.is_some() {
                        return err(format!("{name}: only a cache has a master bus"));
                    }
                    let [[a, b]] = m.addr.as_slice() else {
                        return err(format!("{name}: needs exactly one addr range"));
                    };
                    let range = Range::new(base + a.bytes()?, base + b.bytes()?);
                    if m.kind == "scratch" {
                        for sl in &slaves {
                            self.scratch_ports.push((name.clone(), sl.clone(), port));
                        }
                    }
                    self.built.memories.insert(name.clone(), range);
                    self.mems.push(PendingMem {
                        block: MemBlock {
                            name,
                            owner: nu,
                            attach: Vec::new(),
                            numeric,
                            range,
                            port,
                            load_latency: m.latency,
                        },
                        slaves,
                    });
                }
                "cache" => {
                    let [slave] = slaves.as_slice() else {
                        return err(format!("{name}: a cache is a slave on one bus"));
                    };
                    let Some(master) = &m.master else {
                        return err(format!("{name}: a cache needs master"));
                    };
                    self.caches.push(PendingCache {
                        name,
                        port,
                        slave: slave.clone(),
                        master: resolve(master, path, ppath)?,
                        latency: m.latency,
                    });
                }
                k => return err(format!("{name}: unknown memory type {k:?}")),
            }
        }
        Ok(())
    }

    /// The port of the one scratch memory that is a slave on one of
    /// `buses`.
    fn smem_port(&self, name: &str, buses: &[String]) -> Result<ResourceId, ConfigError> {
        let mut found: Vec<&(String, String, ResourceId)> = self
            .scratch_ports
            .iter()
            .filter(|s| buses.contains(&s.1))
            .collect();
        found.dedup_by(|a, b| a.0 == b.0);
        match found.as_slice() {
            [s] => Ok(s.2),
            [] => err(format!("{name}: no scratch memory is a slave on {buses:?}")),
            [a, b, ..] => err(format!(
                "{name}: {} and {} are slaves on {buses:?}",
                a.0, b.0
            )),
        }
    }

    fn compute(
        &mut self,
        sec: &NodeSec,
        path: &str,
        ppath: Option<&str>,
        nu: NeutrinoId,
    ) -> Result<(), ConfigError> {
        for c in &sec.compute {
            let name = format!("{path}.{}", c.name);
            let Some(master) = &c.master else {
                return err(format!("{name}: needs master"));
            };
            let masters: Vec<String> = master
                .list()
                .into_iter()
                .map(|b| resolve(b, path, ppath))
                .collect::<Result<_, _>>()?;
            let mut atts = Vec::new();
            for m in &masters {
                let bus = self.bus(m, &name)?;
                atts.push(self.built.sim.attach(bus, &c.name, false));
            }
            let utilization = || opt_rate(&c.utilization).map(|u| u.unwrap_or(Ratio::ONE));
            let params = match c.kind.as_str() {
                "tensor" => {
                    let mut rates = Vec::new();
                    for (k, v) in &c.rate {
                        let d = match k.as_str() {
                            "bf16" => Dtype::Bf16,
                            "f32" => Dtype::F32,
                            "fp8" => Dtype::Fp8,
                            "mxfp4" => Dtype::Mxfp4,
                            _ => return err(format!("{name}: unknown data type {k:?}")),
                        };
                        rates.push((d, v.rate()?));
                    }
                    rates.sort_by_key(|r| r.0 as u8);
                    let Some(max) = rates
                        .iter()
                        .map(|r| r.1)
                        .max_by(|a, b| (a.num * b.den).cmp(&(b.num * a.den)))
                    else {
                        return err(format!("{name}: tensor needs rate"));
                    };
                    let array = self.built.sim.resources_mut().add_flow(
                        &format!("{name}.array"),
                        Kind::TensorOps,
                        max,
                    );
                    let smem_port = self.smem_port(&name, &masters)?;
                    let quantize = match c.quantize.as_deref().unwrap_or("tile") {
                        "tile" => true,
                        "none" => false,
                        q => return err(format!("{name}: quantize {q:?}")),
                    };
                    let row_panel = match c.operand_reuse.as_deref().unwrap_or("none") {
                        "none" => false,
                        "row_panel" => true,
                        r => return err(format!("{name}: operand_reuse {r:?}")),
                    };
                    PeParams::Atlas {
                        quantize,
                        row_panel,
                        acc_buffer: c
                            .acc_buffer
                            .as_ref()
                            .map(Num::bytes)
                            .transpose()?
                            .unwrap_or(0),
                        array,
                        rates,
                        smem_port,
                        utilization: utilization()?,
                    }
                }
                "vector" => {
                    let Some(ops) = c.rate.get("lane_op") else {
                        return err(format!("{name}: vector needs rate.lane_op"));
                    };
                    let vector = self.built.sim.resources_mut().add_flow(
                        &format!("{name}.vector"),
                        Kind::VectorOps,
                        ops.rate()?,
                    );
                    PeParams::Muon {
                        vector,
                        masters: atts,
                        utilization: utilization()?,
                        overhead: c.overhead.unwrap_or(0),
                    }
                }
                "dma" => PeParams::Dma {
                    links: Vec::new(),
                    latency: c.latency.unwrap_or(0),
                    masters: atts,
                },
                k => return err(format!("{name}: unknown compute type {k:?}")),
            };
            let id = self.built.sim.add_pe(
                nu,
                PeConfig {
                    name: name.clone(),
                    max_inflight: c.inflight.unwrap_or(1),
                    max_outstanding: c.outstanding.unwrap_or(1),
                    params,
                },
            );
            self.built.pes.insert(name, id);
        }
        Ok(())
    }
}

/// The table check of Section 3.3 (ready/valid): every owner takes at
/// least one invoke into its inbound invoke table, admits one into its
/// scoreboard (execution table above the issue headroom), and every
/// requester can hold one in its outbound invoke table.
fn check_tables(nus: &[TableSizes]) -> Result<(), ConfigError> {
    for t in nus {
        if t.inbound < 1 || t.outbound < 1 {
            return err(format!(
                "{}: inbound invokes {} and outbound invokes {} must be at least 1",
                t.name, t.inbound, t.outbound
            ));
        }
        if t.execution <= t.headroom || t.scoreboard < 1 {
            return err(format!(
                "{}: execution table {} must exceed the issue headroom {} (scoreboard {})",
                t.name, t.execution, t.headroom, t.scoreboard
            ));
        }
    }
    Ok(())
}

/// The table sizes of one Neutrino, for [`check_tables`].
struct TableSizes {
    name: String,
    scoreboard: usize,
    execution: usize,
    headroom: usize,
    inbound: usize,
    outbound: usize,
}

/// The Neutrino of node `name` with `pes` PEs (the default issue headroom).
fn nu_config(
    name: &str,
    s: &NuSec,
    granule: u64,
    pes: usize,
) -> Result<NeutrinoConfig, ConfigError> {
    let d = NeutrinoConfig::default();
    let c = d.costs;
    Ok(NeutrinoConfig {
        name: name.to_string(),
        tables: DepConfig {
            scoreboard_capacity: s.scoreboard.unwrap_or(d.tables.scoreboard_capacity),
            execution_capacity: s.execution.unwrap_or(d.tables.execution_capacity),
            pending_capacity: s.pending.unwrap_or(d.tables.pending_capacity),
            version_capacity: s.versions.unwrap_or(d.tables.version_capacity),
            inbound_ref_capacity: s.inbound_refs.unwrap_or(d.tables.inbound_ref_capacity),
            issue_headroom: s.issue_headroom.unwrap_or(pes),
            granule,
        },
        costs: super::NuCosts {
            invoke: s.c_invoke.unwrap_or(c.invoke),
            wait: s.c_wait.unwrap_or(c.wait),
            version: s.c_version.unwrap_or(c.version),
            vreset: s.c_vreset.unwrap_or(c.vreset),
            complete: s.c_complete.unwrap_or(c.complete),
            load: s.c_load.unwrap_or(c.load),
            store: s.c_store.unwrap_or(c.store),
            spawn: s.c_spawn.unwrap_or(c.spawn),
            resolve: s.c_resolve.unwrap_or(c.resolve),
        },
        scalar_rate: opt_rate(&s.scalar_rate)?.unwrap_or(d.scalar_rate),
        complete_rate: opt_rate(&s.complete_rate)?.unwrap_or(d.complete_rate),
        outbound_deps: s.outbound_deps.unwrap_or(d.outbound_deps),
        outbound_invokes: s.outbound_invokes.unwrap_or(d.outbound_invokes),
        inbound_invokes: s.inbound_invokes.unwrap_or(d.inbound_invokes),
        t_retry: s.t_retry.unwrap_or(d.t_retry),
    })
}
