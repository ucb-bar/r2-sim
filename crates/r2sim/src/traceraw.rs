//! The raw trace dump for the r2-vis converter (`--trace-raw DIR`).
//!
//! The format belongs to r2-vis: `r2-vis/docs/trace-raw.md`, version 1.8
//! (1.8: a forwarded VersionReset has an inbound_invokes entry that ends
//! when it takes effect at the head of the owner's table, and the version
//! entry it frees ends then; a task record has `flags`, bit 0 untracked,
//! and is 112 B; no dependency names an untracked task as producer) and
//! 1.9 (`dst_mask` in place of the task record's `pad`: the mask of the
//! destination range in granules; all ones for a plain range).
//! A dependency's unversioned producer holds the granule of its address
//! under the producer's mask.
//! The dump is `manifest.json` (written before the run), `strings.txt`,
//! one file of fixed-size little-endian records per record type, and
//! `summary.json` (written after the run, also after a fatal error).
//!
//! The writer is a [`Tracer`]: it only observes. Records are encoded into
//! fixed arrays (no allocation per record) and appended to a 1 MiB
//! buffered writer per file, optionally through one zstd stream per file.
//! It keeps only the open task records, the open dependency records and
//! the producer map:
//!
//! * a versioned dependency resolves to the task that produced
//!   (base, version); an unversioned one to the latest invoked task whose
//!   unversioned `dst` contains the address;
//! * a local dependency resolves at the producer's completion plus the
//!   consumer's `c_resolve` (at the consumer's invoke when the producer is
//!   already done); a remote one at the arrival of its `AwaitAck(done)` or
//!   `Resolve`;
//! * a dependency on a pending entry (no producer yet) waits for the next
//!   invoke that writes its key.
//!
//! Table entries (`entries.bin`) are kept open until they leave their
//! table; an outbound invoke entry waits for its task's invoke at the
//! owner, which gives its gid.
//!
//! Unlimited links (links without a resource) get ids after the real
//! resources and records synthesized from the granted interval of the
//! chunk, load, store or message hop that uses them (at least 1 cycle).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use nu_model::{Dep, NeutrinoId};
use serde_json::{json, Value};
use sim_core::tracer as tr;
use sim_core::{
    AttachId, Built, BusTopology, Charge, ChunkGrant, Component, Cycle, Dir, Link, Owner, PeParams,
    ResourceId, Role, RouteUse, Sim, Stats, Table, TableEntry, TaskEntered, Topology, Tracer, Who,
    MESSAGE_KINDS,
};

const BUF: usize = 1 << 20;
const NULL16: u16 = u16::MAX;
const NULL32: u32 = u32::MAX;
const NULL64: u64 = u64::MAX;
/// No slab entry.
const NONE: u32 = u32::MAX;

/// `tasks.flags` bit 0: an untracked task (1.8).
const UNTRACKED: u32 = 1;

/// One slot in a charge record's `usage` (trace-raw 1.7).
const USAGE_ONE: u32 = 65_536;

/// Record layouts, in file order (trace-raw.md Section 4).
const TASKS: &[(&str, &str)] = &[
    ("gid", "u32"),
    ("nu", "u16"),
    ("src_nu", "u16"),
    ("pe", "u16"),
    ("opcode", "u16"),
    ("version", "u32"),
    ("label", "u64"),
    ("dst_base", "u64"),
    ("dst_len", "u64"),
    ("call", "u64"),
    ("invoke", "u64"),
    ("ready", "u64"),
    ("issue", "u64"),
    ("complete", "u64"),
    ("retire", "u64"),
    ("queued", "u64"),
    ("sent", "u64"),
    ("flags", "u32"),
    ("dst_mask", "u32"),
];
const CHUNKS: &[(&str, &str)] = &[
    ("gid", "u32"),
    ("index", "u32"),
    ("start", "u64"),
    ("end", "u64"),
    ("culprit", "u16"),
    ("queue_culprit", "u16"),
    ("pad", "u32"),
];
const DEPS: &[(&str, &str)] = &[
    ("consumer", "u32"),
    ("producer", "u32"),
    ("resolved", "u64"),
    ("index", "u16"),
    ("state", "u8"),
    ("flags", "u8"),
    ("pad", "u32"),
];
const CHARGES: &[(&str, &str)] = &[
    ("start", "u64"),
    ("amount", "u64"),
    ("span", "u32"),
    ("usage", "u32"),
    ("gid", "u32"),
    ("resource", "u16"),
    ("dir", "u8"),
    ("sub", "u8"),
];
const PROGRAMS: &[(&str, &str)] = &[
    ("nu", "u16"),
    ("pad", "u16"),
    ("name", "u32"),
    ("spawn", "u32"),
    ("pad2", "u32"),
    ("start", "u64"),
    ("end", "u64"),
];
const MARKS: &[(&str, &str)] = &[
    ("time", "u64"),
    ("name", "u32"),
    ("nu", "u16"),
    ("pad", "u16"),
];
const ENTRIES: &[(&str, &str)] = &[
    ("start", "u64"),
    ("end", "u64"),
    ("base", "u64"),
    ("len", "u64"),
    ("gid", "u32"),
    ("version", "u32"),
    ("nu", "u16"),
    ("peer", "u16"),
    ("table", "u8"),
    ("flags", "u8"),
    ("pad", "u16"),
];

/// A record layout: (field, type) in file order.
type Layout = &'static [(&'static str, &'static str)];

/// The record types: file stem, layout and record size.
pub const RECORDS: [(&str, Layout, usize); 7] = [
    ("tasks", TASKS, 112),
    ("chunks", CHUNKS, 32),
    ("deps", DEPS, 24),
    ("charges", CHARGES, 32),
    ("programs", PROGRAMS, 32),
    ("marks", MARKS, 16),
    ("entries", ENTRIES, 48),
];
const T_TASKS: usize = 0;
const T_CHUNKS: usize = 1;
const T_DEPS: usize = 2;
const T_CHARGES: usize = 3;
const T_PROGRAMS: usize = 4;
const T_MARKS: usize = 5;
const T_ENTRIES: usize = 6;

/// One record being encoded: little-endian fields in layout order.
struct Rec<const N: usize> {
    b: [u8; N],
    at: usize,
}

impl<const N: usize> Rec<N> {
    #[inline(always)]
    fn new() -> Self {
        Self { b: [0; N], at: 0 }
    }
    #[inline(always)]
    fn put<const K: usize>(&mut self, v: [u8; K]) -> &mut Self {
        self.b[self.at..self.at + K].copy_from_slice(&v);
        self.at += K;
        self
    }
    #[inline(always)]
    fn u8(&mut self, v: u8) -> &mut Self {
        self.put([v])
    }
    #[inline(always)]
    fn u16(&mut self, v: u16) -> &mut Self {
        self.put(v.to_le_bytes())
    }
    #[inline(always)]
    fn u32(&mut self, v: u32) -> &mut Self {
        self.put(v.to_le_bytes())
    }
    #[inline(always)]
    fn u64(&mut self, v: u64) -> &mut Self {
        self.put(v.to_le_bytes())
    }
    #[inline(always)]
    fn done(&self) -> &[u8; N] {
        debug_assert_eq!(self.at, N, "record fields fill the record");
        &self.b
    }
}

fn o16(v: Option<u16>) -> u16 {
    v.unwrap_or(NULL16)
}
fn o32(v: Option<u32>) -> u32 {
    v.unwrap_or(NULL32)
}
fn o64(v: Option<u64>) -> u64 {
    v.unwrap_or(NULL64)
}
/// Cycles as a `u32` field (saturating).
fn c32(v: Cycle) -> u32 {
    v.min(NULL32 as u64 - 1) as u32
}

/// One output file. The simulation thread fills 1 MiB buffers; a writer
/// thread writes them (through one zstd stream when compressing) and
/// returns them for reuse, so neither the system calls nor the
/// compression run on the simulation thread.
struct Sink {
    buf: Vec<u8>,
    full: Option<std::sync::mpsc::SyncSender<Vec<u8>>>,
    empty: std::sync::mpsc::Receiver<Vec<u8>>,
    thread: Option<std::thread::JoinHandle<std::io::Result<()>>>,
}

/// Writes the buffers that arrive on `full` and returns them on `empty`.
fn pump(
    w: &mut impl Write,
    full: std::sync::mpsc::Receiver<Vec<u8>>,
    empty: std::sync::mpsc::Sender<Vec<u8>>,
) -> std::io::Result<()> {
    for mut b in full {
        w.write_all(&b)?;
        b.clear();
        // The simulation thread may be gone at the end.
        let _ = empty.send(b);
    }
    Ok(())
}

/// Buffers in flight per file before the simulation thread waits (32
/// MiB: absorbs a slow disk for a while).
const IN_FLIGHT: usize = 32;

impl Sink {
    fn create(path: &Path, zstd: Option<i32>) -> std::io::Result<Self> {
        let path = match zstd {
            None => path.to_path_buf(),
            Some(_) => {
                let mut p = path.as_os_str().to_owned();
                p.push(".zst");
                PathBuf::from(p)
            }
        };
        let file = File::create(&path)?;
        let (full_tx, full_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(IN_FLIGHT);
        let (empty_tx, empty_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let name = format!(
            "r2-trace-{}",
            path.file_name().map_or("".into(), |n| n.to_string_lossy())
        );
        let thread = std::thread::Builder::new()
            .name(name)
            .spawn(move || match zstd {
                None => {
                    let mut f = file;
                    pump(&mut f, full_rx, empty_tx)?;
                    f.flush()
                }
                Some(level) => {
                    let w = BufWriter::with_capacity(BUF, file);
                    let mut z = zstd::stream::write::Encoder::new(w, level)?;
                    pump(&mut z, full_rx, empty_tx)?;
                    z.finish()?.flush()
                }
            })?;
        Ok(Self {
            buf: Vec::with_capacity(BUF),
            full: Some(full_tx),
            empty: empty_rx,
            thread: Some(thread),
        })
    }

    #[inline(always)]
    fn write(&mut self, b: &[u8]) -> std::io::Result<()> {
        if self.buf.len() + b.len() > BUF {
            self.send()?;
        }
        self.buf.extend_from_slice(b);
        Ok(())
    }

    /// Hands the buffer to the writer thread and takes an empty one.
    #[cold]
    #[inline(never)]
    fn send(&mut self) -> std::io::Result<()> {
        let next = self
            .empty
            .try_recv()
            .unwrap_or_else(|_| Vec::with_capacity(BUF));
        let full = std::mem::replace(&mut self.buf, next);
        let tx = self.full.as_ref().expect("open sink");
        if tx.send(full).is_err() {
            // The writer thread stopped on an error; finish() reports it.
            return Err(std::io::Error::other("trace writer thread stopped"));
        }
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        let sent = if self.buf.is_empty() {
            Ok(())
        } else {
            self.send()
        };
        drop(self.full.take());
        let joined = self
            .thread
            .take()
            .expect("writer thread")
            .join()
            .map_err(|_| std::io::Error::other("trace writer thread panicked"))?;
        joined.and(sent)
    }
}

/// The record types of [`Filter::records`]: the files, and `messages`
/// (the protocol message records of `charges.bin`).
pub const RECORD_TYPES: [&str; 8] = [
    "tasks", "chunks", "deps", "charges", "messages", "programs", "marks", "entries",
];

/// Which records the dump keeps (architecture Section 7.3, trace-raw.md
/// Section 1.1). Every filter is optional.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filter {
    /// Record types of [`RECORD_TYPES`]; `None`: all.
    pub records: Option<Vec<String>>,
    /// Subtree paths ("root.gpc0"); `None`: the whole tree.
    pub nodes: Option<Vec<String>>,
    /// `[lo, hi)` in cycles.
    pub window: Option<(u64, u64)>,
}

impl Filter {
    /// True when a record type is kept.
    fn keeps(&self, kind: &str) -> bool {
        self.records
            .as_ref()
            .is_none_or(|r| r.iter().any(|k| k == kind))
    }

    /// True when some filter is set.
    fn partial(&self) -> bool {
        let all = self
            .records
            .as_ref()
            .is_none_or(|r| RECORD_TYPES.iter().all(|k| r.iter().any(|x| x == k)));
        !all || self.nodes.is_some() || self.window.is_some()
    }
}

/// What the dump writes: the record types, the subtree, the window and
/// the compression.
#[derive(Debug, Clone, Default)]
pub struct Opts {
    pub dir: PathBuf,
    /// zstd level; `None`: uncompressed.
    pub zstd: Option<i32>,
    pub filter: Filter,
}

/// `trace.yaml` (architecture Section 7.5); every key is optional.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceConfig {
    records: Option<Vec<String>>,
    nodes: Option<Vec<String>>,
    window: Option<[u64; 2]>,
    /// zstd level; 0: off.
    zstd: Option<i32>,
}

/// The dump options from `--trace-config FILE` (if any), then the
/// `trace.KEY=VALUE` overrides (VALUE is YAML), then `--trace-zstd`.
pub fn opts(
    dir: PathBuf,
    file: Option<&Path>,
    sets: &[String],
    zstd_flag: bool,
) -> Result<Opts, String> {
    let mut map = match file {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            match serde_yaml::from_str::<serde_yaml::Value>(&text)
                .map_err(|e| format!("{}: {e}", p.display()))?
            {
                serde_yaml::Value::Mapping(m) => m,
                serde_yaml::Value::Null => serde_yaml::Mapping::new(),
                _ => return Err(format!("{}: not a mapping", p.display())),
            }
        }
        None => serde_yaml::Mapping::new(),
    };
    for s in sets {
        let (k, v) = s
            .strip_prefix("trace.")
            .and_then(|kv| kv.split_once('='))
            .ok_or_else(|| format!("--set {s}: want trace.KEY=VALUE"))?;
        let v: serde_yaml::Value =
            serde_yaml::from_str(v).map_err(|e| format!("--set {s}: {e}"))?;
        map.insert(serde_yaml::Value::String(k.to_string()), v);
    }
    let c: TraceConfig = serde_yaml::from_value(serde_yaml::Value::Mapping(map))
        .map_err(|e| format!("trace configuration: {e}"))?;
    let zstd = match c.zstd {
        Some(0) => None,
        Some(l) if (1..=19).contains(&l) => Some(l),
        Some(l) => return Err(format!("trace configuration: zstd level {l} (0 to 19)")),
        None => zstd_flag.then_some(1),
    };
    Ok(Opts {
        dir,
        zstd,
        filter: Filter {
            records: c.records,
            nodes: c.nodes,
            window: c.window.map(|[lo, hi]| (lo, hi)),
        },
    })
}

/// The run's identity for the manifest.
#[derive(Debug, Clone, Default)]
pub struct RunInfo {
    pub config: String,
    pub program: String,
    pub params_file: Option<String>,
    /// `--param key=value` overrides.
    pub params: Vec<String>,
    /// `--set PATH=VALUE` configuration overrides, then the other flags
    /// that change the model (`--acc-buffer`, `--operand-reuse`,
    /// `--quantize`, `--cocharge`, `--flow`) as `"--flag value"`.
    pub overrides: Vec<String>,
    pub granule: u64,
}

/// The result of the dump, read after the run.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Records written per record type (order of [`RECORDS`]).
    pub counts: [u64; 7],
    /// The first write error.
    pub error: Option<String>,
}

/// An open task record (written at retire or at the end).
#[derive(Debug, Clone, Copy)]
struct OpenTask {
    nu: u16,
    src: u16,
    pe: Option<u16>,
    opcode: u16,
    version: Option<u32>,
    label: Option<u64>,
    dst_base: u64,
    dst_len: u64,
    call: Cycle,
    invoke: Cycle,
    ready: Option<Cycle>,
    issue: Option<Cycle>,
    complete: Option<Cycle>,
    queued: Option<Cycle>,
    sent: Option<Cycle>,
    /// Bit 0: untracked (1.8).
    flags: u32,
    /// The mask of `dst` in granules (1.9).
    dst_mask: u32,
    /// Resolved deps kept until the task's retire decides the window.
    parked: u32,
    /// Its invoke call (source Neutrino, number), when it has one: the key
    /// of its owner name in `named`.
    call_key: Option<(u32, u64)>,
}

/// An open dependency record, in a slab with linked lists.
#[derive(Debug, Clone, Copy)]
struct OpenDep {
    consumer: u64,
    /// The consumer's Neutrino (its `c_resolve`).
    nu: u32,
    producer: Option<u64>,
    key: Dep,
    index: u16,
    state: u8,
    flags: u8,
    /// A remote request is open for it.
    requested: bool,
    /// The resolution time of a parked record.
    resolved: Option<Cycle>,
    /// Next entry of the list it is on.
    next: u32,
}

const REMOTE: u8 = 1;
const VERSIONED: u8 = 2;

/// A program as a charge's owner: this bit and its spawn task's gid; the
/// root program (no spawn task) is `ROOT_PROGRAM` (trace-raw 1.6).
const PROGRAM: u32 = 0x8000_0000;
const ROOT_PROGRAM: u32 = 0xFFFF_FFFE;

/// A charge's owner: a gid (task or program reference; `None`: nothing),
/// or an invoke call whose task has no gid yet.
#[derive(Debug, Clone, Copy)]
enum Own {
    Gid(Option<u32>),
    Held((u32, u64)),
}

/// An open table entry (written when it leaves or at the end).
#[derive(Debug, Clone, Copy)]
struct OpenEntry {
    start: Cycle,
    base: u64,
    len: u64,
    version: Option<u64>,
    gid: Option<u64>,
    peer: Option<NeutrinoId>,
    flags: u8,
}

/// A table entry's key: (Neutrino, table, id).
type EntryKey = (u32, u8, u64);

/// The raw trace writer.
pub struct RawTracer {
    files: [Option<Sink>; 7],
    strings: BufWriter<File>,
    string_ids: HashMap<String, u32>,
    out: Rc<RefCell<Outcome>>,
    counts: [u64; 7],
    error: Option<std::io::Error>,
    /// Real resources; unlimited links have ids from here on.
    unlimited: HashMap<Link, u16>,
    topo: Topology,
    /// Unlimited link ids of the route between two attachments.
    route_links: Vec<Option<Span>>,
    /// Link ids of routes and folds.
    links: Vec<u16>,
    /// Control routes by (from, to) Neutrino: per rated hop, the unlimited
    /// links folded into it; and all unlimited links in route order.
    folds: Vec<Option<Fold>>,
    resolve: Vec<Cycle>,
    tasks: IdMap<u64, OpenTask>,
    /// Producer of (base, version).
    versioned: IdMap<(u64, u64), u64>,
    /// Latest unversioned writer by `dst` base: (end, gid, mask).
    unversioned: BTreeMap<u64, (u64, u64, u32)>,
    /// The Neutrino granule (the unit of a `dst` mask).
    granule: u64,
    max_unversioned: u64,
    deps: Vec<OpenDep>,
    free: Vec<u32>,
    /// Local deps waiting for a producer's completion: list heads.
    waiting: IdMap<u64, u32>,
    /// Deps on a pending entry: versioned by key, unversioned in a list.
    pending_v: IdMap<(u64, u64), u32>,
    pending_u: Vec<u32>,
    /// Open remote requests: (requester, request) -> dep.
    remote: IdMap<(u32, u32), u32>,
    /// The remote deps of the last invoked task (their requests follow).
    last_remote: Vec<u32>,
    /// Running programs by Neutrino: (name, spawn, start).
    programs: IdMap<u32, Program>,
    /// Owners (trace-raw 1.6): the gid of an invoke call (source, number)
    /// from its task's invoke to its retire, and the charge records of an
    /// invoke call held until its task has a gid.
    named: IdMap<(u32, u64), u32>,
    held: IdMap<(u32, u64), Vec<[u8; 32]>>,
    held_now: usize,
    held_peak: usize,
    /// Open table entries.
    entries: IdMap<EntryKey, OpenEntry>,
    /// Outbound invoke entries that left before their task's invoke at
    /// the owner (which gives their gid), with their end.
    unnamed: IdMap<(u32, u64), (OpenEntry, Cycle)>,
    entries_on: bool,
    /// Filters: kept Neutrinos (subtree), kept resources (by owner, the
    /// unlimited links included), the window and the charge kinds.
    keep_nu: Vec<bool>,
    keep_res: Vec<bool>,
    lo: Cycle,
    hi: Cycle,
    windowed: bool,
    subtree: bool,
    charges: bool,
    messages: bool,
    deps_on: bool,
}

/// A running program: (name string id, spawn gid, start).
type Program = (u32, Option<u64>, Cycle);

/// A run of link ids in the link store: (first, count).
type Span = (u32, u32);

/// The unlimited links of a control route.
#[derive(Debug, Default)]
struct Fold {
    /// Per rated hop, the unlimited links folded into it.
    hops: Vec<Span>,
    /// All unlimited links, in route order.
    all: Span,
}

/// A hasher for integer keys (the multiply-rotate step of FxHash).
#[derive(Default, Clone, Copy)]
struct IdHasher(u64);

impl std::hash::Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u8(&mut self, v: u8) {
        self.write_u64(v as u64)
    }
    fn write_u16(&mut self, v: u16) {
        self.write_u64(v as u64)
    }
    fn write_u32(&mut self, v: u32) {
        self.write_u64(v as u64)
    }
    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64)
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

/// A map with integer keys.
type IdMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<IdHasher>>;

impl RawTracer {
    /// Creates the dump directory, writes `manifest.json` and opens the
    /// record files. The returned handle holds the counts after the run.
    pub fn create(
        o: &Opts,
        b: &Built,
        run: &RunInfo,
    ) -> Result<(Self, Rc<RefCell<Outcome>>), String> {
        let ctx = |e: std::io::Error| format!("{}: {e}", o.dir.display());
        let topo = b.sim.topology();
        let f = &o.filter;
        let nodes: Vec<String> = (0..topo.len())
            .map(|i| topo.path_name(NeutrinoId(i as u32)))
            .collect();
        if let Some(r) = &f.records {
            if let Some(k) = r.iter().find(|k| !RECORD_TYPES.contains(&k.as_str())) {
                return Err(format!(
                    "unknown record type {k:?} (known: {RECORD_TYPES:?})"
                ));
            }
        }
        if let Some(ns) = &f.nodes {
            if let Some(n) = ns.iter().find(|n| !nodes.contains(n)) {
                return Err(format!("no node {n:?}"));
            }
        }
        if let Some((lo, hi)) = f.window {
            if lo >= hi {
                return Err(format!("empty window [{lo}, {hi})"));
            }
        }
        std::fs::create_dir_all(&o.dir).map_err(ctx)?;
        let in_subtree = |name: &str| {
            f.nodes.as_ref().is_none_or(|ns| {
                ns.iter()
                    .any(|n| name == n || name.starts_with(&format!("{n}.")))
            })
        };
        let keep_nu: Vec<bool> = nodes.iter().map(|n| in_subtree(n)).collect();
        let unlimited = unlimited_links(&b.sim);
        let manifest = manifest(b, run, &unlimited, f);
        // A resource is kept when its owner node is (a resource with no
        // owner only without a subtree filter).
        let keep_res: Vec<bool> = manifest["resources"]
            .as_array()
            .expect("resources")
            .iter()
            .map(|r| match r["owner"].as_u64() {
                Some(o) => keep_nu[o as usize],
                None => f.nodes.is_none(),
            })
            .collect();
        let text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
        std::fs::write(o.dir.join("manifest.json"), text).map_err(ctx)?;
        let mut files: [Option<Sink>; 7] = Default::default();
        let (charges, messages) = (f.keeps("charges"), f.keeps("messages"));
        for (i, (stem, _, _)) in RECORDS.iter().enumerate() {
            let on = f.keeps(stem) || (i == T_CHARGES && messages);
            if on {
                let path = o.dir.join(format!("{stem}.bin"));
                files[i] = Some(Sink::create(&path, o.zstd).map_err(ctx)?);
            }
        }
        let strings = BufWriter::new(File::create(o.dir.join("strings.txt")).map_err(ctx)?);
        let out = Rc::new(RefCell::new(Outcome::default()));
        let resolve = (0..topo.len())
            .map(|i| b.sim.resolve_cost(NeutrinoId(i as u32)))
            .collect();
        let t = Self {
            files,
            strings,
            string_ids: HashMap::new(),
            out: out.clone(),
            counts: [0; 7],
            error: None,
            unlimited,
            topo: topo.links_only(),
            route_links: vec![None; topo.attachment_count() * topo.attachment_count()],
            links: Vec::new(),
            folds: (0..topo.len() * topo.len()).map(|_| None).collect(),
            resolve,
            tasks: IdMap::default(),
            versioned: IdMap::default(),
            unversioned: BTreeMap::new(),
            granule: run.granule.max(1),
            max_unversioned: 0,
            deps: Vec::new(),
            free: Vec::new(),
            waiting: IdMap::default(),
            pending_v: IdMap::default(),
            pending_u: Vec::new(),
            remote: IdMap::default(),
            last_remote: Vec::new(),
            programs: IdMap::default(),
            named: IdMap::default(),
            held: IdMap::default(),
            held_now: 0,
            held_peak: 0,
            entries: IdMap::default(),
            unnamed: IdMap::default(),
            entries_on: f.keeps("entries"),
            keep_nu,
            keep_res,
            lo: f.window.map_or(0, |w| w.0),
            hi: f.window.map_or(Cycle::MAX, |w| w.1),
            windowed: f.window.is_some(),
            subtree: f.nodes.is_some(),
            charges,
            messages,
            deps_on: f.keeps("deps"),
        };
        Ok((t, out))
    }

    #[inline(always)]
    fn emit(&mut self, kind: usize, b: &[u8]) {
        if let Some(f) = self.files[kind].as_mut() {
            if let Err(e) = f.write(b) {
                self.error.get_or_insert(e);
            }
            self.counts[kind] += 1;
        }
    }

    /// True when `[start, end)` overlaps the window (an empty interval:
    /// when its cycle is in it).
    #[inline(always)]
    fn in_window(&self, start: Cycle, end: Cycle) -> bool {
        start < self.hi && (end > self.lo || (end == start && start >= self.lo))
    }

    /// True when task `t` (retired at `retire`, or open at the end) is
    /// kept: its Neutrino is in the subtree and `[invoke, retire)`
    /// overlaps the window.
    fn task_kept(&self, t: &OpenTask, retire: Option<Cycle>) -> bool {
        self.keep_nu[t.nu as usize] && self.in_window(t.invoke, retire.unwrap_or(Cycle::MAX))
    }

    /// The string id of `s`, written to `strings.txt` at first use.
    fn string(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.string_ids.get(s) {
            return i;
        }
        let i = self.string_ids.len() as u32;
        // A string is one line.
        let line = s.replace('\n', " ");
        if let Err(e) = writeln!(self.strings, "{line}") {
            self.error.get_or_insert(e);
        }
        self.string_ids.insert(s.to_string(), i);
        i
    }

    // ---- records ----

    fn write_task(&mut self, gid: u64, t: &OpenTask, retire: Option<Cycle>) {
        if !self.task_kept(t, retire) {
            return;
        }
        let mut r = Rec::<112>::new();
        r.u32(gid as u32)
            .u16(t.nu)
            .u16(t.src)
            .u16(o16(t.pe))
            .u16(t.opcode)
            .u32(o32(t.version))
            .u64(o64(t.label))
            .u64(t.dst_base)
            .u64(t.dst_len)
            .u64(t.call)
            .u64(t.invoke)
            .u64(o64(t.ready))
            .u64(o64(t.issue))
            .u64(o64(t.complete))
            .u64(o64(retire))
            .u64(o64(t.queued))
            .u64(o64(t.sent))
            .u32(t.flags)
            .u32(t.dst_mask);
        self.emit(T_TASKS, r.done());
    }

    #[allow(clippy::too_many_arguments)]
    /// One charge record (trace-raw 1.7): `usage` in 1/65536 slot; `sub`
    /// the message kind of a protocol message or the rate class of a
    /// charge on a resource with rate classes.
    fn write_charge(
        &mut self,
        start: Cycle,
        amount: u64,
        span: Cycle,
        usage: u32,
        owner: Own,
        resource: u16,
        dir: Dir,
        sub: Option<u8>,
    ) {
        if !(self.keep_res[resource as usize] && self.in_window(start, start + span)) {
            return;
        }
        let gid = match owner {
            Own::Gid(g) => g,
            Own::Held(_) => None,
        };
        let mut r = Rec::<32>::new();
        r.u64(start)
            .u64(amount)
            .u32(c32(span))
            .u32(usage)
            .u32(o32(gid))
            .u16(resource)
            .u8(dir as u8)
            .u8(sub.unwrap_or(u8::MAX));
        match owner {
            Own::Gid(_) => self.emit(T_CHARGES, r.done()),
            Own::Held(k) => {
                if self.files[T_CHARGES].is_some() {
                    self.held.entry(k).or_default().push(*r.done());
                    self.held_now += 1;
                    self.held_peak = self.held_peak.max(self.held_now);
                }
            }
        }
    }

    /// The owner of a charge (trace-raw 1.6): a task gid, a program
    /// reference, or an invoke whose task has no gid yet (its records are
    /// held until the task's invoke).
    fn owner(&self, who: Who) -> Own {
        let program = |nu: NeutrinoId| {
            Own::Gid(self.programs.get(&nu.0).map(|p| match p.1 {
                Some(spawn) => PROGRAM | spawn as u32,
                None => ROOT_PROGRAM,
            }))
        };
        let invoke = |src: NeutrinoId, n: u64| match self.named.get(&(src.0, n)) {
            Some(&g) => Own::Gid(Some(g)),
            None => Own::Held((src.0, n)),
        };
        match who {
            Who::Task(g) => Own::Gid(Some(g as u32)),
            Who::Program(nu) => program(nu),
            Who::Invoke { nu, n } => invoke(nu, n),
            Who::Message { owner, .. } => match owner {
                Owner::None => Own::Gid(None),
                Owner::Task(g) => Own::Gid(Some(g as u32)),
                Owner::Program { spawn } => Own::Gid(Some(match spawn {
                    Some(g) => PROGRAM | g as u32,
                    None => ROOT_PROGRAM,
                })),
                Owner::Invoke { src, n } => invoke(src, n),
            },
        }
    }

    /// Records on the unlimited links `ids` over `[start, end)` (at least
    /// one cycle).
    #[allow(clippy::too_many_arguments)]
    fn write_unlimited(
        &mut self,
        ids: Span,
        start: Cycle,
        end: Cycle,
        amount: u64,
        owner: Own,
        dir: Dir,
        msg: Option<u8>,
    ) {
        let span = end.saturating_sub(start).max(1);
        for k in ids.0..ids.0 + ids.1 {
            let id = self.links[k as usize];
            // One notional slot on an unlimited link.
            self.write_charge(start, amount, span, USAGE_ONE, owner, id, dir, msg);
        }
    }

    fn write_dep(&mut self, d: &OpenDep, resolved: Option<Cycle>) {
        let mut r = Rec::<24>::new();
        r.u32(d.consumer as u32)
            .u32(o32(d.producer.map(|p| p as u32)))
            .u64(o64(resolved))
            .u16(d.index)
            .u8(d.state)
            .u8(d.flags)
            .u32(0);
        self.emit(T_DEPS, r.done());
    }

    fn write_program(&mut self, nu: u32, (name, spawn, start): Program, end: Option<Cycle>) {
        let mut r = Rec::<32>::new();
        r.u16(nu as u16)
            .u16(0)
            .u32(name)
            .u32(o32(spawn.map(|s| s as u32)))
            .u32(0)
            .u64(start)
            .u64(o64(end));
        self.emit(T_PROGRAMS, r.done());
    }

    fn write_entry(&mut self, (nu, table, _): EntryKey, e: &OpenEntry, end: Option<Cycle>) {
        if !(self.keep_nu[nu as usize] && self.in_window(e.start, end.unwrap_or(Cycle::MAX))) {
            return;
        }
        let mut r = Rec::<48>::new();
        r.u64(e.start)
            .u64(o64(end))
            .u64(e.base)
            .u64(e.len)
            .u32(o32(e.gid.map(|g| g as u32)))
            .u32(o32(e.version.map(|v| v as u32)))
            .u16(nu as u16)
            .u16(o16(e.peer.map(|p| p.0 as u16)))
            .u8(table)
            .u8(e.flags)
            .u16(0);
        self.emit(T_ENTRIES, r.done());
    }

    /// Entry `k` left at `end`: written, or kept until its gid is known
    /// (an outbound invoke whose task has not entered its owner yet).
    fn close_entry(&mut self, k: EntryKey, e: OpenEntry, end: Cycle) {
        let invoke = k.1 == Table::OutboundInvokes as u8 && e.flags & tr::INVOKE_VRESET == 0;
        if invoke && e.gid.is_none() {
            self.unnamed.insert((k.0, k.2), (e, end));
        } else {
            self.write_entry(k, &e, Some(end));
        }
    }

    // ---- links ----

    /// Stores `ids` in the link store.
    fn store(&mut self, ids: &[u16]) -> Span {
        let at = self.links.len() as u32;
        self.links.extend_from_slice(ids);
        (at, ids.len() as u32)
    }

    /// The unlimited links of the route between two attachments.
    fn route_ids(&mut self, from: AttachId, to: AttachId) -> Span {
        let key = from.0 as usize * self.topo.attachment_count() + to.0 as usize;
        if let Some(v) = self.route_links[key] {
            return v;
        }
        let ids: Vec<u16> = self
            .topo
            .links(from, to)
            .into_iter()
            .filter(|(_, lane)| lane.is_none())
            .map(|(l, _)| self.unlimited[&l])
            .collect();
        let span = self.store(&ids);
        self.route_links[key] = Some(span);
        span
    }

    /// The fold of the control route from `a` to `b`: an unlimited link
    /// belongs to the rated hop before it, or to the first one.
    fn fold(&mut self, a: NeutrinoId, b: NeutrinoId) -> &Fold {
        let key = a.0 as usize * self.topo.len() + b.0 as usize;
        if self.folds[key].is_none() {
            let mut hops: Vec<Vec<u16>> = Vec::new();
            let mut all = Vec::new();
            if let (Some(x), Some(y)) = (self.topo.hub(a), self.topo.hub(b)) {
                let mut before: Vec<u16> = Vec::new();
                for (l, lane) in self.topo.links(x, y) {
                    match lane {
                        Some(_) => hops.push(std::mem::take(&mut before)),
                        None => {
                            let id = self.unlimited[&l];
                            all.push(id);
                            match hops.last_mut() {
                                Some(h) => h.push(id),
                                None => before.push(id),
                            }
                        }
                    }
                }
                // Links before the first rated hop belong to it.
                if let Some(first) = hops.first_mut() {
                    first.append(&mut before);
                }
            }
            let f = Fold {
                hops: hops.iter().map(|h| self.store(h)).collect(),
                all: self.store(&all),
            };
            self.folds[key] = Some(f);
        }
        self.folds[key].as_ref().expect("fold")
    }

    // ---- dependencies ----

    fn alloc(&mut self, d: OpenDep) -> u32 {
        match self.free.pop() {
            Some(i) => {
                self.deps[i as usize] = d;
                i
            }
            None => {
                self.deps.push(d);
                self.deps.len() as u32 - 1
            }
        }
    }

    /// Dep `i` resolved at `resolved`: written now when its consumer is
    /// sure to be kept, parked on the consumer when only its retire can
    /// tell (it resolved before the window), freed otherwise.
    fn resolve_dep(&mut self, i: u32, resolved: Cycle) {
        let d = self.deps[i as usize];
        let park = self.windowed && resolved <= self.lo;
        match self.tasks.get_mut(&d.consumer).filter(|_| park) {
            Some(t) => {
                let dep = &mut self.deps[i as usize];
                dep.resolved = Some(resolved);
                dep.next = t.parked;
                t.parked = i;
                return;
            }
            _ => self.write_dep(&d, Some(resolved)),
        }
        self.free.push(i);
    }

    /// Writes (when `keep`) and frees the parked deps of a consumer.
    fn unpark(&mut self, mut i: u32, keep: bool) {
        while i != NONE {
            let d = self.deps[i as usize];
            if keep {
                self.write_dep(&d, d.resolved);
            }
            self.free.push(i);
            i = d.next;
        }
    }

    /// The producer of `key`, if invoked.
    fn producer(&self, key: Dep) -> Option<u64> {
        match key {
            Dep::Versioned { addr, version } => self.versioned.get(&(addr, version)).copied(),
            Dep::Unversioned(a) => {
                let lo = a.saturating_sub(self.max_unversioned);
                let g = self.granule;
                self.unversioned
                    .range(lo..=a)
                    .filter(|(&base, &(end, _, mask))| {
                        end > a
                            && nu_model::Dst {
                                span: nu_model::Range::new(base, end),
                                mask,
                            }
                            .holds(g, a)
                    })
                    .map(|(_, &(_, gid, _))| gid)
                    .max()
            }
        }
    }

    /// The producer's state at an invoke: 1 in the scoreboard, 2
    /// executing, 3 done (a retired task is done).
    fn state(&self, p: u64) -> u8 {
        match self.tasks.get(&p) {
            None => 3,
            Some(t) if t.complete.is_some() => 3,
            Some(t) if t.issue.is_some() => 2,
            Some(_) => 1,
        }
    }

    fn wait_on(&mut self, producer: u64, i: u32) {
        let head = self.waiting.get(&producer).copied().unwrap_or(NONE);
        self.deps[i as usize].next = head;
        self.waiting.insert(producer, i);
    }

    /// Task `gid` writes `key`: deps that wait on a pending entry of it
    /// now wait for its completion.
    fn adopt(&mut self, gid: u64, versioned: Option<(u64, u64)>, dst: nu_model::Dst) {
        let mut adopted: Vec<u32> = Vec::new();
        match versioned {
            Some(_) if self.pending_v.is_empty() => {}
            Some(k) => {
                let mut i = self.pending_v.remove(&k).unwrap_or(NONE);
                while i != NONE {
                    adopted.push(i);
                    i = self.deps[i as usize].next;
                }
            }
            None if self.pending_u.is_empty() => {}
            None => {
                let (deps, g) = (&self.deps, self.granule);
                self.pending_u.retain(|&i| {
                    let a = deps[i as usize].key.addr();
                    let hit = dst.holds(g, a);
                    if hit {
                        adopted.push(i);
                    }
                    !hit
                });
            }
        }
        for i in adopted {
            let d = &mut self.deps[i as usize];
            d.producer = Some(gid);
            d.state = 0;
            if d.flags & REMOTE == 0 {
                self.wait_on(gid, i);
            }
        }
    }
}

impl Tracer for RawTracer {
    fn task_invoke(&mut self, e: &TaskEntered) {
        let d = e.desc;
        let open = OpenTask {
            nu: e.nu.0 as u16,
            src: e.call.src.0 as u16,
            pe: d.pe.map(|p| p.0 as u16),
            opcode: d.opcode.0,
            version: e.version.map(|v| v as u32),
            label: d.label,
            dst_base: d.dst.base,
            dst_len: d.dst.len(),
            call: e.call.call,
            invoke: e.time,
            ready: None,
            issue: None,
            complete: None,
            queued: e.call.queued,
            sent: e.call.sent,
            flags: d.untracked as u32 * UNTRACKED,
            dst_mask: d.dst_mask,
            parked: NONE,
            call_key: None,
        };
        assert!(
            e.gid < 0x7FFF_FFFE,
            "task gid {} collides with program references",
            e.gid
        );
        let key = (e.call.n != 0).then_some((e.call.src.0, e.call.n));
        self.tasks.insert(
            e.gid,
            OpenTask {
                call_key: key,
                ..open
            },
        );
        // The invoke's held charges (its cost and messages) get its gid.
        if let Some(k) = key {
            self.named.insert(k, e.gid as u32);
            if let Some(recs) = self.held.remove(&k) {
                self.held_now -= recs.len();
                let gid = (e.gid as u32).to_le_bytes();
                for mut r in recs {
                    r[24..28].copy_from_slice(&gid);
                    self.emit(T_CHARGES, &r);
                }
            }
        }
        if self.entries_on && e.call.seq != 0 {
            let k = (e.call.src.0, Table::OutboundInvokes as u8, e.call.seq);
            if let Some(o) = self.entries.get_mut(&k) {
                o.gid = Some(e.gid);
            } else if let Some((mut o, end)) = self.unnamed.remove(&(k.0, k.2)) {
                o.gid = Some(e.gid);
                self.write_entry(k, &o, Some(end));
            }
        }
        // This task as a producer; no lookup finds an untracked task.
        let vkey = e.version.map(|v| (d.dst.base, v));
        if !d.untracked {
            match vkey {
                Some(k) => {
                    self.versioned.insert(k, e.gid);
                }
                None => {
                    self.unversioned
                        .insert(d.dst.base, (d.dst.bound, e.gid, d.dst_mask));
                    self.max_unversioned = self.max_unversioned.max(d.dst.len());
                }
            }
            self.adopt(e.gid, vkey, d.held());
        }
        // Its dependencies, when it can be kept.
        self.last_remote.clear();
        if !self.deps_on || !self.keep_nu[e.nu.0 as usize] || e.time >= self.hi {
            return;
        }
        for (k, &key) in d.deps.iter().enumerate() {
            let remote = k < 64 && e.remote_deps >> k & 1 == 1;
            let producer = self.producer(key).filter(|&p| p != e.gid);
            let mut flags = 0;
            if remote {
                flags |= REMOTE;
            }
            if matches!(key, Dep::Versioned { .. }) {
                flags |= VERSIONED;
            }
            let state = producer.map_or(0, |p| self.state(p));
            let i = self.alloc(OpenDep {
                consumer: e.gid,
                nu: e.nu.0,
                producer,
                key,
                index: k as u16,
                state,
                flags,
                requested: false,
                resolved: None,
                next: NONE,
            });
            match (remote, producer) {
                (true, _) => self.last_remote.push(i),
                (false, Some(_)) if state == 3 => self.resolve_dep(i, e.time),
                (false, Some(p)) => self.wait_on(p, i),
                (false, None) => match key {
                    Dep::Versioned { addr, version } => {
                        let head = self
                            .pending_v
                            .get(&(addr, version))
                            .copied()
                            .unwrap_or(NONE);
                        self.deps[i as usize].next = head;
                        self.pending_v.insert((addr, version), i);
                    }
                    Dep::Unversioned(_) => self.pending_u.push(i),
                },
            }
        }
    }

    fn task_ready(&mut self, gid: u64, time: Cycle) {
        if let Some(t) = self.tasks.get_mut(&gid) {
            t.ready = Some(time);
        }
    }

    fn task_issue(&mut self, gid: u64, time: Cycle, _chunks: usize) {
        if let Some(t) = self.tasks.get_mut(&gid) {
            t.issue = Some(time);
        }
    }

    fn task_complete(&mut self, gid: u64, time: Cycle) {
        if let Some(t) = self.tasks.get_mut(&gid) {
            t.complete = Some(time);
        }
        let mut i = self.waiting.remove(&gid).unwrap_or(NONE);
        while i != NONE {
            let next = self.deps[i as usize].next;
            let nu = self.deps[i as usize].nu as usize;
            let at = time + self.resolve.get(nu).copied().unwrap_or(0);
            self.resolve_dep(i, at);
            i = next;
        }
    }

    fn task_retire(&mut self, gid: u64, time: Cycle) {
        if let Some(t) = self.tasks.remove(&gid) {
            if let Some(k) = t.call_key {
                self.named.remove(&k);
            }
            self.write_task(gid, &t, Some(time));
            if t.parked != NONE {
                let keep = self.task_kept(&t, Some(time));
                self.unpark(t.parked, keep);
            }
        }
    }

    fn wants_charges(&self) -> bool {
        self.charges || self.messages
    }

    fn chunk_grant(&mut self, c: &ChunkGrant) {
        if !self.in_window(c.start, c.end) {
            return;
        }
        if self.subtree
            && !self
                .tasks
                .get(&c.gid)
                .is_some_and(|t| self.keep_nu[t.nu as usize])
        {
            return;
        }
        let mut r = Rec::<32>::new();
        r.u32(c.gid as u32)
            .u32(c.index as u32)
            .u64(c.start)
            .u64(c.end)
            .u16(o16(c.culprit.map(|r| r.0 as u16)))
            .u16(o16(c.queue_culprit.map(|r| r.0 as u16)))
            .u32(0);
        self.emit(T_CHUNKS, r.done());
    }

    fn charge(&mut self, c: &Charge, who: Who) {
        if !(if matches!(who, Who::Message { .. }) {
            self.messages
        } else {
            self.charges
        }) {
            return;
        }
        let msg = match who {
            Who::Message { kind, .. } => Some(kind),
            _ => None,
        };
        let owner = self.owner(who);
        // Slots x 65536, rounded down.
        let usage = (c.usage as u128 * USAGE_ONE as u128 / sim_core::SCALE as u128) as u32;
        self.write_charge(
            c.start,
            c.amount,
            c.span,
            usage,
            owner,
            c.resource.0 as u16,
            c.dir,
            msg.or(c.class),
        );
        if let Who::Message {
            from,
            to,
            hop,
            bytes,
            ..
        } = who
        {
            let ids = match self.fold(from, to).hops.get(hop) {
                Some(&h) if h.1 > 0 => h,
                _ => return,
            };
            self.write_unlimited(
                ids,
                c.start,
                c.start + c.span,
                bytes,
                owner,
                Dir::Write,
                msg,
            );
        }
    }

    fn route(&mut self, r: &RouteUse, start: Cycle, end: Cycle, who: Who) {
        if !self.charges {
            return;
        }
        let ids = self.route_ids(r.from, r.to);
        if ids.1 == 0 {
            return;
        }
        let owner = self.owner(who);
        self.write_unlimited(ids, start, end, r.bytes, owner, r.dir, None);
    }

    fn message_free(&mut self, who: Who, start: Cycle, end: Cycle) {
        let Who::Message {
            kind,
            from,
            to,
            bytes,
            ..
        } = who
        else {
            return;
        };
        if !self.messages {
            return;
        }
        let msg = Some(kind);
        let owner = self.owner(who);
        let ids = self.fold(from, to).all;
        self.write_unlimited(ids, start, end, bytes, owner, Dir::Write, msg);
    }

    fn dep_remote_open(&mut self, nu: NeutrinoId, req: u32, gid: u64, key: Dep) {
        let pos = self.last_remote.iter().position(|&i| {
            let d = &self.deps[i as usize];
            d.consumer == gid && d.key == key && !d.requested
        });
        if let Some(p) = pos {
            let i = self.last_remote.swap_remove(p);
            self.deps[i as usize].requested = true;
            self.remote.insert((nu.0, req), i);
        }
    }

    fn dep_remote_done(&mut self, nu: NeutrinoId, req: u32, time: Cycle) {
        if let Some(i) = self.remote.remove(&(nu.0, req)) {
            let d = &mut self.deps[i as usize];
            if d.producer.is_none() {
                let key = d.key;
                let p = self.producer(key);
                self.deps[i as usize].producer = p;
            }
            self.resolve_dep(i, time);
        }
    }

    fn entry_open(&mut self, e: &TableEntry, time: Cycle) {
        if !self.entries_on {
            return;
        }
        let k = (e.nu.0, e.table as u8, e.id);
        match self.entries.get_mut(&k) {
            Some(o) => {
                if e.version.is_some() {
                    o.version = e.version;
                }
                if o.len == 0 {
                    o.len = e.len;
                }
                o.flags |= e.flags;
            }
            None => {
                let o = OpenEntry {
                    start: time,
                    base: e.base,
                    len: e.len,
                    version: e.version,
                    gid: e.gid,
                    peer: e.peer,
                    flags: e.flags,
                };
                self.entries.insert(k, o);
            }
        }
    }

    fn entry_close(&mut self, e: &TableEntry, time: Cycle) {
        if !self.entries_on {
            return;
        }
        let k = (e.nu.0, e.table as u8, e.id);
        let Some(mut o) = self.entries.remove(&k) else {
            return;
        };
        o.version = e.version.or(o.version);
        o.gid = e.gid.or(o.gid);
        o.peer = e.peer.or(o.peer);
        o.flags |= e.flags;
        self.close_entry(k, o, time);
    }

    fn entries_clear(&mut self, nu: NeutrinoId, table: Table, time: Cycle) {
        if !self.entries_on {
            return;
        }
        let mut gone: Vec<(EntryKey, OpenEntry)> = self
            .entries
            .iter()
            .filter(|(k, _)| k.0 == nu.0 && k.1 == table as u8)
            .map(|(k, o)| (*k, *o))
            .collect();
        gone.sort_by_key(|g| g.0);
        for (k, o) in gone {
            self.entries.remove(&k);
            self.close_entry(k, o, time);
        }
    }

    fn program_start(&mut self, nu: NeutrinoId, name: &'static str, spawn: Option<u64>, t: Cycle) {
        let id = self.string(name);
        self.programs.insert(nu.0, (id, spawn, t));
    }

    fn program_return(&mut self, nu: NeutrinoId, time: Cycle) {
        if let Some(p) = self.programs.remove(&nu.0) {
            if self.keep_nu[nu.0 as usize] && self.in_window(p.2, time) {
                self.write_program(nu.0, p, Some(time));
            }
        }
    }

    fn mark(
        &mut self,
        time: Cycle,
        nu: NeutrinoId,
        name: &str,
        _totals: &dyn Fn() -> Vec<[u64; 3]>,
    ) {
        if !(self.keep_nu[nu.0 as usize] && self.in_window(time, time)) {
            return;
        }
        let id = self.string(name);
        let mut r = Rec::<16>::new();
        r.u64(time).u32(id).u16(nu.0 as u16).u16(0);
        self.emit(T_MARKS, r.done());
    }

    fn finish(mut self: Box<Self>, _stats: &mut Stats) {
        // Open records: tasks that did not retire, deps never resolved,
        // programs that did not return.
        let mut open: Vec<(u64, OpenTask)> = self.tasks.iter().map(|(g, t)| (*g, *t)).collect();
        open.sort_by_key(|t| t.0);
        for (gid, t) in &open {
            self.write_task(*gid, t, None);
            let keep = self.task_kept(t, None);
            self.unpark(t.parked, keep);
        }
        // Deps never resolved: their consumers did not retire.
        let mut live = vec![true; self.deps.len()];
        for &i in &self.free {
            live[i as usize] = false;
        }
        let unresolved: Vec<OpenDep> = self
            .deps
            .iter()
            .zip(&live)
            .filter(|(_, &l)| l)
            .map(|(d, _)| *d)
            .collect();
        for d in unresolved {
            let kept = self
                .tasks
                .get(&d.consumer)
                .is_some_and(|t| self.task_kept(t, None));
            if kept {
                self.write_dep(&d, None);
            }
        }
        self.tasks.clear();
        let mut progs: Vec<(u32, Program)> = self.programs.drain().collect();
        progs.sort_by_key(|p| p.0);
        for (nu, p) in progs {
            if self.keep_nu[nu as usize] && self.in_window(p.2, Cycle::MAX) {
                self.write_program(nu, p, None);
            }
        }
        // Entries still in their tables, and outbound invokes whose task
        // never entered its owner.
        let mut open: Vec<(EntryKey, OpenEntry)> = self.entries.drain().collect();
        open.sort_by_key(|e| (e.1.start, e.0));
        for (k, o) in open {
            self.write_entry(k, &o, None);
        }
        let mut unnamed: Vec<_> = self.unnamed.drain().collect();
        unnamed.sort_by_key(|u| u.0);
        for ((nu, seq), (o, end)) in unnamed {
            self.write_entry((nu, Table::OutboundInvokes as u8, seq), &o, Some(end));
        }
        // Held charges whose invoke never reached its owner: no owner.
        let mut held: Vec<_> = self.held.drain().collect();
        held.sort_by_key(|h| h.0);
        for (_, recs) in held {
            for r in recs {
                self.emit(T_CHARGES, &r);
            }
        }
        if self.held_peak > 0 {
            eprintln!(
                "raw trace: at most {} charge records held for their owner",
                self.held_peak
            );
        }
        let mut err = self.error.take();
        if let Err(e) = self.strings.flush() {
            err.get_or_insert(e);
        }
        for f in self.files.iter_mut() {
            if let Some(f) = f.take() {
                if let Err(e) = f.finish() {
                    err.get_or_insert(e);
                }
            }
        }
        let mut out = self.out.borrow_mut();
        out.counts = self.counts;
        out.error = err.map(|e| e.to_string());
    }
}

// ---- the manifest ----

/// Ids of the unlimited links, after the real resources, in a fixed
/// order: attachment ports, shared buses, link directions.
fn unlimited_links(sim: &Sim) -> HashMap<Link, u16> {
    let topo = sim.topology();
    let mut next = sim.resources().len() as u16;
    let mut m = HashMap::new();
    let mut add = |l: Link| {
        m.insert(l, next);
        next += 1;
    };
    for a in 0..topo.attachment_count() {
        let id = AttachId(a as u32);
        let att = topo.attachment(id);
        if topo.bus(att.bus).topology == BusTopology::Xbar {
            if att.ingress.is_none() {
                add(Link::In(id));
            }
            if att.egress.is_none() {
                add(Link::Out(id));
            }
        }
    }
    for b in 0..topo.bus_count() {
        let bus = topo.bus(sim_core::BusId(b as u32));
        if bus.topology == BusTopology::Shared && bus.lane.is_none() {
            add(Link::Bus(sim_core::BusId(b as u32)));
        }
    }
    for (i, br) in topo.fabric_links().iter().enumerate() {
        if br.ab.is_none() {
            add(Link::Fabric(i as u32, true));
        }
        if br.ba.is_none() {
            add(Link::Fabric(i as u32, false));
        }
    }
    m
}

/// The owner node of a dotted name: the longest node path it starts with.
fn owner_of(name: &str, nodes: &[String]) -> Option<usize> {
    (0..nodes.len())
        .filter(|&i| name == nodes[i] || name.starts_with(&format!("{}.", nodes[i])))
        .max_by_key(|&i| nodes[i].len())
}

fn ratio(r: sim_core::Ratio) -> Value {
    json!([r.num, r.den])
}

fn bw(r: Option<sim_core::Ratio>) -> Value {
    r.map_or(Value::Null, |r| json!(r.num as f64 / r.den as f64))
}

fn hex(v: u64) -> String {
    format!("{v:#x}")
}

/// The git commit at build time ("-dirty" with uncommitted changes).
fn sim_commit() -> &'static str {
    option_env!("R2SIM_COMMIT").unwrap_or("unknown")
}

/// Now in RFC 3339, UTC.
fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

fn manifest(b: &Built, run: &RunInfo, unlimited: &HashMap<Link, u16>, f: &Filter) -> Value {
    let sim = &b.sim;
    let topo = sim.topology();
    let res = sim.resources();
    let lay = &b.layout;
    let nodes: Vec<String> = (0..topo.len())
        .map(|i| topo.path_name(NeutrinoId(i as u32)))
        .collect();
    let lane_id = |lane: Option<ResourceId>, link: Link| -> Value {
        match lane {
            Some(r) => json!(r.0),
            None => unlimited.get(&link).map_or(Value::Null, |&i| json!(i)),
        }
    };

    let node_rows: Vec<Value> = (0..topo.len())
        .map(|i| {
            let n = topo.node(NeutrinoId(i as u32));
            let c = sim.neutrino_config(NeutrinoId(i as u32));
            let t = &c.tables;
            json!({
                "id": i, "path": nodes[i], "parent": n.parent.map(|p| p.0),
                "neutrino": {
                    "scoreboard": t.scoreboard_capacity,
                    "pending": t.pending_capacity,
                    "execution": t.execution_capacity,
                    "versions": t.version_capacity,
                    "inbound_refs": t.inbound_ref_capacity,
                    "inbound_invokes": c.inbound_invokes,
                    "outbound_deps": c.outbound_deps,
                    "outbound_invokes": c.outbound_invokes,
                },
            })
        })
        .collect();

    let mut pe_paths = vec![String::new(); sim.pe_count()];
    for (path, id) in b.pe_paths() {
        pe_paths[id.0 as usize] = path;
    }
    let mut role: Vec<Option<Value>> = vec![None; res.len()];
    let mut set_role = |r: ResourceId, v: Value| {
        let slot = &mut role[r.0 as usize];
        if slot.is_none() {
            *slot = Some(v);
        }
    };
    for (i, m) in lay.memories.iter().enumerate() {
        if let Some(p) = m.port {
            set_role(p, json!({"type": "mem_port", "memory": i}));
        }
        if let Some(w) = m.wport {
            set_role(w, json!({"type": "mem_port", "memory": i, "write": true}));
        }
    }
    let pe_rows: Vec<Value> = (0..sim.pe_count())
        .map(|i| {
            let id = sim_core::PeId(i as u32);
            let cfg = sim.pe_config(id);
            let (kind, own): (&str, Vec<ResourceId>) = match &cfg.params {
                PeParams::Atlas { array, .. } => ("tensor", vec![*array]),
                PeParams::Muon { vector, .. } => ("vector", vec![*vector]),
                PeParams::Dma { links, .. } => ("dma", links.clone()),
            };
            for r in own {
                set_role(r, json!({"type": "pe", "pe": i}));
            }
            json!({"id": i, "path": pe_paths[i], "node": sim.pe_neutrino(id).0, "type": kind})
        })
        .collect();
    for i in 0..topo.len() {
        let (s, c) = sim.neutrino_resources(NeutrinoId(i as u32));
        set_role(s, json!({"type": "scalar", "node": i}));
        set_role(c, json!({"type": "complete", "node": i}));
    }

    let mut bus_rate: Vec<Option<sim_core::Ratio>> = vec![None; topo.bus_count()];
    // The write channel rate of each bus (version 1.5).
    let mut bus_wr: Vec<Option<sim_core::Ratio>> = vec![None; topo.bus_count()];
    let att_rows: Vec<Value> = (0..topo.attachment_count())
        .map(|a| {
            let id = AttachId(a as u32);
            let att = topo.attachment(id);
            let bus = topo.bus(att.bus);
            let xbar = bus.topology == BusTopology::Xbar;
            for (lane, dir) in [(att.ingress, "in"), (att.egress, "out")] {
                if let Some(r) = lane {
                    set_role(r, json!({"type": "port", "attachment": a, "dir": dir}));
                    bus_rate[att.bus.0 as usize] = res.flow_rate(r);
                }
            }
            // The write channel (version 1.5).
            for (lane, dir) in [(att.ingress_wr, "in"), (att.egress_wr, "out")] {
                if let Some(r) = lane {
                    set_role(
                        r,
                        json!({"type": "port", "attachment": a, "dir": dir, "write": true}),
                    );
                    bus_wr[att.bus.0 as usize] = res.flow_rate(r);
                }
            }
            let info = lay.attachments.get(a);
            let short = att
                .name
                .strip_prefix(&format!("{}.", bus.name))
                .unwrap_or(&att.name);
            let (inp, out) = if xbar {
                (
                    lane_id(att.ingress, Link::In(id)),
                    lane_id(att.egress, Link::Out(id)),
                )
            } else {
                (Value::Null, Value::Null)
            };
            let mut row = json!({
                "id": a, "bus": att.bus.0, "name": short,
                "component": info.map(|i| i.component.clone()),
                "component_type": info.map(|i| match i.kind {
                    Component::Memory => "memory",
                    Component::Neutrino => "neutrino",
                    Component::Link => "link",
                    Component::Compute => "compute",
                }),
                "role": info.map(|i| match i.role {
                    Role::Master => "master",
                    Role::Slave => "slave",
                    Role::Both => "both",
                }),
                "in": inp, "out": out,
            });
            if att.ingress_wr.is_some() || att.egress_wr.is_some() {
                row["in_wr"] = json!(att.ingress_wr.map(|r| r.0));
                row["out_wr"] = json!(att.egress_wr.map(|r| r.0));
            }
            row
        })
        .collect();
    let bus_rows: Vec<Value> = (0..topo.bus_count())
        .map(|i| {
            let id = sim_core::BusId(i as u32);
            let bus = topo.bus(id);
            if let Some(r) = bus.lane {
                set_role(r, json!({"type": "bus", "bus": i}));
                bus_rate[i] = res.flow_rate(r);
            }
            if let Some(r) = bus.lane_wr {
                set_role(r, json!({"type": "bus", "bus": i, "write": true}));
                bus_wr[i] = res.flow_rate(r);
            }
            let mut row = json!({
                "id": i, "path": bus.name, "node": owner_of(&bus.name, &nodes),
                "topology": if bus.topology == BusTopology::Xbar { "xbar" } else { "bus" },
                "control": bus.control, "bw_bytes": bw(bus_rate[i]), "latency": bus.latency,
            });
            if bus_wr[i].is_some() {
                row["wr_bytes"] = bw(bus_wr[i]);
            }
            row
        })
        .collect();
    let link_rows: Vec<Value> = topo
        .fabric_links()
        .iter()
        .enumerate()
        .map(|(i, l)| {
            for (lane, dir) in [(l.ab, "ab"), (l.ba, "ba")] {
                if let Some(r) = lane {
                    set_role(r, json!({"type": "link", "link": i, "dir": dir}));
                }
            }
            for (lane, dir) in [(l.ab_wr, "ab"), (l.ba_wr, "ba")] {
                if let Some(r) = lane {
                    set_role(
                        r,
                        json!({"type": "link", "link": i, "dir": dir, "write": true}),
                    );
                }
            }
            let info = lay.links.get(i);
            let mut row = json!({
                "id": i, "path": info.map(|x| x.path.clone()),
                "node": info.map(|x| x.node.0),
                "control": info.map(|x| x.control),
                "from_bus": l.a.0, "to_bus": l.b.0,
                "from_attachment": l.att_a.0, "to_attachment": l.att_b.0,
                "ab": lane_id(l.ab, Link::Fabric(i as u32, true)),
                "ba": lane_id(l.ba, Link::Fabric(i as u32, false)),
                "bw_bytes": bw(info.and_then(|x| x.rate)),
                "beat_bytes": info.map(|x| x.beat),
                "latency": l.latency,
                "cache": info.and_then(|x| x.cache),
            });
            // The write channel (version 1.5).
            if let Some(w) = l.ab_wr {
                row["wr_bytes"] = bw(res.flow_rate(w));
                row["ab_wr"] = json!(w.0);
                row["ba_wr"] = json!(l.ba_wr.map(|r| r.0));
            }
            // A hash link's address rewrite (architecture Section 3.1).
            if let Some(h) = info.and_then(|x| x.hash.as_ref()) {
                row["hash"] = json!({
                    "range": [hex(h.range.base), hex(h.range.bound)],
                    "to_range": [hex(h.to_range.base), hex(h.to_range.bound)],
                    "index": h.index.iter().map(|m| hex(*m)).collect::<Vec<_>>(),
                    "drop": h.drop,
                    "granule": h.granule(),
                    "slice_bytes": h.slice_bytes(),
                });
            }
            row
        })
        .collect();
    let mem_rows: Vec<Value> = lay
        .memories
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let ranges: Vec<Value> = m
                .ranges
                .iter()
                .map(|r| json!([hex(r.base), hex(r.bound)]))
                .collect();
            let mut row = json!({
                "id": i, "path": m.path, "node": m.node.0, "type": m.kind, "ranges": ranges,
                "attachments": m.attachments.iter().map(|a| a.0).collect::<Vec<_>>(),
                "port": m.port.map(|p| p.0),
                "bw_bytes": bw(m.port.and_then(|p| res.flow_rate(p))),
                "latency": m.latency,
            });
            if let Some(w) = m.wport {
                row["wport"] = json!(w.0);
                row["wr_bytes"] = bw(res.flow_rate(w));
            }
            row
        })
        .collect();

    let mut res_rows: Vec<Value> = (0..res.len())
        .map(|i| {
            let id = ResourceId(i as u32);
            let r = res.get(id);
            json!({
                "id": i, "name": r.name, "kind": format!("{:?}", r.kind),
                // Units per core cycle; slots per core cycle (1.7).
                "rate": res.flow_rate(id).map_or(Value::Null, |_| ratio(res.flow(id).core_rate())),
                "capacity": res.flow_rate(id).map_or(Value::Null, |_| ratio(res.flow(id).capacity())),
                "clock": res.flow_rate(id).map_or(json!([1, 1]), |_| ratio(res.flow(id).clock())),
                "beat_bytes": res.flow_rate(id).and_then(|_| res.flow(id).beat()),
                "owner": owner_of(&r.name, &nodes),
                "role": role[i].clone().unwrap_or(json!({"type": "other"})),
            })
        })
        .collect();
    // Rate classes (1.7): an Atlas array's rate per operand type, per core
    // cycle, in the order its charges' `sub` indexes.
    for i in 0..b.sim.pe_count() {
        if let sim_core::PeParams::Atlas { array, rates, .. } =
            &b.sim.pe_config(sim_core::PeId(i as u32)).params
        {
            let clock = res.flow(*array).clock();
            let classes: Vec<Value> = rates
                .iter()
                .map(|(d, r)| {
                    let name = match d {
                        sim_core::Dtype::Bf16 => "bf16",
                        sim_core::Dtype::F32 => "f32",
                        sim_core::Dtype::Fp8 => "fp8",
                        sim_core::Dtype::Mxfp4 => "mxfp4",
                    };
                    let core = sim_core::Ratio::new(r.num * clock.num, r.den * clock.den);
                    json!({"name": name, "rate": ratio(core)})
                })
                .collect();
            if !classes.is_empty() {
                res_rows[array.0 as usize]["rates"] = Value::Array(classes);
            }
        }
    }
    let mut links: Vec<(&Link, &u16)> = unlimited.iter().collect();
    links.sort_by_key(|l| *l.1);
    for (l, &id) in links {
        let (name, role) = match *l {
            Link::In(a) => (
                format!("{}.in", topo.attachment(a).name),
                json!({"type": "port", "attachment": a.0, "dir": "in"}),
            ),
            Link::Out(a) => (
                format!("{}.out", topo.attachment(a).name),
                json!({"type": "port", "attachment": a.0, "dir": "out"}),
            ),
            Link::Bus(b) => (topo.bus(b).name.clone(), json!({"type": "bus", "bus": b.0})),
            Link::Fabric(i, ab) => {
                let path = lay
                    .links
                    .get(i as usize)
                    .map_or(format!("link{i}"), |l| l.path.clone());
                let dir = if ab { "ab" } else { "ba" };
                (
                    format!("{path}.{dir}"),
                    json!({"type": "link", "link": i, "dir": dir}),
                )
            }
        };
        res_rows.push(json!({
            "id": id, "name": name, "kind": "Bandwidth", "rate": null,
            "capacity": null, "clock": [1, 1], "beat_bytes": null,
            "owner": owner_of(&name, &nodes), "role": role,
        }));
    }

    let records: serde_json::Map<String, Value> = RECORDS
        .iter()
        .map(|(stem, layout, _)| {
            let l: Vec<Value> = layout.iter().map(|(n, t)| json!([n, t])).collect();
            (stem.to_string(), Value::Array(l))
        })
        .collect();
    let kinds: Vec<&str> = (0..nu_prog::label::KINDS)
        .map(nu_prog::label::kind_name)
        .collect();
    json!({
        "format": "r2raw",
        "version": [1, 9],
        "run": {
            "config": run.config, "program": run.program,
            "params": {"file": run.params_file, "overrides": run.params},
            "overrides": run.overrides,
            "sim_commit": sim_commit(), "clock_ghz": b.clock_ghz, "granule": run.granule,
            "created": now_utc(),
        },
        "filters": filters_json(f),
        "records": records,
        "nodes": node_rows, "pes": pe_rows, "buses": bus_rows, "attachments": att_rows,
        "links": link_rows, "memories": mem_rows, "resources": res_rows,
        "opcodes": sim.profiles().names().collect::<Vec<_>>(),
        "message_kinds": MESSAGE_KINDS,
        "label": {
            "fields": [
                {"name": "kind", "shift": 56, "bits": 8, "enum": kinds},
                {"name": "step", "shift": 48, "bits": 8},
                {"name": "layer", "shift": 40, "bits": 8},
                {"name": "unit", "shift": 32, "bits": 8},
                {"name": "idx", "shift": 0, "bits": 32},
            ],
            "format": "{kind} s{step} l{layer} u{unit} #{idx}",
        },
    })
}

/// The `filters` object of the manifest: `records` lists the files
/// written, `types` the record types kept (`charges` and `messages` share
/// `charges.bin`).
fn filters_json(f: &Filter) -> Value {
    let records = f.records.as_ref().map(|_| {
        RECORDS
            .iter()
            .map(|r| r.0)
            .filter(|stem| f.keeps(stem) || (*stem == "charges" && f.keeps("messages")))
            .collect::<Vec<_>>()
    });
    json!({
        "nodes": f.nodes,
        "records": records,
        "window": f.window.map(|(lo, hi)| [lo, hi]),
        "types": f.records.as_ref().map(|_| {
            RECORD_TYPES.iter().filter(|k| f.keeps(k)).collect::<Vec<_>>()
        }),
        "partial": f.partial(),
    })
}

/// One phase row of `summary.json`.
pub struct PhaseRow<'a> {
    pub stage: &'a str,
    pub name: &'a str,
    pub layer: u32,
    pub start: u64,
    pub end: u64,
    pub estimate: Option<f64>,
    pub roofline: Option<f64>,
    pub bound_by: &'a str,
}

/// Writes `summary.json` after the run (also after a fatal error).
pub fn write_summary(
    dir: &Path,
    stats: &Stats,
    wall: f64,
    out: &Outcome,
    phases: &[PhaseRow],
) -> Result<(), String> {
    let mut counts = serde_json::Map::new();
    for (i, (stem, _, _)) in RECORDS.iter().enumerate() {
        counts.insert(stem.to_string(), json!(out.counts[i]));
    }
    let phases: Vec<Value> = phases
        .iter()
        .map(|p| {
            json!({
                "stage": p.stage, "name": p.name, "layer": p.layer, "start": p.start,
                "end": p.end, "estimate": p.estimate, "roofline": p.roofline,
                "bound_by": p.bound_by,
            })
        })
        .collect();
    let v = json!({
        "end_time": stats.end_time, "events": stats.events, "wall_seconds": wall,
        "counts": counts,
        "fatal": serde_json::to_value(&stats.fatal).map_err(|e| e.to_string())?,
        "write_error": out.error,
        "phases": phases,
    });
    let text = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("summary.json"), text).map_err(|e| format!("{}: {e}", dir.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_fill_their_record_sizes() {
        let size = |t: &str| match t {
            "u8" => 1,
            "u16" => 2,
            "u32" => 4,
            "u64" => 8,
            _ => panic!("{t}"),
        };
        for (stem, layout, n) in RECORDS {
            let sum: usize = layout.iter().map(|f| size(f.1)).sum();
            assert_eq!(sum, n, "{stem}");
        }
    }

    #[test]
    fn rfc3339_dates() {
        let s = now_utc();
        assert_eq!(s.len(), 20);
        assert!(s.ends_with('Z') && s.as_bytes()[10] == b'T');
    }
}
