//! The backends of the functional plane and their dispatch.
//!
//! Each FU kind (`dma`, `atlas`, `muon`) has a list of backends; every
//! backend works on one [`Store`]. At the issue of a task the timing thread
//! selects its backend ([`Dispatch`]): the first backend of the list of
//! the task's FU kind. The op carries the index; the worker applies it
//! with that backend ([`Functional`]). A Neutrino store writes the store
//! directly.
//!
//! The dispatch detects backend switches:
//! * a matmul of an open chain (an Atlas PE and a C base, opened by a
//!   matmul with `keep`) on another backend than the one that holds its
//!   partial sums stops the run (only a selection rule of a test,
//!   [`SelectFn`], makes this switch);
//! * a task on a backend that is not `null` that reads a range that a
//!   `null` task wrote last gets a warning ([`NullRead`]), once per range
//!   of a `null` task (lists such as `atlas: [null]` with `dma: [builtin]`
//!   make this switch); a later write by such a task removes the mark.

use std::collections::{BTreeMap, HashMap};

use nu_model::{Addr, NeutrinoId, Range};

use crate::func::{FuncError, FuncOp, FunctionalBackend, NullBackend, OpKind};
use crate::profile::{split_row, Args, Descriptor, Dtype, PeId};
use crate::store::Store;
use crate::time::Cycle;

/// The FU kinds of the functional plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuKind {
    /// The `dma` opcode (DMA engines).
    Dma,
    /// The `atlas.matmul` opcode (Atlas).
    Atlas,
    /// The `muon.vector` opcode (Muon).
    Muon,
}

impl FuKind {
    /// The three kinds, in the order of [`Units`].
    pub const ALL: [FuKind; 3] = [FuKind::Dma, FuKind::Atlas, FuKind::Muon];

    /// The kind's key in the configuration.
    pub fn key(self) -> &'static str {
        match self {
            FuKind::Dma => "dma",
            FuKind::Atlas => "atlas",
            FuKind::Muon => "muon",
        }
    }

    /// The FU kind of a task; `None` for a task without data (`join`,
    /// `spawn`, `imem_copy`).
    pub fn of(args: &Args) -> Option<FuKind> {
        match args {
            Args::Dma { .. } => Some(FuKind::Dma),
            Args::Matmul { .. } => Some(FuKind::Atlas),
            Args::Vector { .. } => Some(FuKind::Muon),
            Args::Join | Args::Spawn { .. } | Args::ImemCopy { .. } => None,
        }
    }
}

/// Per FU kind, the backends that the run can select: indices into
/// [`Backends::list`]. The first of each list is the one the fixed rule
/// selects.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Units {
    pub dma: Vec<usize>,
    pub atlas: Vec<usize>,
    pub muon: Vec<usize>,
}

impl Units {
    /// The list of `kind`.
    pub fn of(&self, kind: FuKind) -> &[usize] {
        match kind {
            FuKind::Dma => &self.dma,
            FuKind::Atlas => &self.atlas,
            FuKind::Muon => &self.muon,
        }
    }
}

/// The backends of a run and the lists of the FU kinds.
pub struct Backends {
    /// The backend instances; one instance per name.
    pub list: Vec<Box<dyn FunctionalBackend>>,
    /// The lists of the FU kinds.
    pub units: Units,
}

impl Backends {
    /// `backend` for every FU kind.
    pub fn one(backend: Box<dyn FunctionalBackend>) -> Self {
        let units = Units {
            dma: vec![0],
            atlas: vec![0],
            muon: vec![0],
        };
        Self {
            list: vec![backend],
            units,
        }
    }

    /// The backends of the lists `names` (dma, atlas, muon): one instance
    /// per distinct name, made by `make`. An error names the key of the
    /// list (`functional.units.<kind>`) and the backend.
    pub fn from_names(
        names: [&[String]; 3],
        mut make: impl FnMut(&str) -> Result<Box<dyn FunctionalBackend>, String>,
    ) -> Result<Self, String> {
        let mut list: Vec<Box<dyn FunctionalBackend>> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut lists: [Vec<usize>; 3] = Default::default();
        for (k, kind) in FuKind::ALL.iter().enumerate() {
            if names[k].is_empty() {
                return Err(format!("functional.units.{}: an empty list", kind.key()));
            }
            for n in names[k] {
                let i = match index.get(n) {
                    Some(&i) => i,
                    None => {
                        let b =
                            make(n).map_err(|e| format!("functional.units.{}: {e}", kind.key()))?;
                        list.push(b);
                        index.insert(n.clone(), list.len() - 1);
                        list.len() - 1
                    }
                };
                lists[k].push(i);
            }
        }
        let [dma, atlas, muon] = lists;
        Ok(Self {
            list,
            units: Units { dma, atlas, muon },
        })
    }

    /// The index of a `null` backend, added when the lists have none.
    fn null_index(&mut self) -> usize {
        match self.list.iter().position(|b| b.name() == "null") {
            Some(i) => i,
            None => {
                self.list.push(Box::new(NullBackend));
                self.list.len() - 1
            }
        }
    }
}

/// A read by a task on a computing backend of a range that a `null` task
/// wrote last (its values are not computed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NullRead {
    /// The cycle of the reading task's issue.
    pub time: Cycle,
    /// The Neutrino of the reading task.
    pub nu: NeutrinoId,
    /// The gid of the reading task.
    pub reader: Option<u64>,
    /// The gid of the `null` task.
    pub writer: Option<u64>,
    /// The destination range of the `null` task.
    pub range: Range,
}

/// The output of a `null` task, by the granules it wrote last.
#[derive(Default)]
struct NullMarks {
    /// Granule index → the seq of the `null` task that wrote it last.
    granules: BTreeMap<u64, u64>,
    /// The `null` tasks by seq: (gid, dst span, warned).
    writers: HashMap<u64, (Option<u64>, Range, bool)>,
    warnings: Vec<NullRead>,
}

/// Where the functional plane keeps its memory.
enum Memory {
    /// No backend of the run keeps data (every list is `null`).
    None,
    Store(Store),
}

/// The worker side of the functional plane: the memory, the backends, and
/// the marks of the output of `null` tasks. After the run, it serves the
/// reads of `--image-out` and of tests.
pub struct Functional {
    memory: Memory,
    backends: Vec<Box<dyn FunctionalBackend>>,
    names: Vec<&'static str>,
    null: usize,
    marks: NullMarks,
}

impl Drop for Functional {
    /// Drops the backends before the store: a plug-in's `close` may read
    /// or write the store's blocks, which the store unmaps when it drops.
    fn drop(&mut self) {
        self.backends.clear();
    }
}

impl Functional {
    /// A plane of `backends` over `store` (or no memory, when no backend
    /// keeps data); `null`: the index of the `null` backend.
    pub(crate) fn new(
        backends: Vec<Box<dyn FunctionalBackend>>,
        null: usize,
        store: Option<Store>,
    ) -> Self {
        let names = backends.iter().map(|b| b.name()).collect();
        Self {
            memory: store.map_or(Memory::None, Memory::Store),
            backends,
            names,
            null,
            marks: NullMarks::default(),
        }
    }

    /// True when the plane keeps data: a worker thread applies its ops.
    pub(crate) fn has_data(&self) -> bool {
        !matches!(self.memory, Memory::None)
    }

    /// The store, when the plane has one.
    pub fn store(&self) -> Option<&Store> {
        match &self.memory {
            Memory::Store(s) => Some(s),
            _ => None,
        }
    }

    /// Reads `len` bytes at `addr`.
    pub fn read(&mut self, addr: Addr, len: u64) -> Result<Vec<u8>, FuncError> {
        match &mut self.memory {
            Memory::None => Err(FuncError("the Null backend has no data".into())),
            Memory::Store(s) => s.read(addr.0, len),
        }
    }

    /// Writes `bytes` at `addr` (the memory image, before the run).
    pub fn write(&mut self, addr: Addr, bytes: &[u8]) -> Result<(), FuncError> {
        match &mut self.memory {
            Memory::None => Err(FuncError("the Null backend has no data".into())),
            Memory::Store(s) => s.write(addr.0, bytes),
        }
    }

    /// Configures the backends before the run.
    pub(crate) fn configure(&mut self) -> Result<(), FuncError> {
        match &mut self.memory {
            Memory::Store(s) => {
                for b in &mut self.backends {
                    b.configure(s)?;
                }
                Ok(())
            }
            Memory::None => Ok(()),
        }
    }

    /// The name of the backend that applies `op` (`store` for a Neutrino
    /// store).
    pub(crate) fn backend_name(&self, op: &FuncOp) -> &'static str {
        match &op.kind {
            OpKind::Store { .. } => "store",
            OpKind::Task(_) => self.names.get(op.backend).copied().unwrap_or("null"),
        }
    }

    /// The reads of `null` output, in the order found.
    pub(crate) fn take_warnings(&mut self) -> Vec<NullRead> {
        std::mem::take(&mut self.marks.warnings)
    }

    /// Applies `op` with its backend.
    pub(crate) fn apply(&mut self, op: &FuncOp) -> Result<(), FuncError> {
        let store = match &mut self.memory {
            Memory::None => return Ok(()),
            Memory::Store(s) => s,
        };
        let desc = match &op.kind {
            OpKind::Store { addr, bytes } => {
                unmark(
                    &mut self.marks,
                    store.granule(),
                    &Range::from_len(addr.0, bytes.len() as u64),
                    None,
                );
                return store.write(addr.0, bytes);
            }
            OpKind::Task(d) => d,
        };
        let g = store.granule();
        let null = op.backend == self.null;
        if FuKind::of(&desc.args).is_some() {
            if null {
                mark(&mut self.marks, store, op, desc);
            } else if !self.marks.granules.is_empty() {
                for (addr, len) in reads(desc, op.tile) {
                    check(&mut self.marks, g, op, addr, len);
                }
                unmark(&mut self.marks, g, &desc.dst, Some(desc));
            }
        }
        self.backends[op.backend].apply(op, store)
    }
}

/// Records that the `null` task `op` wrote the granules of its `dst` (a
/// `dst` outside one block of the store: none).
fn mark(m: &mut NullMarks, store: &Store, op: &FuncOp, desc: &Descriptor) {
    let g = store.granule();
    let dst = desc.held();
    if !store.in_block(dst.span.base, dst.span.len()) {
        return;
    }
    for i in dst.span.base / g..dst.span.bound.div_ceil(g) {
        if dst.holds(g, i * g) {
            m.granules.insert(i, op.seq);
        }
    }
    m.writers.insert(op.seq, (op.gid, dst.span, false));
}

/// Removes the marks of the granules that a write of `range` covers (all
/// of them for a store; those its masked `dst` holds for a task).
fn unmark(m: &mut NullMarks, g: u64, range: &Range, desc: Option<&Descriptor>) {
    if m.granules.is_empty() || range.is_empty() {
        return;
    }
    let (lo, hi) = (range.base / g, range.bound.div_ceil(g));
    let held: Vec<u64> = m
        .granules
        .range(lo..hi)
        .map(|(&i, _)| i)
        .filter(|&i| desc.is_none_or(|d| d.held().holds(g, i * g)))
        .collect();
    for i in held {
        m.granules.remove(&i);
    }
}

/// Warns once per `null` task when `op` reads `[addr, addr + len)` and a
/// granule of it was last written by that task.
fn check(m: &mut NullMarks, g: u64, op: &FuncOp, addr: u64, len: u64) {
    if len == 0 {
        return;
    }
    let (lo, hi) = (addr / g, (addr + len).div_ceil(g));
    let seqs: Vec<u64> = m.granules.range(lo..hi).map(|(_, &s)| s).collect();
    for s in seqs {
        if let Some((writer, range, warned)) = m.writers.get_mut(&s) {
            if !*warned {
                *warned = true;
                m.warnings.push(NullRead {
                    time: op.time,
                    nu: op.nu,
                    reader: op.gid,
                    writer: *writer,
                    range: *range,
                });
            }
        }
    }
}

/// The byte ranges that a task reads: the source rows of a DMA; A, B (by
/// its stored shape, with the halves of split operands by the Atlas tile
/// `tile`), and C when it accumulates, of a matmul (row by row); the read
/// entries of a Muon task's kernel (the registry's `time`).
fn reads(desc: &Descriptor, tile: [u64; 2]) -> Vec<(u64, u64)> {
    match desc.args {
        Args::Dma {
            src, bytes, rows, ..
        } => match rows {
            None => vec![(src.0, bytes)],
            Some(r) => (0..r.count)
                .map(|i| (src.0 + i * r.src_stride, r.bytes))
                .collect(),
        },
        Args::Matmul {
            a,
            b,
            a2,
            b2,
            lda,
            ldb,
            ldc,
            m,
            n,
            k,
            in_dtype,
            b_dtype,
            out_dtype,
            accumulate,
            transpose_b,
            ..
        } => {
            let row = |dt: Dtype, n: u64| dt.bytes_for(n);
            let mut out = Vec::new();
            for r in 0..m {
                let (base, i) = match a2 {
                    Some(a2) if split_row(r, tile[0]).0 => (a2.0, split_row(r, tile[0]).1),
                    Some(_) => (a.0, split_row(r, tile[0]).1),
                    None => (a.0, r),
                };
                out.push((base + row(in_dtype, i * lda), row(in_dtype, k)));
            }
            // B: k rows; with b2, each half has the columns that
            // `split_row` gives it.
            // With transpose_b, B is stored as n rows of k elements.
            let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
            let halves: Vec<(u64, u64)> = match b2 {
                None => vec![(b.0, b_cols)],
                Some(b2) => {
                    let width = |hi: bool| {
                        (0..n)
                            .filter(|&c| split_row(c, tile[1]).0 == hi)
                            .map(|c| split_row(c, tile[1]).1 + 1)
                            .max()
                            .unwrap_or(0)
                    };
                    vec![(b.0, width(false)), (b2.0, width(true))]
                }
            };
            for (base, cols) in halves {
                for p in 0..b_rows {
                    out.push((base + row(b_dtype, p * ldb), row(b_dtype, cols)));
                }
            }
            if accumulate {
                for r in 0..m {
                    out.push((desc.dst.base + row(out_dtype, r * ldc), row(out_dtype, n)));
                }
            }
            out
        }
        Args::Vector { .. } => crate::profile::vector_traffic(desc)
            .map(|t| t.reads)
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// The timing thread's side of the dispatch: the backend lists, the index
/// of `null` and the open matmul chains.
pub(crate) struct Dispatch {
    units: Units,
    null: usize,
    names: Vec<&'static str>,
    /// Open chains: (Atlas PE, C base) → the backend that holds them.
    chains: HashMap<(Option<PeId>, u64), usize>,
    /// A selection that replaces the fixed rule (tests).
    select: Option<SelectFn>,
}

/// A selection rule for tests: the task and the index the fixed rule
/// selects; the index to use.
pub type SelectFn = Box<dyn Fn(&Descriptor, usize) -> usize>;

impl Dispatch {
    /// The dispatch of `backends`, and the worker's plane over `store`;
    /// `None` for `store` when no backend keeps data.
    pub(crate) fn new(
        mut backends: Backends,
        select: Option<SelectFn>,
    ) -> (Self, Vec<Box<dyn FunctionalBackend>>, usize) {
        let null = backends.null_index();
        let names = backends.list.iter().map(|b| b.name()).collect();
        let d = Self {
            units: backends.units,
            null,
            names,
            chains: HashMap::new(),
            select,
        };
        (d, backends.list, null)
    }

    /// The backend of a task. An error when a matmul of an open chain goes
    /// to another backend than the one that holds the chain.
    pub(crate) fn select(&mut self, desc: &Descriptor) -> Result<usize, String> {
        let Some(kind) = FuKind::of(&desc.args) else {
            return Ok(self.null);
        };
        if self.names.is_empty() {
            return Ok(0);
        }
        let mut b = self.units.of(kind)[0];
        if let Some(f) = &self.select {
            b = f(desc, b);
        }
        if let Args::Matmul { keep, .. } = desc.args {
            let key = (desc.pe, desc.dst.base);
            if let Some(&held) = self.chains.get(&key) {
                if held != b {
                    return Err(format!(
                        "a matmul of the chain on C {:#x} goes to backend {}, but backend {} holds its partial sums",
                        desc.dst.base, self.names[b], self.names[held]
                    ));
                }
            }
            if keep {
                self.chains.insert(key, b);
            } else {
                self.chains.remove(&key);
            }
        }
        Ok(b)
    }
}
