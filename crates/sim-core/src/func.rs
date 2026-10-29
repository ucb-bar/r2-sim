//! Functional plane interface and worker protocol (r2sim spec Sections 3.3
//! and 2.3).
//!
//! The timing thread emits one [`FuncOp`] per issued PE task and per
//! Neutrino store, in issue order, into a bounded op queue. One worker
//! thread applies them in order. A load is a sync point: it does not travel
//! through the op queue. The timing thread posts it in a separate slot with
//! the seq of the last writer of the range, and blocks. The worker checks
//! the slot between ops and serves the load as soon as it has applied that
//! seq, whether or not it has applied more. For a race-free program the
//! value is the same either way; the timing plane rejects racy loads
//! (r2sim spec Section 3.4).
//!
//! A backend without data (`Null`) runs without a worker thread; a load is
//! then an error.
//!
//! With the thread timeline on, both threads record their intervals in
//! wall-clock time ([`ThreadRecord`]): the timing thread its waits for the
//! worker, the worker each op and load and its waits for work. Each thread
//! keeps its own list; [`FuncPlane::finish`] merges them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use nu_model::{Addr, NeutrinoId, Range};

use crate::config::MemBlock;
use crate::profile::{Args, Descriptor};
use crate::stats::ThreadRecord;
use crate::time::Cycle;

/// What one functional op does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpKind {
    /// An issued PE task.
    Task(Descriptor),
    /// A Neutrino store to external memory.
    Store {
        /// The first byte address of the store.
        addr: Addr,
        /// The stored bytes.
        bytes: Vec<u8>,
    },
}

/// One functional op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncOp {
    /// 1, 2, 3, ... in emission order.
    pub seq: u64,
    /// The emission time in cycles.
    pub time: Cycle,
    /// The owner prefix: the Neutrino that emitted the op.
    pub nu: NeutrinoId,
    /// The task gid, for task ops.
    pub gid: Option<u64>,
    /// What the op does.
    pub kind: OpKind,
}

/// A backend error, as text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncError(pub String);

/// A functional backend. It runs on the worker thread.
pub trait FunctionalBackend: Send {
    /// Called once before the run with the memory blocks, the sync-only
    /// ranges and the Neutrino granule (the unit of a destination range's
    /// mask).
    fn configure(&mut self, memories: &[MemBlock], sync_only: &[Range], granule: u64);
    /// False for a backend that keeps no data; the simulator then runs it
    /// without a worker thread and a load is an error.
    fn has_data(&self) -> bool;
    /// Applies one op; ops arrive in `seq` order.
    fn apply(&mut self, op: &FuncOp) -> Result<(), FuncError>;
    /// Reads `len` bytes starting at `addr`.
    fn read(&mut self, addr: Addr, len: u64) -> Result<Vec<u8>, FuncError>;
    /// Host-side initialization before the run.
    fn write(&mut self, addr: Addr, bytes: &[u8]) -> Result<(), FuncError>;
    /// The time a child process spent on the last applied op, in ns by its
    /// own clock, for the thread timeline; `None` when the backend does
    /// not report it.
    fn child_ns(&self) -> Option<u64> {
        None
    }
}

/// Discards every op. A load is an error: the run declared no
/// data-dependent control.
#[derive(Debug, Default)]
pub struct NullBackend;

impl FunctionalBackend for NullBackend {
    fn configure(&mut self, _: &[MemBlock], _: &[Range], _: u64) {}

    fn has_data(&self) -> bool {
        false
    }

    fn apply(&mut self, _: &FuncOp) -> Result<(), FuncError> {
        Ok(())
    }

    fn read(&mut self, _: Addr, _: u64) -> Result<Vec<u8>, FuncError> {
        Err(FuncError("the Null backend has no data".into()))
    }

    fn write(&mut self, _: Addr, _: &[u8]) -> Result<(), FuncError> {
        Err(FuncError("the Null backend has no data".into()))
    }
}

/// The first error on the worker, with the op that caused it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkerError {
    pub seq: u64,
    pub time: Cycle,
    pub nu: NeutrinoId,
    pub gid: Option<u64>,
    pub error: FuncError,
}

/// The answer to a load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoadReply {
    pub bytes: Vec<u8>,
    /// The worker's applied seq when it served the load.
    pub applied: u64,
}

struct LoadReq {
    seq_needed: u64,
    addr: Addr,
    len: u64,
}

#[derive(Default)]
struct Chan {
    ops: VecDeque<FuncOp>,
    load: Option<LoadReq>,
    reply: Option<Result<LoadReply, FuncError>>,
    applied: u64,
    error: Option<WorkerError>,
    closed: bool,
}

pub(crate) struct Shared {
    chan: Mutex<Chan>,
    to_worker: Condvar,
    to_timing: Condvar,
    failed: AtomicBool,
}

/// The wall-clock records of one thread.
pub(crate) struct Timeline {
    /// The zero of the clock, shared by both threads.
    t0: Instant,
    records: Vec<ThreadRecord>,
}

impl Timeline {
    fn new(t0: Instant) -> Self {
        Self {
            t0,
            records: Vec::new(),
        }
    }

    /// Records `state` of `thread` from `from` to `to`, with the other
    /// fields of `r`.
    fn push(
        &mut self,
        thread: &'static str,
        state: &'static str,
        from: Instant,
        to: Instant,
        r: ThreadRecord,
    ) {
        let ns = |t: Instant| t.saturating_duration_since(self.t0).as_nanos() as u64;
        self.records.push(ThreadRecord {
            thread,
            state,
            start: ns(from),
            end: ns(to),
            ..r
        });
    }
}

/// The timing thread's timeline: its records, the start of its current
/// `run` interval and the seq of the last emitted op.
pub(crate) struct Control {
    timeline: Timeline,
    run_from: Instant,
    seq: u64,
}

impl Control {
    /// Ends the current `run` interval at `from` and records the wait
    /// `state` from `from` to now; `cycle` is the simulated cycle, `r` the
    /// other fields of the wait.
    fn wait(&mut self, state: &'static str, from: Instant, cycle: Cycle, r: ThreadRecord) {
        let to = Instant::now();
        let at = ThreadRecord {
            cycle: Some(cycle),
            seq: Some(self.seq),
            ..ThreadRecord::default()
        };
        let run_from = self.run_from;
        self.timeline
            .push("control", "run", run_from, from, at.clone());
        self.timeline.push(
            "control",
            state,
            from,
            to,
            ThreadRecord {
                cycle: at.cycle,
                seq: at.seq,
                ..r
            },
        );
        self.run_from = to;
    }
}

/// The timing thread's handle on the functional plane.
pub(crate) enum FuncPlane {
    /// A backend without data, on the timing thread.
    Inline(Box<dyn FunctionalBackend>),
    Worker {
        shared: Arc<Shared>,
        handle: JoinHandle<(Box<dyn FunctionalBackend>, Option<Timeline>)>,
        capacity: usize,
        /// The timing thread's timeline, when on.
        control: Option<Control>,
    },
}

/// Why the timing thread could not complete a functional call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlaneError {
    Worker(WorkerError),
    /// A load with a backend that has no data, or a backend read error.
    Load(FuncError),
}

impl FuncPlane {
    /// Starts the worker of a backend with data. `timeline`: both threads
    /// record their intervals (not for a backend without data).
    pub fn start(mut backend: Box<dyn FunctionalBackend>, capacity: usize, timeline: bool) -> Self {
        if !backend.has_data() {
            return FuncPlane::Inline(backend);
        }
        let t0 = Instant::now();
        let mut worker_timeline = timeline.then(|| Timeline::new(t0));
        let shared = Arc::new(Shared {
            chan: Mutex::new(Chan::default()),
            to_worker: Condvar::new(),
            to_timing: Condvar::new(),
            failed: AtomicBool::new(false),
        });
        let sh = shared.clone();
        let handle = std::thread::Builder::new()
            .name("r2-func".into())
            .spawn(move || {
                worker(&mut *backend, &sh, worker_timeline.as_mut());
                (backend, worker_timeline)
            })
            .expect("spawn the functional worker");
        FuncPlane::Worker {
            shared,
            handle,
            capacity,
            control: timeline.then(|| Control {
                timeline: Timeline::new(t0),
                run_from: t0,
                seq: 0,
            }),
        }
    }

    /// True when the worker has failed. Cheap; for periodic polling.
    pub fn failed(&self) -> bool {
        match self {
            FuncPlane::Inline(_) => false,
            FuncPlane::Worker { shared, .. } => shared.failed.load(Ordering::Relaxed),
        }
    }

    pub fn worker_error(&self) -> Option<WorkerError> {
        match self {
            FuncPlane::Inline(_) => None,
            FuncPlane::Worker { shared, .. } => lock(shared).error.clone(),
        }
    }

    /// Queues an op. Blocks while the queue is full, unless the worker has
    /// failed.
    pub fn emit(&mut self, op: FuncOp) -> Result<(), PlaneError> {
        match self {
            FuncPlane::Inline(b) => b.apply(&op).map_err(|error| {
                PlaneError::Worker(WorkerError {
                    seq: op.seq,
                    time: op.time,
                    nu: op.nu,
                    gid: op.gid,
                    error,
                })
            }),
            FuncPlane::Worker {
                shared,
                capacity,
                control,
                ..
            } => {
                let mut c = lock(shared);
                let full = c.error.is_none() && c.ops.len() >= *capacity;
                let from = (full && control.is_some()).then(Instant::now);
                while c.error.is_none() && c.ops.len() >= *capacity {
                    c = shared.to_timing.wait(c).expect("functional plane lock");
                }
                if let Some(e) = &c.error {
                    return Err(PlaneError::Worker(e.clone()));
                }
                if let Some(ctl) = control.as_mut() {
                    if let Some(from) = from {
                        ctl.wait("emit_wait", from, op.time, ThreadRecord::default());
                    }
                    ctl.seq = op.seq;
                }
                c.ops.push_back(op);
                shared.to_worker.notify_one();
                Ok(())
            }
        }
    }

    /// A sync point: blocks until the worker has applied `seq_needed`, then
    /// returns the bytes it read. `cycle`: the simulated cycle of the load.
    pub fn load(
        &mut self,
        seq_needed: u64,
        addr: Addr,
        len: u64,
        cycle: Cycle,
    ) -> Result<LoadReply, PlaneError> {
        match self {
            FuncPlane::Inline(b) => b
                .read(addr, len)
                .map(|bytes| LoadReply { bytes, applied: 0 })
                .map_err(PlaneError::Load),
            FuncPlane::Worker {
                shared, control, ..
            } => {
                let from = Instant::now();
                let mut c = lock(shared);
                if let Some(e) = &c.error {
                    return Err(PlaneError::Worker(e.clone()));
                }
                c.load = Some(LoadReq {
                    seq_needed,
                    addr,
                    len,
                });
                shared.to_worker.notify_one();
                let reply = loop {
                    if let Some(e) = &c.error {
                        return Err(PlaneError::Worker(e.clone()));
                    }
                    if let Some(r) = c.reply.take() {
                        break r;
                    }
                    c = shared.to_timing.wait(c).expect("functional plane lock");
                };
                drop(c);
                if let Some(ctl) = control.as_mut() {
                    let r = ThreadRecord {
                        addr: Some(addr.0),
                        len: Some(len),
                        seq_needed: Some(seq_needed),
                        ..ThreadRecord::default()
                    };
                    ctl.wait("load_wait", from, cycle, r);
                }
                reply.map_err(PlaneError::Load)
            }
        }
    }

    /// Stops the worker after it has applied every queued op (or at once
    /// after an error) and returns the backend. `Err` carries a worker
    /// panic message; the second item is the error of an op that failed
    /// after the timing plane last looked (an op after the last load).
    /// `cycle`: the simulated cycle at the end of the run. The last item
    /// is the thread timeline (empty when off), ordered by start.
    #[allow(clippy::type_complexity)]
    pub fn finish(
        self,
        cycle: Cycle,
    ) -> (
        Result<(), String>,
        Option<WorkerError>,
        Option<Box<dyn FunctionalBackend>>,
        Vec<ThreadRecord>,
    ) {
        match self {
            FuncPlane::Inline(b) => (Ok(()), None, Some(b), Vec::new()),
            FuncPlane::Worker {
                shared,
                handle,
                control,
                ..
            } => {
                let from = Instant::now();
                lock(&shared).closed = true;
                shared.to_worker.notify_one();
                let joined = handle.join();
                let error = lock(&shared).error.clone();
                let mut records = Vec::new();
                if let Some(mut ctl) = control {
                    ctl.wait("finish_wait", from, cycle, ThreadRecord::default());
                    records = ctl.timeline.records;
                }
                match joined {
                    Ok((b, worker)) => {
                        records.extend(worker.into_iter().flat_map(|w| w.records));
                        records.sort_by_key(|r| (r.start, r.end));
                        (Ok(()), error, Some(b), records)
                    }
                    Err(p) => {
                        let msg = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "unknown panic".into());
                        (Err(msg), error, None, records)
                    }
                }
            }
        }
    }
}

fn lock(shared: &Shared) -> std::sync::MutexGuard<'_, Chan> {
    shared.chan.lock().unwrap_or_else(|p| p.into_inner())
}

// A transient value of the worker loop: the size difference costs nothing.
#[allow(clippy::large_enum_variant)]
enum Work {
    Apply(FuncOp),
    Serve(LoadReq),
}

/// The kind and size fields of `op` for the thread timeline.
fn op_fields(op: &FuncOp) -> ThreadRecord {
    let mut r = ThreadRecord {
        seq: Some(op.seq),
        nu: Some(op.nu.0),
        ..ThreadRecord::default()
    };
    let (kind, bytes) = match &op.kind {
        OpKind::Store { bytes, .. } => ("store", Some(bytes.len() as u64)),
        OpKind::Task(d) => match &d.args {
            Args::Dma { bytes, .. } => ("dma", Some(*bytes)),
            Args::Matmul { m, n, k, .. } => {
                (r.m, r.n, r.k) = (Some(*m), Some(*n), Some(*k));
                ("matmul", None)
            }
            Args::Vector { .. } => ("vector", None),
            Args::Join => ("join", None),
            Args::Spawn { .. } => ("spawn", None),
            Args::ImemCopy { bytes, .. } => ("imem_copy", Some(*bytes)),
        },
    };
    r.op = Some(kind);
    r.bytes = bytes;
    r
}

fn worker(
    backend: &mut dyn FunctionalBackend,
    shared: &Shared,
    mut timeline: Option<&mut Timeline>,
) {
    // A panic in the backend must not leave the timing thread blocked.
    struct PanicGuard<'a>(&'a Shared);
    impl Drop for PanicGuard<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                let mut c = lock(self.0);
                if c.error.is_none() {
                    c.error = Some(WorkerError {
                        seq: c.applied + 1,
                        time: 0,
                        nu: NeutrinoId(0),
                        gid: None,
                        error: FuncError("functional worker panicked".into()),
                    });
                }
                self.0.failed.store(true, Ordering::Relaxed);
                self.0.to_timing.notify_all();
            }
        }
    }
    let _guard = PanicGuard(shared);

    loop {
        let mut idle_from = None;
        let work = {
            let mut c = lock(shared);
            loop {
                let serve = c
                    .load
                    .as_ref()
                    .is_some_and(|l| c.error.is_some() || c.applied >= l.seq_needed);
                if serve {
                    break Work::Serve(c.load.take().expect("load"));
                }
                if c.error.is_some() {
                    c.ops.clear();
                } else if let Some(op) = c.ops.pop_front() {
                    shared.to_timing.notify_one();
                    break Work::Apply(op);
                }
                if c.closed && c.ops.is_empty() && c.load.is_none() {
                    if let (Some(t), Some(from)) = (timeline.as_deref_mut(), idle_from) {
                        t.push(
                            "worker",
                            "wait",
                            from,
                            Instant::now(),
                            ThreadRecord::default(),
                        );
                    }
                    return;
                }
                if timeline.is_some() && idle_from.is_none() {
                    idle_from = Some(Instant::now());
                }
                c = shared.to_worker.wait(c).expect("functional plane lock");
            }
        };
        if let (Some(t), Some(from)) = (timeline.as_deref_mut(), idle_from) {
            t.push(
                "worker",
                "wait",
                from,
                Instant::now(),
                ThreadRecord::default(),
            );
        }
        match work {
            Work::Apply(op) => {
                let from = Instant::now();
                let r = backend.apply(&op);
                if let Some(t) = timeline.as_deref_mut() {
                    let fields = ThreadRecord {
                        child_ns: backend.child_ns(),
                        ..op_fields(&op)
                    };
                    t.push("worker", "apply", from, Instant::now(), fields);
                }
                let mut c = lock(shared);
                match r {
                    Ok(()) => c.applied = op.seq,
                    Err(error) => {
                        c.error = Some(WorkerError {
                            seq: op.seq,
                            time: op.time,
                            nu: op.nu,
                            gid: op.gid,
                            error,
                        });
                        shared.failed.store(true, Ordering::Relaxed);
                        shared.to_timing.notify_all();
                    }
                }
            }
            Work::Serve(l) => {
                let from = Instant::now();
                let failed = lock(shared).error.is_some();
                let r = if failed {
                    Err(FuncError("functional worker failed".into()))
                } else {
                    backend.read(l.addr, l.len)
                };
                let mut c = lock(shared);
                let applied = c.applied;
                if let Some(t) = timeline.as_deref_mut() {
                    let fields = ThreadRecord {
                        seq: Some(applied),
                        addr: Some(l.addr.0),
                        len: Some(l.len),
                        ..ThreadRecord::default()
                    };
                    t.push("worker", "load", from, Instant::now(), fields);
                }
                c.reply = Some(r.map(|bytes| LoadReply { bytes, applied }));
                shared.to_timing.notify_all();
            }
        }
    }
}
