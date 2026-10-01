//! Functional plane interface and worker protocol (architecture Sections 4.4
//! and 4.6).
//!
//! The timing thread emits one [`FuncOp`] per issued PE task and per
//! Neutrino store, in issue order, into a bounded op queue. One worker
//! thread applies them in order. A load is a sync point: it does not travel
//! through the op queue. The timing thread posts it in a separate slot with
//! the seq of the last writer of the range, and blocks. The worker checks
//! the slot between ops and serves the load as soon as it has applied that
//! seq, whether or not it has applied more. For a race-free program the
//! value is the same either way; the timing plane rejects racy loads
//! (Section 4.4).
//!
//! A backend without data (`Null`) runs without a worker thread; a load is
//! then an error.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use nu_model::{Addr, NeutrinoId, Range};

use crate::config::MemBlock;
use crate::profile::Descriptor;
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
    /// Called once before the run with the memory blocks and the
    /// sync-only ranges.
    fn configure(&mut self, memories: &[MemBlock], sync_only: &[Range]);
    /// False for a backend that keeps no data; the simulator then runs it
    /// without a worker thread and a load is an error.
    fn has_data(&self) -> bool;
    /// Applies one op; ops arrive in `seq` order.
    fn apply(&mut self, op: &FuncOp) -> Result<(), FuncError>;
    /// Reads `len` bytes starting at `addr`.
    fn read(&mut self, addr: Addr, len: u64) -> Result<Vec<u8>, FuncError>;
    /// Host-side initialization before the run.
    fn write(&mut self, addr: Addr, bytes: &[u8]) -> Result<(), FuncError>;
}

/// Discards every op. A load is an error: the run declared no
/// data-dependent control.
#[derive(Debug, Default)]
pub struct NullBackend;

impl FunctionalBackend for NullBackend {
    fn configure(&mut self, _: &[MemBlock], _: &[Range]) {}

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

/// The timing thread's handle on the functional plane.
pub(crate) enum FuncPlane {
    /// A backend without data, on the timing thread.
    Inline(Box<dyn FunctionalBackend>),
    Worker {
        shared: Arc<Shared>,
        handle: JoinHandle<Box<dyn FunctionalBackend>>,
        capacity: usize,
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
    pub fn start(mut backend: Box<dyn FunctionalBackend>, capacity: usize) -> Self {
        if !backend.has_data() {
            return FuncPlane::Inline(backend);
        }
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
                worker(&mut *backend, &sh);
                backend
            })
            .expect("spawn the functional worker");
        FuncPlane::Worker {
            shared,
            handle,
            capacity,
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
                shared, capacity, ..
            } => {
                let mut c = lock(shared);
                while c.error.is_none() && c.ops.len() >= *capacity {
                    c = shared.to_timing.wait(c).expect("functional plane lock");
                }
                if let Some(e) = &c.error {
                    return Err(PlaneError::Worker(e.clone()));
                }
                c.ops.push_back(op);
                shared.to_worker.notify_one();
                Ok(())
            }
        }
    }

    /// A sync point: blocks until the worker has applied `seq_needed`, then
    /// returns the bytes it read.
    pub fn load(&mut self, seq_needed: u64, addr: Addr, len: u64) -> Result<LoadReply, PlaneError> {
        match self {
            FuncPlane::Inline(b) => b
                .read(addr, len)
                .map(|bytes| LoadReply { bytes, applied: 0 })
                .map_err(PlaneError::Load),
            FuncPlane::Worker { shared, .. } => {
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
                loop {
                    if let Some(e) = &c.error {
                        return Err(PlaneError::Worker(e.clone()));
                    }
                    if let Some(r) = c.reply.take() {
                        return r.map_err(PlaneError::Load);
                    }
                    c = shared.to_timing.wait(c).expect("functional plane lock");
                }
            }
        }
    }

    /// Stops the worker after it has applied every queued op (or at once
    /// after an error) and returns the backend. `Err` carries a worker
    /// panic message.
    pub fn finish(self) -> (Result<(), String>, Option<Box<dyn FunctionalBackend>>) {
        match self {
            FuncPlane::Inline(b) => (Ok(()), Some(b)),
            FuncPlane::Worker { shared, handle, .. } => {
                lock(&shared).closed = true;
                shared.to_worker.notify_one();
                match handle.join() {
                    Ok(b) => (Ok(()), Some(b)),
                    Err(p) => {
                        let msg = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "unknown panic".into());
                        (Err(msg), None)
                    }
                }
            }
        }
    }
}

fn lock(shared: &Shared) -> std::sync::MutexGuard<'_, Chan> {
    shared.chan.lock().unwrap_or_else(|p| p.into_inner())
}

enum Work {
    Apply(FuncOp),
    Serve(LoadReq),
}

fn worker(backend: &mut dyn FunctionalBackend, shared: &Shared) {
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
                    return;
                }
                c = shared.to_worker.wait(c).expect("functional plane lock");
            }
        };
        match work {
            Work::Apply(op) => {
                let r = backend.apply(&op);
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
                let failed = lock(shared).error.is_some();
                let r = if failed {
                    Err(FuncError("functional worker failed".into()))
                } else {
                    backend.read(l.addr, l.len)
                };
                let mut c = lock(shared);
                let applied = c.applied;
                c.reply = Some(r.map(|bytes| LoadReply { bytes, applied }));
                shared.to_timing.notify_all();
            }
        }
    }
}
