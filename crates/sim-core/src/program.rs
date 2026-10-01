//! Neutrino programs as coroutines (architecture Section 6.1).
//!
//! A program is an `async fn` over a [`NuCtx`]. Every hardware operation is
//! an awaited call. The call future puts a [`Request`] into the Neutrino's
//! mailbox and returns `Pending` once; the scheduler handles the request,
//! puts the response into the mailbox, and polls the program again. There
//! is no async runtime; the scheduler polls with a no-op waker. A program
//! has at most one outstanding call, so it must not await anything but
//! `NuCtx` calls (no `join!` of two calls).

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use nu_model::{Addr, Dep, NeutrinoId, Range, TaskId};

use crate::profile::Descriptor;
use crate::time::Cycle;

/// A program for one Neutrino.
pub trait NeutrinoProgram {
    /// Builds the program's future over `ctx`; the scheduler polls it.
    fn run(self, ctx: NuCtx) -> impl Future<Output = ()> + 'static;
}

impl<F, Fut> NeutrinoProgram for F
where
    F: FnOnce(NuCtx) -> Fut,
    Fut: Future<Output = ()> + 'static,
{
    fn run(self, ctx: NuCtx) -> impl Future<Output = ()> + 'static {
        self(ctx)
    }
}

/// The program's handle on an invoked task: the base address of `dst` and
/// the version. A local versioned invoke gets the owner's version; a
/// forwarded invoke has version 0 (the owner assigns it on arrival, and the
/// program infers it) and no local task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskHandle {
    /// The local task entry; `None` for a forwarded invoke.
    pub task: Option<TaskId>,
    /// Index into [`crate::stats::Stats::tasks`] when the timeline is on;
    /// `None` for a forwarded invoke.
    pub gid: Option<u64>,
    /// The base address of the range the task writes.
    pub addr: u64,
    /// The version of a local versioned `dst`; 0 for an unversioned or a
    /// forwarded invoke.
    pub version: u64,
}

impl TaskHandle {
    /// The dependency key that names this task's output: `(addr, version)`
    /// for a versioned `dst`, else the address. For a forwarded invoke the
    /// program names the version it infers instead.
    pub fn key(&self) -> Dep {
        match self.version {
            0 => Dep::Unversioned(self.addr),
            version => Dep::Versioned {
                addr: self.addr,
                version,
            },
        }
    }
}

/// A call from a program to its Neutrino.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Request {
    Invoke(Descriptor),
    Wait(Dep),
    Version(u64, u64),
    VersionReset,
    VReset(u64),
    Cycles(u64),
    Load {
        addr: Addr,
        len: u64,
    },
    Store {
        addr: Addr,
        bytes: Vec<u8>,
    },
    /// A named point in time: records the resource totals when marks are on
    /// (phase-scoped statistics). Costs nothing.
    Mark(String),
    /// `spawn_child`; the program travels in the mailbox.
    Spawn {
        child: NeutrinoId,
        done: Range,
        deps: Vec<Dep>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Response {
    Invoked(TaskHandle),
    Bytes(Vec<u8>),
    Done,
}

/// A program value ready to start on a Neutrino.
pub(crate) type StartFn = Box<dyn FnOnce(NuCtx) -> Pin<Box<dyn Future<Output = ()>>>>;

#[derive(Default)]
pub(crate) struct Mailbox {
    pub request: Option<Request>,
    /// The program of a `Spawn` request.
    pub spawn: Option<StartFn>,
    pub response: Option<Response>,
}

/// The program's view of its Neutrino.
#[derive(Debug, Clone)]
pub struct NuCtx {
    id: NeutrinoId,
    mb: Rc<RefCell<Mailbox>>,
    clock: Rc<Cell<Cycle>>,
    children: Rc<Vec<NeutrinoId>>,
}

impl std::fmt::Debug for Mailbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mailbox")
            .field("request", &self.request)
            .field("spawn", &self.spawn.is_some())
            .field("response", &self.response)
            .finish()
    }
}

impl NuCtx {
    pub(crate) fn new(
        id: NeutrinoId,
        mb: Rc<RefCell<Mailbox>>,
        clock: Rc<Cell<Cycle>>,
        children: Vec<NeutrinoId>,
    ) -> Self {
        Self {
            id,
            mb,
            clock,
            children: Rc::new(children),
        }
    }

    /// The Neutrino this program runs on.
    pub fn id(&self) -> NeutrinoId {
        self.id
    }

    /// The direct children of this Neutrino, in configuration order.
    pub fn children(&self) -> &[NeutrinoId] {
        &self.children
    }

    /// Starts `program` on the direct child `child` once `deps` are done.
    /// The spawn is a task in this Neutrino's tables with `dst = done` (a
    /// sync-only range this Neutrino owns); it completes when the child
    /// program returns, so `wait(done)` waits for the child.
    pub async fn spawn_child(
        &self,
        child: NeutrinoId,
        done: Range,
        deps: Vec<Dep>,
        program: impl NeutrinoProgram + 'static,
    ) -> TaskHandle {
        let start: StartFn = Box::new(move |ctx| Box::pin(program.run(ctx)));
        self.mb.borrow_mut().spawn = Some(start);
        match self.call(Request::Spawn { child, done, deps }).await {
            Response::Invoked(h) => h,
            r => panic!("spawn_child answered with {r:?}"),
        }
    }

    /// The global clock.
    pub fn now(&self) -> Cycle {
        self.clock.get()
    }

    /// Invokes a task. Stalls while the scoreboard or the outbound dep queue is
    /// full. A `dst` in another Neutrino's prefix forwards the invoke to its
    /// owner (forwarded invoke); it stalls while the outbound invoke queue is
    /// full, and returns once the descriptor is queued.
    pub async fn invoke(&self, desc: Descriptor) -> TaskHandle {
        match self.call(Request::Invoke(desc)).await {
            Response::Invoked(h) => h,
            r => panic!("invoke answered with {r:?}"),
        }
    }

    /// Blocks until version `version` of the range at base address `addr`
    /// is done.
    pub async fn wait(&self, addr: u64, version: u64) {
        self.call(Request::Wait(Dep::Versioned { addr, version }))
            .await;
    }

    /// Blocks until every unversioned producer whose `dst` contains `addr`
    /// is done.
    pub async fn wait_unversioned(&self, addr: u64) {
        self.call(Request::Wait(Dep::Unversioned(addr))).await;
    }

    /// Blocks until `key` is done ([`NuCtx::wait`] or
    /// [`NuCtx::wait_unversioned`] by the key's kind).
    pub async fn wait_key(&self, key: Dep) {
        self.call(Request::Wait(key)).await;
    }

    /// Sets the version counter of the versioned range at base address
    /// `addr` (creates its version table entry); the next local invoke on
    /// it gets `v + 1`.
    pub async fn version(&self, addr: u64, v: u64) {
        self.call(Request::Version(addr, v)).await;
    }

    /// Clears the caller's version table.
    pub async fn vreset_all(&self) {
        self.call(Request::VersionReset).await;
    }

    /// Frees the version table entry at base address `addr`. Routes by
    /// address: local for the caller's own range, otherwise a `VersionReset`
    /// message to the owner. An absent entry is a no-op.
    pub async fn vreset(&self, addr: u64) {
        self.call(Request::VReset(addr)).await;
    }

    /// External load: memory latency and bandwidth, then a functional sync.
    /// The value is the functional state after the last writer of the range.
    pub async fn load(&self, addr: Addr, len: u64) -> Vec<u8> {
        match self.call(Request::Load { addr, len }).await {
            Response::Bytes(b) => b,
            r => panic!("load answered with {r:?}"),
        }
    }

    /// External store: memory latency and bandwidth, and a functional op.
    /// Does not wait for the functional plane.
    pub async fn store(&self, addr: Addr, bytes: Vec<u8>) {
        self.call(Request::Store { addr, bytes }).await;
    }

    /// Charges `n` scalar cycles.
    pub async fn cycles(&self, n: u64) {
        self.call(Request::Cycles(n)).await;
    }

    /// Records the resource totals under `name` when marks are on
    /// ([`crate::Sim::set_marks`]); no cost.
    pub async fn mark(&self, name: &str) {
        self.call(Request::Mark(name.to_string())).await;
    }

    fn call(&self, req: Request) -> Call {
        Call {
            mb: self.mb.clone(),
            req: Some(req),
        }
    }
}

struct Call {
    mb: Rc<RefCell<Mailbox>>,
    req: Option<Request>,
}

impl Future for Call {
    type Output = Response;

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Response> {
        if let Some(req) = self.req.take() {
            let mut mb = self.mb.borrow_mut();
            assert!(mb.request.is_none(), "two outstanding NuCtx calls");
            mb.request = Some(req);
            return Poll::Pending;
        }
        match self.mb.borrow_mut().response.take() {
            Some(r) => Poll::Ready(r),
            None => Poll::Pending,
        }
    }
}
