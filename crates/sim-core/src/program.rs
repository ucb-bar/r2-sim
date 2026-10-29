//! Neutrino programs as coroutines (r2sim spec Section 4.2).
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

/// The key of the fence: address 0, which no node owns.
pub(crate) const FENCE: Dep = Dep::Unversioned(0);

/// A call from a program to its Neutrino.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Request {
    Invoke(Descriptor),
    Wait(Dep),
    Version(u64, u64),
    VersionReset,
    VReset(u64),
    /// The instruction `epoch`.
    Epoch(u64),
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
    /// Empties every cache with no traffic (the `cache` experiment).
    CacheFlush,
    /// `spawn_child`; the program travels in the mailbox. `pc`: the spawn
    /// descriptor of an RV program (no `c_desc`).
    Spawn {
        child: NeutrinoId,
        done: Range,
        deps: Vec<Dep>,
        pc: Option<u64>,
        /// The call path of the RV instruction at `pc`.
        call_path: Vec<u32>,
    },
    /// The program stops the run with an error (an RV program's fault).
    Fail(String),
}

impl Request {
    /// The name of the call (a deadlock report).
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Request::Invoke(_) => "invoke",
            Request::Wait(FENCE) => "fence",
            Request::Wait(_) => "wait",
            Request::Version(..) => "version",
            Request::VersionReset => "version_reset",
            Request::VReset(_) => "vreset",
            Request::Epoch(_) => "epoch",
            Request::Cycles(_) => "cycles",
            Request::Load { .. } => "load",
            Request::Store { .. } => "store",
            Request::Mark(_) => "mark",
            Request::CacheFlush => "cache_flush",
            Request::Spawn { .. } => "spawn",
            Request::Fail(_) => "fail",
        }
    }

    /// The key of a wait.
    pub(crate) fn key(&self) -> Option<Dep> {
        match self {
            Request::Wait(k) => Some(*k),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Response {
    Invoked(TaskHandle),
    Bytes(Vec<u8>),
    Done,
}

/// A running program.
pub(crate) type ProgramFuture = Pin<Box<dyn Future<Output = ()>>>;

/// The name of a program for a tracer: fixed, or found when the program
/// starts (an RV program names itself by the record in its IMEM, which
/// holds the image only after the copy that the spawn waits for).
pub enum ProgramName {
    Fixed(&'static str),
    AtStart(Box<dyn FnOnce() -> &'static str>),
}

/// A program value ready to start on a Neutrino, and its name for a
/// tracer.
pub(crate) struct StartFn {
    pub run: Box<dyn FnOnce(NuCtx) -> ProgramFuture>,
    pub name: ProgramName,
    /// The epoch of the started program; above `EPOCH_MAX` is an error at
    /// the spawn.
    pub epoch: u16,
}

#[derive(Default)]
pub(crate) struct Mailbox {
    pub request: Option<Request>,
    /// The program of a `Spawn` request.
    pub spawn: Option<StartFn>,
    pub response: Option<Response>,
}

/// The memories of a Neutrino's scalar core, held in the control plane
/// (RV programs, 2026-10-23): its IMEM, the local window `[0,
/// imem_bytes)` of the core, and its data scratch. Each is allocated, as
/// zeros, on its first use.
#[derive(Debug, Default)]
pub struct CoreMem {
    imem_bytes: u64,
    scratch_bytes: u64,
    imem: Vec<u8>,
    scratch: Vec<u8>,
    /// The image at offset 0 of the IMEM: the last image load or IMEM
    /// copy there.
    pub image: Option<ImemImage>,
}

/// An image placed at offset 0 of an IMEM: its bytes, and for an image
/// of the memory image (`--image`) its file and node; none for a copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImemImage {
    pub bytes: u64,
    pub file: Option<String>,
    pub node: Option<String>,
}

impl CoreMem {
    pub(crate) fn new(imem_bytes: u64, scratch_bytes: u64) -> Self {
        Self {
            imem_bytes,
            scratch_bytes,
            ..Self::default()
        }
    }

    /// The IMEM.
    pub fn imem(&mut self) -> &mut [u8] {
        if self.imem.is_empty() {
            self.imem = vec![0; self.imem_bytes as usize];
        }
        &mut self.imem
    }

    /// The data scratch.
    pub fn scratch(&mut self) -> &mut [u8] {
        if self.scratch.is_empty() {
            self.scratch = vec![0; self.scratch_bytes as usize];
        }
        &mut self.scratch
    }
}

/// A Neutrino's scalar core as a program sees it: its memories, its
/// global prefix (the base of its own space) and the global range of its
/// data scratch (none for a node without a sync-only range).
#[derive(Debug, Clone)]
pub struct Core {
    pub mem: Rc<RefCell<CoreMem>>,
    pub prefix: u64,
    pub scratch: Option<Range>,
}

/// The program's view of its Neutrino.
#[derive(Debug, Clone)]
pub struct NuCtx {
    id: NeutrinoId,
    mb: Rc<RefCell<Mailbox>>,
    clock: Rc<Cell<Cycle>>,
    children: Rc<Vec<NeutrinoId>>,
    /// The Neutrino's `c_plan`.
    plan_cost: Cycle,
    core: Core,
    /// The epoch of the Neutrino: the one the program started with, or
    /// the last `set_epoch`.
    epoch: Rc<Cell<u64>>,
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
        plan_cost: Cycle,
        core: Core,
        epoch: u64,
    ) -> Self {
        Self {
            id,
            mb,
            clock,
            children: Rc::new(children),
            plan_cost,
            core,
            epoch: Rc::new(Cell::new(epoch)),
        }
    }

    /// The epoch of the Neutrino: the one its program started with, or the
    /// last [`NuCtx::set_epoch`].
    pub fn epoch(&self) -> u64 {
        self.epoch.get()
    }

    /// The instruction `epoch e`: an epoch change as at a program start
    /// with epoch `e`. It clears the version table and sets the epoch;
    /// pending and task entries stay, and the checks of a start apply.
    /// Costs `c_version`. An epoch above 255 stops the run (`Invalid`).
    pub async fn set_epoch(&self, e: u64) {
        self.call(Request::Epoch(e)).await;
        self.epoch.set(e);
    }

    /// The Neutrino's scalar core: IMEM, data scratch and prefix.
    pub fn core(&self) -> &Core {
        &self.core
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
        self.spawn_named(child, "child", done, deps, program).await
    }

    /// [`NuCtx::spawn_child`] with a program name for a tracer: a short
    /// static name per program type.
    pub async fn spawn_named(
        &self,
        child: NeutrinoId,
        name: &'static str,
        done: Range,
        deps: Vec<Dep>,
        program: impl NeutrinoProgram + 'static,
    ) -> TaskHandle {
        self.spawn_program(child, ProgramName::Fixed(name), done, deps, None, program)
            .await
    }

    /// [`NuCtx::spawn_named`] with a [`ProgramName`], and the PC of the
    /// spawn descriptor of an RV program (which charges no `c_desc`). The
    /// child starts in epoch 0.
    pub async fn spawn_program(
        &self,
        child: NeutrinoId,
        name: ProgramName,
        done: Range,
        deps: Vec<Dep>,
        pc: Option<u64>,
        program: impl NeutrinoProgram + 'static,
    ) -> TaskHandle {
        self.spawn_in_epoch(child, name, done, deps, pc, 0, program)
            .await
    }

    /// [`NuCtx::spawn_program`] with the epoch of the child's program: its
    /// start clears the child's version table and sets the child's epoch.
    /// An epoch above `EPOCH_MAX` stops the run at the spawn.
    #[allow(clippy::too_many_arguments)]
    pub async fn spawn_in_epoch(
        &self,
        child: NeutrinoId,
        name: ProgramName,
        done: Range,
        deps: Vec<Dep>,
        pc: Option<u64>,
        epoch: u16,
        program: impl NeutrinoProgram + 'static,
    ) -> TaskHandle {
        self.spawn_with(child, name, done, deps, pc, Vec::new(), epoch, program)
            .await
    }

    /// The spawn descriptor of an RV program: [`NuCtx::spawn_in_epoch`]
    /// with the PC of the instruction that made it and the call path of
    /// that instruction ([`Descriptor::call_path`]).
    #[allow(clippy::too_many_arguments)]
    pub async fn spawn_rv(
        &self,
        child: NeutrinoId,
        name: ProgramName,
        done: Range,
        deps: Vec<Dep>,
        pc: u64,
        call_path: Vec<u32>,
        epoch: u16,
        program: impl NeutrinoProgram + 'static,
    ) -> TaskHandle {
        self.spawn_with(child, name, done, deps, Some(pc), call_path, epoch, program)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn spawn_with(
        &self,
        child: NeutrinoId,
        name: ProgramName,
        done: Range,
        deps: Vec<Dep>,
        pc: Option<u64>,
        call_path: Vec<u32>,
        epoch: u16,
        program: impl NeutrinoProgram + 'static,
    ) -> TaskHandle {
        let start = StartFn {
            run: Box::new(move |ctx| Box::pin(program.run(ctx))),
            name,
            epoch,
        };
        self.mb.borrow_mut().spawn = Some(start);
        match self
            .call(Request::Spawn {
                child,
                done,
                deps,
                pc,
                call_path,
            })
            .await
        {
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
    ///
    /// Panics on an untracked descriptor: an untracked task has no handle
    /// ([`NuCtx::invoke_untracked`]).
    pub async fn invoke(&self, desc: Descriptor) -> TaskHandle {
        assert!(
            !desc.untracked,
            "an untracked task has no handle: use invoke_untracked"
        );
        match self.call(Request::Invoke(desc)).await {
            Response::Invoked(h) => h,
            r => panic!("invoke answered with {r:?}"),
        }
    }

    /// Invokes `desc` as an untracked task (sets its flag): it runs on a PE
    /// of this Neutrino, is never forwarded, and returns no handle. Nothing
    /// can depend on it; [`NuCtx::fence`] and the program's return wait for
    /// it. Stalls as [`NuCtx::invoke`] does on a full scoreboard.
    pub async fn invoke_untracked(&self, mut desc: Descriptor) {
        desc.untracked = true;
        match self.call(Request::Invoke(desc)).await {
            Response::Done => {}
            r => panic!("untracked invoke answered with {r:?}"),
        }
    }

    /// The fence, `wait 0`: blocks until every untracked task of the
    /// program is done (`c_wait`; the answer comes `c_resolve` cycles after
    /// the last one completes).
    pub async fn fence(&self) {
        self.call(Request::Wait(FENCE)).await;
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

    /// Stops the run with the error `msg` (a fatal `Invalid` error); the
    /// call does not return.
    pub async fn fail(&self, msg: String) {
        self.call(Request::Fail(msg)).await;
    }

    /// Charges `n` scalar cycles.
    pub async fn cycles(&self, n: u64) {
        self.call(Request::Cycles(n)).await;
    }

    /// Charges the planning of `tasks` tasks: `c_plan` scalar cycles each,
    /// in one block. Nothing happens when that is 0.
    pub async fn plan(&self, tasks: u64) {
        let n = self.plan_cost * tasks;
        if n > 0 {
            self.cycles(n).await;
        }
    }

    /// Records the resource totals under `name` when marks are on
    /// ([`crate::Sim::set_marks`]); no cost.
    pub async fn mark(&self, name: &str) {
        self.call(Request::Mark(name.to_string())).await;
    }

    /// Empties every cache of the run with no traffic and no cost (a
    /// measurement tool's flush outside the measured time; the `cache`
    /// experiment of `sim::cache`). Records the caches' counters.
    pub async fn cache_flush(&self) {
        self.call(Request::CacheFlush).await;
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
