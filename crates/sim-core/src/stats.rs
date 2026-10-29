//! Run outputs (r2sim spec Section 11): plain structs that serialize to
//! the `--stats` JSON.

use nu_model::{Addr, Dep, DepError, NeutrinoId, OpenPending, Range, TaskId, Warning};

use crate::profile::{OpcodeId, PeId};
use crate::resource::{ResourceId, ResourceStats};
use crate::time::Cycle;

/// The granted interval of one chunk of a task; all zero for a chunk not
/// yet requested.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct ChunkRec {
    /// Cycle the chunk's demand starts.
    pub start: Cycle,
    /// Cycle the chunk ends (exclusive).
    pub end: Cycle,
    /// `None` for a chunk with no demand, or in zero-time mode.
    pub culprit: Option<ResourceId>,
}

/// One task from invoke to retire.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TaskTimeline {
    /// The Neutrino whose tables track the task (the owner of `dst`).
    pub nu: NeutrinoId,
    /// The task's id in the tables of `nu`.
    pub task: TaskId,
    /// The descriptor's opcode.
    pub opcode: OpcodeId,
    /// The executing PE; `None` for `join` and `spawn`.
    pub pe: Option<PeId>,
    /// The descriptor's label (software identity).
    pub label: Option<u64>,
    /// Entry into the scoreboard.
    pub invoke: Cycle,
    /// Cycle the last dependency resolved (after `c_resolve`).
    pub ready: Option<Cycle>,
    /// Cycle the task entered the execution table.
    pub issue: Option<Cycle>,
    /// The task's chunks in request order; set at issue.
    pub chunks: Vec<ChunkRec>,
    /// Cycle the completion reached the tables (after `c_complete`).
    pub complete: Option<Cycle>,
    /// Cycle the entry left the tables: at completion for a versioned
    /// task; for an unversioned one when a new producer of its range
    /// supersedes it or its program returns.
    pub retire: Option<Cycle>,
    /// The version of a versioned `dst`.
    pub version: Option<u64>,
}

/// A warning that does not stop the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum SimWarning {
    /// A warning from the dependency tables.
    Dep(Warning),
    /// A load of `range` while the task `writer` (a gid) writes it.
    Race {
        /// The loaded range.
        range: Range,
        /// Gid of the issued task that writes the range.
        writer: u64,
    },
    /// A pending entry was created over `range`, which a completed task
    /// (`writer`, a gid) or a store (`None`) already wrote. Usually a
    /// consumer invoked after its producer retired: it will not issue.
    LateConsumer {
        /// The range of the new pending entry.
        range: Range,
        /// Gid of a completed task that wrote the range; `None` for a store.
        writer: Option<u64>,
    },
}

/// One warning, with the time and place it was raised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct WarningRec {
    /// Cycle of the warning.
    pub time: Cycle,
    /// The Neutrino that raised it.
    pub nu: NeutrinoId,
    /// The task whose invoke raised it, if any.
    pub gid: Option<u64>,
    /// The warning.
    pub warning: SimWarning,
}

/// The errors that stop a run.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum FatalError {
    /// A software error from the dependency tables.
    Dep(DepError),
    /// A range that no Neutrino owns, or outside every memory block.
    Unowned(Range),
    /// A load of `range` while the task `writer` (a gid) is issued and
    /// not done: its value would depend on the functional worker's speed.
    Race {
        /// The loaded range.
        range: Range,
        /// Gid of the issued task that writes the range.
        writer: u64,
    },
    /// A functional backend error; `seq` is the failing op (0 for a load).
    Func {
        /// Func seq of the failing op; 0 for a load.
        seq: u64,
        /// The backend's error message.
        msg: String,
    },
    /// The functional worker panicked.
    WorkerPanic(String),
    /// An operation the program may not do (a PE of another Neutrino, a
    /// spawn on a busy or non-child Neutrino).
    Invalid(String),
    /// A feature of a later step.
    Unsupported(&'static str),
    /// No task issued or completed, no PE ran and no message moved for
    /// `idle` cycles (checked at an `Await` resend), or, when no event is
    /// left, messages are left in the channels or a program has not
    /// returned.
    Deadlock {
        /// Cycles since the last issue, completion or message move.
        idle: Cycle,
        /// Every Neutrino's table occupancy at the error.
        tables: Vec<TableState>,
        /// Every channel stage that holds messages.
        stages: Vec<StageState>,
        /// Every program that has not returned.
        programs: Vec<BlockedProgram>,
    },
}

/// A program that has not returned, and what it waits on (a deadlock
/// report).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct BlockedProgram {
    /// The program's Neutrino.
    pub nu: NeutrinoId,
    /// Where it is blocked.
    pub blocked: Blocked,
    /// Its last call (`return`: the return waits for its untracked tasks).
    pub call: &'static str,
    /// The key of a wait (not the fence).
    pub key: Option<Dep>,
    /// The owner of `key`.
    pub owner: Option<NeutrinoId>,
}

/// A channel stage that holds messages (a deadlock report).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct StageState {
    /// `<lane or Neutrino source>.<class>`.
    pub stage: String,
    /// Messages held.
    pub messages: usize,
}

/// One channel stage over the run.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct StageStat {
    /// `<lane>.<class>`, or `<Neutrino>.source.<class>` (for invokes the
    /// outbound invoke table).
    pub stage: String,
    /// Cycles its head could not move (head-of-line).
    pub hol_cycles: Cycle,
    /// The most messages it held at once.
    pub peak: usize,
}

/// Occupancy of one Neutrino's tables (a deadlock report).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TableState {
    /// The Neutrino.
    pub nu: NeutrinoId,
    /// Scoreboard entries.
    pub scoreboard: usize,
    /// Execution table entries.
    pub execution: usize,
    /// Pending table entries.
    pub pending: usize,
    /// Version table entries.
    pub versions: usize,
    /// Inbound ref table entries.
    pub inbound_refs: usize,
    /// Inbound invoke table entries.
    pub inbound_invokes: usize,
    /// Outbound dep table entries in use.
    pub outbound_deps: usize,
    /// Outbound invoke table entries in use.
    pub outbound_invokes: usize,
}

/// The error that stopped the run. Stats after `time` are not valid.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Fatal {
    /// Cycle of the error.
    pub time: Cycle,
    /// The Neutrino where the error happened (0 for a worker panic).
    pub nu: NeutrinoId,
    /// Gid of the task involved, if any.
    pub gid: Option<u64>,
    /// The error.
    pub error: FatalError,
}

/// A pending entry still open at the end of the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct OpenPendingRec {
    /// The Neutrino whose tables hold the entry.
    pub nu: NeutrinoId,
    /// The open entry.
    pub pending: OpenPending,
    /// The last completed writer of any byte of the range, when the
    /// open-pending table report is on: `Some(Some(gid))` for a task, `Some(None)`
    /// for a store, `None` if nothing wrote it. For an unversioned range, a
    /// task here was superseded before its consumer was invoked.
    pub last_writer: Option<Option<u64>>,
}

/// A full queue that stalls a program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Stall {
    /// The scoreboard is full.
    Scoreboard,
    /// The outbound dep table entries are full.
    OutboundDeps,
    /// Scoreboard and execution table leave no issue headroom.
    Headroom,
    /// The slots for forwarded invokes not yet accepted are full.
    OutboundInvokes,
}

/// Where an unfinished program is blocked at the end of the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Blocked {
    /// Waiting for a covering entry to complete (local or remote).
    Wait,
    /// Stalled on a full queue.
    Stall(Stall),
    /// Waiting for its request to execute or its response to arrive.
    Busy,
}

/// One interval of the wall-clock timeline of a thread. Times are
/// nanoseconds from the start of the functional plane, on one monotonic
/// clock. The fields that do not apply to a record are `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ThreadRecord {
    /// `control` (the timing thread) or `worker` (the functional worker).
    pub thread: &'static str,
    /// Control: `run` (the event loop), `emit_wait` (no room in the op
    /// queue), `load_wait` (a load waits for the worker's reply) or
    /// `finish_wait` (the end of the run waits for the worker to apply the
    /// ops left). Worker: `apply` (one op), `load` (one load served) or
    /// `wait` (no work).
    pub state: &'static str,
    /// Start, in ns.
    pub start: u64,
    /// End, in ns.
    pub end: u64,
    /// Control: the simulated cycle at the end of the interval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycle: Option<u64>,
    /// Control: the seq of the last emitted op at the end of the interval.
    /// Worker `apply`: the op's seq; `load`: the last applied seq.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
    /// Worker `apply`: `dma`, `matmul`, `vector`, `join`, `spawn`,
    /// `imem_copy` or `store`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op: Option<&'static str>,
    /// Worker `apply`: the Neutrino that emitted the op.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nu: Option<u32>,
    /// Worker `apply`: the bytes of a DMA, an IMEM copy or a store.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Worker `apply` of a matmul: m, n and k.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub m: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<u64>,
    /// `load_wait` and worker `load`: the address and length of the load.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub addr: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub len: Option<u64>,
    /// `load_wait`: the seq the load waits for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq_needed: Option<u64>,
    /// Worker `apply` with a process backend that reports it: the time the
    /// child process spent on the op, in ns, by its own clock.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_ns: Option<u64>,
}

/// One functional sync point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct LoadRec {
    /// Cycle the load was served.
    pub time: Cycle,
    /// The loading Neutrino.
    pub nu: NeutrinoId,
    /// Start address of the load.
    pub addr: Addr,
    /// Bytes loaded.
    pub len: u64,
    /// The last writer's seq that the load waited for.
    pub seq_needed: u64,
    /// Ops emitted when the load was posted.
    pub emitted: u64,
    /// Ops the worker had applied when it served the load.
    pub applied: u64,
}

/// The outputs of one run; `r2sim --stats` serializes them into JSON.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Stats {
    /// Resource totals at program marks, when on.
    pub marks: Vec<Mark>,
    /// Time of the last event.
    pub end_time: Cycle,
    /// Events processed by the run loop.
    pub events: u64,
    /// Indexed by task gid; empty when the timeline is off.
    pub tasks: Vec<TaskTimeline>,
    /// Tasks invoked, with or without the timeline.
    pub invoked: u64,
    /// Indexed by `ResourceId`.
    pub resources: Vec<ResourceStats>,
    /// Warnings in the order raised.
    pub warnings: Vec<WarningRec>,
    /// The first fatal error; `None` if no error stopped the run.
    pub fatal: Option<Fatal>,
    /// pending entries still open at the end.
    pub open_pending: Vec<OpenPendingRec>,
    /// Programs that did not finish.
    pub unfinished: Vec<(NeutrinoId, Blocked)>,
    /// Every Neutrino's table occupancy at the end of the run.
    pub tables_at_end: Vec<TableState>,
    /// Functional ops emitted.
    pub func_ops: u64,
    /// Functional sync points, one per load, in order.
    pub loads: Vec<LoadRec>,
    /// The wall-clock timeline of the timing thread and the functional
    /// worker, when on ([`crate::Sim::set_thread_timeline`]); ordered by
    /// start. Not in the stats JSON.
    #[serde(skip)]
    pub thread_timeline: Vec<ThreadRecord>,
    /// Fabric messages sent (remote protocol, forwarded invoke, spawn).
    pub messages: u64,
    /// `Await`s rejected by the owner for capacity.
    pub rejects: u64,
    /// Rejected `Await`s resent.
    pub resends: u64,
    /// Program cycles stalled on a full outbound invoke table, per
    /// Neutrino.
    pub outbound_stall: Vec<Cycle>,
    /// Program cycles stalled on a local invoke (a full scoreboard or no
    /// issue headroom), per Neutrino.
    pub table_stall: Vec<Cycle>,
    /// Per channel stage: head-of-line cycles and peak occupancy.
    pub stages: Vec<StageStat>,
    /// Messages by beats (1 to 4).
    pub beats: [u64; 4],
    /// Messages sent, by kind.
    pub messages_by_kind: std::collections::BTreeMap<String, u64>,
    /// Messages between different subtrees of the root (between gpcs, or
    /// to or from the device), by kind.
    pub cross_messages_by_kind: std::collections::BTreeMap<String, u64>,
    /// Ticket flow: tickets per requester (its bit set at an owner).
    pub tickets: Vec<u64>,
    /// Ticket flow: `Reject`s per requester.
    pub ticket_rejects: Vec<u64>,
    /// Ticket flow: `InvokeAck`s per requester.
    pub invoke_acks: Vec<u64>,
    /// Ticket flow: cycles from each `RetryAck` to its ticket's last
    /// replay, per requester.
    pub ticket_wait: Vec<Cycle>,
    /// Ticket flow: the most bits set at once in each owner's bitmap.
    pub ticket_bitmap_peak: Vec<usize>,
    /// Ticket flow: the most tickets one requester held at once.
    pub tickets_held_peak: Vec<usize>,
    /// The most outbound invoke table entries in use at once, per
    /// Neutrino (under ticket flow: requests not yet acknowledged).
    pub outbound_invoke_peak: Vec<usize>,
    /// The most scoreboard entries held at once, per Neutrino.
    pub scoreboard_peak: Vec<usize>,
    /// The most version table entries held at once, per Neutrino.
    pub version_peak: Vec<usize>,
    /// The most inbound invoke table entries held at once, per Neutrino.
    pub inbound_invoke_peak: Vec<usize>,
    /// Per cached DRAM (the `cache` experiment): its counters and flushes
    /// (left out of the JSON when there is none).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub caches: Vec<crate::sim::cache::CacheStats>,
}

/// Resource totals at a program mark: per resource (by id), busy, culprit
/// and delay cycles so far.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Mark {
    /// Cycle of the mark.
    pub time: Cycle,
    /// The name the program gave the mark.
    pub name: String,
    /// Per resource: `[busy, culprit, delay]` cycles.
    pub res: Vec<[u64; 3]>,
}
