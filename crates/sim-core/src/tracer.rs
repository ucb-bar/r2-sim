//! Observation points of the timing plane (architecture Section 7).
//!
//! The simulator calls a [`Tracer`] at fixed points and never reads
//! anything back: no hook returns a value, so a traced run and an untraced
//! run take the same cycles. With no tracer installed every point costs one
//! branch. [`Collector`] is the in-memory tracer behind the per-task
//! timeline, the resource interval log and the program marks of
//! [`Stats`]; the CSV tracer (`--trace-out`) exports from it, and the raw
//! writer of `r2sim` is another implementation.

use nu_model::{Dep, NeutrinoId, TaskId};

use crate::profile::Descriptor;
use crate::resource::{Dir, LogEntry, ResourceId};
use crate::sim::Sim;
use crate::stats::{ChunkRec, Mark, Stats, TaskTimeline};
use crate::time::Cycle;
use crate::topology::AttachId;

/// When and where the program called `invoke` for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskCall {
    /// The Neutrino whose program called `invoke`.
    pub src: NeutrinoId,
    /// The cycle of the call (before `c_invoke` and any stall).
    pub call: Cycle,
    /// A forwarded invoke: the end of `c_invoke`, its entry into the
    /// outbound invoke table of `src`.
    pub queued: Option<Cycle>,
    /// A forwarded invoke: the cycle the Invoke message left the outbound
    /// invoke table onto the channel.
    pub sent: Option<Cycle>,
    /// A forwarded invoke: its outbound invoke entry at `src` (the
    /// [`TableEntry::id`] of its [`Table::OutboundInvokes`] entry); 0 for
    /// a local invoke.
    pub seq: u64,
    /// The invoke's number among the invoke and spawn calls of `src`
    /// (from 1; 0: none). With `src` it names the invoke before its task
    /// has a gid ([`Owner::Invoke`]).
    pub n: u64,
}

/// A Neutrino table whose entries a tracer can follow. The scoreboard and
/// the execution table follow from the task hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Table {
    /// Keys named before their producer's invoke.
    Pending = 0,
    /// The version table: one entry per versioned base address.
    Versions = 1,
    /// Owner side: remote `Await`s registered on keys not yet done.
    InboundRefs = 2,
    /// Owner side: forwarded invokes waiting to enter the scoreboard.
    InboundInvokes = 3,
    /// Requester side: open remote dependency requests.
    OutboundDeps = 4,
    /// Requester side: forwarded invokes and `VersionReset`s not yet out
    /// of the outbound invoke table.
    OutboundInvokes = 5,
}

/// One entry of a [`Table`] (trace-raw.md Section 4.7 gives the meaning of
/// every field per table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableEntry {
    /// The Neutrino whose table holds the entry.
    pub nu: NeutrinoId,
    pub table: Table,
    /// Unique among the open entries of (`nu`, `table`).
    pub id: u64,
    /// The key's address or the range base.
    pub base: u64,
    /// The range length; 0 for a key.
    pub len: u64,
    pub version: Option<u64>,
    pub gid: Option<u64>,
    /// The other Neutrino.
    pub peer: Option<NeutrinoId>,
    pub flags: u8,
}

impl TableEntry {
    /// Entry `id` of `table` of `nu` with no field set.
    pub fn new(nu: NeutrinoId, table: Table, id: u64) -> Self {
        Self {
            nu,
            table,
            id,
            base: 0,
            len: 0,
            version: None,
            gid: None,
            peer: None,
            flags: 0,
        }
    }

    /// With the address and version of `key`.
    pub fn key(self, key: Dep) -> Self {
        Self {
            base: key.addr(),
            version: key.version(),
            ..self
        }
    }
}

/// [`Table::Pending`] flags: the lookup that created the entry.
pub const PENDING_WAIT: u8 = 1;
pub const PENDING_AWAIT: u8 = 2;
/// [`Table::Versions`] flag: `nu.version` created the entry.
pub const VERSION_SET: u8 = 1;
/// [`Table::OutboundDeps`] flags.
pub const DEP_REGISTERED: u8 = 1;
pub const DEP_REJECTED: u8 = 2;
/// [`Table::OutboundInvokes`] flags.
pub const INVOKE_VRESET: u8 = 1;
pub const INVOKE_REPLAYED: u8 = 2;

/// A task that entered its owner's tables.
#[derive(Debug, Clone, Copy)]
pub struct TaskEntered<'a> {
    /// The task's global id (invoke order).
    pub gid: u64,
    /// The Neutrino whose tables track it.
    pub nu: NeutrinoId,
    /// Its id in those tables.
    pub task: TaskId,
    pub desc: &'a Descriptor,
    /// The version of a versioned `dst`.
    pub version: Option<u64>,
    pub call: TaskCall,
    /// The cycle it entered the scoreboard.
    pub time: Cycle,
    /// Bit `i`: dependency `i` of `desc` is owned by another Neutrino (a
    /// remote request follows; the first 64 dependencies).
    pub remote_deps: u64,
}

/// What a calendar charge is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Who {
    /// A chunk of task `gid` (or its completion cost).
    Task(u64),
    /// Hop `hop` of a protocol message of kind `kind` from `from` to `to`.
    Message {
        /// Index in [`crate::MESSAGE_KINDS`].
        kind: u8,
        from: NeutrinoId,
        to: NeutrinoId,
        hop: usize,
        bytes: u64,
        /// What the message is for.
        owner: Owner,
    },
    /// Program work of a Neutrino: a NuCtx call's cost, a load or a store.
    Program(NeutrinoId),
    /// The scalar cost of invoke (or spawn) call `n` of `nu`
    /// (`c_desc` and `c_invoke`): the task it invokes, whose gid exists
    /// only from its invoke at the owner.
    Invoke { nu: NeutrinoId, n: u64 },
}

/// What a protocol message is for (trace-raw 1.6, the owner of its
/// charges).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    /// Nothing.
    None,
    /// Task `gid`.
    Task(u64),
    /// The program that sent or waits for the message, by its spawn task
    /// (`None`: a program started with no spawn task, the root program).
    Program { spawn: Option<u64> },
    /// The task of invoke call `n` of `src` ([`TaskCall::n`]), before its
    /// gid exists.
    Invoke { src: NeutrinoId, n: u64 },
}

/// One direction of one calendar charge of a flow resource.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Charge {
    pub resource: ResourceId,
    pub dir: Dir,
    /// Units of this direction (MACs, lane-ops, bytes, instructions).
    pub amount: u64,
    /// First cycle of the whole charge.
    pub start: Cycle,
    /// Cycles of the whole charge.
    pub span: Cycle,
    /// Slots this direction holds over `span`, in
    /// [`crate::resource::SCALE`] units; the directions of a charge sum to
    /// its usage.
    pub usage: u64,
    /// Slot-cycles of this direction.
    pub cycles: f64,
    /// The rate class the charge ran at, on a resource whose rate depends
    /// on the operand type (see [`crate::resource::Demand::class`]).
    pub class: Option<u8>,
}

/// A part of a chunk that moves bytes over a fabric route: from attachment
/// `from` to attachment `to`, in direction `dir`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteUse {
    pub from: AttachId,
    pub to: AttachId,
    pub bytes: u64,
    pub dir: Dir,
}

/// A chunk's granted interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkGrant {
    pub gid: u64,
    /// Chunk index in the task.
    pub index: usize,
    pub start: Cycle,
    pub end: Cycle,
    /// The resource with the longest own duration.
    pub culprit: Option<ResourceId>,
    /// The resource whose earliest start delayed the chunk most.
    pub queue_culprit: Option<ResourceId>,
}

/// The observation points. Every method has an empty default.
#[allow(unused_variables)]
pub trait Tracer {
    /// The run starts; the topology, resources and memories are final.
    fn begin(&mut self, sim: &Sim) {}
    /// True when the tracer uses [`Tracer::charge`]: the resource plane
    /// then records every charge (asked once, at the start of the run).
    fn wants_charges(&self) -> bool {
        true
    }
    /// A task entered its owner's tables.
    fn task_invoke(&mut self, t: &TaskEntered) {}
    /// Its last dependency resolved (after `c_resolve`).
    fn task_ready(&mut self, gid: u64, time: Cycle) {}
    /// It entered the execution table; it has `chunks` chunks.
    fn task_issue(&mut self, gid: u64, time: Cycle, chunks: usize) {}
    /// Its completion reached the tables.
    fn task_complete(&mut self, gid: u64, time: Cycle) {}
    /// Its entry left the tables.
    fn task_retire(&mut self, gid: u64, time: Cycle) {}
    /// A chunk's interval was granted.
    fn chunk_grant(&mut self, c: &ChunkGrant) {}
    /// A calendar charge (one call per direction with a nonzero amount;
    /// `span` is 0 when the charge rounds to no cycle).
    fn charge(&mut self, c: &Charge, who: Who) {}
    /// The bytes of `r` moved over its fabric route during `[start, end)`
    /// (a chunk, a load or a store). The rated links of the route are also
    /// charged; the unlimited ones are not.
    fn route(&mut self, r: &RouteUse, start: Cycle, end: Cycle, who: Who) {}
    /// A message whose control route has no rated link left at `start`
    /// and arrived at `end`.
    fn message_free(&mut self, who: Who, start: Cycle, end: Cycle) {}
    /// Neutrino `nu` opened remote request `req` for dependency `key` of
    /// task `gid` (an Await).
    fn dep_remote_open(&mut self, nu: NeutrinoId, req: u32, gid: u64, key: Dep) {}
    /// The remote request `req` of `nu` was satisfied (AwaitAck done or
    /// Resolve).
    fn dep_remote_done(&mut self, nu: NeutrinoId, req: u32, time: Cycle) {}
    /// Entry `e` entered its table at `time`. For an entry that is already
    /// open this is an update: a `version` replaces the entry's, a `len`
    /// fills a zero one, and `flags` are added.
    fn entry_open(&mut self, e: &TableEntry, time: Cycle) {}
    /// Entry `e` left its table at `time` (which can be later than now: an
    /// outbound invoke frees at the end of its transmission). A set
    /// `version`, `gid` or `peer` replaces the open entry's; `flags` are
    /// added. An `id` that is not open is ignored.
    fn entry_close(&mut self, e: &TableEntry, time: Cycle) {}
    /// Every entry of `table` of `nu` left at `time`.
    fn entries_clear(&mut self, nu: NeutrinoId, table: Table, time: Cycle) {}
    /// A program started on `nu`; `spawn` is the parent's spawn task.
    fn program_start(
        &mut self,
        nu: NeutrinoId,
        name: &'static str,
        spawn: Option<u64>,
        time: Cycle,
    ) {
    }
    /// The program of `nu` returned.
    fn program_return(&mut self, nu: NeutrinoId, time: Cycle) {}
    /// A program mark; `totals` gives the per-resource `[busy, culprit,
    /// delay]` cycles at this point (computed only if called).
    fn mark(
        &mut self,
        time: Cycle,
        nu: NeutrinoId,
        name: &str,
        totals: &dyn Fn() -> Vec<[u64; 3]>,
    ) {
    }
    /// The run ended (normally or on a fatal error, in `stats.fatal`).
    fn finish(self: Box<Self>, stats: &mut Stats) {}
}

/// Several tracers in order.
pub struct Tee(pub Vec<Box<dyn Tracer>>);

impl Tracer for Tee {
    fn begin(&mut self, sim: &Sim) {
        self.0.iter_mut().for_each(|t| t.begin(sim));
    }
    fn wants_charges(&self) -> bool {
        self.0.iter().any(|t| t.wants_charges())
    }
    fn task_invoke(&mut self, e: &TaskEntered) {
        self.0.iter_mut().for_each(|t| t.task_invoke(e));
    }
    fn task_ready(&mut self, gid: u64, time: Cycle) {
        self.0.iter_mut().for_each(|t| t.task_ready(gid, time));
    }
    fn task_issue(&mut self, gid: u64, time: Cycle, chunks: usize) {
        self.0
            .iter_mut()
            .for_each(|t| t.task_issue(gid, time, chunks));
    }
    fn task_complete(&mut self, gid: u64, time: Cycle) {
        self.0.iter_mut().for_each(|t| t.task_complete(gid, time));
    }
    fn task_retire(&mut self, gid: u64, time: Cycle) {
        self.0.iter_mut().for_each(|t| t.task_retire(gid, time));
    }
    fn chunk_grant(&mut self, c: &ChunkGrant) {
        self.0.iter_mut().for_each(|t| t.chunk_grant(c));
    }
    fn charge(&mut self, c: &Charge, who: Who) {
        self.0.iter_mut().for_each(|t| t.charge(c, who));
    }
    fn route(&mut self, r: &RouteUse, start: Cycle, end: Cycle, who: Who) {
        self.0.iter_mut().for_each(|t| t.route(r, start, end, who));
    }
    fn message_free(&mut self, who: Who, start: Cycle, end: Cycle) {
        self.0
            .iter_mut()
            .for_each(|t| t.message_free(who, start, end));
    }
    fn dep_remote_open(&mut self, nu: NeutrinoId, req: u32, gid: u64, key: Dep) {
        self.0
            .iter_mut()
            .for_each(|t| t.dep_remote_open(nu, req, gid, key));
    }
    fn dep_remote_done(&mut self, nu: NeutrinoId, req: u32, time: Cycle) {
        self.0
            .iter_mut()
            .for_each(|t| t.dep_remote_done(nu, req, time));
    }
    fn entry_open(&mut self, e: &TableEntry, time: Cycle) {
        self.0.iter_mut().for_each(|t| t.entry_open(e, time));
    }
    fn entry_close(&mut self, e: &TableEntry, time: Cycle) {
        self.0.iter_mut().for_each(|t| t.entry_close(e, time));
    }
    fn entries_clear(&mut self, nu: NeutrinoId, table: Table, time: Cycle) {
        self.0
            .iter_mut()
            .for_each(|t| t.entries_clear(nu, table, time));
    }
    fn program_start(
        &mut self,
        nu: NeutrinoId,
        name: &'static str,
        spawn: Option<u64>,
        time: Cycle,
    ) {
        self.0
            .iter_mut()
            .for_each(|t| t.program_start(nu, name, spawn, time));
    }
    fn program_return(&mut self, nu: NeutrinoId, time: Cycle) {
        self.0.iter_mut().for_each(|t| t.program_return(nu, time));
    }
    fn mark(
        &mut self,
        time: Cycle,
        nu: NeutrinoId,
        name: &str,
        totals: &dyn Fn() -> Vec<[u64; 3]>,
    ) {
        self.0
            .iter_mut()
            .for_each(|t| t.mark(time, nu, name, totals));
    }
    fn finish(self: Box<Self>, stats: &mut Stats) {
        for t in self.0 {
            t.finish(stats);
        }
    }
}

/// What the [`Collector`] keeps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollectorConfig {
    /// The per-task timeline ([`Stats::tasks`]).
    pub tasks: bool,
    /// The interval log of every flow resource
    /// ([`crate::ResourceStats::log`]).
    pub log: bool,
    /// The program marks with resource totals ([`Stats::marks`]).
    pub marks: bool,
}

impl CollectorConfig {
    /// True when something is kept.
    pub fn any(&self) -> bool {
        self.tasks || self.log || self.marks
    }
}

/// The in-memory tracer: the per-task timeline, the interval log and the
/// marks, moved into [`Stats`] at the end of the run.
#[derive(Debug, Default)]
pub struct Collector {
    cfg: CollectorConfig,
    tasks: Vec<TaskTimeline>,
    log: Vec<Vec<LogEntry>>,
    marks: Vec<Mark>,
}

impl Collector {
    pub fn new(cfg: CollectorConfig) -> Self {
        Self {
            cfg,
            ..Self::default()
        }
    }

    fn task(&mut self, gid: u64) -> Option<&mut TaskTimeline> {
        self.tasks.get_mut(gid as usize)
    }
}

impl Tracer for Collector {
    fn wants_charges(&self) -> bool {
        self.cfg.log
    }

    fn begin(&mut self, sim: &Sim) {
        if self.cfg.log {
            self.log = vec![Vec::new(); sim.resources().len()];
        }
    }

    fn task_invoke(&mut self, e: &TaskEntered) {
        if !self.cfg.tasks {
            return;
        }
        debug_assert_eq!(e.gid as usize, self.tasks.len(), "gids in invoke order");
        self.tasks.push(TaskTimeline {
            nu: e.nu,
            task: e.task,
            opcode: e.desc.opcode,
            pe: e.desc.pe,
            label: e.desc.label,
            invoke: e.time,
            ready: None,
            issue: None,
            chunks: Vec::new(),
            complete: None,
            retire: None,
            version: e.version,
        });
    }

    fn task_ready(&mut self, gid: u64, time: Cycle) {
        if let Some(t) = self.task(gid) {
            t.ready = Some(time);
        }
    }

    fn task_issue(&mut self, gid: u64, time: Cycle, chunks: usize) {
        if let Some(t) = self.task(gid) {
            t.issue = Some(time);
            t.chunks = vec![ChunkRec::default(); chunks];
        }
    }

    fn task_complete(&mut self, gid: u64, time: Cycle) {
        if let Some(t) = self.task(gid) {
            t.complete = Some(time);
        }
    }

    fn task_retire(&mut self, gid: u64, time: Cycle) {
        if let Some(t) = self.task(gid) {
            t.retire = Some(time);
        }
    }

    fn chunk_grant(&mut self, c: &ChunkGrant) {
        if let Some(t) = self.task(c.gid) {
            t.chunks[c.index] = ChunkRec {
                start: c.start,
                end: c.end,
                culprit: c.culprit,
            };
        }
    }

    fn charge(&mut self, c: &Charge, _who: Who) {
        // The interval log keeps intervals of at least one cycle.
        if !self.cfg.log || c.span == 0 {
            return;
        }
        // The directions of one charge come one after the other: one entry.
        let log = &mut self.log[c.resource.0 as usize];
        let end = c.start + c.span;
        let k = c.dir as usize;
        match log.last_mut() {
            Some(e) if e.start == c.start && e.end == end && e.dir[k] == 0.0 => e.dir[k] = c.cycles,
            _ => {
                let mut dir = [0.0; 3];
                dir[k] = c.cycles;
                log.push(LogEntry {
                    start: c.start,
                    end,
                    dir,
                });
            }
        }
    }

    fn mark(
        &mut self,
        time: Cycle,
        _nu: NeutrinoId,
        name: &str,
        totals: &dyn Fn() -> Vec<[u64; 3]>,
    ) {
        if self.cfg.marks {
            self.marks.push(Mark {
                time,
                name: name.to_string(),
                res: totals(),
            });
        }
    }

    fn finish(self: Box<Self>, stats: &mut Stats) {
        let c = *self;
        if c.cfg.tasks {
            stats.tasks = c.tasks;
        }
        if c.cfg.marks {
            stats.marks = c.marks;
        }
        if c.cfg.log {
            for (r, log) in stats.resources.iter_mut().zip(c.log) {
                r.log = Some(log);
            }
        }
    }
}
