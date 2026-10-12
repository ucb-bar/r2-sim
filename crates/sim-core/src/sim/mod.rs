//! The timing plane: Neutrino and PE actors, the task lifecycle, and the
//! run loop (architecture Section 4.2).
//!
//! ```text
//! program call -> cost on ScalarOps -> NuExec: tables.invoke/wait/version/vreset
//! ready -> PE ready FIFO -> issue (holds, execution table slot) -> chunks -> ChunkEnd ...
//! last ChunkEnd -> c_complete on the completion port -> Complete: tables.complete -> effects
//! load/store -> cost on ScalarOps -> NuExec: memory chunk -> MemDone: sync or op
//! ```
//!
//! Messages, remote dependencies and `spawn_child` are in [`remote`].

mod remote;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use nu_model::{
    Addr, AddressHash, Dep, DepError, DepTables, Effects, NeutrinoId, OutboundDeps, OwnerMap,
    Phase, Purpose, Range, TaskId, TaskSpec, WaitId, WaitStatus,
};

use crate::config::{MemBlock, NeutrinoConfig, PeConfig, RaceCheck};
use crate::event::{Event, EventQueue};
use crate::func::{FuncOp, FuncPlane, FunctionalBackend, NullBackend, OpKind, PlaneError};
use crate::profile::{
    Args, Descriptor, NodeConfig, Occupancy, OpcodeId, PeId, Profile, ProfileRegistry,
};
use crate::program::{Mailbox, NeutrinoProgram, NuCtx, Request, Response, StartFn, TaskHandle};
use crate::resource::{Demand, Dir, Kind, PathHop, ResourceId, Resources};
use crate::stats::{
    Blocked, Fatal, FatalError, LoadRec, OpenPendingRec, SimWarning, Stall, Stats, WarningRec,
};
use crate::time::{Cycle, Ratio};
use crate::topology::{AttachId, Attachment, Bus, BusId, BusTopology, Lane, Topology};
use crate::tracer::{
    ChunkGrant, Collector, CollectorConfig, RouteUse, Table, TableEntry, TaskCall, TaskEntered,
    Tee, Tracer, Who,
};
use crate::writers::LastWriters;
use remote::Msg;
pub use remote::{Flow, MESSAGE_KINDS};

/// Calls the tracer `$t` with `$e` when one is installed: one branch.
macro_rules! trace {
    ($s:ident, $t:ident => $e:expr) => {
        if let Some($t) = $s.tracer.as_mut() {
            $e;
        }
    };
}
pub(crate) use trace;

/// Calendars are pruned every this many events. Measured: 16 gives 3.8 M
/// events/s on the benchmark against 2.8 M at 1024 (deeper calendars).
const PRUNE_EVERY: u64 = 16;

/// The worker error flag is polled every this many events.
const POLL_WORKER_EVERY: u64 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgState {
    /// No program, or it returned.
    Idle,
    /// A request is being charged, executed, or answered.
    Busy,
    /// The request stalled on a full queue; retried when a slot frees.
    Stalled(Stall),
    /// Blocked on a local wait.
    Waiting(WaitId),
    /// Blocked on a remote wait; the tag is the request's `Purpose::Wait`.
    WaitingRemote(u32),
}

struct TaskRec {
    gid: u64,
    desc: Descriptor,
    /// Computed at the first issue attempt; moves to the PE at issue.
    profile: Option<Profile>,
    /// Occupancy held from issue to completion.
    holds: Vec<Occupancy>,
    /// Func seq of the task's op, set at issue.
    seq: u64,
    /// The child program of a `spawn` task, until it is sent.
    spawn: Option<StartFn>,
}

/// The interval a reservation was granted.
#[derive(Debug, Clone, Copy)]
pub(super) struct Reserved {
    start: Cycle,
    pub(super) end: Cycle,
    culprit: Option<ResourceId>,
    queue_culprit: Option<ResourceId>,
}

impl Reserved {
    /// An empty interval at `t`.
    fn at(t: Cycle) -> Self {
        Self {
            start: t,
            end: t,
            culprit: None,
            queue_culprit: None,
        }
    }
}

/// A task that entered the tables.
pub(super) struct Entered {
    task: TaskId,
    gid: u64,
    dst: Range,
    version: Option<u64>,
    ready: bool,
}

/// Why a task did not enter the tables. A stall hands the descriptor back
/// to the program; a refused inbound invoke stays in the inbound invoke
/// table.
pub(super) enum EnterFail {
    Stall(Stall, Box<Descriptor>),
    Refused(Box<Descriptor>),
    Fatal(FatalError),
}

/// How a task enters a Neutrino's tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Entry {
    /// The Neutrino's own invoke: stalls when a table is full.
    Local,
    /// A forwarded invoke leaving the inbound invoke table: it enters only
    /// when all of it fits (a scoreboard entry, pending entries, outbound
    /// dep entries); otherwise it stays in the table. The call is the
    /// requester's, for a tracer.
    Inbound(TaskCall),
}

/// Something that waits for an execution table slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionWaiter {
    Pe(PeId),
    /// A task without a PE (`join`, `spawn`).
    NoPe(TaskId),
}

struct Neutrino {
    id: NeutrinoId,
    cfg: NeutrinoConfig,
    tables: DepTables,
    scalar: ResourceId,
    /// Completion port: `c_complete` is charged here.
    complete_port: ResourceId,
    program: Option<Pin<Box<dyn Future<Output = ()>>>>,
    mb: Rc<RefCell<Mailbox>>,
    /// The request whose cost is charged, or that stalled.
    op: Option<Request>,
    /// The cycle the program made the request in `op`.
    op_call: Cycle,
    /// Invoke and spawn calls of the program so far (their numbers,
    /// [`TaskCall::n`]).
    calls: u64,
    /// The name of the program (for a tracer).
    prog_name: &'static str,
    state: ProgState,
    /// Indexed by `TaskId`; `None` after retire.
    tasks: Vec<Option<TaskRec>>,
    execution_waiters: VecDeque<ExecutionWaiter>,
    /// Requester side of the remote dependency protocol.
    reqs: OutboundDeps,
    /// The program of a stalled `Spawn` request.
    pending_spawn: Option<StartFn>,
    /// The parent's spawn task, while a spawned program runs.
    parent_task: Option<(NeutrinoId, TaskId)>,
    /// Tag of the next remote program wait.
    next_wait_tag: u32,
    /// Since when the program is stalled on a full outbound invoke table.
    outbound_stall_since: Option<Cycle>,
    /// Ready/valid: inbound invoke table entries taken by invokes that
    /// left the channel and have not arrived yet.
    inbound_reserved: usize,
    /// Stages whose head waits for an inbound invoke table entry, in the
    /// order they blocked.
    inbound_waiters: VecDeque<crate::event::Stage>,
    /// Ticket flow, owner side: the ticket bitmap and the grants.
    tickets: remote::Tickets,
    /// Ticket flow, requester side: the requests to each owner not yet
    /// acknowledged, in order, and their number.
    held: std::collections::BTreeMap<u32, VecDeque<remote::Held>>,
    held_len: usize,
    /// The last request sequence number.
    next_seq: u64,
    /// Since when the program is stalled on a full scoreboard or on issue
    /// headroom (a local invoke).
    table_stall_since: Option<Cycle>,
    /// Ticket flow, requester side: the tickets held, by owner.
    ticket: std::collections::BTreeMap<u32, remote::Ticket>,
    /// The most tickets held at once.
    ticket_peak: usize,
    /// The most outbound invoke table entries in use at once.
    outbound_peak: usize,
    /// Inbound invoke table: accepted forwarded invokes (how they enter,
    /// and the descriptor) waiting for a scoreboard entry or for issue
    /// headroom, in arrival order.
    inbound: VecDeque<(Entry, Descriptor)>,
    /// The most entries the inbound invoke table held at once.
    inbound_peak: usize,
    /// A `DrainInbound` event is queued.
    drain_scheduled: bool,
}

impl Neutrino {
    fn rec(&self, t: TaskId) -> &TaskRec {
        self.tasks[t.0 as usize].as_ref().expect("live task record")
    }

    fn rec_mut(&mut self, t: TaskId) -> &mut TaskRec {
        self.tasks[t.0 as usize].as_mut().expect("live task record")
    }
}

/// The address rewrite of a hash link.
#[derive(Debug, Clone)]
struct HashLink {
    #[allow(dead_code)]
    link: usize,
    owner: NeutrinoId,
    numeric: bool,
    hash: AddressHash,
}

struct Running {
    /// The Neutrino that tracks the task (the PE's own).
    nu: NeutrinoId,
    task: TaskId,
    gid: u64,
    profile: Profile,
    /// Next chunk to request.
    next: usize,
    ended: usize,
    /// A transfer (its chunks have a fabric path): chunks are issued at
    /// the engine's beat rate, not when the previous one ends (Section
    /// 5.3, issue-driven chunks).
    stream: bool,
    /// Streaming: the next chunk is due but the outstanding cap holds it.
    held: bool,
    /// Streaming: the latest end of the chunks issued so far.
    last_end: Cycle,
}

struct Pe {
    nu: NeutrinoId,
    cfg: PeConfig,
    /// Ready tasks as (tracking Neutrino, task).
    ready: VecDeque<(NeutrinoId, TaskId)>,
    running: Vec<Running>,
    /// Atlas accumulator memory (H3): held C tiles (dst, bytes), most
    /// recent first.
    acc: Vec<(Range, u64)>,
}

/// Capacity of the op queue to the functional worker.
const FUNC_CAPACITY: usize = 4096;

/// Ticket flow counters, per requester unless noted.
#[derive(Debug, Default)]
struct TicketStats {
    /// Tickets: bits set at an owner.
    tickets: Vec<u64>,
    /// `Reject`s received.
    rejects: Vec<u64>,
    /// `InvokeAck`s received.
    acks: Vec<u64>,
    /// Cycles from a `RetryAck` to the ticket's last replay.
    wait: Vec<Cycle>,
}

/// One simulation: the timing plane and the resource plane.
pub struct Sim {
    now: Cycle,
    clock: Rc<Cell<Cycle>>,
    queue: EventQueue,
    nus: Vec<Neutrino>,
    pes: Vec<Pe>,
    res: Resources,
    memories: Vec<MemBlock>,
    /// The routing function of transfers, built at the start of the run.
    routes: crate::profile::TransferRoutes,
    /// The hops of the transfer chunk being reserved (kept across chunks).
    hop_buf: Vec<PathHop>,

    /// Hash links (architecture Section 3.1): fabric link index, owner,
    /// fidelity, rule.
    hashes: Vec<HashLink>,
    profiles: ProfileRegistry,
    /// The backend before the run; it moves into `plane` at the start.
    backend: Option<Box<dyn FunctionalBackend>>,
    plane: Option<FuncPlane>,
    /// Host writes applied to the backend before the run.
    init: Vec<(Addr, Vec<u8>)>,
    /// Last writers are tracked for a backend with data, the late-consumer
    /// check or the open-pending table report.
    track_writers: bool,
    late_consumer_check: bool,
    open_pending_report: bool,
    writers: LastWriters,
    owners: OwnerMap,
    sync_only: Vec<Range>,
    topo: Topology,
    /// Messages in flight, by id.
    msgs: Vec<Option<Msg>>,
    msg_free: Vec<u32>,
    /// The channel stages (Section 3.3, ready/valid).
    stages: HashMap<crate::event::Stage, remote::StageQ>,
    /// Messages a hop stage holds (a pipeline register).
    hop_queue: usize,
    flow: Flow,
    /// Ticket flow counters.
    ticket_stats: TicketStats,
    msg_bytes: u64,
    /// Extra cycles per remote-protocol message.
    msg_latency: u64,
    /// Port rate of each bus; `None`: unlimited.
    bus_rates: Vec<Option<crate::time::Ratio>>,
    /// The write channel's rate of each bus (`wr_bytes`).
    bus_wr_rates: Vec<Option<crate::time::Ratio>>,
    /// Beat of each bus's resources; `None`: one lane.
    bus_beats: Vec<Option<u64>>,
    /// Cycle of the last task issue or completion.
    last_progress: Cycle,
    /// Idle cycles with continuing replays that stop the run as a retry
    /// livelock; `None`: 10 x the largest `t_retry` plus the longest route
    /// latency.
    livelock_cycles: Option<Cycle>,
    livelock: Cycle,
    func_seq: u64,
    next_gid: u64,
    zero_time: bool,
    /// What the in-memory [`Collector`] keeps.
    collect: CollectorConfig,
    /// A tracer installed with [`Sim::set_tracer`]; it moves into
    /// `tracer` at the start of the run.
    extra: Option<Box<dyn Tracer>>,
    /// The tracer of the run; `None`: no observation point does anything.
    tracer: Option<Box<dyn Tracer>>,
    race: RaceCheck,
    stats: Stats,
}

impl Sim {
    pub fn new(profiles: ProfileRegistry) -> Self {
        Self {
            now: 0,
            clock: Rc::new(Cell::new(0)),
            queue: EventQueue::default(),
            nus: Vec::new(),
            pes: Vec::new(),
            res: Resources::new(),
            memories: Vec::new(),
            routes: Default::default(),
            hop_buf: Vec::new(),
            hashes: Vec::new(),
            profiles,
            backend: Some(Box::new(NullBackend)),
            plane: None,
            init: Vec::new(),
            track_writers: false,
            late_consumer_check: false,
            open_pending_report: true,
            writers: LastWriters::default(),
            owners: OwnerMap::default(),
            sync_only: Vec::new(),
            topo: Topology::default(),
            msgs: Vec::new(),
            msg_free: Vec::new(),
            stages: HashMap::new(),
            hop_queue: 1,
            flow: Flow::ReadyValid,
            ticket_stats: TicketStats::default(),
            msg_bytes: 64,
            msg_latency: 0,
            bus_rates: Vec::new(),
            bus_wr_rates: Vec::new(),
            bus_beats: Vec::new(),
            last_progress: 0,
            livelock_cycles: None,
            livelock: Cycle::MAX,
            func_seq: 0,
            next_gid: 0,
            zero_time: false,
            collect: CollectorConfig {
                tasks: true,
                ..CollectorConfig::default()
            },
            extra: None,
            tracer: None,
            race: RaceCheck::Fatal,
            stats: Stats::default(),
        }
    }

    /// Functional check mode: every reservation, latency and Neutrino cost
    /// takes zero time. The order of events is unchanged.
    pub fn set_zero_time(&mut self, on: bool) {
        self.zero_time = on;
    }

    /// Turns the per-task timeline in [`Stats::tasks`] on or off (default
    /// on).
    pub fn set_timeline(&mut self, on: bool) {
        self.collect.tasks = on;
    }

    /// Record the resource totals (busy, culprit and delay cycles) at each
    /// program [`NuCtx::mark`] in [`Stats::marks`] (default off).
    pub fn set_marks(&mut self, on: bool) {
        self.collect.marks = on;
    }

    /// Turns the per-interval log of every flow resource
    /// ([`crate::ResourceStats::log`]) on or off (default off).
    pub fn set_log(&mut self, on: bool) {
        self.collect.log = on;
    }

    /// Installs a tracer for the run, beside the in-memory collector of
    /// [`Sim::set_timeline`], [`Sim::set_marks`] and [`Sim::set_log`].
    pub fn set_tracer(&mut self, t: Box<dyn Tracer>) {
        self.extra = Some(t);
    }

    /// Warn when a lookup creates a pending entry over a range that a completed
    /// task already wrote (default off). The typical cause is a consumer
    /// invoked after its producer retired. It also fires for a consumer
    /// invoked before the next producer of a reused buffer, which is legal.
    pub fn set_late_consumer_check(&mut self, on: bool) {
        self.late_consumer_check = on;
    }

    /// Report every pending entry still open at the end with the last completed
    /// writer of its range (default on). Precise: an open entry never
    /// issued its waiters.
    pub fn set_open_pending_report(&mut self, on: bool) {
        self.open_pending_report = on;
    }

    /// What a racy load does (default: stop the run).
    pub fn set_race_check(&mut self, race: RaceCheck) {
        self.race = race;
    }

    /// Writes `bytes` into the functional state before the run.
    pub fn init_memory(&mut self, addr: Addr, bytes: Vec<u8>) {
        self.init.push((addr, bytes));
    }

    /// All resources of the simulation.
    pub fn resources(&self) -> &Resources {
        &self.res
    }

    /// All resources of the simulation, mutable.
    pub fn resources_mut(&mut self) -> &mut Resources {
        &mut self.res
    }

    /// The opcode to profile function registry.
    pub fn profiles(&self) -> &ProfileRegistry {
        &self.profiles
    }

    /// Sets the functional backend for the run (default: [`NullBackend`]).
    pub fn set_functional(&mut self, func: Box<dyn FunctionalBackend>) {
        self.backend = Some(func);
    }

    /// Messages one hop stage holds per protocol class (default 1).
    pub fn set_hop_queue(&mut self, n: usize) {
        assert!(n > 0);
        self.hop_queue = n;
    }

    /// Flow control of forwarded invokes (default ready/valid).
    pub fn set_flow(&mut self, flow: Flow) {
        self.flow = flow;
    }

    /// The scoreboard entries of Neutrino `nu` above its issue headroom:
    /// the tasks it can hold for all requesters without room to issue.
    pub fn scoreboard_room(&self, nu: NeutrinoId) -> u32 {
        let c = self.nus[nu.0 as usize].tables.config();
        c.scoreboard_capacity.saturating_sub(c.issue_headroom) as u32
    }

    /// The execution table capacity of Neutrino `nu`.
    pub fn execution_capacity(&self, nu: NeutrinoId) -> usize {
        self.nus[nu.0 as usize].tables.config().execution_capacity
    }

    /// The version table capacity of Neutrino `nu`.
    pub fn version_capacity(&self, nu: NeutrinoId) -> usize {
        self.nus[nu.0 as usize].tables.config().version_capacity
    }

    /// Extra cycles per remote-protocol message (default 0).
    pub(crate) fn set_msg_latency(&mut self, cycles: u64) {
        self.msg_latency = cycles;
    }

    /// Size of one remote-protocol or hierarchy message (default 64 B).
    pub(crate) fn set_msg_bytes(&mut self, bytes: u64) {
        self.msg_bytes = bytes;
    }

    /// The hardware tree.
    pub fn topology(&self) -> &Topology {
        &self.topo
    }

    /// The memory blocks, in insertion order.
    pub fn memories(&self) -> &[MemBlock] {
        &self.memories
    }

    /// Adds a root Neutrino that owns `owned` (its memory and sync-only
    /// ranges). `cfg.name` is its dotted name from the root; the tree node
    /// takes the last component. Creates its scalar and completion-port
    /// resources.
    pub fn add_neutrino(&mut self, cfg: NeutrinoConfig, owned: &[Range]) -> NeutrinoId {
        self.topo.push(short_name(&cfg.name), None);
        self.push_neutrino(cfg, owned)
    }

    /// Adds a Neutrino under `parent`, as [`Sim::add_neutrino`].
    pub fn add_child_neutrino(
        &mut self,
        parent: NeutrinoId,
        cfg: NeutrinoConfig,
        owned: &[Range],
    ) -> NeutrinoId {
        self.topo.push(short_name(&cfg.name), Some(parent));
        self.push_neutrino(cfg, owned)
    }

    /// Adds a bus named `name` of `topology` at `rate` bytes per cycle
    /// (`None`: unlimited, no resources); `control`: a control bus. A shared
    /// bus has one resource (`<name>`); an xbar has ports per attachment
    /// ([`Sim::attach`]).
    pub fn add_bus(
        &mut self,
        name: &str,
        topology: BusTopology,
        rate: Option<crate::time::Ratio>,
        latency: Cycle,
        control: bool,
    ) -> BusId {
        let lane = match topology {
            BusTopology::Shared => self.lane(name, rate),
            BusTopology::Xbar => None,
        };
        self.bus_rates.push(rate);
        self.bus_wr_rates.push(None);
        self.bus_beats.push(None);
        self.topo.add_bus(Bus {
            name: name.to_string(),
            topology,
            lane,
            latency,
            lane_wr: None,
            control,
            block: None,
        })
    }

    /// The rate of `bus` (its ports' or its shared lane's); `None`:
    /// unlimited.
    pub fn bus_rate(&self, bus: BusId) -> Option<Ratio> {
        self.bus_rates[bus.0 as usize]
    }

    /// Splits the resources of `bus` into lanes of `beat` bytes: its shared
    /// lane, and the ports of the attachments made from now on.
    /// Gives a bus a write channel at `rate` (`wr_bytes`, Section 3.5): a
    /// write lane for a `Shared` bus, write ports for the attachments of an
    /// `Xbar`. Call it after [`Sim::set_bus_beat`] and before any
    /// attachment on the bus.
    ///
    /// # Panics
    ///
    /// When the bus already has an attachment: it would have no write
    /// ports, and its writes would fall back to the read channel.
    pub fn set_bus_wr(&mut self, bus: BusId, rate: crate::time::Ratio) {
        let b = self.topo.bus(bus);
        if let Some(a) = (0..self.topo.attachment_count())
            .map(|i| self.topo.attachment(AttachId(i as u32)))
            .find(|a| a.bus == bus)
        {
            panic!(
                "set_bus_wr on {} after its attachment {}: give the bus its write channel before attaching to it",
                b.name, a.name
            );
        }
        self.bus_wr_rates[bus.0 as usize] = Some(rate);
        let b = self.topo.bus(bus);
        if b.topology == BusTopology::Shared {
            let name = format!("{}.wr", b.name);
            let lane = self.lane(&name, Some(rate));
            if let (Some(r), Some(beat)) = (lane, self.bus_beats[bus.0 as usize]) {
                self.res.set_beat(r, beat);
            }
            self.topo.set_bus_wr(bus, lane);
        }
    }

    /// Gives fabric link `link` a write channel per direction.
    pub fn set_link_wr(&mut self, link: usize, ab: Lane, ba: Lane) {
        self.topo.set_link_wr(link, ab, ba);
    }

    pub fn set_bus_beat(&mut self, bus: BusId, beat: u64) {
        self.bus_beats[bus.0 as usize] = Some(beat);
        if let Some(r) = self.topo.bus(bus).lane {
            self.res.set_beat(r, beat);
        }
    }

    /// A bandwidth resource named `name` at `rate`; none when unlimited.
    pub fn lane(&mut self, name: &str, rate: Option<crate::time::Ratio>) -> Lane {
        rate.map(|r| self.res.add_flow(name, Kind::Bandwidth, r))
    }

    /// Attaches `name` to `bus`. On a limited xbar it gets an ingress and
    /// an egress port at the bus rate (`<bus>.<name>.in`, `.out`), unless
    /// `port` is false (a compute master: never a route endpoint that
    /// charges).
    pub fn attach(&mut self, bus: BusId, name: &str, port: bool) -> AttachId {
        let b = self.topo.bus(bus);
        let full = format!("{}.{name}", b.name);
        let rate = self.bus_rates[bus.0 as usize].filter(|_| port);
        let (ingress, egress) = match b.topology {
            BusTopology::Xbar => (
                self.lane(&format!("{full}.in"), rate),
                self.lane(&format!("{full}.out"), rate),
            ),
            BusTopology::Shared => (None, None),
        };
        if let Some(beat) = self.bus_beats[bus.0 as usize] {
            for r in [ingress, egress].into_iter().flatten() {
                self.res.set_beat(r, beat);
            }
        }
        // The write channel (`wr_bytes`), after the read channel.
        let wr_rate = self.bus_wr_rates[bus.0 as usize].filter(|_| port);
        let (ingress_wr, egress_wr) = match (self.topo.bus(bus).topology, wr_rate) {
            (BusTopology::Xbar, Some(r)) => (
                self.lane(&format!("{full}.in.wr"), Some(r)),
                self.lane(&format!("{full}.out.wr"), Some(r)),
            ),
            _ => (None, None),
        };
        if let Some(beat) = self.bus_beats[bus.0 as usize] {
            for r in [ingress_wr, egress_wr].into_iter().flatten() {
                self.res.set_beat(r, beat);
            }
        }
        self.topo.add_attachment(Attachment {
            bus,
            name: full,
            ingress,
            egress,
            ingress_wr,
            egress_wr,
        })
    }

    /// Adds a link from bus `a` (attached as `a_name`) to bus `b` (attached
    /// as `b_name`) with the resource of each direction (`ab` from `a` to
    /// `b`).
    #[allow(clippy::too_many_arguments)]
    pub fn add_link(
        &mut self,
        a: BusId,
        a_name: &str,
        b: BusId,
        b_name: &str,
        ab: Lane,
        ba: Lane,
        latency: Cycle,
    ) {
        let (att_a, att_b) = (self.attach(a, a_name, true), self.attach(b, b_name, true));
        self.topo.add_link(att_a, att_b, ab, ba, latency);
    }

    /// Sets the transaction size (`block_bytes`) of a bus.
    pub fn set_bus_block(&mut self, bus: BusId, block: u64) {
        self.topo.set_bus_block(bus, block);
    }

    /// Sets the transaction size (`block_bytes`) of the fabric link of
    /// index `link` (in [`Topology::fabric_links`] order).
    pub fn set_link_block(&mut self, link: usize, block: u64) {
        self.topo.set_link_block(link, block);
    }

    /// The attachments of Neutrino `nu`: on its control bus (messages) and
    /// on its data buses (loads and stores).
    pub fn set_hub(&mut self, nu: NeutrinoId, control: AttachId, data: Vec<AttachId>) {
        self.topo.set_hub(nu, control, data);
    }

    /// Idle cycles (no issue, no completion) with continuing resends that
    /// stop the run as a retry livelock (default: 10 x the largest `t_retry`
    /// plus the longest route latency).
    pub fn set_livelock_cycles(&mut self, cycles: Cycle) {
        self.livelock_cycles = Some(cycles);
    }

    /// The occupancy of every Neutrino's tables.
    pub(super) fn table_states(&self) -> Vec<crate::stats::TableState> {
        self.nus
            .iter()
            .map(|n| crate::stats::TableState {
                nu: n.id,
                scoreboard: n.tables.scoreboard_len(),
                execution: n.tables.execution_len(),
                pending: n.tables.pending_len(),
                versions: n.tables.versions_len(),
                inbound_refs: n.tables.inbound_refs_len(),
                inbound_invokes: n.inbound.len(),
                outbound_deps: n.reqs.open_count(),
                outbound_invokes: self
                    .stages
                    .get(&crate::event::Stage::Source(n.id.0, remote::INVOKE))
                    .map_or(0, |q| q.occ())
                    .max(n.held_len),
            })
            .collect()
    }

    /// Computes the fabric routes; fails on two shortest routes between
    /// two buses. The run does this itself if the caller did not.
    pub fn finish_fabric(&mut self) -> Result<(), String> {
        self.topo.finish()
    }

    fn push_neutrino(&mut self, cfg: NeutrinoConfig, owned: &[Range]) -> NeutrinoId {
        let id = NeutrinoId(self.nus.len() as u32);
        for r in owned {
            self.owners.insert_range(r, id);
        }
        let scalar = self.res.add_flow(
            &format!("{}.scalar", cfg.name),
            Kind::ScalarOps,
            cfg.scalar_rate,
        );
        let complete_port = self.res.add_flow(
            &format!("{}.complete", cfg.name),
            Kind::ScalarOps,
            cfg.complete_rate,
        );
        let cfg_outbound_deps = cfg.outbound_deps;
        self.nus.push(Neutrino {
            id,
            tables: DepTables::new(cfg.tables),
            cfg,
            scalar,
            complete_port,
            program: None,
            mb: Rc::new(RefCell::new(Mailbox::default())),
            op: None,
            op_call: 0,
            calls: 0,
            prog_name: "",
            state: ProgState::Idle,
            tasks: Vec::new(),
            execution_waiters: VecDeque::new(),
            reqs: OutboundDeps::new(id, cfg_outbound_deps),
            pending_spawn: None,
            parent_task: None,
            next_wait_tag: 0,
            outbound_stall_since: None,
            inbound_reserved: 0,
            inbound_waiters: VecDeque::new(),
            tickets: remote::Tickets::default(),
            held: Default::default(),
            held_len: 0,
            next_seq: 0,
            ticket: Default::default(),
            ticket_peak: 0,
            table_stall_since: None,
            outbound_peak: 0,
            inbound: VecDeque::new(),
            inbound_peak: 0,
            drain_scheduled: false,
        });
        id
    }

    /// Marks `range` as sync-only: no storage, no functional effect.
    pub fn add_sync_only(&mut self, range: Range) {
        self.sync_only.push(range);
    }

    /// Adds a PE on `nu` and returns its ID. Panics if `cfg.max_inflight` or
    /// `cfg.max_outstanding` is 0.
    pub fn add_pe(&mut self, nu: NeutrinoId, cfg: PeConfig) -> PeId {
        assert!(cfg.max_inflight > 0 && cfg.max_outstanding > 0);
        self.pes.push(Pe {
            nu,
            cfg,
            ready: VecDeque::new(),
            running: Vec::new(),
            acc: Vec::new(),
        });
        PeId(self.pes.len() as u32 - 1)
    }

    /// Applies `f` to the knobs (quantize, acc_buffer, row_panel) of every
    /// Atlas PE.
    fn each_atlas(&mut self, mut f: impl FnMut(&mut bool, &mut u64, &mut bool)) {
        for p in &mut self.pes {
            if let crate::config::PeParams::Atlas {
                quantize,
                acc_buffer,
                row_panel,
                ..
            } = &mut p.cfg.params
            {
                f(quantize, acc_buffer, row_panel);
            }
        }
    }

    /// Sets `quantize` on every Atlas PE (Section 5.5 knob).
    pub fn set_atlas_quantize(&mut self, quantize: bool) {
        self.each_atlas(|q, _, _| *q = quantize);
    }

    /// The configuration of PE `pe`.
    pub fn pe_config(&self, pe: PeId) -> &crate::config::PeConfig {
        &self.pes[pe.0 as usize].cfg
    }

    /// The Neutrino of PE `pe`.
    pub fn pe_neutrino(&self, pe: PeId) -> NeutrinoId {
        self.pes[pe.0 as usize].nu
    }

    /// Number of PEs.
    pub fn pe_count(&self) -> usize {
        self.pes.len()
    }

    /// The configuration of Neutrino `nu`.
    pub fn neutrino_config(&self, nu: NeutrinoId) -> &NeutrinoConfig {
        &self.nus[nu.0 as usize].cfg
    }

    /// The scalar resource and the completion port of Neutrino `nu`.
    pub fn neutrino_resources(&self, nu: NeutrinoId) -> (ResourceId, ResourceId) {
        let n = &self.nus[nu.0 as usize];
        (n.scalar, n.complete_port)
    }

    /// The dependency resolution latency (`c_resolve`) of Neutrino `nu`
    /// (0 in zero-time mode).
    pub fn resolve_cost(&self, nu: NeutrinoId) -> Cycle {
        if self.zero_time {
            0
        } else {
            self.nus[nu.0 as usize].cfg.costs.resolve
        }
    }

    /// Atlas accumulator memory in bytes (H3; 0 = none).
    pub fn set_atlas_acc_buffer(&mut self, bytes: u64) {
        self.each_atlas(|_, a, _| *a = bytes);
    }

    /// Atlas operand reuse (H1): keep the A row panel across a row's
    /// output tiles.
    pub fn set_atlas_row_panel(&mut self, on: bool) {
        self.each_atlas(|_, _, r| *r = on);
    }

    /// Adds a memory block.
    pub fn add_memory(&mut self, mem: MemBlock) {
        self.memories.push(mem);
    }

    /// Adds the address rewrite of fabric link `link` (Section 3.1), whose
    /// node `owner` owns its linear range.
    pub fn add_hash(&mut self, link: usize, owner: NeutrinoId, numeric: bool, hash: AddressHash) {
        self.hashes.push(HashLink {
            link,
            owner,
            numeric,
            hash,
        });
    }

    /// The hash links' rules.
    pub fn hashes(&self) -> impl Iterator<Item = &AddressHash> {
        self.hashes.iter().map(|h| &h.hash)
    }

    /// A compute task on `pe` whose `dst` or source lies in a hash link's
    /// linear range: the error. A compute PE charges one block per
    /// operand and is not routed through a hash link.
    fn hashed_compute(&self, pe: PeId, desc: &Descriptor) -> Option<String> {
        if self.hashes.is_empty() {
            return None;
        }
        let p = &self.pes[pe.0 as usize].cfg;
        if matches!(p.params, crate::config::PeParams::Dma { .. }) {
            return None;
        }
        let src = match desc.args {
            Args::Vector { src, .. } => src,
            _ => None,
        };
        let addrs = [Some(Addr(desc.dst.base)), src];
        addrs.into_iter().flatten().find_map(|a| {
            self.hashes
                .iter()
                .find(|h| h.hash.range.contains(a))
                .map(|h| {
                    format!(
                        "address {:#x} of the task is in the linear range {} of a hash link; \
                         a compute PE ({}) does not address a hashed range",
                        a.0, h.hash.range, p.name
                    )
                })
        })
    }

    /// Starts `program` on `nu` at the current time.
    pub fn spawn(&mut self, nu: NeutrinoId, program: impl NeutrinoProgram) {
        self.spawn_named(nu, "main", program);
    }

    /// [`Sim::spawn`] with a program name for a tracer.
    pub fn spawn_named(
        &mut self,
        nu: NeutrinoId,
        name: &'static str,
        program: impl NeutrinoProgram,
    ) {
        let n = &mut self.nus[nu.0 as usize];
        assert!(
            n.program.is_none(),
            "Neutrino {nu:?} already runs a program"
        );
        let ctx = NuCtx::new(
            nu,
            n.mb.clone(),
            self.clock.clone(),
            self.topo.node(nu).children.clone(),
            n.cfg.costs.plan,
        );
        // A program start resets the Neutrino's version table (Section 3.2).
        n.tables.vreset_all();
        n.program = Some(Box::pin(program.run(ctx)));
        n.prog_name = name;
        n.state = ProgState::Busy;
        let now = self.now;
        trace!(self, t => t.entries_clear(nu, Table::Versions, now));
        trace!(self, t => t.program_start(nu, name, None, now));
        self.queue.push(self.now, Event::NuResume(nu));
    }

    /// Runs until the queue is empty or an error stops the run.
    pub fn run(self) -> Stats {
        self.run_with_backend().0
    }

    /// Like [`Sim::run`], and returns the functional backend (`None` after
    /// a worker panic) so that its final state can be inspected.
    pub fn run_with_backend(mut self) -> (Stats, Option<Box<dyn FunctionalBackend>>) {
        if let Err(e) = self.topo.finish() {
            panic!("fabric: {e}");
        }
        let pes: Vec<&PeConfig> = self.pes.iter().map(|p| &p.cfg).collect();
        let hashes: Vec<&AddressHash> = self.hashes.iter().map(|h| &h.hash).collect();
        self.routes =
            crate::profile::TransferRoutes::build(&pes, &self.memories, &hashes, &self.topo);
        self.livelock = self.livelock_cycles.unwrap_or_else(|| {
            let t_retry = self.nus.iter().map(|n| n.cfg.t_retry).max().unwrap_or(0);
            10 * t_retry + self.topo.max_latency()
        });
        let mut backend = self.backend.take().expect("backend");
        // The functional plane sees linear addresses: a store per hash
        // link's linear range beside the blocks.
        let mut stores = self.memories.clone();
        for h in &self.hashes {
            stores.push(MemBlock {
                numeric: h.numeric,
                ..MemBlock::plain(String::new(), h.owner, h.hash.range, ResourceId(0), 0)
            });
        }
        backend.configure(&stores, &self.sync_only);
        for (addr, bytes) in std::mem::take(&mut self.init) {
            if let Err(e) = backend.write(addr, &bytes) {
                panic!("init_memory at {:#x}: {}", addr.0, e.0);
            }
        }
        self.track_writers =
            backend.has_data() || self.late_consumer_check || self.open_pending_report;
        self.plane = Some(FuncPlane::start(backend, FUNC_CAPACITY));
        let n = self.nus.len();
        self.ticket_stats = TicketStats {
            tickets: vec![0; n],
            rejects: vec![0; n],
            acks: vec![0; n],
            wait: vec![0; n],
        };
        self.stats.outbound_stall = vec![0; self.nus.len()];
        self.stats.table_stall = vec![0; self.nus.len()];
        self.start_tracer();
        while let Some((t, ev)) = self.queue.pop() {
            debug_assert!(t >= self.now, "event in the past");
            self.now = t;
            self.clock.set(t);
            self.stats.events += 1;
            if self.stats.events.is_multiple_of(PRUNE_EVERY) {
                self.res.prune(t);
            }
            self.dispatch(ev);
            if self.stats.events.is_multiple_of(POLL_WORKER_EVERY) && self.plane().failed() {
                self.worker_failed();
            }
            if self.stats.fatal.is_some() {
                break;
            }
        }
        self.finish()
    }

    /// Installs the tracer of the run: the collector and the tracer of
    /// [`Sim::set_tracer`]. Programs spawned before the run start now.
    fn start_tracer(&mut self) {
        let mut list: Vec<Box<dyn Tracer>> = Vec::new();
        if self.collect.any() {
            list.push(Box::new(Collector::new(self.collect)));
        }
        list.extend(self.extra.take());
        let mut t = match list.len() {
            0 => return,
            1 => list.pop().expect("one tracer"),
            _ => Box::new(Tee(list)),
        };
        self.res.set_record(t.wants_charges());
        t.begin(self);
        for n in self.nus.iter().filter(|n| n.program.is_some()) {
            t.program_start(n.id, n.prog_name, None, self.now);
        }
        self.tracer = Some(t);
    }

    fn plane(&self) -> &FuncPlane {
        self.plane.as_ref().expect("functional plane")
    }

    /// Records the worker's error as the fatal error of the run.
    fn worker_failed(&mut self) {
        if let Some(e) = self.plane().worker_error() {
            self.set_fatal(Fatal {
                time: e.time,
                nu: e.nu,
                gid: e.gid,
                error: FatalError::Func {
                    seq: e.seq,
                    msg: e.error.0,
                },
            });
        }
    }

    fn finish(mut self) -> (Stats, Option<Box<dyn FunctionalBackend>>) {
        self.worker_failed();
        // A message still in a stage, or an invoke in an inbound table,
        // when no event is left: nothing can move it.
        if self.stats.fatal.is_none()
            && (self.stages.values().any(|q| q.occ() > 0)
                || self
                    .nus
                    .iter()
                    .any(|n| !n.inbound.is_empty() || n.held_len > 0))
        {
            self.deadlock(NeutrinoId(0));
        }
        self.stats.stages = self.stage_stats();
        let (joined, backend) = self.plane.take().expect("functional plane").finish();
        if let Err(msg) = joined {
            self.set_fatal(Fatal {
                time: self.now,
                nu: NeutrinoId(0),
                gid: None,
                error: FatalError::WorkerPanic(msg),
            });
        }
        self.stats.end_time = self.now;
        self.stats.resources = self.res.stats();
        self.stats.func_ops = self.func_seq;
        let t = std::mem::take(&mut self.ticket_stats);
        self.stats.tickets = t.tickets;
        self.stats.ticket_rejects = t.rejects;
        self.stats.invoke_acks = t.acks;
        self.stats.ticket_wait = t.wait;
        self.stats.ticket_bitmap_peak = self.nus.iter().map(|n| n.tickets.peak).collect();
        self.stats.tickets_held_peak = self.nus.iter().map(|n| n.ticket_peak).collect();
        self.stats.outbound_invoke_peak = self.nus.iter().map(|n| n.outbound_peak).collect();
        self.stats.inbound_invoke_peak = self.nus.iter().map(|n| n.inbound_peak).collect();
        self.stats.scoreboard_peak = self
            .nus
            .iter()
            .map(|n| n.tables.scoreboard_peak())
            .collect();
        self.stats.version_peak = self.nus.iter().map(|n| n.tables.version_peak()).collect();
        self.stats.tables_at_end = self.table_states();
        for n in &self.nus {
            for p in n.tables.open_pending() {
                let last_writer = if self.open_pending_report {
                    self.writers.any_writer(&Range::from_len(p.addr, 1))
                } else {
                    None
                };
                self.stats.open_pending.push(OpenPendingRec {
                    nu: n.id,
                    pending: p,
                    last_writer,
                });
            }
            let blocked = match n.state {
                ProgState::Idle => continue,
                ProgState::Busy => Blocked::Busy,
                ProgState::Stalled(e) => Blocked::Stall(e),
                ProgState::Waiting(_) | ProgState::WaitingRemote(_) => Blocked::Wait,
            };
            self.stats.unfinished.push((n.id, blocked));
        }
        if let Some(t) = self.tracer.take() {
            t.finish(&mut self.stats);
        }
        (self.stats, backend)
    }

    fn dispatch(&mut self, ev: Event) {
        match ev {
            Event::NuResume(nu) => self.resume(nu),
            Event::NuExec(nu) => self.exec(nu),
            Event::ChunkEnd { pe, nu, task } => self.chunk_end(pe, nu, task),
            Event::ChunkIssue { pe, nu, task } => self.chunk_issue(pe, nu, task),
            Event::Complete { nu, task } => self.complete(nu, task),
            Event::MemDone(nu) => self.mem_done(nu),
            Event::MsgArrive { msg } => self.msg_arrive(msg),
            Event::ResendAwait { nu, owner, msg } => self.resend_await(nu, owner, msg),
            Event::HopArrive { msg } => self.hop_arrive(msg),
            Event::StageFree { stage } => self.stage_free(stage),
            Event::Ready { nu, task } => self.on_ready(nu, task),
            Event::DrainInbound(nu) => {
                self.nus[nu.0 as usize].drain_scheduled = false;
                self.drain_inbound(nu);
            }
            Event::WaitDone { nu, wait } => {
                if self.nus[nu.0 as usize].state == ProgState::Waiting(wait) {
                    self.respond(nu, Response::Done);
                }
            }
        }
    }

    // ---- program driver ----

    fn resume(&mut self, nu: NeutrinoId) {
        let n = &mut self.nus[nu.0 as usize];
        let Some(fut) = n.program.as_mut() else {
            return;
        };
        let mut cx = Context::from_waker(Waker::noop());
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                n.program = None;
                n.state = ProgState::Idle;
                let now = self.now;
                trace!(self, t => t.program_return(nu, now));
                let n = &mut self.nus[nu.0 as usize];
                let parent = n.parent_task.take();
                // Program return frees the done unversioned entries
                // (Section 3.2); the version table resets at the next start.
                let fx = n.tables.program_returned();
                self.effects(nu, None, fx);
                if let Some((parent, task)) = parent {
                    self.send(nu, parent, remote::MsgKind::Return { task });
                }
            }
            Poll::Pending => {
                let req =
                    n.mb.borrow_mut()
                        .request
                        .take()
                        .expect("a program awaited something that is not a NuCtx call");
                if matches!(req, Request::Spawn { .. }) {
                    n.pending_spawn = n.mb.borrow_mut().spawn.take();
                }
                self.start_request(nu, req);
            }
        }
    }

    /// Charges the request's cost on the Neutrino's scalar resource: for
    /// an invoke or a spawn, the descriptor's assembly (`c_desc`) and then
    /// its issue.
    fn start_request(&mut self, nu: NeutrinoId, req: Request) {
        let n = &self.nus[nu.0 as usize];
        let c = &n.cfg.costs;
        let desc = match &req {
            Request::Invoke(d) => c.desc(d.deps.len(), d.wire_bytes()),
            // The spawn descriptor of `Descriptor::wire_bytes`: 20 B of
            // header and `dst`, 8 per dependency, 4 for the child.
            Request::Spawn { deps, .. } => c.desc(deps.len(), 24 + 8 * deps.len() as u64),
            _ => 0,
        };
        let cost = match &req {
            Request::Invoke(_) => c.invoke,
            Request::Wait(_) => c.wait,
            Request::Version(..) | Request::VersionReset => c.version,
            Request::VReset(_) => c.vreset,
            Request::Cycles(k) => *k,
            Request::Load { .. } => c.load,
            Request::Store { .. } => c.store,
            Request::Spawn { .. } => c.spawn,
            Request::Mark(_) => 0,
        };
        // An invoke's cost is its task's (trace-raw 1.6), named by the
        // call's number until the task has a gid.
        let who = match &req {
            Request::Invoke(_) | Request::Spawn { .. } => {
                let n = &mut self.nus[nu.0 as usize];
                n.calls += 1;
                Who::Invoke { nu, n: n.calls }
            }
            _ => Who::Program(nu),
        };
        let r = self.nus[nu.0 as usize].scalar;
        if desc > 0 {
            self.charge(r, desc, who);
        }
        let end = self.charge(r, cost, who);
        let n = &mut self.nus[nu.0 as usize];
        n.op = Some(req);
        n.op_call = self.now;
        n.state = ProgState::Busy;
        self.queue.push(end, Event::NuExec(nu));
    }

    /// Charges `c_complete` on `nu`'s completion port; the completion of
    /// `t` reaches the tables at its end.
    pub(super) fn complete_after_charge(&mut self, nu: NeutrinoId, t: TaskId) {
        let n = &self.nus[nu.0 as usize];
        let (r, cost) = (n.complete_port, n.cfg.costs.complete);
        let gid = n.rec(t).gid;
        let end = self.charge(r, cost, Who::Task(gid));
        self.queue.push(end, Event::Complete { nu, task: t });
    }

    fn charge(&mut self, r: ResourceId, cost: u64, who: Who) -> Cycle {
        self.reserve(&[Demand::work(r, cost)], self.now, who).end
    }

    /// Co-charges `demand` for `who` at or after `earliest`. In zero-time
    /// mode everything takes place at now.
    fn reserve(&mut self, demand: &[Demand], earliest: Cycle, who: Who) -> Reserved {
        if self.zero_time {
            return Reserved::at(self.now);
        }
        let g = self.res.reserve(demand, earliest);
        self.granted(g, earliest, who)
    }

    /// Reserves a transfer chunk along `path` in route order (Section 5.4,
    /// data-path rule) for `who`, starting at or after `earliest`, at most
    /// at the master's `rate`.
    fn reserve_path(
        &mut self,
        path: &[PathHop],
        rate: Ratio,
        earliest: Cycle,
        who: Who,
    ) -> Reserved {
        if self.zero_time {
            return Reserved::at(self.now);
        }
        let g = self.res.reserve_path(path, rate, earliest);
        self.granted(g, earliest, who)
    }

    /// The reservation of `g` (none: an empty one at `earliest`); the
    /// tracer gets its charges.
    fn granted(
        &mut self,
        g: Option<crate::resource::Grant>,
        earliest: Cycle,
        who: Who,
    ) -> Reserved {
        let g = match g {
            Some(g) => Reserved {
                start: g.start,
                end: g.end,
                culprit: Some(g.culprit),
                queue_culprit: g.queue_culprit,
            },
            None => Reserved::at(earliest),
        };
        if let Some(t) = self.tracer.as_mut() {
            for c in self.res.charges.drain(..) {
                t.charge(&c, who);
            }
        }
        g
    }

    fn respond(&mut self, nu: NeutrinoId, r: Response) {
        let n = &mut self.nus[nu.0 as usize];
        n.state = ProgState::Busy;
        n.mb.borrow_mut().response = Some(r);
        self.queue.push(self.now, Event::NuResume(nu));
    }

    /// Executes the pending request after its cost.
    fn exec(&mut self, nu: NeutrinoId) {
        let req = self.nus[nu.0 as usize].op.take().expect("pending request");
        match req {
            Request::Invoke(desc) => {
                self.exec_invoke(nu, desc);
                if !matches!(
                    self.nus[nu.0 as usize].state,
                    ProgState::Stalled(Stall::Scoreboard | Stall::Headroom)
                ) {
                    self.table_stall_end(nu);
                }
            }
            Request::Wait(key) => {
                let range = Range::from_len(key.addr(), 1);
                match self.owners.owner_of_range(&range) {
                    None => return self.fatal(nu, None, FatalError::Unowned(range)),
                    Some(o) if o != nu => return self.remote_wait(nu, o, key),
                    Some(_) => {}
                }
                match self.nus[nu.0 as usize].tables.wait(key) {
                    Ok((status, fx)) => {
                        self.trace_pending(nu, &fx, crate::tracer::PENDING_WAIT, None);
                        self.effects(nu, None, fx);
                        match status {
                            WaitStatus::Done => self.respond(nu, Response::Done),
                            WaitStatus::Pending(w) => {
                                self.nus[nu.0 as usize].state = ProgState::Waiting(w)
                            }
                        }
                    }
                    Err(e) => self.dep_error(nu, None, Request::Wait(key), e),
                }
            }
            Request::Version(addr, v) => {
                if let Err(e) = self.own_range(nu, Range::from_len(addr, 1), "version") {
                    return self.fatal(nu, None, e);
                }
                match self.nus[nu.0 as usize].tables.set_version(addr, v) {
                    Ok(()) => {
                        let now = self.now;
                        trace!(self, t => t.entry_open(&TableEntry {
                            base: addr,
                            version: Some(v),
                            flags: crate::tracer::VERSION_SET,
                            ..TableEntry::new(nu, Table::Versions, addr)
                        }, now));
                        self.respond(nu, Response::Done)
                    }
                    Err(e) => self.fatal(nu, None, FatalError::Dep(e)),
                }
            }
            Request::VersionReset => {
                self.nus[nu.0 as usize].tables.vreset_all();
                let now = self.now;
                trace!(self, t => t.entries_clear(nu, Table::Versions, now));
                self.respond(nu, Response::Done);
            }
            Request::VReset(addr) => {
                let range = Range::from_len(addr, 1);
                match self.owners.owner_of_range(&range) {
                    None => return self.fatal(nu, None, FatalError::Unowned(range)),
                    Some(o) if o != nu => {
                        // Routed by address: a VersionReset message to the
                        // owner, in the invoke channel.
                        return self.vreset_remote(nu, o, addr);
                    }
                    Some(_) => {}
                }
                self.nus[nu.0 as usize].tables.vreset(addr);
                let now = self.now;
                trace!(self, t => t.entry_close(&TableEntry::new(nu, Table::Versions, addr), now));
                // A version table entry freed: forwarded invokes that wait
                // for one can enter.
                self.unstall(nu);
                self.respond(nu, Response::Done);
            }
            Request::Cycles(_) => self.respond(nu, Response::Done),
            Request::Mark(name) => {
                if let Some(mut t) = self.tracer.take() {
                    let res = &self.res;
                    let totals = || {
                        res.stats()
                            .into_iter()
                            .map(|r| [r.busy_cycles, r.culprit_cycles, r.delay_cycles])
                            .collect()
                    };
                    t.mark(self.now, nu, &name, &totals);
                    self.tracer = Some(t);
                }
                self.respond(nu, Response::Done)
            }
            req @ (Request::Load { .. } | Request::Store { .. }) => self.exec_mem(nu, req),
            Request::Spawn { child, done, deps } => {
                let n = &mut self.nus[nu.0 as usize];
                let start = n.pending_spawn.take().expect("spawn program");
                if self.topo.node(child).parent != Some(nu) {
                    let msg = format!("spawn on {child:?}, which is not a direct child");
                    return self.fatal(nu, None, FatalError::Invalid(msg));
                }
                let desc = Descriptor {
                    opcode: OpcodeId::SPAWN,
                    pe: None,
                    dst: done,
                    versioned: false,
                    inferred_version: None,
                    deps,
                    label: None,
                    args: Args::Spawn { child },
                };
                self.nus[nu.0 as usize].pending_spawn = Some(start);
                self.exec_invoke(nu, desc);
            }
        }
    }

    /// A load or store: a chunk on the memory block's port after its
    /// latency, then `MemDone`.
    fn exec_mem(&mut self, nu: NeutrinoId, req: Request) {
        let (addr, len) = match &req {
            Request::Load { addr, len } => (*addr, *len),
            Request::Store { addr, bytes } => (*addr, bytes.len() as u64),
            _ => unreachable!("not a memory request"),
        };
        let load = matches!(req, Request::Load { .. });
        // One piece per block it touches: a piece never crosses a block
        // boundary (Section 5.3). The access is done when its last piece
        // is.
        let (mut a, stop, mut done) = (addr.0, addr.0 + len, self.now);
        loop {
            let Some((i, run)) = self.routes.locate(a) else {
                return self.fatal(nu, None, FatalError::Unowned(Range::from_len(addr.0, len)));
            };
            let n = run.min(stop - a);
            let n = n.min(self.mem_grid(nu, i, a, load));
            match self.mem_piece(nu, i, n, load) {
                Ok(end) => done = done.max(end),
                Err(msg) => return self.fatal(nu, None, FatalError::Invalid(msg)),
            }
            a += n;
            if a >= stop {
                break;
            }
        }
        self.nus[nu.0 as usize].op = Some(req);
        self.queue.push(done, Event::MemDone(nu));
    }

    /// The most a load or store piece at `a` in memory block `i` may move:
    /// the `block_bytes` grid of the block's port and of the route the
    /// data takes (Section 5.3).
    fn mem_grid(&self, nu: NeutrinoId, i: usize, a: u64, load: bool) -> u64 {
        let mem = &self.memories[i];
        let hub = self.topo.data_hub(nu);
        let route = if load {
            self.topo.best_path(&mem.attach, hub)
        } else {
            self.topo.best_path(hub, &mem.attach)
        };
        let blocks = route.map_or(&[][..], |p| &p.blocks[..]);
        crate::topology::grid_run(a, blocks).min(mem.block.map_or(u64::MAX, |b| b - a % b))
    }

    /// Reserves `n` bytes of a load or store in memory block `i`;
    /// returns the cycle it is done.
    fn mem_piece(&mut self, nu: NeutrinoId, i: usize, n: u64, load: bool) -> Result<Cycle, String> {
        let mem = &self.memories[i];
        // A transfer in data order (Section 5.4, data-path rule). A load:
        // the request takes the route to the block and the block latency,
        // then the data comes back from the port over the return route. A
        // store: the data takes the route to the block, then the port;
        // the block latency follows the port.
        let dir = if load { Dir::Read } else { Dir::Write };
        let (port, block_latency, attach) = (mem.port_for(dir), mem.load_latency, &mem.attach);
        let hub = self.topo.data_hub(nu);
        let (to_block, from_block) = (
            self.topo.best_path(hub, attach).cloned(),
            self.topo.best_path(attach, hub).cloned(),
        );
        let (Some(to_block), Some(from_block)) = (to_block, from_block) else {
            return Err(format!("no data route between {:?} and {}", nu, mem.name));
        };
        let hop = |resource, latency| PathHop {
            resource,
            amount: n,
            dir,
            latency,
        };
        let (path, lead, tail): (Vec<PathHop>, Cycle, Cycle) = if load {
            let mut p = vec![hop(port, from_block.lead)];
            p.extend(from_block.hops.iter().map(|h| hop(h.lane, h.latency)));
            (p, to_block.latency + block_latency, 0)
        } else {
            let mut p: Vec<PathHop> = to_block
                .hops_for(Dir::Write)
                .iter()
                .map(|h| hop(h.lane, h.latency))
                .collect();
            p.push(hop(port, 0));
            (p, to_block.lead, block_latency)
        };
        let earliest = self.now + if self.zero_time { 0 } else { lead };
        let rate = Ratio::int(self.nus[nu.0 as usize].cfg.beat_bytes);
        let pair = self
            .tracer
            .is_some()
            .then(|| self.topo.best_pair(hub, attach))
            .flatten();
        let mut g = self.reserve_path(&path, rate, earliest, Who::Program(nu));
        if !self.zero_time {
            g.end += tail;
        }
        if let Some((from, to)) = pair {
            let r = RouteUse {
                from,
                to,
                bytes: n,
                dir,
            };
            trace!(self, t => t.route(&r, g.start, g.end, Who::Program(nu)));
        }
        Ok(g.end)
    }

    /// The memory access of a load or store is done: sync or emit.
    fn mem_done(&mut self, nu: NeutrinoId) {
        match self.nus[nu.0 as usize].op.take().expect("memory request") {
            Request::Load { addr, len } => self.load(nu, addr, len),
            Request::Store { addr, bytes } => {
                let range = Range::from_len(addr.0, bytes.len() as u64);
                let seq = self.emit(nu, None, OpKind::Store { addr, bytes });
                if self.track_writers {
                    self.writers.insert(range, seq, None);
                }
                self.respond(nu, Response::Done);
            }
            _ => unreachable!("not a memory request"),
        }
    }

    /// A functional sync point (Section 4.4).
    fn load(&mut self, nu: NeutrinoId, addr: Addr, len: u64) {
        let range = Range::from_len(addr.0, len);
        // Race check in the owner's tables (a debug check with no cost): a
        // writer that is issued and not done. Writers still in the scoreboard are not
        // a race: their op is not emitted yet.
        let owner = self.owners.owner_of_range(&range).unwrap_or(nu);
        let racer = self.nus[owner.0 as usize]
            .tables
            .overlapping(&range)
            .find(|(_, phase, _)| *phase == Phase::Executing)
            .map(|(t, ..)| t);
        if let Some(t) = racer {
            let writer = self.nus[owner.0 as usize].rec(t).gid;
            match self.race {
                RaceCheck::Fatal => {
                    return self.fatal(nu, Some(writer), FatalError::Race { range, writer });
                }
                RaceCheck::Warn => self.warn(nu, Some(writer), SimWarning::Race { range, writer }),
            }
        }
        let seq_needed = self.writers.max_seq(&range);
        let emitted = self.func_seq;
        let plane = self.plane.as_mut().expect("functional plane");
        match plane.load(seq_needed, addr, len) {
            Ok(reply) => {
                self.stats.loads.push(LoadRec {
                    time: self.now,
                    nu,
                    addr,
                    len,
                    seq_needed,
                    emitted,
                    applied: reply.applied,
                });
                self.respond(nu, Response::Bytes(reply.bytes));
            }
            Err(PlaneError::Worker(_)) => self.worker_failed(),
            Err(PlaneError::Load(e)) => {
                self.fatal(nu, None, FatalError::Func { seq: 0, msg: e.0 });
            }
        }
    }

    /// Emits one functional op and returns its seq.
    fn emit(&mut self, nu: NeutrinoId, gid: Option<u64>, kind: OpKind) -> u64 {
        self.func_seq += 1;
        let op = FuncOp {
            seq: self.func_seq,
            time: self.now,
            nu,
            gid,
            kind,
        };
        let plane = self.plane.as_mut().expect("functional plane");
        if plane.emit(op).is_err() {
            self.worker_failed();
        }
        self.func_seq
    }

    /// A `version` names a range that `nu` owns: versions of
    /// another Neutrino's range are the program's inference (Section 3.2).
    fn own_range(&self, nu: NeutrinoId, range: Range, call: &str) -> Result<(), FatalError> {
        match self.owners.owner_of_range(&range) {
            None => Err(FatalError::Unowned(range)),
            Some(o) if o != nu => Err(FatalError::Invalid(format!(
                "{call} of {range}, a range of another Neutrino"
            ))),
            Some(_) => Ok(()),
        }
    }

    fn exec_invoke(&mut self, nu: NeutrinoId, desc: Descriptor) {
        match self.owners.owner_of_range(&desc.dst) {
            None => return self.fatal(nu, None, FatalError::Unowned(desc.dst)),
            Some(o) if o != nu => return self.forward_invoke(nu, o, desc),
            Some(_) => {}
        }
        match self.enter_task(nu, desc, Entry::Local) {
            Ok(e) => {
                let handle = TaskHandle {
                    task: Some(e.task),
                    gid: Some(e.gid),
                    addr: e.dst.base,
                    version: e.version.unwrap_or(0),
                };
                self.respond(nu, Response::Invoked(handle));
                if e.ready {
                    self.on_ready(nu, e.task);
                }
            }
            Err(EnterFail::Stall(why, desc)) => self.stall(nu, Request::Invoke(*desc), why),
            Err(EnterFail::Refused(_)) => unreachable!("a local invoke is never refused"),
            Err(EnterFail::Fatal(e)) => self.fatal(nu, None, e),
        }
    }

    /// Enters a task into `nu`'s tables: the program's own invoke, or a
    /// forwarded one that arrived by message. A full scoreboard stalls
    /// a local invoke and rejects a forwarded one; so does a full
    /// outbound dep queue. A full pending table is fatal for a local invoke (only the
    /// program could free a slot) and rejects a forwarded one. No state
    /// changes on failure.
    pub(super) fn enter_task(
        &mut self,
        nu: NeutrinoId,
        desc: Descriptor,
        entry: Entry,
    ) -> Result<Entered, EnterFail> {
        let forwarded = entry != Entry::Local;
        let call = match entry {
            Entry::Local => TaskCall {
                src: nu,
                call: self.nus[nu.0 as usize].op_call,
                queued: None,
                sent: None,
                seq: 0,
                n: self.nus[nu.0 as usize].calls,
            },
            Entry::Inbound(call) => call,
        };
        if let Some(pe) = desc.pe {
            // The PE must be the owner's (Section 3.3): a write into another
            // node's memory is a pull by that memory owner's DMA engine.
            if self.pes[pe.0 as usize].nu != nu {
                let msg = format!("invoke on {pe:?}, a PE of another Neutrino");
                return Err(EnterFail::Fatal(FatalError::Invalid(msg)));
            }
        }
        if forwarded && desc.opcode == OpcodeId::SPAWN {
            return Err(EnterFail::Fatal(FatalError::Unsupported("remote spawn")));
        }
        // Split the deps by owner; remote ones count as external deps.
        let mut local = Vec::with_capacity(desc.deps.len());
        let mut remote = Vec::new();
        let mut remote_deps = 0u64;
        for (i, d) in desc.deps.iter().enumerate() {
            match self.owners.owner_of(Addr(d.addr())) {
                None => {
                    let r = Range::from_len(d.addr(), 1);
                    return Err(EnterFail::Fatal(FatalError::Unowned(r)));
                }
                Some(o) if o == nu => local.push(*d),
                Some(o) => {
                    if i < 64 {
                        remote_deps |= 1 << i;
                    }
                    remote.push((o, *d));
                }
            }
        }
        let reqs = &self.nus[nu.0 as usize].reqs;
        if remote.len() > reqs.capacity() {
            let msg = format!(
                "{} remote deps exceed the outbound dep capacity",
                remote.len()
            );
            return Err(EnterFail::Fatal(FatalError::Invalid(msg)));
        }
        if reqs.free() < remote.len() {
            return Err(if forwarded {
                EnterFail::Refused(Box::new(desc))
            } else {
                EnterFail::Stall(Stall::OutboundDeps, Box::new(desc))
            });
        }
        let spec = TaskSpec {
            dst: desc.dst,
            versioned: desc.versioned,
            deps: local,
        };
        // Debug check of the program's inference (Section 3.3): invokes of
        // one requester reach the owner in program order, so the version
        // the program infers is the one the owner assigns (the counter + 1).
        if cfg!(debug_assertions) && desc.versioned {
            let t = &self.nus[nu.0 as usize].tables;
            let assigns = t.version_counter(desc.dst.base).unwrap_or(0) + 1;
            if let Some(v) = desc.inferred_version.filter(|v| *v != assigns) {
                let msg = format!(
                    "{}: the program infers version {v}, the owner assigns {assigns}",
                    desc.dst
                );
                return Err(EnterFail::Fatal(FatalError::Invalid(msg)));
            }
        }
        let tables = &mut self.nus[nu.0 as usize].tables;
        let inv = match tables.invoke(spec, remote.len() as u32) {
            Ok(inv) => inv,
            Err(
                DepError::ScoreboardFull
                | DepError::PendingFull
                | DepError::NoHeadroom
                | DepError::VersionsFull,
            ) if forwarded => return Err(EnterFail::Refused(Box::new(desc))),
            Err(DepError::NoHeadroom) => {
                return Err(EnterFail::Stall(Stall::Headroom, Box::new(desc)))
            }
            Err(DepError::ScoreboardFull) => {
                return Err(EnterFail::Stall(Stall::Scoreboard, Box::new(desc)))
            }
            Err(e) => return Err(EnterFail::Fatal(FatalError::Dep(e))),
        };
        let gid = self.next_gid;
        self.next_gid += 1;
        self.stats.invoked += 1;
        let now = self.now;
        trace!(self, t => t.task_invoke(&TaskEntered {
            gid,
            nu,
            task: inv.task,
            desc: &desc,
            version: inv.version,
            call,
            time: now,
            remote_deps,
        }));
        let dst = desc.dst;
        if self.tracer.is_some() {
            self.trace_invoke_entries(nu, gid, dst, inv.version, &inv.effects);
        }
        let n = &mut self.nus[nu.0 as usize];
        let slot = inv.task.0 as usize;
        if n.tasks.len() <= slot {
            n.tasks.resize_with(slot + 1, || None);
        }
        let spawn = if desc.opcode == OpcodeId::SPAWN {
            n.pending_spawn.take()
        } else {
            None
        };
        n.tasks[slot] = Some(TaskRec {
            gid,
            desc,
            profile: None,
            holds: Vec::new(),
            seq: 0,
            spawn,
        });
        for (owner, range) in remote {
            self.open_remote(nu, owner, range, Purpose::TaskDep(inv.task));
        }
        self.effects(nu, Some(gid), inv.effects);
        Ok(Entered {
            task: inv.task,
            gid,
            dst,
            version: inv.version,
            ready: inv.ready,
        })
    }

    /// The tracer's view of an invoke's table changes: the pending entries
    /// it created and adopted, and its version table entry.
    fn trace_invoke_entries(
        &mut self,
        nu: NeutrinoId,
        gid: u64,
        dst: Range,
        version: Option<u64>,
        fx: &Effects,
    ) {
        self.trace_pending(nu, fx, 0, None);
        let now = self.now;
        let Some(t) = self.tracer.as_mut() else {
            return;
        };
        for p in &fx.adopted {
            let e = TableEntry {
                gid: Some(gid),
                ..TableEntry::new(nu, Table::Pending, p.0 as u64)
            };
            t.entry_close(&e, now);
        }
        if let Some(v) = version {
            t.entry_open(
                &TableEntry {
                    base: dst.base,
                    len: dst.len(),
                    version: Some(v),
                    gid: Some(gid),
                    ..TableEntry::new(nu, Table::Versions, dst.base)
                },
                now,
            );
        }
    }

    /// The tracer's view of the pending entries a lookup created: `flags`
    /// names the lookup, `peer` a remote requester.
    pub(super) fn trace_pending(
        &mut self,
        nu: NeutrinoId,
        fx: &Effects,
        flags: u8,
        peer: Option<NeutrinoId>,
    ) {
        let now = self.now;
        let Some(t) = self.tracer.as_mut() else {
            return;
        };
        for &(p, key) in &fx.pending_created {
            let e = TableEntry {
                peer,
                flags,
                ..TableEntry::new(nu, Table::Pending, p.0 as u64).key(key)
            };
            t.entry_open(&e, now);
        }
    }

    /// A full scoreboard stalls the request. Everything else stops the run,
    /// including a full pending table: only this program can free a pending table slot, and it
    /// is the one that would wait (Section 3.2).
    fn dep_error(&mut self, nu: NeutrinoId, gid: Option<u64>, req: Request, e: DepError) {
        match e {
            DepError::ScoreboardFull => self.stall(nu, req, Stall::Scoreboard),
            _ => self.fatal(nu, gid, FatalError::Dep(e)),
        }
    }

    /// Parks the request until a slot of `why` frees.
    fn stall(&mut self, nu: NeutrinoId, req: Request, why: Stall) {
        let now = self.now;
        let n = &mut self.nus[nu.0 as usize];
        n.op = Some(req);
        n.state = ProgState::Stalled(why);
        if matches!(why, Stall::Scoreboard | Stall::Headroom) {
            n.table_stall_since.get_or_insert(now);
        }
    }

    /// Retries a stalled request. It stalls again if the slot is still
    /// missing.
    fn unstall(&mut self, nu: NeutrinoId) {
        // Inbound invokes move in a separate event: `unstall` runs inside
        // the issue of a task, before that task is on its PE.
        let n = &mut self.nus[nu.0 as usize];
        if !n.inbound.is_empty() && !n.drain_scheduled {
            n.drain_scheduled = true;
            self.queue.push(self.now, Event::DrainInbound(nu));
        }
        let n = &mut self.nus[nu.0 as usize];
        if let ProgState::Stalled(_) = n.state {
            n.state = ProgState::Busy;
            self.queue.push(self.now, Event::NuExec(nu));
        }
    }

    /// The program of `nu` issued its stalled request: the time it waited
    /// for a scoreboard entry or issue headroom ends.
    fn table_stall_end(&mut self, nu: NeutrinoId) {
        if let Some(t) = self.nus[nu.0 as usize].table_stall_since.take() {
            self.stats.table_stall[nu.0 as usize] += self.now - t;
        }
    }

    fn fatal(&mut self, nu: NeutrinoId, gid: Option<u64>, error: FatalError) {
        let time = self.now;
        self.set_fatal(Fatal {
            time,
            nu,
            gid,
            error,
        });
    }

    /// Keeps the first fatal error.
    fn set_fatal(&mut self, f: Fatal) {
        if self.stats.fatal.is_none() {
            self.stats.fatal = Some(f);
        }
    }

    /// Turns table effects into events.
    fn effects(&mut self, nu: NeutrinoId, gid: Option<u64>, fx: Effects) {
        for w in fx.warnings {
            self.warn(nu, gid, SimWarning::Dep(w));
        }
        if self.late_consumer_check {
            // A versioned key named before its producer is legal.
            for (_, key) in &fx.pending_created {
                if let Dep::Unversioned(a) = *key {
                    let range = Range::from_len(a, 1);
                    if let Some(writer) = self.writers.any_writer(&range) {
                        self.warn(nu, gid, SimWarning::LateConsumer { range, writer });
                    }
                }
            }
        }
        for w in fx.resolves {
            let now = self.now;
            let id = remote::ref_id(w.requester, w.request);
            trace!(self, t => t.entry_close(&TableEntry::new(nu, Table::InboundRefs, id), now));
            self.send(nu, w.requester, remote::MsgKind::Resolve { req: w.request });
        }
        let resolve = self.nus[nu.0 as usize].cfg.costs.resolve;
        for w in fx.waits_done {
            if resolve > 0 && !self.zero_time {
                self.queue
                    .push(self.now + resolve, Event::WaitDone { nu, wait: w });
            } else if self.nus[nu.0 as usize].state == ProgState::Waiting(w) {
                self.respond(nu, Response::Done);
            }
        }
        let freed = !fx.retired.is_empty() || !fx.adopted.is_empty();
        for t in fx.retired {
            let n = &mut self.nus[nu.0 as usize];
            let rec = n.tasks[t.0 as usize].take().expect("retired task record");
            let now = self.now;
            trace!(self, t => t.task_retire(rec.gid, now));
        }
        for t in fx.ready {
            self.resolved(nu, t);
        }
        if freed {
            self.unstall(nu);
            let waiters: Vec<ExecutionWaiter> = self.nus[nu.0 as usize]
                .execution_waiters
                .drain(..)
                .collect();
            for w in waiters {
                match w {
                    ExecutionWaiter::Pe(pe) => self.try_issue(pe),
                    ExecutionWaiter::NoPe(t) => self.issue_no_pe(nu, t),
                }
            }
        }
    }

    fn warn(&mut self, nu: NeutrinoId, gid: Option<u64>, warning: SimWarning) {
        self.stats.warnings.push(WarningRec {
            time: self.now,
            nu,
            gid,
            warning,
        });
    }

    // ---- task lifecycle ----

    /// The last dependency of `t` resolved: ready after the resolution
    /// latency (`c_resolve`).
    pub(super) fn resolved(&mut self, nu: NeutrinoId, t: TaskId) {
        let resolve = self.nus[nu.0 as usize].cfg.costs.resolve;
        if resolve > 0 && !self.zero_time {
            self.queue
                .push(self.now + resolve, Event::Ready { nu, task: t });
        } else {
            self.on_ready(nu, t);
        }
    }

    fn on_ready(&mut self, nu: NeutrinoId, t: TaskId) {
        let rec = self.nus[nu.0 as usize].rec(t);
        let (gid, pe) = (rec.gid, rec.desc.pe);
        let now = self.now;
        trace!(self, t => t.task_ready(gid, now));
        match pe {
            None => self.issue_no_pe(nu, t),
            Some(pe) => {
                self.pes[pe.0 as usize].ready.push_back((nu, t));
                self.try_issue(pe);
            }
        }
    }

    /// After a task of `nu` entered the execution table outside a PE: the
    /// issue time, and the invokes that waited for room.
    fn issued(&mut self, nu: NeutrinoId, gid: u64) {
        let now = self.now;
        trace!(self, t => t.task_issue(gid, now, 0));
        self.unstall(nu);
    }

    /// A task without a PE enters the execution table. A join completes after the
    /// Neutrino's completion cost; a spawn sends its program to the child
    /// and completes when the child reports back.
    fn issue_no_pe(&mut self, nu: NeutrinoId, t: TaskId) {
        let n = &mut self.nus[nu.0 as usize];
        if let Err(e) = n.tables.issue(t) {
            assert_eq!(e, DepError::ExecutionFull);
            n.execution_waiters.push_back(ExecutionWaiter::NoPe(t));
            return;
        }
        self.last_progress = self.now;
        let rec = n.rec_mut(t);
        let gid = rec.gid;
        let spawn = match rec.desc.args {
            Args::Spawn { child } => Some((child, rec.spawn.take().expect("spawn program"))),
            _ => None,
        };
        self.issued(nu, gid);
        match spawn {
            Some((child, start)) => self.send(nu, child, remote::MsgKind::Start { task: t, start }),
            None => self.complete_after_charge(nu, t),
        }
    }

    /// Issues ready tasks of `pe` in order while slots, holds and execution table
    /// entries allow.
    fn try_issue(&mut self, pe: PeId) {
        loop {
            let p = &self.pes[pe.0 as usize];
            if p.running.len() >= p.cfg.max_inflight as usize {
                return;
            }
            let Some(&(nu, t)) = p.ready.front() else {
                return;
            };
            let n = &self.nus[nu.0 as usize];
            if n.tables.execution_len() >= n.tables.config().execution_capacity {
                let w = ExecutionWaiter::Pe(pe);
                let n = &mut self.nus[nu.0 as usize];
                if !n.execution_waiters.contains(&w) {
                    n.execution_waiters.push_back(w);
                }
                return;
            }
            if n.rec(t).profile.is_none() {
                if let Some(msg) = self.hashed_compute(pe, &n.rec(t).desc) {
                    let gid = n.rec(t).gid;
                    return self.fatal(nu, Some(gid), FatalError::Invalid(msg));
                }
                let n = &self.nus[nu.0 as usize];
                let acc = acc_flags(&mut self.pes[pe.0 as usize], &n.rec(t).desc);
                let p = &self.pes[pe.0 as usize];
                let node = NodeConfig {
                    pe: &p.cfg,
                    pe_id: pe,
                    routes: &self.routes,
                    memories: &self.memories,
                    topology: &self.topo,
                    acc,
                };
                let prof = self.profiles.profile(&n.rec(t).desc, &node);
                self.nus[nu.0 as usize].rec_mut(t).profile = Some(prof);
            }
            let holds = self.nus[nu.0 as usize]
                .rec(t)
                .profile
                .as_ref()
                .expect("profile")
                .holds
                .clone();
            if let Some(h) = holds.iter().find(|h| !self.res.fits(h.resource, h.amount)) {
                self.res.enqueue(h.resource, pe.0 as u64);
                return;
            }
            for h in &holds {
                assert!(self.res.try_acquire(h.resource, h.amount));
            }

            let n = &mut self.nus[nu.0 as usize];
            n.tables.issue(t).expect("execution table slot checked");
            self.last_progress = self.now;
            let rec = n.rec_mut(t);
            let gid = rec.gid;
            let profile = rec.profile.take().expect("profile");
            rec.holds = holds;
            let desc = rec.desc.clone();
            self.pes[pe.0 as usize].ready.pop_front();
            let now = self.now;
            trace!(self, t => t.task_issue(gid, now, profile.chunks.len()));
            let seq = self.emit(nu, Some(gid), OpKind::Task(desc));
            self.nus[nu.0 as usize].rec_mut(t).seq = seq;
            self.unstall(nu);

            let nchunks = profile.chunks.len();
            let stream = profile.chunks.iter().any(|c| c.transfer.is_some());
            let p = &mut self.pes[pe.0 as usize];
            p.running.push(Running {
                nu,
                task: t,
                gid,
                profile,
                next: 0,
                ended: 0,
                stream,
                held: false,
                last_end: 0,
            });
            if nchunks == 0 {
                self.finish_task(pe, nu, t);
                continue;
            }
            if stream {
                self.issue_chunk(pe, nu, t);
                continue;
            }
            let ahead = nchunks.min(p.cfg.max_outstanding as usize);
            for _ in 0..ahead {
                self.request_chunk(pe, nu, t);
            }
        }
    }

    /// Issues the next chunk of transfer `t` now (Section 5.3): reserves it
    /// at or after now + its latency, and schedules the issue of the one
    /// after it at now + this chunk's bytes at the engine's beat rate.
    /// Without an outstanding cap only the last chunk gets an end event,
    /// at the latest end of all of them.
    fn issue_chunk(&mut self, pe: PeId, nu: NeutrinoId, t: TaskId) {
        let p = &self.pes[pe.0 as usize];
        let capped = p.cfg.max_outstanding != u32::MAX;
        let r = &p.running[running_pos(p, nu, t)];
        let chunk = &r.profile.chunks[r.next];
        let bytes = chunk.transfer.as_ref().map_or(0, |c| c.bytes);
        let rate = chunk.rate;
        let issue = if self.zero_time {
            0
        } else {
            (bytes as u128 * rate.den as u128).div_ceil(rate.num as u128) as u64
        };
        let end = self.reserve_chunk(pe, nu, t, capped);
        let p = &mut self.pes[pe.0 as usize];
        let ri = running_pos(p, nu, t);
        let r = &mut p.running[ri];
        r.last_end = r.last_end.max(end);
        if r.next < r.profile.chunks.len() {
            self.queue
                .push(self.now + issue, Event::ChunkIssue { pe, nu, task: t });
        } else if !capped {
            let end = r.last_end;
            self.queue.push(end, Event::ChunkEnd { pe, nu, task: t });
        }
    }

    /// The next chunk of transfer `t` is due: issued unless the
    /// outstanding cap holds it until a chunk ends.
    fn chunk_issue(&mut self, pe: PeId, nu: NeutrinoId, t: TaskId) {
        let p = &mut self.pes[pe.0 as usize];
        let cap = p.cfg.max_outstanding as usize;
        let ri = running_pos(p, nu, t);
        let r = &mut p.running[ri];
        if r.next - r.ended >= cap {
            r.held = true;
            return;
        }
        self.issue_chunk(pe, nu, t);
    }

    /// Reserves the next chunk of `t` at or after now + its latency.
    fn request_chunk(&mut self, pe: PeId, nu: NeutrinoId, t: TaskId) {
        self.reserve_chunk(pe, nu, t, true);
    }

    /// Reserves the next chunk of `t` at or after now + its latency and,
    /// with `end_event`, schedules its end; returns its end.
    fn reserve_chunk(&mut self, pe: PeId, nu: NeutrinoId, t: TaskId, end_event: bool) -> Cycle {
        let p = &mut self.pes[pe.0 as usize];
        let ri = running_pos(p, nu, t);
        let r = &mut p.running[ri];
        let i = r.next;
        r.next += 1;
        let chunk = &r.profile.chunks[i];
        let earliest = self.now + chunk.latency;
        let gid = r.gid;
        // The profile lives in the PE; a moved-out demand avoids a borrow of
        // `self.pes` across the reservation.
        let (routes, rate) = (chunk.routes, chunk.rate);
        let demand = std::mem::take(&mut r.profile.chunks[i].demand);
        let transfer = r.profile.chunks[i].transfer.take();
        let g = match transfer {
            None => self.reserve(&demand, earliest, Who::Task(gid)),
            Some(tc) => {
                // The hops into a buffer kept across chunks.
                let mut hops = std::mem::take(&mut self.hop_buf);
                tc.hops_into(&mut hops);
                let g = self.reserve_path(&hops, rate, earliest, Who::Task(gid));
                self.hop_buf = hops;
                g
            }
        };
        if let Some(tr) = self.tracer.as_mut() {
            tr.chunk_grant(&ChunkGrant {
                gid,
                index: i,
                start: g.start,
                end: g.end,
                culprit: g.culprit,
                queue_culprit: g.queue_culprit,
            });
            for u in routes.iter().flatten().filter(|u| u.bytes > 0) {
                tr.route(u, g.start, g.end, Who::Task(gid));
            }
        }
        if end_event {
            self.queue.push(g.end, Event::ChunkEnd { pe, nu, task: t });
        }
        g.end
    }

    fn chunk_end(&mut self, pe: PeId, nu: NeutrinoId, t: TaskId) {
        let p = &mut self.pes[pe.0 as usize];
        let capped = p.cfg.max_outstanding != u32::MAX;
        let i = running_pos(p, nu, t);
        let r = &mut p.running[i];
        let n = r.profile.chunks.len();
        // An uncapped transfer has one end event, for all its chunks.
        r.ended = if r.stream && !capped { n } else { r.ended + 1 };
        let (more, all) = (r.next < n, r.ended == n);
        if r.stream {
            if more && r.held {
                r.held = false;
                self.issue_chunk(pe, nu, t);
            }
        } else if more {
            self.request_chunk(pe, nu, t);
        }
        if all {
            self.finish_task(pe, nu, t);
            self.try_issue(pe);
        }
    }

    /// The PE is done with `t`: free its slot and send the completion.
    fn finish_task(&mut self, pe: PeId, nu: NeutrinoId, t: TaskId) {
        let p = &mut self.pes[pe.0 as usize];
        let i = running_pos(p, nu, t);
        p.running.swap_remove(i);
        self.complete_after_charge(nu, t);
    }

    fn complete(&mut self, nu: NeutrinoId, t: TaskId) {
        self.last_progress = self.now;
        let n = &mut self.nus[nu.0 as usize];
        let (gid, seq, dst) = {
            let rec = n.rec(t);
            (rec.gid, rec.seq, rec.desc.dst)
        };
        let holds = std::mem::take(&mut n.rec_mut(t).holds);
        let fx = n.tables.complete(t);
        let now = self.now;
        trace!(self, t => t.task_complete(gid, now));
        if self.track_writers {
            self.writers.insert(dst, seq, Some(gid));
        }
        let mut retry = Vec::new();
        for h in holds {
            retry.extend(self.res.release(h.resource, h.amount));
        }
        self.effects(nu, Some(gid), fx);
        for w in retry {
            self.try_issue(PeId(w as u32));
        }
    }
}

/// The Atlas accumulator memory's decision for matmul `desc` issued on `p`
/// (H3), and its new state: a held dst is read from the buffer; a `keep`
/// task whose C fits stays in it; tiles evicted for room are written back.
fn acc_flags(p: &mut Pe, desc: &Descriptor) -> crate::profile::AccFlags {
    let mut f = crate::profile::AccFlags::default();
    let crate::config::PeParams::Atlas { acc_buffer, .. } = p.cfg.params else {
        return f;
    };
    let Args::Matmul {
        m,
        n,
        accumulate,
        keep,
        out_dtype,
        ..
    } = desc.args
    else {
        return f;
    };
    if acc_buffer == 0 {
        return f;
    }
    let bytes = m * n * out_dtype.bytes();
    let held = p.acc.iter().position(|x| x.0 == desc.dst);
    if let Some(i) = held {
        p.acc.remove(i);
        f.skip_read = accumulate;
    }
    // A tile of the same dst not kept by this task is written by it.
    if keep && bytes <= acc_buffer {
        f.skip_write = true;
        p.acc.insert(0, (desc.dst, bytes));
        let mut used: u64 = p.acc.iter().map(|x| x.1).sum();
        while used > acc_buffer {
            let (_, b) = p.acc.pop().expect("held tile");
            used -= b;
            f.flush_bytes += b;
        }
    }
    f
}

/// The index of task `t` of `nu` among the tasks running on `p`.
fn running_pos(p: &Pe, nu: NeutrinoId, t: TaskId) -> usize {
    p.running
        .iter()
        .position(|r| r.task == t && r.nu == nu)
        .expect("running")
}

/// The last component of a dotted name.
fn short_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}
