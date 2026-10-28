//! The dependency tables of one Neutrino (Neutrino spec Section 6).
//!
//! A task lives in three phases:
//!
//! * pending table: a dependency key that a consumer, a wait or a remote
//!   request named before its producer was invoked;
//! * scoreboard: invoked, depcount counts down;
//! * execution table: executing, then done. A versioned task retires at
//!   completion (the version table carries its state from then on); an
//!   unversioned task's done entry stays until a new producer of its range
//!   supersedes it or the program returns.
//!
//! Writes carry ranges; a dependency is one address plus a version. On
//! versioned data it is `(a, v)` with `a` the base address of the
//! producer's `dst`; on unversioned data it is an address inside one
//! producer's `dst`. The version table maps a base address to its version
//! counter only. A lookup of `(a, v)` checks the scoreboard and the
//! execution table first (their entries carry `dst` and version), then the
//! counter (`v` not after the counter: done), and otherwise waits on a pending entry
//! keyed `(a, v)` for the producer's invoke. Hazards are RAW only; hardware
//! counts no readers.
//!
//! A tag names one entry. The model resolves dependencies to tags at invoke
//! and wakes consumers at completion through per-entry waiter lists, which
//! is observably equal to a tag broadcast. A task keeps one [`TaskId`] from
//! invoke to retire; the scoreboard and the execution table are two
//! capacity counts over the same entries, selected by [`Phase`].
//!
//! The tables have no notion of time. Every mutation returns [`Effects`],
//! which the caller turns into events.
//!
//! Epochs: a version is `epoch << EPOCH_SHIFT | count`. Each Neutrino has a
//! current epoch, set by a program start. A base without a version table
//! entry reads as the epoch base `epoch << EPOCH_SHIFT`, both for version
//! assignment and for the lookup. Epochs compare as serial numbers of 8
//! bits ([`version_order`]): the numbering wraps from 255 to 0, and two
//! epochs that a lookup compares must be less than 128 apart. So a key of
//! an earlier epoch is done (unless its producer is still in flight), and
//! a key of a later epoch waits on a pending entry. With epoch 0
//! everywhere a base without an entry reads as 0, as without epochs.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::addr::Range;
use crate::dst::{Dst, FULL_MASK};
use crate::remote::Awaiter;
use crate::slab::Slab;

/// The bit of a version where its epoch starts; the count is below it.
pub const EPOCH_SHIFT: u32 = 24;
/// The highest epoch.
pub const EPOCH_MAX: u64 = 255;
/// The bits of the count of a version.
pub const COUNT_MASK: u64 = (1 << EPOCH_SHIFT) - 1;

/// The epoch of version `v`.
pub fn epoch_of(v: u64) -> u64 {
    v >> EPOCH_SHIFT
}

/// The serial order of epochs `a` and `b`: with d = (a - b) mod 256, `a`
/// is after `b` for d in 1..=127, before it for d in 129..=255, and equal
/// for d = 0. `None` (no order) for d = 128.
pub fn epoch_order(a: u64, b: u64) -> Option<Ordering> {
    match a.wrapping_sub(b) & EPOCH_MAX {
        0 => Some(Ordering::Equal),
        1..=127 => Some(Ordering::Greater),
        128 => None,
        _ => Some(Ordering::Less),
    }
}

/// The order of versions `a` and `b`: the serial order of their epochs
/// ([`epoch_order`]), and in one epoch the order of their counts.
pub fn version_order(a: u64, b: u64) -> Option<Ordering> {
    match epoch_order(epoch_of(a), epoch_of(b))? {
        Ordering::Equal => Some((a & COUNT_MASK).cmp(&(b & COUNT_MASK))),
        o => Some(o),
    }
}

/// A task entry, stable from invoke to retire. The task instance ID on the
/// issue port.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(pub u32);

/// A pending table entry.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PendingId(pub u32);

/// A pending local program wait.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WaitId(pub u32);

/// A dependency key: what a consumer, a wait or a remote request names.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dep {
    /// Version `version` of the versioned range whose base address is
    /// `addr`.
    Versioned {
        /// The base address of the producer's `dst`.
        addr: u64,
        /// The version.
        version: u64,
    },
    /// An address inside one unversioned producer's `dst`.
    Unversioned(u64),
}

impl Dep {
    /// The named address.
    pub fn addr(&self) -> u64 {
        match *self {
            Dep::Versioned { addr, .. } | Dep::Unversioned(addr) => addr,
        }
    }

    /// The named version; `None` for an unversioned address.
    pub fn version(&self) -> Option<u64> {
        match *self {
            Dep::Versioned { version, .. } => Some(version),
            Dep::Unversioned(_) => None,
        }
    }
}

/// Names one entry of the dependency tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tag {
    /// A pending entry.
    Pending(PendingId),
    /// A task entry in the scoreboard or the execution table.
    Task(TaskId),
}

/// The phase of a task entry. `Scoreboard` counts against the scoreboard
/// capacity, `Executing` and `Done` against the execution table capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Invoked; waits for its depcount to reach zero.
    Scoreboard,
    /// Issued and executing.
    Executing,
    /// Completed, unversioned; retires when superseded or at program
    /// return.
    Done,
}

/// The dependency-relevant part of a task descriptor. `deps` are the local
/// dependencies; remote ones are counted with `external_deps` at invoke.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSpec {
    /// The range the task writes.
    pub dst: Range,
    /// `dst` is versioned: the tables assign the next version of its base.
    pub versioned: bool,
    /// The granules of `dst` the task holds ([`FULL_MASK`]: all; Neutrino
    /// spec, granules and destination ranges).
    pub mask: u32,
    /// An untracked task (Neutrino spec Section 3.1): no version, no
    /// lookup finds it, it adopts and supersedes nothing, and it retires at
    /// completion. The tables count it until it retires (the fence).
    pub untracked: bool,
    /// The local dependencies.
    pub deps: Vec<Dep>,
}

impl TaskSpec {
    /// An unversioned task.
    pub fn new(dst: Range, deps: Vec<Dep>) -> Self {
        Self {
            dst,
            versioned: false,
            mask: FULL_MASK,
            untracked: false,
            deps,
        }
    }

    /// A versioned task.
    pub fn versioned(dst: Range, deps: Vec<Dep>) -> Self {
        Self {
            dst,
            versioned: true,
            mask: FULL_MASK,
            untracked: false,
            deps,
        }
    }

    /// The same task with the mask `mask`.
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.mask = mask;
        self
    }

    /// The masked destination range.
    pub fn dst(&self) -> Dst {
        Dst {
            span: self.dst,
            mask: self.mask,
        }
    }

    /// An untracked task.
    pub fn untracked(dst: Range, deps: Vec<Dep>) -> Self {
        Self {
            dst,
            versioned: false,
            mask: FULL_MASK,
            untracked: true,
            deps,
        }
    }
}

/// Capacities and alignment of the dependency tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepConfig {
    /// Scoreboard entries (invoked, not issued).
    pub scoreboard_capacity: usize,
    /// Execution table entries (executing or done).
    pub execution_capacity: usize,
    /// Pending table entries.
    pub pending_capacity: usize,
    /// Version table entries (versioned base addresses).
    pub version_capacity: usize,
    /// Owner-side inbound ref table entries.
    pub inbound_ref_capacity: usize,
    /// Execution table entries kept for issue (Neutrino spec Section 6.2): an invoke
    /// enters the scoreboard only while scoreboard and execution table
    /// entries together stay below `execution_capacity - issue_headroom`.
    pub issue_headroom: usize,
    /// Range alignment in bytes; a power of two.
    pub granule: u64,
}

/// A request the tables refuse. Capacity variants are stalls or rejects for
/// the caller to handle; the others are software errors.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepError {
    /// A destination range that holds no granule (bound 0; the spec's
    /// error `Empty`).
    Empty(Range),
    /// A destination range whose ends are not multiples of the granule.
    /// r2-sim only: the spec's ranges are in granules.
    Misaligned(Range),
    /// A dependency of the invoked task resolves to a pending entry that
    /// the task itself would adopt: the task would wait on itself.
    SelfDependency {
        /// The dependency address.
        dep: u64,
        /// The `dst` of the invoked task.
        dst: Range,
    },
    /// Two live versioned ranges share a base address with a different
    /// bound or mask.
    BaseConflict {
        /// The shared base address.
        base: u64,
        /// The bound on record.
        bound: u64,
        /// The bound of the new range.
        new_bound: u64,
        /// The mask on record.
        mask: u32,
        /// The mask of the new range.
        new_mask: u32,
    },
    /// The version an invoke gets is still held by a task in the tables
    /// (the range was released while its producer was in flight, then
    /// reused).
    VersionInFlight {
        /// The base address.
        addr: u64,
        /// The version.
        version: u64,
    },
    /// The next version of a base would carry its count into the next
    /// epoch (the count reached `COUNT_MASK`).
    VersionOverflow {
        /// The base address.
        addr: u64,
        /// The version counter.
        counter: u64,
    },
    /// At a program start, a pending entry of a versioned key whose epoch
    /// is not the start's epoch and not after it: no producer can adopt
    /// it.
    StalePending {
        /// The base address of the key.
        addr: u64,
        /// The version of the key.
        version: u64,
        /// The epoch of the start.
        epoch: u64,
    },
    /// A lookup compared a key with the counter of its base, and their
    /// epochs are 128 apart: no order.
    EpochWindow {
        /// The base address of the key.
        addr: u64,
        /// The version of the key.
        version: u64,
        /// The counter of the base (the epoch base without an entry).
        counter: u64,
    },
    /// A lookup found a key of an earlier epoch done by the counter, but
    /// its base never reached the key's count in that epoch at this
    /// Neutrino.
    Unproduced {
        /// The base address of the key.
        addr: u64,
        /// The version of the key.
        version: u64,
        /// The highest version of the base in the key's epoch; `None`
        /// when the base had no write in that epoch.
        highest: Option<u64>,
    },
    /// Scoreboard and execution table leave no issue headroom (stall).
    NoHeadroom,
    /// The scoreboard has no free slot for this invoke (stall).
    ScoreboardFull,
    /// The execution table has no free slot for the issue (stall).
    ExecutionFull,
    /// The pending table has no free slot for a new entry.
    PendingFull,
    /// The version table has no free slot for a new base address.
    VersionsFull,
    /// The owner-side inbound ref table is full.
    InboundRefsFull,
}

/// A condition that is legal but suspicious.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Warning {
    /// `addr` is in the `dst` of `count` executing unversioned entries at
    /// one lookup: writers that the program did not order. Done entries and
    /// entries still in the scoreboard do not count.
    MultipleCovers {
        /// The looked-up address.
        addr: u64,
        /// Executing entries that contain `addr`.
        count: u32,
    },
    /// An unversioned key and a versioned write meet at one address: no
    /// write of the versioned range adopts or resolves the unversioned
    /// key, so its waiters may never wake. Raised when an unversioned
    /// lookup makes a pending entry for an address that is the base of a
    /// version table entry or that a versioned task in the tables holds,
    /// and when a versioned invoke's `dst` holds the address of an
    /// unversioned pending entry.
    UnversionedWait {
        /// The address of the unversioned key.
        addr: u64,
        /// The versioned key: the counter of the base, or the key of the
        /// versioned task.
        versioned: Dep,
    },
}

/// What a mutation released or changed. The caller turns it into events.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Effects {
    /// Scoreboard entries whose depcount reached zero.
    pub ready: Vec<TaskId>,
    /// Local program waits that are satisfied.
    pub waits_done: Vec<WaitId>,
    /// Remote requesters to send `Resolve`.
    pub resolves: Vec<Awaiter>,
    /// Task entries that retired.
    pub retired: Vec<TaskId>,
    /// Pending entries adopted by an invoke; their slots are free.
    pub adopted: Vec<PendingId>,
    /// Pending entries this mutation created, with their keys.
    pub pending_created: Vec<(PendingId, Dep)>,
    /// Suspicious conditions found during the mutation.
    pub warnings: Vec<Warning>,
}

impl Effects {
    /// Appends `other` to `self`.
    pub fn merge(&mut self, other: Effects) {
        self.ready.extend(other.ready);
        self.waits_done.extend(other.waits_done);
        self.resolves.extend(other.resolves);
        self.retired.extend(other.retired);
        self.adopted.extend(other.adopted);
        self.pending_created.extend(other.pending_created);
        self.warnings.extend(other.warnings);
    }
}

/// Result of an invoke.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invoked {
    /// The new task entry.
    pub task: TaskId,
    /// The version of a versioned `dst`.
    pub version: Option<u64>,
    /// Depcount is zero; the task can issue.
    pub ready: bool,
    /// What the invoke released or changed.
    pub effects: Effects,
}

/// Result of a local wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitStatus {
    /// The key is done.
    Done,
    /// The key is not done; the ID appears in [`Effects::waits_done`] when
    /// it is.
    Pending(WaitId),
}

/// Result of a remote registration at the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Registration {
    /// The key is done; no `Resolve` follows.
    Done,
    /// An inbound ref is recorded; the requester appears in
    /// [`Effects::resolves`] at completion.
    Registered,
}

/// An open pending entry, for end-of-run reporting.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenPending {
    /// The pending entry.
    pub id: PendingId,
    /// The address of the key.
    pub addr: u64,
    /// The version of a versioned key.
    pub version: Option<u64>,
    /// Number of consumers, waits and inbound refs waiting on the entry.
    pub waiters: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Waiter {
    Task(TaskId),
    Wait(WaitId),
    Remote(u32),
}

#[derive(Debug, Clone)]
struct Task {
    dst: Range,
    /// The granules of `dst` it holds.
    mask: u32,
    /// The version of a versioned `dst`.
    version: Option<u64>,
    phase: Phase,
    depcount: u32,
    waiters: Vec<Waiter>,
    /// Invoke order, to tell the newest of several covers.
    seq: u64,
    /// Unversioned: a newer producer covers `dst`; retires at completion.
    superseded: bool,
    /// Untracked: no lookup finds it; retires at completion.
    untracked: bool,
}

#[derive(Debug, Clone)]
struct PendingEntry {
    key: Dep,
    waiters: Vec<Waiter>,
}

/// One versioned base address: the version counter. The bound and mask of
/// the range are a simulator check (two live ranges with one base and a
/// different bound or mask are a software error); they are not hardware
/// state.
#[derive(Debug, Clone, Copy)]
struct VersionEntry {
    counter: u64,
    bound: Option<(u64, u32)>,
}

#[derive(Debug, Clone)]
struct InboundRef {
    waiter: Awaiter,
    remaining: u32,
}

/// Where a lookup of a key resolves.
enum Found {
    /// Done; nothing to wait for.
    Done,
    /// These entries (some may be done unversioned tasks).
    Tags(Vec<Tag>),
    /// Nothing produces the key yet: a new pending entry.
    New,
}

/// Per-dependency plan of an invoke, computed before any mutation.
enum DepPlan {
    Done,
    Covers(Vec<Tag>),
    /// Index into the invoke's list of new pending keys.
    NewPending(usize),
}

/// What an invoke does to the tables, planned before any change.
struct InvokePlan {
    /// Per dep: done, its covers or its new pending entry.
    plans: Vec<DepPlan>,
    /// Keys of the new pending entries the deps need.
    new_pending: Vec<Dep>,
    /// Pending entries the task adopts.
    adopt: Vec<usize>,
    /// The version of a versioned `dst`.
    version: Option<u64>,
    /// Unversioned tasks whose `dst` the task covers.
    supersede: Vec<TaskId>,
}

/// The pending table, scoreboard, execution table and version table of one
/// Neutrino, plus its owner-side inbound ref table.
#[derive(Debug, Clone)]
pub struct DepTables {
    cfg: DepConfig,
    tasks: Slab<Task>,
    /// Versioned tasks in the tables by (base, version): the lookup of the
    /// entries' `(dst, version)` fields.
    in_flight: HashMap<(u64, u64), TaskId>,
    pending: Slab<PendingEntry>,
    /// Pending entries by key.
    pending_keys: HashMap<Dep, PendingId>,
    versions: HashMap<u64, VersionEntry>,
    waits: Slab<u32>,
    inbound_refs: Slab<InboundRef>,
    scoreboard_len: usize,
    execution_len: usize,
    next_seq: u64,
    scoreboard_peak: usize,
    version_peak: usize,
    /// Untracked tasks that have not retired.
    untracked_len: usize,
    /// Fences (program waits on the key 0) that wait for `untracked_len`
    /// to reach 0.
    fences: Vec<WaitId>,
    /// The current epoch (set by a program start).
    epoch: u64,
    /// The highest version per (epoch, base) that a version assignment
    /// or `nu.version` reached: a simulator check, not hardware state.
    produced: HashMap<(u64, u64), u64>,
}

impl DepTables {
    /// Builds empty tables; panics if `cfg.granule` is not a power of two.
    pub fn new(cfg: DepConfig) -> Self {
        assert!(
            cfg.granule.is_power_of_two(),
            "granule must be a power of two"
        );
        Self {
            cfg,
            tasks: Slab::default(),
            in_flight: HashMap::new(),
            pending: Slab::default(),
            pending_keys: HashMap::new(),
            versions: HashMap::new(),
            waits: Slab::default(),
            inbound_refs: Slab::default(),
            scoreboard_len: 0,
            execution_len: 0,
            next_seq: 0,
            scoreboard_peak: 0,
            version_peak: 0,
            untracked_len: 0,
            fences: Vec::new(),
            epoch: 0,
            produced: HashMap::new(),
        }
    }

    /// The current epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The counter of a base without a version table entry.
    fn epoch_base(&self) -> u64 {
        self.epoch << EPOCH_SHIFT
    }

    /// Records that base `addr` reached version `v`.
    fn record_produced(&mut self, addr: u64, v: u64) {
        let h = self
            .produced
            .entry((epoch_of(v) & EPOCH_MAX, addr))
            .or_insert(v);
        if version_order(v, *h) == Some(Ordering::Greater) {
            *h = v;
        }
    }

    /// The configuration the tables were built with.
    pub fn config(&self) -> &DepConfig {
        &self.cfg
    }

    /// Task entries in the scoreboard phase.
    pub fn scoreboard_len(&self) -> usize {
        self.scoreboard_len
    }

    /// Task entries in the execution table (executing or done).
    pub fn execution_len(&self) -> usize {
        self.execution_len
    }

    /// Open pending entries.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Version table entries in use.
    pub fn versions_len(&self) -> usize {
        self.versions.len()
    }

    /// The most version table entries held at once.
    pub fn version_peak(&self) -> usize {
        self.version_peak
    }

    /// The version counter of base address `addr`; `None` without an
    /// entry.
    pub fn version_counter(&self, addr: u64) -> Option<u64> {
        self.versions.get(&addr).map(|e| e.counter)
    }

    /// Entries in use in the owner-side inbound ref table.
    pub fn inbound_refs_len(&self) -> usize {
        self.inbound_refs.len()
    }

    /// Local program waits not yet satisfied.
    pub fn pending_waits(&self) -> usize {
        self.waits.len()
    }

    /// `None` when the task retired.
    pub fn phase(&self, t: TaskId) -> Option<Phase> {
        self.tasks.get(t.0 as usize).map(|e| e.phase)
    }

    /// The remaining depcount of a task; `None` when the task retired.
    pub fn depcount(&self, t: TaskId) -> Option<u32> {
        self.tasks.get(t.0 as usize).map(|e| e.depcount)
    }

    /// Pending entries still open, for end-of-run reporting.
    pub fn open_pending(&self) -> impl Iterator<Item = OpenPending> + '_ {
        self.pending.iter().map(|(i, p)| OpenPending {
            id: PendingId(i as u32),
            addr: p.key.addr(),
            version: p.key.version(),
            waiters: p.waiters.len(),
        })
    }

    /// Task entries whose `dst` overlaps `range`, with their phase. For
    /// diagnostics such as the race check at a load; a linear scan.
    pub fn overlapping<'a>(
        &'a self,
        range: &'a Range,
    ) -> impl Iterator<Item = (TaskId, Phase, Range)> + 'a {
        self.tasks
            .iter()
            .filter(|(_, t)| t.held().overlaps(self.cfg.granule, range))
            .map(|(i, t)| (TaskId(i as u32), t.phase, t.dst))
    }

    /// The unversioned tracked task entries whose `dst` contains `addr`
    /// (no lookup finds an untracked task).
    fn covers(&self, addr: u64) -> Vec<Tag> {
        // Linear scan, like the hardware CAM. Fine at table sizes of a few
        // hundred; add an interval index if they grow to thousands.
        self.tasks
            .iter()
            .filter(|(_, t)| {
                t.version.is_none() && !t.untracked && t.held().holds(self.cfg.granule, addr)
            })
            .map(|(i, _)| Tag::Task(TaskId(i as u32)))
            .collect()
    }

    /// Looks `key` up: an in-flight producer in the scoreboard or the
    /// execution table, then the version counter (a versioned key), then
    /// the pending table. A base without an entry reads as the epoch base.
    /// The key is done when its version is not after the counter.
    /// `EpochWindow` when the two have no order; `Unproduced` when a key of
    /// an earlier epoch is done by the counter although its base never
    /// reached it in that epoch.
    fn find(&self, key: &Dep) -> Result<Found, DepError> {
        if let Some(&p) = self.pending_keys.get(key) {
            return Ok(Found::Tags(vec![Tag::Pending(p)]));
        }
        Ok(match *key {
            Dep::Unversioned(a) => {
                let tags = self.bind(self.covers(a));
                if tags.is_empty() {
                    Found::New
                } else {
                    Found::Tags(tags)
                }
            }
            Dep::Versioned { addr, version } => {
                if let Some(&t) = self.in_flight.get(&(addr, version)) {
                    return Ok(Found::Tags(vec![Tag::Task(t)]));
                }
                let counter = self
                    .versions
                    .get(&addr)
                    .map_or(self.epoch_base(), |e| e.counter);
                match version_order(version, counter) {
                    None => {
                        return Err(DepError::EpochWindow {
                            addr,
                            version,
                            counter,
                        })
                    }
                    Some(Ordering::Greater) => Found::New,
                    Some(_) => {
                        self.check_produced(addr, version)?;
                        Found::Done
                    }
                }
            }
        })
    }

    /// A key of an earlier epoch that a lookup finds done: its base must
    /// have reached its version in that epoch.
    fn check_produced(&self, addr: u64, version: u64) -> Result<(), DepError> {
        let e = epoch_of(version);
        if epoch_order(e, self.epoch) != Some(Ordering::Less) {
            return Ok(());
        }
        let highest = self.produced.get(&(e & EPOCH_MAX, addr)).copied();
        if highest.is_some_and(|h| version_order(version, h) != Some(Ordering::Greater)) {
            return Ok(());
        }
        Err(DepError::Unproduced {
            addr,
            version,
            highest,
        })
    }

    /// Enters a task in the scoreboard. `external_deps` counts the
    /// dependencies in other Neutrinos; each is cleared with
    /// [`DepTables::external_resolved`]. No state changes on error.
    pub fn invoke(&mut self, spec: TaskSpec, external_deps: u32) -> Result<Invoked, DepError> {
        self.invoke_in(spec, external_deps)
    }

    /// True while an invoke may enter: scoreboard and execution table
    /// entries together below `execution_capacity - issue_headroom`.
    pub fn headroom_free(&self) -> bool {
        self.scoreboard_len + self.execution_len
            < self
                .cfg
                .execution_capacity
                .saturating_sub(self.cfg.issue_headroom)
    }

    /// Free scoreboard entries.
    pub fn scoreboard_free(&self) -> usize {
        self.cfg
            .scoreboard_capacity
            .saturating_sub(self.scoreboard_len)
    }

    /// The most scoreboard entries held at once.
    pub fn scoreboard_peak(&self) -> usize {
        self.scoreboard_peak
    }

    /// Untracked tasks that have not retired.
    pub fn untracked_len(&self) -> usize {
        self.untracked_len
    }

    /// The version a versioned invoke of `dst` gets.
    fn plan_version(&self, spec: &TaskSpec) -> Result<Option<u64>, DepError> {
        debug_assert!(
            !(spec.untracked && spec.versioned),
            "a versioned untracked task"
        );
        if !spec.versioned || spec.untracked {
            return Ok(None);
        }
        let addr = spec.dst.base;
        let mask = spec.dst().canonical_mask(self.cfg.granule);
        let v = match self.versions.get(&addr) {
            Some(e) => {
                if let Some((bound, m)) = e.bound.filter(|b| *b != (spec.dst.bound, mask)) {
                    return Err(DepError::BaseConflict {
                        base: addr,
                        bound,
                        new_bound: spec.dst.bound,
                        mask: m,
                        new_mask: mask,
                    });
                }
                if e.counter & COUNT_MASK == COUNT_MASK {
                    return Err(DepError::VersionOverflow {
                        addr,
                        counter: e.counter,
                    });
                }
                e.counter + 1
            }
            None if self.versions.len() >= self.cfg.version_capacity => {
                return Err(DepError::VersionsFull)
            }
            None => self.epoch_base() + 1,
        };
        if self.in_flight.contains_key(&(addr, v)) {
            return Err(DepError::VersionInFlight { addr, version: v });
        }
        Ok(Some(v))
    }

    /// Checks an invoke against the tables and plans its lookups and
    /// adoptions; changes nothing (a failed invoke leaves no trace).
    fn plan_invoke(&self, spec: &TaskSpec) -> Result<InvokePlan, DepError> {
        self.check_range(&spec.dst)?;
        if self.scoreboard_len >= self.cfg.scoreboard_capacity {
            return Err(DepError::ScoreboardFull);
        }
        if !self.headroom_free() {
            return Err(DepError::NoHeadroom);
        }
        let version = self.plan_version(spec)?;
        let own_key = version.map(|v| Dep::Versioned {
            addr: spec.dst.base,
            version: v,
        });

        // Plan every lookup against the tables and against the pending
        // entries that earlier deps of this invoke create.
        let mut plans = Vec::with_capacity(spec.deps.len());
        let mut new_pending: Vec<Dep> = Vec::new();
        for d in &spec.deps {
            // An untracked task adopts nothing, so it never waits on
            // itself.
            let self_dep = match *d {
                _ if spec.untracked => false,
                _ if Some(*d) == own_key => true,
                Dep::Unversioned(a) if version.is_none() => {
                    spec.dst().holds(self.cfg.granule, a) && self.covers(a).is_empty()
                }
                _ => false,
            };
            if self_dep {
                return Err(DepError::SelfDependency {
                    dep: d.addr(),
                    dst: spec.dst,
                });
            }
            let plan = match new_pending.iter().position(|np| np == d) {
                Some(k) => DepPlan::NewPending(k),
                None => match self.find(d)? {
                    Found::New => {
                        new_pending.push(*d);
                        DepPlan::NewPending(new_pending.len() - 1)
                    }
                    Found::Done => DepPlan::Done,
                    Found::Tags(tags) => DepPlan::Covers(tags),
                },
            };
            plans.push(plan);
        }
        let mut adopt = Vec::new();
        for (i, p) in self.pending.iter() {
            if !spec.untracked && adopts(&spec.dst(), self.cfg.granule, own_key, &p.key) {
                adopt.push(i);
            }
        }
        let room = self.cfg.pending_capacity.saturating_sub(self.pending.len());
        if new_pending.len() > room {
            return Err(DepError::PendingFull);
        }
        let supersede = if version.is_none() && !spec.untracked {
            self.tasks
                .iter()
                .filter(|(_, t)| {
                    t.version.is_none()
                        && !t.untracked
                        && spec.dst().covers(self.cfg.granule, &t.held())
                })
                .map(|(i, _)| TaskId(i as u32))
                .collect()
        } else {
            Vec::new()
        };
        Ok(InvokePlan {
            plans,
            new_pending,
            adopt,
            version,
            supersede,
        })
    }

    fn invoke_in(&mut self, spec: TaskSpec, external_deps: u32) -> Result<Invoked, DepError> {
        let InvokePlan {
            plans,
            new_pending,
            adopt,
            version,
            supersede,
        } = self.plan_invoke(&spec)?;

        // Commit.
        let mut effects = Effects::default();
        self.next_seq += 1;
        let task = TaskId(self.tasks.insert(Task {
            dst: spec.dst,
            mask: spec.mask,
            version,
            phase: Phase::Scoreboard,
            depcount: external_deps,
            seq: self.next_seq,
            waiters: Vec::new(),
            superseded: false,
            untracked: spec.untracked,
        }) as u32);
        self.scoreboard_len += 1;
        self.untracked_len += spec.untracked as usize;
        self.scoreboard_peak = self.scoreboard_peak.max(self.scoreboard_len);
        if let Some(v) = version {
            let mask = spec.dst().canonical_mask(self.cfg.granule);
            self.versions.insert(
                spec.dst.base,
                VersionEntry {
                    counter: v,
                    bound: Some((spec.dst.bound, mask)),
                },
            );
            self.version_peak = self.version_peak.max(self.versions.len());
            self.in_flight.insert((spec.dst.base, v), task);
            self.record_produced(spec.dst.base, v);
        }

        if let Some(v) = version {
            self.warn_unversioned_pending(spec.dst(), v, &mut effects);
        }
        let new_pending_ids: Vec<PendingId> = new_pending
            .iter()
            .map(|k| self.insert_pending(*k))
            .collect();
        for k in &new_pending {
            self.warn_versioned_at(*k, &mut effects);
        }
        effects.pending_created.extend(
            new_pending_ids
                .iter()
                .copied()
                .zip(new_pending.iter().copied()),
        );
        for (i, plan) in plans.into_iter().enumerate() {
            let d = spec.deps[i];
            let tags = match plan {
                DepPlan::Done => continue,
                DepPlan::NewPending(k) => vec![Tag::Pending(new_pending_ids[k])],
                DepPlan::Covers(tags) => tags,
            };
            self.warn_covers(&tags, d, &mut effects);
            for tag in tags {
                self.attach(task, tag);
            }
        }

        for i in adopt {
            let p = self.pending.remove(i).expect("pending");
            self.pending_keys.remove(&p.key);
            self.task_mut(task).waiters.extend(p.waiters);
            effects.adopted.push(PendingId(i as u32));
        }
        for t in supersede {
            if self.task(t).phase == Phase::Done {
                self.retire(t, &mut effects);
            } else {
                self.task_mut(t).superseded = true;
            }
        }
        let ready = self.task(task).depcount == 0;
        Ok(Invoked {
            task,
            version,
            ready,
            effects,
        })
    }

    /// Clears one external dependency of `t`. Returns true when the task
    /// became ready.
    pub fn external_resolved(&mut self, t: TaskId) -> bool {
        let e = self.task_mut(t);
        assert_eq!(e.phase, Phase::Scoreboard, "external dep of an issued task");
        assert!(e.depcount > 0, "external_resolved at depcount 0");
        e.depcount -= 1;
        e.depcount == 0
    }

    /// Moves a ready task from the scoreboard to the execution table.
    /// `ExecutionFull` is a stall.
    pub fn issue(&mut self, t: TaskId) -> Result<(), DepError> {
        let e = self.task(t);
        assert_eq!(e.phase, Phase::Scoreboard, "issue of an issued task");
        assert_eq!(e.depcount, 0, "issue of a task that is not ready");
        if self.execution_len >= self.cfg.execution_capacity {
            return Err(DepError::ExecutionFull);
        }
        self.task_mut(t).phase = Phase::Executing;
        self.scoreboard_len -= 1;
        self.execution_len += 1;
        Ok(())
    }

    /// Marks an executing task done and wakes its waiters. A versioned or
    /// untracked task retires (its version counter already says done, or no
    /// lookup can find it); an unversioned one retires only if superseded.
    pub fn complete(&mut self, t: TaskId) -> Effects {
        let e = self.task_mut(t);
        assert_eq!(
            e.phase,
            Phase::Executing,
            "complete of a task not executing"
        );
        e.phase = Phase::Done;
        let waiters = std::mem::take(&mut e.waiters);
        let (versioned, superseded) = (e.version.is_some(), e.superseded);
        let untracked = e.untracked;
        let mut out = Effects::default();
        self.wake(waiters, &mut out);
        if versioned || superseded || untracked {
            self.retire(t, &mut out);
        }
        out
    }

    /// `nu.version`: sets the counter of base address `addr`, creating the
    /// entry.
    pub fn set_version(&mut self, addr: u64, v: u64) -> Result<(), DepError> {
        let full = self.versions.len() >= self.cfg.version_capacity;
        match self.versions.get_mut(&addr) {
            Some(e) => e.counter = v,
            None if full => return Err(DepError::VersionsFull),
            None => {
                self.versions.insert(
                    addr,
                    VersionEntry {
                        counter: v,
                        bound: None,
                    },
                );
                self.version_peak = self.version_peak.max(self.versions.len());
            }
        }
        self.record_produced(addr, v);
        Ok(())
    }

    /// `nu.vreset` (local or from a `VersionReset` message): frees the entry of
    /// base address `addr`; an absent entry is a no-op. A producer still in
    /// the tables keeps resolving its consumers by its own entry. The epoch
    /// does not change.
    pub fn vreset(&mut self, addr: u64) {
        self.versions.remove(&addr);
    }

    /// `nu.vreset_all`: clears the version table. The epoch does not
    /// change.
    pub fn vreset_all(&mut self) {
        self.versions.clear();
    }

    /// A program start with epoch `epoch`: clears the version table and
    /// sets the epoch; pending and task entries stay. `StalePending` when a
    /// pending entry names a versioned key whose epoch is not `epoch` and
    /// not after it (nothing changes then). The records of the highest
    /// versions keep only the epochs before `epoch`: after the start, the
    /// numbers of the others are the current epoch, later epochs or the
    /// epoch with no order. Panics on an epoch above [`EPOCH_MAX`].
    pub fn start(&mut self, epoch: u64) -> Result<(), DepError> {
        assert!(epoch <= EPOCH_MAX, "epoch {epoch} above {EPOCH_MAX}");
        let stale = self.pending.iter().find_map(|(_, p)| match p.key {
            Dep::Versioned { addr, version }
                if !matches!(
                    epoch_order(epoch_of(version), epoch),
                    Some(Ordering::Equal | Ordering::Greater)
                ) =>
            {
                Some((addr, version))
            }
            _ => None,
        });
        if let Some((addr, version)) = stale {
            return Err(DepError::StalePending {
                addr,
                version,
                epoch,
            });
        }
        self.produced
            .retain(|&(e, _), _| epoch_order(e, epoch) == Some(Ordering::Less));
        self.versions.clear();
        self.epoch = epoch;
        Ok(())
    }

    /// The program returned: done unversioned entries retire, and executing
    /// ones retire at completion.
    pub fn program_returned(&mut self) -> Effects {
        let mut out = Effects::default();
        let unversioned: Vec<(TaskId, Phase)> = self
            .tasks
            .iter()
            .filter(|(_, t)| t.version.is_none() && !t.untracked)
            .map(|(i, t)| (TaskId(i as u32), t.phase))
            .collect();
        for (t, phase) in unversioned {
            match phase {
                Phase::Done => self.retire(t, &mut out),
                _ => self.task_mut(t).superseded = true,
            }
        }
        out
    }

    /// A local program wait: pending until `key` is done. Creates a
    /// pending entry if nothing produces `key` yet.
    pub fn wait(&mut self, key: Dep) -> Result<(WaitStatus, Effects), DepError> {
        let (pending, out) = self.lookup_pending(key, false)?;
        if pending.is_empty() {
            return Ok((WaitStatus::Done, out));
        }
        let id = WaitId(self.waits.insert(pending.len() as u32) as u32);
        self.add_waiter(&pending, Waiter::Wait(id));
        Ok((WaitStatus::Pending(id), out))
    }

    /// A fence (a program wait on the key 0): pending until every untracked
    /// task in the tables has retired. Does no lookup.
    pub fn fence(&mut self) -> WaitStatus {
        if self.untracked_len == 0 {
            return WaitStatus::Done;
        }
        let id = WaitId(self.waits.insert(1) as u32);
        self.fences.push(id);
        WaitStatus::Pending(id)
    }

    /// Owner side of a remote `Await`. Records an inbound ref when `key`
    /// is not done. `InboundRefsFull` and `PendingFull` mean reject.
    pub fn register_remote(
        &mut self,
        key: Dep,
        waiter: Awaiter,
    ) -> Result<(Registration, Effects), DepError> {
        let (pending, out) = self.lookup_pending(key, true)?;
        if pending.is_empty() {
            return Ok((Registration::Done, out));
        }
        let id = self.inbound_refs.insert(InboundRef {
            waiter,
            remaining: pending.len() as u32,
        }) as u32;
        self.add_waiter(&pending, Waiter::Remote(id));
        Ok((Registration::Registered, out))
    }

    /// The entries of `key` that are not done; a new pending entry if
    /// nothing produces it. No state changes on error.
    fn lookup_pending(&mut self, key: Dep, remote: bool) -> Result<(Vec<Tag>, Effects), DepError> {
        let mut out = Effects::default();
        let pending: Vec<Tag> = match self.find(&key)? {
            Found::Done => Vec::new(),
            Found::Tags(tags) => {
                self.warn_covers(&tags, key, &mut out);
                tags.into_iter()
                    .filter(|t| match t {
                        Tag::Task(t) => self.task(*t).phase != Phase::Done,
                        Tag::Pending(_) => true,
                    })
                    .collect()
            }
            Found::New => {
                if remote && self.inbound_refs.len() >= self.cfg.inbound_ref_capacity {
                    return Err(DepError::InboundRefsFull);
                }
                let p = self.new_pending(key, &mut out)?;
                return Ok((vec![Tag::Pending(p)], out));
            }
        };
        if remote && !pending.is_empty() && self.inbound_refs.len() >= self.cfg.inbound_ref_capacity
        {
            return Err(DepError::InboundRefsFull);
        }
        Ok((pending, out))
    }

    /// The covers a lookup binds to: the newest (by invoke order), and every
    /// other cover that is not done (unordered writers, or ordered ones still
    /// running). Older done covers are earlier values of a recycled range.
    fn bind(&self, tags: Vec<Tag>) -> Vec<Tag> {
        if tags.len() <= 1 {
            return tags;
        }
        let newest = tags
            .iter()
            .filter_map(|t| match t {
                Tag::Task(id) => Some((self.task(*id).seq, *t)),
                Tag::Pending(_) => None,
            })
            .max_by_key(|(seq, _)| *seq)
            .map(|(_, t)| t);
        tags.into_iter()
            .filter(|t| match t {
                Tag::Task(id) => Some(*t) == newest || self.task(*id).phase != Phase::Done,
                Tag::Pending(_) => true,
            })
            .collect()
    }

    /// Covering writers that are executing (issued, not done). Two or more
    /// at one lookup are unordered writers (Neutrino spec Section 7.4); covers still in the
    /// scoreboard are ordered by their deps and do not count.
    fn executing_count(&self, tags: &[Tag]) -> u32 {
        tags.iter()
            .filter(|t| match t {
                Tag::Task(t) => self.task(*t).phase == Phase::Executing,
                Tag::Pending(_) => false,
            })
            .count() as u32
    }

    fn add_waiter(&mut self, tags: &[Tag], w: Waiter) {
        for tag in tags {
            match tag {
                Tag::Task(t) => self.task_mut(*t).waiters.push(w),
                Tag::Pending(p) => self
                    .pending
                    .get_mut(p.0 as usize)
                    .expect("pending")
                    .waiters
                    .push(w),
            }
        }
    }

    /// Consumer `c` looks up `tag`; it waits unless `tag` is done.
    fn attach(&mut self, c: TaskId, tag: Tag) {
        let waits = match tag {
            Tag::Task(p) => {
                let e = self.task_mut(p);
                if e.phase != Phase::Done {
                    e.waiters.push(Waiter::Task(c));
                    true
                } else {
                    false
                }
            }
            Tag::Pending(p) => {
                self.pending_mut(p).waiters.push(Waiter::Task(c));
                true
            }
        };
        if waits {
            self.task_mut(c).depcount += 1;
        }
    }

    fn wake(&mut self, waiters: Vec<Waiter>, out: &mut Effects) {
        for w in waiters {
            match w {
                Waiter::Task(c) => {
                    let e = self.task_mut(c);
                    e.depcount -= 1;
                    if e.depcount == 0 {
                        out.ready.push(c);
                    }
                }
                Waiter::Wait(id) => {
                    let n = self.waits.get_mut(id.0 as usize).expect("wait");
                    *n -= 1;
                    if *n == 0 {
                        self.waits.remove(id.0 as usize);
                        out.waits_done.push(id);
                    }
                }
                Waiter::Remote(id) => {
                    let r = self.inbound_refs.get_mut(id as usize).expect("inbound ref");
                    r.remaining -= 1;
                    if r.remaining == 0 {
                        let r = self.inbound_refs.remove(id as usize).expect("inbound ref");
                        out.resolves.push(r.waiter);
                    }
                }
            }
        }
    }

    fn retire(&mut self, t: TaskId, out: &mut Effects) {
        let e = self.tasks.remove(t.0 as usize).expect("live task");
        debug_assert_eq!(e.phase, Phase::Done);
        if let Some(v) = e.version {
            self.in_flight.remove(&(e.dst.base, v));
        }
        self.execution_len -= 1;
        out.retired.push(t);
        if e.untracked {
            self.untracked_len -= 1;
            if self.untracked_len == 0 {
                for id in std::mem::take(&mut self.fences) {
                    self.waits.remove(id.0 as usize);
                    out.waits_done.push(id);
                }
            }
        }
    }

    fn insert_pending(&mut self, key: Dep) -> PendingId {
        let p = PendingId(self.pending.insert(PendingEntry {
            key,
            waiters: Vec::new(),
        }) as u32);
        self.pending_keys.insert(key, p);
        p
    }

    /// A new pending entry for `key`, recorded in `out`; `PendingFull` when
    /// the pending table has no free entry.
    fn new_pending(&mut self, key: Dep, out: &mut Effects) -> Result<PendingId, DepError> {
        if self.pending.len() >= self.cfg.pending_capacity {
            return Err(DepError::PendingFull);
        }
        let p = self.insert_pending(key);
        out.pending_created.push((p, key));
        self.warn_versioned_at(key, out);
        Ok(p)
    }

    /// Warns when a new pending entry of an unversioned key names the base
    /// of a version table entry or an address that a versioned task in
    /// the tables holds.
    fn warn_versioned_at(&self, key: Dep, out: &mut Effects) {
        let Dep::Unversioned(addr) = key else {
            return;
        };
        let entry = self.versions.get(&addr).map(|e| Dep::Versioned {
            addr,
            version: e.counter,
        });
        let task = || {
            self.tasks
                .iter()
                .filter(|(_, t)| t.version.is_some() && t.held().holds(self.cfg.granule, addr))
                .max_by_key(|(_, t)| t.seq)
                .map(|(_, t)| Dep::Versioned {
                    addr: t.dst.base,
                    version: t.version.expect("versioned"),
                })
        };
        if let Some(versioned) = entry.or_else(task) {
            out.warnings
                .push(Warning::UnversionedWait { addr, versioned });
        }
    }

    /// Warns when a versioned invoke of `dst` (version `v`) holds the
    /// address of an unversioned pending entry.
    fn warn_unversioned_pending(&self, dst: Dst, v: u64, out: &mut Effects) {
        let versioned = Dep::Versioned {
            addr: dst.span.base,
            version: v,
        };
        for (_, p) in self.pending.iter() {
            if let Dep::Unversioned(addr) = p.key {
                if dst.holds(self.cfg.granule, addr) {
                    out.warnings
                        .push(Warning::UnversionedWait { addr, versioned });
                }
            }
        }
    }

    /// Warns when a lookup of `key` resolves to more than one executing
    /// cover.
    fn warn_covers(&self, tags: &[Tag], key: Dep, out: &mut Effects) {
        let count = self.executing_count(tags);
        if count > 1 {
            out.warnings.push(Warning::MultipleCovers {
                addr: key.addr(),
                count,
            });
        }
    }

    fn pending_mut(&mut self, p: PendingId) -> &mut PendingEntry {
        self.pending.get_mut(p.0 as usize).expect("pending")
    }

    fn task(&self, t: TaskId) -> &Task {
        self.tasks.get(t.0 as usize).expect("live task")
    }

    fn task_mut(&mut self, t: TaskId) -> &mut Task {
        self.tasks.get_mut(t.0 as usize).expect("live task")
    }

    fn check_range(&self, r: &Range) -> Result<(), DepError> {
        if r.is_empty() {
            Err(DepError::Empty(*r))
        } else if !r.is_aligned(self.cfg.granule) {
            Err(DepError::Misaligned(*r))
        } else {
            Ok(())
        }
    }
}

/// True when an invoke of `dst` (with key `own` if versioned) adopts the
/// pending key `key`: the same versioned key, or an unversioned address
/// whose granule an unversioned `dst` holds.
fn adopts(dst: &Dst, g: u64, own: Option<Dep>, key: &Dep) -> bool {
    match (own, key) {
        (Some(own), _) => own == *key,
        (None, Dep::Versioned { .. }) => false,
        (None, Dep::Unversioned(a)) => dst.holds(g, *a),
    }
}

impl Task {
    /// The granules the task holds.
    fn held(&self) -> Dst {
        Dst {
            span: self.dst,
            mask: self.mask,
        }
    }
}
