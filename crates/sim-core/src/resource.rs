//! The resource plane (architecture Section 5).
//!
//! A flow resource has a rational rate and a calendar of busy intervals.
//! A chunk occupies it exclusively for whole cycles; a per-resource
//! fractional carry keeps the long-run throughput equal to the rate. An
//! occupancy resource has a size and a FIFO of waiters.
//!
//! [`Resources::reserve`] co-charges one chunk over all resources of its
//! demand vector (Section 5.4) and returns the granted interval and the
//! culprit.

use std::collections::{BTreeMap, VecDeque};

use crate::time::{gcd, Cycle, Ratio};

/// Index of one resource instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub struct ResourceId(pub u32);

/// Resource kinds. An open enumeration: new kinds come with new code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Kind {
    /// Tensor ops (MACs) of an Atlas array; rate in MACs per cycle.
    TensorOps,
    /// Lane-ops of a Muon; rate in lane-ops per cycle.
    VectorOps,
    /// Neutrino instructions (scalar resource, completion port); rate in
    /// instructions per cycle.
    ScalarOps,
    /// Bytes of memory ports, links and crossbars; rate in bytes per cycle.
    Bandwidth,
    /// Slots of an occupancy resource.
    Slots,
}

impl Kind {
    /// True for a flow kind; `Slots` is the only occupancy kind.
    pub fn is_flow(self) -> bool {
        !matches!(self, Kind::Slots)
    }
}

/// What a demand's amount is: bytes read, bytes written, or work (MACs,
/// lane-ops, instructions). Reads and writes are charged alike; the
/// direction is recorded in the statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Dir {
    /// Bytes read: toward the reader (a memory read, a DMA's source side).
    Read,
    /// Bytes written: toward the target (a memory write, a DMA's
    /// destination side, a message).
    Write,
    /// Compute: tensor ops, vector ops, scalar instructions.
    Work,
}

impl Dir {
    fn index(self) -> usize {
        self as usize
    }
}

/// One demand of a chunk on one flow resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Demand {
    /// The flow resource charged.
    pub resource: ResourceId,
    /// MACs, lane-ops, bytes, instructions.
    pub amount: u64,
    /// Exclusive efficiency in (0, 1] (Section 5.6).
    pub utilization: Ratio,
    /// Read, write or work.
    pub dir: Dir,
}

impl Demand {
    /// A demand of `amount` on `resource` at full utilization.
    pub fn new(resource: ResourceId, amount: u64, dir: Dir) -> Self {
        Self {
            resource,
            amount,
            utilization: Ratio::ONE,
            dir,
        }
    }

    /// `amount` bytes read on `resource`.
    pub fn read(resource: ResourceId, amount: u64) -> Self {
        Self::new(resource, amount, Dir::Read)
    }

    /// `amount` bytes written on `resource`.
    pub fn write(resource: ResourceId, amount: u64) -> Self {
        Self::new(resource, amount, Dir::Write)
    }

    /// `amount` units of work on `resource`.
    pub fn work(resource: ResourceId, amount: u64) -> Self {
        Self::new(resource, amount, Dir::Work)
    }
}

/// Cycles split by direction, indexed by [`Dir`] (read, write, work).
pub type DirCycles = [u64; 3];

/// Splits `d` charged cycles in proportion to `amounts` (by direction);
/// the parts sum to `d`.
fn split(d: u64, amounts: &DirCycles) -> DirCycles {
    let tot: u128 = amounts.iter().map(|&a| a as u128).sum();
    if tot == 0 {
        return [0, 0, d];
    }
    let part = |a: u64| (d as u128 * a as u128 / tot) as u64;
    let (r, k) = (part(amounts[0]), part(amounts[2]));
    [r, d - r - k, k]
}

/// The interval granted to one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grant {
    /// First cycle of the interval.
    pub start: Cycle,
    /// End of the interval (exclusive).
    pub end: Cycle,
    /// The resource with the longest own duration.
    pub culprit: ResourceId,
    /// The resource whose own earliest start was latest, when the chunk
    /// started after its requested earliest cycle.
    pub queue_culprit: Option<ResourceId>,
}

/// Sub-cycle remainder in cycles, `num / den`, `num < den`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Carry {
    num: u128,
    den: u128,
}

impl Carry {
    const ZERO: Carry = Carry { num: 0, den: 1 };
}

/// A flow resource: a rate and a calendar.
#[derive(Debug, Clone)]
pub struct FlowResource {
    rate: Ratio,
    /// Busy intervals, start -> end; disjoint, merged when adjacent.
    busy: BTreeMap<Cycle, Cycle>,
    /// Credit from earlier round-ups; subtracted from the next duration.
    carry: Carry,
    busy_total: u64,
    /// `busy_total` by direction.
    busy_dir: DirCycles,
    culprit_total: u64,
    delay_total: u64,
}

/// One charged interval and its cycles by direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct LogEntry {
    /// First cycle.
    pub start: Cycle,
    /// End (exclusive).
    pub end: Cycle,
    /// Its cycles by direction (read, write, work); they sum to
    /// `end - start`.
    pub dir: DirCycles,
}

impl FlowResource {
    fn new(rate: Ratio) -> Self {
        assert!(rate.num > 0, "flow resource with zero rate");
        Self {
            rate,
            busy: BTreeMap::new(),
            carry: Carry::ZERO,
            busy_total: 0,
            busy_dir: [0; 3],
            culprit_total: 0,
            delay_total: 0,
        }
    }

    /// Units per cycle.
    pub fn rate(&self) -> Ratio {
        self.rate
    }

    /// Whole cycles for `amount` at `util`, after the carry, and the new
    /// carry. The exact duration is `amount / (rate * util)`.
    fn duration(&self, amount: u64, util: Ratio) -> (u64, Carry) {
        assert!(
            util.num > 0 && util.num <= util.den,
            "utilization must be in (0, 1]"
        );
        // Fast path: integer rate, full utilization, no carry, exact.
        if self.carry.num == 0 && self.rate.den == 1 && util.num == util.den {
            let r = self.rate.num;
            if amount.is_multiple_of(r) {
                return (amount / r, Carry::ZERO);
            }
        }
        let e_num = amount as u128 * self.rate.den as u128 * util.den as u128;
        let e_den = self.rate.num as u128 * util.num as u128;
        let c = self.carry;
        // x = e - c over den = e_den * c.den. On overflow, drop the carry.
        let (Some(a), Some(b), Some(den)) = (
            e_num.checked_mul(c.den),
            c.num.checked_mul(e_den),
            e_den.checked_mul(c.den),
        ) else {
            let d = e_num.div_ceil(e_den);
            return (d as u64, reduce(d * e_den - e_num, e_den));
        };
        if a <= b {
            return (0, reduce(b - a, den));
        }
        let x = a - b;
        let d = x.div_ceil(den);
        (d as u64, reduce(d * den - x, den))
    }

    /// The earliest `t' >= t` with `[t', t' + d)` free.
    pub(crate) fn earliest(&self, mut t: Cycle, d: u64) -> Cycle {
        if d == 0 {
            return t;
        }
        if let Some((_, &end)) = self.busy.range(..=t).next_back() {
            t = t.max(end);
        }
        for (&s, &e) in self.busy.range(t..) {
            if s - t >= d {
                return t;
            }
            t = t.max(e);
        }
        t
    }

    /// Charges `[start, start + d)`; returns the cycles by direction.
    fn charge(&mut self, start: Cycle, d: u64, amounts: &DirCycles) -> DirCycles {
        if d == 0 {
            return [0; 3];
        }
        let parts = split(d, amounts);
        for (b, p) in self.busy_dir.iter_mut().zip(parts) {
            *b += p;
        }
        let mut s = start;
        let mut e = start + d;
        if let Some((&ps, &pe)) = self.busy.range(..=s).next_back() {
            debug_assert!(pe <= s, "charge over a busy interval");
            if pe == s {
                self.busy.remove(&ps);
                s = ps;
            }
        }
        if let Some(ne) = self.busy.remove(&e) {
            e = ne;
        }
        debug_assert!(
            self.busy.range(s..e).next().is_none(),
            "charge over a busy interval"
        );
        self.busy.insert(s, e);
        self.busy_total += d;
        parts
    }

    /// Drops intervals that end at or before `now`.
    fn prune(&mut self, now: Cycle) {
        while let Some((&s, &e)) = self.busy.first_key_value() {
            if e > now {
                break;
            }
            self.busy.remove(&s);
        }
    }
}

fn reduce(num: u128, den: u128) -> Carry {
    if num == 0 {
        return Carry::ZERO;
    }
    let g = gcd(num, den);
    Carry {
        num: num / g,
        den: den / g,
    }
}

/// Who waits on an occupancy resource. Opaque to the resource plane.
pub type Waiter = u64;

/// An occupancy resource: a size and a FIFO of waiters.
#[derive(Debug, Clone)]
pub struct OccupancyResource {
    size: u64,
    used: u64,
    peak: u64,
    waiters: VecDeque<Waiter>,
}

impl OccupancyResource {
    /// Total units.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Units held now.
    pub fn used(&self) -> u64 {
        self.used
    }

    /// The largest amount held at once.
    pub fn peak(&self) -> u64 {
        self.peak
    }
}

#[derive(Debug, Clone)]
pub(crate) enum ResourceState {
    Flow(FlowResource),
    Occupancy(OccupancyResource),
}

/// One resource: its name, kind and state.
#[derive(Debug, Clone)]
pub struct Resource {
    /// Name, such as `<node>.scalar` or `<node>.<memory>.port`.
    pub name: String,
    /// The resource kind.
    pub kind: Kind,
    pub(crate) state: ResourceState,
}

/// Per-resource statistics at the end of a run.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ResourceStats {
    /// The resource's name.
    pub name: String,
    /// The resource's kind.
    pub kind: Kind,
    /// Flow: charged cycles. Occupancy: 0.
    pub busy_cycles: u64,
    /// The part of `busy_cycles` charged for reads.
    pub busy_read: u64,
    /// The part charged for writes.
    pub busy_write: u64,
    /// The part charged for work (compute).
    pub busy_work: u64,
    /// Occupancy: the largest amount held at once.
    pub peak: u64,
    /// Flow: charged intervals, when logging is on.
    pub log: Option<Vec<LogEntry>>,
    /// Flow: cycles of the chunks for which this resource was the culprit
    /// (the longest own duration).
    pub culprit_cycles: u64,
    /// Flow: start delay of chunks (granted start minus requested earliest)
    /// attributed to this resource: the one whose own earliest start was
    /// latest, that is, the one the chunk queued on.
    pub delay_cycles: u64,
}

/// How one chunk charges the resources of its demand vector (Section 5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoCharge {
    /// Every resource over `[t, t + D)`, `D` the longest own duration (the
    /// first rule of Section 5.4). A fast resource shared by slower paths is
    /// held for the slow duration.
    Longest,
    /// Each resource over its own duration from `t`; the chunk still ends at
    /// `t + D`. A fast shared resource can serve other chunks meanwhile. The
    /// default (Section 5.4, decided 2026-09-22).
    #[default]
    Own,
}

/// All resources of one simulation.
#[derive(Debug, Clone, Default)]
pub struct Resources {
    res: Vec<Resource>,
    cocharge: CoCharge,
    /// Record every charge in `charges` (a tracer is installed).
    record: bool,
    /// The charges of the last `reserve`, when recording.
    pub(crate) charges: Vec<crate::tracer::Charge>,
    /// Scratch buffers of `reserve`, kept to avoid allocation.
    merged: Vec<Demand>,
    /// Per merged demand: its amount by direction.
    merged_dir: Vec<DirCycles>,
    own: Vec<(ResourceId, u64, Carry, Cycle)>,
}

impl Resources {
    /// Records the charges of every `reserve` for a tracer.
    pub(crate) fn set_record(&mut self, on: bool) {
        self.record = on;
    }

    /// No resources yet.
    pub fn new() -> Self {
        Self {
            res: Vec::new(),
            cocharge: CoCharge::Own,
            record: false,
            charges: Vec::new(),
            merged: Vec::new(),
            merged_dir: Vec::new(),
            own: Vec::new(),
        }
    }

    /// Sets the co-charge rule (default `Own`).
    pub fn set_cocharge(&mut self, mode: CoCharge) {
        self.cocharge = mode;
    }

    /// Adds a flow resource of `rate` units per cycle and returns its id.
    /// Panics for an occupancy kind.
    pub fn add_flow(&mut self, name: &str, kind: Kind, rate: Ratio) -> ResourceId {
        assert!(kind.is_flow(), "{kind:?} is not a flow kind");
        self.push(name, kind, ResourceState::Flow(FlowResource::new(rate)))
    }

    /// Adds an occupancy resource of `size` units and returns its id. Panics
    /// for a flow kind.
    pub fn add_occupancy(&mut self, name: &str, kind: Kind, size: u64) -> ResourceId {
        assert!(!kind.is_flow(), "{kind:?} is not an occupancy kind");
        self.push(
            name,
            kind,
            ResourceState::Occupancy(OccupancyResource {
                size,
                used: 0,
                peak: 0,
                waiters: VecDeque::new(),
            }),
        )
    }

    fn push(&mut self, name: &str, kind: Kind, state: ResourceState) -> ResourceId {
        self.res.push(Resource {
            name: name.to_string(),
            kind,
            state,
        });
        ResourceId(self.res.len() as u32 - 1)
    }

    /// The resource `id`.
    pub fn get(&self, id: ResourceId) -> &Resource {
        &self.res[id.0 as usize]
    }

    /// Number of resources.
    pub fn len(&self) -> usize {
        self.res.len()
    }

    /// The resource with this name.
    pub fn by_name(&self, name: &str) -> Option<ResourceId> {
        self.res
            .iter()
            .position(|r| r.name == name)
            .map(|i| ResourceId(i as u32))
    }

    /// True when there are no resources.
    pub fn is_empty(&self) -> bool {
        self.res.is_empty()
    }

    /// The rate of flow resource `id`; `None` for an occupancy resource.
    pub fn flow_rate(&self, id: ResourceId) -> Option<Ratio> {
        match &self.res[id.0 as usize].state {
            ResourceState::Flow(f) => Some(f.rate()),
            ResourceState::Occupancy(_) => None,
        }
    }

    /// The flow state of resource `id`. Panics for an occupancy resource.
    pub fn flow(&self, id: ResourceId) -> &FlowResource {
        match &self.res[id.0 as usize].state {
            ResourceState::Flow(f) => f,
            ResourceState::Occupancy(_) => panic!("{} is not a flow resource", self.get(id).name),
        }
    }

    fn flow_mut(&mut self, id: ResourceId) -> &mut FlowResource {
        let r = &mut self.res[id.0 as usize];
        match &mut r.state {
            ResourceState::Flow(f) => f,
            ResourceState::Occupancy(_) => panic!("{} is not a flow resource", r.name),
        }
    }

    fn occ_mut(&mut self, id: ResourceId) -> &mut OccupancyResource {
        match &mut self.res[id.0 as usize].state {
            ResourceState::Occupancy(o) => o,
            ResourceState::Flow(_) => panic!("resource {} is not occupancy", id.0),
        }
    }

    /// Co-charges one chunk at or after `earliest` (Section 5.4). Demands on
    /// the same resource with the same utilization are summed. Returns
    /// `None` for an empty demand vector.
    pub fn reserve(&mut self, demand: &[Demand], earliest: Cycle) -> Option<Grant> {
        let mut merged = std::mem::take(&mut self.merged);
        let mut merged_dir = std::mem::take(&mut self.merged_dir);
        let mut own = std::mem::take(&mut self.own);
        merge(demand, &mut merged, &mut merged_dir);
        own.clear();
        let grant = self.co_charge(&merged, &merged_dir, &mut own, earliest);
        self.merged = merged;
        self.merged_dir = merged_dir;
        self.own = own;
        grant
    }

    fn co_charge(
        &mut self,
        merged: &[Demand],
        merged_dir: &[DirCycles],
        own: &mut Vec<(ResourceId, u64, Carry, Cycle)>,
        earliest: Cycle,
    ) -> Option<Grant> {
        if merged.is_empty() {
            return None;
        }
        // Step 1: own durations and own earliest starts.
        own.extend(merged.iter().map(|d| {
            let f = self.flow(d.resource);
            let (dur, carry) = f.duration(d.amount, d.utilization);
            (d.resource, dur, carry, f.earliest(earliest, dur))
        }));
        // Step 2.
        let mut t = own.iter().map(|o| o.3).max().expect("nonempty");
        let big_d = own.iter().map(|o| o.1).max().expect("nonempty");
        // Step 3: fixpoint; t only advances. Each resource must be free for
        // the length it is charged.
        let own_mode = self.cocharge == CoCharge::Own;
        let charged = |d: u64| if own_mode { d } else { big_d };
        loop {
            let next = own
                .iter()
                .map(|o| self.flow(o.0).earliest(t, charged(o.1)))
                .max()
                .expect("nonempty");
            if next == t {
                break;
            }
            t = next;
        }
        // Step 4. A resource charged longer than its own duration keeps no
        // carry: the extra charge already exceeds its sub-cycle remainder.
        for (i, &(r, d, carry, _)) in own.iter().enumerate() {
            let f = self.flow_mut(r);
            let parts = f.charge(t, charged(d), &merged_dir[i]);
            f.carry = if charged(d) == d { carry } else { Carry::ZERO };
            // A charge of 0 cycles (a carry) is recorded too: its amount
            // still moves.
            if self.record {
                for (k, dir) in [Dir::Read, Dir::Write, Dir::Work].into_iter().enumerate() {
                    if merged_dir[i][k] > 0 {
                        self.charges.push(crate::tracer::Charge {
                            resource: r,
                            dir,
                            amount: merged_dir[i][k],
                            start: t,
                            span: charged(d),
                            cycles: parts[k],
                        });
                    }
                }
            }
        }
        // Culprit: longest own duration; on a tie, the latest own start.
        let culprit = own
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(a.3.cmp(&b.3)).then(b.0.cmp(&a.0)))
            .expect("nonempty")
            .0;
        self.flow_mut(culprit).culprit_total += big_d;
        let queue_culprit = (t > earliest).then(|| {
            let queued = own
                .iter()
                .max_by(|a, b| a.3.cmp(&b.3).then(b.0.cmp(&a.0)))
                .expect("nonempty")
                .0;
            self.flow_mut(queued).delay_total += t - earliest;
            queued
        });
        Some(Grant {
            start: t,
            end: t + big_d,
            culprit,
            queue_culprit,
        })
    }

    /// Drops calendar intervals that end at or before `now`.
    pub fn prune(&mut self, now: Cycle) {
        for r in &mut self.res {
            if let ResourceState::Flow(f) = &mut r.state {
                f.prune(now);
            }
        }
    }

    /// Acquires `amount` units if they fit and nobody waits ahead.
    pub fn try_acquire(&mut self, id: ResourceId, amount: u64) -> bool {
        let o = self.occ_mut(id);
        assert!(
            amount <= o.size,
            "acquire of {amount} exceeds the size {}",
            o.size
        );
        if o.waiters.is_empty() && o.used + amount <= o.size {
            o.used += amount;
            o.peak = o.peak.max(o.used);
            true
        } else {
            false
        }
    }

    /// True when `amount` units would be granted now.
    pub fn fits(&self, id: ResourceId, amount: u64) -> bool {
        match &self.get(id).state {
            ResourceState::Occupancy(o) => o.waiters.is_empty() && o.used + amount <= o.size,
            ResourceState::Flow(_) => panic!("fits on a flow resource"),
        }
    }

    /// Queues `w` on an occupancy resource.
    pub fn enqueue(&mut self, id: ResourceId, w: Waiter) {
        self.occ_mut(id).waiters.push_back(w);
    }

    /// Releases `amount` units and returns the waiters to retry, in FIFO
    /// order. Retried waiters that still do not fit enqueue again.
    pub fn release(&mut self, id: ResourceId, amount: u64) -> Vec<Waiter> {
        let o = self.occ_mut(id);
        assert!(o.used >= amount, "release of more than is held");
        o.used -= amount;
        o.waiters.drain(..).collect()
    }

    /// Statistics of every resource, indexed by `ResourceId`.
    pub fn stats(&self) -> Vec<ResourceStats> {
        self.res
            .iter()
            .map(|r| match &r.state {
                ResourceState::Flow(f) => ResourceStats {
                    name: r.name.clone(),
                    kind: r.kind,
                    busy_cycles: f.busy_total,
                    busy_read: f.busy_dir[0],
                    busy_write: f.busy_dir[1],
                    busy_work: f.busy_dir[2],
                    peak: 0,
                    log: None,
                    culprit_cycles: f.culprit_total,
                    delay_cycles: f.delay_total,
                },
                ResourceState::Occupancy(o) => ResourceStats {
                    name: r.name.clone(),
                    kind: r.kind,
                    busy_cycles: 0,
                    busy_read: 0,
                    busy_write: 0,
                    busy_work: 0,
                    peak: o.peak,
                    log: None,
                    culprit_cycles: 0,
                    delay_cycles: 0,
                },
            })
            .collect()
    }
}

/// Sums demands on the same resource and utilization into `out`; drops
/// zero amounts.
fn merge(demand: &[Demand], out: &mut Vec<Demand>, dirs: &mut Vec<DirCycles>) {
    out.clear();
    dirs.clear();
    for d in demand.iter().filter(|d| d.amount > 0) {
        match out.iter().position(|o| o.resource == d.resource) {
            Some(i) => {
                assert_eq!(
                    out[i].utilization, d.utilization,
                    "two demands on one resource with different utilization"
                );
                out[i].amount += d.amount;
                dirs[i][d.dir.index()] += d.amount;
            }
            None => {
                out.push(*d);
                let mut a = [0; 3];
                a[d.dir.index()] = d.amount;
                dirs.push(a);
            }
        }
    }
}
