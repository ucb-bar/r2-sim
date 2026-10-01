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

/// One demand of a chunk on one flow resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Demand {
    /// The flow resource charged.
    pub resource: ResourceId,
    /// MACs, lane-ops, bytes, instructions.
    pub amount: u64,
    /// Exclusive efficiency in (0, 1] (Section 5.6).
    pub utilization: Ratio,
}

impl Demand {
    /// A demand of `amount` on `resource` at full utilization.
    pub fn new(resource: ResourceId, amount: u64) -> Self {
        Self {
            resource,
            amount,
            utilization: Ratio::ONE,
        }
    }
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
    culprit_total: u64,
    delay_total: u64,
    log: Option<Vec<(Cycle, Cycle)>>,
}

impl FlowResource {
    fn new(rate: Ratio, log: bool) -> Self {
        assert!(rate.num > 0, "flow resource with zero rate");
        Self {
            rate,
            busy: BTreeMap::new(),
            carry: Carry::ZERO,
            busy_total: 0,
            culprit_total: 0,
            delay_total: 0,
            log: log.then(Vec::new),
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

    fn charge(&mut self, start: Cycle, d: u64) {
        if d == 0 {
            return;
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
        if let Some(log) = &mut self.log {
            log.push((start, start + d));
        }
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
    /// Occupancy: the largest amount held at once.
    pub peak: u64,
    /// Flow: charged intervals, when logging is on.
    pub log: Option<Vec<(Cycle, Cycle)>>,
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
    log: bool,
    cocharge: CoCharge,
    /// Scratch buffers of `reserve`, kept to avoid allocation.
    merged: Vec<Demand>,
    own: Vec<(ResourceId, u64, Carry, Cycle)>,
}

impl Resources {
    /// Turns the per-interval log of every flow resource on or off (for
    /// resources already added and later ones).
    pub fn set_log(&mut self, on: bool) {
        self.log = on;
        for r in &mut self.res {
            if let ResourceState::Flow(f) = &mut r.state {
                f.log = on.then(Vec::new);
            }
        }
    }

    /// `log` turns on the per-interval log of every flow resource.
    pub fn new(log: bool) -> Self {
        Self {
            res: Vec::new(),
            log,
            cocharge: CoCharge::Own,
            merged: Vec::new(),
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
        self.push(
            name,
            kind,
            ResourceState::Flow(FlowResource::new(rate, self.log)),
        )
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
        let mut own = std::mem::take(&mut self.own);
        merge(demand, &mut merged);
        own.clear();
        let grant = self.co_charge(&merged, &mut own, earliest);
        self.merged = merged;
        self.own = own;
        grant
    }

    fn co_charge(
        &mut self,
        merged: &[Demand],
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
        for &(r, d, carry, _) in own.iter() {
            let f = self.flow_mut(r);
            f.charge(t, charged(d));
            f.carry = if charged(d) == d { carry } else { Carry::ZERO };
        }
        // Culprit: longest own duration; on a tie, the latest own start.
        let culprit = own
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(a.3.cmp(&b.3)).then(b.0.cmp(&a.0)))
            .expect("nonempty")
            .0;
        self.flow_mut(culprit).culprit_total += big_d;
        if t > earliest {
            let queued = own
                .iter()
                .max_by(|a, b| a.3.cmp(&b.3).then(b.0.cmp(&a.0)))
                .expect("nonempty")
                .0;
            self.flow_mut(queued).delay_total += t - earliest;
        }
        Some(Grant {
            start: t,
            end: t + big_d,
            culprit,
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
                    peak: 0,
                    log: f.log.clone(),
                    culprit_cycles: f.culprit_total,
                    delay_cycles: f.delay_total,
                },
                ResourceState::Occupancy(o) => ResourceStats {
                    name: r.name.clone(),
                    kind: r.kind,
                    busy_cycles: 0,
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
fn merge(demand: &[Demand], out: &mut Vec<Demand>) {
    out.clear();
    for d in demand.iter().filter(|d| d.amount > 0) {
        match out.iter_mut().find(|o| o.resource == d.resource) {
            Some(o) => {
                assert_eq!(
                    o.utilization, d.utilization,
                    "two demands on one resource with different utilization"
                );
                o.amount += d.amount;
            }
            None => out.push(*d),
        }
    }
}
