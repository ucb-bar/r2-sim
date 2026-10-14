//! The resource plane (architecture Section 5).
//!
//! A flow resource has a rate per cycle of its own clock, a clock ratio to
//! the core clock and a capacity in slots per core cycle (Section 5.2): a
//! slot moves one beat per core cycle. Its calendar holds the slots in use
//! over time in fixed point ([`SCALE`] units per slot). A chunk holds a
//! constant usage of slots for whole cycles, at least one, and nothing
//! carries over to the next chunk. An occupancy resource has a size and a
//! FIFO of waiters.
//!
//! [`Resources::reserve`] co-charges a compute chunk over the resources of
//! its demand vector and [`Resources::reserve_path`] places a transfer
//! chunk hop by hop (Section 5.4); both return the granted interval and
//! the culprit.

use std::collections::VecDeque;

use crate::time::{Cycle, Ratio};

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
/// lane-ops, instructions). Every hop is charged in the direction the
/// data moves; a write uses the write channel of a hop that has one
/// (`wr_bytes`, architecture Section 3.5), a read the read channel. The
/// direction is also recorded in the statistics.
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
    /// The master's beat in bytes on a data resource (the chunk's beat
    /// there is the smaller of it and the resource's beat; Section 5.4);
    /// 0: the resource's beat.
    pub beat: u64,
    /// The rate class of a work demand on a resource whose rate depends
    /// on the operand type (the index of the class an Atlas tile runs at,
    /// in the order of its `rates`), for a tracer.
    pub class: Option<u8>,
}

impl Demand {
    /// A demand of `amount` on `resource` at full utilization.
    pub fn new(resource: ResourceId, amount: u64, dir: Dir) -> Self {
        Self {
            resource,
            amount,
            utilization: Ratio::ONE,
            dir,
            beat: 0,
            class: None,
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

/// Fixed-point units of one slot: usages and capacities are integer
/// multiples of `1 / SCALE` slot. 2^16 x 720,720 is exact for every
/// power-of-two beat ratio up to 2^16 and every clock denominator up to 16;
/// a needed usage rounds up and a capacity rounds down, so a calendar
/// never holds more than the capacity.
pub const SCALE: u64 = 65_536 * 720_720;

/// Slot-cycles split by direction, in [`SCALE`] units, indexed by [`Dir`]
/// (read, write, work).
pub type DirSlots = [u128; 3];

/// Amounts by direction, indexed by [`Dir`] (read, write, work).
pub type DirAmounts = [u64; 3];

/// Splits `total` in proportion to `amounts` (by direction); the parts sum
/// to `total`.
fn split(total: u128, amounts: &DirAmounts) -> DirSlots {
    let tot: u128 = amounts.iter().map(|&a| a as u128).sum();
    if tot == 0 {
        return [0, 0, total];
    }
    let part = |a: u64| total * a as u128 / tot;
    let (r, k) = (part(amounts[0]), part(amounts[2]));
    [r, total - r - k, k]
}

/// `ceil(a / b)` in u128.
fn ceil_div(a: u128, b: u128) -> u128 {
    a.div_ceil(b)
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

/// A flow resource: a rate, a clock ratio, a capacity in slots and a
/// calendar of the slots in use over time (Section 5.2).
#[derive(Debug, Clone)]
pub struct FlowResource {
    /// Units per cycle of the resource's own clock.
    rate: Ratio,
    /// Its clock over the core clock (Section 4.1).
    clock: Ratio,
    /// The beat in bytes of a data resource (`beat_bytes`); `None`: one
    /// slot whose beat is the whole rate (a compute unit, or a data
    /// resource built without a beat).
    beat: Option<u64>,
    /// Slots per core cycle, in [`SCALE`] units: `rate / beat x clock`, or
    /// `clock` without a beat (rounded down).
    cap: u64,
    /// The capacity as `(p, q)` when it is not an integer: its slots then
    /// exist at fixed cycles (Section 5.2, whole slots by cycle), and a
    /// window's length depends on its start. `None` for an integer
    /// capacity.
    frac: Option<(u64, u64)>,
    /// Slots in use from each key until the next one, in [`SCALE`] units;
    /// 0 after the last. Sorted by time. A calendar holds a few tens of
    /// keys after pruning, so a sorted deque beats a tree on every
    /// operation.
    used: VecDeque<(Cycle, u64)>,
    /// Slot-cycles charged, in [`SCALE`] units.
    busy_total: u128,
    /// `busy_total` by direction.
    busy_dir: DirSlots,
    /// Bytes charged (read and write).
    bytes: u64,
    culprit_total: u64,
    delay_total: u64,
}

/// One charged interval and its slot-cycles by direction.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct LogEntry {
    /// First cycle.
    pub start: Cycle,
    /// End (exclusive).
    pub end: Cycle,
    /// Its slot-cycles by direction (read, write, work).
    pub dir: [f64; 3],
}

impl FlowResource {
    fn new(rate: Ratio) -> Self {
        assert!(rate.num > 0, "flow resource with zero rate");
        Self {
            rate,
            clock: Ratio::ONE,
            beat: None,
            cap: SCALE,
            frac: None,
            used: VecDeque::new(),
            busy_total: 0,
            busy_dir: [0; 3],
            bytes: 0,
            culprit_total: 0,
            delay_total: 0,
        }
    }

    /// Units per cycle of the resource's own clock.
    pub fn rate(&self) -> Ratio {
        self.rate
    }

    /// The clock ratio to the core clock.
    pub fn clock(&self) -> Ratio {
        self.clock
    }

    /// Units per core cycle: `rate x clock`.
    pub fn core_rate(&self) -> Ratio {
        Ratio::new(
            self.rate.num * self.clock.num,
            self.rate.den * self.clock.den,
        )
    }

    /// The beat in bytes; `None` for a one-slot resource.
    pub fn beat(&self) -> Option<u64> {
        self.beat
    }

    /// Slots per core cycle, exactly: `rate / beat x clock`, or `clock`.
    pub fn capacity(&self) -> Ratio {
        match self.beat {
            Some(b) => Ratio::new(
                self.rate.num * self.clock.num,
                self.rate.den * self.clock.den * b,
            ),
            None => self.clock,
        }
    }

    /// Slots per core cycle.
    pub fn slots(&self) -> f64 {
        self.cap as f64 / SCALE as f64
    }

    fn set_cap(&mut self) {
        let c = self.capacity();
        self.cap = (c.num as u128 * SCALE as u128 / c.den as u128) as u64;
        self.frac = (c.den != 1).then_some((c.num, c.den));
        assert!(self.cap > 0, "a flow resource with no capacity");
    }

    /// The slots of the cycles `[0, t)`: `floor(t x C)` (Section 5.2).
    fn slots_to(t: Cycle, p: u64, q: u64) -> u128 {
        t as u128 * p as u128 / q as u128
    }

    /// The end of a window from `t0` that needs `n` of the resource's
    /// slots (the first t1 with `floor(t1 x C) - floor(t0 x C) >= n`).
    fn slot_end(t0: Cycle, n: u128, p: u64, q: u64) -> Cycle {
        ceil_div((Self::slots_to(t0, p, q) + n) * q as u128, p as u128) as Cycle
    }

    /// Whole cycles from `t0` for `beats` at `u` slots (Section 5.2): with
    /// an integer capacity `ceil(beats / u)`; otherwise the chunk moves u
    /// / C of the slots that exist in its window, so it needs `N =
    /// ceil(beats x C / u)` of them (`beats` when it holds the whole
    /// capacity), and its window ends at the cycle that holds the N-th. At
    /// least one cycle.
    #[inline]
    fn window(&self, t0: Cycle, beats: u64, u: u64) -> u64 {
        let Some((p, q)) = self.frac else {
            return Self::dur(beats, u);
        };
        let n = if u >= self.cap {
            beats as u128
        } else {
            ceil_div(
                beats as u128 * p as u128 * SCALE as u128,
                q as u128 * u as u128,
            )
        };
        (Self::slot_end(t0, n, p, q) - t0).max(1)
    }

    /// Whole cycles from `t0` of a work demand that holds the whole
    /// capacity: its beats are `ceil(amount / (rate x util))` (own
    /// cycles), one per slot.
    fn work_window(&self, t0: Cycle, amount: u64, util: Ratio) -> u64 {
        let Some((p, q)) = self.frac else {
            return self.work_duration(amount, util);
        };
        if amount == 0 {
            return 0;
        }
        let r = self.rate;
        let beats = ceil_div(
            amount as u128 * r.den as u128 * util.den as u128,
            r.num as u128 * util.num as u128,
        )
        .max(1);
        (Self::slot_end(t0, beats, p, q) - t0).max(1)
    }

    /// The earliest `t' >= t` with `u` slots free over `[t', t' +
    /// len(t'))`: with an integer capacity the length does not depend on
    /// the start; otherwise the search repeats from each candidate until a
    /// window of its own length is free.
    #[inline]
    fn earliest_by(&self, t: Cycle, u: u64, len: impl Fn(Cycle) -> u64) -> Cycle {
        if self.frac.is_none() {
            return self.earliest(t, len(t), u);
        }
        let mut t = t;
        loop {
            let s = self.earliest(t, len(t), u);
            if s == t {
                return t;
            }
            t = s;
        }
    }

    /// The earliest start `>= t` of a window for `beats` at `u` slots,
    /// and the window's length there.
    #[inline]
    fn earliest_window(&self, t: Cycle, beats: u64, u: u64) -> (Cycle, u64) {
        if self.frac.is_none() {
            let d = Self::dur(beats, u);
            return (self.earliest(t, d, u), d);
        }
        let s = self.earliest_by(t, u, |x| self.window(x, beats, u));
        (s, self.window(s, beats, u))
    }

    /// Splits the rate into slots of `beat` bytes (Section 5.2).
    fn set_beat(&mut self, beat: u64) {
        assert!(beat > 0, "beat of 0 bytes");
        self.beat = Some(beat);
        self.set_cap();
    }

    /// Sets the clock ratio (Section 4.1).
    fn set_clock(&mut self, clock: Ratio) {
        assert!(clock.num > 0, "clock 0");
        self.clock = clock;
        self.set_cap();
    }

    /// The beat of a data resource: `beat_bytes`, or its rate without one.
    fn data_beat(&self) -> u64 {
        self.beat.unwrap_or((self.rate.num / self.rate.den).max(1))
    }

    /// Whole core cycles of a work demand: `ceil(amount / (rate x clock x
    /// util))`, at least one for a nonzero amount; 0 for none.
    fn work_duration(&self, amount: u64, util: Ratio) -> u64 {
        assert!(
            util.num > 0 && util.num <= util.den,
            "utilization must be in (0, 1]"
        );
        if amount == 0 {
            return 0;
        }
        let (r, c) = (self.rate, self.clock);
        if r.den == 1 && c.num == c.den && util.num == util.den {
            return amount.div_ceil(r.num);
        }
        let n = amount as u128 * r.den as u128 * c.den as u128 * util.den as u128;
        let d = r.num as u128 * c.num as u128 * util.num as u128;
        n.div_ceil(d).max(1) as u64
    }

    /// The beat and the beats of `amount` bytes for a master of `beat`
    /// bytes (0: none), at `util`.
    fn beats(&self, amount: u64, beat: u64, util: Ratio) -> (u64, u64) {
        let b = match beat {
            0 => self.data_beat(),
            m => m.min(self.data_beat()),
        };
        let n = amount.div_ceil(b);
        let n = if util.num == util.den {
            n
        } else {
            ceil_div(n as u128 * util.den as u128, util.num as u128) as u64
        };
        (b, n)
    }

    /// Whole cycles for `beats` at `u` slots: `ceil(beats / u)`, at least
    /// one.
    fn dur(beats: u64, u: u64) -> u64 {
        ceil_div(beats as u128 * SCALE as u128, u as u128).max(1) as u64
    }

    /// Keys at or before `t`.
    fn keys_to(&self, t: Cycle) -> usize {
        self.used.partition_point(|&(k, _)| k <= t)
    }

    /// Slots in use at `t`.
    fn used_at(&self, t: Cycle) -> u64 {
        match self.keys_to(t) {
            0 => 0,
            i => self.used[i - 1].1,
        }
    }

    /// The earliest `t' >= t` with `u` slots free over `[t', t' + d)`: one
    /// forward pass over the keys from `t`.
    pub(crate) fn earliest(&self, t: Cycle, d: u64, u: u64) -> Cycle {
        if d == 0 {
            return t;
        }
        // The tail: at or after the last key every slot is free.
        if self.used.back().is_none_or(|&(last, _)| last <= t) {
            return t;
        }
        let free = self.cap - u.min(self.cap);
        let mut cand = t;
        // Slots in use at `cand`; they change only at keys, and the last
        // key's value is 0.
        let i = self.keys_to(t);
        let mut busy = i > 0 && self.used[i - 1].1 > free;
        for &(at, v) in self.used.range(i..) {
            if busy {
                // No window yet: it may start at this key.
                cand = at;
                busy = v > free;
            } else if at >= cand + d {
                return cand;
            } else if v > free {
                // The window from `cand` breaks here.
                cand = at;
                busy = true;
            }
        }
        cand
    }

    /// The earliest `t' >= t` at which some slots are free.
    fn first_free(&self, t: Cycle) -> Cycle {
        if self.used_at(t) < self.cap {
            return t;
        }
        let i = self.keys_to(t);
        self.used
            .range(i..)
            .find(|&&(_, v)| v < self.cap)
            .map_or(t, |&(at, _)| at)
    }

    /// The slots free throughout `[t, t + d)`.
    fn min_free(&self, t: Cycle, d: u64) -> u64 {
        let i = self.keys_to(t);
        let mut most = if i > 0 { self.used[i - 1].1 } else { 0 };
        for &(at, v) in self.used.range(i..) {
            if at >= t + d {
                break;
            }
            most = most.max(v);
        }
        self.cap - most
    }

    /// The index of key `t`, inserted with the slots in use there if it is
    /// not a key.
    fn key(&mut self, t: Cycle) -> usize {
        let i = self.used.partition_point(|&(k, _)| k < t);
        if self.used.get(i).is_none_or(|&(k, _)| k != t) {
            let v = if i == 0 { 0 } else { self.used[i - 1].1 };
            self.used.insert(i, (t, v));
        }
        i
    }

    /// Removes key `i` when it repeats the slots in use before it.
    fn coalesce(&mut self, i: usize) -> bool {
        let before = if i == 0 { 0 } else { self.used[i - 1].1 };
        let same = before == self.used[i].1;
        if same {
            self.used.remove(i);
        }
        same
    }

    /// Adds `delta` slots in use over `[t, t + d)`.
    fn add(&mut self, t: Cycle, d: u64, delta: i64) {
        if d == 0 {
            return;
        }
        let e = t + d;
        let it = self.key(t);
        let ie = self.key(e);
        for (_, v) in self.used.range_mut(it..ie) {
            let n = *v as i64 + delta;
            debug_assert!(n >= 0 && n as u64 <= self.cap, "slots over capacity");
            *v = n as u64;
        }
        // Coalesce at the two ends.
        let ie = if self.coalesce(it) { ie - 1 } else { ie };
        self.coalesce(ie);
    }

    /// Holds `u` slots over `[t, t + d)` (the calendar only).
    fn occupy(&mut self, t: Cycle, d: u64, u: u64) {
        self.add(t, d, u as i64);
    }

    /// Frees `u` slots over `[t, t + d)` held by [`FlowResource::occupy`].
    fn release(&mut self, t: Cycle, d: u64, u: u64) {
        self.add(t, d, -(u as i64));
    }

    /// Counts `u` slots over `d` cycles and the bytes of `amounts` in the
    /// statistics; returns the slot-cycles by direction.
    fn account(&mut self, d: u64, u: u64, amounts: &DirAmounts) -> DirSlots {
        let sc = d as u128 * u as u128;
        let parts = split(sc, amounts);
        for (b, p) in self.busy_dir.iter_mut().zip(parts) {
            *b += p;
        }
        self.busy_total += sc;
        self.bytes += amounts[0] + amounts[1];
        parts
    }

    /// Drops the calendar before `now`.
    #[inline]
    fn prune(&mut self, now: Cycle) {
        // Nothing before `now`: the common case, without a search.
        if self.used.front().is_none_or(|&(k, _)| k >= now) {
            return;
        }
        let v = self.used_at(now);
        let gone = self.used.partition_point(|&(k, _)| k < now);
        self.used.drain(..gone);
        if v > 0 && self.used.front().is_none_or(|&(k, _)| k != now) {
            self.used.push_front((now, v));
        }
    }
}

/// A pool for the data in flight into a node through one link (Section
/// 5.4, data-path rule 7): a calendar of the bytes held over time, at most
/// `cap` at any cycle.
#[derive(Debug, Clone)]
pub(crate) struct Pool {
    /// Its occupancy resource (name, statistics, queue culprit).
    resource: ResourceId,
    cap: u64,
    /// Bytes held from each key until the next one; 0 after the last.
    used: VecDeque<(Cycle, u64)>,
    peak: u64,
    /// Byte-cycles held.
    held: u128,
    /// Cycles chunks waited for room.
    waited: u64,
}

impl Pool {
    fn keys_to(&self, t: Cycle) -> usize {
        self.used.partition_point(|&(k, _)| k <= t)
    }

    /// The most bytes held at any cycle of `[t0, t1)`.
    fn most(&self, t0: Cycle, t1: Cycle) -> u64 {
        let i = self.keys_to(t0);
        let mut most = if i > 0 { self.used[i - 1].1 } else { 0 };
        for &(at, v) in self.used.range(i..) {
            if at >= t1 {
                break;
            }
            most = most.max(v);
        }
        most
    }

    /// True when `bytes` more fit over `[t0, t1)`.
    fn fits(&self, t0: Cycle, t1: Cycle, bytes: u64) -> bool {
        self.most(t0, t1) + bytes <= self.cap
    }

    /// The first cycle from which `bytes` more fit up to `t1`: the end of
    /// the last interval in `[t0, t1)` where they do not (`t0` if they fit
    /// throughout).
    fn room_from(&self, t0: Cycle, t1: Cycle, bytes: u64) -> Cycle {
        let free = self.cap.saturating_sub(bytes);
        let i = self.keys_to(t0);
        let mut over = i > 0 && self.used[i - 1].1 > free;
        let mut from = t0;
        for &(at, v) in self.used.range(i..) {
            if at >= t1 {
                break;
            }
            if over && v <= free {
                from = at;
            }
            over = v > free;
            if over {
                from = t1;
            }
        }
        if over {
            // Held to `t1` and beyond: room from the first key after it
            // that frees enough.
            let j = self.keys_to(t1.saturating_sub(1));
            return self
                .used
                .range(j..)
                .find(|&&(_, v)| v <= free)
                .map_or(t1, |&(at, _)| at);
        }
        from
    }

    fn key(&mut self, t: Cycle) -> usize {
        let i = self.used.partition_point(|&(k, _)| k < t);
        if self.used.get(i).is_none_or(|&(k, _)| k != t) {
            let v = if i == 0 { 0 } else { self.used[i - 1].1 };
            self.used.insert(i, (t, v));
        }
        i
    }

    /// Holds `bytes` over `[t0, t1)`.
    fn hold(&mut self, t0: Cycle, t1: Cycle, bytes: u64) {
        if t1 <= t0 || bytes == 0 {
            return;
        }
        let it = self.key(t0);
        let ie = self.key(t1);
        for (_, v) in self.used.range_mut(it..ie) {
            *v += bytes;
            debug_assert!(*v <= self.cap, "pool over its bytes");
            self.peak = self.peak.max(*v);
        }
        let before = |u: &VecDeque<(Cycle, u64)>, i: usize| if i == 0 { 0 } else { u[i - 1].1 };
        let ie = if before(&self.used, it) == self.used[it].1 {
            self.used.remove(it);
            ie - 1
        } else {
            ie
        };
        if before(&self.used, ie) == self.used[ie].1 {
            self.used.remove(ie);
        }
        self.held += (t1 - t0) as u128 * bytes as u128;
    }

    /// Drops the calendar before `now`.
    #[inline]
    fn prune(&mut self, now: Cycle) {
        if self.used.front().is_none_or(|&(k, _)| k >= now) {
            return;
        }
        let i = self.keys_to(now);
        let v = if i > 0 { self.used[i - 1].1 } else { 0 };
        let gone = self.used.partition_point(|&(k, _)| k < now);
        self.used.drain(..gone);
        if v > 0 && self.used.front().is_none_or(|&(k, _)| k != now) {
            self.used.push_front((now, v));
        }
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
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ResourceStats {
    /// The resource's name.
    pub name: String,
    /// The resource's kind.
    pub kind: Kind,
    /// Flow: slots per core cycle (Section 5.2). Occupancy: 0.
    pub slots: f64,
    /// Flow: units (bytes, MACs, lane-ops, instructions) per core cycle.
    /// Occupancy: 0.
    pub rate: f64,
    /// Flow: charged slot-cycles, rounded to whole ones. Occupancy: 0.
    pub busy_cycles: u64,
    /// Flow: charged slot-cycles, unrounded.
    pub busy_slot_cycles: f64,
    /// The part of `busy_cycles` charged for reads.
    pub busy_read: u64,
    /// The part charged for writes.
    pub busy_write: u64,
    /// The part charged for work (compute).
    pub busy_work: u64,
    /// Flow: bytes charged (reads and writes); the byte utilization is
    /// `bytes / (rate x clock x cycles)`.
    pub bytes: u64,
    /// Occupancy: the largest amount held at once.
    pub peak: u64,
    /// Occupancy: its size (an in-flight pool: `inflight_bytes`).
    pub size: u64,
    /// An in-flight pool: the byte-cycles held (the mean occupancy is
    /// this over the run's end time). 0 otherwise.
    pub held_cycles: f64,
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
    merged_dir: Vec<DirAmounts>,
    own: Vec<Own>,
    /// Scratch buffer of `reserve_path`.
    plan: Vec<Placed>,
    /// The (usage, length) of a compute chunk's narrow option (kept
    /// across chunks).
    narrow_plan: Vec<(u64, u64)>,
    /// The in-flight pools (Section 5.4, data-path rule 7).
    pools: Vec<Pool>,
}

/// One hop of a transfer chunk, in route order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathHop {
    /// The flow resource.
    pub resource: ResourceId,
    /// Bytes.
    pub amount: u64,
    pub dir: Dir,
    /// Cycles from this hop's start to the next hop's earliest start.
    pub latency: Cycle,
    /// The in-flight pool whose hold ends with this hop's window: the
    /// chunk crossed the pool's link into its node before this hop
    /// (Section 5.4, data-path rule 7).
    pub pool: Option<PoolMark>,
}

/// A pool on a route (Section 5.4, data-path rule 7): its index, and the
/// route latency from the pool's link to the hop that carries the mark
/// (0 when the hop is the link itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolMark {
    pub pool: u32,
    pub skip: Cycle,
}

/// What the in-flight pools need of a transfer chunk (Section 5.4,
/// data-path rule 7): its issue cycle, the latency of its source block,
/// and a pool whose link the chunk crosses after its last hop (the hold
/// then ends with the chunk).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Admit {
    pub issue: Cycle,
    pub src_latency: Cycle,
    pub tail: Option<PoolMark>,
}

/// One demand of a compute chunk while it is co-charged.
#[derive(Debug, Clone, Copy)]
struct Own {
    resource: ResourceId,
    /// Own duration with the whole capacity.
    dur: u64,
    /// The cycles it is charged.
    charged: u64,
    /// The slots it holds, in [`SCALE`] units.
    usage: u64,
    /// Its own earliest start.
    start: Cycle,
    /// Beats on a data resource; `None` on a work resource.
    beats: Option<u64>,
    /// The demand's rate class, for a tracer.
    class: Option<u8>,
    /// A work demand's amount and utilization (its window on a resource
    /// with a fractional capacity, Section 5.2).
    amount: u64,
    util: Ratio,
}

/// A hop's window while a transfer is placed.
#[derive(Debug, Clone, Copy)]
struct Placed {
    start: Cycle,
    dur: u64,
    /// The slots it holds, in [`SCALE`] units.
    usage: u64,
    /// Start delay on this hop.
    delay: Cycle,
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
            plan: Vec::new(),
            narrow_plan: Vec::new(),
            pools: Vec::new(),
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

    /// Splits flow resource `id` into slots of `beat` bytes (Section 5.2).
    pub fn set_beat(&mut self, id: ResourceId, beat: u64) {
        self.flow_mut(id).set_beat(beat);
    }

    /// Sets the clock ratio of flow resource `id` (Section 4.1).
    pub fn set_clock(&mut self, id: ResourceId, clock: Ratio) {
        self.flow_mut(id).set_clock(clock);
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
        merged_dir: &[DirAmounts],
        own: &mut Vec<Own>,
        earliest: Cycle,
    ) -> Option<Grant> {
        if merged.is_empty() {
            return None;
        }
        // Step 1: own durations with the whole capacity, and the chunk's D.
        own.extend(merged.iter().map(|d| {
            let f = self.flow(d.resource);
            let beats = (d.dir != Dir::Work).then(|| f.beats(d.amount, d.beat, d.utilization).1);
            let dur = match beats {
                Some(n) => FlowResource::dur(n, f.cap),
                None => f.work_duration(d.amount, d.utilization),
            };
            Own {
                resource: d.resource,
                dur,
                charged: 0,
                usage: f.cap,
                start: 0,
                beats,
                class: d.class,
                amount: d.amount,
                util: d.utilization,
            }
        }));
        let big_d = own.iter().map(|o| o.dur).max().expect("nonempty");
        // What each resource holds: a data resource the slots its own
        // rate needs over D, `beats / D` (Section 5.4), for D; a work
        // resource its whole capacity for its own duration (or D under the
        // longest rule).
        let own_mode = self.cocharge == CoCharge::Own;
        for o in own.iter_mut() {
            let f = self.flow(o.resource);
            match o.beats {
                Some(n) => {
                    o.usage =
                        (ceil_div(n as u128 * SCALE as u128, big_d as u128) as u64).clamp(1, f.cap);
                    o.charged = big_d;
                }
                None => o.charged = if own_mode { o.dur } else { big_d },
            }
            o.start = f.earliest_by(earliest, o.usage, |x| {
                self.own_len(o, o.usage, o.charged, x, own_mode)
            });
        }
        // Step 2.
        let mut t = own.iter().map(|o| o.start).max().expect("nonempty");
        // Step 3: fixpoint; t only advances. Each resource must have its
        // slots free for the length it is charged (from t, on a resource
        // with a fractional capacity).
        loop {
            let next = own
                .iter()
                .map(|o| {
                    self.flow(o.resource).earliest_by(t, o.usage, |x| {
                        self.own_len(o, o.usage, o.charged, x, own_mode)
                    })
                })
                .max()
                .expect("nonempty");
            if next == t {
                break;
            }
            t = next;
        }
        for o in own.iter_mut() {
            o.charged = self.own_len(o, o.usage, o.charged, t, own_mode);
        }
        let mut end = t + own.iter().map(|o| o.charged).max().expect("nonempty");
        // Starting on the free slots (Section 5.4): a chunk that its full
        // usage delays may start at `earliest` on the slots free there and
        // run longer, if it then ends strictly sooner.
        if own_mode && t > earliest {
            let mut plan = std::mem::take(&mut self.narrow_plan);
            if let Some((t2, e2)) = self.narrow(own, earliest, &mut plan) {
                if e2 < end {
                    (t, end) = (t2, e2);
                    for (o, &(u, c)) in own.iter_mut().zip(&plan) {
                        (o.usage, o.charged) = (u, c);
                    }
                }
            }
            self.narrow_plan = plan;
        }
        // Step 4.
        for (i, o) in own.iter().enumerate() {
            let f = self.flow_mut(o.resource);
            f.occupy(t, o.charged, o.usage);
            let parts = f.account(o.charged, o.usage, &merged_dir[i]);
            let (span, usage, class) = (o.charged, o.usage, o.class);
            self.record_charge(o.resource, &merged_dir[i], t, span, usage, &parts, class);
        }
        // Culprit: longest own duration; on a tie, the latest own start.
        let culprit = own
            .iter()
            .max_by(|a, b| {
                a.dur
                    .cmp(&b.dur)
                    .then(a.start.cmp(&b.start))
                    .then(b.resource.cmp(&a.resource))
            })
            .expect("nonempty")
            .resource;
        self.flow_mut(culprit).culprit_total += end - t;
        let queue_culprit = (t > earliest).then(|| {
            let queued = own
                .iter()
                .max_by(|a, b| a.start.cmp(&b.start).then(b.resource.cmp(&a.resource)))
                .expect("nonempty")
                .resource;
            self.flow_mut(queued).delay_total += t - earliest;
            queued
        });
        Some(Grant {
            start: t,
            end,
            culprit,
            queue_culprit,
        })
    }

    /// The length of `o`'s window from `t` at `usage`: `fixed` on an
    /// integer capacity; on a fractional one the window that holds its
    /// beats in the slots from `t` (Section 5.2), except a work demand
    /// charged the chunk's D under the longest rule.
    #[inline]
    fn own_len(&self, o: &Own, usage: u64, fixed: u64, t: Cycle, own_mode: bool) -> u64 {
        let f = self.flow(o.resource);
        if f.frac.is_none() {
            return fixed;
        }
        match o.beats {
            Some(n) => f.window(t, n, usage),
            None if own_mode => f.work_window(t, o.amount, o.util),
            None => fixed,
        }
    }

    /// The narrow option of a compute chunk (Section 5.4, starting on the
    /// free slots): on each data resource u' = the slots that stay free
    /// from `earliest` over the window (at most its full usage), found by
    /// shrinking u' while the window lengthens; D' = the longest of the
    /// work resources' own durations and `ceil(beats / u')` on each data
    /// resource; each data resource holds `beats / D'` slots for D', a
    /// work resource its full usage. Returns the start found by the same
    /// fixpoint and the chunk's end, with the (usage, length) of each
    /// resource in `plan`; `None` without a data resource or with one that
    /// has no free slot.
    fn narrow(
        &self,
        own: &[Own],
        earliest: Cycle,
        plan: &mut Vec<(u64, u64)>,
    ) -> Option<(Cycle, u64)> {
        if own.iter().all(|o| o.beats.is_none()) {
            return None;
        }
        plan.clear();
        plan.extend(own.iter().map(|o| {
            let f = self.flow(o.resource);
            match o.beats {
                Some(_) => ((f.cap - f.used_at(earliest)).min(o.usage), 0),
                None => (o.usage, o.charged),
            }
        }));
        let mut d2;
        loop {
            if own
                .iter()
                .zip(plan.iter())
                .any(|(o, p)| o.beats.is_some() && p.0 == 0)
            {
                return None;
            }
            d2 = own
                .iter()
                .zip(plan.iter())
                .map(|(o, &(u, _))| match o.beats {
                    Some(n) => self.flow(o.resource).window(earliest, n, u),
                    None => o.dur,
                })
                .max()
                .expect("nonempty");
            let mut changed = false;
            for (o, p) in own.iter().zip(plan.iter_mut()) {
                if o.beats.is_some() {
                    let m = self.flow(o.resource).min_free(earliest, d2);
                    if m < p.0 {
                        p.0 = m;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for (o, p) in own.iter().zip(plan.iter_mut()) {
            if let Some(n) = o.beats {
                let u =
                    (ceil_div(n as u128 * SCALE as u128, d2 as u128) as u64).clamp(1, p.0.max(1));
                *p = (u, d2);
            }
        }
        let mut t = earliest;
        loop {
            let next = own
                .iter()
                .zip(plan.iter())
                .map(|(o, &(u, c))| {
                    self.flow(o.resource)
                        .earliest_by(t, u, |x| self.own_len(o, u, c, x, true))
                })
                .max()
                .expect("nonempty");
            if next == t {
                break;
            }
            t = next;
        }
        for (o, p) in own.iter().zip(plan.iter_mut()) {
            p.1 = self.own_len(o, p.0, p.1, t, true);
        }
        let end = t + plan.iter().map(|p| p.1).max().expect("nonempty");
        Some((t, end))
    }

    /// Records one charge per direction with a nonzero amount, when
    /// recording: its usage (`usage` split in proportion to the amounts)
    /// and its slot-cycles.
    #[allow(clippy::too_many_arguments)]
    fn record_charge(
        &mut self,
        r: ResourceId,
        amounts: &DirAmounts,
        start: Cycle,
        span: Cycle,
        usage: u64,
        parts: &DirSlots,
        class: Option<u8>,
    ) {
        if !self.record {
            return;
        }
        let tot: u128 = amounts.iter().map(|&a| a as u128).sum();
        for (k, dir) in [Dir::Read, Dir::Write, Dir::Work].into_iter().enumerate() {
            if amounts[k] > 0 {
                self.charges.push(crate::tracer::Charge {
                    resource: r,
                    dir,
                    amount: amounts[k],
                    start,
                    span,
                    usage: (usage as u128 * amounts[k] as u128 / tot.max(1)) as u64,
                    cycles: parts[k] as f64 / SCALE as f64,
                    class,
                });
            }
        }
    }

    /// Reserves a transfer chunk along `hops` in route order (Section 5.4,
    /// data-path rule) for a master that moves one beat of `beat` bytes
    /// per cycle. Hop 0 at or after `earliest`, hop i+1 at or after hop
    /// i's start plus its latency. The chunk's beat on hop i is the
    /// smaller of its beat on hop i-1 and the hop's beat; it needs r /
    /// b_i slots there (r the rate from hop i-1, rule 2) and takes the
    /// earlier-ending of two windows (rule 3, sharing): the earliest with
    /// the need free throughout, and one from the earliest cycle with any
    /// slot free on the slots that stay free over it. Each hop keeps that
    /// window: data waits between hops when a later hop is busy. The
    /// chunk ends at the last window's end plus the last hop's latency.
    /// Returns `None` for no hops.
    pub fn reserve_path(&mut self, hops: &[PathHop], beat: u64, earliest: Cycle) -> Option<Grant> {
        let admit = Admit {
            issue: earliest,
            ..Admit::default()
        };
        self.place(hops, beat, Ratio::int(beat), earliest, false, admit)
    }

    /// [`Resources::reserve_path`] for a master of `beat` bytes that moves
    /// `rate` bytes per core cycle (a clocked engine, Section 4.1); the
    /// in-flight pools of the hops take the chunk by `admit` (Section
    /// 5.4, data-path rule 7).
    pub fn reserve_path_at(
        &mut self,
        hops: &[PathHop],
        beat: u64,
        rate: Ratio,
        earliest: Cycle,
        admit: Admit,
    ) -> Option<Grant> {
        self.place(hops, beat, rate, earliest, false, admit)
    }

    /// Adds an in-flight pool of `bytes` named `name` (Section 5.4,
    /// data-path rule 7): an occupancy resource for its statistics and as
    /// a queue culprit. Returns its index for [`PoolMark`].
    pub fn add_pool(&mut self, name: &str, bytes: u64) -> u32 {
        assert!(bytes > 0, "a pool of 0 bytes");
        let resource = self.add_occupancy(name, Kind::Slots, bytes);
        self.pools.push(Pool {
            resource,
            cap: bytes,
            used: VecDeque::new(),
            peak: 0,
            held: 0,
            waited: 0,
        });
        self.pools.len() as u32 - 1
    }

    /// Reserves a protocol message of `hop.amount` bytes on one hop
    /// (Section 5.4, details): one slot, or the whole capacity of a hop
    /// with less than one, for the message's beats at the smaller of
    /// `beat` and the hop's beat; store and forward, so the sender's
    /// earlier hops do not narrow it.
    pub fn reserve_message(&mut self, hop: PathHop, beat: u64, earliest: Cycle) -> Grant {
        let admit = Admit {
            issue: earliest,
            ..Admit::default()
        };
        self.place(&[hop], beat, Ratio::int(beat), earliest, true, admit)
            .expect("one hop")
    }

    /// Books the hops of a transfer chunk from `earliest` in route order
    /// (rules 1 to 3) into `plan`.
    fn book(
        &mut self,
        hops: &[PathHop],
        beat: u64,
        rate: Ratio,
        earliest: Cycle,
        message: bool,
        plan: &mut Vec<Placed>,
    ) {
        plan.clear();
        // The rate from the previous hop, in bytes per cycle x SCALE, and
        // the beat there.
        let mut r = rate.num as u128 * SCALE as u128 / rate.den as u128;
        let mut b_prev = beat;
        let mut want = earliest;
        for h in hops {
            let f = self.flow(h.resource);
            let b = b_prev.min(f.data_beat());
            let beats = h.amount.div_ceil(b);
            let need = if message {
                SCALE.min(f.cap)
            } else {
                (ceil_div(r, b as u128).min(f.cap as u128) as u64).max(1)
            };
            let (s, d) = f.earliest_window(want, beats, need);
            let (mut s, mut u, mut d) = (s, need, d);
            if s > want && !message {
                // Window B: from the first cycle with a free slot, on the
                // slots that stay free while the window lengthens.
                let tb = f.first_free(want);
                if tb < s {
                    let mut ub = need.min(f.cap - f.used_at(tb));
                    let mut db;
                    loop {
                        db = f.window(tb, beats, ub);
                        let m = f.min_free(tb, db);
                        if m >= ub || m == 0 {
                            ub = if m == 0 { 0 } else { ub };
                            break;
                        }
                        ub = m;
                    }
                    if ub > 0 && tb + db < s + d {
                        (s, u, d) = (tb, ub, db);
                    }
                }
            }
            self.flow_mut(h.resource).occupy(s, d, u);
            plan.push(Placed {
                start: s,
                dur: d,
                usage: u,
                delay: s - want,
            });
            r = u as u128 * b as u128;
            b_prev = b;
            want = s + h.latency;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn place(
        &mut self,
        hops: &[PathHop],
        beat: u64,
        rate: Ratio,
        earliest: Cycle,
        message: bool,
        admit: Admit,
    ) -> Option<Grant> {
        if hops.is_empty() {
            return None;
        }
        let mut plan = std::mem::take(&mut self.plan);
        let pooled = !message && (admit.tail.is_some() || hops.iter().any(|h| h.pool.is_some()));
        let mut e = earliest;
        // The pool whose room set the final e.
        let mut waited_on: Option<u32> = None;
        loop {
            self.book(hops, beat, rate, e, message, &mut plan);
            if !pooled {
                break;
            }
            match self.admit(hops, &plan, e, admit) {
                None => break,
                Some((next, p)) => {
                    // Not admitted: free the route and try again later.
                    for (h, q) in hops.iter().zip(plan.iter()) {
                        self.flow_mut(h.resource).release(q.start, q.dur, q.usage);
                    }
                    e = next;
                    waited_on = Some(p);
                }
            }
        }
        let pool_wait = e - earliest;
        // Statistics, records, culprits.
        for (h, p) in hops.iter().zip(plan.iter()) {
            let mut amounts = [0; 3];
            amounts[h.dir.index()] = h.amount;
            let parts = self.flow_mut(h.resource).account(p.dur, p.usage, &amounts);
            self.record_charge(h.resource, &amounts, p.start, p.dur, p.usage, &parts, None);
        }
        // The data arrives the last hop's latency after its window.
        let start = plan[0].start;
        let end = plan.iter().map(|p| p.start + p.dur).max().expect("hops")
            + hops.last().expect("hops").latency;
        // The first hop with the longest window: the narrowest one (the hops
        // after it are narrowed to its rate).
        let (ci, cp) = plan
            .iter()
            .enumerate()
            .max_by_key(|(i, p)| (p.dur, std::cmp::Reverse(*i)))
            .expect("hops");
        let culprit = hops[ci].resource;
        self.flow_mut(culprit).culprit_total += cp.dur;
        let (qi, qp) = plan
            .iter()
            .enumerate()
            .max_by_key(|(i, p)| (p.delay, *i))
            .expect("hops");
        let queue_culprit = match waited_on {
            // The pool's wait counts in full; it is the queue culprit when
            // it is longer than every hop's delay.
            Some(p) => {
                let pool = &mut self.pools[p as usize];
                pool.waited += pool_wait;
                let r = pool.resource;
                if pool_wait >= qp.delay {
                    Some(r)
                } else {
                    let q = hops[qi].resource;
                    self.flow_mut(q).delay_total += qp.delay;
                    Some(q)
                }
            }
            None => (qp.delay > 0).then(|| {
                let q = hops[qi].resource;
                self.flow_mut(q).delay_total += qp.delay;
                q
            }),
        };
        self.plan = plan;
        Some(Grant {
            start,
            end,
            culprit,
            queue_culprit,
        })
    }

    /// The pool test of a booked chunk (rule 7): each pool on the route
    /// must hold the chunk's bytes from its request crossing the link,
    /// max(issue, e - L_src - L_req), to the end of the marked hop's
    /// window (the chunk's end for the tail). Holds them and returns
    /// `None` when every pool has room; otherwise holds nothing and
    /// returns the next e (the latest that any pool asks for) and that
    /// pool.
    fn admit(
        &mut self,
        hops: &[PathHop],
        plan: &[Placed],
        e: Cycle,
        admit: Admit,
    ) -> Option<(Cycle, u32)> {
        let end = plan.iter().map(|p| p.start + p.dur).max().expect("hops")
            + hops.last().expect("hops").latency;
        let mut retry: Option<(Cycle, u32)> = None;
        let mut holds: [(u32, Cycle, Cycle, u64); 4] = [(0, 0, 0, 0); 4];
        let mut n = 0;
        let mut at = 0;
        let mut test = |pools: &[Pool], m: PoolMark, at: Cycle, h1: Cycle, bytes: u64| {
            let back = admit.src_latency + at.saturating_sub(m.skip);
            let h0 = admit.issue.max(e.saturating_sub(back));
            let pool = &pools[m.pool as usize];
            // A chunk larger than the pool holds the whole pool.
            let bytes = bytes.min(pool.cap);
            if pool.fits(h0, h1, bytes) {
                assert!(n < holds.len(), "more than four pools on one route");
                holds[n] = (m.pool, h0, h1, bytes);
                n += 1;
            } else {
                let next = (pool.room_from(h0, h1, bytes) + back).max(e + 1);
                if retry.is_none_or(|r| next > r.0) {
                    retry = Some((next, m.pool));
                }
            }
        };
        for (h, p) in hops.iter().zip(plan) {
            if let Some(m) = h.pool {
                test(&self.pools, m, at, p.start + p.dur, h.amount);
            }
            at += h.latency;
        }
        if let Some(m) = admit.tail {
            test(&self.pools, m, at, end, hops.last().expect("hops").amount);
        }
        if retry.is_none() {
            for &(p, h0, h1, bytes) in &holds[..n] {
                self.pools[p as usize].hold(h0, h1, bytes);
            }
        }
        retry
    }

    /// Drops calendar intervals that end at or before `now`.
    pub fn prune(&mut self, now: Cycle) {
        for r in &mut self.res {
            if let ResourceState::Flow(f) = &mut r.state {
                f.prune(now);
            }
        }
        for p in &mut self.pools {
            p.prune(now);
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

    /// Statistics of every resource, indexed by `ResourceId`. An in-flight
    /// pool's row has its peak bytes, held byte-cycles and, in
    /// `delay_cycles`, the cycles chunks waited for it.
    pub fn stats(&self) -> Vec<ResourceStats> {
        let mut out = self.stats_rows();
        for p in &self.pools {
            let r = &mut out[p.resource.0 as usize];
            r.peak = p.peak;
            r.held_cycles = p.held as f64;
            r.delay_cycles = p.waited;
        }
        out
    }

    fn stats_rows(&self) -> Vec<ResourceStats> {
        self.res
            .iter()
            .map(|r| match &r.state {
                ResourceState::Flow(f) => ResourceStats {
                    name: r.name.clone(),
                    kind: r.kind,
                    slots: f.slots(),
                    rate: f.core_rate().to_f64(),
                    busy_cycles: whole(f.busy_total),
                    busy_slot_cycles: f.busy_total as f64 / SCALE as f64,
                    busy_read: whole(f.busy_dir[0]),
                    busy_write: whole(f.busy_dir[1]),
                    busy_work: whole(f.busy_dir[2]),
                    bytes: f.bytes,
                    size: 0,
                    held_cycles: 0.0,
                    peak: 0,
                    log: None,
                    culprit_cycles: f.culprit_total,
                    delay_cycles: f.delay_total,
                },
                ResourceState::Occupancy(o) => ResourceStats {
                    name: r.name.clone(),
                    kind: r.kind,
                    slots: 0.0,
                    rate: 0.0,
                    busy_cycles: 0,
                    busy_slot_cycles: 0.0,
                    busy_read: 0,
                    busy_write: 0,
                    busy_work: 0,
                    bytes: 0,
                    size: o.size,
                    held_cycles: 0.0,
                    peak: o.peak,
                    log: None,
                    culprit_cycles: 0,
                    delay_cycles: 0,
                },
            })
            .collect()
    }
}

/// Slot-cycles in [`SCALE`] units, rounded to whole slot-cycles.
fn whole(sc: u128) -> u64 {
    ((sc + SCALE as u128 / 2) / SCALE as u128) as u64
}

/// Sums demands on the same resource and utilization into `out`; drops
/// zero amounts.
fn merge(demand: &[Demand], out: &mut Vec<Demand>, dirs: &mut Vec<DirAmounts>) {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{FlowResource, SCALE};
    use crate::time::Ratio;

    /// The calendar as it was before 2026-10-08: a tree, with the search
    /// that looks up every key it passes. The reference for the deque.
    #[derive(Default)]
    struct Tree(BTreeMap<u64, u64>);

    impl Tree {
        fn used_at(&self, t: u64) -> u64 {
            self.0.range(..=t).next_back().map_or(0, |(_, &v)| v)
        }

        fn earliest(&self, cap: u64, t: u64, d: u64, k: u64) -> u64 {
            use std::ops::Bound::{Excluded, Unbounded};
            if d == 0 {
                return t;
            }
            let free = cap - k.min(cap);
            let mut cand = t;
            'search: loop {
                if self.used_at(cand) > free {
                    match self.0.range((Excluded(cand), Unbounded)).next() {
                        Some((&next, _)) => cand = next,
                        None => return cand,
                    }
                    continue;
                }
                for (&at, &v) in self.0.range((Excluded(cand), Excluded(cand + d))) {
                    if v > free {
                        cand = at;
                        continue 'search;
                    }
                }
                return cand;
            }
        }

        fn add(&mut self, t: u64, d: u64, delta: i64) {
            if d == 0 {
                return;
            }
            let e = t + d;
            let (vt, ve) = (self.used_at(t), self.used_at(e));
            self.0.entry(t).or_insert(vt);
            self.0.entry(e).or_insert(ve);
            for (_, v) in self.0.range_mut(t..e) {
                *v = (*v as i64 + delta) as u64;
            }
            let before =
                |m: &BTreeMap<u64, u64>, x: u64| m.range(..x).next_back().map_or(0, |(_, &v)| v);
            if before(&self.0, t) == self.0[&t] {
                self.0.remove(&t);
            }
            if before(&self.0, e) == self.0[&e] {
                self.0.remove(&e);
            }
        }

        fn prune(&mut self, now: u64) {
            let v = self.used_at(now);
            while let Some((&k, _)) = self.0.first_key_value() {
                if k >= now {
                    break;
                }
                self.0.pop_first();
            }
            if v > 0 {
                self.0.entry(now).or_insert(v);
            }
        }
    }

    #[test]
    fn the_calendar_matches_the_tree_on_random_reservations() {
        // A linear congruential generator: the same sequence every run.
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |n: u64| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (x >> 33) % n
        };
        // Usages in quarter slots, on capacities of 1 to 8 slots.
        let q = SCALE / 4;
        for slots in [1u64, 2, 4, 8] {
            let mut f = FlowResource::new(Ratio::int(64 * slots));
            f.set_beat(64);
            let cap = slots * SCALE;
            assert_eq!(f.cap, cap);
            let mut tree = Tree::default();
            let mut held: Vec<(u64, u64, u64)> = Vec::new();
            let mut now = 0;
            for _ in 0..20_000 {
                match rnd(10) {
                    // Reserve at the earliest window.
                    0..=5 => {
                        let (t, d, k) = (now + rnd(200), 1 + rnd(40), (1 + rnd(4 * slots)) * q);
                        let s = f.earliest(t, d, k);
                        assert_eq!(s, tree.earliest(cap, t, d, k), "earliest({t}, {d}, {k})");
                        f.occupy(s, d, k);
                        tree.add(s, d, k as i64);
                        held.push((s, d, k));
                    }
                    // Release a held window (pool admission does).
                    6 | 7 if !held.is_empty() => {
                        let (s, d, k) = held.swap_remove(rnd(held.len() as u64) as usize);
                        if s >= now {
                            f.release(s, d, k);
                            tree.add(s, d, -(k as i64));
                        }
                    }
                    // Time passes.
                    _ => {
                        now += rnd(30);
                        f.prune(now);
                        tree.prune(now);
                        held.retain(|&(s, _, _)| s >= now);
                    }
                }
                let keys: Vec<(u64, u64)> = f.used.iter().copied().collect();
                let want: Vec<(u64, u64)> = tree.0.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(keys, want);
            }
        }
    }
}
