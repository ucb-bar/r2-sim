//! A window of the task graph, built before it is invoked, so that every
//! dependency can name the version its producer gets (Section 3.2,
//! Versions).
//!
//! Rules the builder relies on:
//! * Every task is versioned: data ranges, and joins over a ring of sync
//!   ranges (unversioned done joins would stay in the owner's execution
//!   table until their range is reused, more than it holds). A builder
//!   names a dependency by an address; it resolves to the latest earlier
//!   task in program order whose `dst` contains it, and the invoke names
//!   `(base, v)` of that producer.
//! * The owner assigns every version, a forwarded invoke's too. The driver
//!   infers the version of a forwarded invoke by counting its invokes to
//!   the range ([`VersionState`]); invokes of one requester reach the
//!   owner in program order, so the inference is exact (the owner checks it
//!   in debug builds).
//! * A versioned writer also depends on the previous version of its
//!   `dst` in the window: that orders it after that producer.
//! * A dependency with no producer in the window is dropped: the driver
//!   ends each window with a barrier, so earlier producers are done. The
//!   exception is a carried producer of the previous window that may still
//!   run at the barrier ([`Graph::carry`]); a dependency on it is kept.
//! * The versioned ranges of a window must fit the owners' version tables;
//!   the releases that make room, and the releases of the previous
//!   window's one-off signals, go out before the window's first invoke.
//! * A dependency on data at another node names the data directly: the
//!   version disambiguates, so no fresh signal is needed. [`Graph::pin`]
//!   names the producer that is current when it is called, for a consumer
//!   added later; [`Graph::pin_task`] names one task (a join on a reused
//!   ring range).

use std::cell::RefCell;
use std::rc::Rc;

use nu_model::{Dep, Range};
use sim_core::{Descriptor, NuCtx, OpcodeId};

use crate::versions::VersionState;

/// `desc` with the present `deps` and a label. A dependency is named by the
/// base address of the range the consumer reads (an address inside its
/// producer's `dst`).
pub(crate) fn with_deps(mut desc: Descriptor, deps: &[Option<Range>], label: u64) -> Descriptor {
    desc.deps = deps
        .iter()
        .flatten()
        .map(|r| Dep::Unversioned(r.base))
        .collect();
    desc.label = Some(label);
    desc
}

/// The most bytes of one message: 4 beats of 64 bytes (Section 3.3).
const WIRE_LIMIT: u64 = 256;

/// The most deps of one join: its descriptor (20 bytes plus 8 per dep)
/// fits one message.
pub(crate) const MAX_JOIN_DEPS: usize = 29;

/// The deps beyond what fits one message with `desc` (its fixed part plus
/// 8 bytes per dep), leaving one dep for the join that takes them; empty
/// when all fit.
pub(crate) fn excess_deps(desc: &mut Descriptor) -> Vec<Option<Range>> {
    let fixed = desc.wire_bytes() - 8 * desc.deps.len() as u64;
    let max = ((WIRE_LIMIT - fixed) / 8) as usize;
    if desc.deps.len() <= max {
        return Vec::new();
    }
    desc.deps
        .split_off(max - 1)
        .into_iter()
        .map(|d| Some(Range::from_len(d.addr(), 1)))
        .collect()
}

/// Identifies a task of a window.
pub(crate) type TaskIx = usize;

/// Pin tokens are ranges in an address space that no node owns: bit 63,
/// the window in bits 32..63, the pin index times 512 below.
const PIN_BASE: u64 = 1 << 63;
const PIN_STRIDE: u64 = 512;

fn pin_token(window: u64, i: usize) -> Range {
    Range::from_len(
        PIN_BASE | (window << 32) | (i as u64 * PIN_STRIDE),
        PIN_STRIDE,
    )
}

/// (window, index) of a pin token's address.
fn pin_of(a: u64) -> Option<(u64, usize)> {
    (a & PIN_BASE != 0).then(|| {
        let x = a & !PIN_BASE;
        (x >> 32, ((x & 0xffff_ffff) / PIN_STRIDE) as usize)
    })
}

/// A dependency of a task in the window, resolved at flush.
#[derive(Debug, Clone, Copy)]
enum DepRef {
    /// On task `.0` of the window.
    Task(TaskIx),
    /// A key known before the window (a carried producer).
    Key(Dep),
}

#[derive(Default)]
pub(crate) struct Graph {
    tasks: Vec<(Descriptor, Vec<DepRef>)>,
    /// The own space of the driver: its invokes are local; the others are
    /// forwarded.
    driver_own: Option<Range>,
    /// The program's version bookkeeping, shared with its other graphs.
    versions: Rc<RefCell<VersionState>>,
    /// Deps dropped because their producer is in an earlier window.
    pub(crate) dropped: u64,
    /// Tasks invoked through this graph, all windows.
    pub(crate) invoked: u64,
    /// Carried tasks of this window, and the (dst, key) usable in the
    /// next.
    carry_out: Vec<TaskIx>,
    carry_in: Vec<(Range, Dep)>,
    /// (dst, key) of the tasks of the last flushed window, in program
    /// order.
    flushed: Vec<(Range, Dep)>,
    /// Windows flushed so far (the window of a pin token).
    window: u64,
    /// Pins of the current window: what a dependency on the token names,
    /// or none when the producer is in an earlier window.
    pins: Vec<Option<DepRef>>,
    /// Pins of the last flushed window, for [`Graph::key`].
    flushed_pins: Vec<Option<DepRef>>,
    /// Joins added, all windows.
    pub(crate) joins: u64,
}

impl Graph {
    /// An empty graph for a driver whose own space is `own`.
    pub(crate) fn new(own: Range, versions: Rc<RefCell<VersionState>>) -> Self {
        Self {
            driver_own: Some(own),
            versions,
            ..Self::default()
        }
    }

    /// Adds a task; its `dst` is versioned. The deps are addresses (see
    /// [`with_deps`]) or pin tokens.
    pub(crate) fn add(&mut self, mut desc: Descriptor) -> TaskIx {
        desc.versioned = true;
        let signal = desc.opcode == OpcodeId::JOIN;
        if signal {
            self.joins += 1;
        }
        let mut refs = Vec::with_capacity(desc.deps.len() + 1);
        for d in std::mem::take(&mut desc.deps) {
            let Dep::Unversioned(a) = d else {
                refs.push(DepRef::Key(d));
                continue;
            };
            if let Some((w, i)) = pin_of(a) {
                // A pin of an earlier window names a producer that is done
                // at that window's barrier.
                match self
                    .pins
                    .get(i)
                    .copied()
                    .flatten()
                    .filter(|_| w == self.window)
                {
                    Some(x) => refs.push(x),
                    None => self.dropped += 1,
                }
                continue;
            }
            match self.resolve(a) {
                Some(x) => refs.push(x),
                None => self.dropped += 1,
            }
        }
        self.versions.borrow_mut().touch(desc.dst, signal);
        // The previous version of the same range (write after write).
        let prev = self
            .latest(desc.dst.base)
            .filter(|&p| self.tasks[p].0.dst == desc.dst);
        if let Some(p) = prev {
            if !refs.iter().any(|x| matches!(x, DepRef::Task(q) if *q == p)) {
                refs.push(DepRef::Task(p));
            }
        }
        self.tasks.push((desc, refs));
        self.tasks.len() - 1
    }

    /// What a dependency on address `a` names: the latest task of the
    /// window whose `dst` contains it, else a carried producer with that
    /// base; none when the producer is in an earlier window.
    fn resolve(&self, a: u64) -> Option<DepRef> {
        match self.latest(a) {
            Some(p) => Some(DepRef::Task(p)),
            None => self
                .carry_in
                .iter()
                .find(|(dst, _)| dst.base == a)
                .map(|(_, k)| DepRef::Key(*k)),
        }
    }

    /// The latest task whose `dst` contains address `a`.
    fn latest(&self, a: u64) -> Option<TaskIx> {
        self.tasks
            .iter()
            .rposition(|(t, _)| t.dst.base <= a && a < t.dst.bound)
    }

    pub(crate) fn dst(&self, t: TaskIx) -> Range {
        self.tasks[t].0.dst
    }

    /// A token for the current producer of `r` in the window (the latest
    /// task whose `dst` contains its base): a later dependency on the token
    /// binds to that producer even if `r` is written again in between.
    /// With no producer in the window, a dependency on the token is
    /// dropped.
    pub(crate) fn pin(&mut self, r: Range) -> Range {
        // A pin of a pin names the same producer.
        let x = match pin_of(r.base) {
            Some((w, i)) => self
                .pins
                .get(i)
                .copied()
                .flatten()
                .filter(|_| w == self.window),
            None => self.resolve(r.base),
        };
        self.pins.push(x);
        pin_token(self.window, self.pins.len() - 1)
    }

    /// A token for task `t` of the window.
    pub(crate) fn pin_task(&mut self, t: TaskIx) -> Range {
        self.pins.push(Some(DepRef::Task(t)));
        pin_token(self.window, self.pins.len() - 1)
    }

    /// Marks task `t` as carried into the next window: the next window may
    /// depend on its `dst`, so a signal is not released after the barrier.
    pub(crate) fn carry(&mut self, t: TaskIx) {
        self.versions.borrow_mut().keep(self.dst(t));
        self.carry_out.push(t);
    }

    /// The key a program wait after the flush names for range `r` (or a
    /// pin token): the version of the last flushed producer that contains
    /// its base. `None` for a pin whose producer is in an earlier window
    /// (done at its barrier).
    pub(crate) fn key(&self, r: &Range) -> Option<Dep> {
        if let Some((w, i)) = pin_of(r.base) {
            let pinned = self.flushed_pins.get(i).copied().flatten();
            return match pinned.filter(|_| w + 1 == self.window)? {
                DepRef::Task(p) => Some(self.flushed[p].1),
                DepRef::Key(k) => Some(k),
            };
        }
        let a = r.base;
        Some(
            match self
                .flushed
                .iter()
                .rev()
                .find(|(dst, _)| dst.base <= a && a < dst.bound)
            {
                Some(&(_, k)) => k,
                None => Dep::Unversioned(a),
            },
        )
    }

    /// Invokes the window in program order and clears it.
    pub(crate) async fn flush(&mut self, ctx: &NuCtx) {
        let releases = self.versions.borrow_mut().take_releases();
        for a in releases {
            ctx.vreset(a).await;
        }
        let mut keys: Vec<(Range, Dep)> = Vec::with_capacity(self.tasks.len());
        for (mut d, refs) in std::mem::take(&mut self.tasks) {
            d.deps = refs
                .iter()
                .map(|x| match *x {
                    DepRef::Task(p) => keys[p].1,
                    DepRef::Key(k) => k,
                })
                .collect();
            let dst = d.dst;
            let local = self.driver_own.is_some_and(|o| o.covers(&dst));
            // The owner assigns the version; for a forwarded invoke the
            // driver infers it (and the owner checks the inference).
            let inferred = (!local).then(|| self.versions.borrow_mut().next_version(dst.base));
            d.inferred_version = inferred;
            let h = ctx.invoke(d).await;
            let version = inferred.unwrap_or(h.version);
            keys.push((
                dst,
                Dep::Versioned {
                    addr: dst.base,
                    version,
                },
            ));
            self.invoked += 1;
        }
        self.carry_in = std::mem::take(&mut self.carry_out)
            .into_iter()
            .map(|t| keys[t])
            .collect();
        self.flushed = keys;
        self.flushed_pins = std::mem::take(&mut self.pins);
        self.window += 1;
        self.versions.borrow_mut().end_window();
    }
}
