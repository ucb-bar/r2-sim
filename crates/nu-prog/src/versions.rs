//! The driver's version bookkeeping (architecture Section 3.2, Versions),
//! shared by every graph of one program.
//!
//! * The owner assigns the version of every invoke. For a forwarded invoke
//!   the driver infers it with one counter per remote base address: this
//!   is software inference, exact because the one driver invokes every
//!   producer of a range and its invokes reach the owner in program order.
//! * Owners run no program, so no program start resets their version
//!   tables. The driver keeps the versioned ranges live at each owner
//!   within the owner's table size and resets ranges (`nu.vreset`, a
//!   `VersionReset` message for a remote owner) to make room.
//! * After a window's barrier every consumer of its one-off signals (the
//!   joins on the signal ring) has bound, so the driver releases them
//!   before the next window. A release resets the owner's counter, so the
//!   driver's inference for that base restarts too.

use std::collections::{HashMap, HashSet};

use nu_model::{NeutrinoId, Range};

/// When the driver resets (`vreset`) versioned data ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VresetPolicy {
    /// Only when an owner's table would overflow: the least recently used
    /// range whose last use was before the last barrier (it is done).
    #[default]
    Lru,
    /// Every range a window used, after the window's barrier.
    Barrier,
}

impl VresetPolicy {
    /// Parses "lru" or "barrier".
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "lru" => Ok(Self::Lru),
            "barrier" => Ok(Self::Barrier),
            _ => Err(format!("unknown vreset_policy {s:?} (lru, barrier)")),
        }
    }
}

#[derive(Debug)]
struct Owner {
    nu: NeutrinoId,
    own: Range,
    capacity: usize,
    /// Live versioned base addresses: the window of their last use and the
    /// bound of the range.
    live: HashMap<u64, (u64, u64)>,
    /// Signal bases used in the current window.
    signals: Vec<u64>,
}

#[derive(Debug, Default)]
pub(crate) struct VersionState {
    mode: VresetPolicy,
    /// The inferred version counter of each remote base address.
    remote: HashMap<u64, u64>,
    owners: Vec<Owner>,
    /// The window being built; windows end with a barrier.
    window: u64,
    /// Bases to release before the next window's first invoke: data, then
    /// signals.
    queued_data: Vec<u64>,
    queued_signal: Vec<u64>,
    /// Signal bases carried into the next window (not released).
    kept: HashSet<u64>,
    /// Releases of data ranges, all windows.
    pub(crate) vreset_data: u64,
    /// Releases of one-off signals, all windows.
    pub(crate) vreset_signal: u64,
}

impl VersionState {
    /// Bookkeeping for the owners `(nu, own space, version table size)`.
    pub(crate) fn new(owners: Vec<(NeutrinoId, Range, usize)>) -> Self {
        Self {
            owners: owners
                .into_iter()
                .map(|(nu, own, capacity)| Owner {
                    nu,
                    own,
                    capacity,
                    live: HashMap::new(),
                    signals: Vec::new(),
                })
                .collect(),
            ..Self::default()
        }
    }

    pub(crate) fn set_mode(&mut self, mode: VresetPolicy) {
        self.mode = mode;
    }

    /// The same owners and policy, nothing tracked yet (another driver's
    /// own bookkeeping).
    pub(crate) fn fresh(&self) -> Self {
        Self {
            mode: self.mode,
            owners: self
                .owners
                .iter()
                .map(|o| Owner {
                    nu: o.nu,
                    own: o.own,
                    capacity: o.capacity,
                    live: HashMap::new(),
                    signals: Vec::new(),
                })
                .collect(),
            ..Self::default()
        }
    }

    /// A program Start reset the version tables of the owners whose own
    /// spaces are `spaces` (the gpcs at a stage start): their counters and
    /// live ranges start again, and nothing is left to reset there.
    pub(crate) fn reset_spaces(&mut self, spaces: &[Range]) {
        let inside = |a: u64| spaces.iter().any(|r| r.covers(&Range::from_len(a, 1)));
        self.remote.retain(|a, _| !inside(*a));
        self.queued_data.retain(|a| !inside(*a));
        self.queued_signal.retain(|a| !inside(*a));
        self.kept.retain(|a| !inside(*a));
        for o in &mut self.owners {
            if spaces.contains(&o.own) {
                o.live.clear();
                o.signals.clear();
            }
        }
    }

    /// The inferred version of the next forwarded invoke on base `addr`.
    pub(crate) fn next_version(&mut self, addr: u64) -> u64 {
        let v = self.remote.entry(addr).or_insert(0);
        *v += 1;
        *v
    }

    /// A local invoke on base `addr` got version `v` from its owner.
    pub(crate) fn note_version(&mut self, addr: u64, v: u64) {
        self.remote.insert(addr, v);
    }

    /// A versioned invoke on `dst` in the current window; `signal` for a
    /// join on the signal ring. Panics when the owner's table cannot hold
    /// the window's ranges.
    pub(crate) fn touch(&mut self, dst: Range, signal: bool) {
        let window = self.window;
        let a = dst.base;
        let o = self
            .owners
            .iter_mut()
            .find(|o| o.own.covers(&dst))
            .unwrap_or_else(|| panic!("no owner of versioned range {dst}"));
        if signal && !o.signals.contains(&a) {
            o.signals.push(a);
        }
        match o.live.get(&a).copied() {
            Some((_, bound)) if bound == dst.bound => {
                o.live.insert(a, (window, bound));
                return;
            }
            Some((w, bound)) => {
                // A new range at a live base: two ranges with one base are
                // a software error at the owner, so the old one goes first.
                assert!(
                    w < window,
                    "{dst} and the range to {bound:#x} share a base in one window at {:?}",
                    o.nu
                );
                o.live.remove(&a);
                self.queued_data.push(a);
            }
            None => {}
        }
        if o.live.len() >= o.capacity {
            let victim = o
                .live
                .iter()
                .filter(|(_, (w, _))| *w < window)
                .min_by_key(|(a, (w, _))| (*w, **a))
                .map(|(a, _)| *a)
                .unwrap_or_else(|| {
                    panic!(
                        "window needs more than {} versioned ranges at {:?}",
                        o.capacity, o.nu
                    )
                });
            o.live.remove(&victim);
            self.queued_data.push(victim);
        }
        o.live.insert(a, (window, dst.bound));
    }

    /// A carried join's range: its consumers are in the next window.
    pub(crate) fn keep(&mut self, dst: Range) {
        self.kept.insert(dst.base);
    }

    /// The bases to release before the current window's first invoke. The
    /// owner's counter of a released base restarts, and so does the
    /// inference.
    pub(crate) fn take_releases(&mut self) -> Vec<u64> {
        self.vreset_data += self.queued_data.len() as u64;
        self.vreset_signal += self.queued_signal.len() as u64;
        let mut out = std::mem::take(&mut self.queued_data);
        out.append(&mut self.queued_signal);
        for a in &out {
            self.remote.remove(a);
        }
        out
    }

    /// The window was invoked; its barrier follows.
    pub(crate) fn end_window(&mut self) {
        let kept = std::mem::take(&mut self.kept);
        for o in &mut self.owners {
            for a in std::mem::take(&mut o.signals) {
                if !kept.contains(&a) {
                    o.live.remove(&a);
                    self.queued_signal.push(a);
                }
            }
            if self.mode == VresetPolicy::Barrier {
                let mut used: Vec<u64> = o
                    .live
                    .keys()
                    .filter(|a| !kept.contains(a))
                    .copied()
                    .collect();
                used.sort_unstable();
                for a in &used {
                    o.live.remove(a);
                }
                self.queued_data.extend(used);
            }
        }
        self.window += 1;
    }
}
