//! The timing event queue, ordered by (time, seq).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use nu_model::{AwaitMsg, NeutrinoId, TaskId};

use crate::profile::PeId;
use crate::time::Cycle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Poll the Neutrino's program.
    NuResume(NeutrinoId),
    /// Execute the Neutrino's pending request; its cost has elapsed.
    NuExec(NeutrinoId),
    /// The current chunk of `task` on `pe` ended.
    ChunkEnd {
        pe: PeId,
        /// The Neutrino that tracks `task` (another one for a put).
        nu: NeutrinoId,
        task: TaskId,
    },
    /// The next chunk of transfer `task` on `pe` is due for issue
    /// (r2sim spec Section 5.4.1, issue-driven chunks).
    ChunkIssue {
        pe: PeId,
        nu: NeutrinoId,
        task: TaskId,
    },
    /// The completion of `task` reaches the tables (after `c_complete`).
    Complete { nu: NeutrinoId, task: TaskId },
    /// The memory access of the Neutrino's load or store is done.
    MemDone(NeutrinoId),
    /// A fabric message arrives; it waits in the message slab.
    MsgArrive { msg: u32 },
    /// `task` becomes ready after the local resolution latency.
    Ready { nu: NeutrinoId, task: TaskId },
    /// A pending local wait of `nu` completes after the resolution latency.
    WaitDone {
        nu: NeutrinoId,
        wait: nu_model::WaitId,
    },
    /// `nu` moves forwarded invokes from its inbound invoke table into its
    /// scoreboard (after the event that freed an entry).
    DrainInbound(NeutrinoId),
    /// `nu` resends an `Await` that `owner` rejected, after `t_retry`.
    ResendAwait {
        nu: NeutrinoId,
        owner: NeutrinoId,
        msg: AwaitMsg,
    },
    /// Message `msg` reaches the stage after its current hop.
    HopArrive { msg: u32 },
    /// A message finished leaving `stage`: its entry is free.
    StageFree { stage: Stage },
}

/// A message stage of one protocol class (Neutrino spec Section 9.2, ready/valid): the
/// requester's source queue (for invokes its outbound invoke table), or
/// the one-message stage at the end of a hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    /// The source queue of Neutrino `.0` for class `.1`.
    Source(u32, u8),
    /// The stage at the end of control lane `.0` for class `.1`.
    Hop(crate::resource::ResourceId, u8),
}

#[derive(Debug)]
struct Entry {
    time: Cycle,
    seq: u64,
    event: Event,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        (self.time, self.seq) == (other.time, other.seq)
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    // Reversed: BinaryHeap is a max-heap.
    fn cmp(&self, other: &Self) -> Ordering {
        (other.time, other.seq).cmp(&(self.time, self.seq))
    }
}

#[derive(Debug, Default)]
pub struct EventQueue {
    heap: BinaryHeap<Entry>,
    seq: u64,
}

impl EventQueue {
    pub fn push(&mut self, time: Cycle, event: Event) {
        self.seq += 1;
        self.heap.push(Entry {
            time,
            seq: self.seq,
            event,
        });
    }

    pub fn pop(&mut self) -> Option<(Cycle, Event)> {
        self.heap.pop().map(|e| (e.time, e.event))
    }
}
