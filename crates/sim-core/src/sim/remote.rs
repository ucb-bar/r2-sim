//! Fabric messages, the cross-Neutrino protocols (architecture Section
//! 3.3), and `spawn_child` (Section 6.1).
//!
//! Every protocol class (invoke, dependency, hierarchy) has its own
//! channel (ready/valid): per class a source queue at the sender, one
//! message stage at the end of every hop of the route (a pipeline register
//! of `hop_queue` messages), and the receiver. For invokes the source queue
//! is the requester's outbound invoke table and the receiver the owner's
//! inbound invoke table. A message is transmitted over a hop's control
//! lane only when the stage at its end has room; it holds that stage from
//! the start of its transmission until it moves on. Stages never reorder,
//! so one requester's invokes reach an owner in program order. A stage
//! head that cannot move blocks the stage (head-of-line); blocked stages
//! are served in the order they blocked. Arrival at the receiver is the
//! time the message leaves the last stage into the receiver's table.
//!
//! Ticket flow (CHI-style) differs only at a full inbound invoke table.
//! The invoke channel is full duplex: the owner answers every accepted
//! `Invoke` or `VersionReset` with an `InvokeAck`, and the requester keeps
//! each request in its outbound invoke table until its `InvokeAck` or
//! `RetryAck` (one beat each). On a full inbound table the owner answers
//! `RetryAck` and sets the requester's bit in its ticket bitmap; while the
//! bit is set, every request of that requester without a ticket gets a
//! `Reject` (one beat). Channels keep order, so at `RetryAck(X)` the
//! requester knows that every earlier request to that owner was accepted
//! and that every later one already sent will be rejected: it marks X and
//! those for replay and sends nothing new. When an inbound entry frees,
//! the owner reserves it and sends a `PCrdGrant` to the next set bit
//! (round-robin); the requester replays its oldest marked request with
//! the ticket, which the owner accepts into the reserved entry. The bit
//! clears when the last replayed request is accepted. Acceptance order is
//! program order.
//!
//! A message is `ceil(bytes / msg_bytes)` beats of `msg_bytes`, at most 4.
//! `VersionReset` travels in the invoke channel and takes an entry of the
//! owner's inbound invoke table, as an `Invoke` does (ready/valid: it
//! reserves the entry when it leaves the last stage; ticket flow: a full
//! table answers it `RetryAck` and it is replayed). It takes effect when it
//! reaches the head of the table: it frees the version entry of its base
//! (absent: no-op) and needs no other resource. So a requester's
//! `VersionReset` takes effect after its earlier invokes to that owner have
//! entered (architecture Section 3.3; decided 2026-10-20). A local `vreset`
//! is immediate. A rejected `Await` is resent
//! after `t_retry`.

use nu_model::{
    handle_await, AckAction, AwaitAck, AwaitMsg, Dep, NeutrinoId, Purpose, RequestId, TaskId,
};

use super::{trace, EnterFail, Entry, Inbound, ProgState, Sim};
use crate::event::{Event, Stage};
use crate::profile::{Descriptor, OpcodeId};
use crate::program::{NuCtx, Response, StartFn};
use crate::program::{Request, TaskHandle};
use crate::stats::{FatalError, Stall};
use crate::time::Cycle;
use crate::tracer::{self as tr, Owner, Table, TableEntry, TaskCall, Who};

/// The id of the inbound ref of request `request` of `requester`.
pub(super) fn ref_id(requester: NeutrinoId, request: RequestId) -> u64 {
    (requester.0 as u64) << 32 | request.0 as u64
}

/// The id of the inbound invoke entry of outbound entry `seq` of `src`.
fn inbound_id(src: NeutrinoId, seq: u64) -> u64 {
    (src.0 as u64) << 40 | seq
}

/// Most beats of one message.
pub(crate) const MAX_BEATS: u64 = 4;

/// The protocol classes: each has its own channel.
pub(crate) const INVOKE: u8 = 0;
pub(crate) const DEPENDENCY: u8 = 1;
pub(crate) const HIERARCHY: u8 = 2;

/// The name of class `c`.
pub(crate) fn class_name(c: u8) -> &'static str {
    ["invoke", "dependency", "hierarchy"][c as usize]
}

pub(crate) enum MsgKind {
    /// Requester to owner: tell me when the producer of a key is done.
    Await { msg: AwaitMsg },
    /// Owner to requester: registered, done or reject.
    AwaitAck { req: RequestId, ack: AwaitAck },
    /// Owner to requester: the registered producer completed.
    Resolve { req: RequestId },
    /// Parent to child: start this program; report to `task`.
    Start { task: TaskId, start: StartFn },
    /// Child to parent: the program of spawn task `task` returned.
    Return { task: TaskId },
    /// Invoker to owner: a forwarded invoke with its descriptor, and the
    /// invoker's call for a tracer.
    Invoke {
        desc: Box<Descriptor>,
        call: TaskCall,
    },
    /// Requester to owner: free the version table entry at base address
    /// `addr` (an absent entry is a no-op).
    VersionReset { addr: u64 },
    /// Owner to requester, ticket flow: the request was accepted.
    InvokeAck,
    /// Owner to requester, ticket flow: the inbound invoke table was full;
    /// the requester holds a ticket.
    RetryAck,
    /// Owner to requester, ticket flow: the request came while the
    /// requester's bit was set.
    Reject,
    /// Owner to requester, ticket flow: an inbound entry is reserved for
    /// the requester's oldest request marked for replay.
    PCrdGrant,
}

/// The names of the protocol message kinds.
pub const MESSAGE_KINDS: [&str; 11] = [
    "Await",
    "AwaitAck",
    "Resolve",
    "Start",
    "Return",
    "Invoke",
    "VersionReset",
    "InvokeAck",
    "RetryAck",
    "Reject",
    "PCrdGrant",
];

impl MsgKind {
    /// The kind's index in [`MESSAGE_KINDS`].
    fn index(&self) -> u8 {
        match self {
            MsgKind::Await { .. } => 0,
            MsgKind::AwaitAck { .. } => 1,
            MsgKind::Resolve { .. } => 2,
            MsgKind::Start { .. } => 3,
            MsgKind::Return { .. } => 4,
            MsgKind::Invoke { .. } => 5,
            MsgKind::VersionReset { .. } => 6,
            MsgKind::InvokeAck => 7,
            MsgKind::RetryAck => 8,
            MsgKind::Reject => 9,
            MsgKind::PCrdGrant => 10,
        }
    }

    fn name(&self) -> &'static str {
        MESSAGE_KINDS[self.index() as usize]
    }

    /// The channel of the message.
    fn class(&self) -> u8 {
        match self {
            MsgKind::Invoke { .. }
            | MsgKind::VersionReset { .. }
            | MsgKind::InvokeAck
            | MsgKind::RetryAck
            | MsgKind::Reject
            | MsgKind::PCrdGrant => INVOKE,
            MsgKind::Start { .. } | MsgKind::Return { .. } => HIERARCHY,
            _ => DEPENDENCY,
        }
    }

    /// Bytes on the wire before rounding to beats.
    fn bytes(&self) -> u64 {
        match self {
            MsgKind::Invoke { desc, .. } => desc.wire_bytes(),
            _ => 1,
        }
    }
}

/// Flow control of forwarded invokes (Section 3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Flow {
    /// Ready/valid: an invoke waits in the channel until the owner's
    /// inbound invoke table has room; nothing is rejected.
    #[default]
    ReadyValid,
    /// As ready/valid, but a full inbound invoke table answers with a
    /// `RetryAck` and a ticket and grants the entry later (`PCrdGrant`);
    /// requests are held at the requester until acknowledged.
    Ticket,
}

/// Owner side of ticket flow.
#[derive(Debug, Default)]
pub(crate) struct Tickets {
    /// Requesters whose bit is set, with their requests not yet accepted
    /// (the one answered `RetryAck` and the rejected ones).
    held: std::collections::BTreeMap<u32, u32>,
    /// Requesters with a grant (or its resend) in flight, and when their
    /// entry was reserved.
    granted: std::collections::BTreeMap<u32, Cycle>,
    /// The requester granted last (round-robin).
    last: u32,
    /// The most bits set at once.
    pub(crate) peak: usize,
}

/// Ticket flow, requester side: a request held until it is acknowledged.
#[derive(Debug)]
pub(crate) struct Held {
    seq: u64,
    kind: HeldKind,
    state: HeldState,
}

#[derive(Debug, Clone)]
enum HeldKind {
    Invoke(Box<Descriptor>, TaskCall),
    VersionReset(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeldState {
    /// In the source queue, not yet sent.
    Queued,
    /// Sent (first time or replayed).
    Sent,
    /// Marked for replay.
    Replay,
}

/// Ticket flow, requester side: a ticket at one owner.
#[derive(Debug, Default)]
pub(crate) struct Ticket {
    /// Requests to replay, oldest first (sequence numbers).
    replay: std::collections::VecDeque<u64>,
    /// Rejects still to come.
    rejects_due: u32,
    /// When the `RetryAck` arrived.
    since: Cycle,
}

/// One message in the fabric.
pub(crate) struct Msg {
    from: NeutrinoId,
    to: NeutrinoId,
    kind: MsgKind,
    class: u8,
    beats: u64,
    /// The route's hops and the latency before the first one.
    hops: Vec<crate::topology::Hop>,
    lead: Cycle,
    /// The next hop to take.
    next: usize,
    /// The sender's beat rate. A message is held whole in each hop stage
    /// (store and forward), so a hop is not narrowed by the hops before
    /// it: it moves at the smaller of this rate and its own.
    rate: crate::time::Ratio,
    /// Part of an `Await` retry (does not count as progress).
    retry: bool,
    /// Ticket flow: a replay that carries the ticket.
    ticketed: bool,
    /// A request of the invoke channel: its outbound invoke entry at its
    /// requester (a sequence number).
    seq: u64,
    /// An invoke: when the owner reserved its inbound invoke entry.
    reserved: Option<Cycle>,
    /// What the message is for, for a tracer.
    owner: Owner,
}

/// One stage of a channel.
#[derive(Debug, Default)]
pub(crate) struct StageQ {
    /// Messages in the stage, in order.
    fifo: std::collections::VecDeque<u32>,
    /// Entries held: messages on their way into the stage, in it, and
    /// leaving it.
    occ: usize,
    /// Stages whose head waits for room here, in the order they blocked.
    waiters: std::collections::VecDeque<Stage>,
    /// Since when the head could not move.
    hol_since: Option<Cycle>,
    /// Cycles the head could not move.
    pub(crate) hol: Cycle,
    /// The most entries held at once.
    pub(crate) peak: usize,
}

impl StageQ {
    /// Entries held.
    pub(crate) fn occ(&self) -> usize {
        self.occ
    }
}

impl Sim {
    /// Per stage that held a message: its head-of-line cycles and its peak
    /// (Stats).
    pub(super) fn stage_stats(&self) -> Vec<crate::stats::StageStat> {
        let mut out: Vec<_> = self
            .stages
            .iter()
            .filter(|(_, q)| q.peak > 0)
            .map(|(s, q)| crate::stats::StageStat {
                stage: self.stage_name(*s),
                hol_cycles: q.hol + q.hol_since.map_or(0, |t| self.now - t),
                peak: q.peak,
            })
            .collect();
        out.sort_by(|a, b| a.stage.cmp(&b.stage));
        out
    }

    /// Sends a message: it enters the sender's source queue of its class.
    pub(super) fn send(&mut self, from: NeutrinoId, to: NeutrinoId, kind: MsgKind) {
        self.send_msg(from, to, kind, false);
    }

    fn send_msg(&mut self, from: NeutrinoId, to: NeutrinoId, kind: MsgKind, retry: bool) {
        let owner = self.msg_owner(from, to, &kind);
        self.send_owned(from, to, kind, retry, owner);
    }

    /// [`Sim::send`] with the message's owner for a tracer (trace-raw 1.6).
    fn send_owned(
        &mut self,
        from: NeutrinoId,
        to: NeutrinoId,
        kind: MsgKind,
        retry: bool,
        owner: Owner,
    ) {
        let Some((id, class)) = self.make_msg(from, to, kind, retry, owner) else {
            return;
        };
        let src = Stage::Source(from.0, class);
        self.enqueue(src, id);
        self.advance(src);
    }

    /// What message `kind` from `from` to `to` is for (trace-raw 1.6): an
    /// invoke's task, the consumer of a remote dependency (or the waiting
    /// program), a spawn's task, or the sending program.
    fn msg_owner(&self, from: NeutrinoId, to: NeutrinoId, kind: &MsgKind) -> Owner {
        if self.tracer.is_none() {
            return Owner::None;
        }
        let spawn = |nu: NeutrinoId, t: TaskId| Owner::Task(self.nus[nu.0 as usize].rec(t).gid);
        match kind {
            MsgKind::Invoke { call, .. } => Owner::Invoke {
                src: call.src,
                n: call.n,
            },
            MsgKind::Await { msg } => self.request_owner(msg.from, msg.request),
            MsgKind::AwaitAck { req, .. } | MsgKind::Resolve { req } => {
                self.request_owner(to, *req)
            }
            MsgKind::Start { task, .. } => spawn(from, *task),
            MsgKind::Return { task } => spawn(to, *task),
            _ => self.program_owner(from),
        }
    }

    /// The program running on `nu`, by its spawn task (`Owner::None` when
    /// `nu` runs none). Resolved when a message is made, so that its hops
    /// keep the owner after the program returns.
    fn program_owner(&self, nu: NeutrinoId) -> Owner {
        let n = &self.nus[nu.0 as usize];
        if n.program.is_none() {
            return Owner::None;
        }
        Owner::Program {
            spawn: n
                .parent_task
                .map(|(p, t)| self.nus[p.0 as usize].rec(t).gid),
        }
    }

    /// The owner of remote request `req` of `nu`: its consumer task, or
    /// `nu`'s program for a program wait.
    fn request_owner(&self, nu: NeutrinoId, req: RequestId) -> Owner {
        let n = &self.nus[nu.0 as usize];
        match n.reqs.purpose(req) {
            Some(Purpose::TaskDep(t)) => Owner::Task(n.rec(t).gid),
            Some(Purpose::Wait(_)) => self.program_owner(nu),
            None => Owner::None,
        }
    }

    /// The child of the root that `nu` is in (`None`: the root).
    fn subtree_of(&self, nu: NeutrinoId) -> Option<NeutrinoId> {
        let mut cur = nu;
        loop {
            match self.topo.node(cur).parent {
                None => return None,
                Some(p) if self.topo.node(p).parent.is_none() => return Some(cur),
                Some(p) => cur = p,
            }
        }
    }

    /// Builds a message on the control route from `from` to `to` and
    /// counts it; `None` (and a fatal error) for more than 4 beats.
    fn make_msg(
        &mut self,
        from: NeutrinoId,
        to: NeutrinoId,
        kind: MsgKind,
        retry: bool,
        owner: Owner,
    ) -> Option<(u32, u8)> {
        let beats = kind.bytes().div_ceil(self.msg_bytes).max(1);
        if beats > MAX_BEATS {
            let what = match &kind {
                MsgKind::Invoke { desc, .. } => format!(
                    " (opcode {}, {} deps, label {:?})",
                    desc.opcode.0,
                    desc.deps.len(),
                    desc.label
                ),
                _ => String::new(),
            };
            let msg = format!(
                "{} message of {} bytes exceeds {MAX_BEATS} beats of {} bytes{what}",
                kind.name(),
                kind.bytes(),
                self.msg_bytes
            );
            self.fatal(from, None, FatalError::Invalid(msg));
            return None;
        }
        self.stats.beats[beats as usize - 1] += 1;
        let path = self.topo.control_path(from, to);
        let (mut hops, mut lead) = (path.hops.clone(), path.lead);
        // Spawns are not remote-protocol messages.
        if !matches!(kind, MsgKind::Start { .. } | MsgKind::Return { .. }) {
            lead += self.msg_latency;
        }
        if self.zero_time {
            lead = 0;
            for h in &mut hops {
                h.latency = 0;
            }
        }
        *self
            .stats
            .messages_by_kind
            .entry(kind.name().to_string())
            .or_insert(0) += 1;
        self.stats.messages += 1;
        if self.subtree_of(from) != self.subtree_of(to) {
            *self
                .stats
                .cross_messages_by_kind
                .entry(kind.name().to_string())
                .or_insert(0) += 1;
        }
        let class = kind.class();
        let msg = Msg {
            from,
            to,
            kind,
            class,
            beats,
            hops,
            lead,
            next: 0,
            rate: crate::time::Ratio::int(self.nus[from.0 as usize].cfg.beat_bytes),
            retry,
            ticketed: false,
            seq: 0,
            reserved: None,
            owner,
        };
        let id = match self.msg_free.pop() {
            Some(i) => {
                self.msgs[i as usize] = Some(msg);
                i
            }
            None => {
                self.msgs.push(Some(msg));
                self.msgs.len() as u32 - 1
            }
        };
        Some((id, class))
    }

    /// Puts message `id` into stage `s` (its entry is already held for a
    /// hop stage; a source entry is taken here).
    fn enqueue(&mut self, s: Stage, id: u32) {
        let q = self.stages.entry(s).or_default();
        if matches!(s, Stage::Source(..)) {
            q.occ += 1;
        }
        q.fifo.push_back(id);
        q.peak = q.peak.max(q.occ);
    }

    /// Entries a stage can hold.
    fn stage_cap(&self, s: Stage) -> usize {
        match s {
            Stage::Source(nu, INVOKE) => self.nus[nu as usize].cfg.outbound_invokes,
            Stage::Source(..) => usize::MAX,
            Stage::Hop(..) => self.hop_queue,
        }
    }

    fn stage_occ(&self, s: Stage) -> usize {
        self.stages.get(&s).map_or(0, |q| q.occ)
    }

    /// Moves the messages at the head of stage `s` on while the next stage
    /// (or the receiver) has room.
    pub(super) fn advance(&mut self, s: Stage) {
        loop {
            let Some(&id) = self.stages.get(&s).and_then(|q| q.fifo.front()) else {
                return;
            };
            let m = self.msgs[id as usize].as_ref().expect("message in flight");
            // Ticket flow: a requester that holds a ticket sends only its
            // replays.
            if let Stage::Source(nu, INVOKE) = s {
                if !m.ticketed && !self.nus[nu as usize].ticket.is_empty() {
                    return;
                }
            }
            let (to, class, next, retry) = (m.to, m.class, m.next, m.retry);
            let (from, kind, owner) = (m.from, m.kind.index(), m.owner);
            if let Some(&hop) = m.hops.get(next) {
                let t = Stage::Hop(hop.lane, class);
                if self.stage_occ(t) >= self.stage_cap(t) {
                    return self.block(s, t);
                }
                let (beats, lead) = (m.beats, m.lead);
                self.stages.entry(t).or_default().occ += 1;
                self.leave(s);
                self.sent_from_source(s, id);
                let earliest = self.now + if next == 0 { lead } else { 0 };
                let bytes = beats * self.msg_bytes;
                let who = Who::Message {
                    kind,
                    from,
                    to,
                    hop: next,
                    bytes,
                    owner,
                };
                // A message is a write toward its receiver.
                let rate = self.msgs[id as usize].as_ref().expect("message").rate;
                let hop_path = crate::resource::PathHop {
                    resource: hop.lane,
                    amount: bytes,
                    dir: crate::resource::Dir::Write,
                    latency: 0,
                    pool: None,
                };
                let beat = rate.num / rate.den;
                let end = self.reserve_message(hop_path, beat, earliest, who).end;
                self.close_outbound(s, id, end);
                self.msgs[id as usize].as_mut().expect("message").next += 1;
                self.queue.push(end, Event::StageFree { stage: s });
                self.queue
                    .push(end + hop.latency, Event::HopArrive { msg: id });
            } else {
                // The receiver. A ready/valid invoke or VersionReset needs a
                // free entry in the owner's inbound invoke table.
                let is_invoke = matches!(
                    m.kind,
                    MsgKind::Invoke { .. } | MsgKind::VersionReset { .. }
                );
                let lead = if m.hops.is_empty() { m.lead } else { 0 };
                let free = (m.hops.is_empty() && self.tracer.is_some()).then(|| Who::Message {
                    kind,
                    from,
                    to,
                    hop: 0,
                    bytes: m.beats * self.msg_bytes,
                    owner: m.owner,
                });
                if is_invoke && self.flow == Flow::ReadyValid {
                    let n = &self.nus[to.0 as usize];
                    if n.inbound.len() + n.inbound_reserved >= n.cfg.inbound_invokes {
                        return self.block_inbound(s, to);
                    }
                    self.nus[to.0 as usize].inbound_reserved += 1;
                    let now = self.now;
                    self.msgs[id as usize].as_mut().expect("message").reserved = Some(now);
                }
                self.close_outbound(s, id, self.now);
                self.leave(s);
                self.sent_from_source(s, id);
                let q = self.stages.get_mut(&s).expect("stage");
                q.occ -= 1;
                self.queue
                    .push(self.now + lead, Event::MsgArrive { msg: id });
                if let Some(w) = free {
                    let now = self.now;
                    trace!(self, t => t.message_free(w, now, now + lead));
                }
                self.wake(s);
            }
            if !retry {
                self.last_progress = self.now;
            }
        }
    }

    /// The head of stage `s` leaves it (its entry frees later).
    fn leave(&mut self, s: Stage) {
        let now = self.now;
        let q = self.stages.get_mut(&s).expect("stage");
        q.fifo.pop_front();
        if let Some(t) = q.hol_since.take() {
            q.hol += now - t;
        }
    }

    /// The head of `s` waits for room in `t`.
    fn block(&mut self, s: Stage, t: Stage) {
        let now = self.now;
        self.stages
            .get_mut(&s)
            .expect("stage")
            .hol_since
            .get_or_insert(now);
        let w = &mut self.stages.entry(t).or_default().waiters;
        if !w.contains(&s) {
            w.push_back(s);
        }
    }

    /// The head of `s` waits for a free entry in `nu`'s inbound invoke
    /// table.
    fn block_inbound(&mut self, s: Stage, nu: NeutrinoId) {
        let now = self.now;
        self.stages
            .get_mut(&s)
            .expect("stage")
            .hol_since
            .get_or_insert(now);
        let w = &mut self.nus[nu.0 as usize].inbound_waiters;
        if !w.contains(&s) {
            w.push_back(s);
        }
    }

    /// An entry of stage `s` is free: the stages waiting for it move on,
    /// in the order they blocked, and a program stalled on its outbound
    /// invoke table retries.
    fn wake(&mut self, s: Stage) {
        let waiters = std::mem::take(&mut self.stages.get_mut(&s).expect("stage").waiters);
        for w in waiters {
            self.advance(w);
        }
        if let Stage::Source(nu, INVOKE) = s {
            self.unstall(NeutrinoId(nu));
        }
    }

    /// A message finished its transmission out of stage `stage`.
    pub(super) fn stage_free(&mut self, stage: Stage) {
        self.stages.get_mut(&stage).expect("stage").occ -= 1;
        self.wake(stage);
    }

    /// Message `id` reached the stage at the end of its last hop.
    pub(super) fn hop_arrive(&mut self, id: u32) {
        let m = self.msgs[id as usize].as_ref().expect("message in flight");
        let s = Stage::Hop(m.hops[m.next - 1].lane, m.class);
        self.enqueue(s, id);
        self.advance(s);
    }

    /// The stages waiting for `nu`'s inbound invoke table move on.
    fn wake_inbound(&mut self, nu: NeutrinoId) {
        let waiters = std::mem::take(&mut self.nus[nu.0 as usize].inbound_waiters);
        for w in waiters {
            self.advance(w);
        }
    }

    /// Stages that hold messages, for the deadlock report.
    pub(super) fn stage_states(&self) -> Vec<crate::stats::StageState> {
        let mut out: Vec<_> = self
            .stages
            .iter()
            .filter(|(_, q)| q.occ > 0)
            .map(|(s, q)| crate::stats::StageState {
                stage: self.stage_name(*s),
                messages: q.occ,
            })
            .collect();
        out.sort_by(|a, b| a.stage.cmp(&b.stage));
        out
    }

    /// `<lane or Neutrino>.<class>` of stage `s`.
    pub(super) fn stage_name(&self, s: Stage) -> String {
        match s {
            Stage::Source(nu, c) => format!(
                "{}.source.{}",
                self.nus[nu as usize].cfg.name,
                class_name(c)
            ),
            Stage::Hop(r, c) => format!("{}.{}", self.res.get(r).name, class_name(c)),
        }
    }

    pub(super) fn msg_arrive(&mut self, id: u32) {
        let mut msg = self.msgs[id as usize].take().expect("message in flight");
        self.msg_free.push(id);
        let (from, to) = (msg.from, msg.to);
        if self.flow == Flow::Ticket && msg.class == INVOKE {
            let accept = matches!(
                msg.kind,
                MsgKind::Invoke { .. } | MsgKind::VersionReset { .. }
            );
            if accept {
                // An answer is for its request (trace-raw 1.6): the invoked
                // task, or the program that sent a VersionReset.
                let owner = msg.owner;
                match self.ticket_answer(from, to, &mut msg) {
                    MsgKind::InvokeAck => {
                        self.send_owned(to, from, MsgKind::InvokeAck, false, owner)
                    }
                    answer => {
                        self.send_owned(to, from, answer, false, owner);
                        return self.grant(to);
                    }
                }
            }
        }
        match msg.kind {
            MsgKind::Await { msg: m } => {
                let ack = match handle_await(&mut self.nus[to.0 as usize].tables, &m) {
                    Ok((ack, fx)) => {
                        if self.tracer.is_some() {
                            self.trace_await(&m, ack, &fx);
                        }
                        self.effects(to, None, fx);
                        if ack == AwaitAck::Reject {
                            self.stats.rejects += 1;
                        }
                        ack
                    }
                    Err(e) => return self.fatal(to, None, FatalError::Dep(e)),
                };
                let kind = MsgKind::AwaitAck {
                    req: m.request,
                    ack,
                };
                self.send_msg(to, from, kind, ack == AwaitAck::Reject);
            }
            MsgKind::AwaitAck { req, ack } => match self.nus[to.0 as usize].reqs.on_ack(req, ack) {
                AckAction::Satisfied(purpose) => {
                    self.trace_dep_done(to, req.0, purpose, 0);
                    self.dep_satisfied(to, purpose)
                }
                AckAction::Wait => {}
                AckAction::Retry { msg, .. } => {
                    let now = self.now;
                    trace!(self, t => t.entry_open(&TableEntry {
                        flags: tr::DEP_REJECTED,
                        ..TableEntry::new(to, Table::OutboundDeps, req.0 as u64)
                    }, now));
                    let t = self.nus[to.0 as usize].cfg.t_retry;
                    self.queue.push(
                        self.now + t,
                        Event::ResendAwait {
                            nu: to,
                            owner: from,
                            msg,
                        },
                    );
                }
            },
            MsgKind::Resolve { req } => {
                let purpose = self.nus[to.0 as usize].reqs.on_resolve(req);
                self.trace_dep_done(to, req.0, purpose, tr::DEP_REGISTERED);
                self.dep_satisfied(to, purpose);
            }
            MsgKind::Start { task, start } => self.start_child(from, to, task, start),
            MsgKind::Invoke { desc, call } => {
                if self.flow == Flow::ReadyValid {
                    self.nus[to.0 as usize].inbound_reserved -= 1;
                }
                if self.tracer.is_some() {
                    let e = TableEntry {
                        base: desc.dst.base,
                        len: desc.dst.len(),
                        peer: Some(from),
                        ..TableEntry::new(to, Table::InboundInvokes, inbound_id(from, call.seq))
                    };
                    let start = msg.reserved.unwrap_or(self.now);
                    trace!(self, t => t.entry_open(&e, start));
                }
                let n = &mut self.nus[to.0 as usize];
                n.inbound
                    .push_back(Inbound::Invoke(Entry::Inbound(call), *desc));
                n.inbound_peak = n.inbound_peak.max(n.inbound.len());
                self.drain_inbound(to);
            }
            MsgKind::VersionReset { addr } => {
                // An entry of the inbound invoke table; it takes effect at
                // the head (drain_inbound), after the requester's earlier
                // invokes have entered.
                if self.flow == Flow::ReadyValid {
                    self.nus[to.0 as usize].inbound_reserved -= 1;
                }
                let seq = msg.seq;
                if self.tracer.is_some() {
                    let e = TableEntry {
                        base: addr,
                        peer: Some(from),
                        flags: tr::INVOKE_VRESET,
                        ..TableEntry::new(to, Table::InboundInvokes, inbound_id(from, seq))
                    };
                    let start = msg.reserved.unwrap_or(self.now);
                    trace!(self, t => t.entry_open(&e, start));
                }
                let n = &mut self.nus[to.0 as usize];
                n.inbound.push_back(Inbound::VReset { from, seq, addr });
                n.inbound_peak = n.inbound_peak.max(n.inbound.len());
                self.drain_inbound(to);
            }
            MsgKind::Return { task } => self.complete_after_charge(to, task),
            MsgKind::InvokeAck => self.ack_arrive(to, from),
            MsgKind::RetryAck => self.retry_arrive(to, from),
            MsgKind::Reject => self.reject_arrive(to, from),
            MsgKind::PCrdGrant => self.grant_arrive(to, from),
        }
        if self.flow == Flow::Ticket {
            self.grant(to);
        }
    }

    /// Ticket flow, owner side: the answer to request `msg` from `req`. A
    /// replay with the ticket is accepted into its reserved entry; while
    /// `req`'s bit is set any other request is rejected; an invoke that
    /// finds the inbound invoke table full sets the bit (`RetryAck`).
    /// Otherwise the request is accepted (`InvokeAck`).
    fn ticket_answer(&mut self, req: NeutrinoId, owner: NeutrinoId, msg: &mut Msg) -> MsgKind {
        let n = &mut self.nus[owner.0 as usize];
        if msg.ticketed {
            msg.reserved = n.tickets.granted.remove(&req.0);
            n.inbound_reserved -= 1;
            let left = n.tickets.held.get_mut(&req.0).expect("bit of a replay");
            *left -= 1;
            if *left == 0 {
                n.tickets.held.remove(&req.0);
            }
            return MsgKind::InvokeAck;
        }
        if let Some(left) = n.tickets.held.get_mut(&req.0) {
            *left += 1;
            return MsgKind::Reject;
        }
        let full = n.inbound.len() + n.inbound_reserved >= n.cfg.inbound_invokes;
        if matches!(
            msg.kind,
            MsgKind::Invoke { .. } | MsgKind::VersionReset { .. }
        ) && full
        {
            n.tickets.held.insert(req.0, 1);
            n.tickets.peak = n.tickets.peak.max(n.tickets.held.len());
            self.ticket_stats.tickets[req.0 as usize] += 1;
            return MsgKind::RetryAck;
        }
        MsgKind::InvokeAck
    }

    /// Ticket flow, owner side: while inbound entries are free, reserve
    /// one for the next requester with a set bit and no grant in flight
    /// (round-robin after the last one granted) and send it a `PCrdGrant`.
    pub(super) fn grant(&mut self, owner: NeutrinoId) {
        loop {
            let n = &mut self.nus[owner.0 as usize];
            if n.inbound.len() + n.inbound_reserved >= n.cfg.inbound_invokes {
                return;
            }
            let t = &n.tickets;
            let next = t
                .held
                .keys()
                .copied()
                .filter(|r| !t.granted.contains_key(r))
                .min_by_key(|&r| (r <= t.last, r));
            let Some(r) = next else {
                return;
            };
            n.tickets.granted.insert(r, self.now);
            n.tickets.last = r;
            n.inbound_reserved += 1;
            // For the requester's replays: its program (trace-raw 1.6).
            let who = self.program_owner(NeutrinoId(r));
            self.send_owned(owner, NeutrinoId(r), MsgKind::PCrdGrant, false, who);
        }
    }

    /// Ticket flow, requester side: message `id` left the invoke source
    /// queue `s`; its held request is sent.
    fn sent_from_source(&mut self, s: Stage, id: u32) {
        let Stage::Source(nu, INVOKE) = s else {
            return;
        };
        let now = self.now;
        if let Some(MsgKind::Invoke { call, .. }) =
            self.msgs[id as usize].as_mut().map(|m| &mut m.kind)
        {
            call.sent = Some(now);
        }
        if self.flow != Flow::Ticket {
            return;
        }
        let m = self.msgs[id as usize].as_ref().expect("message");
        let (to, seq) = (m.to.0, m.seq);
        if let Some(h) = self.nus[nu as usize]
            .held
            .get_mut(&to)
            .and_then(|q| q.iter_mut().find(|h| h.seq == seq))
        {
            h.state = HeldState::Sent;
        }
    }

    /// Ticket flow, requester side: `owner` accepted the oldest request
    /// held for it; its outbound entry frees.
    fn ack_arrive(&mut self, nu: NeutrinoId, owner: NeutrinoId) {
        let n = &mut self.nus[nu.0 as usize];
        let h = n.held.get_mut(&owner.0).and_then(|q| q.pop_front());
        let seq = h.as_ref().map(|h| h.seq);
        match h {
            Some(h) if h.state == HeldState::Sent => {}
            h => {
                let msg = format!("InvokeAck from {owner:?} for {h:?}");
                return self.fatal(nu, None, FatalError::Invalid(msg));
            }
        }
        n.held_len -= 1;
        self.ticket_stats.acks[nu.0 as usize] += 1;
        if let Some(seq) = seq {
            let now = self.now;
            trace!(self, t => t.entry_close(&TableEntry::new(nu, Table::OutboundInvokes, seq), now));
        }
        self.unstall(nu);
    }

    /// Ticket flow, requester side: `owner` answered the oldest request
    /// held for it with `RetryAck`. Every earlier request was acknowledged;
    /// every later one already sent will be rejected. All of them are
    /// marked for replay, oldest first.
    fn retry_arrive(&mut self, nu: NeutrinoId, owner: NeutrinoId) {
        let now = self.now;
        let n = &mut self.nus[nu.0 as usize];
        let q = n.held.get_mut(&owner.0);
        let ok = q
            .as_ref()
            .and_then(|q| q.front())
            .is_some_and(|h| h.state == HeldState::Sent);
        if !ok || n.ticket.contains_key(&owner.0) {
            let msg = format!("RetryAck from {owner:?} with no request sent first");
            return self.fatal(nu, None, FatalError::Invalid(msg));
        }
        let mut t = Ticket {
            since: now,
            ..Ticket::default()
        };
        for h in q.expect("held").iter_mut() {
            if h.state == HeldState::Sent {
                h.state = HeldState::Replay;
                t.replay.push_back(h.seq);
            }
        }

        t.rejects_due = t.replay.len() as u32 - 1;
        if let Some(tracer) = self.tracer.as_mut() {
            for &seq in &t.replay {
                let e = TableEntry {
                    flags: tr::INVOKE_REPLAYED,
                    ..TableEntry::new(nu, Table::OutboundInvokes, seq)
                };
                tracer.entry_open(&e, now);
            }
        }
        let n = &mut self.nus[nu.0 as usize];
        n.ticket.insert(owner.0, t);
        n.ticket_peak = n.ticket_peak.max(n.ticket.len());
    }

    /// Ticket flow, requester side: a request sent behind the one that got
    /// the ticket was rejected; it is already marked for replay.
    fn reject_arrive(&mut self, nu: NeutrinoId, owner: NeutrinoId) {
        self.ticket_stats.rejects[nu.0 as usize] += 1;
        let t = self.nus[nu.0 as usize].ticket.get_mut(&owner.0);
        match t {
            Some(t) if t.rejects_due > 0 => t.rejects_due -= 1,
            _ => {
                let msg = format!("Reject from {owner:?} with none due");
                self.fatal(nu, None, FatalError::Invalid(msg));
            }
        }
    }

    /// Ticket flow, requester side: `owner` reserved an entry; the oldest
    /// request marked for replay goes with the ticket, ahead of every
    /// other request. After the last replay the ticket ends.
    fn grant_arrive(&mut self, nu: NeutrinoId, owner: NeutrinoId) {
        let now = self.now;
        let n = &mut self.nus[nu.0 as usize];
        let Some(seq) = n
            .ticket
            .get_mut(&owner.0)
            .and_then(|t| t.replay.pop_front())
        else {
            let msg = format!("PCrdGrant from {owner:?} with nothing to replay");
            return self.fatal(nu, None, FatalError::Invalid(msg));
        };
        let t = n.ticket.get(&owner.0).expect("ticket");
        if t.replay.is_empty() {
            if t.rejects_due > 0 {
                let msg = format!("last replay to {owner:?} before its rejects");
                return self.fatal(nu, None, FatalError::Invalid(msg));
            }
            self.ticket_stats.wait[nu.0 as usize] += now - t.since;
            n.ticket.remove(&owner.0);
        }
        let h = n
            .held
            .get_mut(&owner.0)
            .and_then(|q| q.iter_mut().find(|h| h.seq == seq))
            .expect("held request to replay");
        h.state = HeldState::Sent;
        let kind = match h.kind.clone() {
            HeldKind::Invoke(desc, call) => MsgKind::Invoke { desc, call },
            HeldKind::VersionReset(addr) => MsgKind::VersionReset { addr },
        };
        self.stats.resends += 1;
        let who = self.msg_owner(nu, owner, &kind);
        let Some((id, _)) = self.make_msg(nu, owner, kind, false, who) else {
            return;
        };
        let m = self.msgs[id as usize].as_mut().expect("message");
        m.ticketed = true;
        m.seq = seq;
        let src = Stage::Source(nu.0, INVOKE);
        let q = self.stages.entry(src).or_default();
        q.occ += 1;
        q.peak = q.peak.max(q.occ);
        q.fifo.push_front(id);
        self.advance(src);
    }

    /// Ticket flow, requester side: holds a request to `owner` until it is
    /// acknowledged, and sends it into the invoke source queue.
    fn send_held(&mut self, nu: NeutrinoId, owner: NeutrinoId, kind: MsgKind, seq: u64) {
        let held = match &kind {
            MsgKind::Invoke { desc, call } => HeldKind::Invoke(desc.clone(), *call),
            MsgKind::VersionReset { addr } => HeldKind::VersionReset(*addr),
            _ => unreachable!("only requests are held"),
        };
        let who = self.msg_owner(nu, owner, &kind);
        let Some((id, _)) = self.make_msg(nu, owner, kind, false, who) else {
            return;
        };
        let n = &mut self.nus[nu.0 as usize];
        n.held.entry(owner.0).or_default().push_back(Held {
            seq,
            kind: held,
            state: HeldState::Queued,
        });
        n.held_len += 1;
        n.outbound_peak = n.outbound_peak.max(n.held_len);
        self.msgs[id as usize].as_mut().expect("message").seq = seq;
        let src = Stage::Source(nu.0, INVOKE);
        self.enqueue(src, id);
        self.advance(src);
    }

    /// Entries of `nu`'s outbound invoke table in use: the invoke-class
    /// source queue, or under ticket flow every request not yet
    /// acknowledged.
    fn outbound_used(&self, nu: NeutrinoId) -> usize {
        if self.flow == Flow::Ticket {
            self.nus[nu.0 as usize].held_len
        } else {
            self.stage_occ(Stage::Source(nu.0, INVOKE))
        }
    }

    /// Sends request `kind` from `nu` to `owner` in the invoke channel; it
    /// takes outbound invoke entry `seq` (the next sequence number).
    fn send_request(&mut self, nu: NeutrinoId, owner: NeutrinoId, mut kind: MsgKind) {
        let n = &mut self.nus[nu.0 as usize];
        n.next_seq += 1;
        let seq = n.next_seq;
        let mut e = TableEntry {
            peer: Some(owner),
            ..TableEntry::new(nu, Table::OutboundInvokes, seq)
        };
        match &mut kind {
            MsgKind::Invoke { desc, call } => {
                call.seq = seq;
                (e.base, e.len) = (desc.dst.base, desc.dst.len());
            }
            MsgKind::VersionReset { addr } => {
                (e.base, e.flags) = (*addr, tr::INVOKE_VRESET);
            }
            _ => unreachable!("only requests take an outbound invoke entry"),
        }
        let now = self.now;
        trace!(self, t => t.entry_open(&e, now));
        if self.flow == Flow::Ticket {
            self.send_held(nu, owner, kind, seq);
        } else {
            let who = self.msg_owner(nu, owner, &kind);
            let Some((id, class)) = self.make_msg(nu, owner, kind, false, who) else {
                return;
            };
            self.msgs[id as usize].as_mut().expect("message").seq = seq;
            let src = Stage::Source(nu.0, class);
            self.enqueue(src, id);
            self.advance(src);
            let occ = self.stage_occ(Stage::Source(nu.0, INVOKE));
            let n = &mut self.nus[nu.0 as usize];
            n.outbound_peak = n.outbound_peak.max(occ);
        }
    }

    /// True (and the program stalled on `req`) when `nu`'s outbound invoke
    /// table is full.
    fn outbound_full(&mut self, nu: NeutrinoId, req: Request) -> Option<Request> {
        if self.outbound_used(nu) < self.nus[nu.0 as usize].cfg.outbound_invokes {
            let n = &mut self.nus[nu.0 as usize];
            if let Some(t) = n.outbound_stall_since.take() {
                self.stats.outbound_stall[nu.0 as usize] += self.now - t;
            }
            return Some(req);
        }
        let now = self.now;
        self.nus[nu.0 as usize]
            .outbound_stall_since
            .get_or_insert(now);
        self.stall(nu, req, Stall::OutboundInvokes);
        None
    }

    /// Forwards an invoke whose `dst` `owner` owns (Section 3.3, remote
    /// invoke). The program continues at once; the invoke waits in the
    /// outbound invoke table while the channel is busy.
    pub(super) fn forward_invoke(&mut self, nu: NeutrinoId, owner: NeutrinoId, desc: Descriptor) {
        if desc.opcode == OpcodeId::SPAWN {
            return self.fatal(nu, None, FatalError::Unsupported("remote spawn"));
        }
        let Some(Request::Invoke(desc)) = self.outbound_full(nu, Request::Invoke(desc)) else {
            return;
        };
        let addr = desc.dst.base;
        let call = TaskCall {
            src: nu,
            call: self.nus[nu.0 as usize].op_call,
            queued: Some(self.now),
            sent: None,
            // Set by `send_request`.
            seq: 0,
            n: self.nus[nu.0 as usize].calls,
        };
        self.send_request(
            nu,
            owner,
            MsgKind::Invoke {
                desc: Box::new(desc),
                call,
            },
        );
        self.respond(
            nu,
            Response::Invoked(TaskHandle {
                task: None,
                gid: None,
                addr,
                // The owner assigns the version on arrival.
                version: 0,
            }),
        );
    }

    /// A program `vreset` of a range that `owner` owns: a `VersionReset`
    /// message in the invoke channel, behind the program's earlier invokes
    /// to that owner.
    pub(super) fn vreset_remote(&mut self, nu: NeutrinoId, owner: NeutrinoId, addr: u64) {
        if self.outbound_full(nu, Request::VReset(addr)).is_none() {
            return;
        }
        self.send_request(nu, owner, MsgKind::VersionReset { addr });
        self.respond(nu, Response::Done);
    }

    /// Resends a rejected `Await`. The deadlock detector runs here too: a
    /// run whose only events are `Await` retries makes no progress.
    pub(super) fn resend_await(&mut self, nu: NeutrinoId, owner: NeutrinoId, msg: AwaitMsg) {
        if self.deadlocked() {
            return self.deadlock(nu);
        }
        self.stats.resends += 1;
        self.send_msg(nu, owner, MsgKind::Await { msg }, true);
    }

    /// No issue, no completion and no message moved for the detector's
    /// cycles, and no PE runs.
    fn deadlocked(&self) -> bool {
        let idle = self.now.saturating_sub(self.last_progress);
        idle > self.livelock && self.pes.iter().all(|p| p.running.is_empty())
    }

    /// Stops the run as a deadlock with every table and every non-empty
    /// stage.
    pub(super) fn deadlock(&mut self, nu: NeutrinoId) {
        let idle = self.now.saturating_sub(self.last_progress);
        let tables = self.table_states();
        let stages = self.stage_states();
        self.fatal(
            nu,
            None,
            FatalError::Deadlock {
                idle,
                tables,
                stages,
            },
        );
    }

    /// Opens a remote dependency and sends its `Await`. The caller checked
    /// that the outbound dep table has room.
    pub(super) fn open_remote(
        &mut self,
        nu: NeutrinoId,
        owner: NeutrinoId,
        key: Dep,
        purpose: Purpose,
    ) {
        let msg = self.nus[nu.0 as usize]
            .reqs
            .open(owner, key, purpose)
            .expect("remote dependency capacity checked");
        let gid = match purpose {
            Purpose::TaskDep(task) => Some(self.nus[nu.0 as usize].rec(task).gid),
            Purpose::Wait(_) => None,
        };
        let now = self.now;
        if let Some(t) = self.tracer.as_mut() {
            if let Some(gid) = gid {
                t.dep_remote_open(nu, msg.request.0, gid, key);
            }
            let e = TableEntry {
                gid,
                peer: Some(owner),
                ..TableEntry::new(nu, Table::OutboundDeps, msg.request.0 as u64).key(key)
            };
            t.entry_open(&e, now);
        }
        self.send(nu, owner, MsgKind::Await { msg });
    }

    /// The tracer's view of a satisfied remote request of `nu`; `flags`
    /// for its outbound dep entry.
    fn trace_dep_done(&mut self, nu: NeutrinoId, req: u32, purpose: Purpose, flags: u8) {
        let now = self.now;
        let Some(t) = self.tracer.as_mut() else {
            return;
        };
        if let Purpose::TaskDep(_) = purpose {
            t.dep_remote_done(nu, req, now);
        }
        let e = TableEntry {
            flags,
            ..TableEntry::new(nu, Table::OutboundDeps, req as u64)
        };
        t.entry_close(&e, now);
    }

    /// The tracer's view of an `Await` at its owner: the pending entries
    /// its lookup created and, when registered, its inbound ref. The ref's
    /// task is the consumer at the requester.
    fn trace_await(&mut self, m: &AwaitMsg, ack: AwaitAck, fx: &nu_model::Effects) {
        self.trace_pending(m.to, fx, tr::PENDING_AWAIT, Some(m.from));
        if ack != AwaitAck::Registered {
            return;
        }
        let r = &self.nus[m.from.0 as usize];
        let gid = match r.reqs.purpose(m.request) {
            Some(Purpose::TaskDep(t)) => Some(r.rec(t).gid),
            _ => None,
        };
        let e = TableEntry {
            gid,
            peer: Some(m.from),
            ..TableEntry::new(m.to, Table::InboundRefs, ref_id(m.from, m.request)).key(m.key)
        };
        let now = self.now;
        trace!(self, t => t.entry_open(&e, now));
    }

    /// Ready/valid: message `id` leaves the invoke source queue `s` (a
    /// request's outbound invoke entry), which frees at `end`.
    fn close_outbound(&mut self, s: Stage, id: u32, end: Cycle) {
        let Stage::Source(nu, INVOKE) = s else {
            return;
        };
        if self.flow != Flow::ReadyValid || self.tracer.is_none() {
            return;
        }
        let seq = self.msgs[id as usize].as_ref().expect("message").seq;
        let e = TableEntry::new(NeutrinoId(nu), Table::OutboundInvokes, seq);
        trace!(self, t => t.entry_close(&e, end));
    }

    /// A program wait on a key that `owner` owns.
    pub(super) fn remote_wait(&mut self, nu: NeutrinoId, owner: NeutrinoId, key: Dep) {
        let n = &mut self.nus[nu.0 as usize];
        if n.reqs.free() == 0 {
            return self.stall(nu, Request::Wait(key), Stall::OutboundDeps);
        }
        let tag = n.next_wait_tag;
        n.next_wait_tag = n.next_wait_tag.wrapping_add(1);
        n.state = ProgState::WaitingRemote(tag);
        self.open_remote(nu, owner, key, Purpose::Wait(tag));
    }

    /// A remote dependency of `nu` is satisfied (`AwaitAck(done)` or
    /// `Resolve`); its outbound dep entry is free.
    fn dep_satisfied(&mut self, nu: NeutrinoId, purpose: Purpose) {
        match purpose {
            Purpose::TaskDep(t) => {
                if self.nus[nu.0 as usize].tables.external_resolved(t) {
                    self.resolved(nu, t);
                }
            }
            Purpose::Wait(tag) => {
                if self.nus[nu.0 as usize].state == ProgState::WaitingRemote(tag) {
                    self.respond(nu, Response::Done);
                }
            }
        }
        // An outbound dep entry freed: a stalled program may go on.
        self.unstall(nu);
    }

    /// Moves invokes from the head of `nu`'s inbound invoke table into the
    /// scoreboard while the head fits: issue headroom, a version table
    /// entry for a new versioned range, a scoreboard entry, pending entries
    /// and outbound dep entries. Their
    /// dependencies are resolved then. A head that does not fit waits, in
    /// order. A freed entry lets the channel deliver the next invoke.
    pub(super) fn drain_inbound(&mut self, nu: NeutrinoId) {
        let mut moved = false;
        loop {
            let n = &mut self.nus[nu.0 as usize];
            if n.inbound.is_empty() {
                break;
            }
            let (entry, desc) = match n.inbound.pop_front().expect("inbound entry") {
                Inbound::Invoke(entry, desc) => (entry, desc),
                Inbound::VReset { from, seq, addr } => {
                    // Frees the version entry of its base (absent: no-op);
                    // needs no other resource.
                    self.nus[nu.0 as usize].tables.vreset(addr);
                    moved = true;
                    let now = self.now;
                    trace!(self, t => t.entry_close(&TableEntry {
                        peer: Some(from),
                        ..TableEntry::new(nu, Table::Versions, addr)
                    }, now));
                    trace!(self, t => t.entry_close(&TableEntry {
                        flags: tr::INVOKE_VRESET,
                        ..TableEntry::new(nu, Table::InboundInvokes, inbound_id(from, seq))
                    }, now));
                    // A freed version entry lets a local invoke that waits
                    // for one enter.
                    self.unstall(nu);
                    continue;
                }
            };
            match self.enter_task(nu, desc, entry) {
                Ok(e) => {
                    moved = true;
                    if let (Some(t), Entry::Inbound(call)) = (self.tracer.as_mut(), entry) {
                        let id = inbound_id(call.src, call.seq);
                        let x = TableEntry {
                            gid: Some(e.gid),
                            version: e.version,
                            ..TableEntry::new(nu, Table::InboundInvokes, id)
                        };
                        t.entry_close(&x, self.now);
                    }
                    if e.ready {
                        self.on_ready(nu, e.task);
                    }
                }
                Err(EnterFail::Refused(desc)) => {
                    self.nus[nu.0 as usize]
                        .inbound
                        .push_front(Inbound::Invoke(entry, *desc));
                    break;
                }
                Err(EnterFail::Fatal(e)) => return self.fatal(nu, None, e),
                Err(EnterFail::Stall(..)) => unreachable!("a forwarded invoke never stalls"),
            }
        }
        if moved {
            self.wake_inbound(nu);
            if self.flow == Flow::Ticket {
                self.grant(nu);
            }
        }
    }

    /// A `Start` message reached the child.
    fn start_child(&mut self, parent: NeutrinoId, child: NeutrinoId, task: TaskId, start: StartFn) {
        let n = &mut self.nus[child.0 as usize];
        if n.program.is_some() || n.state != ProgState::Idle {
            let msg = format!("spawn on {child:?}, which runs a program");
            return self.fatal(parent, None, FatalError::Invalid(msg));
        }
        let ctx = NuCtx::new(
            child,
            n.mb.clone(),
            self.clock.clone(),
            self.topo.node(child).children.clone(),
            n.cfg.costs.plan,
        );
        // A program start resets the Neutrino's version table (Section 3.2).
        n.tables.vreset_all();
        n.program = Some((start.run)(ctx));
        n.prog_name = start.name;
        n.parent_task = Some((parent, task));
        n.state = ProgState::Busy;
        if let Some(a) = self.asm.as_mut() {
            a.start(child, start.name, Some(parent), self.now);
        }
        if let Some(t) = self.tracer.as_mut() {
            let gid = self.nus[parent.0 as usize].rec(task).gid;
            t.entries_clear(child, Table::Versions, self.now);
            t.program_start(child, start.name, Some(gid), self.now);
        }
        self.queue.push(self.now, Event::NuResume(child));
    }
}
