//! Cross-Neutrino dependency protocol.
//!
//! Requester A asks owner B: "await me when the producer of key K is
//! done", where K is a version `(a, v)` or an unversioned address. B rejects
//! (inbound ref table or pending table full), answers `Done` (K is done),
//! or answers `Registered` and sends `Resolve` at completion. A retries on reject. The hardware mechanism is open; this is
//! the working model of architecture Section 3.3.
//!
//! This module holds the requester-side bookkeeping ([`OutboundDeps`])
//! and the owner-side entry point ([`handle_await`]); the inbound ref table
//! itself is part of [`DepTables`]. Message transport and latency belong to
//! `sim-core`.

use crate::addr::NeutrinoId;
use crate::scoreboard::{Dep, DepError, DepTables, Effects, Registration, TaskId};
use crate::slab::Slab;

/// Index of an open request in the requester's queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u32);

/// Identifies a requester's open request from the owner's side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Awaiter {
    /// The Neutrino that opened the request.
    pub requester: NeutrinoId,
    /// The request ID in the requester's queue.
    pub request: RequestId,
}

/// Why the requester opened the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// One external dependency of a local task.
    TaskDep(TaskId),
    /// A program-level wait; the tag is the caller's.
    Wait(u32),
}

/// Requester to owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwaitMsg {
    /// The requester.
    pub from: NeutrinoId,
    /// The owner of the key's address.
    pub to: NeutrinoId,
    /// The requester's request ID.
    pub request: RequestId,
    /// The key whose producer the requester waits for.
    pub key: Dep,
}

/// Owner to requester.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwaitAck {
    /// Inbound ref table or pending table full; retry.
    Reject,
    /// The key is not done; `Resolve` follows later.
    Registered,
    /// The key is done; the dependency is satisfied.
    Done,
}

/// What the requester must do after an `AwaitAck`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckAction {
    /// The dependency is satisfied; the request is closed.
    Satisfied(Purpose),
    /// Registered at the owner; wait for `Resolve`.
    Wait,
    /// Resend the message; `attempt` counts rejections so far.
    Retry {
        /// The message to resend.
        msg: AwaitMsg,
        /// Rejections of this request so far.
        attempt: u32,
    },
}

/// Error of [`OutboundDeps::open`]: every request slot is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutboundDepsFull;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReqState {
    Sent,
    Registered,
}

#[derive(Debug, Clone)]
struct Req {
    msg: AwaitMsg,
    purpose: Purpose,
    state: ReqState,
    rejections: u32,
}

/// Requester-side outbound dep table: the open remote dependencies.
#[derive(Debug, Clone)]
pub struct OutboundDeps {
    me: NeutrinoId,
    capacity: usize,
    reqs: Slab<Req>,
}

impl OutboundDeps {
    /// Builds an empty queue for Neutrino `me` with `capacity` request slots.
    pub fn new(me: NeutrinoId, capacity: usize) -> Self {
        Self {
            me,
            capacity,
            reqs: Slab::default(),
        }
    }

    /// Open requests.
    pub fn open_count(&self) -> usize {
        self.reqs.len()
    }

    /// Total request slots.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Request slots not in use.
    pub fn free(&self) -> usize {
        self.capacity.saturating_sub(self.reqs.len())
    }

    /// The purpose of open request `id`.
    pub fn purpose(&self, id: RequestId) -> Option<Purpose> {
        self.reqs.get(id.0 as usize).map(|r| r.purpose)
    }

    /// Opens a request and returns the message to send.
    pub fn open(
        &mut self,
        owner: NeutrinoId,
        key: Dep,
        purpose: Purpose,
    ) -> Result<AwaitMsg, OutboundDepsFull> {
        if self.reqs.len() >= self.capacity {
            return Err(OutboundDepsFull);
        }
        let placeholder = AwaitMsg {
            from: self.me,
            to: owner,
            request: RequestId(0),
            key,
        };
        let id = self.reqs.insert(Req {
            msg: placeholder,
            purpose,
            state: ReqState::Sent,
            rejections: 0,
        });
        let req = self.reqs.get_mut(id).expect("just inserted");
        req.msg.request = RequestId(id as u32);
        Ok(req.msg)
    }

    /// Applies an `AwaitAck`.
    pub fn on_ack(&mut self, id: RequestId, ack: AwaitAck) -> AckAction {
        let i = id.0 as usize;
        let req = self.reqs.get_mut(i).expect("open request");
        match ack {
            AwaitAck::Reject => {
                assert_eq!(req.state, ReqState::Sent, "reject after registration");
                req.rejections += 1;
                AckAction::Retry {
                    msg: req.msg,
                    attempt: req.rejections,
                }
            }
            AwaitAck::Registered => {
                req.state = ReqState::Registered;
                AckAction::Wait
            }
            AwaitAck::Done => {
                let purpose = req.purpose;
                self.reqs.remove(i);
                AckAction::Satisfied(purpose)
            }
        }
    }

    /// Applies a `Resolve`: the registered producer completed; the request
    /// closes.
    pub fn on_resolve(&mut self, id: RequestId) -> Purpose {
        let i = id.0 as usize;
        assert_eq!(
            self.reqs.get(i).expect("open request").state,
            ReqState::Registered,
            "Resolve before AwaitAck(registered)"
        );
        self.reqs.remove(i).expect("open request").purpose
    }
}

/// Owner-side handling of one incoming `await` message.
///
/// Looks the key up in the owner's tables (Section 3.3): the pending table
/// and the version table for a version, the containing producer for an
/// unversioned address. A full
/// inbound ref table or a full pending table gives `Reject`. A done key gives
/// `Done`. Otherwise the tables record an inbound ref and the ack is
/// `Registered`; the completion that clears it reports the requester in
/// [`Effects::resolves`], and the caller sends `Resolve`. A software error
/// is returned as `Err`.
pub fn handle_await(
    tables: &mut DepTables,
    msg: &AwaitMsg,
) -> Result<(AwaitAck, Effects), DepError> {
    let waiter = Awaiter {
        requester: msg.from,
        request: msg.request,
    };
    match tables.register_remote(msg.key, waiter) {
        Ok((Registration::Done, fx)) => Ok((AwaitAck::Done, fx)),
        Ok((Registration::Registered, fx)) => Ok((AwaitAck::Registered, fx)),
        Err(DepError::InboundRefsFull | DepError::PendingFull) => {
            Ok((AwaitAck::Reject, Effects::default()))
        }
        Err(e) => Err(e),
    }
}
