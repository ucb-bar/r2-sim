//! Neutrino dependency model.
//!
//! Pure state machines with no notion of time. `sim-core` drives them from
//! the timing plane and turns their returned effects into events.
//!
//! * [`addr`]: global addresses, half-open ranges, granule alignment, and the
//!   prefix-to-owner map.
//! * [`scoreboard`]: the dependency tables of one Neutrino (pending table,
//!   scoreboard, execution table, version table), RAW lookup by version or
//!   under the subset rule, retire, epochs, and the owner-side inbound ref
//!   table.
//! * [`remote`]: requester-side state of the cross-Neutrino dependency
//!   protocol and the owner-side `await` handler.

pub mod addr;
pub mod dst;
pub mod remote;
pub mod scoreboard;
mod slab;

pub use addr::{Addr, AddressHash, AddressSet, NeutrinoId, OwnerMap, Range};
pub use dst::{Dst, FULL_MASK};
pub use remote::{
    handle_await, AckAction, AwaitAck, AwaitMsg, Awaiter, OutboundDeps, OutboundDepsFull, Purpose,
    RequestId,
};
pub use scoreboard::{
    epoch_of, epoch_order, version_order, Dep, DepConfig, DepError, DepTables, Effects, Invoked,
    OpenPending, PendingId, Phase, Registration, Tag, TaskId, TaskSpec, WaitId, WaitStatus,
    Warning, COUNT_MASK, EPOCH_MAX, EPOCH_SHIFT,
};
