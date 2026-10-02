//! Radiance 2 simulator kernel (architecture Sections 4 to 6).
//!
//! * [`time`]: the global clock and exact rationals.
//! * [`event`]: the event queue, ordered by (time, seq).
//! * [`resource`]: flow and occupancy resources, calendars, co-charging.
//! * [`profile`]: descriptors, demand profiles, the opcode registry.
//! * [`config`]: static configuration of Neutrinos, PEs and memories.
//! * [`program`]: Neutrino programs as coroutines over [`NuCtx`].
//! * [`func`]: the functional plane interface, `Null`, and the worker protocol.
//! * [`topology`]: the node tree, links and data paths.
//! * [`sim`]: the timing plane and the run loop.
//! * [`stats`]: run outputs.

pub mod config;
pub(crate) mod event;
pub mod func;
pub mod profile;
pub mod program;
pub mod resource;
pub mod sim;
pub mod stats;
pub mod time;
pub mod topology;
mod writers;

pub use config::{
    build as build_config, build_file as build_config_file, build_str as build_config_str, Built,
    ConfigError, MemBlock, NeutrinoConfig, NuCosts, PeConfig, PeParams, RaceCheck,
};
pub use func::{FuncError, FuncOp, FunctionalBackend, NullBackend, OpKind};
pub use profile::{
    Args, Chunk, Descriptor, Dtype, NodeConfig, Occupancy, OpcodeId, PeId, Profile, ProfileFn,
    ProfileRegistry,
};
pub use program::{NeutrinoProgram, NuCtx, TaskHandle};
pub use resource::{
    CoCharge, Demand, Dir, DirCycles, Grant, Kind, LogEntry, ResourceId, ResourceStats, Resources,
};
pub use sim::{Flow, Sim};
pub use stats::{
    Blocked, ChunkRec, Fatal, FatalError, LoadRec, OpenPendingRec, SimWarning, Stall, Stats,
    TableState, TaskTimeline, WarningRec,
};
pub use time::{Cycle, Ratio};
pub use topology::{
    AttachId, Attachment, Bus, BusId, BusTopology, Hop, Lane, Path, TopoNode, Topology,
};
