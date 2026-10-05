//! Static configuration of one simulation: the types that code or the TOML
//! loader ([`yaml`], architecture Section 8) builds.

use nu_model::{DepConfig, NeutrinoId, Range};

use crate::resource::ResourceId;
use crate::time::{Cycle, Ratio};

/// Cycle costs of the Neutrino instructions (Section 3.2). `complete` is
/// charged on the completion port and `resolve` is a delay (no resource);
/// the others are charged on the Neutrino's scalar resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NuCosts {
    /// Cost of `invoke`: the issue of an assembled descriptor.
    pub invoke: Cycle,
    /// Assembly of a descriptor before `invoke` (and `spawn_child`):
    /// `c_desc = desc_base + desc_dep x deps + desc_word x wire words`
    /// (8 B words of [`crate::Descriptor::wire_bytes`]).
    pub desc_base: Cycle,
    pub desc_dep: Cycle,
    pub desc_word: Cycle,
    /// Planning, per task that a program plans ([`crate::NuCtx::plan`]).
    pub plan: Cycle,
    /// Cost of `wait`.
    pub wait: Cycle,
    /// Cost of `version`.
    pub version: Cycle,
    /// Cost of `vreset`.
    pub vreset: Cycle,
    /// Cost of `c_complete`, charged on the completion port.
    pub complete: Cycle,
    /// Issue cost of an external load or store; the memory latency and
    /// bandwidth come from the memory block.
    pub load: Cycle,
    /// Issue cost of an external store.
    pub store: Cycle,
    /// Cost of `spawn_child` on the parent.
    pub spawn: Cycle,
    /// Local dependency resolution: cycles from a task's last dependency
    /// resolving to the task being ready to issue, and from a pending local
    /// wait's range completing to the program's wake (default 0).
    pub resolve: Cycle,
}

impl NuCosts {
    /// `c_desc` of a descriptor with `deps` dependencies and `wire_bytes`
    /// on the wire.
    pub fn desc(&self, deps: usize, wire_bytes: u64) -> Cycle {
        self.desc_base + self.desc_dep * deps as Cycle + self.desc_word * wire_bytes.div_ceil(8)
    }
}

impl Default for NuCosts {
    fn default() -> Self {
        Self {
            invoke: 8,
            // An RV64 estimate at IPC 1 for assembling a descriptor in
            // scratch.
            desc_base: 8,
            desc_dep: 3,
            desc_word: 1,
            plan: 0,
            wait: 2,
            version: 1,
            vreset: 1,
            complete: 4,
            load: 1,
            store: 1,
            spawn: 16,
            resolve: 0,
        }
    }
}

/// Configuration of one Neutrino: its tables, costs, rates and queue
/// sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeutrinoConfig {
    /// Node name; the names of its resources start with it.
    pub name: String,
    /// Capacities of the dependency tables and their granule.
    pub tables: DepConfig,
    /// Cycle costs of the Neutrino instructions.
    pub costs: NuCosts,
    /// Instruction throughput per cycle; program work and `nu.*` calls.
    pub scalar_rate: Ratio,
    /// Throughput of the completion port; `c_complete` is charged there.
    pub complete_rate: Ratio,
    /// Outbound dep table: remote dependencies open at once.
    pub outbound_deps: usize,
    /// Outbound invoke table, in messages: the first stage of the
    /// requester's invoke channel (ready/valid flow), or every request not
    /// yet acknowledged (ticket flow).
    pub outbound_invokes: usize,
    /// Inbound invoke table: forwarded invokes that arrived and wait for a
    /// scoreboard entry, in messages (the last stage of the invoke
    /// channel).
    pub inbound_invokes: usize,
    /// Cycles before a rejected Await is resent.
    pub t_retry: Cycle,
}

impl Default for NeutrinoConfig {
    fn default() -> Self {
        Self {
            name: "nu".into(),
            tables: DepConfig {
                scoreboard_capacity: 64,
                execution_capacity: 64,
                pending_capacity: 16,
                version_capacity: 256,
                inbound_ref_capacity: 16,
                issue_headroom: 0,
                granule: 512,
            },
            costs: NuCosts::default(),
            scalar_rate: Ratio::ONE,
            complete_rate: Ratio::ONE,
            outbound_deps: 16,
            outbound_invokes: 16,
            inbound_invokes: 64,
            t_retry: 50,
        }
    }
}

/// PE-specific parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeParams {
    /// A DMA engine.
    Dma {
        /// Extra resources on every transfer, in addition to the two memory
        /// ports and the fabric routes.
        links: Vec<ResourceId>,
        /// The engine's attachments to the buses it masters: a transfer
        /// takes the fewest-bridge route from the source block to one of
        /// them and from one of them to the destination block.
        masters: Vec<crate::topology::AttachId>,
        /// Latency of every chunk (Section 5.7).
        latency: Cycle,
    },
    /// A tensor unit.
    Atlas {
        /// The tensor-op resource, at the highest rate of `rates` (MACs per
        /// cycle).
        array: ResourceId,
        /// MACs per cycle by operand type (Section 8.2); a GEMM whose
        /// slower operand type has rate r charges `MACs * max / r`. Empty:
        /// every type at the array rate.
        rates: Vec<(crate::profile::Dtype, Ratio)>,
        /// The SMEM port resource.
        smem_port: ResourceId,
        /// Efficiency of the array, in (0, 1].
        utilization: Ratio,
        /// Pad m and n of every chunk to the descriptor's tile shape for
        /// tensor ops and operand bytes (K exact): a 51-token row costs a
        /// full 128-row tile. False counts the exact shape.
        quantize: bool,
        /// Keep the A row panel across the output tiles of a row (H1,
        /// Section 5.5): A is read from SMEM once per row panel instead of
        /// once per output tile. Default false ("none").
        row_panel: bool,
        /// Accumulator memory inside the Atlas, bytes (H3, Section 5.5;
        /// default 0). A task marked `keep` leaves its C there; the next
        /// accumulate task on the same dst reads it from there. C goes to
        /// SMEM at the end of the chain or on eviction.
        acc_buffer: u64,
    },
    /// A SIMT core modeled as an opaque vector PE (Section 3.4).
    Muon {
        /// The vector-op resource (lane-ops per cycle).
        vector: ResourceId,
        /// The unit's attachments to the buses it masters: a task charges
        /// the block that holds its `dst`, over the route with the fewest
        /// bridges.
        masters: Vec<crate::topology::AttachId>,
        /// Efficiency of the vector resource, in (0, 1].
        utilization: Ratio,
        /// Fixed cycles per task before its demand (launch, reduction tail).
        overhead: Cycle,
    },
}

/// Configuration of one PE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeConfig {
    /// PE name within its node.
    pub name: String,
    /// Tasks running at once on this PE.
    pub max_inflight: u32,
    /// Chunks of one task requested ahead (Section 5.3).
    pub max_outstanding: u32,
    /// Kind-specific parameters.
    pub params: PeParams,
}

/// A memory block: its address range and its port bandwidth resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemBlock {
    /// Name, `<node path>.<memory name>`.
    pub name: String,
    /// The Neutrino whose node holds the block.
    pub owner: NeutrinoId,
    /// The block's attachments to the buses it is a slave on; fabric
    /// routes to it end at one of them (none: no fabric, an empty route).
    pub attach: Vec<crate::topology::AttachId>,
    /// False in a `null`-fidelity subtree: the Numeric backend drops writes
    /// and rejects reads.
    pub numeric: bool,
    /// Address range of the block.
    pub range: Range,
    /// Port bandwidth resource (bytes per cycle).
    pub port: ResourceId,
    /// Latency of a Neutrino load or store before its port bandwidth.
    pub load_latency: Cycle,
}

/// What the timing plane does with a racy load (Section 4.4): a load of a
/// range that a task writes while it is issued and not done.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RaceCheck {
    /// Stop the run (default).
    #[default]
    Fatal,
    /// Record a warning and serve the load.
    Warn,
}

mod yaml;

pub use self::yaml::{
    build, build_file, build_str, AttachInfo, BridgeInfo, Built, Component, ConfigError, Layout,
    MemInfo, Role,
};
