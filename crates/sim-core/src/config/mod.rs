//! Static configuration of one simulation: the types that code or the TOML
//! loader ([`yaml`], r2sim spec Section 10) builds.

use nu_model::{Addr, AddressSet, DepConfig, NeutrinoId, Range};

use crate::resource::{Dir, ResourceId};
use crate::time::{Cycle, Ratio};

/// Cycle costs of the Neutrino instructions (Neutrino spec Section 5.7). `complete` is
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
    /// Data width in bytes of the Neutrino as a master (loads, stores,
    /// protocol messages): the start of a transfer's narrowing chain
    /// (r2sim spec Section 5.6).
    pub beat_bytes: u64,
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
            beat_bytes: 64,
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
        /// takes the fewest-link route from the source block to one of
        /// them and from one of them to the destination block.
        masters: Vec<crate::topology::AttachId>,
        /// Latency of every chunk (r2sim spec Section 5.9).
        latency: Cycle,
        /// Chunk size of every transfer (a transaction of the fabric);
        /// `None`: the descriptor's `tile_bytes`.
        burst: Option<u64>,
        /// Data width in bytes: the engine moves at most one beat per
        /// cycle, the start of a transfer's narrowing chain (r2sim spec Section 5.6).
        beat: u64,
        /// Clock ratio to the core clock (r2sim spec Section 2.1): the engine moves
        /// `beat x clock` bytes per core cycle.
        clock: Ratio,
    },
    /// A tensor unit.
    Atlas {
        /// The tensor-op resource, at the highest rate of `rates` (MACs per
        /// cycle).
        array: ResourceId,
        /// Data width as a master (`beat_bytes`, default 64): operands and
        /// C move in beats of this size, or of the port's if smaller
        /// (r2sim spec Section 5.6).
        beat: u64,
        /// MACs per cycle by operand type (Radiance 2 spec Section 5.2); a GEMM whose
        /// slower operand type has rate r charges `MACs * max / r`. Empty:
        /// every type at the array rate.
        rates: Vec<(crate::profile::Dtype, Ratio)>,
        /// The SMEM port resource; `None` on a banked SMEM, whose operand
        /// and C bytes charge the ports of the banks by address
        /// (r2sim spec Section 6.2).
        smem_port: Option<ResourceId>,
        /// The SMEM write port (`wr_bytes`), charged by C out and the
        /// flush instead of `smem_port`.
        smem_wport: Option<ResourceId>,
        /// Efficiency of the array, in (0, 1].
        utilization: Ratio,
        /// Pad m and n of every chunk to the descriptor's tile shape for
        /// tensor ops and operand bytes (K exact): a 51-token row costs a
        /// full 128-row tile. False counts the exact shape.
        quantize: bool,
        /// Keep the A row panel across the output tiles of a row (r2sim
        /// spec Section 6.2): A is read from SMEM once per row panel instead of
        /// once per output tile. Default false ("none").
        row_panel: bool,
        /// Accumulator memory inside the Atlas, bytes (r2sim spec Section 6.2;
        /// default 0). A task marked `keep` leaves its C there; the next
        /// accumulate task on the same dst reads it from there. C goes to
        /// SMEM at the end of the chain or on eviction.
        acc_buffer: u64,
        /// The array's attachments to the buses it masters (`lbus`): the
        /// tracer draws its SMEM traffic on that route; nothing is
        /// charged there (its bytes are charged on the SMEM port).
        masters: Vec<crate::topology::AttachId>,
    },
    /// A SIMT core modeled as an opaque vector PE (r2sim spec Section 6.3).
    Muon {
        /// The vector-op resource (lane-ops per cycle).
        vector: ResourceId,
        /// Data width as a master (`beat_bytes`, default 64): its bytes
        /// move in beats of this size, or of a hop's if smaller (Section
        /// 5.4).
        beat: u64,
        /// The unit's attachments to the buses it masters: a task charges
        /// the block that holds its `dst`, over the route with the fewest
        /// links.
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
    /// A transfer (chunks with a fabric path): the cap on its chunks
    /// issued and not ended, `u32::MAX` for none (r2sim spec Section
    /// 5.4.1, issue-driven chunks). Another task: its chunks requested ahead.
    pub max_outstanding: u32,
    /// Kind-specific parameters.
    pub params: PeParams,
}

/// A memory block: its address sets and its port bandwidth resources.
/// A banked memory is one block per bank (Radiance 2 spec Section 3.1).
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
    /// The block's contiguous range.
    pub range: Range,
    /// The block's addresses: the decomposition of `range`.
    pub sets: Vec<AddressSet>,
    /// True for a bank of a banked memory (`<memory>.b<i>`).
    pub banked: bool,
    /// Port bandwidth resource (bytes per cycle); also the write port
    /// without `wport`.
    pub port: ResourceId,
    /// Write port (`wr_bytes`), charged by write demands.
    pub wport: Option<ResourceId>,
    /// Transaction size of the port(s) (`block_bytes`): a transfer is cut
    /// at its multiples (r2sim spec Section 5.4).
    pub block: Option<u64>,
    /// Latency of a Neutrino load or store before its port bandwidth.
    pub load_latency: Cycle,
}

impl MemBlock {
    /// A plain block of `range` with one port for both directions.
    pub fn plain(
        name: String,
        owner: NeutrinoId,
        range: Range,
        port: ResourceId,
        load_latency: Cycle,
    ) -> Self {
        Self {
            name,
            owner,
            attach: Vec::new(),
            numeric: true,
            range,
            sets: AddressSet::decompose(&range),
            banked: false,
            port,
            wport: None,
            block: None,
            load_latency,
        }
    }

    /// The name of the memory: the block's, or for a bank
    /// (`<memory>.b<i>`) its memory's.
    pub fn memory_name(&self) -> &str {
        match self.name.rsplit_once('.') {
            Some((m, _)) if self.banked => m,
            _ => &self.name,
        }
    }

    /// True when the block holds `addr`.
    pub fn holds(&self, addr: Addr) -> bool {
        self.sets.iter().any(|s| s.contains(addr))
    }

    /// The contiguous bytes from `addr` in the block (0 when it does not
    /// hold `addr`), across adjacent sets of a decomposed range.
    pub fn run(&self, addr: u64) -> u64 {
        let mut total = 0u64;
        let mut a = addr;
        while let Some(r) = self.sets.iter().map(|s| s.run(a)).find(|&r| r > 0) {
            total = total.saturating_add(r);
            match a.checked_add(r) {
                Some(next) => a = next,
                None => break,
            }
        }
        total
    }

    /// The most a chunk at `addr` may move in the block: its run, cut at
    /// the block's `block_bytes` grid.
    pub fn chunk_run(&self, addr: u64) -> u64 {
        let grid = self.block.map_or(u64::MAX, |b| b - addr % b);
        self.run(addr).min(grid)
    }

    /// True when the block holds every byte of `r` (a nonempty range).
    pub fn covers(&self, r: &Range) -> bool {
        !r.is_empty() && self.run(r.base) >= r.len()
    }

    /// The port that a demand in direction `dir` charges.
    pub fn port_for(&self, dir: Dir) -> ResourceId {
        match (dir, self.wport) {
            (Dir::Write, Some(w)) => w,
            _ => self.port,
        }
    }
}

/// What the timing plane does with a racy load (r2sim spec Section 3.4): a load of a
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
    build, build_file, build_str, AttachInfo, Built, Component, ConfigError, Layout, LinkInfo,
    MemInfo, Role,
};
