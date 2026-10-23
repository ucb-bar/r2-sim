//! RV programs on the Neutrino model (2026-10-23): the scalar core of a
//! Neutrino runs an RV64 image that the toolchain built, and its `nu.*`
//! instructions become the requests of a [`sim_core::NuCtx`].
//!
//! * [`Abi`]: the numbers of the ABI for one configuration
//!   ([`sim_core::abi`]), per node.
//! * [`decode()`]: a descriptor of ABI version 1, as the program built it
//!   in its data scratch, to a [`sim_core::Descriptor`].
//! * [`RvProgram`]: the interpreter, a Neutrino program.
//! * [`launch`]: `--program rv`, the root program found by its record in
//!   the root's IMEM.

mod decode;
mod program;

pub use decode::{
    decode, Decoded, ARGS_BYTES, OP_DMA, OP_IMEM_COPY, OP_JOIN, OP_MATMUL, OP_SPAWN, OP_VECTOR,
    PE_IMEM,
};
pub use program::{find_record, launch, records, Record, RvProgram, DEFAULT_LIMIT};

use nu_model::{NeutrinoId, Range};
use sim_core::{abi, Built, PeId};

/// The ABI of one node.
#[derive(Debug, Clone)]
pub struct NodeAbi {
    /// The dotted path ("root.gpc0.sm1").
    pub path: String,
    /// Its own space; its base is the global prefix.
    pub own: Range,
    /// Its PEs in index order.
    pub pes: Vec<PeId>,
    /// Its data scratch, or why it has none.
    pub scratch: Result<Range, String>,
    /// Bytes of its IMEM.
    pub imem_bytes: u64,
    /// The cycles of its instruction classes.
    pub scalar: sim_core::ScalarCosts,
    /// Its scalar core: IMEM, data scratch, prefix.
    pub core: sim_core::Core,
}

/// The ABI numbers of a configuration ([`sim_core::abi`]), per node.
#[derive(Debug, Clone)]
pub struct Abi {
    /// The granule, as a shift.
    pub granule_shift: u32,
    /// The bits of the granule address in a key.
    pub key_bits: u32,
    /// Indexed by [`NeutrinoId`].
    pub nodes: Vec<NodeAbi>,
}

impl Abi {
    /// The numbers of the configuration `b`.
    pub fn new(b: &Built) -> Self {
        let nodes = (0..b.sim.topology().len())
            .map(|i| {
                let nu = NeutrinoId(i as u32);
                NodeAbi {
                    path: b.sim.topology().path_name(nu),
                    own: b.own_space(nu),
                    pes: abi::pe_order(b, nu),
                    scratch: abi::data_scratch(b, nu),
                    imem_bytes: abi::imem_bytes(b, nu),
                    scalar: b.sim.neutrino_config(nu).scalar,
                    core: b.sim.core(nu),
                }
            })
            .collect();
        Self {
            granule_shift: abi::granule_shift(b),
            key_bits: abi::key_addr_bits(b),
            nodes,
        }
    }

    /// The node whose own space holds `addr`.
    pub fn owner_of(&self, addr: u64) -> Option<NeutrinoId> {
        self.nodes
            .iter()
            .position(|n| n.own.base <= addr && addr < n.own.bound)
            .map(|i| NeutrinoId(i as u32))
    }

    /// The node `nu`.
    pub fn node(&self, nu: NeutrinoId) -> &NodeAbi {
        &self.nodes[nu.0 as usize]
    }

    /// The dependency of key `key`: version `key >> K` of the granule
    /// address in the low K bits; version 0 is an unversioned key.
    pub fn dep(&self, key: u64) -> nu_model::Dep {
        let (version, granule) = (
            key.checked_shr(self.key_bits).unwrap_or(0),
            key & low_bits(self.key_bits),
        );
        let addr = granule << self.granule_shift;
        match version {
            0 => nu_model::Dep::Unversioned(addr),
            version => nu_model::Dep::Versioned { addr, version },
        }
    }
}

/// The low `n` bits set.
fn low_bits(n: u32) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1 << n) - 1
    }
}
