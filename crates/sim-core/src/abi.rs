//! The numbers of the Neutrino ABI that a configuration fixes (RV
//! programs, 2026-10-23), in one place: the simulator, which runs the RV
//! programs, and the toolchain, which builds them, compute them here.
//!
//! * address bits: the bits of a global byte address, those of the
//!   highest own-space bound;
//! * key address bits: the address bits less the granule shift; a key is
//!   `version << key_addr_bits | granule address`;
//! * the PE order of a node: its PEs by increasing [`PeId`], the PE index
//!   of a descriptor;
//! * the data scratch of a node: `scratch_bytes` at offset `S` of its own
//!   space, `S` the end of its sync-only range;
//! * the IMEM: `imem_bytes`, the local window `[0, imem_bytes)` of the
//!   scalar core; offset 0 traps and the start code is at
//!   [`START_VECTOR`].

use nu_model::{NeutrinoId, Range};

use crate::{Built, PeId};

/// The ABI version of the descriptors and program records.
pub const VERSION: u32 = 1;
/// The IMEM offset where a program starts; offset 0 holds a trap.
pub const START_VECTOR: u64 = 4;
/// Bytes of a program record in an IMEM: entry u64, abi_version u32,
/// args_bytes u32, stack_bytes u32, reserved u32, name [u8; 32], node
/// [u8; 16].
pub const RECORD_BYTES: u64 = 72;
/// The CSR that holds the global prefix of the node.
pub const CSR_PREFIX: u16 = 0xfc0;
/// The read-only CSR that holds the epoch of the node: the one its
/// program started with.
pub const CSR_EPOCH: u16 = 0xfc1;

/// The granule of the configuration, as a shift.
pub fn granule_shift(b: &Built) -> u32 {
    b.sim
        .neutrino_config(NeutrinoId(0))
        .tables
        .granule
        .trailing_zeros()
}

/// The bits of a global byte address: those of the highest own-space
/// bound.
pub fn addr_bits(b: &Built) -> u32 {
    let end = (0..b.sim.topology().len())
        .map(|i| b.own_space(NeutrinoId(i as u32)).bound)
        .max()
        .unwrap_or(1);
    64 - (end.max(2) - 1).leading_zeros()
}

/// The bits of the granule address in a key; the version is above them.
pub fn key_addr_bits(b: &Built) -> u32 {
    addr_bits(b) - granule_shift(b)
}

/// The PEs of node `nu` in index order: by increasing [`PeId`].
pub fn pe_order(b: &Built, nu: NeutrinoId) -> Vec<PeId> {
    let path = b.sim.topology().path_name(nu);
    b.pe_paths()
        .into_iter()
        .filter(|(p, _)| p.rsplit_once('.').is_some_and(|(n, _)| n == path))
        .map(|(_, id)| id)
        .collect()
}

/// The offset `S` of the data scratch of `nu` in its own space: the end
/// of its sync-only range. `None` for a node without one.
pub fn scratch_offset(b: &Built, nu: NeutrinoId) -> Option<u64> {
    b.sync_only(nu).map(|s| s.bound - b.own_space(nu).base)
}

/// The global range of the data scratch of `nu`.
pub fn scratch_range(b: &Built, nu: NeutrinoId) -> Option<Range> {
    let at = b.own_space(nu).base + scratch_offset(b, nu)?;
    Some(Range::from_len(at, b.sim.neutrino_config(nu).scratch_bytes))
}

/// The data scratch of `nu`, checked: the node has a sync-only range, and
/// the scratch lies in its own space and outside every memory block.
pub fn data_scratch(b: &Built, nu: NeutrinoId) -> Result<Range, String> {
    let path = b.sim.topology().path_name(nu);
    let r = scratch_range(b, nu)
        .ok_or_else(|| format!("{path}: no sync-only range, so no data scratch"))?;
    let own = b.own_space(nu);
    if !own.covers(&r) {
        return Err(format!(
            "{path}: the data scratch {r} does not fit in the own space {own}"
        ));
    }
    let blocks = b.sim.functional_blocks();
    if let Some(m) = blocks.iter().find(|m| m.range.overlaps(&r)) {
        return Err(format!(
            "{path}: the data scratch {r} overlaps the memory {} {}",
            m.name, m.range
        ));
    }
    Ok(r)
}

/// Bytes of the IMEM of `nu`.
pub fn imem_bytes(b: &Built, nu: NeutrinoId) -> u64 {
    b.sim.neutrino_config(nu).imem_bytes
}

/// The node type of a node: the last component of its path without its
/// digits (`root`, `gpc0` -> `gpc`, `sm3` -> `sm`). A program record names
/// the type it runs on.
pub fn node_type(path: &str) -> &str {
    path.rsplit('.')
        .next()
        .unwrap_or(path)
        .trim_end_matches(|c: char| c.is_ascii_digit())
}
