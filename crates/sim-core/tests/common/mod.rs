//! A single-Neutrino node: DRAM, one link, SMEM, one DMA engine, one Atlas.
//!
//! | resource | rate |
//! |---|---|
//! | dram port | 64 B/cycle |
//! | link | 32 B/cycle |
//! | smem port | 128 B/cycle |
//! | atlas array | 4096 MAC/cycle |
//! | scalar | 1 instr/cycle |
//! | completion port | 1 /cycle |
//!
//! Costs: c_invoke 8, c_wait 2, c_version 1, c_vreset 1, c_complete 4;
//! no descriptor assembly (c_base, c_dep, c_word 0).
//!
//! DMA and matmul outputs are versioned data ranges; each test range is
//! written once unless a test says otherwise, so its consumers name
//! version 1 ([`v1`]).

#![allow(dead_code)]

use nu_model::{Addr, Dep, DepConfig, NeutrinoId, Range};
use sim_core::{
    Args, Descriptor, Dtype, Kind, MemBlock, NeutrinoConfig, NuCosts, OpcodeId, PeConfig, PeId,
    PeParams, ProfileRegistry, Ratio, ResourceId, Sim,
};

pub const DRAM: u64 = 0x0000_0000;
pub const SMEM: u64 = 0x1000_0000;
pub const SYNC: u64 = 0x2000_0000;

pub struct Node {
    pub sim: Sim,
    pub nu: NeutrinoId,
    pub dma: PeId,
    pub atlas: PeId,
    pub dma_op: OpcodeId,
    pub mm_op: OpcodeId,
    pub dram_port: ResourceId,
    pub link: ResourceId,
    pub smem_port: ResourceId,
    pub array: ResourceId,
}

pub struct Knobs {
    pub scoreboard: usize,
    pub execution: usize,
    pub pending: usize,
    pub dma_inflight: u32,
    pub dma_outstanding: u32,
    pub dma_latency: u64,
    /// The Neutrino's local dependency-resolution latency (`c_resolve`).
    pub resolve: u64,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            scoreboard: 16,
            execution: 16,
            pending: 4,
            dma_inflight: 2,
            dma_outstanding: 1,
            dma_latency: 100,
            resolve: 0,
        }
    }
}

pub fn node(k: Knobs) -> Node {
    let mut sim = Sim::new(ProfileRegistry::with_builtins());
    let r = sim.resources_mut();
    let dram_port = r.add_flow("dram.port", Kind::Bandwidth, Ratio::int(64));
    let link = r.add_flow("link", Kind::Bandwidth, Ratio::int(32));
    let smem_port = r.add_flow("smem.port", Kind::Bandwidth, Ratio::int(128));
    let array = r.add_flow("atlas.array", Kind::TensorOps, Ratio::int(4096));
    let nu = sim.add_neutrino(
        NeutrinoConfig {
            name: "sm0".into(),
            tables: DepConfig {
                scoreboard_capacity: k.scoreboard,
                execution_capacity: k.execution,
                pending_capacity: k.pending,
                version_capacity: 16,
                inbound_ref_capacity: 4,
                issue_headroom: 0,
                granule: 512,
            },
            // The hand computations exclude descriptor assembly.
            costs: NuCosts {
                resolve: k.resolve,
                desc_base: 0,
                desc_dep: 0,
                desc_word: 0,
                ..NuCosts::default()
            },
            scalar_rate: Ratio::ONE,
            complete_rate: Ratio::ONE,
            ..NeutrinoConfig::default()
        },
        &[Range::new(0, 0x4000_0000)],
    );
    sim.add_sync_only(Range::new(SYNC, SYNC + 0x100_0000));
    sim.add_memory(MemBlock::plain(
        "dram".into(),
        nu,
        Range::new(DRAM, SMEM),
        dram_port,
        100,
    ));
    sim.add_memory(MemBlock::plain(
        "smem".into(),
        nu,
        Range::new(SMEM, SMEM + 0x8_0000),
        smem_port,
        10,
    ));
    let dma = sim.add_pe(
        nu,
        PeConfig {
            name: "dma0".into(),
            max_inflight: k.dma_inflight,
            max_outstanding: k.dma_outstanding,
            params: PeParams::Dma {
                links: vec![link],
                latency: k.dma_latency,
                masters: Vec::new(),
                burst: None,
                // Wide enough not to narrow the paths.
                beat: 1024,
            },
        },
    );
    let atlas = sim.add_pe(
        nu,
        PeConfig {
            name: "atlas0".into(),
            max_inflight: 1,
            max_outstanding: 1,
            params: PeParams::Atlas {
                masters: Vec::new(),
                row_panel: false,
                acc_buffer: 0,
                array,
                rates: Vec::new(),
                smem_port: Some(smem_port),
                smem_wport: None,
                utilization: Ratio::ONE,
                quantize: true,
            },
        },
    );
    let dma_op = sim.profiles().lookup("dma").unwrap();
    let mm_op = sim.profiles().lookup("atlas.matmul").unwrap();
    Node {
        sim,
        nu,
        dma,
        atlas,
        dma_op,
        mm_op,
        dram_port,
        link,
        smem_port,
        array,
    }
}

impl Node {
    /// DMA of `bytes` from DRAM offset `src` to SMEM offset `dst`.
    pub fn dma(&self, src: u64, dst: u64, bytes: u64, tile_bytes: u64) -> Descriptor {
        Descriptor {
            opcode: self.dma_op,
            pe: Some(self.dma),
            dst: Range::from_len(SMEM + dst, bytes),
            versioned: true,
            inferred_version: None,
            deps: vec![],
            label: None,
            args: Args::Dma {
                src: Addr(DRAM + src),
                dst: Addr(SMEM + dst),
                bytes,
                tile_bytes,
            },
        }
    }

    /// Square bf16 matmul of edge `e` with output tiles of edge `tile`.
    /// A and B are at the first two deps (or C when there are none).
    pub fn matmul(&self, c: Range, deps: Vec<Dep>, e: u64, tile: u64) -> Descriptor {
        let a = deps.first().map_or(c.base, |d| d.addr());
        let b = deps.get(1).map_or(a, |d| d.addr());
        Descriptor {
            opcode: self.mm_op,
            pe: Some(self.atlas),
            dst: c,
            versioned: true,
            inferred_version: None,
            deps,
            label: None,
            args: Args::Matmul {
                a: Addr(a),
                b: Addr(b),
                a2: None,
                b2: None,
                lda: e,
                ldb: e,
                ldc: e,
                m: e,
                n: e,
                k: e,
                in_dtype: Dtype::Bf16,
                b_dtype: Dtype::Bf16,
                out_dtype: Dtype::Bf16,
                accumulate: false,
                keep: false,
                tile: [tile, tile],
            },
        }
    }
}

pub fn smem(off: u64, len: u64) -> Range {
    Range::from_len(SMEM + off, len)
}

pub fn sync(off: u64) -> Range {
    Range::from_len(SYNC + off * 512, 512)
}

/// Version `version` of the versioned range `range`.
pub fn ver(range: Range, version: u64) -> Dep {
    Dep::Versioned {
        addr: range.base,
        version,
    }
}

/// The first version of `range`.
pub fn v1(range: Range) -> Dep {
    ver(range, 1)
}
