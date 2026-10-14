//! Synthetic throughput benchmark: groups of DMA -> matmul -> join, 4
//! chunks per DMA and matmul, spread over N Neutrinos, timing-only with the
//! Null backend.
//!
//! Usage: `cargo run --release -p sim-core --example bench -- [tasks]
//! [neutrinos] [timeline: on|off]`. Defaults: 1000000 1 on.

use std::time::Instant;

use nu_model::{Addr, Dep, DepConfig, Range};
use sim_core::{
    Args, Descriptor, Dtype, Kind, MemBlock, NeutrinoConfig, NuCosts, NuCtx, PeConfig, PeParams,
    ProfileRegistry, Ratio, Sim,
};

const SPAN: u64 = 1 << 40;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let tasks: u64 = a.get(1).map_or(1_000_000, |s| s.parse().unwrap());
    let nnu: u64 = a.get(2).map_or(1, |s| s.parse().unwrap());
    let timeline = a.get(3).is_none_or(|s| s != "off");
    let groups_per_nu = tasks / 3 / nnu;

    let mut sim = Sim::new(ProfileRegistry::with_builtins());
    sim.set_timeline(timeline);
    // Diagnostics off: they keep a writer map that grows with the number
    // of distinct buffers, and this benchmark never reuses one.
    sim.set_open_pending_report(false);
    sim.set_late_consumer_check(false);
    let dma_op = sim.profiles().lookup("dma").unwrap();
    let mm_op = sim.profiles().lookup("atlas.matmul").unwrap();
    let dram_port = sim
        .resources_mut()
        .add_flow("dram.port", Kind::Bandwidth, Ratio::int(256));
    // A device root holds DRAM; the SMs are its children without links.
    let device = sim.add_neutrino(
        NeutrinoConfig {
            name: "device".into(),
            ..NeutrinoConfig::default()
        },
        &[Range::new(0, SPAN)],
    );
    sim.add_memory(MemBlock::plain(
        "dram".into(),
        device,
        Range::new(0, SPAN),
        dram_port,
        100,
    ));
    for i in 0..nnu {
        let base = SPAN * (1 + 3 * i);
        let r = sim.resources_mut();
        let link = r.add_flow(&format!("link{i}"), Kind::Bandwidth, Ratio::int(64));
        let smem_port = r.add_flow(&format!("smem{i}"), Kind::Bandwidth, Ratio::int(128));
        let array = r.add_flow(&format!("array{i}"), Kind::TensorOps, Ratio::int(4096));
        let nu = sim.add_child_neutrino(
            device,
            NeutrinoConfig {
                name: format!("sm{i}"),
                tables: DepConfig {
                    scoreboard_capacity: 256,
                    execution_capacity: 256,
                    pending_capacity: 64,
                    version_capacity: 64,
                    inbound_ref_capacity: 16,
                    issue_headroom: 0,
                    granule: 512,
                },
                costs: NuCosts::default(),
                scalar_rate: Ratio::ONE,
                complete_rate: Ratio::ONE,
                ..NeutrinoConfig::default()
            },
            &[Range::new(base, base + 3 * SPAN)],
        );
        let (smem, sync) = (base, base + SPAN);
        sim.add_memory(MemBlock::plain(
            format!("smem{i}"),
            nu,
            Range::new(smem, smem + SPAN),
            smem_port,
            10,
        ));
        sim.add_sync_only(Range::new(sync, sync + SPAN));
        let dma = sim.add_pe(
            nu,
            PeConfig {
                name: format!("dma{i}"),
                max_inflight: 4,
                max_outstanding: 2,
                params: PeParams::Dma {
                    links: vec![link],
                    latency: 50,
                    masters: Vec::new(),
                    burst: None,
                    // Wide enough not to narrow the paths.
                    beat: 1024,
                    clock: sim_core::Ratio::ONE,
                },
            },
        );
        let atlas = sim.add_pe(
            nu,
            PeConfig {
                name: format!("atlas{i}"),
                max_inflight: 1,
                max_outstanding: 1,
                params: PeParams::Atlas {
                    // Wide enough not to narrow the ports.
                    beat: 1024,
                    masters: Vec::new(),
                    rates: Vec::new(),
                    row_panel: false,
                    acc_buffer: 0,
                    array,
                    smem_port: Some(smem_port),
                    smem_wport: None,
                    utilization: Ratio::ONE,
                    quantize: true,
                },
            },
        );
        sim.spawn(nu, move |ctx: NuCtx| async move {
            // A ring of SLOTS buffers: the DMA into a slot waits for the
            // previous join of that slot (its readers are done), and the
            // join then supersedes it.
            const SLOTS: u64 = 8;
            let mut last = None;
            for g in 0..groups_per_nu {
                let slot = g % SLOTS;
                let a = Range::from_len(smem + slot * 65536, 8192);
                let c = Range::from_len(smem + slot * 65536 + 16384, 32768);
                let j = Range::from_len(sync + slot * 512, 512);
                let war = if g >= SLOTS {
                    vec![Dep::Unversioned(j.base)]
                } else {
                    vec![]
                };
                let ha = ctx
                    .invoke(Descriptor {
                        opcode: dma_op,
                        pe: Some(dma),
                        dst: a,
                        versioned: true,
                        inferred_version: None,
                        deps: war,
                        label: None,
                        args: Args::Dma {
                            src: Addr((g * 8192) % (SPAN / 2)),
                            dst: Addr(a.base),
                            bytes: 8192,
                            tile_bytes: 2048,
                        },
                    })
                    .await;
                let hc = ctx
                    .invoke(Descriptor {
                        opcode: mm_op,
                        pe: Some(atlas),
                        dst: c,
                        versioned: true,
                        inferred_version: None,
                        deps: vec![ha.key()],
                        label: None,
                        args: Args::Matmul {
                            a: Addr(a.base),
                            b: Addr(a.base),
                            a2: None,
                            b2: None,
                            lda: 64,
                            ldb: 128,
                            ldc: 128,
                            m: 128,
                            n: 128,
                            k: 64,
                            in_dtype: Dtype::Bf16,
                            b_dtype: Dtype::Bf16,
                            out_dtype: Dtype::Bf16,
                            accumulate: false,
                            keep: false,
                            tile: [64, 64],
                        },
                    })
                    .await;
                ctx.invoke(Descriptor::join(j, vec![hc.key()])).await;
                last = Some(j);
            }
            if let Some(j) = last {
                ctx.wait_unversioned(j.base).await;
            }
        });
    }

    let t0 = Instant::now();
    let s = sim.run();
    let wall = t0.elapsed().as_secs_f64();
    assert!(s.fatal.is_none(), "{:?}", s.fatal);
    assert!(s.unfinished.is_empty(), "{:?}", s.unfinished);
    let hwm = std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|l| l.starts_with("VmHWM"))
        .unwrap()
        .to_string();
    println!(
        "neutrinos {nnu} timeline {timeline} tasks {} events {} wall {:.3} s  {:.2} M events/s  {:.2} M tasks/s  sim end {} cycles  {hwm}",
        s.invoked,
        s.events,
        wall,
        s.events as f64 / wall / 1e6,
        s.invoked as f64 / wall / 1e6,
        s.end_time
    );
}
