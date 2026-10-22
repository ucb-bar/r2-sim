//! A program defined outside nu-prog, run through the r2sim library.
//!
//! From the repository root: `cargo run --release --example custom`. It
//! runs on configs/2p.yaml and writes out/custom/stats.json and the tracer
//! files (`--trace-out`) in out/custom/trace/.
//!
//! The device asks SM 0 of gpc0 to process 64 KiB of its SMEM on its Muon,
//! then SM 1 pulls the result into its own SMEM with its own DMA engine (a
//! write into a node's memory is a pull by its owner; Neutrino spec Section
//! 3), and the device waits for the copy.

use nu_model::{Dep, Range};
use nu_prog::Launch;
use sim_core::{Descriptor, NuCtx};

fn main() -> Result<(), String> {
    let o = r2sim::Options {
        config: Some("configs/2p.yaml".into()),
        stats: Some("out/custom/stats.json".into()),
        trace_out: Some("out/custom/trace".into()),
        ..Default::default()
    };
    std::fs::create_dir_all("out/custom").map_err(|e| e.to_string())?;
    r2sim::run_program(&o, |b| {
        let (s0, s1) = (
            b.memory("root.gpc0.sm0.smem"),
            b.memory("root.gpc0.sm1.smem"),
        );
        let (muon, dma) = (b.pe("root.gpc0.sm0.muon"), b.pe("root.gpc0.sm1.dma"));
        let bytes = 64 << 10;
        let (x, y) = (
            Range::from_len(s0.base, bytes),
            Range::from_len(s1.base, bytes),
        );
        let program = move |ctx: NuCtx| async move {
            // 8 lane-ops per byte on SM 0's Muon, in place: version 1 of x
            // (the device's invoke is forwarded to SM 0, the owner of x).
            let v = Descriptor::vector(muon, x, 8 * bytes, bytes, bytes).with_inferred_version(1);
            ctx.invoke(v).await;
            let key = Dep::Versioned {
                addr: x.base,
                version: 1,
            };
            // SM 1 pulls x once it is done.
            let copy = Descriptor::dma(dma, x.base, y, bytes)
                .with_deps(vec![key])
                .with_inferred_version(1);
            ctx.invoke(copy).await;
            ctx.wait(y.base, 1).await;
        };
        Ok(Launch::new(nu_model::NeutrinoId(0), "custom", program))
    })
}
