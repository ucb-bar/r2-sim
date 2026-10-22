//! `--program hello`: the smallest program on configs/2p.yaml. The device
//! moves A and B (128 x 128 bf16) from the DRAM of gpc0 into the SMEM of
//! root.gpc0.sm0, multiplies them there, copies C back, and waits for it.

use nu_model::{Dep, Range};
use sim_core::{Built, Descriptor, Dtype, NuCtx};

use crate::Launch;

/// The program, ready to spawn on the device.
pub fn launch(b: &Built) -> Result<Launch, String> {
    let (dram, smem) = (
        b.try_memory("root.gpc0.hbm")
            .ok_or("hello: no root.gpc0.hbm")?,
        b.try_memory("root.gpc0.sm0.smem")
            .ok_or("hello: no root.gpc0.sm0.smem")?,
    );
    let (sm_dma, atlas, gpc_dma) = (
        b.pe("root.gpc0.sm0.dma"),
        b.pe("root.gpc0.sm0.atlas"),
        b.pe("root.gpc0.dma"),
    );
    const N: u64 = 128;
    let bytes = N * N * 2;
    // Operands in DRAM and in SMEM, each 32 KiB apart.
    let at = move |r: Range, i: u64| Range::from_len(r.base + i * bytes, bytes);
    let (a, bb, c) = (at(smem, 0), at(smem, 1), at(smem, 2));
    let v1 = |r: Range| Dep::Versioned {
        addr: r.base,
        version: 1,
    };
    let program = move |ctx: NuCtx| async move {
        // Compute follows data (Neutrino spec Section 3, rules N3.1 to
        // N3.3): a task runs at the owner of its destination. The SMEM is
        // root.gpc0.sm0's, so the two loads are DMAs on that SM's engine (a
        // pull by the owner), and the device's invokes are forwarded there.
        for (src, dst) in [(at(dram, 0), a), (at(dram, 1), bb)] {
            ctx.invoke(Descriptor::dma(sm_dma, src.base, dst, bytes).with_inferred_version(1))
                .await;
        }
        // The matmul runs on that SM's Atlas once both loads are done. The
        // device names their keys with the versions it infers (version 1
        // of each range; Neutrino spec Section 7.8).
        let mm = Descriptor::matmul(atlas, a.base, bb.base, c, N, N, N, Dtype::Bf16)
            .with_deps(vec![v1(a), v1(bb)])
            .with_inferred_version(1);
        ctx.invoke(mm).await;
        // C goes back to the DRAM, which gpc0 owns: a DMA on the cluster's
        // engine that pulls C from the SM once the matmul is done.
        let out = at(dram, 2);
        let back = Descriptor::dma(gpc_dma, c.base, out, bytes)
            .with_deps(vec![v1(c)])
            .with_inferred_version(1);
        ctx.invoke(back).await;
        // The device waits for the copy (Neutrino spec Section 5.4).
        ctx.wait(out.base, 1).await;
    };
    Ok(Launch::new(nu_model::NeutrinoId(0), "hello", program))
}
