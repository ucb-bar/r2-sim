# RV test images

Each image `NAME.bin` has an image file `NAME.toml` that loads it into
the IMEM of the root node, for `r2sim run --config CONFIG --program rv
--image NAME.toml`. `hello.bin`, `three-level.bin` and `probe2.bin` run on
configs/2p-smem.yaml, and `encoder.bin` runs on configs/2p.yaml. The
golden check (`tools/golden.sh`) runs hello, three-level and the encoder.
The images have ABI version 2: they were built by the r2-toolchain at its
ABI version 2 (`r2-build --config CONFIG` in a copy of the example).

`hello.bin` (400 bytes) is a device program as an RV64 program: it loads
two tiles into the SMEM of an SM, multiplies them there and copies the
product back. The image was built by the r2-toolchain
(`r2-build --config configs/2p-smem.yaml`) from this source:

```rust
#[nu_program(node = root, stack = 4096)]
fn hello(_: &Args) {
    let dram = gpc::prefix(0) | gpc::HBM;
    let smem = sm::prefix(0, 0) | sm::SMEM;
    let (a, b, c) = (smem, smem + BYTES, smem + 2 * BYTES);
    for (from, to) in [(dram, a), (dram + BYTES, b)] {
        let load = desc::dma(sm::PE_DMA, dst(to, BYTES), from, to, BYTES);
        unsafe { insn::invoke(load.addr()) };
    }
    let mm = desc::matmul(sm::PE_ATLAS, dst(c, BYTES), a, b, N, N, N, desc::DT_BF16)
        .with_deps([key(a, 1), key(b, 1)]);
    unsafe { insn::invoke(mm.addr()) };
    let out = dram + 2 * BYTES;
    let back = desc::dma(gpc::PE_DMA, dst(out, BYTES), c, out, BYTES).with_deps([key(c, 1)]);
    unsafe { insn::invoke(back.addr()) };
    insn::wait(key(out, 1));
}
```

`N` is 128 and `BYTES` is `N * N * 2`: A and B are 128 x 128 bf16
matrices. The image holds the trap word at offset 0, the start code at
offset 4, the program, and its record at offset 0x148 (entry 0x2a, name
`hello`, node type `root`). The program expects A at `root.gpc0.hbm+0x0`
and B at `root.gpc0.hbm+0x8000`, and writes C to `root.gpc0.hbm+0x10000`.

`three-level.bin` (832 bytes) is a program on each level of the 2p
hierarchy, built by the r2-toolchain from one crate. Each program is its
own image; a parent embeds the image of each child program it starts.

```rust
/// One 128 x 128 x k bf16 matmul on this SM's Atlas: A, B and C in this
/// SM's SMEM.
#[nu_program(node = sm, stack = 2048)]
fn sm_tile(args: &TileArgs) {
    let smem = insn::prefix() | sm::SMEM;
    let (a, b, c) = (smem, smem + TILE, smem + 2 * TILE);
    let mm = desc::matmul(sm::PE_ATLAS, dst(c, TILE), a, b, 128, 128, args.k, desc::DT_BF16);
    unsafe { insn::invoke(mm.addr()) };
    insn::wait(key(c, 1));
}

/// Starts `sm_tile` on each SM of this cluster and waits for all of them.
#[nu_program(node = gpc, stack = 2048)]
fn gpc_fanout(args: &FanoutArgs) {
    let sync = insn::prefix() | gpc::SYNC;
    let tile = TileArgs { k: args.k };
    // Two granules of the sync-only space per SM: the IMEM copy, then done.
    let rec = |i: u32| sync + u64::from(i) * 2 * GRANULE;
    for i in 0..sm::COUNT {
        let (copy, done) = (dst(rec(i), GRANULE), dst(rec(i) + GRANULE, GRANULE));
        insn::spawn(i as u16, &sm_tile, &tile, copy, done);
    }
    for i in 0..sm::COUNT {
        insn::wait(key(rec(i) + GRANULE, 0));
    }
}

/// Starts `gpc_fanout` on each cluster and waits for all of them.
#[nu_program(node = root, stack = 2048)]
fn device(_: &Args) {
    let sync = root::prefix() | root::SYNC;
    let fan = FanoutArgs { k: 256 };
    let rec = |i: u32| sync + u64::from(i) * 2 * GRANULE;
    for i in 0..gpc::COUNT {
        let (copy, done) = (dst(rec(i), GRANULE), dst(rec(i) + GRANULE, GRANULE));
        insn::spawn(i as u16, &gpc_fanout, &fan, copy, done);
    }
    for i in 0..gpc::COUNT {
        insn::wait(key(rec(i) + GRANULE, 0));
    }
}
```

`TILE` is 128 x 128 x 2 bytes, and `TileArgs` and `FanoutArgs` hold one
`k: u32`. `insn::spawn` invokes an IMEM_COPY of the child image (its
record is the `copy` range) and then a SPAWN that depends on the copy.
The records: `device` at offset 0x2f8 (node type `root`), and in the
embedded images `gpc_fanout` (node type `gpc`, a 544-byte image at
offset 0xd8 of the device image) and `sm_tile` (node type `sm`, a
224-byte image at offset 0xf8 of the cluster image). Each record has
entry 0x2a.

`encoder.bin` (15,336 bytes) is a vision encoder kernel (27 layers of
width 1152, 16 heads of 72, an FFN of 4304, three images of 256 patches,
and a projector to a width of 2048), built by the r2-toolchain
(`r2-build --config configs/2p.yaml`). It models time only: the
functional plane does not compute its values. It runs on configs/2p.yaml:
the Atlas operands are in the tensor memory of each SM, and on
configs/2p-smem.yaml the run stops with `Unowned`. The image holds every
program of the kernel: the device program `encoder` (node type `root`,
entry 0x2a) embeds the image of the cluster program `encoder_cluster`
(node type `gpc`, entry 0x316, 15,016 bytes), which embeds the image of
the SM program `encoder_step` (node type `sm`, entry 0x2068, 11,224
bytes). The cluster image is at offset 248 of the device image, and the
SM image at offset 3,704 of the cluster image (3,952 of the device
image). The device starts `encoder_cluster` on each cluster. A cluster
holds one half of the patches of each image and runs the steps of the
encoder in stream order: for each step it starts one `encoder_step`
program on each SM, in epoch step + 1. A GEMM step splits its output
tiles over the SMs, and an attention step splits the heads; attention
reads the keys and values of the other half of each image from the HBM
of the other cluster. A step program returns without waiting for its
tasks, and the programs of the next step name the keys of those tasks.

`probe2.bin` (392 bytes) has one task of each new descriptor form of ABI
version 2 on root.gpc0.sm0: a 2D copy, a matmul with B transposed, and a
Muon task that runs the registry kernel `epilogue` (id 2, an 80-byte
argument block) with the flag `BIAS`: out = bf16(C + bias). Its record is
at offset 0x140 (entry 0x2a, name `probe2`, node type `root`). The
argument block is the struct that the toolchain generates from the
kernel registry (crates/sim-core/kernels.yaml). `trace` and `null` write
nothing for a Muon kernel (the short form `--functional numeric` puts the
Muon on `null`), the run stops with an error when the Muon's backend is
`numeric` (`--set functional.units.muon=[numeric]`), and `python`
computes the epilogue. Load A (128 rows of 128 bf16) at the HBM of gpc0,
B ([128][72] bf16) at offset 0x8000 of the SMEM of root.gpc0.sm0, and the
bias (128 fp32) at offset 0x100000 of the HBM; C is at SMEM offset
0x10000 and out at 0x18000.

```rust
const N: u32 = 128;
const BYTES: u64 = (N * N * 2) as u64;

#[nu_program(node = root, stack = 4096)]
fn probe2(_: &Args) {
    let dram = gpc::prefix(0) | gpc::HBM;
    let smem = sm::prefix(0, 0) | sm::SMEM;
    let (a, b, c, out) = (smem, smem + BYTES, smem + 2 * BYTES, smem + 3 * BYTES);
    let bias = dram + 0x10_0000;
    // A 2D copy: 128 rows of 144 B at a pitch of 256 B
    let load = desc::dma2d(sm::PE_DMA, dst(a, BYTES), dram, a, 128, 144, 256, 144);
    unsafe { insn::invoke(load.addr()) };
    // A matmul with B transposed (k = 72)
    let mm = desc::matmul(sm::PE_ATLAS, dst(c, BYTES), a, b, N, N, 72, desc::DT_BF16)
        .transpose_b()
        .with_deps([key(a, 1)]);
    unsafe { insn::invoke(mm.addr()) };
    // The epilogue of C: out = bf16(C + bias)
    let ep = kernels::EpilogueArgs {
        c,
        out,
        bias,
        rows: N,
        cols: N,
        ld_c: N,
        ld_out: N,
        flags: kernels::epilogue::BIAS,
        ..Default::default()
    };
    let e = desc::muon(sm::PE_MUON, dst(out, BYTES), &ep, (N * N) as u64, BYTES, BYTES)
        .with_deps([key(c, 1)]);
    unsafe { insn::invoke(e.addr()) };
    insn::wait(key(out, 1));
}
```
