# RV test images

Each image `NAME.bin` has an image file `NAME.toml` that loads it into
the IMEM of the root node, for `r2sim run --config configs/2p-smem.yaml
--program rv --image NAME.toml`. The golden check (`tools/golden.sh`)
runs the three images.

`hello.bin` (384 bytes) is a device program as an RV64 program: it loads
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
offset 4, the program, and its record at offset 0x138 (entry 0x2a, name
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

`encoder.bin` (5768 bytes) is a vision encoder kernel (27 layers of width
1152, 16 heads of 72, an FFN of 4304, three images of 256 patches, and a
projector to a width of 2048), built by the r2-toolchain for
configs/2p-smem.yaml. It models time only: the functional plane does not
compute its values. The image holds every program of the kernel: the
device program `encoder` (node type `root`) embeds the image of the
cluster program `encoder_cluster` (node type `gpc`), which embeds the
images of the SM programs `gemm` and `attention` (node type `sm`). The
device starts `encoder_cluster` on each cluster. A cluster holds one half
of the patches of each image and runs the steps of the encoder one after
the other on its SMs: a GEMM splits its output tiles over the SMs, and
attention splits the heads. Attention reads the keys and values of the
other half of each image from the HBM of the other cluster.
