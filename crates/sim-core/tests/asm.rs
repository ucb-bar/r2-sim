//! The assembly-style listing (`sim_core::asm`): instructions of
//! hand-built descriptors, and a recorded program's listing.

use nu_model::{Addr, Dep, Range};
use sim_core::asm::Symbols;
use sim_core::{build_config, Args, Built, Descriptor, Dtype, NuCtx, ProfileRegistry};

fn built() -> Built {
    let text = "global:
  neutrino_granule: 512
  address: {fields: [{name: sm, bits: 2}], offset_bits: 24}
device:
  name: root
  fabric:
    - {name: sbus}
    - {name: cbus, control: true}
  memory:
    - {name: hbm, type: dram, addr: [[0x100000, 0x110000]], bw_bytes: 64, slave: $this.sbus}
    - {name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}
  neutrino: {c_base: 0, c_dep: 0, c_word: 0}
  children:
    - name: sm0
      fabric:
        - {name: smsbus}
        - {name: smcbus, control: true}
        - {name: lbus}
        - {name: smsbus_to_sbus, topology: link, from: $this.smsbus, to: $parent.sbus}
        - {name: smcbus_to_cbus, topology: link, from: $this.smcbus, to: $parent.cbus, control: true}
      memory:
        - {name: smem, type: scratch, addr: [[0x0, 0x10000]], bw_bytes: 128, slave: [$this.lbus, $this.smsbus]}
        - {name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}
      compute:
        - {name: atlas, type: tensor, rate: {bf16: 512}, master: [$this.lbus]}
        - {name: dma, type: dma, master: [$this.lbus, $this.smsbus]}
      neutrino: {c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}
";
    build_config(text, ProfileRegistry::with_builtins()).expect("config")
}

#[test]
fn a_descriptor_prints_as_one_instruction() {
    let b = built();
    let sym = Symbols::from_built(&b);
    let smem = b.memory("root.sm0.smem").base;
    let hbm = b.memory("root.hbm").base;
    let dma = Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe("root.sm0.dma")),
        dst: Range::from_len(smem + 0x400, 0x800),
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: Some(3),
        deps: vec![
            Dep::Versioned {
                addr: smem,
                version: 2,
            },
            Dep::Unversioned(hbm + 0x40),
        ],
        label: Some(7),
        args: Args::Dma {
            src: Addr(hbm + 0x1000),
            dst: Addr(smem + 0x400),
            bytes: 2048,
            tile_bytes: 512,
        },
    };
    assert_eq!(
        sym.task(&dma),
        "dma dst=root.sm0.smem+0x400:0x800@v3 pe=root.sm0.dma \
         deps=[root.sm0.smem+0x0@v2,root.hbm+0x40] from=root.hbm+0x1000 \
         to=root.sm0.smem+0x400 bytes=2048 tile=512 label=7"
    );
    let mm = Descriptor {
        opcode: b.sim.profiles().lookup("atlas.matmul").unwrap(),
        pe: Some(b.pe("root.sm0.atlas")),
        dst: Range::from_len(smem + 0x8000, 0x2000),
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Matmul {
            a: Addr(smem),
            b: Addr(smem + 0x1000),
            a2: None,
            b2: None,
            lda: 32,
            ldb: 64,
            ldc: 64,
            m: 64,
            n: 64,
            k: 32,
            in_dtype: Dtype::Bf16,
            b_dtype: Dtype::Fp8,
            out_dtype: Dtype::F32,
            accumulate: true,
            keep: false,
            tile: [64, 64],
        },
    };
    assert_eq!(
        sym.task(&mm),
        "mm dst=root.sm0.smem+0x8000:0x2000@v? pe=root.sm0.atlas deps=[] \
         a=root.sm0.smem+0x0 b=root.sm0.smem+0x1000 lda=32 ldb=64 ldc=64 m=64 n=64 \
         k=32 types=bf16,fp8,f32 acc=1 keep=0 tile=64x64"
    );
    // An address in a sync-only range, and one in no symbol.
    assert_eq!(
        sym.addr(b.sync_only(b.nu("root.sm0")).unwrap().base + 0x200),
        "root.sm0.sync+0x200"
    );
    assert_eq!(sym.addr(0x7_0000_0000), "0x700000000");
}

#[test]
fn a_recorded_program_lists_its_requests_in_order() {
    let b = built();
    let sym = Symbols::from_built(&b);
    let sm = b.nu("root.sm0");
    let smem = b.memory("root.sm0.smem").base;
    let d = Descriptor::join(Range::from_len(b.sync_only(sm).unwrap().base, 512), vec![]);
    let mut sim = b.sim;
    sim.set_asm(true);
    sim.spawn_named(sm, "demo", move |ctx: NuCtx| async move {
        ctx.mark("go").await;
        let h = ctx.invoke(d).await;
        ctx.wait_unversioned(h.addr).await;
        ctx.cycles(5).await;
        ctx.store(Addr(smem), vec![0; 8]).await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert_eq!(s.asm.len(), 1);
    let text = sym.listing(&s.asm[0]);
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.split(';').next().unwrap().trim())
        .collect();
    assert_eq!(
        lines,
        [
            ".program root.sm0 demo",
            "mark \"go\"",
            "join dst=root.sm0.sync+0x0:0x200 deps=[]",
            "wait root.sm0.sync+0x0",
            "cycles 5",
            "store root.sm0.smem+0x0 len=8",
        ]
    );
    assert!(text.lines().nth(1).unwrap().ends_with("; t=0"));
}

#[test]
fn an_untracked_task_and_the_fence_print_their_syntax() {
    // `dma untracked dst=...` (the flag right after the mnemonic) and the
    // fence `wait 0`; the DMA writes the root's HBM from sm0's engine.
    let b = built();
    let sym = Symbols::from_built(&b);
    let sm = b.nu("root.sm0");
    let (smem, hbm) = (b.memory("root.sm0.smem").base, b.memory("root.hbm").base);
    let d = Descriptor {
        opcode: b.sim.profiles().lookup("dma").unwrap(),
        pe: Some(b.pe("root.sm0.dma")),
        dst: Range::from_len(hbm + 0x800, 0x200),
        versioned: false,
        untracked: true,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        deps: vec![],
        label: None,
        args: Args::Dma {
            src: Addr(smem),
            dst: Addr(hbm + 0x800),
            bytes: 512,
            tile_bytes: 512,
        },
    };
    let want = "dma untracked dst=root.hbm+0x800:0x200 pe=root.sm0.dma deps=[] \
                from=root.sm0.smem+0x0 to=root.hbm+0x800 bytes=512 tile=512";
    assert_eq!(sym.task(&d), want);
    let mut sim = b.sim;
    sim.set_asm(true);
    sim.spawn_named(sm, "store", move |ctx: NuCtx| async move {
        ctx.invoke_untracked(d).await;
        ctx.fence().await;
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let text = sym.listing(&s.asm[0]);
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.split(';').next().unwrap().trim())
        .collect();
    assert_eq!(lines, [".program root.sm0 store", want, "wait 0"]);
}

#[test]
fn a_masked_range_prints_its_mask() {
    // `/0x<mask>` after the range when the mask is not all ones; nothing
    // for a plain range.
    let b = built();
    let sym = Symbols::from_built(&b);
    let smem = b.memory("root.sm0.smem").base;
    let mut d = Descriptor::join(Range::from_len(smem + 0x400, 0x800), vec![]);
    assert_eq!(sym.task(&d), "join dst=root.sm0.smem+0x400:0x800 deps=[]");
    d = d.with_mask(0x2).versioned();
    d.inferred_version = Some(1);
    assert_eq!(
        sym.task(&d),
        "join dst=root.sm0.smem+0x400:0x800/0x2@v1 deps=[]"
    );
}
