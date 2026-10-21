//! The minimal L2 cache model (`sim::cache`, an experiment): an L2 of two
//! slices of 16 lines of 256 B caches a DRAM of two channels; channel c is
//! cached by slice c. Every address below is in channel 0, so one slice
//! of 16 lines holds them.

use nu_model::{Addr, Range};
use sim_core::{build_config, Args, Built, Descriptor, NuCtx, ProfileRegistry, Stats};

const LINE: u64 = 256;

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
    - {name: l2, type: scratch, addr: [[0x100000, 0x102000]], banks: 2, bw_bytes: 64, latency: 10,
       slave: $this.sbus, cache: {of: $this.hbm, line_bytes: 256}}
    - {name: hbm, type: dram, addr: [[0x200000, 0x400000]], banks: 2, bw_bytes: 64, latency: 50,
       slave: $this.sbus}
    - {name: buf, type: scratch, addr: [[0x0, 0x40000]], bw_bytes: 64, slave: $this.sbus}
    - {name: sync, type: sync, addr: [[0xF00000, 0x1000000]]}
  compute:
    - {name: dma, type: dma, latency: 4, master: [$this.sbus]}
  neutrino: {c_invoke: 0, c_base: 0, c_dep: 0, c_word: 0}
";
    build_config(text, ProfileRegistry::with_builtins()).expect("config")
}

/// One step of a program: a store of `lines` lines from `buf` into DRAM
/// line `first` (`store`), a read of them into `buf`, or a flush.
#[derive(Clone, Copy)]
enum Step {
    Store(u64, u64),
    Read(u64, u64),
    Flush,
}

/// Runs `steps` in order on the root's DMA engine, each after the one
/// before ends.
fn run(steps: &[Step]) -> Stats {
    let b = built();
    let root = b.nu("root");
    let (hbm, buf) = (b.memory("root.hbm").base, b.memory("root.buf").base);
    let (dma, pe) = (b.sim.profiles().lookup("dma").unwrap(), b.pe("root.dma"));
    let steps = steps.to_vec();
    let mut sim = b.sim;
    sim.spawn(root, move |ctx: NuCtx| async move {
        for (i, s) in steps.into_iter().enumerate() {
            let (src, dst, bytes) = match s {
                Step::Store(first, n) => (buf, hbm + first * LINE, n * LINE),
                Step::Read(first, n) => (hbm + first * LINE, buf, n * LINE),
                Step::Flush => {
                    ctx.cache_flush().await;
                    continue;
                }
            };
            let d = Descriptor {
                opcode: dma,
                pe: Some(pe),
                dst: Range::from_len(dst, bytes.div_ceil(512) * 512),
                versioned: false,
                inferred_version: None,
                deps: vec![],
                label: Some(i as u64),
                args: Args::Dma {
                    src: Addr(src),
                    dst: Addr(dst),
                    bytes,
                    tile_bytes: bytes,
                },
            };
            let h = ctx.invoke(d).await;
            ctx.wait_unversioned(h.addr).await;
        }
    });
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    s
}

fn bytes(s: &Stats, name: &str) -> u64 {
    s.resources
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("{name}"))
        .bytes
}

#[test]
fn a_store_stream_larger_than_the_cache_writes_back_the_excess() {
    // 40 lines into a slice of 16: the first 24 are written back when the
    // later ones evict them, from the slice to the channel. The DRAM
    // channel sees only the write-backs; the slice the 40 stores and the
    // 24 write-back reads.
    let s = run(&[Step::Store(0, 40)]);
    let c = &s.caches[0];
    assert_eq!(c.name, "root.l2");
    assert_eq!((c.slices, c.lines_per_slice), (2, 16));
    let k = c.counts;
    assert_eq!((k.write_allocs, k.write_hits, k.allocs), (40, 0, 40));
    assert_eq!((k.dirty_evictions, k.clean_evictions), (24, 0));
    assert_eq!(k.writeback_bytes, 24 * LINE);
    assert_eq!(k.peak_lines, 16);
    assert_eq!(bytes(&s, "root.hbm.b0.port"), 24 * LINE);
    assert_eq!(bytes(&s, "root.hbm.b1.port"), 0);
    assert_eq!(bytes(&s, "root.l2.b0.port"), (40 + 24) * LINE);
    assert_eq!(bytes(&s, "root.l2.b1.port"), 0);
}

#[test]
fn a_new_line_evicts_the_least_recently_used_one() {
    // Lines 0..16 fill the slice clean; reading line 0 again makes line 1
    // the least recently used. A store of line 16 evicts line 1 (clean: no
    // write-back), so line 0 still hits and line 1 misses (and evicts line
    // 2, now the least recently used).
    let s = run(&[
        Step::Read(0, 16),
        Step::Read(0, 1),
        Step::Store(16, 1),
        Step::Read(0, 1),
        Step::Read(1, 1),
        Step::Read(2, 1),
    ]);
    let k = s.caches[0].counts;
    assert_eq!((k.read_hits, k.read_misses), (2, 18));
    assert_eq!((k.clean_evictions, k.dirty_evictions), (3, 0));
    // The 16 first reads come from the channel, the two hits from the
    // slice, the misses of lines 1 and 2 from the channel again.
    assert_eq!(bytes(&s, "root.hbm.b0.port"), 18 * LINE);
    assert_eq!(bytes(&s, "root.l2.b0.port"), (2 + 1) * LINE);
}

#[test]
fn a_flush_empties_the_cache_without_traffic() {
    // 8 dirty lines, a flush (no write-back), then the same lines read:
    // every read misses.
    let s = run(&[Step::Store(0, 8), Step::Flush, Step::Read(0, 8)]);
    let c = &s.caches[0];
    let k = c.counts;
    assert_eq!((k.read_hits, k.read_misses), (0, 8));
    assert_eq!((k.dirty_evictions, k.writeback_bytes), (0, 0));
    assert_eq!(c.flushes.len(), 1);
    assert_eq!(c.flushes[0].counts.write_allocs, 8);
    assert_eq!(c.flushes[0].counts.peak_lines, 8);
    assert_eq!(bytes(&s, "root.hbm.b0.port"), 8 * LINE);
}
