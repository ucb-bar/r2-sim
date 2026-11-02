//! The 2D DMA (ABI version 2) end to end on one node: a scratch memory of
//! 4096 B/cycle in 64 B beats (64 slots) and a copy engine of 1024 B/cycle
//! (a 1024 B beat): the engine's runs per cycle limit a chunk of many
//! short rows, and the copy moves each row at its strides.

use nu_model::{Addr, Range};
use sim_core::{build_config, Args, Descriptor, NuCtx, ProfileRegistry, Rows, Stats};

fn config(runs_per_cycle: u64) -> String {
    format!(
        r#"
global:
  neutrino_granule: 512
  address:
    fields: [{{name: x, bits: 1}}]
    offset_bits: 24
device:
  name: root
  fidelity: numeric
  fabric:
    - {{name: sbus}}
    - {{name: cbus, control: true}}
  memory:
    - {{name: mem, type: scratch, addr: [[0x100000, 0x200000]], bw_bytes: 4096, beat_bytes: 64, latency: 0, slave: $this.sbus}}
  compute:
    - {{name: dma, type: dma, latency: 0, beat_bytes: 1024, runs_per_cycle: {runs_per_cycle}, master: [$this.sbus]}}
  neutrino: {{}}
"#
    )
}

/// A copy of 32 rows of 32 B from a source pitch of `src_stride` to a
/// destination pitch of `dst_stride` (one chunk: the tile holds the 32
/// rows), on `runs_per_cycle`; the stats and the destination's bytes.
fn copy(runs_per_cycle: u64, src_stride: u64, dst_stride: u64) -> (Stats, Vec<u8>) {
    let b =
        build_config(&config(runs_per_cycle), ProfileRegistry::with_builtins()).expect("config");
    let root = b.nu("root");
    let pe = b.pe("root.dma");
    let (src, dst) = (0x10_1000u64, 0x14_0000u64);
    let rows = Rows {
        count: 32,
        bytes: 32,
        src_stride,
        dst_stride,
    };
    let mut d = Descriptor::dma(pe, src, Range::from_len(dst, 32 * dst_stride), 32 * 32);
    if let Args::Dma { rows: r, .. } = &mut d.args {
        *r = Some(rows);
    }
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.set_functional(Box::new(func::Numeric::new()));
    let data: Vec<u8> = (0..32 * src_stride).map(|i| (i % 251) as u8).collect();
    sim.init_memory(Addr(src), data);
    sim.spawn(root, move |ctx: NuCtx| async move {
        let h = ctx.invoke(d).await;
        ctx.wait(h.addr, h.version).await;
    });
    let (s, plane) = sim.run_with_backend();
    let out = plane.unwrap().read(Addr(dst), 32 * dst_stride).unwrap();
    (s, out)
}

/// The cycles of the copy's one chunk.
fn chunk_cycles(s: &Stats) -> u64 {
    let t = &s.tasks[0];
    assert_eq!(t.chunks.len(), 1);
    t.chunks[0].end - t.chunks[0].start
}

#[test]
fn the_runs_per_cycle_limit_a_chunk_of_short_rows() {
    // 32 runs of 32 B from a pitch of 64 B into a contiguous destination.
    // On the hops: the read side charges one 64 B beat per run (32 beats)
    // and the write side the 1024 B as one run (16 beats), at the engine's
    // 1024 B/cycle on 16 of the memory's 64 B slots: 2 cycles. The engine
    // starts r runs per cycle, so the chunk lasts at least 32 / r cycles:
    // 32 and 8 for r = 1 and 4; for r = 32 the hops' 2 cycles.
    let cycles: Vec<u64> = [1, 4, 32]
        .map(|r| chunk_cycles(&copy(r, 64, 32).0))
        .to_vec();
    assert_eq!(cycles, [32, 8, 2]);
}

#[test]
fn a_copy_with_both_sides_contiguous_is_one_row() {
    // Pitch 32 on both sides: one contiguous run of 1024 B, 16 beats on 16
    // slots at the engine's rate, whatever the runs per cycle.
    assert_eq!(chunk_cycles(&copy(1, 32, 32).0), 1);
}

#[test]
fn each_row_goes_from_its_source_pitch_to_its_destination_pitch() {
    let (s, out) = copy(1, 64, 48);
    assert_eq!(s.fatal, None);
    for r in 0..32u64 {
        let got = &out[(r * 48) as usize..(r * 48 + 32) as usize];
        let want: Vec<u8> = (r * 64..r * 64 + 32).map(|i| (i % 251) as u8).collect();
        assert_eq!(got, want, "row {r}");
        // The gap after each destination row stays zero.
        assert_eq!(
            &out[(r * 48 + 32) as usize..(r * 48 + 48) as usize],
            &[0; 16]
        );
    }
}
