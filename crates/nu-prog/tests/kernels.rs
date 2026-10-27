//! The kernel-sequence program on configs/rtx6000-half.yaml with tiny
//! shapes: CTA counts, MACs and lane-ops, bytes per tensor and memory, the
//! first-touch rule and the barrier between kernels.

use nu_prog::kernels::Report;
use sim_core::{build_config_file, ProfileRegistry, Stats};

/// Runs `kernels` with the TOML `params`; the stats and the report.
fn run(params: &str) -> (Stats, Report) {
    run_with("rtx6000-half.yaml", params, &[])
}

/// `run` on configuration `config` with the overrides `sets`.
fn run_with(config: &str, params: &str, sets: &[&str]) -> (Stats, Report) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs")
        .join(config);
    let mut b = build_config_file(&path, sets, ProfileRegistry::with_builtins()).expect("config");
    // C stays in the Atlas accumulator until the last k-tile.
    b.sim.set_atlas_acc_buffer(1 << 17);
    let l = nu_prog::launch("kernels", &b, params, &[]).expect("launch");
    let mut sim = b.sim;
    sim.set_timeline(true);
    sim.spawn_named(l.nu, l.name, l.program);
    let s = sim.run();
    assert_eq!(s.fatal, None);
    assert!(s.unfinished.is_empty());
    let r = l.kernels.expect("kernels report").borrow().clone();
    (s, r)
}

/// The bytes of the resources whose names start with `prefix` and end
/// with `suffix`.
fn bytes(s: &Stats, prefix: &str, suffix: &str) -> u64 {
    s.resources
        .iter()
        .filter(|r| r.name.starts_with(prefix) && r.name.ends_with(suffix))
        .map(|r| r.bytes)
        .sum()
}

const GEMM: &str = r#"
tensors = [
  {name = "a", bytes = 131072},
  {name = "b", bytes = 131072},
  {name = "c", bytes = 131072},
]
[[kernels]]
name = "g"
op = "gemm"
inputs = ["a", "b"]
output = "c"
m = 256
n = 256
k = 256
bm = 128
bn = 128
bk = 64
stages = 2
epilogue_ops = 2
"#;

#[test]
fn a_gemm_reads_each_tile_range_once_from_the_gddr7() {
    // 2 x 2 tiles of 128 x 128, 4 k-tiles each. A has 2 x 4 tile ranges
    // and B 4 x 2: 16 first reads of 16 KiB, 256 KiB from the GDDR7. Each
    // CTA reads its A row band and B column band (64 KiB each), so the
    // other 256 KiB come from the L2. C (128 KiB) goes to the L2.
    let (s, r) = run(GEMM);
    let k = &r.kernels[0];
    assert_eq!(k.ctas, 4);
    assert_eq!(k.macs, 256 * 256 * 256);
    assert_eq!(k.lane_ops, 256 * 256 * 2);
    assert_eq!((k.dram_reads, k.dram_read_bytes), (16, 256 << 10));
    assert_eq!(k.l2_read_bytes, 256 << 10);
    assert_eq!((k.l2_write_bytes, k.dram_write_bytes), (128 << 10, 0));
    let t = |n: &str| r.tensors.iter().find(|t| t.name == n).unwrap();
    assert_eq!(
        (t("a").dram_read_bytes, t("a").l2_read_bytes),
        (128 << 10, 128 << 10)
    );
    assert_eq!(t("c").write_bytes, 128 << 10);
    // The model's memories moved the same bytes.
    assert_eq!(bytes(&s, "root.hbm.", ".port"), 256 << 10);
    assert_eq!(bytes(&s, "root.l2.", ".port"), 256 << 10);
    assert_eq!(bytes(&s, "root.l2.", ".wport"), 128 << 10);
    assert!(k.start < k.end && r.end == k.end);
}

#[test]
fn under_cache_home_every_access_goes_to_the_gddr7() {
    let (s, r) = run(&format!("cache = \"home\"\n{GEMM}"));
    let k = &r.kernels[0];
    assert_eq!((k.dram_read_bytes, k.l2_read_bytes), (512 << 10, 0));
    assert_eq!((k.dram_write_bytes, k.l2_write_bytes), (128 << 10, 0));
    assert_eq!(bytes(&s, "root.hbm.", ".port"), 640 << 10);
    assert_eq!(bytes(&s, "root.l2.", "port"), 0);
}

#[test]
fn serial_split_k_reads_and_adds_the_previous_slice() {
    // One 128 x 128 tile, 4 k-tiles in 2 slices: 2 CTAs. Slice 1 reads
    // slice 0's tile (32 KiB) from the L2 and adds it (2 lane-ops per
    // element); C is written twice. Slice 1's partial read starts after
    // slice 0's tile is written.
    let (s, r) = run(r#"
tensors = [
  {name = "a", bytes = 65536},
  {name = "b", bytes = 65536},
  {name = "c", bytes = 32768},
]
[[kernels]]
name = "g"
op = "gemm"
inputs = ["a", "b"]
output = "c"
m = 128
n = 128
k = 256
bk = 64
split_k = 2
"#);
    let k = &r.kernels[0];
    assert_eq!(k.ctas, 2);
    assert_eq!(k.macs, 128 * 128 * 256);
    assert_eq!(k.lane_ops, 128 * 128 * 2);
    let c = r.tensors.iter().find(|t| t.name == "c").unwrap();
    assert_eq!((c.write_bytes, c.l2_read_bytes), (64 << 10, 32 << 10));
    // The stores are DMAs on each CTA's own SM; the device runs no task.
    // Slice 1's SM has the add (a Muon task). Its partial read, add and
    // store become ready only after slice 0's store is done, and the
    // slices store the tile in slice order.
    let root = nu_model::NeutrinoId(0);
    assert!(s.tasks.iter().all(|t| t.nu != root || t.pe.is_none()));
    let mut nus: Vec<nu_model::NeutrinoId> = s
        .tasks
        .iter()
        .filter(|t| t.nu != root && t.pe.is_some())
        .map(|t| t.nu)
        .collect();
    nus.sort();
    nus.dedup();
    assert_eq!(nus.len(), 2);
    // The store: the last DMA (opcode of the first task, a load) to end.
    let dma = s
        .tasks
        .iter()
        .find(|t| t.nu != root && t.pe.is_some())
        .unwrap()
        .opcode;
    let store = |nu| {
        s.tasks
            .iter()
            .filter(|t| t.nu == nu && t.opcode == dma)
            .max_by_key(|t| t.complete.unwrap())
            .unwrap()
    };
    let adds = |nu| {
        s.tasks
            .iter()
            .filter(|t| t.nu == nu && t.opcode != dma && t.pe.is_some())
            .count()
    };
    // Slice 0's SM runs 4 matmuls; slice 1's SM 4 matmuls and the add.
    let (s0, s1) = if adds(nus[0]) < adds(nus[1]) {
        (nus[0], nus[1])
    } else {
        (nus[1], nus[0])
    };
    let first = store(s0).complete.unwrap();
    let after = s
        .tasks
        .iter()
        .filter(|t| t.nu == s1 && t.pe.is_some() && t.ready.unwrap() >= first)
        .count();
    assert_eq!(after, 3);
    assert!(store(s1).chunks[0].start >= first);
}

#[test]
fn split_k_slices_write_in_order_without_neutrino_costs() {
    // The gemm_skinny test of the card on rtx6000.yaml with no Neutrino
    // costs. When the device's copy engine wrote the tiles, slice 1's write
    // could reach the device before slice 0's and the run deadlocked. The
    // SMs now store the tiles themselves, and slice 1 depends on slice 0's
    // store record.
    let mut sets = Vec::new();
    for node in ["root", "root.gpc*", "root.gpc*.sm*"] {
        for c in ["c_invoke", "c_base", "c_dep", "c_word"] {
            sets.push(format!("{node}.neutrino.{c}=0"));
        }
    }
    let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
    let (_, r) = run_with(
        "rtx6000.yaml",
        r#"
tensors = [
  {name = "a", bytes = 2097152},
  {name = "b", bytes = 33554432},
  {name = "c", bytes = 2097152},
]
[[kernels]]
name = "g"
op = "gemm"
inputs = ["a", "b"]
output = "c"
m = 256
n = 4096
k = 4096
bk = 32
stages = 4
split_k = 2
"#,
        &sets,
    );
    let k = &r.kernels[0];
    assert_eq!(k.ctas, 64 * 2);
    assert!(k.end > k.start && r.end == k.end);
}

#[test]
fn a_layernorm_sends_no_invoke_and_no_await_to_the_device() {
    // Every task is the SM's own, the stores included: the only messages
    // are the programs' starts and returns.
    let (s, r) = run(r#"
tensors = [
  {name = "x", bytes = 262144},
  {name = "y", bytes = 262144},
]
[[kernels]]
name = "ln"
op = "layernorm"
inputs = ["x"]
output = "y"
rows = 512
cols = 256
ctas_per_sm = 4
"#);
    assert_eq!(r.kernels[0].ctas, 512);
    let kinds: Vec<&str> = s.messages_by_kind.keys().map(String::as_str).collect();
    assert!(
        kinds.iter().all(|k| ["Start", "Return"].contains(k)),
        "{kinds:?}"
    );
}

#[test]
fn rows_ops_count_their_rows() {
    // 64 rows of 256 bf16, 4 rows per CTA: 16 CTAs, 2 per SM; the add
    // reads both inputs once from the GDDR7.
    let (_, r) = run(r#"
tensors = [
  {name = "x", bytes = 32768},
  {name = "y", bytes = 32768},
  {name = "z", bytes = 32768},
]
[[kernels]]
name = "add"
op = "elementwise"
inputs = ["x", "y"]
output = "z"
rows = 64
cols = 256
rows_per_cta = 4
ops_per_elem = 3
ctas_per_sm = 2
"#);
    let k = &r.kernels[0];
    assert_eq!(k.ctas, 16);
    assert_eq!(k.lane_ops, 64 * 256 * 3);
    assert_eq!((k.dram_read_bytes, k.l2_read_bytes), (64 << 10, 0));
    assert_eq!(k.dram_reads, 32);
}

#[test]
fn attention_counts_its_kv_tiles() {
    // 2 heads over 1 KV head, 256 tokens, head_dim 64, 64 x 64 tiles,
    // causal: 4 Q tiles per head see 1, 2, 3, 4 KV tiles (10 per head).
    // Each KV tile: S (64 x 64 x 64) and P V (64 x 64 x 64). K and V are
    // read once from the GDDR7 and then from the L2.
    let (_, r) = run(r#"
tensors = [
  {name = "q", bytes = 65536},
  {name = "k", bytes = 32768},
  {name = "v", bytes = 32768},
  {name = "o", bytes = 65536},
]
[[kernels]]
name = "att"
op = "attention"
inputs = ["q", "k", "v"]
output = "o"
heads = 2
kv_heads = 1
seq = 256
head_dim = 64
bq = 64
bkv = 64
stages = 1
causal = true
softmax_ops = 12
rescale_ops = 1
norm_ops = 1
"#);
    let k = &r.kernels[0];
    assert_eq!(k.ctas, 8);
    let tiles = 2 * 10;
    assert_eq!(k.macs, tiles * 2 * 64 * 64 * 64);
    assert_eq!(k.lane_ops, tiles * (64 * 64 * 12 + 64 * 64) + 8 * 64 * 64);
    assert_eq!(k.dram_read_bytes, (64 + 32 + 32) << 10);
    assert_eq!(k.l2_read_bytes, 2 * tiles * (8 << 10) - (64 << 10));
}

const ATT: &str = r#"
tensors = [
  {name = "q", bytes = 131072},
  {name = "k", bytes = 131072},
  {name = "v", bytes = 131072},
  {name = "o", bytes = 131072},
]
[[kernels]]
name = "att"
op = "attention"
inputs = ["q", "k", "v"]
output = "o"
heads = 4
seq = 256
head_dim = 64
bq = 128
bkv = 128
stages = 1
ctas_per_sm = 2
"#;

#[test]
fn attention_holds_s_and_o_in_the_accumulator_by_default() {
    // 2 CTAs per SM: 64 KiB of SMEM each. Q, K and V take 48 KiB; with S
    // and O in SMEM (s_in = "smem") they need 112 KiB. In the accumulator
    // S takes 128 x 128 x 4 and O 128 x 64 x 2: 80 KiB per CTA, more than
    // the 64 KiB share of a 128 KiB accumulator; a 256 KiB one holds it.
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/rtx6000-half.yaml");
    let built = |acc: u64| {
        let mut b =
            build_config_file::<&str>(&path, &[], ProfileRegistry::with_builtins()).unwrap();
        b.sim.set_atlas_acc_buffer(acc);
        b
    };
    let b = built(1 << 18);
    let smem = ATT.replace("ctas_per_sm = 2", "ctas_per_sm = 2\ns_in = \"smem\"");
    let e = nu_prog::launch("kernels", &b, &smem, &[])
        .err()
        .expect("error");
    assert!(
        e.contains("S 32768 B and 2 x O 16384 B do not fit in 65536 B"),
        "{e}"
    );
    let e = nu_prog::launch("kernels", &built(1 << 17), ATT, &[])
        .err()
        .expect("error");
    assert!(
        e.contains("S and O of 81920 B per tensor unit do not fit in the CTA's 65536 B"),
        "{e}"
    );
    let l = nu_prog::launch("kernels", &b, ATT, &[]).expect("launch");
    let mut sim = b.sim;
    sim.spawn_named(l.nu, l.name, l.program);
    let s = sim.run();
    assert_eq!(s.fatal, None);
    let r = l.kernels.unwrap().borrow().clone();
    assert_eq!(r.kernels[0].ctas, 8);
    // SMEM bytes per CTA (2 KV tiles of 128): Q in; per KV tile K and V
    // in, S = Q K^T reads Q and K, P V reads P and V; then O written by
    // the last P V, read and written by the normalize, read by the pull.
    // The online softmax moves none, and S never reaches SMEM.
    let tile = 128 * 64 * 2;
    let per_kv = 2 * tile + 2 * tile + (128 * 128 * 2 + tile);
    let per_cta = tile + 2 * per_kv + tile + 2 * tile + tile;
    let smem: u64 = s
        .resources
        .iter()
        .filter(|r| r.name.contains(".smem."))
        .map(|r| r.bytes)
        .sum();
    assert_eq!(smem, 8 * per_cta as u64);
}

#[test]
fn a_kernel_starts_after_every_cta_of_the_one_before() {
    // softmax, then layernorm on its output; tasks carry their kernel's
    // index as the label's layer field.
    let (s, r) = run(r#"
tensors = [
  {name = "x", bytes = 65536},
  {name = "y", bytes = 65536, home = "l2"},
  {name = "z", bytes = 65536, home = "l2"},
]
[[kernels]]
name = "sm"
op = "softmax"
inputs = ["x"]
output = "y"
rows = 128
cols = 256
[[kernels]]
name = "ln"
op = "layernorm"
inputs = ["y"]
output = "z"
rows = 128
cols = 256
"#);
    let (k0, k1) = (&r.kernels[0], &r.kernels[1]);
    assert_eq!((k0.ctas, k1.ctas), (128, 128));
    assert!(k1.start >= k0.end);
    let of = |l: u32| {
        s.tasks
            .iter()
            .filter(move |t| t.label.map(|x| nu_prog::label::split(x).2) == Some(l))
    };
    assert!(of(0).all(|t| t.complete.unwrap() <= k0.end));
    assert!(of(1).all(|t| t.invoke >= k0.end));
    assert_eq!(of(1).count(), 128 * 3);
    // y is read from where softmax wrote it: the L2.
    assert_eq!((k1.dram_read_bytes, k1.l2_read_bytes), (0, 64 << 10));
}

#[test]
fn a_cta_that_does_not_fit_in_its_smem_is_an_error() {
    let b = build_config_file::<&str>(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/rtx6000-half.yaml"),
        &[],
        ProfileRegistry::with_builtins(),
    )
    .unwrap();
    let p = GEMM.replace("stages = 2", "stages = 2\nctas_per_sm = 4");
    let e = nu_prog::launch("kernels", &b, &p, &[])
        .err()
        .expect("error");
    assert!(e.contains("do not fit"), "{e}");
    let p = GEMM.replace("op = \"gemm\"", "op = \"conv\"");
    let e = nu_prog::launch("kernels", &b, &p, &[])
        .err()
        .expect("error");
    assert!(e.contains("unknown op"), "{e}");
}

const TWO: &str = r#"
tensors = [
  {name = "x", bytes = 65536},
  {name = "y", bytes = 65536, home = "l2"},
  {name = "z", bytes = 65536, home = "l2"},
]
[[kernels]]
name = "sm"
op = "softmax"
inputs = ["x"]
output = "y"
rows = 128
cols = 256
[[kernels]]
name = "ln"
op = "layernorm"
inputs = ["y"]
output = "z"
rows = 128
cols = 256
"#;

#[test]
fn a_cold_l2_per_kernel_reads_every_kernels_inputs_from_the_gddr7() {
    // cache = per_kernel: the L2 is empty at each kernel start, and every
    // tensor's home is the GDDR7, so layernorm reads softmax's output from
    // the GDDR7. Within a kernel the GEMM's repeated tile reads still hit
    // the L2.
    let (_, r) = run(&format!("cache = \"per_kernel\"\n{TWO}"));
    let k1 = &r.kernels[1];
    assert_eq!((k1.dram_read_bytes, k1.l2_read_bytes), (64 << 10, 0));
    let (_, r) = run(&format!("cache = \"per_kernel\"\n{GEMM}"));
    let k = &r.kernels[0];
    assert_eq!((k.dram_read_bytes, k.l2_read_bytes), (256 << 10, 256 << 10));
}

#[test]
fn posted_outputs_let_the_next_cta_start_before_the_write_ends() {
    // 1024 one-row CTAs over the SMs, one at a time per SM: the next row
    // starts when the Muon is done instead of after the write.
    let rows = TWO
        .replace("rows = 128", "rows = 1024")
        .replace("bytes = 65536", "bytes = 524288");
    let (_, posted) = run(&rows);
    let (_, waited) = run(&format!("posted_outputs = false\n{rows}"));
    assert_eq!(posted.kernels[0].ctas, waited.kernels[0].ctas);
    assert!(posted.end < waited.end, "{} vs {}", posted.end, waited.end);
}

#[test]
fn fa2_load_order_and_warps_shorten_an_attention_cta() {
    // One CTA, 8 KV tiles, single K and V buffers. K_{j+1} loads once
    // every warp's S_j is done (fa2) instead of after P V; warps overlap
    // one warp's softmax with another's matmuls. The work is the same.
    let att = |kv: &str, warps: u64| {
        run(&format!(
            r#"
tensors = [
  {{name = "q", bytes = 16384}},
  {{name = "k", bytes = 131072}},
  {{name = "v", bytes = 131072}},
  {{name = "o", bytes = 16384}},
]
[[kernels]]
name = "att"
op = "attention"
inputs = ["q", "k", "v"]
output = "o"
seq = 128
seq_kv = 1024
head_dim = 64
bq = 128
bkv = 128
stages = 1
kv_release = "{kv}"
warps = {warps}
"#
        ))
        .1
    };
    let (old, fa2, w4) = (att("after_pv", 1), att("fa2", 1), att("fa2", 4));
    assert_eq!(old.kernels[0].macs, w4.kernels[0].macs);
    assert_eq!(old.kernels[0].lane_ops, w4.kernels[0].lane_ops);
    assert!(fa2.end < old.end, "{} vs {}", fa2.end, old.end);
    assert!(w4.end < fa2.end, "{} vs {}", w4.end, fa2.end);
}

/// One layernorm row of 8192 bf16 on rtx6000-half.yaml, with the extra
/// kernel keys `extra`.
fn one_row(extra: &str) -> (Stats, Report) {
    run(&format!(
        r#"
tensors = [
  {{name = "x", bytes = 16384}},
  {{name = "y", bytes = 16384}},
]
[[kernels]]
name = "ln"
op = "layernorm"
inputs = ["x"]
output = "y"
rows = 1
cols = 8192
ops_per_elem = 8
{extra}
"#
    ))
}

/// The SM tasks of a run with opcode `name`, by issue.
fn of_opcode<'a>(s: &'a Stats, name: &str) -> Vec<&'a sim_core::TaskTimeline> {
    let op = ProfileRegistry::with_builtins().lookup(name).unwrap();
    let mut v: Vec<&sim_core::TaskTimeline> = s
        .tasks
        .iter()
        .filter(|t| t.opcode == op && t.nu != nu_model::NeutrinoId(0))
        .collect();
    v.sort_by_key(|t| t.issue);
    v
}

#[test]
fn a_row_in_pieces_starts_its_muon_before_the_row_has_arrived() {
    // piece 0 is the whole row (as before pieces): one load, one Muon
    // task after it. Pieces of 2048 elements: 4 loads, 4 Muon tasks and 4
    // stores; the first Muon task starts before the last load ends.
    let (whole, rw) = one_row("");
    let (zero, rz) = one_row("piece = 0");
    assert_eq!(whole.end_time, zero.end_time);
    assert_eq!(rw.kernels[0].lane_ops, rz.kernels[0].lane_ops);
    let (s, r) = one_row("piece = 2048");
    let (dmas, muons) = (of_opcode(&s, "dma"), of_opcode(&s, "muon.vector"));
    assert_eq!((dmas.len(), muons.len()), (8, 4), "4 loads and 4 stores");
    assert_eq!(r.kernels[0].lane_ops, rw.kernels[0].lane_ops);
    // Without pieces the Muon task waits for the whole row's load; with
    // them the first starts before that load would have ended.
    let (wl, wm) = (of_opcode(&whole, "dma"), of_opcode(&whole, "muon.vector"));
    let row_arrived = wl[0].complete.unwrap();
    assert!(wm[0].issue.unwrap() >= row_arrived);
    assert!(muons[0].issue.unwrap() < row_arrived);
    assert!(s.end_time < whole.end_time);
}

#[test]
fn simt_util_scales_a_muon_tasks_lane_ops() {
    // simt_util 0.5 doubles the charged lane-ops and an isolated Muon
    // task's time; the report keeps the work.
    let span = |s: &Stats| {
        let m = of_opcode(s, "muon.vector")[0];
        m.chunks.last().unwrap().end - m.chunks[0].start
    };
    let (full, rf) = one_row("");
    let (half, rh) = one_row("simt_util = 0.5");
    let (kf, kh) = (&rf.kernels[0], &rh.kernels[0]);
    assert_eq!(kf.lane_ops, kh.lane_ops);
    assert_eq!(kh.lane_ops_charged, 2 * kf.lane_ops_charged);
    assert_eq!(kh.simt_util, 0.5);
    assert_eq!(span(&half), 2 * span(&full));
    for bad in ["simt_util = 0.0", "simt_util = 1.5"] {
        let b = build_config_file::<&str>(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/rtx6000-half.yaml"),
            &[],
            ProfileRegistry::with_builtins(),
        )
        .unwrap();
        let p = format!(
            "tensors = [{{name = \"x\", bytes = 16384}}, {{name = \"y\", bytes = 16384}}]\n[[kernels]]\nname = \"ln\"\nop = \"layernorm\"\ninputs = [\"x\"]\noutput = \"y\"\nrows = 1\ncols = 8192\n{bad}\n"
        );
        let e = nu_prog::launch("kernels", &b, &p, &[])
            .err()
            .expect("error");
        assert!(e.contains("simt_util"), "{e}");
    }
}
