//! The Muon kernel registry (kernels.yaml): the parse of the file, the
//! layout of the argument blocks, each check on a bad registry, the trace
//! label of a task, and the traffic of a task from its block (`time`).

use sim_core::kernels::{self, FieldType, Traffic};

#[test]
fn the_registry_holds_kernel_0_and_the_four_encoder_kernels() {
    let k = kernels::registry();
    let names: Vec<(u32, &str, u64)> = k.iter().map(|k| (k.id, k.name.as_str(), k.bytes)).collect();
    assert_eq!(
        names,
        [
            (0, "opaque", 24),
            (1, "row_stats", 32),
            (2, "epilogue", 80),
            (3, "softmax_stats", 40),
            (4, "softmax_apply", 40),
        ]
    );
    assert_eq!(kernels::by_id(3).unwrap().name, "softmax_stats");
    assert!(kernels::by_id(5).is_none());
    assert_eq!(kernels::name(0), Some("opaque"));
    assert_eq!(kernels::name(9), None);
}

#[test]
fn the_epilogue_block_is_80_bytes_with_eps_at_72_and_flags_at_76() {
    let e = kernels::by_id(2).unwrap();
    let at = |n: &str| {
        let f = e.field(n).unwrap();
        (f.offset, f.ty)
    };
    assert_eq!(e.bytes, 80);
    assert_eq!(at("c"), (0, FieldType::U64));
    assert_eq!(at("stats"), (40, FieldType::U64));
    assert_eq!(at("rows"), (48, FieldType::U32));
    assert_eq!(at("k"), (68, FieldType::U32));
    assert_eq!(at("eps"), (72, FieldType::F32));
    assert_eq!(at("flags"), (76, FieldType::U32));
    let flags: Vec<(&str, u32)> = e.flags.iter().map(|(n, &b)| (n.as_str(), b)).collect();
    assert_eq!(
        flags,
        [
            ("BIAS", 2),
            ("C_F32", 16),
            ("GELU_TANH", 8),
            ("NORM", 1),
            ("RES", 4)
        ]
    );
    assert_eq!(
        kernels::flags(),
        [
            ("NORM", 1),
            ("BIAS", 2),
            ("RES", 4),
            ("GELU_TANH", 8),
            ("C_F32", 16)
        ]
    );
}

/// A registry of one kernel with `fields` (and `more`, YAML lines of the
/// entry), and an empty `time` when `more` has none.
fn one(id: u32, name: &str, fields: &str, more: &str) -> String {
    let time = if more.contains("  time:") {
        ""
    } else {
        "  time: {}\n"
    };
    format!("- id: {id}\n  name: {name}\n  summary: s\n  semantics: s\n  fields: {fields}\n{more}{time}")
}

fn bad(yaml: &str, part: &str) {
    let e = kernels::parse(yaml).unwrap_err();
    assert!(e.contains(part), "{e:?} does not contain {part:?}");
}

#[test]
fn each_check_rejects_a_bad_registry() {
    let ok = "[[a, u64, x]]";
    assert!(kernels::parse(&one(1, "k", ok, "")).is_ok());
    // Id 0 is a kernel like the others.
    assert!(kernels::parse(&one(0, "k", ok, "")).is_ok());
    bad(
        &format!("{}{}", one(0, "k", ok, ""), one(0, "j", ok, "")),
        "j (id 0): the id is not unique",
    );
    bad(
        &format!("{}{}", one(1, "k", ok, ""), one(1, "j", ok, "")),
        "j (id 1): the id is not unique",
    );
    bad(
        &format!("{}{}", one(1, "k", ok, ""), one(2, "k", ok, "")),
        "k (id 2): the name is not unique",
    );
    bad(
        &one(1, "k", "[[a, u16, x]]", ""),
        "field a: unknown type u16",
    );
    bad(
        &one(1, "k", "[[a, u32, x], [b, u64, x]]", ""),
        "field b at offset 4 is not aligned to 8 bytes (implicit padding)",
    );
    bad(
        &one(1, "k", "[[a, u64, x], [b, f32, x]]", ""),
        "12 bytes, not a multiple of 8",
    );
    bad(
        &one(1, "k", ok, "  colour: red\n"),
        "unknown field `colour`",
    );
}

#[test]
fn a_flag_is_one_bit_with_one_name_across_the_kernels() {
    let f = "[[a, u32, x], [flags, u32, x]]";
    bad(
        &one(1, "k", f, "  flags: {A: 3}\n"),
        "flag A = 3 is not one bit",
    );
    bad(
        &format!(
            "{}{}",
            one(1, "k", f, "  flags: {A: 1}\n"),
            one(2, "j", f, "  flags: {A: 2}\n")
        ),
        "j (id 2): flag A = 2, another kernel has A = 1",
    );
    bad(
        &format!(
            "{}{}",
            one(1, "k", f, "  flags: {A: 1}\n"),
            one(2, "j", f, "  flags: {B: 1}\n")
        ),
        "j (id 2): flag B = 1, another kernel names bit 1 A",
    );
    // The same flag with the same bit in two kernels.
    assert!(kernels::parse(&format!(
        "{}{}",
        one(1, "k", f, "  flags: {A: 1}\n"),
        one(2, "j", f, "  flags: {A: 1}\n")
    ))
    .is_ok());
}

#[test]
fn the_label_of_a_task_is_its_kernel_and_its_flags() {
    let mut block = vec![0u8; 80];
    block[76..80].copy_from_slice(&(2u32 | 8).to_le_bytes());
    assert_eq!(kernels::label(2, &block), 2 | (10 << 32));
    // A kernel without flags, and the opaque task.
    assert_eq!(kernels::label(4, &[0xff; 40]), 4);
    assert_eq!(kernels::label(0, &[]), 0);
}

#[test]
fn each_check_of_time_rejects_a_bad_registry() {
    let f = "[[a, u64, x], [n, u32, x], [flags, u32, x], [s, f32, x], [pad, u32, x]]";
    let k = |time: &str| one(5, "k", f, &format!("  flags: {{A: 1}}\n  time: {time}\n"));
    assert!(kernels::parse(&k(
        "{ops: [n * 2 + 3, {expr: n, if: A}, {expr: 4, unless: n}], reads: [{field: a, bytes: n}], writes: [{field: dst, bytes: n * n}]}"
    ))
    .is_ok());
    bad(
        "- id: 1\n  name: k\n  summary: s\n  semantics: s\n  fields: [[a, u64, x]]\n",
        "missing field `time`",
    );
    bad(
        &k("{ops: [n * m]}"),
        "k (id 5): time: unknown field m in \"n * m\"",
    );
    bad(
        &k("{ops: [n * s]}"),
        "time: field s is f32, not an integer field in \"n * s\"",
    );
    bad(&k("{ops: [n + ]}"), "time: an empty operand in \"n +\"");
    bad(
        &k("{reads: [{field: n, bytes: 4}]}"),
        "time: field n: not a u64 field of the block and not dst",
    );
    bad(
        &k("{writes: [{field: out, bytes: 4}]}"),
        "time: field out: not a u64 field of the block and not dst",
    );
    bad(
        &k("{ops: [{expr: n, if: B}]}"),
        "time: B is neither a flag of the kernel nor a field of the block",
    );
    bad(
        &k("{reads: [{field: a, bytes: n, unless: s}]}"),
        "time: field s is f32, not an integer field",
    );
    bad(
        &k("{reads: [{field: a, bytes: n, if: A, unless: n}]}"),
        "time: both if and unless in one entry",
    );
    bad(&k("{ops: [n], extra: []}"), "unknown field `extra`");
    // A flag needs the field flags.
    bad(
        &one(
            5,
            "k",
            "[[a, u64, x]]",
            "  flags: {A: 1}\n  time: {ops: [{expr: 1, if: A}]}\n",
        ),
        "time: flag A: the block has no u32 field flags",
    );
}

/// The argument block of kernel `id` with `values` by field name (the
/// other fields 0).
fn block(id: u32, values: &[(&str, u64)]) -> Vec<u8> {
    let k = kernels::by_id(id).unwrap();
    let mut b = vec![0u8; k.bytes as usize];
    for &(name, v) in values {
        let f = k.field(name).unwrap();
        let o = f.offset as usize;
        match f.ty {
            FieldType::U64 => b[o..o + 8].copy_from_slice(&v.to_le_bytes()),
            _ => b[o..o + 4].copy_from_slice(&(v as u32).to_le_bytes()),
        }
    }
    b
}

fn traffic(id: u32, values: &[(&str, u64)], dst: u64) -> Traffic {
    kernels::by_id(id)
        .unwrap()
        .traffic(&block(id, values), dst)
        .unwrap()
}

fn t(ops: u64, reads: &[(u64, u64)], writes: &[(u64, u64)]) -> Traffic {
    Traffic {
        ops,
        latency: 0,
        reads: reads.to_vec(),
        writes: writes.to_vec(),
    }
}

const DST: u64 = 0x9_0000;

#[test]
fn the_traffic_of_the_opaque_kernel_is_its_block() {
    let v = [
        ("src", 0x1000),
        ("ops", 500),
        ("bytes_in", 256),
        ("bytes_out", 128),
    ];
    assert_eq!(traffic(0, &v, DST), t(500, &[(0x1000, 256)], &[(DST, 128)]));
    // src 0: the read goes to the destination.
    let v0 = [("ops", 500), ("bytes_in", 256), ("bytes_out", 128)];
    assert_eq!(traffic(0, &v0, DST), t(500, &[(DST, 256)], &[(DST, 128)]));
    // No bytes: no entry.
    assert_eq!(traffic(0, &[("ops", 7)], DST), t(7, &[], &[]));
    // The field latency is the task's fixed latency.
    assert_eq!(
        traffic(0, &[("ops", 7), ("latency", 300)], DST),
        Traffic {
            latency: 300,
            ..t(7, &[], &[])
        }
    );
}

#[test]
fn the_latency_of_a_task_is_the_sum_of_its_terms() {
    let f = "[[n, u32, x], [flags, u32, x]]";
    let yaml = one(
        5,
        "k",
        f,
        "  flags: {A: 1}\n  time: {latency: [n * 2 + 10, {expr: 100, if: A}, {expr: 7, unless: n}]}\n",
    );
    let k = &kernels::parse(&yaml).unwrap()[0];
    let at = |n: u32, flags: u32| {
        let b: Vec<u8> = [n.to_le_bytes(), flags.to_le_bytes()].concat();
        let t = k.traffic(&b, DST).unwrap();
        (t.ops, t.latency)
    };
    assert_eq!(at(3, 0), (0, 16));
    assert_eq!(at(3, 1), (0, 116), "the term of flag A");
    assert_eq!(at(0, 1), (0, 117), "the term unless n");
    // An unknown field in a term is an error at load.
    bad(
        &one(5, "k", f, "  time: {latency: [{expr: m, if: n}]}\n"),
        "k (id 5): time: unknown field m in \"m\"",
    );
    bad(
        &one(5, "k", f, "  time: {latency: [{expr: 1, if: B}]}\n"),
        "time: B is neither a flag of the kernel nor a field of the block",
    );
}

#[test]
fn the_traffic_of_row_stats() {
    // The encoder's chunk: 128 rows of 128 columns after 128 merged ones:
    // ops 128 x 128 x 4; reads src 128 x 128 x 2 and the statistics 128 x
    // 8; writes the statistics.
    let (src, stats) = (0x10_0000, 0x20_0000);
    let v = [
        ("src", src),
        ("stats", stats),
        ("rows", 128),
        ("cols", 128),
        ("ld", 256),
    ];
    let merged = [v.as_slice(), &[("count", 128)]].concat();
    assert_eq!(
        traffic(1, &merged, DST),
        t(65_536, &[(src, 32_768), (stats, 1_024)], &[(stats, 1_024)])
    );
    // The first chunk (count 0) does not read the statistics.
    assert_eq!(
        traffic(1, &v, DST),
        t(65_536, &[(src, 32_768)], &[(stats, 1_024)])
    );
}

#[test]
fn the_traffic_of_epilogue_by_its_flags() {
    const NORM: u64 = 1;
    const BIAS: u64 = 2;
    const RES: u64 = 4;
    const GELU_TANH: u64 = 8;
    const C_F32: u64 = 16;
    let (c, out, res, bias, u, stats) =
        (0x1_0000, 0x2_0000, 0x3_0000, 0x4_0000, 0x5_0000, 0x6_0000);
    let run = |flags: u64| {
        traffic(
            2,
            &[
                ("c", c),
                ("out", out),
                ("res", res),
                ("bias", bias),
                ("u", u),
                ("stats", stats),
                ("rows", 128),
                ("cols", 128),
                ("flags", flags),
            ],
            DST,
        )
    };
    let tile = 128 * 128;
    // The encoder's FC1: NORM, BIAS and GELU_TANH, C in bf16 and in fp32.
    let fc1 = NORM | BIAS | GELU_TANH;
    let norm_reads = [(bias, 512), (u, 512), (stats, 1_024)];
    assert_eq!(
        run(fc1),
        t(
            tile * 11,
            &[[(c, tile * 2)].as_slice(), &norm_reads].concat(),
            &[(out, tile * 2)]
        )
    );
    assert_eq!(
        run(fc1 | C_F32),
        t(
            tile * 11,
            &[[(c, tile * 4)].as_slice(), &norm_reads].concat(),
            &[(out, tile * 2)]
        )
    );
    // No flag: C only, no lane operation.
    assert_eq!(run(0), t(0, &[(c, tile * 2)], &[(out, tile * 2)]));
    // O and FC2: BIAS and RES.
    assert_eq!(
        run(BIAS | RES),
        t(
            tile * 2,
            &[(c, tile * 2), (res, tile * 2), (bias, 512)],
            &[(out, tile * 2)]
        )
    );
}

#[test]
fn the_traffic_of_the_softmax() {
    let (s0, s1, stats) = (0x1_0000, 0x1_0100, 0x3_0000);
    let shape = [("rows", 128), ("cols", 128), ("ld", 256)];
    let v = [
        [("s0", s0), ("s1", s1), ("stats", stats)].as_slice(),
        &shape,
    ]
    .concat();
    assert_eq!(
        traffic(3, &v, DST),
        t(98_304, &[(s0, 32_768), (s1, 32_768)], &[(stats, 1_024)])
    );
    let (src, dst) = (0x1_0000, 0x5_0000);
    let v = [
        [("src", src), ("dst", dst), ("stats", stats)].as_slice(),
        &shape,
    ]
    .concat();
    assert_eq!(
        traffic(4, &v, DST),
        t(49_152, &[(src, 32_768), (stats, 1_024)], &[(dst, 32_768)])
    );
    // The block's dst 0: the task's destination.
    let v = [[("src", src), ("stats", stats)].as_slice(), &shape].concat();
    assert_eq!(
        traffic(4, &v, DST),
        t(49_152, &[(src, 32_768), (stats, 1_024)], &[(DST, 32_768)])
    );
}

#[test]
fn a_block_of_the_wrong_size_and_an_overflow_are_errors() {
    let k = kernels::by_id(2).unwrap();
    assert_eq!(
        k.traffic(&[0; 72], DST).unwrap_err(),
        "kernel epilogue (id 2): an argument block of 72 bytes, not 80"
    );
    let big = block(1, &[("rows", u32::MAX as u64), ("cols", u32::MAX as u64)]);
    assert_eq!(
        kernels::by_id(1).unwrap().traffic(&big, DST).unwrap_err(),
        "kernel row_stats (id 1): time: \"rows * cols * 4\" overflows u64"
    );
}
