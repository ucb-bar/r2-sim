//! The Muon kernel registry (kernels.yaml): the parse of the file, the
//! layout of the argument blocks, each check on a bad registry, and the
//! trace label of a task.

use sim_core::kernels::{self, FieldType};

#[test]
fn the_registry_holds_the_four_encoder_kernels() {
    let k = kernels::registry();
    let names: Vec<(u32, &str, u64)> = k.iter().map(|k| (k.id, k.name.as_str(), k.bytes)).collect();
    assert_eq!(
        names,
        [
            (1, "row_stats", 32),
            (2, "epilogue", 80),
            (3, "softmax_stats", 40),
            (4, "softmax_apply", 40),
        ]
    );
    assert_eq!(kernels::by_id(3).unwrap().name, "softmax_stats");
    assert!(kernels::by_id(0).is_none() && kernels::by_id(5).is_none());
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
/// entry).
fn one(id: u32, name: &str, fields: &str, more: &str) -> String {
    format!("- id: {id}\n  name: {name}\n  summary: s\n  semantics: s\n  fields: {fields}\n{more}")
}

fn bad(yaml: &str, part: &str) {
    let e = kernels::parse(yaml).unwrap_err();
    assert!(e.contains(part), "{e:?} does not contain {part:?}");
}

#[test]
fn each_check_rejects_a_bad_registry() {
    let ok = "[[a, u64, x]]";
    assert!(kernels::parse(&one(1, "k", ok, "")).is_ok());
    bad(&one(0, "k", ok, ""), "k (id 0): id 0 is the opaque task");
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
