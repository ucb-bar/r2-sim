use func::{bf16_to_f32, f32_to_bf16, Numeric};
use nu_model::{Addr, Range};
use sim_core::{FunctionalBackend, MemBlock, ResourceId};

#[test]
fn bf16_rounds_to_nearest_even() {
    assert_eq!(f32_to_bf16(1.0), 0x3f80);
    assert_eq!(bf16_to_f32(0x3f80), 1.0);
    // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even (1.0).
    assert_eq!(f32_to_bf16(1.0 + 1.0 / 256.0), 0x3f80);
    // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: to even.
    assert_eq!(f32_to_bf16(1.0 + 3.0 / 256.0), 0x3f82);
    // Just above halfway rounds up.
    assert_eq!(f32_to_bf16(1.0 + 1.0 / 256.0 + 1.0 / 65536.0), 0x3f81);
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
}

fn numeric() -> Numeric {
    let mut n = Numeric::new();
    n.configure(
        &[MemBlock::plain(
            "m".into(),
            nu_model::NeutrinoId(0),
            Range::new(0x10_0000, 0x30_0000),
            ResourceId(0),
            0,
        )],
        &[Range::new(0x20_0000, 0x21_0000)],
    );
    n
}

#[test]
fn sparse_pages_cross_boundaries_and_read_zero() {
    let mut n = numeric();
    let data: Vec<u8> = (0..200u8).collect();
    // Crosses a 64 KiB page boundary.
    n.write(Addr(0x10_ffa0), &data).unwrap();
    assert_eq!(n.read(Addr(0x10_ffa0), 200).unwrap(), data);
    assert_eq!(n.read(Addr(0x18_0000), 4).unwrap(), vec![0; 4]);
    assert!(n.read(Addr(0x2f_fff0), 32).is_err(), "outside the block");
}

#[test]
fn writes_to_sync_only_are_ignored() {
    let mut n = numeric();
    n.write(Addr(0x20_0000), &[1, 2, 3]).unwrap();
    assert_eq!(n.read(Addr(0x20_0000), 3).unwrap(), vec![0; 3]);
}
