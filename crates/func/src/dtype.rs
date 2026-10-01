//! bf16 conversion by hand.

/// bf16 bits to f32: the bf16 is the high half of the f32.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 to bf16 bits, rounding to nearest even. NaN stays NaN.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x40;
    }
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}
