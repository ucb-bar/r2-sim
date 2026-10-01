//! The label scheme: software identity of a task, for the tracer and for
//! Trace records. `kind:8 | step:8 | layer:8 | unit:8 | index:32`.

/// Task kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Kind {
    Router = 1,
    Qkv = 2,
    Attn = 3,
    Wo = 4,
    FfnGate = 5,
    FfnUp = 6,
    FfnDown = 7,
    Vec = 8,
    DmaW = 9,
    DmaAct = 10,
    Join = 11,
}

/// The tracer name of kind number `k` ("none" for 0, "other" when unknown).
pub fn kind_name(k: u8) -> &'static str {
    const NAMES: [&str; 12] = [
        "none", "router", "qkv", "attn", "wo", "ffn_gate", "ffn_up", "ffn_down", "vec", "dma_w",
        "dma_act", "join",
    ];
    NAMES.get(k as usize).copied().unwrap_or("other")
}

pub(crate) fn label(kind: Kind, step: u32, layer: u32, unit: u32, index: u32) -> u64 {
    ((kind as u64) << 56)
        | (((step & 0xff) as u64) << 48)
        | (((layer & 0xff) as u64) << 40)
        | (((unit & 0xff) as u64) << 32)
        | index as u64
}

/// The router output of (step, layer): the only label the Trace backend
/// looks up in the AdaMoE program.
pub fn router(step: u32, layer: u32) -> u64 {
    label(Kind::Router, step, layer, 0, 0)
}

/// (kind, step, layer, unit, index) of a label.
pub fn split(l: u64) -> (u8, u32, u32, u32, u32) {
    (
        (l >> 56) as u8,
        ((l >> 48) & 0xff) as u32,
        ((l >> 40) & 0xff) as u32,
        ((l >> 32) & 0xff) as u32,
        l as u32,
    )
}
