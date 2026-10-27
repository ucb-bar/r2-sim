//! The label scheme: software identity of a task, for the tracer and for
//! Trace records. `kind:8 | step:8 | layer:8 | unit:8 | index:32`.

/// Task kinds of the kernels program. The numbers 1 to 11 but 8 were
/// the kinds of earlier programs; [`kind_name`] keeps their names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Kind {
    Vec = 8,
    Load = 12,
    Store = 13,
    Matmul = 14,
    Softmax = 15,
    Norm = 16,
    Eltwise = 17,
    Rescale = 18,
}

/// The number of kind numbers (0, none, included).
pub const KINDS: u8 = 19;

/// The tracer name of kind number `k` ("none" for 0, "other" when unknown).
pub fn kind_name(k: u8) -> &'static str {
    const NAMES: [&str; KINDS as usize] = [
        "none", "router", "qkv", "attn", "wo", "ffn_gate", "ffn_up", "ffn_down", "vec", "dma_w",
        "dma_act", "join", "load", "store", "matmul", "softmax", "norm", "eltwise", "rescale",
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
