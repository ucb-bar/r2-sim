//! The label scheme: software identity of a task, for the tracers.
//! `kind:8 | step:8 | layer:8 | unit:8 | index:32`.

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
