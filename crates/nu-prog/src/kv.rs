//! The prefix KV cache in HBM: written by the VLM prefill, read by the
//! action expert. One entry per layer: K then V, each `prefix x head_dim`
//! bf16, row-major (one MQA KV head), each padded to the granule.

use nu_model::Range;

/// The HBM layout of the prefix KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    /// Address of the K of layer 0.
    pub base: u64,
    /// Bytes per layer (K and V).
    pub stride: u64,
    /// Layers in the cache.
    pub layers: u32,
    /// Prefix entries (keys) per layer.
    pub prefix: u64,
    /// Elements per key or value.
    pub head_dim: u64,
}

impl KvLayout {
    /// The layout at `base` for `layers` layers of `prefix x head_dim` K and
    /// V, each padded to `granule` bytes.
    pub fn new(base: u64, layers: u32, prefix: u64, head_dim: u64, granule: u64) -> Self {
        let half = (prefix * head_dim * 2).div_ceil(granule) * granule;
        Self {
            base,
            stride: 2 * half,
            layers,
            prefix,
            head_dim,
        }
    }

    /// K of layer `l`.
    pub fn k(&self, l: u32) -> Range {
        Range::from_len(self.base + l as u64 * self.stride, self.bytes())
    }

    /// V of layer `l`.
    pub fn v(&self, l: u32) -> Range {
        Range::from_len(
            self.base + l as u64 * self.stride + self.stride / 2,
            self.bytes(),
        )
    }

    /// Unpadded bytes of one K (or V).
    pub fn bytes(&self) -> u64 {
        self.prefix * self.head_dim * 2
    }

    /// The first address after the last layer.
    pub fn end(&self) -> u64 {
        self.base + self.layers as u64 * self.stride
    }
}
