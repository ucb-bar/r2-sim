//! Destination ranges with a mask (2026-10-21; Neutrino spec, granules and
//! destination ranges).
//!
//! The spec's destination range is base, bound and mask, in granules:
//! granule `base + i`, `0 <= i < bound`, belongs to the range when
//! `i & !mask == 0`. A set mask bit is a bit of `i` that can vary, as in
//! Rocket's address sets; all ones holds the whole span. r2-sim keeps the
//! span in bytes (aligned to the granule) and the mask in granules: [`Dst`].

use crate::addr::Range;

/// The mask of a plain range: every granule of the span.
pub const FULL_MASK: u32 = u32::MAX;

/// A destination range: a byte span aligned to the granule and the mask of
/// the granules it holds.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dst {
    /// The span: granules `[span.base / g, span.bound / g)`.
    pub span: Range,
    /// The granules of the span the range holds ([`FULL_MASK`]: all).
    pub mask: u32,
}

impl Dst {
    /// A plain range: the whole span.
    pub fn plain(span: Range) -> Self {
        Self {
            span,
            mask: FULL_MASK,
        }
    }

    /// The span's granules.
    fn granules(&self, g: u64) -> u64 {
        self.span.len() / g
    }

    /// True when every granule of the span is held.
    pub fn is_whole(&self, g: u64) -> bool {
        let n = self.granules(g);
        n == 0 || (n - 1) & !(self.mask as u64) == 0
    }

    /// True when the range holds the granule of byte `addr`.
    pub fn holds(&self, g: u64, addr: u64) -> bool {
        self.span.base <= addr
            && addr < self.span.bound
            && ((addr - self.span.base) / g) & !(self.mask as u64) == 0
    }

    /// The held granules as maximal byte runs, in address order.
    pub fn runs(&self, g: u64) -> Vec<Range> {
        if self.is_whole(g) {
            return if self.span.is_empty() {
                Vec::new()
            } else {
                vec![self.span]
            };
        }
        let (n, m) = (self.granules(g), self.mask as u64);
        let mut out: Vec<Range> = Vec::new();
        // The offsets i with i & !m == 0, ascending: the submasks of m.
        let mut i = 0u64;
        loop {
            if i >= n {
                break;
            }
            let a = self.span.base + i * g;
            match out.last_mut() {
                Some(r) if r.bound == a => r.bound = a + g,
                _ => out.push(Range::from_len(a, g)),
            }
            let next = (i | !m).wrapping_add(1) & m;
            if next == 0 {
                break;
            }
            i = next;
        }
        out
    }

    /// True when the range holds a granule that `r` touches.
    pub fn overlaps(&self, g: u64, r: &Range) -> bool {
        if !self.span.overlaps(r) {
            return false;
        }
        if self.is_whole(g) {
            return true;
        }
        self.runs(g).iter().any(|x| x.overlaps(r))
    }

    /// True when every granule that `r` touches is held.
    pub fn covers_bytes(&self, g: u64, r: &Range) -> bool {
        if r.is_empty() {
            return true;
        }
        if !self.span.covers(r) {
            return false;
        }
        if self.is_whole(g) {
            return true;
        }
        let first = r.base / g * g;
        (first..r.bound)
            .step_by(g as usize)
            .all(|a| self.holds(g, a))
    }

    /// True when every granule of `other` is held.
    pub fn covers(&self, g: u64, other: &Dst) -> bool {
        if other.is_whole(g) {
            return self.covers_bytes(g, &other.span);
        }
        other.runs(g).iter().all(|r| self.covers_bytes(g, r))
    }

    /// True when the two ranges share a granule.
    pub fn meets(&self, g: u64, other: &Dst) -> bool {
        if !self.span.overlaps(&other.span) {
            return false;
        }
        if other.is_whole(g) {
            return self.overlaps(g, &other.span);
        }
        other.runs(g).iter().any(|r| self.overlaps(g, r))
    }

    /// The mask as the spec states it: [`FULL_MASK`] when the span is
    /// whole, so that two masks of one set of granules compare equal.
    pub fn canonical_mask(&self, g: u64) -> u32 {
        if self.is_whole(g) {
            FULL_MASK
        } else {
            self.mask
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The warp-tile example of the spec, granule 128 B: warp 0 holds the
    /// even granules of 128 from its base, warp 1 the odd ones.
    #[test]
    fn the_warp_tile_example() {
        let g = 128;
        let c = 0x10000;
        let w0 = Dst {
            span: Range::from_len(c, 128 * g),
            mask: 0x7e,
        };
        let w1 = Dst {
            span: Range::from_len(c + g, 128 * g),
            mask: 0x7e,
        };
        assert!(w0.holds(g, c) && w0.holds(g, c + 2 * g) && w0.holds(g, c + 126 * g));
        assert!(!w0.holds(g, c + g) && !w0.holds(g, c + 127 * g));
        assert_eq!(w0.runs(g).len(), 64);
        assert!(!w0.meets(g, &w1) && !w1.meets(g, &w0));
        assert!(w0.span.overlaps(&w1.span));
        let whole = Dst::plain(Range::from_len(c, 256 * g));
        assert!(whole.covers(g, &w0) && whole.covers(g, &w1));
        assert!(!w0.covers(g, &whole));
        assert!(w0.covers_bytes(g, &Range::from_len(c + 2 * g + 5, 100)));
        assert!(!w0.covers_bytes(g, &Range::from_len(c, 2 * g)));
    }

    #[test]
    fn a_mask_with_every_bit_of_the_span_is_whole() {
        let g = 512;
        let d = Dst {
            span: Range::from_len(0x1000, 4 * g),
            mask: 0b11,
        };
        assert!(d.is_whole(g));
        assert_eq!(d.canonical_mask(g), FULL_MASK);
        assert_eq!(d.runs(g), vec![d.span]);
    }
}
