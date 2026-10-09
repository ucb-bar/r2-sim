//! Global addresses, ranges, address sets, and ownership.
//!
//! Addresses are 64 bit. The high bits are prefix fields that select a node
//! of the hardware tree; the owner of an address is the Neutrino whose
//! address sets contain it (architecture Section 3.1). The field layout
//! comes from the configuration, so this module only sees the result: the
//! owners' address sets. Task write ranges and transfer spans stay
//! contiguous [`Range`]s; ownership and memory blocks use [`AddressSet`]s.

use std::fmt;

/// A global address.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Addr(pub u64);

/// Identifies one Neutrino instance in the device.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NeutrinoId(pub u32);

/// A half-open range `[base, bound)` in the global address space.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Range {
    /// First address in the range.
    pub base: u64,
    /// One past the last address in the range.
    pub bound: u64,
}

impl Range {
    /// Builds a range; panics if `bound < base`.
    pub fn new(base: u64, bound: u64) -> Self {
        assert!(bound >= base, "range bound {bound:#x} below base {base:#x}");
        Self { base, bound }
    }

    /// Builds a range from a base and a length.
    pub fn from_len(base: u64, len: u64) -> Self {
        Self::new(base, base + len)
    }

    /// Length in bytes.
    pub fn len(&self) -> u64 {
        self.bound - self.base
    }

    /// True when the range has no bytes.
    pub fn is_empty(&self) -> bool {
        self.bound == self.base
    }

    /// True when the two ranges share at least one byte. Empty ranges never
    /// overlap anything.
    pub fn overlaps(&self, other: &Range) -> bool {
        !self.is_empty() && !other.is_empty() && self.base < other.bound && other.base < self.bound
    }

    /// True when `addr` lies inside the range.
    pub fn contains(&self, addr: Addr) -> bool {
        self.base <= addr.0 && addr.0 < self.bound
    }

    /// True when `other` lies entirely inside `self`.
    pub fn covers(&self, other: &Range) -> bool {
        self.base <= other.base && other.bound <= self.bound
    }

    /// True when both ends are multiples of `granule`.
    pub fn is_aligned(&self, granule: u64) -> bool {
        granule.is_power_of_two()
            && self.base.is_multiple_of(granule)
            && self.bound.is_multiple_of(granule)
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:#x}, {:#x})", self.base, self.bound)
    }
}

/// A set of addresses in the Chisel/Rocket sense: `mask` marks the
/// address bits that are free, so the set holds `a` when
/// `(a & !mask) == base`. `base & mask == 0` always. A block of a
/// power-of-two size aligned to its base is `base` with `mask = size - 1`;
/// a slice of an interleaved memory has free bits above and below its
/// selector bits.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AddressSet {
    pub base: u64,
    pub mask: u64,
}

impl AddressSet {
    /// Builds a set; panics if `base` has a bit of `mask`.
    pub fn new(base: u64, mask: u64) -> Self {
        assert!(
            base & mask == 0,
            "address set base {base:#x} has bits of its mask {mask:#x}"
        );
        Self { base, mask }
    }

    /// The block `[base, base + size)`; panics unless `size` is a power of
    /// two and `base` a multiple of it.
    pub fn block(base: u64, size: u64) -> Self {
        assert!(
            size.is_power_of_two() && base.is_multiple_of(size),
            "[{base:#x}, +{size:#x}) is not an aligned power-of-two block"
        );
        Self::new(base, size - 1)
    }

    /// The fewest aligned power-of-two blocks whose union is `r`, in
    /// address order. Empty for an empty range.
    pub fn decompose(r: &Range) -> Vec<AddressSet> {
        let mut out = Vec::new();
        let mut a = r.base;
        while a < r.bound {
            // The largest block that starts at `a` (its alignment) and fits.
            let align = if a == 0 {
                1u64 << 63
            } else {
                1u64 << a.trailing_zeros()
            };
            let mut size = align;
            while size > r.bound - a {
                size >>= 1;
            }
            out.push(Self::block(a, size));
            a += size;
        }
        out
    }

    /// True when the set holds `addr`.
    pub fn contains(&self, addr: Addr) -> bool {
        addr.0 & !self.mask == self.base
    }

    /// True when the two sets share an address.
    pub fn overlaps(&self, other: &AddressSet) -> bool {
        (self.base ^ other.base) & !self.mask & !other.mask == 0
    }

    /// The low part of the mask: its trailing ones.
    fn low(&self) -> u64 {
        self.mask & !(self.mask.wrapping_add(1))
    }

    /// The contiguous bytes from `addr` inside the set (0 when the set
    /// does not hold `addr`): `size - offset` for a block, `G - (addr mod
    /// G)` for a slice of interleave `G`.
    pub fn run(&self, addr: u64) -> u64 {
        if !self.contains(Addr(addr)) {
            return 0;
        }
        let low = self.low();
        (low - (addr & low)).saturating_add(1)
    }

    /// True when the set holds every byte of `r` (a nonempty range).
    pub fn covers(&self, r: &Range) -> bool {
        !r.is_empty() && self.run(r.base) >= r.len()
    }

    /// True when the set is one contiguous block.
    pub fn is_contiguous(&self) -> bool {
        self.mask == self.low()
    }

    /// Bytes in the set (saturated at `u64::MAX`).
    pub fn size(&self) -> u64 {
        1u64.checked_shl(self.mask.count_ones()).unwrap_or(u64::MAX)
    }

    /// The lowest and one past the highest address of the set: the
    /// extent of a contiguous set.
    pub fn extent(&self) -> Range {
        Range::new(self.base, (self.base | self.mask).saturating_add(1))
    }
}

impl fmt::Display for AddressSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{base {:#x}, mask {:#x}}}", self.base, self.mask)
    }
}

/// The address rewrite of a hash link (architecture Section 3.1): a
/// linear range onto a physical one. Slice index bit j is the parity of
/// the linear address ANDed with `index[j]`; the offset inside the slice
/// is the address relative to `range` with the `drop` bits removed;
/// `physical = to_range.base + index x slice_bytes + offset`.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressHash {
    pub range: Range,
    pub to_range: Range,
    pub index: Vec<u64>,
    /// Bit positions, ascending.
    pub drop: Vec<u32>,
}

impl AddressHash {
    /// Builds the rule; `drop` `None` takes the index bits when each mask
    /// has one bit. Errors when the rewrite is not a bijection from
    /// `range` onto `to_range`.
    pub fn new(
        range: Range,
        to_range: Range,
        index: Vec<u64>,
        drop: Option<Vec<u32>>,
    ) -> Result<Self, String> {
        let k = index.len();
        if k == 0 || k > 32 {
            return Err(format!("{} index masks (1 to 32)", k));
        }
        if index.contains(&0) {
            return Err("an index mask of 0".into());
        }
        let mut drop = match drop {
            Some(d) => d,
            None if index.iter().all(|m| m.is_power_of_two()) => {
                index.iter().map(|m| m.trailing_zeros()).collect()
            }
            None => return Err("drop is needed when an index mask has more than one bit".into()),
        };
        drop.sort_unstable();
        if drop.len() != k || drop.windows(2).any(|w| w[0] == w[1]) || drop.iter().any(|&d| d > 62)
        {
            return Err(format!(
                "drop needs {k} distinct bits below 63, one per index mask"
            ));
        }
        let len = range.len();
        if len == 0 || to_range.len() != len {
            return Err(format!(
                "range {range} and to_range {to_range} differ in size"
            ));
        }
        let slices = 1u64 << k;
        if !len.is_multiple_of(slices) {
            return Err(format!("{range} does not split into {slices} slices"));
        }
        // Removing the drop bits maps the range onto [0, len / 2^k) only
        // when every drop bit lies below an aligned period of the range.
        let period = 1u64 << (drop[k - 1] + 1);
        if !len.is_multiple_of(period) || !range.base.is_multiple_of(period) {
            return Err(format!(
                "{range} is not a multiple of {period:#x}, the period of the drop bits"
            ));
        }
        // The index restricted to the drop bits must be invertible over
        // GF(2): then the drop bits follow from the index and the offset.
        let mut rows: Vec<u64> = index
            .iter()
            .map(|m| {
                drop.iter()
                    .enumerate()
                    .fold(0, |r, (i, &d)| r | ((m >> d) & 1) << i)
            })
            .collect();
        for col in 0..k {
            let Some(p) = (col..k).find(|&r| rows[r] >> col & 1 == 1) else {
                return Err(format!(
                    "the index masks restricted to the drop bits {drop:?} are not invertible"
                ));
            };
            rows.swap(col, p);
            for r in 0..k {
                if r != col && rows[r] >> col & 1 == 1 {
                    rows[r] ^= rows[col];
                }
            }
        }
        Ok(Self {
            range,
            to_range,
            index,
            drop,
        })
    }

    /// Bytes of one slice of `to_range`.
    pub fn slice_bytes(&self) -> u64 {
        self.range.len() >> self.index.len()
    }

    /// The slice granule: the lowest bit of any mask. A chunk inside one
    /// granule stays in one slice.
    pub fn granule(&self) -> u64 {
        1 << self
            .index
            .iter()
            .map(|m| m.trailing_zeros())
            .min()
            .unwrap_or(0)
    }

    /// The slice index of linear address `a`.
    pub fn slice(&self, a: u64) -> u64 {
        self.index
            .iter()
            .enumerate()
            .fold(0, |s, (j, m)| s | (((a & m).count_ones() & 1) as u64) << j)
    }

    /// The physical address of linear address `a` (in `range`).
    pub fn rewrite(&self, a: u64) -> u64 {
        let mut off = a - self.range.base;
        for &d in self.drop.iter().rev() {
            off = (off & ((1u64 << d) - 1)) | ((off >> (d + 1)) << d);
        }
        self.to_range.base + self.slice(a) * self.slice_bytes() + off
    }
}

/// Maps addresses to their owning Neutrino by address set.
///
/// Sets are disjoint. Contiguous sets (every prefix and plain block) are
/// kept sorted by base for a binary search; other sets are searched in
/// order.
#[derive(Debug, Clone, Default)]
pub struct OwnerMap {
    /// Contiguous sets by base, as extents.
    blocks: Vec<(Range, NeutrinoId)>,
    /// The other sets.
    sets: Vec<(AddressSet, NeutrinoId)>,
}

impl OwnerMap {
    /// Adds an owner set. Panics if it overlaps an existing one.
    pub fn insert(&mut self, set: AddressSet, owner: NeutrinoId) {
        let all = self
            .blocks
            .iter()
            .flat_map(|(r, o)| AddressSet::decompose(r).into_iter().map(move |s| (s, *o)))
            .chain(self.sets.iter().copied());
        for (s, o) in all {
            assert!(!s.overlaps(&set), "owner set {set} overlaps {s} of {o:?}");
        }
        if set.is_contiguous() {
            let r = set.extent();
            let pos = self.blocks.partition_point(|(b, _)| b.base < r.base);
            self.blocks.insert(pos, (r, owner));
        } else {
            self.sets.push((set, owner));
        }
    }

    /// Adds the sets of the range `r` (its decomposition).
    pub fn insert_range(&mut self, r: &Range, owner: NeutrinoId) {
        for s in AddressSet::decompose(r) {
            self.insert(s, owner);
        }
    }

    /// The owner of `addr` and the contiguous bytes from `addr` in the
    /// set that holds it.
    fn lookup(&self, addr: u64) -> Option<(NeutrinoId, u64)> {
        let i = self.blocks.partition_point(|(r, _)| r.bound <= addr);
        if let Some((r, o)) = self.blocks.get(i).filter(|(r, _)| r.contains(Addr(addr))) {
            return Some((*o, r.bound - addr));
        }
        self.sets
            .iter()
            .find(|(s, _)| s.contains(Addr(addr)))
            .map(|(s, o)| (*o, s.run(addr)))
    }

    /// The Neutrino that owns `addr`, if any.
    pub fn owner_of(&self, addr: Addr) -> Option<NeutrinoId> {
        self.lookup(addr.0).map(|(o, _)| o)
    }

    /// The Neutrino that owns the whole `range` (in one or more of its
    /// sets), or `None` if a byte is unowned or another Neutrino's.
    pub fn owner_of_range(&self, range: &Range) -> Option<NeutrinoId> {
        let (owner, mut run) = self.lookup(range.base)?;
        let mut a = range.base;
        while range.bound - a > run {
            a += run;
            let (o, r) = self.lookup(a)?;
            if o != owner {
                return None;
            }
            run = r;
        }
        Some(owner)
    }
}
