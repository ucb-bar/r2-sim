//! Global addresses, ranges, and prefix ownership.
//!
//! Addresses are 64 bit. The high bits are prefix fields that select a node
//! of the hardware tree; the owner of an address is the Neutrino whose
//! prefix covers it. The field layout comes from the configuration, so this
//! module only sees the result: a sorted list of owner ranges.

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

/// Maps addresses to their owning Neutrino.
///
/// Owner ranges are disjoint and sorted by base. A prefix in the address
/// layout is always a contiguous range, so a sorted list is enough and the
/// map stays independent of the field layout.
#[derive(Debug, Clone, Default)]
pub struct OwnerMap {
    owners: Vec<(Range, NeutrinoId)>,
}

impl OwnerMap {
    /// Adds an owner range. Panics if it overlaps an existing one.
    pub fn insert(&mut self, range: Range, owner: NeutrinoId) {
        for (r, o) in &self.owners {
            assert!(
                !r.overlaps(&range),
                "owner range {range} overlaps {r} of {o:?}"
            );
        }
        let pos = self.owners.partition_point(|(r, _)| r.base < range.base);
        self.owners.insert(pos, (range, owner));
    }

    /// The Neutrino that owns `addr`, if any.
    pub fn owner_of(&self, addr: Addr) -> Option<NeutrinoId> {
        let i = self.owners.partition_point(|(r, _)| r.bound <= addr.0);
        self.owners
            .get(i)
            .filter(|(r, _)| r.contains(addr))
            .map(|(_, o)| *o)
    }

    /// The Neutrino that owns the whole `range`, or `None` if the range is
    /// unowned or crosses an owner boundary.
    pub fn owner_of_range(&self, range: &Range) -> Option<NeutrinoId> {
        let i = self.owners.partition_point(|(r, _)| r.bound <= range.base);
        self.owners
            .get(i)
            .filter(|(r, _)| r.covers(range))
            .map(|(_, o)| *o)
    }
}
