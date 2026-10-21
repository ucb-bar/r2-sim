//! Last-writer tracking for functional sync (r2sim spec Section 3.4).
//!
//! A map from disjoint address intervals to the func seq of the op that
//! wrote them last and the gid of the task that completed last on them. A
//! load of a range waits for the largest seq over the intervals it
//! overlaps. Completed tasks (seq 0 for a join) and Neutrino stores (no
//! gid) enter the map. Unordered writers keep the larger seq, because the
//! worker applies ops in seq order; the gid is always the latest one.

use std::collections::BTreeMap;

use nu_model::Range;

#[derive(Debug, Default)]
pub(crate) struct LastWriters {
    /// base -> (bound, seq, gid); disjoint.
    map: BTreeMap<u64, (u64, u64, Option<u64>)>,
}

impl LastWriters {
    /// Records that op `seq` of task `gid` wrote `range`.
    pub fn insert(&mut self, range: Range, seq: u64, gid: Option<u64>) {
        if range.is_empty() {
            return;
        }
        let (b, e) = (range.base, range.bound);
        let hit: Vec<(u64, u64, u64, Option<u64>)> = self.overlapping(&range).collect();
        for &(base, ..) in &hit {
            self.map.remove(&base);
        }
        // Outside parts keep their seq; inside parts take the larger seq.
        let mut cursor = b;
        for (base, bound, s, g) in hit {
            if base < b {
                self.map.insert(base, (b, s, g));
            }
            if bound > e {
                self.map.insert(e, (bound, s, g));
            }
            let (ib, ie) = (base.max(b), bound.min(e));
            if cursor < ib {
                self.map.insert(cursor, (ib, seq, gid));
            }
            self.map.insert(ib, (ie, s.max(seq), gid));
            cursor = ie;
        }
        if cursor < e {
            self.map.insert(cursor, (e, seq, gid));
        }
    }

    /// The largest seq that wrote any byte of `range`; 0 if none.
    pub fn max_seq(&self, range: &Range) -> u64 {
        self.overlapping(range)
            .map(|(.., s, _)| s)
            .max()
            .unwrap_or(0)
    }

    /// Some writer of any byte of `range`: `Some(gid)` for a task,
    /// `Some(None)` for a store, `None` if nothing wrote it.
    pub fn any_writer(&self, range: &Range) -> Option<Option<u64>> {
        self.overlapping(range).next().map(|(.., g)| g)
    }

    /// The intervals that overlap `range`, in address order, as (base,
    /// bound, seq, gid).
    fn overlapping(
        &self,
        range: &Range,
    ) -> impl Iterator<Item = (u64, u64, u64, Option<u64>)> + '_ {
        let before = self
            .map
            .range(..range.base)
            .next_back()
            .filter(|(_, &(bound, ..))| bound > range.base);
        before
            .into_iter()
            .chain(self.map.range(range.base..range.bound))
            .map(|(&base, &(bound, s, g))| (base, bound, s, g))
    }
}
