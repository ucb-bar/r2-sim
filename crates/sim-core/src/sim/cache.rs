//! A minimal L2 cache model (an experiment, 2026-10-15): a scratch memory
//! with `cache: {of: <dram>, line_bytes: L}` caches that DRAM. Its banks
//! are the slices; DRAM line i (a line of L bytes at a DRAM address) is
//! cached by the slice paired with its DRAM bank (channel): slice =
//! channel mod slices. Each slice keeps its resident lines, their LRU
//! order and a dirty bit per line; its capacity is its bank's bytes / L.
//!
//! The bookkeeping runs when a DMA's chunks are built (before its issue):
//! a chunk whose source is in the DRAM reads the slice on a hit and the
//! channel on a miss (the line becomes resident and clean); a chunk whose
//! destination is in the DRAM writes the slice (write-allocate without a
//! fetch; the line is resident and dirty). A new line in a full slice
//! evicts the slice's LRU line; a dirty victim adds a write-back chunk of
//! L bytes to the evicting task, from the slice to the victim's channel
//! over the device's fabric (the route halves of the cache owner's DMA
//! engine). A program's `cache_flush` empties every cache with no traffic.
//!
//! Not modeled: the fill write into the slice on a miss; sectors (32 B);
//! set associativity and the slice hash; hit-under-miss (a read that hits
//! a line still being filled reads the slice at once); replacement other
//! than LRU; the write-back buffer (a write-back is a chunk of the
//! evicting task, after the chunk that evicts); the bookkeeping order is
//! the order in which tasks' chunks are built, not the order of their
//! bytes in time.

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use crate::profile::{Args, Chunk, Descriptor, Half, PeId, Profile, TransferChunk};
use crate::resource::Dir;
use crate::time::Cycle;
use crate::tracer::RouteUse;

/// The configuration of one cache.
#[derive(Debug, Clone)]
pub(crate) struct CacheConfig {
    /// The caching memory's name (`<node>.<memory>`).
    pub name: String,
    pub line: u64,
    /// The slices (the caching memory's banks) and the cached DRAM's
    /// channels (its banks), as memory block indices in bank order.
    pub slices: Vec<usize>,
    pub channels: Vec<usize>,
    /// Lines per slice.
    pub capacity: u64,
}

/// One slice: resident lines by line number (LRU stamp, dirty) and the
/// LRU order (stamp -> line).
#[derive(Debug, Default, Clone)]
struct Slice {
    lines: HashMap<u64, (u64, bool)>,
    lru: BTreeMap<u64, u64>,
}

/// Counters of one cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct CacheCounts {
    pub read_hits: u64,
    pub read_misses: u64,
    pub write_hits: u64,
    /// Writes to a line that was not resident (allocated dirty).
    pub write_allocs: u64,
    /// Lines made resident (read misses and write allocations).
    pub allocs: u64,
    pub clean_evictions: u64,
    pub dirty_evictions: u64,
    pub writeback_bytes: u64,
    /// The most lines resident at once (over every slice) since the run
    /// start or the last flush.
    pub peak_lines: u64,
}

/// A flush: its cycle and the counters up to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CacheFlush {
    pub time: Cycle,
    pub counts: CacheCounts,
}

/// One cache's statistics (`Stats::caches`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CacheStats {
    pub name: String,
    pub line_bytes: u64,
    pub slices: usize,
    pub lines_per_slice: u64,
    /// Totals over the run.
    pub counts: CacheCounts,
    /// The counters at each flush, in order.
    pub flushes: Vec<CacheFlush>,
}

/// What one access did: the slice (index into the slices) and an evicted
/// line (line number, dirty).
pub(crate) struct Access {
    pub hit: bool,
    pub slice: usize,
    pub victim: Option<(u64, bool)>,
}

/// A cache's state.
#[derive(Debug, Clone)]
pub(crate) struct Cache {
    pub cfg: CacheConfig,
    slices: Vec<Slice>,
    stamp: u64,
    resident: u64,
    counts: CacheCounts,
    flushes: Vec<CacheFlush>,
}

impl Cache {
    pub fn new(cfg: CacheConfig) -> Self {
        let n = cfg.slices.len();
        Self {
            cfg,
            slices: vec![Slice::default(); n],
            stamp: 0,
            resident: 0,
            counts: CacheCounts::default(),
            flushes: Vec::new(),
        }
    }

    /// An access to line `line` of channel `channel` (index into the
    /// channels); `write` marks the line dirty.
    pub fn access(&mut self, line: u64, channel: usize, write: bool) -> Access {
        let si = channel % self.slices.len();
        self.stamp += 1;
        let stamp = self.stamp;
        let s = &mut self.slices[si];
        if let Some(e) = s.lines.get_mut(&line) {
            s.lru.remove(&e.0);
            e.0 = stamp;
            e.1 |= write;
            s.lru.insert(stamp, line);
            if write {
                self.counts.write_hits += 1;
            } else {
                self.counts.read_hits += 1;
            }
            return Access {
                hit: true,
                slice: si,
                victim: None,
            };
        }
        if write {
            self.counts.write_allocs += 1;
        } else {
            self.counts.read_misses += 1;
        }
        self.counts.allocs += 1;
        let mut victim = None;
        if s.lines.len() as u64 >= self.cfg.capacity {
            let (&old, &vline) = s.lru.iter().next().expect("a full slice has lines");
            s.lru.remove(&old);
            let (_, dirty) = s.lines.remove(&vline).expect("resident");
            self.resident -= 1;
            if dirty {
                self.counts.dirty_evictions += 1;
                self.counts.writeback_bytes += self.cfg.line;
            } else {
                self.counts.clean_evictions += 1;
            }
            victim = Some((vline, dirty));
        }
        s.lines.insert(line, (stamp, write));
        s.lru.insert(stamp, line);
        self.resident += 1;
        self.counts.peak_lines = self.counts.peak_lines.max(self.resident);
        Access {
            hit: false,
            slice: si,
            victim,
        }
    }

    /// Empties the cache with no traffic; records the counters.
    pub fn flush(&mut self, now: Cycle) {
        self.flushes.push(CacheFlush {
            time: now,
            counts: self.counts,
        });
        for s in &mut self.slices {
            s.lines.clear();
            s.lru.clear();
        }
        self.resident = 0;
        self.counts.peak_lines = 0;
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            name: self.cfg.name.clone(),
            line_bytes: self.cfg.line,
            slices: self.cfg.slices.len(),
            lines_per_slice: self.cfg.capacity,
            counts: self.counts,
            flushes: self.flushes.clone(),
        }
    }
}

/// The tracer's route use of `bytes` over `half` in direction `dir`.
fn route_use(half: &Half, bytes: u64, dir: Dir) -> Option<RouteUse> {
    half.pair.map(|(from, to)| RouteUse {
        from,
        to,
        bytes,
        dir,
    })
}

impl super::Sim {
    /// At the run start (the routes built): the channel blocks of each
    /// cache, and the DMA engine whose route halves carry its write-backs
    /// (an engine of the slices' owner that reaches a slice and a channel).
    pub(super) fn setup_caches(&mut self) {
        self.cache_of = vec![None; self.memories.len()];
        self.cache_wb.clear();
        for (ci, c) in self.caches.iter().enumerate() {
            for (k, &b) in c.cfg.channels.iter().enumerate() {
                self.cache_of[b] = Some((ci, k));
            }
            let owner = self.memories[c.cfg.slices[0]].owner;
            let pe = (0..self.pes.len())
                .map(|i| PeId(i as u32))
                .find(|&p| {
                    self.pes[p.0 as usize].nu == owner
                        && self.routes.read(p, c.cfg.slices[0]).is_some()
                        && self.routes.write(p, c.cfg.channels[0]).is_some()
                })
                .unwrap_or_else(|| {
                    panic!(
                        "cache {}: no DMA engine of its owner reaches its slices and its DRAM",
                        c.cfg.name
                    )
                });
            self.cache_wb.push(pe);
        }
    }

    /// The cache and channel index of address `a`, if it is in a cached
    /// DRAM.
    fn cached(&self, a: u64) -> Option<(usize, usize)> {
        self.routes.locate(a).and_then(|(b, _)| self.cache_of[b])
    }

    /// The cache bookkeeping of DMA `desc` on engine `pe` (module doc):
    /// redirects the chunks of `prof` that hit or write a cached line to
    /// its slice, and adds a write-back chunk after each chunk that evicts
    /// a dirty line.
    pub(super) fn cache_chunks(&mut self, pe: PeId, desc: &Descriptor, prof: &mut Profile) {
        let Args::Dma { src, dst, .. } = desc.args else {
            return;
        };
        if !prof.chunks.iter().any(|c| c.transfer.is_some()) {
            return;
        }
        // A chunk with a side in a cached DRAM is cut at that side's line
        // boundaries: one line per chunk.
        let mut pieces: Vec<(Chunk, u64)> = Vec::with_capacity(prof.chunks.len());
        let mut at = 0u64;
        for c in std::mem::take(&mut prof.chunks) {
            let Some(t) = &c.transfer else {
                pieces.push((c, at));
                continue;
            };
            let (n, start) = (t.bytes, at);
            at += n;
            let line_of = |a: u64| self.cached(a).map(|(ci, _)| self.caches[ci].cfg.line);
            let (ls, ld) = (line_of(src.0 + start), line_of(dst.0 + start));
            if ls.is_none() && ld.is_none() {
                pieces.push((c, start));
                continue;
            }
            let mut off = 0;
            while off < n {
                let run = |base: u64, l: Option<u64>| {
                    l.map_or(u64::MAX, |l| l - ((base + start + off) % l))
                };
                let k = (n - off).min(run(src.0, ls)).min(run(dst.0, ld));
                let mut p = c.clone();
                let pt = p.transfer.as_mut().expect("a transfer");
                pt.bytes = k;
                for u in p.routes.iter_mut().flatten() {
                    u.bytes = k;
                }
                pieces.push((p, start + off));
                off += k;
            }
        }
        let mut out: Vec<Chunk> = Vec::with_capacity(pieces.len());
        for (i, (mut c, off)) in pieces.into_iter().enumerate() {
            let Some(t) = c.transfer.as_mut() else {
                out.push(c);
                continue;
            };
            let n = t.bytes;
            let (sa, da) = (src.0 + off, dst.0 + off);
            let mut wbs = Vec::new();
            if let Some((ci, ch)) = self.cached(sa) {
                let line = self.caches[ci].cfg.line;
                let a = self.caches[ci].access(sa / line, ch, false);
                if a.hit {
                    let sb = self.caches[ci].cfg.slices[a.slice];
                    if let Some(h) = self.routes.read(pe, sb) {
                        t.read = Rc::clone(h);
                        c.routes[0] = route_use(h, n, Dir::Read);
                        if i == 0 {
                            prof.fill = self.memories[sb].load_latency;
                        }
                    }
                }
                wbs.extend(self.writeback(ci, a.slice, a.victim, t.beat, c.rate));
            }
            if let Some((ci, ch)) = self.cached(da) {
                let line = self.caches[ci].cfg.line;
                let a = self.caches[ci].access(da / line, ch, true);
                let sb = self.caches[ci].cfg.slices[a.slice];
                if let Some(h) = self.routes.write(pe, sb) {
                    t.write = Rc::clone(h);
                    c.routes[1] = route_use(h, n, Dir::Write);
                }
                wbs.extend(self.writeback(ci, a.slice, a.victim, t.beat, c.rate));
            }
            out.push(c);
            out.extend(wbs);
        }
        prof.chunks = out;
    }

    /// The write-back chunk of a dirty `victim` (line, dirty) of slice
    /// `slice` of cache `ci`: one line from the slice to the victim's
    /// channel over the route halves of the cache's write-back engine.
    fn writeback(
        &self,
        ci: usize,
        slice: usize,
        victim: Option<(u64, bool)>,
        beat: u64,
        rate: crate::time::Ratio,
    ) -> Option<Chunk> {
        let (vline, true) = victim? else {
            return None;
        };
        let cfg = &self.caches[ci].cfg;
        let (cb, _) = self
            .routes
            .locate(vline * cfg.line)
            .expect("a resident line is in its DRAM");
        let wb = self.cache_wb[ci];
        let read = self
            .routes
            .read(wb, cfg.slices[slice])
            .expect("checked at setup");
        let write = self
            .routes
            .write(wb, cb)
            .unwrap_or_else(|| panic!("cache {}: no write-back route to a channel", cfg.name));
        Some(Chunk {
            demand: Vec::new(),
            routes: [
                route_use(read, cfg.line, Dir::Read),
                route_use(write, cfg.line, Dir::Write),
            ],
            transfer: Some(TransferChunk {
                read: Rc::clone(read),
                write: Rc::clone(write),
                bytes: cfg.line,
                beat,
            }),
            rate,
            latency: 0,
        })
    }
}
