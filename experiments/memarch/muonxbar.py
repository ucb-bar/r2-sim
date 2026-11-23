"""Cycle model of Muon warp accesses into a banked smem through a crossbar.

Gives, per access pattern, the efficiency of a Muon core's memory port:
(ideal cycles) / (modeled cycles), where ideal is one cycle per warp access.
r2-sim takes it as a scale on the core's smem link bandwidth.

Model:
- `cores` Muon cores of `lanes` lanes each issue one warp access at a time
  (4 B per lane); a core issues its next access once every lane of the current
  one is served. All cores run the same pattern on disjoint regions.
- smem: R banks x S subbanks of macros, `w` bits wide. A macro row holds
  w/32 words; word i is in subbank (i / wpr) mod S, bank (i / (wpr S)) mod R,
  row i / (wpr S R). Each macro serves one row per cycle (1R): lanes that hit
  the same row of a macro are served together (coalesced or broadcast); lanes
  that hit different rows of one macro conflict.
- Crossbar: inputs = cores x lanes, outputs = R x S macros. Full (radix None):
  only macro conflicts. Multistage radix r: an Omega network of k = ceil(log_r n)
  stages of r x r switches (n = r^k >= both sides) with destination-tag routing;
  two requests that need the same inter-stage link in one cycle block, unless
  they go to the same macro row (merged).
- Each cycle, requests are granted in a rotating priority order if their macro
  is free (or already serving that row) and every link of their path is free.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache

PATTERNS = ["row", "col", "stride2", "stride4", "stride32", "gather", "broadcast"]


def addresses(pattern: str, lanes: int, access: int, region: int, rng: random.Random) -> list[int]:
    """Word addresses of warp access number `access` of a core whose data
    starts at word `region`. Row-major tile of 128 fp32 per row."""
    ld = 128
    if pattern == "row":
        return [region + access * lanes + l for l in range(lanes)]
    if pattern == "col":
        # lane l reads row l of column `access` (a transpose read)
        rb = (access // ld) * lanes
        return [region + (rb + l) * ld + access % ld for l in range(lanes)]
    if pattern.startswith("stride"):
        s = int(pattern[6:])
        return [region + (access * lanes + l) * s for l in range(lanes)]
    if pattern == "gather":
        return [region + rng.randrange(16384) for _ in range(lanes)]
    if pattern == "broadcast":
        return [region + access] * lanes
    raise ValueError(pattern)


def place(word: int, banks: int, subbanks: int, wpr: int) -> tuple[int, int]:
    """(macro index, row) of a word."""
    line = word // wpr
    sb = line % subbanks
    bank = (line // subbanks) % banks
    row = line // (subbanks * banks)
    return bank * subbanks + sb, row


@lru_cache(maxsize=None)
def _shape(n_in: int, n_out: int, radix: int | None) -> tuple[int, int]:
    if radix is None:
        return 0, 0
    n = max(n_in, n_out, radix)
    k = max(1, math.ceil(math.log(n, radix) - 1e-9))
    return k, radix**k


def links(src: int, dst: int, k: int, n: int, radix: int) -> list[tuple[int, int]]:
    """Inter-stage links (stage, position) of an Omega route src -> dst."""
    out = []
    for s in range(k - 1):  # the last stage's output is the macro itself
        pos = (src * radix ** (s + 1) + dst // radix ** (k - s - 1)) % n
        out.append((s, pos))
    return out


def efficiency(pattern: str, banks: int, subbanks: int, width: int, radix: int | None,
               lanes: int = 16, cores: int = 2, accesses: int = 256, seed: int = 1) -> float:
    rng = random.Random(seed)
    wpr = max(1, width // 32)
    macros = banks * subbanks
    n_in = lanes * cores
    k, n = _shape(n_in, macros, radix if radix and radix < max(n_in, macros) else None)
    # Regions far apart (and not aligned to the bank cycle, as a real layout).
    regions = [c * (1 << 20) + c * 7 * wpr for c in range(cores)]
    pending = [[] for _ in range(cores)]   # (lane input, macro, row) still to serve
    issued = [0] * cores
    done = 0
    cycle = 0
    prio = 0
    while done < accesses * cores:
        for c in range(cores):
            if not pending[c] and issued[c] < accesses:
                ws = addresses(pattern, lanes, issued[c], regions[c], rng)
                pending[c] = [(c * lanes + l, *place(w, banks, subbanks, wpr)) for l, w in enumerate(ws)]
                issued[c] += 1
        busy_macro: dict[int, int] = {}
        busy_link: dict[tuple[int, int], tuple[int, int]] = {}
        order = [(c, i) for c in range(cores) for i in range(len(pending[c]))]
        order = order[prio % max(1, len(order)):] + order[: prio % max(1, len(order))]
        granted = {c: set() for c in range(cores)}
        for c, i in order:
            src, m, row = pending[c][i]
            if m in busy_macro and busy_macro[m] != row:
                continue
            path = links(src, m, k, n, radix) if k else []
            if any(l in busy_link and busy_link[l] != (m, row) for l in path):
                continue
            busy_macro[m] = row
            for l in path:
                busy_link[l] = (m, row)
            granted[c].add(i)
        for c in range(cores):
            if granted[c]:
                pending[c] = [p for j, p in enumerate(pending[c]) if j not in granted[c]]
                if not pending[c]:
                    done += 1
        cycle += 1
        prio += 1
    # Ideal: each core finishes one access per cycle.
    return accesses / cycle


if __name__ == "__main__":
    import sys

    S = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print(f"banks=2 subbanks={S} width=32")
    print(f"{'pattern':>10} " + " ".join(f"{str(r):>6}" for r in [2, 4, 8, None]))
    for p in PATTERNS:
        print(f"{p:>10} " + " ".join(f"{efficiency(p, 2, S, 32, r):6.3f}" for r in [2, 4, 8, None]))
