"""Area model of the SM memory architectures: SRAM macros, crossbars, logic.

All constants come from params/ (sram_area.csv, logic.yaml) so they can be
swapped without touching code. `python3 area.py` prints the SRAM fit and its
error at every table point.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARAMS = Path(__file__).parent / "params"

# Compiler range (two-port register file, 1R1W).
MIN_BITS, MAX_BITS = 256, 64 * 1024
MIN_W, MAX_W = 32, 256


def load_table(path: Path = PARAMS / "sram_area.csv"):
    """(kbit, width, area_per_bit) for every filled cell."""
    rows = [r for r in path.read_text().splitlines() if r and not r.startswith("#")]
    reader = csv.DictReader(rows)
    pts = []
    for r in reader:
        kbit = float(r["kbit"])
        for col, v in r.items():
            if col.startswith("w") and v:
                pts.append((kbit, int(col[1:]), float(v)))
    return pts


def load_logic(path: Path = PARAMS / "logic.yaml") -> dict:
    """Flat `key: value` YAML (no dependency on PyYAML)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            k, v = (s.strip() for s in line.split(":", 1))
            out[k] = v == "true" if v in ("true", "false") else float(v)
    return out


@dataclass
class SramFit:
    """area = a*bits + b*width + c*depth + d (um^2)."""

    a: float
    b: float
    c: float
    d: float

    def area(self, bits: int, width: int) -> float:
        depth = bits / width
        return self.a * bits + self.b * width + self.c * depth + self.d


def fit_sram(pts=None) -> tuple[SramFit, list]:
    """Least squares on relative error (each point weighted by 1/area)."""
    pts = pts or load_table()
    X, y = [], []
    for kbit, w, apb in pts:
        bits = kbit * 1024
        X.append([bits, w, bits / w, 1.0])
        y.append(apb * bits)
    X, y = np.array(X), np.array(y)
    wgt = 1.0 / y
    coef, *_ = np.linalg.lstsq(X * wgt[:, None], y * wgt, rcond=None)
    fit = SramFit(*coef)
    errs = [
        (kbit, w, apb * kbit * 1024, fit.area(kbit * 1024, w))
        for kbit, w, apb in pts
    ]
    return fit, errs


@dataclass
class Macro:
    """One SRAM macro shape: `depth` words of `width` bits."""

    depth: int
    width: int

    @property
    def bits(self) -> int:
        return self.depth * self.width

    def legal(self) -> bool:
        return MIN_W <= self.width <= MAX_W and MIN_BITS <= self.bits <= MAX_BITS


def array_macros(bits: int, width: int) -> tuple[int, Macro]:
    """Macros of `width` bits that hold `bits`: (count, shape). The depth is a
    power of two; bigger than 64 Kbit splits into several macros."""
    if not MIN_W <= width <= MAX_W:
        raise ValueError(f"macro width {width} outside {MIN_W}..{MAX_W}")
    max_depth = MAX_BITS // width
    depth = 1 << max(0, math.ceil(math.log2(max(1, math.ceil(bits / width)))))
    depth = max(depth, MIN_BITS // width)
    n = max(1, math.ceil(depth / max_depth))
    return n, Macro(min(depth, max_depth), width)


def xbar_area(n_in: int, n_out: int, w: int, radix: int | None, logic: dict,
              pipelined: bool = False) -> float:
    """Crossbar of `w`-bit ports. radix None (or >= N): full crossbar."""
    k_xp, k_reg = logic["k_xp"], logic["k_reg"]
    n = max(n_in, n_out)
    if radix is None or radix >= n:
        area, stages = k_xp * n_in * n_out * w, 1
    else:
        stages = math.ceil(math.log(n, radix) - 1e-9)
        area = k_xp * stages * math.ceil(n / radix) * radix * radix * w
    if pipelined:
        area += k_reg * n * w * stages
    return area


def rotator_area(lanes: int, w: int, logic: dict) -> float:
    """Log-depth barrel rotator / XOR swizzle on a port of `lanes` x `w` bits:
    log2(lanes) stages of one MUX2 per bit."""
    return logic["k_xp"] * lanes * w * math.ceil(math.log2(max(2, lanes)))


def arbiter_area(requesters: int, banks: int, logic: dict) -> float:
    return logic["k_arb"] * requesters * banks


if __name__ == "__main__":
    fit, errs = fit_sram()
    print(f"fit: area = {fit.a:.4f}*bits + {fit.b:.3f}*width + {fit.c:.3f}*depth + {fit.d:.1f}  (um^2)")
    worst = 0.0
    print(f"{'Kbit':>6} {'w':>4} {'table':>9} {'fit':>9} {'err%':>7}")
    for kbit, w, t, f in errs:
        e = 100 * (f - t) / t
        worst = max(worst, abs(e))
        print(f"{kbit:6g} {w:4d} {t:9.1f} {f:9.1f} {e:7.1f}")
    print(f"max |err| = {worst:.1f}%  ({'OK' if worst < 10 else 'FAILS'} < 10%)")
