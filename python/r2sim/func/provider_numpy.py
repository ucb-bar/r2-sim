"""The provider `numpy`: the memory access and the arithmetic of the ops
and the Muon kernels in NumPy, on the CPU. A matrix is a float32 array;
a bf16 store rounds to nearest even (ml_dtypes)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import ml_dtypes
import numpy as np

from r2sim.func.kernels import opaque

if TYPE_CHECKING:
    from r2sim.func.kernels import Task
    from r2sim.func.memory import Memory

F32 = np.float32
DTYPES = {"bf16": ml_dtypes.bfloat16, "f32": np.float32}
SQRT_2_OVER_PI = F32(math.sqrt(2 / math.pi))
GELU_C = F32(0.044715)


class Provider:
    """The memory access and the array operations of the shared ops."""

    name = "numpy"

    def read(self, memory: Memory, base: int, rows: int, cols: int, ld: int,
             dt: str) -> np.ndarray:
        """A rows x cols matrix with leading dimension ld, as fp32."""
        t = DTYPES[dt]
        es = np.dtype(t).itemsize
        if rows == 0 or cols == 0:
            return np.zeros((rows, cols), dtype=F32)
        span = ((rows - 1) * ld + cols) * es
        flat = memory.view(base, span).view(t)
        rows_view = np.lib.stride_tricks.as_strided(
            flat, shape=(rows, cols), strides=(ld * es, es), writeable=False
        )
        return rows_view.astype(F32)

    def write(self, memory: Memory, base: int, ld: int, dt: str, v: np.ndarray) -> None:
        """Writes the fp32 matrix `v` as `dt`, row by row at pitch ld."""
        t = DTYPES[dt]
        es = np.dtype(t).itemsize
        out = v.astype(t)
        for r in range(v.shape[0]):
            memory.write(base + r * ld * es, out[r].view(np.uint8))

    def copy(self, memory: Memory, src: int, dst: int, n: int) -> None:
        """Copies n bytes (the source first, so the ranges may overlap)."""
        memory.write(dst, memory.view(src, n).copy())

    def zeros(self, rows: int, cols: int) -> np.ndarray:
        return np.zeros((rows, cols), dtype=F32)

    def concat(self, parts: list, axis: int) -> np.ndarray:
        return np.concatenate(parts, axis)


def row_stats(t: Task, a: dict) -> None:
    """LayerNorm statistics of one chunk: (mean, M2) per row, merged with
    the statistics of the earlier chunks (Chan et al.) when count > 0."""
    rows, cols, count = a["rows"], a["cols"], a["count"]
    src, stats = t.need(a, "src"), t.need(a, "stats")
    x = t.read(src, rows, cols, a["ld"], "bf16")
    mean_c = x.mean(1, dtype=F32)
    m2_c = np.square(x - mean_c[:, None]).sum(1, dtype=F32)
    if count == 0:
        mean, m2 = mean_c, m2_c
    else:
        old = t.read(stats, rows, 2, 2, "f32")
        mean, m2 = old[:, 0], old[:, 1]
        n = F32(count + cols)
        d = mean_c - mean
        mean = mean + d * F32(cols) / n
        m2 = m2 + m2_c + d * d * F32(count) * F32(cols) / n
    t.write(stats, 2, "f32", np.stack([mean, m2], 1).astype(F32))


def epilogue(t: Task, a: dict) -> None:
    """One out tile of a GEMM: y = C, then NORM, BIAS, GELU_TANH and RES in
    this order, then out = bf16(y)."""
    f = a["flags"]
    rows, cols = a["rows"], a["cols"]
    c_dt = "f32" if f & t.flag("C_F32") else "bf16"
    y = t.read(t.need(a, "c"), rows, cols, a["ld_c"], c_dt)
    if f & t.flag("NORM"):
        st = t.read(t.need(a, "stats", " with NORM"), rows, 2, 2, "f32")
        u = t.read(t.need(a, "u", " with NORM"), 1, cols, cols, "f32")
        rstd = F32(1) / np.sqrt(st[:, 1:2] / F32(a["k"]) + F32(a["eps"]))
        y = rstd * (y - st[:, 0:1] * u)
    if f & t.flag("BIAS"):
        y = y + t.read(t.need(a, "bias", " with BIAS"), 1, cols, cols, "f32")
    if f & t.flag("GELU_TANH"):
        y = F32(0.5) * y * (F32(1) + np.tanh(SQRT_2_OVER_PI * (y + GELU_C * y * y * y)))
    if f & t.flag("RES"):
        y = y + t.read(t.need(a, "res", " with RES"), rows, cols, a["ld_res"], "bf16")
    t.write(t.need(a, "out"), a["ld_out"], "bf16", y.astype(F32))


def softmax_stats(t: Task, a: dict) -> None:
    """Per row over both halves: m = max(scale s), l = sum exp(scale s - m)."""
    rows, cols, ld, scale = a["rows"], a["cols"], a["ld"], F32(a["scale"])
    s = np.concatenate([t.read(t.need(a, "s0"), rows, cols, ld, "bf16"),
                        t.read(t.need(a, "s1"), rows, cols, ld, "bf16")], 1) * scale
    m = s.max(1)
    l = np.exp(s - m[:, None]).sum(1, dtype=F32)
    t.write(t.need(a, "stats"), 2, "f32", np.stack([m, l], 1).astype(F32))


def softmax_apply(t: Task, a: dict) -> None:
    """One half: p = exp(scale s - m) / l, dst = bf16(p) (dst may be src)."""
    rows, cols, ld, scale = a["rows"], a["cols"], a["ld"], F32(a["scale"])
    s = t.read(t.need(a, "src"), rows, cols, ld, "bf16")
    st = t.read(t.need(a, "stats"), rows, 2, 2, "f32")
    p = np.exp(s * scale - st[:, 0:1]) / st[:, 1:2]
    t.write(t.need(a, "dst"), ld, "bf16", p.astype(F32))


KERNELS = {
    "opaque": opaque,
    "row_stats": row_stats,
    "epilogue": epilogue,
    "softmax_stats": softmax_stats,
    "softmax_apply": softmax_apply,
}
