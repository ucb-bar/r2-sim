"""The Muon kernels of the registry (crates/sim-core/kernels.yaml) on the
numpy provider. r2-sim passes the registry in the configuration (key
`kernels`: per kernel id, name, bytes, flags, and the fields with name,
type and offset); a Muon task carries the kernel id and its argument
block, which `Registry.decode` reads by the registry's fields.

The semantics are those of the registry: bf16 storage, fp32 arithmetic,
round to nearest even at each bf16 store. Addresses are global byte
addresses, `ld` values are in elements. Reads and writes go through the
memory of the plane, row by row with the leading dimension; a kernel
writes only inside its task's destination."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Callable

import ml_dtypes
import numpy as np

from r2sim.func.memory import FuncError
from r2sim.func.ops import check_in_dst, read_matrix

if TYPE_CHECKING:
    from r2sim.func.plane import Plane

BF16 = ml_dtypes.bfloat16
F32 = np.float32
FORMATS = {"u64": "<Q", "u32": "<I", "f32": "<f"}
SQRT_2_OVER_PI = F32(math.sqrt(2 / math.pi))
GELU_C = F32(0.044715)


class Kernel:
    """One kernel of the registry: its argument block's layout and flags."""

    def __init__(self, entry: dict) -> None:
        self.id = entry["id"]
        self.name = entry["name"]
        self.bytes = entry["bytes"]
        self.flags = dict(entry.get("flags", {}))
        self.fields = [(f["name"], FORMATS[f["type"]], f["offset"]) for f in entry["fields"]]

    def decode(self, block: bytes) -> dict:
        """The fields of the argument block `block`."""
        if len(block) != self.bytes:
            raise FuncError(f"kernel {self.name} (id {self.id}): an argument block of "
                            f"{len(block)} bytes, not {self.bytes}")
        return {name: struct.unpack_from(fmt, block, off)[0] for name, fmt, off in self.fields}


class Registry:
    """The kernels of the configuration, by id, each with its function."""

    def __init__(self, entries: list[dict]) -> None:
        self.kernels = {e["id"]: Kernel(e) for e in entries}
        missing = [f"{k.name} (id {k.id})" for k in self.kernels.values() if k.name not in KERNELS]
        if missing:
            raise FuncError(f"provider numpy has no implementation of the registry kernels "
                            f"{', '.join(missing)}")

    def run(self, plane: Plane, desc: dict) -> None:
        """Runs the Muon task `desc` (kernel id and block) on the plane."""
        kernel = self.kernels.get(desc["kernel"])
        if kernel is None:
            raise FuncError(f"unknown kernel {desc['kernel']}")
        args = kernel.decode(desc["block"])
        KERNELS[kernel.name](Task(plane, desc, kernel), args)


class Task:
    """The memory access of one Muon task, by the names of its kernel."""

    def __init__(self, plane: Plane, desc: dict, kernel: Kernel) -> None:
        self.plane = plane
        self.desc = desc
        self.kernel = kernel

    def need(self, args: dict, field: str, why: str = "") -> int:
        """The address `field`, which must not be 0."""
        if args[field] == 0:
            raise FuncError(f"kernel {self.kernel.name}: the address {field} is 0{why}")
        return args[field]

    def read(self, addr: int, rows: int, cols: int, ld: int, dt: type) -> np.ndarray:
        """A [rows][cols] matrix of `dt` with leading dimension `ld`, as fp32."""
        return read_matrix(self.plane, addr, rows, cols, ld, dt)

    def write(self, addr: int, ld: int, dt: type, v: np.ndarray) -> None:
        """Writes `v` as `dt` (bf16: rounded to nearest even) row by row;
        each row must be inside the task's destination."""
        es = np.dtype(dt).itemsize
        out = v.astype(dt)
        for r in range(v.shape[0]):
            at = addr + r * ld * es
            check_in_dst(self.plane, self.desc, at, v.shape[1] * es)
            self.plane.memory.write(at, out[r].view(np.uint8))


def row_stats(t: Task, a: dict) -> None:
    """LayerNorm statistics of one chunk: (mean, M2) per row, merged with
    the statistics of the earlier chunks (Chan et al.) when count > 0."""
    rows, cols, count = a["rows"], a["cols"], a["count"]
    src, stats = t.need(a, "src"), t.need(a, "stats")
    x = t.read(src, rows, cols, a["ld"], BF16)
    mean_c = x.mean(1, dtype=F32)
    m2_c = np.square(x - mean_c[:, None]).sum(1, dtype=F32)
    if count == 0:
        mean, m2 = mean_c, m2_c
    else:
        old = t.read(stats, rows, 2, 2, F32)
        mean, m2 = old[:, 0], old[:, 1]
        n = F32(count + cols)
        d = mean_c - mean
        mean = mean + d * F32(cols) / n
        m2 = m2 + m2_c + d * d * F32(count) * F32(cols) / n
    t.write(stats, 2, F32, np.stack([mean, m2], 1).astype(F32))


def epilogue(t: Task, a: dict) -> None:
    """One out tile of a GEMM: y = C, then NORM, BIAS, GELU_TANH and RES in
    this order, then out = bf16(y)."""
    flags = t.kernel.flags
    f = a["flags"]
    rows, cols = a["rows"], a["cols"]
    c_dt = F32 if f & flags["C_F32"] else BF16
    y = t.read(t.need(a, "c"), rows, cols, a["ld_c"], c_dt)
    if f & flags["NORM"]:
        st = t.read(t.need(a, "stats", " with NORM"), rows, 2, 2, F32)
        u = t.read(t.need(a, "u", " with NORM"), 1, cols, cols, F32)
        rstd = F32(1) / np.sqrt(st[:, 1:2] / F32(a["k"]) + F32(a["eps"]))
        y = rstd * (y - st[:, 0:1] * u)
    if f & flags["BIAS"]:
        y = y + t.read(t.need(a, "bias", " with BIAS"), 1, cols, cols, F32)
    if f & flags["GELU_TANH"]:
        y = F32(0.5) * y * (F32(1) + np.tanh(SQRT_2_OVER_PI * (y + GELU_C * y * y * y)))
    if f & flags["RES"]:
        y = y + t.read(t.need(a, "res", " with RES"), rows, cols, a["ld_res"], BF16)
    t.write(t.need(a, "out"), a["ld_out"], BF16, y.astype(F32))


def softmax_stats(t: Task, a: dict) -> None:
    """Per row over both halves: m = max(scale s), l = sum exp(scale s - m)."""
    rows, cols, ld, scale = a["rows"], a["cols"], a["ld"], F32(a["scale"])
    s = np.concatenate([t.read(t.need(a, "s0"), rows, cols, ld, BF16),
                        t.read(t.need(a, "s1"), rows, cols, ld, BF16)], 1) * scale
    m = s.max(1)
    l = np.exp(s - m[:, None]).sum(1, dtype=F32)
    t.write(t.need(a, "stats"), 2, F32, np.stack([m, l], 1).astype(F32))


def softmax_apply(t: Task, a: dict) -> None:
    """One half: p = exp(scale s - m) / l, dst = bf16(p) (dst may be src)."""
    rows, cols, ld, scale = a["rows"], a["cols"], a["ld"], F32(a["scale"])
    s = t.read(t.need(a, "src"), rows, cols, ld, BF16)
    st = t.read(t.need(a, "stats"), rows, 2, 2, F32)
    p = np.exp(s * scale - st[:, 0:1]) / st[:, 1:2]
    t.write(t.need(a, "dst"), ld, BF16, p.astype(F32))


KERNELS: dict[str, Callable[[Task, dict], None]] = {
    "row_stats": row_stats,
    "epilogue": epilogue,
    "softmax_stats": softmax_stats,
    "softmax_apply": softmax_apply,
}
