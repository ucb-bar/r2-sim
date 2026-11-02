"""The provider `torch`: the memory access and the arithmetic of the ops and
the Muon kernels in PyTorch, on the device `cpu` or `cuda`. A matrix is a
float32 tensor on the device. A read is a tensor over the block's memory
(torch.from_numpy of the memory's view: no copy), converted to fp32 on the
device; a store converts on the device (bf16: round to nearest even) and
copies the rows back into the block. A DMA copies bytes on the CPU.

PyTorch is optional: the plug-in finds it only when it was built in the
pixi environment `torch` (`pixi run -e torch build-python`)."""

from __future__ import annotations

import ctypes
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from r2sim.func.memory import FuncError

if TYPE_CHECKING:
    from r2sim.func.kernels import Task
    from r2sim.func.memory import Memory


def f32(x: float) -> float:
    """The fp32 value of the scalar `x` (a tensor op on fp32 tensors
    computes with the scalar in fp32)."""
    return float(np.float32(x))


SQRT_2_OVER_PI = f32(math.sqrt(2 / math.pi))
GELU_C = f32(0.044715)


# A shared library that the dynamic loader did not find.
MISSING_LIBRARY = re.compile(r"([\w.+-]+\.so[\w.]*): cannot open shared object file")


def import_torch():
    """The torch module, or an error that names the environment to build
    the plug-in in.

    PyTorch's libraries find their dependencies in the environment's
    `lib` (MKL, for example) through the RPATH of the `python` executable.
    In r2-sim the executable is r2sim, so a library that the loader does
    not find is loaded from the environment's `lib` (global symbols, so
    its soname satisfies later lookups) and the import runs again."""
    loaded = set()
    while True:
        try:
            import torch
        except ImportError as e:
            raise FuncError(
                f"provider torch: PyTorch is not installed in the Python of this plug-in "
                f"({sys.prefix}): {e}. Build the plug-in in the pixi environment torch: "
                f"pixi run -e torch build-python") from None
        except OSError as e:
            m = MISSING_LIBRARY.search(str(e))
            path = Path(sys.prefix) / "lib" / m.group(1) if m else None
            if path is None or path in loaded or not path.exists():
                raise FuncError(f"provider torch: import torch: {e}") from None
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            loaded.add(path)
        else:
            return torch


class Provider:
    """The memory access and the tensor operations of the shared ops."""

    name = "torch"

    def __init__(self, device: str = "cpu") -> None:
        torch = import_torch()
        if device not in ("cpu", "cuda"):
            raise FuncError(f"provider torch: device {device!r}: this version has cpu, cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise FuncError("provider torch: device cuda: no CUDA device")
        # fp32 matmuls in fp32, not TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.device = torch.device(device)
        self.dtypes = {"bf16": torch.bfloat16, "f32": torch.float32}

    def read(self, memory: Memory, base: int, rows: int, cols: int, ld: int, dt: str):
        """A rows x cols matrix with leading dimension ld, as fp32 on the
        device."""
        torch = self.torch
        t = self.dtypes[dt]
        es = t.itemsize
        if rows == 0 or cols == 0:
            return self.zeros(rows, cols)
        span = ((rows - 1) * ld + cols) * es
        flat = torch.from_numpy(memory.view(base, span)).view(t)
        m = flat.as_strided((rows, cols), (ld, 1))
        return m.to(device=self.device, dtype=torch.float32)

    def write(self, memory: Memory, base: int, ld: int, dt: str, v) -> None:
        """Writes the fp32 matrix `v` as `dt`, row by row at pitch ld."""
        t = self.dtypes[dt]
        out = v.to(t).cpu().contiguous()
        es = t.itemsize
        for r in range(out.shape[0]):
            memory.write(base + r * ld * es, out[r].view(self.torch.uint8).numpy())

    def copy(self, memory: Memory, src: int, dst: int, n: int) -> None:
        """Copies n bytes (the source first, so the ranges may overlap)."""
        target = memory.target(dst, n)
        if target is not None:
            data = self.torch.from_numpy(memory.view(src, n)).clone()
            self.torch.from_numpy(target).copy_(data)

    def zeros(self, rows: int, cols: int):
        return self.torch.zeros((rows, cols), dtype=self.torch.float32, device=self.device)

    def concat(self, parts: list, axis: int):
        return self.torch.cat(parts, axis)


def row_stats(t: Task, a: dict) -> None:
    """LayerNorm statistics of one chunk: (mean, M2) per row, merged with
    the statistics of the earlier chunks (Chan et al.) when count > 0."""
    import torch

    rows, cols, count = a["rows"], a["cols"], a["count"]
    src, stats = t.need(a, "src"), t.need(a, "stats")
    x = t.read(src, rows, cols, a["ld"], "bf16")
    mean_c = x.mean(1)
    m2_c = torch.square(x - mean_c[:, None]).sum(1)
    if count == 0:
        mean, m2 = mean_c, m2_c
    else:
        old = t.read(stats, rows, 2, 2, "f32")
        mean, m2 = old[:, 0], old[:, 1]
        n = f32(count + cols)
        d = mean_c - mean
        mean = mean + d * f32(cols) / n
        m2 = m2 + m2_c + d * d * f32(count) * f32(cols) / n
    t.write(stats, 2, "f32", torch.stack([mean, m2], 1))


def epilogue(t: Task, a: dict) -> None:
    """One out tile of a GEMM: y = C, then NORM, BIAS, GELU_TANH and RES in
    this order, then out = bf16(y)."""
    import torch

    f = a["flags"]
    rows, cols = a["rows"], a["cols"]
    c_dt = "f32" if f & t.flag("C_F32") else "bf16"
    y = t.read(t.need(a, "c"), rows, cols, a["ld_c"], c_dt)
    if f & t.flag("NORM"):
        st = t.read(t.need(a, "stats", " with NORM"), rows, 2, 2, "f32")
        u = t.read(t.need(a, "u", " with NORM"), 1, cols, cols, "f32")
        rstd = 1 / torch.sqrt(st[:, 1:2] / f32(a["k"]) + a["eps"])
        y = rstd * (y - st[:, 0:1] * u)
    if f & t.flag("BIAS"):
        y = y + t.read(t.need(a, "bias", " with BIAS"), 1, cols, cols, "f32")
    if f & t.flag("GELU_TANH"):
        y = 0.5 * y * (1 + torch.tanh(SQRT_2_OVER_PI * (y + GELU_C * y * y * y)))
    if f & t.flag("RES"):
        y = y + t.read(t.need(a, "res", " with RES"), rows, cols, a["ld_res"], "bf16")
    t.write(t.need(a, "out"), a["ld_out"], "bf16", y)


def softmax_stats(t: Task, a: dict) -> None:
    """Per row over both halves: m = max(scale s), l = sum exp(scale s - m)."""
    import torch

    rows, cols, ld, scale = a["rows"], a["cols"], a["ld"], a["scale"]
    s = torch.cat([t.read(t.need(a, "s0"), rows, cols, ld, "bf16"),
                   t.read(t.need(a, "s1"), rows, cols, ld, "bf16")], 1) * scale
    m = s.amax(1)
    l = torch.exp(s - m[:, None]).sum(1)
    t.write(t.need(a, "stats"), 2, "f32", torch.stack([m, l], 1))


def softmax_apply(t: Task, a: dict) -> None:
    """One half: p = exp(scale s - m) / l, dst = bf16(p) (dst may be src)."""
    import torch

    rows, cols, ld, scale = a["rows"], a["cols"], a["ld"], a["scale"]
    s = t.read(t.need(a, "src"), rows, cols, ld, "bf16")
    st = t.read(t.need(a, "stats"), rows, 2, 2, "f32")
    p = torch.exp(s * scale - st[:, 0:1]) / st[:, 1:2]
    t.write(t.need(a, "dst"), ld, "bf16", p)


KERNELS = {
    "row_stats": row_stats,
    "epilogue": epilogue,
    "softmax_stats": softmax_stats,
    "softmax_apply": softmax_apply,
}
