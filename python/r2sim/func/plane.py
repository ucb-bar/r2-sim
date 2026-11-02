"""The state of the functional plane: memory, the Atlas accumulators and
the Muon kernels."""

from __future__ import annotations

import numpy as np

from r2sim.func import ops
from r2sim.func.kernels import Registry
from r2sim.func.memory import Memory


class Plane:
    """Memory, per Atlas (by PE index) the fp32 partial tiles it holds,
    keyed by the C base, and the kernels of the registry."""

    def __init__(self, memory: Memory, kernels: Registry | None = None) -> None:
        self.memory = memory
        self.kernels = kernels or Registry([])
        self.accumulators: dict[int, dict[int, np.ndarray]] = {}

    def accumulator(self, pe: int | None) -> dict[int, np.ndarray]:
        return self.accumulators.setdefault(-1 if pe is None else pe, {})

    def task(self, desc: dict) -> None:
        ops.task(self, desc)
