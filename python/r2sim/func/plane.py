"""The state of the functional plane: memory and the Atlas accumulators."""

from __future__ import annotations

import numpy as np

from r2sim.func import ops
from r2sim.func.memory import Memory


class Plane:
    """Memory, and per Atlas (by PE index) the fp32 partial tiles it holds,
    keyed by the C base."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self.accumulators: dict[int, dict[int, np.ndarray]] = {}

    def accumulator(self, pe: int | None) -> dict[int, np.ndarray]:
        return self.accumulators.setdefault(-1 if pe is None else pe, {})

    def task(self, desc: dict) -> None:
        ops.task(self, desc)
