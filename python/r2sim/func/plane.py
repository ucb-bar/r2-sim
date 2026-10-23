"""The state of the functional plane: memory and the Atlas accumulators."""

from __future__ import annotations

import numpy as np

from r2sim.func import ops
from r2sim.func.memory import Memory


class Plane:
    """Memory, and per Atlas (by PE id) the fp32 partial tiles it holds,
    keyed by the C base."""

    def __init__(self) -> None:
        self.memory = Memory()
        self.accumulators: dict[int, dict[int, np.ndarray]] = {}

    def configure(self, memories: list[dict], sync_only: list[list[int]], granule: int) -> None:
        self.memory.configure(memories, sync_only, granule)
        self.accumulators = {}

    def accumulator(self, pe: int | None) -> dict[int, np.ndarray]:
        return self.accumulators.setdefault(-1 if pe is None else pe, {})

    def task(self, desc: dict) -> None:
        ops.task(self, desc)

    def store(self, addr: int, data: bytes) -> None:
        ops.store(self, addr, data)
