"""The memory of the functional plane, as a backend plug-in sees it: the
blocks of r2-sim's store, each a NumPy array of bytes over the block's
memory (no copy), sorted by base. An access lies inside one block; a write
into the sync-only space has no effect; a block of a null subtree has no
array: a read from it is an error and a write to it has no effect."""

from __future__ import annotations

import bisect
import ctypes
from dataclasses import dataclass

import numpy as np


class FuncError(Exception):
    """An error of the functional plane; it stops the run."""


@dataclass
class Dst:
    """A destination range: a byte span aligned to the granule, and the mask
    of the granules it holds (granule base + i belongs when i & ~mask == 0;
    all ones holds the whole span)."""

    base: int
    bound: int
    mask: int

    def holds(self, granule: int, addr: int) -> bool:
        if not self.base <= addr < self.bound:
            return False
        return ((addr - self.base) // granule) & ~self.mask == 0

    def covers_bytes(self, granule: int, addr: int, length: int) -> bool:
        """True when every granule that [addr, addr + length) touches is held."""
        if length == 0:
            return True
        if not (self.base <= addr and addr + length <= self.bound):
            return False
        n = (self.bound - self.base) // granule
        # Whole: every bit up to the highest set bit of n - 1 is in the mask.
        if n <= 1 or ((1 << (n - 1).bit_length()) - 1) & ~self.mask == 0:
            return True
        first = addr // granule * granule
        return all(self.holds(granule, a) for a in range(first, addr + length, granule))


@dataclass
class Block:
    """One memory block: its global base, its size, its bytes (None for a
    block of a null subtree) and its name."""

    base: int
    size: int
    data: np.ndarray | None
    name: str


def over(ptr: int, size: int) -> np.ndarray:
    """A uint8 array over `size` bytes at address `ptr` of this process."""
    return np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(ptr))


class Memory:
    """The blocks of a run, with the sync-only ranges and the granule."""

    def __init__(self, blocks: list[Block], sync_only: list[tuple[int, int]], granule: int):
        self.blocks = sorted(blocks, key=lambda b: b.base)
        self.bases = [b.base for b in self.blocks]
        self.sync_only = list(sync_only)
        self.granule = max(int(granule), 1)

    @classmethod
    def owned(cls, blocks: list[tuple[int, int, bool]], sync_only: list[tuple[int, int]],
              granule: int) -> Memory:
        """A memory with arrays of its own (for tests): (base, size, data)
        per block, data False for a block of a null subtree."""
        return cls([Block(base, size, np.zeros(size, np.uint8) if data else None, f"b{i}")
                    for i, (base, size, data) in enumerate(blocks)], sync_only, granule)

    def block(self, addr: int, length: int) -> Block:
        i = bisect.bisect_right(self.bases, addr)
        if i > 0:
            b = self.blocks[i - 1]
            off = addr - b.base
            if off <= b.size and length <= b.size - off:
                return b
        raise FuncError(f"[{addr:#x}, {addr + length:#x}) is not inside one memory block")

    def is_null(self, addr: int, length: int) -> bool:
        """True when the range lies in a block of a null subtree."""
        try:
            return self.block(addr, length).data is None
        except FuncError:
            return False

    def _is_sync_only(self, addr: int, length: int) -> bool:
        return any(addr < hi and lo < addr + length for lo, hi in self.sync_only)

    def view(self, addr: int, length: int) -> np.ndarray:
        """The bytes [addr, addr + length), as a view of the block."""
        b = self.block(addr, length)
        if b.data is None:
            raise FuncError(f"read of [{addr:#x}, +{length}) in a null-fidelity block")
        off = addr - b.base
        return b.data[off:off + length]

    def read(self, addr: int, length: int) -> bytes:
        return self.view(addr, length).tobytes()

    def target(self, addr: int, length: int) -> np.ndarray | None:
        """The bytes [addr, addr + length) to write, as a view of the block;
        None where a write has no effect (the sync-only space, a block of
        a null subtree)."""
        if self._is_sync_only(addr, length):
            return None
        b = self.block(addr, length)
        if b.data is None:
            return None
        off = addr - b.base
        return b.data[off:off + length]

    def write(self, addr: int, data) -> None:
        """Writes `data` (bytes or a uint8 array) at `addr`."""
        src = np.frombuffer(data, np.uint8) if isinstance(data, (bytes, bytearray)) else data
        dst = self.target(addr, len(src))
        if dst is not None:
            dst[:] = src
