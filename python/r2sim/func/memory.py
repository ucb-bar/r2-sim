"""Sparse byte memory of the functional plane.

It behaves as the Rust memory of the Numeric and Trace backends: 64 KiB pages
per memory block, allocated on the first write and zero where never written;
an access lies inside one block; writes into the sync-only space have no
effect; a block of null fidelity drops writes and rejects reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PAGE = 1 << 16


class FuncError(Exception):
    """An error of the functional plane; it stops the run."""


@dataclass
class Block:
    """One memory block: a global byte range and its pages."""

    base: int
    bound: int
    numeric: bool
    pages: dict[int, bytearray] = field(default_factory=dict)

    def covers(self, addr: int, length: int) -> bool:
        return self.base <= addr and addr + length <= self.bound


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


class Memory:
    """The memory blocks of a run, with the sync-only ranges."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.sync_only: list[tuple[int, int]] = []
        self.granule = 1

    def configure(self, memories: list[dict], sync_only: list[list[int]], granule: int) -> None:
        self.blocks = []
        for m in memories:
            base, bound = m["range"]
            if not any(b.base == base and b.bound == bound for b in self.blocks):
                self.blocks.append(Block(base, bound, bool(m["numeric"])))
        self.sync_only = [(a, b) for a, b in sync_only]
        self.granule = max(int(granule), 1)

    def block(self, addr: int, length: int) -> Block:
        for b in self.blocks:
            if b.covers(addr, length):
                return b
        raise FuncError(f"[{addr:#x}, {addr + length:#x}) is not inside one memory block")

    def is_null(self, addr: int, length: int) -> bool:
        """True when the range lies in a block of null fidelity."""
        return any(not b.numeric and b.covers(addr, length) for b in self.blocks)

    def _is_sync_only(self, addr: int, length: int) -> bool:
        return any(addr < hi and lo < addr + length for lo, hi in self.sync_only)

    def read(self, addr: int, length: int) -> bytes:
        b = self.block(addr, length)
        if not b.numeric:
            raise FuncError(f"read of [{addr:#x}, +{length}) in a null-fidelity block")
        out = bytearray(length)
        off = 0
        while off < length:
            page, in_page = divmod(addr + off, PAGE)
            n = min(PAGE - in_page, length - off)
            p = b.pages.get(page)
            if p is not None:
                out[off:off + n] = p[in_page:in_page + n]
            off += n
        return bytes(out)

    def write(self, addr: int, data: bytes) -> None:
        length = len(data)
        if self._is_sync_only(addr, length):
            return
        b = self.block(addr, length)
        if not b.numeric:
            return
        off = 0
        while off < length:
            page, in_page = divmod(addr + off, PAGE)
            n = min(PAGE - in_page, length - off)
            p = b.pages.setdefault(page, bytearray(PAGE))
            p[in_page:in_page + n] = data[off:off + n]
            off += n
