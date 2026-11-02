"""The Muon kernels of the registry (crates/sim-core/kernels.yaml), shared
by the providers. r2-sim passes the registry in the configuration (key
`kernels`: per kernel id, name, bytes, flags, and the fields with name,
type and offset); a Muon task carries the kernel id and its argument
block, which `Kernel.decode` reads by the registry's fields. Each provider
has a function per kernel name (`KERNELS` of `provider_numpy` and
`provider_torch`), which gets a `Task` and the decoded fields.

The semantics are those of the registry: bf16 storage, fp32 arithmetic,
round to nearest even at each bf16 store. Addresses are global byte
addresses, `ld` values are in elements. Reads and writes go through the
memory of the plane, row by row with the leading dimension; a kernel
writes only inside its task's destination."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Callable

from r2sim.func.memory import FuncError
from r2sim.func.ops import check_in_dst, read_matrix, size_of

if TYPE_CHECKING:
    from r2sim.func.plane import Plane

FORMATS = {"u64": "<Q", "u32": "<I", "f32": "<f"}


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
    """The kernels of the configuration, by id, each with the function of
    the provider `provider` (name) in `functions` (by kernel name)."""

    def __init__(self, entries: list[dict], provider: str = "numpy",
                 functions: dict[str, Callable] | None = None) -> None:
        self.kernels = {e["id"]: Kernel(e) for e in entries}
        self.functions = functions or {}
        missing = [f"{k.name} (id {k.id})" for k in self.kernels.values()
                   if k.name not in self.functions]
        if missing:
            raise FuncError(f"provider {provider} has no implementation of the registry kernels "
                            f"{', '.join(missing)}")

    def run(self, plane: Plane, desc: dict) -> None:
        """Runs the Muon task `desc` (kernel id and block) on the plane."""
        kernel = self.kernels.get(desc["kernel"])
        if kernel is None:
            raise FuncError(f"unknown kernel {desc['kernel']}")
        args = kernel.decode(desc["block"])
        self.functions[kernel.name](Task(plane, desc, kernel), args)


class Task:
    """The memory access of one Muon task, by the names of its kernel; the
    arrays are those of the plane's provider."""

    def __init__(self, plane: Plane, desc: dict, kernel: Kernel) -> None:
        self.plane = plane
        self.desc = desc
        self.kernel = kernel

    def flag(self, name: str) -> int:
        """The bit of the kernel's flag `name`."""
        return self.kernel.flags[name]

    def need(self, args: dict, field: str, why: str = "") -> int:
        """The address `field`, which must not be 0."""
        if args[field] == 0:
            raise FuncError(f"kernel {self.kernel.name}: the address {field} is 0{why}")
        return args[field]

    def read(self, addr: int, rows: int, cols: int, ld: int, dt: str):
        """A [rows][cols] matrix of `dt` with leading dimension `ld`, as fp32."""
        return read_matrix(self.plane, addr, rows, cols, ld, dt)

    def write(self, addr: int, ld: int, dt: str, v) -> None:
        """Writes `v` as `dt` (bf16: rounded to nearest even) row by row;
        each row must be inside the task's destination."""
        es = size_of(dt)
        rows, cols = v.shape
        for r in range(rows):
            check_in_dst(self.plane, self.desc, addr + r * ld * es, cols * es)
        self.plane.provider.write(self.plane.memory, addr, ld, dt, v)
