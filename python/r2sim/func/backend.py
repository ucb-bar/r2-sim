"""The Python backend of the functional plane: the module that the plug-in
libr2_python.so calls inside the r2-sim process. `create` gets the
configuration (the block table of the store, the sync-only space, the
granule, the PE names and the backend's entry of the configuration) and
returns a backend; r2-sim calls its `apply` once per task, in order.

Each block of the store becomes a NumPy array over the block's memory, so
the ops write r2-sim's memory directly. The ops of `ops` (DMA; matmul with
the accumulator per Atlas and C base) and the Muon kernels of `kernels`
(the registry in the key `kernels` of the backend's entry; the opaque
kernel 0 writes nothing) run on the provider of the entry: `provider:
numpy` (the default), or `provider: torch` with `device: cpu` (the
default) or `cuda`. `threads` (default 1) is the number of threads of the
plug-in's thread pools (`threads`)."""

from __future__ import annotations

import json
import threading

from r2sim.func import ops, threads
from r2sim.func.descriptor import decode
from r2sim.func.memory import Block, FuncError, Memory, over
from r2sim.func.plane import Plane

NO_PE = 0xFFFFFFFF
DEFAULT_TILE = (128, 128)


class Backend:
    """One backend of a run."""

    def __init__(self, memory: Memory, pes: list[str], config: dict) -> None:
        if "kernels" not in config:
            raise FuncError("the configuration has no kernels (the kernel registry)")
        self.thread_count = threads.check(config.get("threads", 1))
        self.plane = Plane(memory, config["kernels"], config.get("provider", "numpy"),
                           config.get("device", "cpu"))
        self.pes = pes
        self.config = config
        # The output tile of each PE's Atlas (the halves of split operands).
        self.tiles = [list(t) for t in config.get("tiles", [])]
        # The pools of the thread that sets them (configure, then the
        # thread of the tasks).
        self.pools_of = None
        self.set_pools()

    def set_pools(self) -> None:
        """Sets the thread pools for the calling thread, once per thread."""
        me = threading.get_ident()
        if self.pools_of != me:
            threads.set_threads(self.thread_count)
            self.pools_of = me

    def pools(self) -> dict[str, int]:
        """The threads of each pool, as the calling thread sees them."""
        return threads.report()

    def apply(self, seq: int, cycle: int, gid: int, neutrino: int, pe: int, desc: bytes,
              block: bytes = b"") -> None:
        """One task: its descriptor bytes, its PE (an index of the PE table;
        NO_PE for none), and the argument block of a Muon task."""
        self.set_pools()
        d = decode(desc, self.plane.memory.granule, None if pe == NO_PE else pe)
        d["block"] = block
        d["tile"] = self.tiles[pe] if pe < len(self.tiles) else list(DEFAULT_TILE)
        ops.task(self.plane, d)


def create(text: str) -> Backend:
    """The backend of the configuration `text` (JSON): `blocks` (base, size,
    ptr, name), `sync_only`,
    `granule`, `pes` and `config`."""
    c = json.loads(text)
    blocks = [Block(b["base"], b["size"], over(b["ptr"], b["size"]), b["name"])
              for b in c["blocks"]]
    memory = Memory(blocks, [tuple(r) for r in c["sync_only"]], c["granule"])
    return Backend(memory, c["pes"], c["config"])
