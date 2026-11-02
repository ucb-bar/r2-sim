"""The thread pools of the plug-in: the BLAS of NumPy (OpenBLAS in the
default environment, MKL in the environment torch), OpenMP and PyTorch.
The ops are tiles of 128 x 128, which a pool of many threads makes slower,
so a backend sets every pool to the number of threads of its entry
(`threads`, default 1).

The pools are set at run time through the libraries' own functions, found
in the libraries that the process has loaded: openblas_set_num_threads,
MKL_Set_Num_Threads, omp_set_num_threads and torch.set_num_threads. The
variables OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS of the
process do not change the result. The OpenMP setting holds for the calling
thread only, so the backend sets the pools on each thread that runs its
ops (configure and the tasks run on different threads)."""

from __future__ import annotations

import ctypes
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from r2sim.func.memory import FuncError

# (name, the library's file name, getter, setter).
POOLS = [
    ("openblas", re.compile(r"libopenblas[^/]*\.so"), "openblas_get_num_threads",
     "openblas_set_num_threads"),
    ("mkl", re.compile(r"libmkl_rt[^/]*\.so"), "MKL_Get_Max_Threads", "MKL_Set_Num_Threads"),
    ("openmp", re.compile(r"lib(gomp|omp|iomp5)[^/]*\.so"), "omp_get_max_threads",
     "omp_set_num_threads"),
]


def check(threads) -> int:
    """The value of the key `threads`: an integer >= 1."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise FuncError(f"threads: {threads!r} is not an integer >= 1")
    return threads


@dataclass
class Pool:
    """One pool of a loaded library: its getter and setter of the number
    of threads."""

    name: str
    path: str
    getter: ctypes._CFuncPtr
    setter: ctypes._CFuncPtr


def loaded() -> list[Pool]:
    """The pools of the libraries that the process has loaded."""
    fields = (line.split(maxsplit=5) for line in Path("/proc/self/maps").read_text().splitlines())
    paths = sorted({f[5] for f in fields if len(f) == 6 and f[5].startswith("/")})
    pools = []
    for name, pattern, get, set_ in POOLS:
        for path in paths:
            if not pattern.search(Path(path).name):
                continue
            lib = ctypes.CDLL(path)
            try:
                getter, setter = getattr(lib, get), getattr(lib, set_)
            except AttributeError:
                continue
            getter.restype, getter.argtypes = ctypes.c_int, []
            setter.restype, setter.argtypes = None, [ctypes.c_int]
            pools.append(Pool(name, path, getter, setter))
    return pools


def set_threads(threads: int) -> None:
    """Sets every pool of the calling thread to `threads` threads."""
    for pool in loaded():
        pool.setter(threads)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(threads)


def report() -> dict[str, int]:
    """The threads of each pool, as the calling thread sees them."""
    out = {pool.name: pool.getter() for pool in loaded()}
    torch = sys.modules.get("torch")
    if torch is not None:
        out["torch"] = torch.get_num_threads()
    return out
