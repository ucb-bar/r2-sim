"""One function per op kind: the input is the descriptor (as the protocol
sends it) and the plane's state; each function changes the state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ml_dtypes
import numpy as np

from r2sim.func.memory import Dst, FuncError

if TYPE_CHECKING:
    from r2sim.func.plane import Plane

DTYPES = {"bf16": ml_dtypes.bfloat16, "f32": np.float32}


def held(desc: dict) -> Dst:
    """The task's masked destination range."""
    base, bound = desc["dst"]
    return Dst(base, bound, desc["mask"])


def dtype_of(name: str) -> type:
    if name not in DTYPES:
        raise FuncError(f"numeric matmul on {name} operands")
    return DTYPES[name]


def dma(plane: Plane, desc: dict) -> None:
    """Copies `bytes` from `src` to `to`; the bytes written must be inside
    the task's masked destination. A copy from a null-fidelity block copies
    nothing."""
    src, to, n = desc["src"], desc["to"], desc["bytes"]
    check_in_dst(plane, desc, to, n)
    if plane.memory.is_null(src, n):
        return
    plane.memory.write(to, plane.memory.read(src, n))


def check_in_dst(plane: Plane, desc: dict, addr: int, length: int) -> None:
    d = held(desc)
    if not d.covers_bytes(plane.memory.granule, addr, length):
        mask = "" if d.mask == 0xFFFFFFFF else f" with mask {d.mask:#x}"
        raise FuncError(
            f"task writes [{addr:#x}, {addr + length:#x}) outside its dst "
            f"[{d.base:#x}, {d.bound:#x}){mask}"
        )


def split_row(r: int, t: int) -> tuple[bool, int]:
    """Where row (or column) r of a split operand with tile t is: (True, i)
    for row i of the high half matrix, (False, i) for the low half."""
    h = t // 2
    ti, j = divmod(r, t)
    return (False, ti * h + j) if j < h else (True, ti * h + j - h)


def read_matrix(plane: Plane, base: int, rows: int, cols: int, ld: int, dt: type) -> np.ndarray:
    """A row-major rows x cols matrix with leading dimension ld, as fp32."""
    es = np.dtype(dt).itemsize
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)
    span = ((rows - 1) * ld + cols) * es
    flat = np.frombuffer(plane.memory.read(base, span), dtype=dt)
    rows_view = np.lib.stride_tricks.as_strided(
        flat, shape=(rows, cols), strides=(ld * es, es), writeable=False
    )
    return rows_view.astype(np.float32)


def write_matrix(plane: Plane, base: int, ld: int, dt: type, v: np.ndarray) -> None:
    es = np.dtype(dt).itemsize
    for r in range(v.shape[0]):
        plane.memory.write(base + r * ld * es, v[r].astype(dt).tobytes())


def matmul_operands(plane: Plane, desc: dict) -> tuple[np.ndarray, np.ndarray]:
    """A (m x k) and B (k x n) in fp32, halves of split operands joined as
    the Rust Numeric backend joins them."""
    m, n, k = desc["m"], desc["n"], desc["k"]
    at, bt = dtype_of(desc["a_dtype"]), dtype_of(desc["b_dtype"])
    tile = desc["tile"]
    if desc["a2"] is None:
        a = read_matrix(plane, desc["a"], m, k, desc["lda"], at)
    else:
        es = np.dtype(at).itemsize
        a = np.empty((m, k), dtype=np.float32)
        for r in range(m):
            hi, i = split_row(r, tile[0])
            base = desc["a2"] if hi else desc["a"]
            a[r] = read_matrix(plane, base + i * desc["lda"] * es, 1, k, desc["lda"], at)[0]
    if desc["b2"] is None:
        b = read_matrix(plane, desc["b"], k, n, desc["ldb"], bt)
    else:
        es = np.dtype(bt).itemsize
        b = np.empty((k, n), dtype=np.float32)
        for c in range(n):
            hi, i = split_row(c, tile[1])
            base = (desc["b2"] if hi else desc["b"]) + i * es
            b[:, c] = read_matrix(plane, base, k, 1, desc["ldb"], bt)[:, 0]
    return a, b


def matmul(plane: Plane, desc: dict) -> None:
    """C (+)= A B in fp32, output stationary: the partial C of a chain stays
    in its Atlas's accumulator (fp32, keyed by the C base) while the tasks
    keep it; a task that accumulates starts from the held partial, or else
    from C in memory in its output type; a task without keep writes C in its
    output type into memory and drops the partial."""
    m, n = desc["m"], desc["n"]
    ot = dtype_of(desc["out_dtype"])
    es = np.dtype(ot).itemsize
    c, ldc = desc["dst"][0], desc["ldc"]
    for r in range(m):
        check_in_dst(plane, desc, c + r * ldc * es, n * es)
    a, b = matmul_operands(plane, desc)
    acc = plane.accumulator(desc["pe"])
    partial = acc.pop(c, None)
    if not desc["accumulate"]:
        start = np.zeros((m, n), dtype=np.float32)
    elif partial is not None:
        start = partial
    else:
        start = read_matrix(plane, c, m, n, ldc, ot)
    result = (start + a @ b).astype(np.float32)
    if desc["keep"]:
        acc[c] = result
    else:
        write_matrix(plane, c, ldc, ot, result)


def vector(plane: Plane, desc: dict) -> None:
    """A Muon task: no effect yet (the descriptor does not name the kernel)."""


def nothing(plane: Plane, desc: dict) -> None:
    """A join, a spawn, or an IMEM_COPY (its bytes are in the control
    plane): no data."""


TASKS = {
    "dma": dma,
    "matmul": matmul,
    "vector": vector,
    "join": nothing,
    "spawn": nothing,
    "imem_copy": nothing,
}


def task(plane: Plane, desc: dict) -> None:
    """Applies one task descriptor."""
    fn = TASKS.get(desc["kind"])
    if fn is None:
        raise FuncError(f"unknown task kind {desc['kind']!r}")
    fn(plane, desc)


def store(plane: Plane, addr: int, data: bytes) -> None:
    """A Neutrino store: writes its bytes."""
    plane.memory.write(addr, data)
