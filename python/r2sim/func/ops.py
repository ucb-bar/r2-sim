"""One function per task kind: the input is the descriptor (the fields of
`descriptor.decode`) and the plane's state; each function changes the
state. The rules of each task (the destination checks, the rows of a 2D
copy, the halves of split operands, the Atlas accumulator) are here; the
arithmetic and the memory access are the plane's provider's
(`provider_numpy`, `provider_torch`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from r2sim.func.memory import Dst, FuncError

if TYPE_CHECKING:
    from r2sim.func.plane import Plane

# The element types of the providers: bf16 storage, fp32 arithmetic.
DTYPES = {"bf16": 2, "f32": 4}


def held(desc: dict) -> Dst:
    """The task's masked destination range."""
    base, bound = desc["dst"]
    return Dst(base, bound, desc["mask"])


def size_of(name: str) -> int:
    """The bytes of an element of type `name`."""
    if name not in DTYPES:
        raise FuncError(f"numeric matmul on {name} operands")
    return DTYPES[name]


def dma(plane: Plane, desc: dict) -> None:
    """Copies `bytes` from `src` to `to`, or `rows` rows of `row_bytes`: row
    r from `src + r * src_stride` to `to + r * dst_stride`. Each written row
    must be inside the task's masked destination. A row from a
    null-fidelity block copies nothing."""
    src, to = desc["src"], desc["to"]
    rows = desc.get("rows", 1)
    n = desc["row_bytes"] if rows > 1 else desc["bytes"]
    for r in range(rows):
        s, d = src + r * desc.get("src_stride", 0), to + r * desc.get("dst_stride", 0)
        check_in_dst(plane, desc, d, n)
        if plane.memory.is_null(s, n):
            continue
        plane.provider.copy(plane.memory, s, d, n)


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


def read_matrix(plane: Plane, base: int, rows: int, cols: int, ld: int, dt: str):
    """A row-major rows x cols matrix of `dt` with leading dimension ld, as
    fp32 (an array of the provider)."""
    size_of(dt)
    return plane.provider.read(plane.memory, base, rows, cols, ld, dt)


def matmul_operands(plane: Plane, desc: dict):
    """A (m x k) and B (k x n) in fp32, halves of split operands joined as
    the Rust builtin backend joins them."""
    m, n, k = desc["m"], desc["n"], desc["k"]
    at, bt = desc["a_dtype"], desc["b_dtype"]
    tile = desc["tile"]
    p = plane.provider
    if desc["a2"] is None:
        a = read_matrix(plane, desc["a"], m, k, desc["lda"], at)
    else:
        es = size_of(at)
        rows = []
        for r in range(m):
            hi, i = split_row(r, tile[0])
            base = desc["a2"] if hi else desc["a"]
            rows.append(read_matrix(plane, base + i * desc["lda"] * es, 1, k, desc["lda"], at))
        a = p.concat(rows, 0)
    if desc.get("transpose_b"):
        if desc["b2"] is not None:
            raise FuncError("a matmul with transpose_b and b2")
        # B stored as n x k: its transpose.
        b = read_matrix(plane, desc["b"], n, k, desc["ldb"], bt).T
    elif desc["b2"] is None:
        b = read_matrix(plane, desc["b"], k, n, desc["ldb"], bt)
    else:
        es = size_of(bt)
        cols = []
        for c in range(n):
            hi, i = split_row(c, tile[1])
            base = (desc["b2"] if hi else desc["b"]) + i * es
            cols.append(read_matrix(plane, base, k, 1, desc["ldb"], bt))
        b = p.concat(cols, 1)
    return a, b


def matmul(plane: Plane, desc: dict) -> None:
    """C (+)= A B in fp32, output stationary: the partial C of a chain stays
    in its Atlas's accumulator (fp32, keyed by the C base) while the tasks
    keep it; a task that accumulates starts from the held partial, or else
    from C in memory in its output type; a task without keep writes C in its
    output type into memory and drops the partial."""
    m, n = desc["m"], desc["n"]
    ot = desc["out_dtype"]
    es = size_of(ot)
    c, ldc = desc["dst"][0], desc["ldc"]
    for r in range(m):
        check_in_dst(plane, desc, c + r * ldc * es, n * es)
    a, b = matmul_operands(plane, desc)
    acc = plane.accumulator(desc["pe"])
    partial = acc.pop(c, None)
    if not desc["accumulate"]:
        start = plane.provider.zeros(m, n)
    elif partial is not None:
        start = partial
    else:
        start = read_matrix(plane, c, m, n, ldc, ot)
    result = start + a @ b
    if desc["keep"]:
        acc[c] = result
    else:
        plane.provider.write(plane.memory, c, ldc, ot, result)


def vector(plane: Plane, desc: dict) -> None:
    """A Muon task: the opaque kernel 0 writes nothing; a kernel of the
    registry runs on the plane (`kernels`)."""
    if desc.get("kernel", 0) != 0:
        plane.kernels.run(plane, desc)


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
