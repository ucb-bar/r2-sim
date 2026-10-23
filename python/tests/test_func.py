"""The Python functional plane, op by op, on descriptors without the
simulator; and the protocol loop on in-memory streams."""

import io

import ml_dtypes
import numpy as np
import pytest

from r2sim.func import protocol as p
from r2sim.func.__main__ import serve
from r2sim.func.memory import FuncError
from r2sim.func.ops import split_row
from r2sim.func.plane import Plane

BF16 = ml_dtypes.bfloat16
FULL = 0xFFFFFFFF
G = 512
MEM = 0x100000  # a numeric block [1 MiB, 2 MiB)
NULL = 0x200000  # a null-fidelity block [2 MiB, 3 MiB)
SYNC = (0x1F0000, 0x200000)  # the sync-only top of the numeric block


@pytest.fixture
def plane() -> Plane:
    pl = Plane()
    pl.configure(
        [
            {"name": "m", "range": [MEM, NULL], "kind": "sram", "numeric": True},
            {"name": "n", "range": [NULL, NULL + MEM], "kind": "sram", "numeric": False},
        ],
        [list(SYNC)],
        G,
    )
    return pl


def dma(src, to, n, dst=None, mask=FULL):
    dst = dst or [to, to + n]
    return {"kind": "dma", "pe": 0, "dst": dst, "mask": mask, "src": src, "to": to, "bytes": n}


def matmul(a, b, c, m, n, k, *, dt="bf16", out="bf16", accumulate=False, keep=False,
           a2=None, b2=None, tile=(128, 128), pe=0):
    es = 2 if out == "bf16" else 4
    return {
        "kind": "matmul", "pe": pe, "dst": [c, c + m * n * es], "mask": FULL,
        "a": a, "b": b, "a2": a2, "b2": b2, "lda": k, "ldb": n, "ldc": n,
        "m": m, "n": n, "k": k, "a_dtype": dt, "b_dtype": dt, "out_dtype": out,
        "accumulate": accumulate, "keep": keep, "tile": list(tile),
    }


def put(plane, addr, x):
    plane.memory.write(addr, np.ascontiguousarray(x).tobytes())


def get(plane, addr, shape, dt):
    n = int(np.prod(shape)) * np.dtype(dt).itemsize
    return np.frombuffer(plane.memory.read(addr, n), dtype=dt).reshape(shape)


def rand(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(BF16)


def test_memory_is_zero_where_never_written(plane):
    assert plane.memory.read(MEM + 0x12345, 100) == bytes(100)


def test_a_write_across_a_page_boundary_reads_back(plane):
    data = bytes(range(256)) * 4
    plane.memory.write(MEM + 0x10000 - 300, data)
    assert plane.memory.read(MEM + 0x10000 - 300, len(data)) == data


def test_sync_only_writes_are_ignored(plane):
    plane.memory.write(SYNC[0], b"\x01" * 8)
    assert plane.memory.read(SYNC[0], 8) == bytes(8)


def test_a_null_block_drops_writes_and_rejects_reads(plane):
    plane.memory.write(NULL, b"\x01")
    with pytest.raises(FuncError, match="null-fidelity"):
        plane.memory.read(NULL, 1)


def test_an_access_across_two_blocks_is_an_error(plane):
    with pytest.raises(FuncError, match="not inside one memory block"):
        plane.memory.read(NULL - 4, 8)


def test_dma_copies_the_bytes(plane):
    plane.memory.write(MEM, b"abcdefgh" * 128)
    plane.task(dma(MEM, MEM + 0x8000, 1024))
    assert plane.memory.read(MEM + 0x8000, 1024) == b"abcdefgh" * 128


def test_dma_from_a_null_block_copies_nothing(plane):
    plane.memory.write(MEM + 0x8000, b"x" * 16)
    plane.task(dma(NULL, MEM + 0x8000, 16))
    assert plane.memory.read(MEM + 0x8000, 16) == b"x" * 16


def test_dma_must_write_inside_its_masked_dst(plane):
    # dst holds granules 0 and 2 of 4 (mask 0b10): granule 2 is inside,
    # granule 1 is in a gap.
    dst = [MEM + 0x10000, MEM + 0x10000 + 4 * G]
    plane.task(dma(MEM, dst[0] + 2 * G, G, dst, 0b10))
    with pytest.raises(FuncError, match="outside its dst .* with mask 0x2"):
        plane.task(dma(MEM, dst[0] + G + 8, 16, dst, 0b10))


def test_matmul_bf16(plane):
    a, b = rand((64, 32), 1), rand((32, 48), 2)
    put(plane, MEM, a)
    put(plane, MEM + 0x10000, b)
    plane.task(matmul(MEM, MEM + 0x10000, MEM + 0x20000, 64, 48, 32))
    want = (a.astype(np.float32) @ b.astype(np.float32)).astype(BF16)
    assert np.array_equal(get(plane, MEM + 0x20000, (64, 48), BF16), want)


def test_matmul_f32_accumulates_from_c_in_memory(plane):
    a = np.ones((4, 4), np.float32)
    b = np.full((4, 4), 2.0, np.float32)
    put(plane, MEM, a)
    put(plane, MEM + 0x1000, b)
    put(plane, MEM + 0x2000, np.full((4, 4), 1.5, np.float32))
    plane.task(matmul(MEM, MEM + 0x1000, MEM + 0x2000, 4, 4, 4, dt="f32", out="f32",
                      accumulate=True))
    assert np.array_equal(get(plane, MEM + 0x2000, (4, 4), np.float32), np.full((4, 4), 9.5))


def test_matmul_rejects_fp8(plane):
    with pytest.raises(FuncError, match="numeric matmul on fp8"):
        plane.task(matmul(MEM, MEM, MEM + 0x2000, 4, 4, 4, dt="fp8"))


def test_the_accumulator_keeps_the_partial_in_fp32(plane):
    # A chain of three tasks on one C: the first two keep C in the
    # accumulator (memory unchanged), the last writes C once in bf16, so C
    # is rounded to bf16 once, not after each task.
    m = n = k = 16
    c = MEM + 0x20000
    a, b = rand((m, k), 3), rand((k, n), 4)
    put(plane, MEM, a)
    put(plane, MEM + 0x10000, b)
    plane.task(matmul(MEM, MEM + 0x10000, c, m, n, k, keep=True))
    plane.task(matmul(MEM, MEM + 0x10000, c, m, n, k, accumulate=True, keep=True))
    assert plane.memory.read(c, m * n * 2) == bytes(m * n * 2)
    assert c in plane.accumulator(0)
    plane.task(matmul(MEM, MEM + 0x10000, c, m, n, k, accumulate=True))
    assert c not in plane.accumulator(0)
    p32 = a.astype(np.float32) @ b.astype(np.float32)
    want = (p32 + p32 + p32).astype(BF16)
    assert np.array_equal(get(plane, c, (m, n), BF16), want)


def test_the_accumulator_is_per_atlas(plane):
    # A partial kept on PE 0 is not seen by an accumulating task on PE 1:
    # PE 1 starts from C in memory (zero).
    m = n = k = 8
    c = MEM + 0x20000
    put(plane, MEM, np.ones((m, k), BF16))
    put(plane, MEM + 0x10000, np.ones((k, n), BF16))
    plane.task(matmul(MEM, MEM + 0x10000, c, m, n, k, keep=True, pe=0))
    plane.task(matmul(MEM, MEM + 0x10000, c, m, n, k, accumulate=True, pe=1))
    assert np.array_equal(get(plane, c, (m, n), BF16), np.full((m, n), 8, BF16))


def test_dma_into_a_gap_of_a_non_power_of_two_dst_is_rejected(plane):
    # The spec's warp tile: bound 127 granules, mask 0x7e. Granule 1 is not
    # held (1 & ~0x7e != 0) although every bit of 126 is in the mask but
    # bit 0.
    dst = [MEM, MEM + 127 * G]
    plane.task(dma(MEM + 0x80000, MEM + 2 * G, G, dst, 0x7E))
    with pytest.raises(FuncError, match="outside its dst"):
        plane.task(dma(MEM + 0x80000, MEM + G, G, dst, 0x7E))


def test_split_row_follows_the_tile_halves():
    # Tile 4: rows 0, 1 of each tile are in the low half, rows 2, 3 high.
    assert [split_row(r, 4) for r in range(8)] == [
        (False, 0), (False, 1), (True, 0), (True, 1),
        (False, 2), (False, 3), (True, 2), (True, 3),
    ]


def test_matmul_with_split_operands(plane):
    # A (8 x 4) split by rows with tile 4 into a (low halves) and a2 (high
    # halves); B (4 x 8) split by columns with tile 4 into b and b2.
    m = n = 8
    k = 4
    a, b = rand((m, k), 5), rand((k, n), 6)
    lo_rows = [r for r in range(m) if not split_row(r, 4)[0]]
    hi_rows = [r for r in range(m) if split_row(r, 4)[0]]
    lo_cols = [c for c in range(n) if not split_row(c, 4)[0]]
    hi_cols = [c for c in range(n) if split_row(c, 4)[0]]
    put(plane, MEM, a[lo_rows])
    put(plane, MEM + 0x1000, a[hi_rows])
    # The halves of B keep B's leading dimension n.
    bl = np.zeros((k, n), BF16)
    bh = np.zeros((k, n), BF16)
    bl[:, :4] = b[:, lo_cols]
    bh[:, :4] = b[:, hi_cols]
    put(plane, MEM + 0x2000, bl)
    put(plane, MEM + 0x3000, bh)
    d = matmul(MEM, MEM + 0x2000, MEM + 0x4000, m, n, k, a2=MEM + 0x1000, b2=MEM + 0x3000,
               tile=(4, 4))
    plane.task(d)
    want = (a.astype(np.float32) @ b.astype(np.float32)).astype(BF16)
    assert np.array_equal(get(plane, MEM + 0x4000, (m, n), BF16), want)


def test_matmul_must_write_inside_its_dst(plane):
    d = matmul(MEM, MEM, MEM + 0x2000, 4, 4, 4)
    d["dst"] = [MEM + 0x2000, MEM + 0x2000 + 16]
    with pytest.raises(FuncError, match="outside its dst"):
        plane.task(d)


def test_store_writes_its_bytes(plane):
    plane.store(MEM + 8, b"\x07" * 8)
    assert plane.memory.read(MEM + 8, 8) == b"\x07" * 8


def test_a_muon_task_has_no_effect(plane):
    plane.task({"kind": "vector", "pe": 0, "dst": [MEM, MEM + 64], "mask": FULL,
                "ops": 1, "bytes_in": 64, "bytes_out": 64, "src": None})
    assert plane.memory.read(MEM, 64) == bytes(64)


def frames(*fs) -> io.BytesIO:
    buf = io.BytesIO()
    for f in fs:
        p.write_frame(buf, *f)
    buf.seek(0)
    return buf


def answers(out: io.BytesIO) -> list:
    out.seek(0)
    got = []
    while (f := p.read_frame(out)) is not None:
        got.append(f)
    return got


def test_the_protocol_loop():
    config = {"memories": [{"name": "m", "range": [MEM, NULL], "kind": "sram", "numeric": True}],
              "sync_only": [], "granule": G}
    inp = frames(
        (p.HELLO, {"protocol": 1}),
        (p.CONFIGURE, config),
        (p.WRITE, {"addr": MEM}, b"\x01\x02\x03\x04"),
        (p.TASK, {"seq": 7, "desc": dma(MEM, MEM + 0x100, 4)}),
        (p.READ, {"addr": MEM + 0x100, "len": 4}),
        (p.TASK, {"seq": 8, "desc": dma(MEM, MEM + 0x100, 4, [MEM, MEM + 4])}),
        (p.FINISH, None),
    )
    out = io.BytesIO()
    assert serve(inp, out) == 0
    got = answers(out)
    assert got[0][0] == p.OK and got[0][1]["protocol"] == 1
    assert [g[0] for g in got[1:4]] == [p.OK, p.OK, p.OK]
    assert got[4] == (p.DATA, None, b"\x01\x02\x03\x04")
    assert got[5][0] == p.ERROR and got[5][1]["seq"] == 8
    assert "outside its dst" in got[5][1]["msg"]


def test_an_unexpected_exception_is_an_error_frame_and_the_loop_goes_on():
    # A matmul descriptor without its operand fields raises a KeyError,
    # not a FuncError: the answer is ERROR with its seq, and the next
    # frame is served.
    config = {"memories": [{"name": "m", "range": [MEM, NULL], "kind": "sram", "numeric": True}],
              "sync_only": [], "granule": G}
    inp = frames(
        (p.HELLO, {"protocol": 1}),
        (p.CONFIGURE, config),
        (p.TASK, {"seq": 5, "desc": {"kind": "matmul", "pe": 0, "dst": [MEM, MEM + 64],
                                     "mask": FULL}}),
        (p.READ, {"addr": MEM, "len": 4}),
    )
    out = io.BytesIO()
    assert serve(inp, out) == 0
    got = answers(out)
    assert got[2][0] == p.ERROR and got[2][1]["seq"] == 5
    assert got[2][1]["msg"].startswith("KeyError: ")
    assert got[3] == (p.DATA, None, bytes(4))


def test_the_loop_exits_at_end_of_file():
    out = io.BytesIO()
    assert serve(frames((p.HELLO, {"protocol": 1})), out) == 0
    assert len(answers(out)) == 1
