"""The Python backend as the plug-in calls it: the descriptor decoder, and
`create` and `apply` over memory blocks given by address (NumPy arrays
here, r2-sim's store in a run)."""

import json
import struct

import ml_dtypes
import numpy as np
import pytest

from r2sim.func.backend import NO_PE, create
from r2sim.func.descriptor import decode
from r2sim.func.memory import FuncError

BF16 = ml_dtypes.bfloat16
G = 512
FULL = 0xFFFFFFFF
BASE = 0x100000


def header(opcode, dst, length, keys=0, mask=FULL, flags=1, pe=0):
    """A descriptor header: dst and length in bytes (multiples of G)."""
    return struct.pack("<HHHHQII", opcode, pe, flags, keys, dst // G, length // G, mask)


def dma_desc(src, to, n, rows=1, row_bytes=None, src_stride=0, dst_stride=0, dst_len=None):
    """A DMA (ABI version 2): one row of n bytes, or `rows` rows."""
    row_bytes = n if row_bytes is None else row_bytes
    return header(2, to, dst_len or max(G, n)) + struct.pack(
        "<QQQQIIII", src, to, n, n, rows, row_bytes, src_stride, dst_stride)


def vector_desc(dst, kernel, args_bytes=0, args=0):
    return header(4, dst, G) + struct.pack("<QQQQIIQ", 1, 64, 64, 0, kernel, args_bytes, args)


def matmul_desc(a, b, c, m, n, k, *, accumulate=False, keep=False, lda=None, ldb=None,
                transpose_b=False):
    flags = (1 if accumulate else 0) | (2 if keep else 0) | (4 if transpose_b else 0)
    args = struct.pack("<QQQQIIIIIIIIHH", a, b, 0, 0, lda or k, ldb or n, n, m, n, k,
                       0, flags, 128, 128) + bytes(4)
    return header(3, c, m * n * 2) + args


def test_the_decoder_gives_the_fields_of_the_ops():
    d = decode(dma_desc(BASE, BASE + 0x1000, 64), G, 3)
    assert d == {"kind": "dma", "pe": 3, "dst": [BASE + 0x1000, BASE + 0x1000 + G], "mask": FULL,
                 "src": BASE, "to": BASE + 0x1000, "bytes": 64, "tile_bytes": 64, "rows": 1,
                 "row_bytes": 64, "src_stride": 0, "dst_stride": 0}
    d2 = decode(dma_desc(BASE, BASE + 0x1000, 4 * 144, 4, 144, 256, 144), G, 0)
    assert (d2["rows"], d2["row_bytes"], d2["src_stride"], d2["dst_stride"]) == (4, 144, 256, 144)
    with pytest.raises(FuncError, match="a DMA of 3 rows of 144 bytes for 576 bytes"):
        decode(dma_desc(BASE, BASE, 576, 3, 144), G, 0)
    m = decode(matmul_desc(1, 2, BASE, 4, 8, 16, accumulate=True), G, None)
    assert (m["kind"], m["a"], m["b"], m["a2"], m["b2"]) == ("matmul", 1, 2, None, None)
    assert (m["lda"], m["ldb"], m["ldc"], m["m"], m["n"], m["k"]) == (16, 8, 8, 4, 8, 16)
    assert (m["a_dtype"], m["out_dtype"], m["accumulate"], m["keep"]) == ("bf16", "bf16", True, False)
    assert m["tile"] == [128, 128] and m["pe"] is None and not m["transpose_b"]
    assert decode(matmul_desc(1, 2, BASE, 4, 8, 16, transpose_b=True), G, 0)["transpose_b"]
    # Two keys between the header and the arguments.
    v = header(4, BASE, G, keys=2) + bytes(16) + struct.pack("<QQQQIIQ", 5, 6, 7, 0, 7, 24, 0x40)
    d = decode(v, G, 0)
    assert (d["ops"], d["src"], d["kernel"], d["args_bytes"], d["args"]) == (5, None, 7, 24, 0x40)
    with pytest.raises(FuncError, match="unknown opcode 9"):
        decode(header(9, BASE, G), G, 0)


class Run:
    """A backend over one 1 MiB block at BASE (a NumPy array)."""

    def __init__(self, provider="numpy"):
        self.mem = np.zeros(1 << 20, np.uint8)
        cfg = {"blocks": [{"base": BASE, "size": len(self.mem), "ptr": self.mem.ctypes.data,
                           "name": "root.mem"}],
               "sync_only": [], "granule": G, "pes": ["root.atlas"],
               "config": {"lib": "libr2_python.so", "provider": provider}}
        self.backend = create(json.dumps(cfg))
        self.seq = 0

    def apply(self, desc, pe=0, block=b""):
        self.seq += 1
        self.backend.apply(self.seq, 0, self.seq, 0, pe, desc, block)

    def put(self, addr, x):
        b = np.ascontiguousarray(x).view(np.uint8).ravel()
        self.mem[addr - BASE:addr - BASE + len(b)] = b

    def get(self, addr, shape, dt):
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        return self.mem[addr - BASE:addr - BASE + n].view(dt).reshape(shape)


def test_a_dma_writes_the_memory_of_the_block():
    r = Run()
    r.put(BASE, np.arange(64, dtype=np.uint8))
    r.apply(dma_desc(BASE, BASE + 0x1000, 64), pe=NO_PE)
    assert np.array_equal(r.mem[0x1000:0x1040], np.arange(64, dtype=np.uint8))


def test_a_chain_of_nine_matmuls_rounds_c_to_bf16_once():
    # C (128 x 128) = A (128 x 1152) B (1152 x 128) in bf16 as 9 tasks of
    # k = 128, the first without accumulate, all but the last with keep.
    # Integer data whose partial sums exceed 256: every fp32 sum is exact,
    # and a rounding of a partial to bf16 would change C.
    m = n = 128
    k, chunk = 1152, 128
    i, p, j = np.arange(m)[:, None], np.arange(k), np.arange(n)
    a = ((i * 7 + p[None, :] * 3) % 9 + 1).astype(np.float32)
    b = ((p[:, None] * 5 + j[None, :] * 11) % 9 + 1).astype(np.float32)
    pa, pb, pc = BASE, BASE + 0x50000, BASE + 0xA0000
    r = Run()
    r.put(pa, a.astype(BF16))
    r.put(pb, b.astype(BF16))
    for q in range(k // chunk):
        last = q == k // chunk - 1
        r.apply(matmul_desc(pa + q * chunk * 2, pb + q * chunk * n * 2, pc, m, n, chunk,
                            accumulate=q > 0, keep=not last, lda=k, ldb=n))
    want = (a @ b).astype(BF16)
    assert np.array_equal(r.get(pc, (m, n), BF16), want)
    # Rounding each partial to bf16 gives another C.
    each = np.zeros((m, n), np.float32)
    for q in range(k // chunk):
        s = slice(q * chunk, (q + 1) * chunk)
        each = (each + a[:, s] @ b[s, :]).astype(BF16).astype(np.float32)
    assert (each.astype(BF16) != want).sum() > 1000


def test_a_provider_other_than_numpy_is_an_error():
    with pytest.raises(FuncError, match="provider 'torch'"):
        Run("torch")


def test_a_2d_copy_moves_each_row_at_its_strides():
    # 128 rows of 144 B at a source pitch of 256 B into a contiguous
    # destination (pitch 144).
    r = Run()
    src = np.arange(128 * 256, dtype=np.uint32).astype(np.uint8)
    r.put(BASE, src)
    to = BASE + 0x10000
    r.apply(dma_desc(BASE, to, 128 * 144, 128, 144, 256, 144, dst_len=0x5000), pe=NO_PE)
    want = src.reshape(128, 256)[:, :144].ravel()
    assert np.array_equal(r.mem[0x10000:0x10000 + 128 * 144], want)


def test_a_matmul_with_b_transposed():
    # B is stored as n x k (k = 72); integer data: exact.
    m, n, k = 16, 32, 72
    i, p, j = np.arange(m)[:, None], np.arange(k), np.arange(n)
    a = ((i * 3 + p[None, :]) % 5).astype(np.float32)
    bt = ((j[:, None] * 7 + p[None, :] * 2) % 7).astype(np.float32)  # n x k
    pa, pb, pc = BASE, BASE + 0x10000, BASE + 0x20000
    r = Run()
    r.put(pa, a.astype(BF16))
    r.put(pb, bt.astype(BF16))
    r.apply(matmul_desc(pa, pb, pc, m, n, k, transpose_b=True, ldb=k))
    assert np.array_equal(r.get(pc, (m, n), BF16), (a @ bt.T).astype(BF16))


def test_a_muon_kernel_is_unknown_until_the_registry():
    r = Run()
    r.apply(vector_desc(BASE, 0))
    with pytest.raises(FuncError, match="unknown kernel 7"):
        r.apply(vector_desc(BASE, 7, 24, 0x40), block=bytes(range(24)))
