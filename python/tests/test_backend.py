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


def dma_desc(src, to, n):
    return header(2, to, max(G, n)) + struct.pack("<QQQQ", src, to, n, n)


def matmul_desc(a, b, c, m, n, k, *, accumulate=False, keep=False, lda=None, ldb=None):
    flags = (1 if accumulate else 0) | (2 if keep else 0)
    args = struct.pack("<QQQQIIIIIIIIHH", a, b, 0, 0, lda or k, ldb or n, n, m, n, k,
                       0, flags, 128, 128) + bytes(4)
    return header(3, c, m * n * 2) + args


def test_the_decoder_gives_the_fields_of_the_ops():
    d = decode(dma_desc(BASE, BASE + 0x1000, 64), G, 3)
    assert d == {"kind": "dma", "pe": 3, "dst": [BASE + 0x1000, BASE + 0x1000 + G], "mask": FULL,
                 "src": BASE, "to": BASE + 0x1000, "bytes": 64, "tile_bytes": 64}
    m = decode(matmul_desc(1, 2, BASE, 4, 8, 16, accumulate=True), G, None)
    assert (m["kind"], m["a"], m["b"], m["a2"], m["b2"]) == ("matmul", 1, 2, None, None)
    assert (m["lda"], m["ldb"], m["ldc"], m["m"], m["n"], m["k"]) == (16, 8, 8, 4, 8, 16)
    assert (m["a_dtype"], m["out_dtype"], m["accumulate"], m["keep"]) == ("bf16", "bf16", True, False)
    assert m["tile"] == [128, 128] and m["pe"] is None
    # Two keys between the header and the arguments.
    v = header(4, BASE, G, keys=2) + bytes(16) + struct.pack("<QQQQ", 5, 6, 7, 0)
    assert decode(v, G, 0)["ops"] == 5 and decode(v, G, 0)["src"] is None
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

    def apply(self, desc, pe=0):
        self.seq += 1
        self.backend.apply(self.seq, 0, self.seq, 0, pe, desc)

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
