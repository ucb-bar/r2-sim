"""numpy against torch on the same inputs: equal bytes for the DMA (one
row and 2D) and for matmuls with integer data (every fp32 sum exact); the
kernels within one bf16 ulp of each other (bf16 outputs) or 1e-5 relative
(fp32 statistics)."""

import ml_dtypes
import numpy as np
import pytest

import test_kernels as tk
from conftest import needs_torch
from test_backend import BASE, Run, dma_desc, matmul_desc

pytestmark = needs_torch

BF16 = ml_dtypes.bfloat16


def both(fn):
    """`fn(run)` on a numpy run and on a torch run: the two memories."""
    out = []
    for p in ("numpy", "torch"):
        r = Run(p)
        fn(r)
        out.append(r.mem)
    return out


def test_dma_one_row_and_2d_give_the_same_bytes():
    src = np.random.default_rng(7).integers(0, 256, 128 * 256, dtype=np.uint8)

    def copies(r):
        r.put(BASE, src)
        r.apply(dma_desc(BASE, BASE + 0x10000, 4096), pe=0xFFFFFFFF)
        r.apply(dma_desc(BASE, BASE + 0x20000, 128 * 144, 128, 144, 256, 160, dst_len=0x5000),
                pe=0xFFFFFFFF)
        # Overlapping ranges: the source is read first.
        r.apply(dma_desc(BASE + 0x10000, BASE + 0x10000 + 512, 2048), pe=0xFFFFFFFF)

    a, b = both(copies)
    assert np.array_equal(a, b)


def test_matmuls_with_integer_data_give_the_same_bytes():
    # A chain with keep and accumulate, then B transposed: integer data
    # whose fp32 sums are exact, so the order of the sums does not matter.
    m = n = 128
    k = 256
    i, p, j = np.arange(m)[:, None], np.arange(k), np.arange(n)
    a = ((i * 7 + p[None, :] * 3) % 9 - 4).astype(BF16)
    b = ((p[:, None] * 5 + j[None, :] * 11) % 9 - 4).astype(BF16)
    pa, pb, pc, pd = BASE, BASE + 0x20000, BASE + 0x40000, BASE + 0x50000

    def chain(r):
        r.put(pa, a)
        r.put(pb, b)
        for q in range(2):
            r.apply(matmul_desc(pa + q * 256, pb + q * 128 * n * 2, pc, m, n, 128,
                                accumulate=q > 0, keep=q == 0, lda=k, ldb=n))
        r.apply(matmul_desc(pa, pb, pd, m, 128, 128, transpose_b=True, lda=k, ldb=n))

    x, y = both(chain)
    assert np.array_equal(x, y)
    assert x[0x40000:0x48000].any()


def bf16_close(x: np.ndarray, y: np.ndarray) -> None:
    """Within one bf16 ulp of the larger magnitude."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    big = np.maximum(np.abs(x), np.abs(y))
    ulp = np.exp2(np.floor(np.log2(np.where(big > 0, big, 1))) - 7)
    assert (np.abs(x - y) <= ulp).all()


def test_row_stats():
    for (x, _), (y, _) in zip(tk.row_stats_case("numpy"), tk.row_stats_case("torch")):
        np.testing.assert_allclose(x, y, rtol=1e-5)


@pytest.mark.parametrize("step,flags,cols", tk.VARIANTS, ids=[v[0] for v in tk.VARIANTS])
def test_epilogue(step, flags, cols):
    x = tk.epilogue_case("numpy", step, flags, cols)[2]
    y = tk.epilogue_case("torch", step, flags, cols)[2]
    bf16_close(x, y)


def test_softmax_stats():
    np.testing.assert_allclose(tk.softmax_stats_case("numpy")[0],
                               tk.softmax_stats_case("torch")[0], rtol=1e-5)


def test_softmax_apply():
    bf16_close(tk.softmax_apply_case("numpy")[0], tk.softmax_apply_case("torch")[0])
