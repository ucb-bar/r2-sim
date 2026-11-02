"""Conformance of the Muon kernels of each provider: each kernel
through the backend's `apply` on arrays, against the fp64 reference of the
registry's semantics (reference.py). Random bf16 inputs from fixed seeds.

Tolerance: a bf16 output is within one bf16 ulp of the reference value
plus a small absolute term near zero (the backend computes in fp32 and
rounds once); an fp32 statistic is within 1e-5 relative."""

import struct

import ml_dtypes
import numpy as np
import pytest

import reference as ref
from binary import registry
from r2sim.func.memory import FuncError
from test_backend import BASE, G, Run, header

BF16 = ml_dtypes.bfloat16
F32 = np.float32
FORMATS = {"u64": "<Q", "u32": "<I", "f32": "<f"}


def kernel(name: str) -> dict:
    return next(k for k in registry() if k["name"] == name)


def block(name: str, **values) -> bytes:
    """The argument block of kernel `name`, by the registry's offsets; the
    fields not given are 0."""
    k = kernel(name)
    b = bytearray(k["bytes"])
    for f in k["fields"]:
        struct.pack_into(FORMATS[f["type"]], b, f["offset"], values.pop(f["name"], 0))
    assert not values, f"no fields {list(values)} in {name}"
    return bytes(b)


def muon(name: str, at: int, length: int, **values) -> tuple[bytes, bytes]:
    """A Muon task of kernel `name` that may write [at, at + length): the
    descriptor and the argument block."""
    k = kernel(name)
    length = -(-length // G) * G
    desc = header(4, at, length) + struct.pack("<QQQQIIQ", 1, 64, 64, 0, k["id"], k["bytes"], 0)
    return desc, block(name, **values)


class Arena:
    """Regions of the backend's memory, each aligned to the granule."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.next = BASE

    def put(self, x: np.ndarray) -> int:
        at = self.alloc(x.nbytes)
        self.run.put(at, x)
        return at

    def alloc(self, n: int) -> int:
        at = self.next
        self.next += -(-n // G) * G
        assert self.next <= BASE + len(self.run.mem)
        return at


def strided(r: Run, x: np.ndarray, ld: int, arena: Arena) -> int:
    """`x` [rows][cols] stored with leading dimension `ld`."""
    rows, cols = x.shape
    full = np.zeros((rows, ld), x.dtype)
    full[:, :cols] = x
    return arena.put(full)


def get_strided(r: Run, at: int, rows: int, cols: int, ld: int, dt) -> np.ndarray:
    return r.get(at, (rows, ld), dt)[:, :cols]


def within_one_ulp(got: np.ndarray, want: np.ndarray, atol: float) -> None:
    """bf16 `got` within one bf16 ulp of the fp64 `want`, plus `atol`."""
    w = np.abs(want)
    ulp = np.exp2(np.floor(np.log2(np.where(w > 0, w, 1))) - 7)
    err = np.abs(got.astype(np.float64) - want)
    bad = err > ulp + atol
    assert not bad.any(), (f"{bad.sum()} of {bad.size} outside: worst error "
                           f"{(err - ulp).max():.3g} over the ulp")


def close_f32(got: np.ndarray, want: np.ndarray) -> None:
    np.testing.assert_allclose(got.astype(np.float64), want, rtol=1e-5, atol=0)


def row_stats_case(provider: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three chunks of row_stats: 128 rows of 332 columns (the chunks 128,
    128 and 76 of the patch embedding), in a slot of leading dimension 384.
    Per chunk, the statistics (fp32) and the reference from the same
    statistics before the chunk; then the statistics against the whole
    rows."""
    rng = np.random.default_rng(1)
    rows, ld = 128, 384
    x = rng.normal(3.0, 2.0, (rows, 332)).astype(BF16)
    r = Run(provider)
    arena = Arena(r)
    src = strided(r, x, ld, arena)
    stats = arena.alloc(rows * 8)
    have, out = None, []
    for c0, cols in [(0, 128), (128, 128), (256, 76)]:
        desc, b = muon("row_stats", stats, rows * 8, src=src + 2 * c0, stats=stats, rows=rows,
                       cols=cols, ld=ld, count=c0)
        r.apply(desc, block=b)
        got = r.get(stats, (rows, 2), F32).copy()
        out.append((got, ref.row_stats(x[:, c0:c0 + cols].astype(np.float64), have, c0)))
        have = got
    whole = x.astype(np.float64)
    out.append((have, np.stack([whole.mean(1), whole.var(1) * 332], 1)))
    return out


def test_row_stats_of_three_chunks_merge_into_the_statistics_of_the_row(provider):
    for got, want in row_stats_case(provider):
        close_f32(got, want)


NORM, BIAS, RES, GELU_TANH, C_F32 = 1, 2, 4, 8, 16

# The epilogue of each GEMM step of the encoder: (step, flags, cols).
VARIANTS = [
    ("patch", BIAS | RES, 128),
    ("qkv", NORM | BIAS, 128),
    ("o", BIAS | RES, 128),
    ("fc1", NORM | BIAS | GELU_TANH, 80),
    ("fc2", BIAS | RES, 128),
    ("projector", NORM | BIAS, 128),
    ("qkv-c-f32", NORM | BIAS | C_F32, 128),
]


def epilogue_case(provider: str, step: str, flags: int, cols: int):
    """The epilogue of `step`: the out tile (bf16), the fp64 reference, and
    the out rows with their gaps (leading dimension 144)."""
    rng = np.random.default_rng(sum(map(ord, step)))
    rows, k, eps = 128, 1152, 1e-6
    ld_c, ld_out, ld_res = 192, 144, 160
    r = Run(provider)
    arena = Arena(r)
    # A LayerNorm input X and a folded weight W': C = X W', the statistics
    # of X and the column sums of W'.
    x = rng.normal(0.3, 1.0, (rows, k)).astype(BF16).astype(np.float64)
    w = rng.normal(0.0, k ** -0.5, (k, cols)).astype(BF16).astype(np.float64)
    c32 = (x @ w).astype(F32)
    c_dt = F32 if flags & C_F32 else BF16
    c = c32.astype(c_dt)
    st = np.stack([x.mean(1), x.var(1) * k], 1).astype(F32)
    u = w.sum(0).astype(F32)
    bias = rng.normal(0.0, 0.1, cols).astype(F32)
    res = rng.normal(0.0, 1.0, (rows, cols)).astype(BF16)
    at_c = strided(r, c, ld_c, arena)
    at_res = strided(r, res, ld_res, arena)
    at_bias, at_u, at_st = arena.put(bias), arena.put(u), arena.put(st)
    out = arena.alloc(rows * ld_out * 2)
    desc, b = muon("epilogue", out, rows * ld_out * 2, c=at_c, out=out, res=at_res, bias=at_bias,
                   u=at_u, stats=at_st, rows=rows, cols=cols, ld_c=ld_c, ld_out=ld_out,
                   ld_res=ld_res, k=k, eps=eps, flags=flags)
    r.apply(desc, block=b)
    got = get_strided(r, out, rows, cols, ld_out, BF16)
    want = ref.epilogue(c.astype(np.float64), flags, res=res.astype(np.float64),
                        bias=bias.astype(np.float64), u=u, stats=st, k=k, eps=F32(eps))
    return got, want, r.get(out, (rows, ld_out), BF16)


@pytest.mark.parametrize("step,flags,cols", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_epilogue(provider, step, flags, cols):
    got, want, rows = epilogue_case(provider, step, flags, cols)
    within_one_ulp(got, want, atol=1e-6 * np.abs(want).max())
    # The gap after each out row stays zero.
    assert not rows[:, cols:].astype(F32).any()


def test_an_address_that_a_flag_needs_is_not_0(provider):
    r = Run(provider)
    desc, b = muon("epilogue", BASE, G, c=BASE + G, out=BASE, rows=1, cols=8, ld_c=8, ld_out=8,
                   flags=BIAS)
    with pytest.raises(FuncError, match="kernel epilogue: the address bias is 0 with BIAS"):
        r.apply(desc, block=b)


def test_a_kernel_writes_only_inside_its_destination(provider):
    r = Run(provider)
    desc, b = muon("epilogue", BASE, G, c=BASE + 4 * G, out=BASE + G, rows=1, cols=8, ld_c=8,
                   ld_out=8)
    with pytest.raises(FuncError, match="outside its dst"):
        r.apply(desc, block=b)


def scores(seed: int):
    """Scores of one unit: two halves [128][128] side by side in a slot of
    leading dimension 256, and the attention scale of a head of 72."""
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 4.0, (128, 256)).astype(BF16)
    return s, 72 ** -0.5


def softmax_stats_case(provider: str):
    """softmax_stats over both halves: the (m, l) rows and the reference."""
    s, scale = scores(3)
    r = Run(provider)
    arena = Arena(r)
    at = arena.put(s)
    stats = arena.alloc(128 * 8)
    desc, b = muon("softmax_stats", stats, 128 * 8, s0=at, s1=at + 256, stats=stats, rows=128,
                   cols=128, ld=256, scale=scale)
    r.apply(desc, block=b)
    want = ref.softmax_stats(s[:, :128].astype(np.float64), s[:, 128:].astype(np.float64),
                             F32(scale))
    return r.get(stats, (128, 2), F32).copy(), want


def test_softmax_stats_over_both_halves(provider):
    close_f32(*softmax_stats_case(provider))


def softmax_apply_case(provider: str):
    """softmax_apply on each half, half 0 into another slot and half 1 in
    place: the probabilities (bf16) and the reference."""
    s, scale = scores(4)
    r = Run(provider)
    arena = Arena(r)
    at = arena.put(s)
    s64 = s.astype(np.float64)
    st = ref.softmax_stats(s64[:, :128], s64[:, 128:], F32(scale)).astype(F32)
    at_st = arena.put(st)
    out = arena.alloc(128 * 256 * 2)
    for src, dst in [(at, out), (at + 256, at + 256)]:
        desc, b = muon("softmax_apply", dst, 128 * 256 * 2 - 256, src=src, dst=dst, stats=at_st,
                       rows=128, cols=128, ld=256, scale=scale)
        r.apply(desc, block=b)
    p = np.concatenate([r.get(out, (128, 256), BF16)[:, :128],
                        r.get(at, (128, 256), BF16)[:, 128:]], 1)
    want = np.concatenate([ref.softmax_apply(s64[:, :128], st, F32(scale)),
                           ref.softmax_apply(s64[:, 128:], st, F32(scale))], 1)
    return p, want


def test_softmax_apply_on_each_half_one_in_place(provider):
    p, want = softmax_apply_case(provider)
    within_one_ulp(p, want, atol=1e-6 * np.abs(want).max())
    assert abs(p.astype(np.float64).sum(1) - 1).max() < 0.02
