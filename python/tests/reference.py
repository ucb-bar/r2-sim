"""fp64 references of the Muon kernels of the registry (kernels.yaml), for
the conformance tests of the backends. Each function takes and returns
NumPy arrays (bf16 inputs as float values), computes in fp64, and leaves
the final rounding to the caller: a backend computes in fp32 and rounds
once to bf16, so the tests compare within a tolerance.
"""

import numpy as np


def row_stats(src: np.ndarray, stats: np.ndarray, count: int) -> np.ndarray:
    """src [rows][cols]; stats [rows][2] (mean, M2) before this chunk.
    Returns the merged (mean, M2) after the chunk (Chan et al.)."""
    x = src.astype(np.float64)
    mean_c = x.mean(1)
    m2_c = ((x - mean_c[:, None]) ** 2).sum(1)
    if count == 0:
        return np.stack([mean_c, m2_c], 1)
    cols = x.shape[1]
    mean, m2 = stats[:, 0].astype(np.float64), stats[:, 1].astype(np.float64)
    n = count + cols
    d = mean_c - mean
    return np.stack([mean + d * cols / n, m2 + m2_c + d * d * count * cols / n], 1)


def gelu_tanh(y: np.ndarray) -> np.ndarray:
    return 0.5 * y * (1 + np.tanh(np.sqrt(2 / np.pi) * (y + 0.044715 * y ** 3)))


NORM, BIAS, RES, GELU_TANH, C_F32 = 1, 2, 4, 8, 16


def epilogue(c, flags, res=None, bias=None, u=None, stats=None, k=0, eps=0.0) -> np.ndarray:
    """One out tile: c [rows][cols]; res [rows][cols]; bias, u [cols];
    stats [rows][2] (mean, M2). Returns y before the bf16 rounding."""
    y = c.astype(np.float64)
    if flags & NORM:
        mean, m2 = stats[:, 0:1].astype(np.float64), stats[:, 1:2].astype(np.float64)
        rstd = 1 / np.sqrt(m2 / k + eps)
        y = rstd * (y - mean * u[None, :].astype(np.float64))
    if flags & BIAS:
        y = y + bias[None, :]
    if flags & GELU_TANH:
        y = gelu_tanh(y)
    if flags & RES:
        y = y + res
    return y


def softmax_stats(s0: np.ndarray, s1: np.ndarray, scale: float) -> np.ndarray:
    """Both halves [rows][cols]. Returns [rows][2] (m, l)."""
    s = np.concatenate([s0, s1], 1).astype(np.float64) * scale
    m = s.max(1)
    return np.stack([m, np.exp(s - m[:, None]).sum(1)], 1)


def softmax_apply(src: np.ndarray, stats: np.ndarray, scale: float) -> np.ndarray:
    """One half [rows][cols] and the stats of both halves. Returns p."""
    m, l = stats[:, 0:1].astype(np.float64), stats[:, 1:2].astype(np.float64)
    return np.exp(src.astype(np.float64) * scale - m) / l


def _self_check() -> None:
    """The chunked statistics equal the statistics of the whole row, and
    the softmax of both halves sums to one per row."""
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, (8, 300))
    st = row_stats(x[:, :128], None, 0)
    st = row_stats(x[:, 128:256], st, 128)
    st = row_stats(x[:, 256:], st, 256)
    assert np.allclose(st[:, 0], x.mean(1)) and np.allclose(st[:, 1], x.var(1) * 300)
    s0, s1 = rng.normal(size=(4, 128)), rng.normal(size=(4, 128))
    ml = softmax_stats(s0, s1, 0.12)
    p = np.concatenate([softmax_apply(s0, ml, 0.12), softmax_apply(s1, ml, 0.12)], 1)
    assert np.allclose(p.sum(1), 1)
    print("reference self-check ok")


if __name__ == "__main__":
    _self_check()
