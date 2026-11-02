"""The thread pools of the plug-in (key `threads` of the backend's entry,
default 1): every pool of the loaded libraries (OpenBLAS in the default
environment; MKL, OpenMP and PyTorch in the environment torch) has that
many threads, on the thread of configure and on the thread of the tasks;
the key decides over what the pools had before."""

import threading

import pytest

from conftest import HAS_TORCH
from r2sim.func import threads
from r2sim.func.memory import FuncError
from test_backend import BASE, Run, dma_desc

# The pools of the BLAS of NumPy in each environment (PyTorch adds its own
# when the provider is torch).
POOLS = {"mkl", "openmp"} if HAS_TORCH else {"openblas"}


def pools_on_another_thread(r: Run) -> dict[str, int]:
    """The pools as a new thread sees them after it applied a task."""
    out = {}

    def task():
        r.apply(dma_desc(BASE, BASE + 0x1000, 64), pe=0xFFFFFFFF)
        out.update(r.backend.pools())

    t = threading.Thread(target=task)
    t.start()
    t.join()
    return out


def test_the_pools_have_one_thread_by_default(provider):
    r = Run(provider)
    got = r.backend.pools()
    assert set(got) >= POOLS | ({"torch"} if provider == "torch" else set()), got
    assert set(got.values()) == {1}, got
    assert set(pools_on_another_thread(r).values()) == {1}


@pytest.mark.parametrize("n", [2, 4])
def test_the_key_sets_every_pool(provider, n):
    # The pools had another number before (an earlier backend of the
    # process, or the variables of the environment).
    threads.set_threads(3)
    r = Run(provider, threads=n)
    assert set(r.backend.pools().values()) == {n}, r.backend.pools()
    assert set(pools_on_another_thread(r).values()) == {n}
    # A second backend of the process with another number.
    r = Run(provider, threads=1)
    assert set(r.backend.pools().values()) == {1}


@pytest.mark.parametrize("bad", [0, -2, 1.5, "2", True, None])
def test_threads_must_be_an_integer_of_at_least_1(bad):
    with pytest.raises(FuncError, match=r"threads: .* is not an integer >= 1"):
        Run(threads=bad)
