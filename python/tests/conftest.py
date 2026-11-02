"""The providers of the Python backend: the tests of the ops and the
kernels run for each. The torch cases need PyTorch (the pixi environment
torch: `pixi run -e torch python -m pytest python/tests`)."""

import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="no PyTorch (pixi environment torch)")
PROVIDERS = ["numpy", pytest.param("torch", marks=needs_torch)]


@pytest.fixture(params=PROVIDERS)
def provider(request) -> str:
    return request.param
