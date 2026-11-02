"""The state of the functional plane: memory, the provider of the
arithmetic, the Atlas accumulators and the Muon kernels."""

from __future__ import annotations

from r2sim.func import ops, provider_numpy, provider_torch
from r2sim.func.kernels import Registry
from r2sim.func.memory import FuncError, Memory

PROVIDERS = ("numpy", "torch")


def make_provider(name: str, device: str = "cpu"):
    """The provider `name` (`numpy`, or `torch` on `device`) and its
    kernels by name."""
    if name == "numpy":
        if device != "cpu":
            raise FuncError(f"provider numpy: device {device!r}: numpy runs on the cpu")
        return provider_numpy.Provider(), provider_numpy.KERNELS
    if name == "torch":
        return provider_torch.Provider(device), provider_torch.KERNELS
    raise FuncError(f"provider {name!r}: this version has {', '.join(PROVIDERS)}")


class Plane:
    """Memory, the provider, per Atlas (by PE index) the fp32 partial tiles
    it holds, keyed by the C base, and the kernels of the registry."""

    def __init__(self, memory: Memory, kernels: list[dict] | None = None,
                 provider: str = "numpy", device: str = "cpu") -> None:
        self.memory = memory
        self.provider, functions = make_provider(provider, device)
        self.kernels = Registry(kernels or [], provider, functions)
        self.accumulators: dict[int, dict] = {}

    def accumulator(self, pe: int | None) -> dict:
        return self.accumulators.setdefault(-1 if pe is None else pe, {})

    def task(self, desc: dict) -> None:
        ops.task(self, desc)
