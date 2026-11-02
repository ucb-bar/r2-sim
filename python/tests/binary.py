"""The r2sim release binary of the end-to-end tests, the kernel registry
that it prints (`r2sim kernels`), and the options of a run on the Python
plug-in with a provider. A run on torch needs the plug-in built in the
pixi environment torch (`pixi run -e torch build-python`)."""

import functools
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def r2sim() -> Path:
    """The release binary: $R2SIM, else the cargo target directory's."""
    if "R2SIM" in os.environ:
        return Path(os.environ["R2SIM"])
    target = Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target"))
    exe = target / "release" / "r2sim"
    assert exe.exists(), f"no {exe}: build with cargo build --release, or set R2SIM"
    return exe


@functools.cache
def registry() -> list[dict]:
    """The kernel registry as r2-sim passes it to a plug-in."""
    out = subprocess.run([str(r2sim()), "kernels"], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def python_plane(provider: str) -> list[str]:
    """The r2sim options of the Python plug-in with `provider`."""
    flags = ["--functional", "python"]
    if provider != "numpy":
        flags += ["--set", f"functional.backends.python={{lib: libr2_python.so, provider: {provider}}}"]
    return flags
