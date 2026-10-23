"""End to end: `r2sim run --program hello` on configs/2p.yaml with the
Python plane (`--functional python`) and a memory image. A and B are
random bf16 matrices from a fixed seed; C, read back with --image-out, must
equal numpy's A @ B computed in fp32 and rounded to bf16 once (exact
equality: the plane accumulates in fp32 as numpy does, and the hello
matmul is one task)."""

import os
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors.numpy import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
BF16 = ml_dtypes.bfloat16


def r2sim() -> Path:
    """The release binary: $R2SIM, else the cargo target directory's."""
    if "R2SIM" in os.environ:
        return Path(os.environ["R2SIM"])
    target = Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target"))
    exe = target / "release" / "r2sim"
    assert exe.exists(), f"no {exe}: build with cargo build --release, or set R2SIM"
    return exe


def test_hello_with_the_python_plane(tmp_path):
    rng = np.random.default_rng(2026)
    a = rng.standard_normal((128, 128)).astype(BF16)
    b = rng.standard_normal((128, 128)).astype(BF16)
    # A as an f32 .npy and B as a bf16 safetensors tensor: both load paths.
    np.save(tmp_path / "a.npy", a.astype(np.float32))
    save_file({"B": b}, str(tmp_path / "b.safetensors"))
    (tmp_path / "image.toml").write_text(
        """
[[entry]]
name = "A"
at = "root.gpc0.hbm+0x0"
npy = "a.npy"
dtype = "bf16"

[[entry]]
name = "B"
at = "root.gpc0.hbm+0x8000"
safetensors = "b.safetensors"
tensor = "B"

[[entry]]
name = "C"
at = "root.gpc0.hbm+0x10000"
dtype = "bf16"
shape = [128, 128]
"""
    )
    out = tmp_path / "out.safetensors"
    subprocess.run(
        [str(r2sim()), "run", "--config", str(REPO / "configs/2p.yaml"), "--program", "hello",
         "--functional", "python", "--image", str(tmp_path / "image.toml"),
         "--image-out", str(out)],
        check=True, stdout=subprocess.DEVNULL,
    )
    got = load_file(str(out))
    assert np.array_equal(got["A"], a) and np.array_equal(got["B"], b)
    want = (a.astype(np.float32) @ b.astype(np.float32)).astype(BF16)
    assert got["C"].dtype == BF16 and got["C"].shape == (128, 128)
    assert np.array_equal(got["C"], want)
