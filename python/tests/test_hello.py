"""End to end: `r2sim run --program rv` with the RV hello fixture
(crates/nu-rv/tests/fixtures/rv/hello.bin) on configs/2p-smem.yaml, with the
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
HELLO_BIN = REPO / "crates/nu-rv/tests/fixtures/rv/hello.bin"


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
        f"""
[[entry]]
imem = "root"
bin = "{HELLO_BIN}"

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
        [str(r2sim()), "run", "--config", str(REPO / "configs/2p-smem.yaml"), "--program", "rv",
         "--functional", "python", "--image", str(tmp_path / "image.toml"),
         "--image-out", str(out)],
        check=True, stdout=subprocess.DEVNULL,
    )
    got = load_file(str(out))
    assert np.array_equal(got["A"], a) and np.array_equal(got["B"], b)
    want = (a.astype(np.float32) @ b.astype(np.float32)).astype(BF16)
    assert got["C"].dtype == BF16 and got["C"].shape == (128, 128)
    assert np.array_equal(got["C"], want)


def test_hello_with_each_backend(tmp_path):
    """C = A @ B with the Numeric backend and with the Python plane."""
    rng = np.random.default_rng(2026)
    a = rng.standard_normal((128, 128)).astype(BF16)
    b = rng.standard_normal((128, 128)).astype(BF16)
    save_file({"A": a, "B": b}, str(tmp_path / "ab.safetensors"))
    (tmp_path / "image.toml").write_text(
        f"""
[[entry]]
imem = "root"
bin = "{HELLO_BIN}"

[[entry]]
at = "root.gpc0.hbm+0x0"
safetensors = "ab.safetensors"
tensor = "A"

[[entry]]
at = "root.gpc0.hbm+0x8000"
safetensors = "ab.safetensors"
tensor = "B"

[[entry]]
name = "C"
at = "root.gpc0.hbm+0x10000"
dtype = "bf16"
shape = [128, 128]
"""
    )
    want = (a.astype(np.float32) @ b.astype(np.float32)).astype(BF16)
    for backend in ["numeric", "python"]:
        out = tmp_path / f"{backend}.safetensors"
        subprocess.run(
            [str(r2sim()), "run", "--config", str(REPO / "configs/2p-smem.yaml"), "--program", "rv",
             "--functional", backend, "--image", str(tmp_path / "image.toml"),
             "--image-out", str(out)],
            check=True, stdout=subprocess.DEVNULL,
        )
        assert np.array_equal(load_file(str(out))["C"], want), backend
