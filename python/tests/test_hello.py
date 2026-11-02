"""End to end: `r2sim run --program rv` with the RV hello fixture
(crates/nu-rv/tests/fixtures/rv/hello.bin) on configs/2p-smem.yaml, with the
Python plug-in (`--functional python`, libr2_python.so next to the r2sim
binary: `pixi run build-python`, or `pixi run -e torch build-python` for
the provider torch) and a memory image. A and B are
random bf16 matrices from a fixed seed; C, read back with --image-out, must
equal numpy's A @ B computed in fp32 and rounded to bf16 once (exact
equality: the plane accumulates in fp32 as numpy does, and the hello
matmul is one task)."""

import json
import subprocess

import ml_dtypes
import numpy as np
from safetensors.numpy import load_file, save_file

from binary import REPO, python_plane, r2sim

BF16 = ml_dtypes.bfloat16
HELLO_BIN = REPO / "crates/nu-rv/tests/fixtures/rv/hello.bin"


def test_hello_with_the_python_plane(tmp_path, provider):
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
         *python_plane(provider), "--image", str(tmp_path / "image.toml"),
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


def test_a_mixed_run_with_the_atlas_on_python(tmp_path):
    """The numeric short form with the Atlas on the Python plug-in: the
    DMAs on numeric, the matmul on python; C exact."""
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
    out = tmp_path / "out.safetensors"
    timeline = tmp_path / "timeline.jsonl"
    subprocess.run(
        [str(r2sim()), "run", "--config", str(REPO / "configs/2p-smem.yaml"), "--program", "rv",
         "--functional", "numeric", "--set", "functional.units.atlas=[python]",
         "--image", str(tmp_path / "image.toml"), "--image-out", str(out),
         "--thread-timeline", str(timeline)],
        check=True, stdout=subprocess.DEVNULL,
    )
    want = (a.astype(np.float32) @ b.astype(np.float32)).astype(BF16)
    assert np.array_equal(load_file(str(out))["C"], want)
    applies = [json.loads(line) for line in timeline.read_text().splitlines()]
    used = [(r["op"], r["backend"]) for r in applies if r["state"] == "apply"]
    assert used == [("dma", "numeric"), ("dma", "numeric"), ("matmul", "python"),
                    ("dma", "numeric")]


KERNELS_GEMM = """
tensors = [
  {name = "a", bytes = 131072},
  {name = "b", bytes = 131072},
  {name = "c", bytes = 131072},
]
[[kernels]]
name = "g"
op = "gemm"
inputs = ["a", "b"]
output = "c"
m = 256
n = 256
k = 256
bm = 128
bn = 128
bk = 64
stages = 2
epilogue_ops = 2
"""


def test_a_task_of_a_rust_program_on_a_plugin_stops_the_run(tmp_path):
    """The `kernels` program builds its descriptors in Rust: they have no
    descriptor bytes, so the Python plug-in cannot apply them."""
    (tmp_path / "gemm.toml").write_text(KERNELS_GEMM)
    r = subprocess.run(
        [str(r2sim()), "run", "--config", str(REPO / "configs/rtx6000-half.yaml"),
         "--program", "kernels", "--params", str(tmp_path / "gemm.toml"),
         "--acc-buffer", "131072", "--functional", "python"],
        capture_output=True, text=True,
    )
    assert r.returncode == 2
    assert "backend python (libr2_python.so)" in r.stdout
    assert "has no descriptor bytes: it is a task of a Rust program" in r.stdout
