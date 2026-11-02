"""End to end: the RV fixture probe2 (crates/nu-rv/tests/fixtures/rv) on
configs/2p-smem.yaml with the Python plug-in. On root.gpc0.sm0 it copies A
(128 rows of 72 bf16 at a pitch of 128 elements in the HBM) into the
SMEM with a 2D copy, computes C = A B^T (B stored [128][72] in the SMEM)
and runs the registry kernel `epilogue` with BIAS: out = bf16(C + bias),
bias fp32 in the HBM."""

import subprocess

import ml_dtypes
import numpy as np
from safetensors.numpy import load_file, save_file

from binary import REPO, r2sim

BF16 = ml_dtypes.bfloat16
PROBE2_BIN = REPO / "crates/nu-rv/tests/fixtures/rv/probe2.bin"
# The SMEM offsets of A, B, C and out (128 x 128 bf16 apart).
A, B, C, OUT = 0, 0x8000, 0x10000, 0x18000


def test_probe2_runs_the_epilogue_on_the_python_plane(tmp_path):
    rng = np.random.default_rng(64)
    a = rng.standard_normal((128, 128)).astype(BF16)
    b = rng.standard_normal((128, 72)).astype(BF16)
    bias = rng.standard_normal(128).astype(np.float32)
    save_file({"A": a, "B": b, "bias": bias}, str(tmp_path / "in.safetensors"))
    entries = [
        ("imem", "root", f'bin = "{PROBE2_BIN}"'),
        ("at", "root.gpc0.hbm+0x0", 'safetensors = "in.safetensors"\ntensor = "A"'),
        ("at", f"root.gpc0.sm0.smem+{B:#x}", 'safetensors = "in.safetensors"\ntensor = "B"'),
        ("at", "root.gpc0.hbm+0x100000", 'safetensors = "in.safetensors"\ntensor = "bias"'),
        ("at", f"root.gpc0.sm0.smem+{C:#x}", 'name = "C"\ndtype = "bf16"\nshape = [128, 128]'),
        ("at", f"root.gpc0.sm0.smem+{OUT:#x}",
         'name = "out"\ndtype = "bf16"\nshape = [128, 128]'),
    ]
    (tmp_path / "image.toml").write_text(
        "".join(f'[[entry]]\n{k} = "{v}"\n{rest}\n\n' for k, v, rest in entries))
    out = tmp_path / "out.safetensors"
    subprocess.run(
        [str(r2sim()), "run", "--config", str(REPO / "configs/2p-smem.yaml"), "--program", "rv",
         "--functional", "python", "--image", str(tmp_path / "image.toml"),
         "--image-out", str(out)],
        check=True, stdout=subprocess.DEVNULL,
    )
    got = load_file(str(out))
    c = (a[:, :72].astype(np.float32) @ b.astype(np.float32).T).astype(BF16)
    assert np.array_equal(got["C"], c)
    assert np.array_equal(got["out"], (c.astype(np.float32) + bias).astype(BF16))
