"""Run memarch kernel points on r2sim: the bridge between a design point
(simconfig.SimPoint) and the kernel (r2-toolchain examples/memarch).

Paths come from the environment, with defaults for this checkout:
  MEMARCH_R2SIM   r2sim binary (default: ../../target/release/r2sim)
  MEMARCH_KERNEL  the kernel example (default: the memarch-ws checkout next to r2-sim)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import simconfig

HERE = Path(__file__).resolve().parent
R2SIM = Path(os.environ.get("MEMARCH_R2SIM", HERE.parents[1] / "target" / "release" / "r2sim"))
KERNEL = Path(os.environ.get(
    "MEMARCH_KERNEL", HERE.parents[2] / "memarch-ws" / "r2-toolchain" / "examples" / "memarch"))

_spec = importlib.util.spec_from_file_location("memarch_run", KERNEL / "host" / "run.py")
krun = importlib.util.module_from_spec(_spec)
sys.modules["memarch_run"] = krun
_spec.loader.exec_module(krun)


def run(sim: simconfig.SimPoint, run_dir: Path | None = None, **kernel) -> dict:
    """Runs the kernel point `kernel` (fields of host/run.py's Point) on the
    design `sim`; the result line (cycles, phase stats) or an error."""
    run_dir = Path(run_dir or tempfile.mkdtemp(prefix="memarch-"))
    cfg = simconfig.write(sim, run_dir / "cfg")
    try:
        return krun.run_point(str(R2SIM), cfg.resolve(), krun.Point(**kernel), run_dir.resolve())
    except ValueError as e:
        return {"point": kernel, "error": str(e)}


if __name__ == "__main__":
    import json
    import sys

    sim = simconfig.SimPoint(tmem_rd=256, tmem_wr=256)
    out = Path(sys.argv[1])
    for name, kw in {
        "gemm_smem": dict(mode="gemm", m=256, n=256, k=256),
        "attn_A": dict(mode="attn", tm=64, kc=64, kvblocks=8, opmem="tmem", accmem="tmem"),
        "attn_C": dict(mode="attn", tm=64, kc=64, kvblocks=8, opmem="smem", accmem="tmem"),
        "attn_B": dict(mode="attn", tm=64, kc=64, kvblocks=8, opmem="smem", accmem="smem"),
        "muon": dict(mode="muon", tasks=4),
    }.items():
        r = run(sim, out / name, **kw)
        busiest = r.get("phase", {}).get("busiest", [])[:4]
        print(name, r.get("cycles"), r.get("error", ""), json.dumps(busiest))
