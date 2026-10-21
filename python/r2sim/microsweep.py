"""Run the `micro` program over parameter grids; one JSON line per point.

Each point is one `r2sim run --program micro` with the mode's parameters,
at the configuration's Neutrino costs (`costs: default`) and with
c_invoke, c_base, c_dep and c_word set to 0 on every Neutrino (`costs:
zero`), so that the control cost is visible. A line holds the point
(mode, parameters, costs, config), the run's end time and the `micro`
object of the run's `--stats` JSON (nu-prog `micro::result`), or the
error of a run that failed.

Usage, from the repository root in the pixi environment:

    PYTHONPATH=python python -m r2sim.microsweep --config CONFIG.yaml \\
        --out points.jsonl [--modes stream,latency] [--sms-max 96] [--jobs 16]

CONFIG needs a copy engine on the device (`root.dma`) for writes into the
device's memories (stream with dir = write and mem = l2 or dram).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COST_KEYS = ["c_invoke", "c_base", "c_dep", "c_word"]
# Every Neutrino of the rtx6000 tree: the device, the gpcs, the SMs.
COST_NODES = ["root", "root.gpc*", "root.gpc*.sm*"]


def grid(**axes):
    """Every combination of the axes, as parameter dictionaries."""
    keys = list(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, values))


def points(modes: list[str], sms_max: int):
    sms = [s for s in [1, 2, 4, 8, 16, 32, 64, 96, 128, 192] if s <= sms_max]
    if "stream" in modes:
        yield from ({"mode": "stream", **p} for p in grid(
            sms=sms, placement=["packed", "spread"],
            mem=["l2", "dram", "smem", "rsmem_gpc", "rsmem_far"], dir=["read", "write"],
            bytes=[1 << 20], chunk=[16384], inflight=[4]))
        # Chunk and transfers in flight, one SM.
        yield from ({"mode": "stream", **p} for p in grid(
            sms=[1], mem=["l2", "dram"], dir=["read"], bytes=[1 << 20],
            chunk=[1024, 4096, 16384, 32768], inflight=[1, 2, 4]))
    if "latency" in modes:
        yield from ({"mode": "latency", **p} for p in grid(
            mem=["l2", "dram", "smem", "rsmem_gpc", "rsmem_far"],
            bytes=[32 << i for i in range(12)]))
    if "atlas" in modes:
        yield from ({"mode": "atlas", **p} for p in grid(
            sms=[1, 16], dtype=["bf16", "fp8", "mxfp4"],
            m=[128], n=[128], k=[1024, 4096]))
    if "muon" in modes:
        yield from ({"mode": "muon", **p} for p in grid(
            sms=[1], lane_ops=[1 << i for i in range(10, 21, 2)], bytes=[0, 65536]))
    if "pingpong" in modes:
        yield from ({"mode": "pingpong", **p} for p in grid(
            pair=["same_gpc", "other_gpc"], rounds=[32]))
    if "barrier" in modes:
        yield from ({"mode": "barrier", **p} for p in grid(sms=[s for s in sms if s >= 2]))
    if "invoke" in modes:
        yield from ({"mode": "invoke", **p} for p in grid(targets=["one"], tasks=[64, 256]))
        yield from ({"mode": "invoke", **p} for p in grid(
            targets=["many"], sms=[s for s in sms if s >= 2], tasks=[256]))


def toml_value(v) -> str:
    if isinstance(v, str):
        return json.dumps(v)
    return str(v).lower() if isinstance(v, bool) else str(v)


def run_point(binary: str, config: str, point: dict, costs: str) -> dict:
    line = {"config": config, "costs": costs, **point}
    args = [binary, "run", "--config", config, "--program", "micro"]
    for k, v in point.items():
        args += ["--param", f"{k}={toml_value(v)}"]
    if costs == "zero":
        for n in COST_NODES:
            for k in COST_KEYS:
                args += ["--set", f"{n}.neutrino.{k}=0"]
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        p = subprocess.run(args + ["--stats", f.name], capture_output=True, text=True)
        if p.returncode != 0:
            line["error"] = (p.stderr.strip().splitlines() or ["exit %d" % p.returncode])[-1]
            return line
        stats = json.loads(Path(f.name).read_text())
    line["end_time"] = stats["end_time"]
    line["wall_seconds"] = stats["wall_seconds"]
    line["result"] = stats.get("micro")
    return line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, type=Path)
    default_binary = Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "release" / "r2sim"
    ap.add_argument("--binary", default=str(default_binary))
    ap.add_argument("--modes", default="stream,latency,atlas,muon,pingpong,barrier,invoke")
    ap.add_argument("--sms-max", type=int, default=96)
    ap.add_argument("--costs", default="default,zero")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    jobs = [(pt, c) for pt in points(a.modes.split(","), a.sms_max) for c in a.costs.split(",")]
    failed = 0
    with a.out.open("w") as out, ThreadPoolExecutor(a.jobs) as pool:
        for line in pool.map(lambda j: run_point(a.binary, a.config, j[0], j[1]), jobs):
            failed += "error" in line
            out.write(json.dumps(line) + "\n")
            out.flush()
    print(f"{len(jobs)} points, {failed} failed -> {a.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
