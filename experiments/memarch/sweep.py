"""The memarch sweep: enumerate, prune analytically, simulate, write JSON lines.

Stage 1: every design that is buildable (< 128 macros, legal shapes) with an
analytic bound >= --bound-min runs the concurrent attention workload.
Stage 2: designs whose attention reaches >= --stage2-min (default 0: all) run
the GEMMs (square, skinny, K-heavy) and the background Muon workloads (a GEMM
beside a transpose, a 4 B gather, a 32 B segment gather). The ideal-memory run
of each kernel point gives its ceiling. Atlas's MAC rate follows the dtype
(archs.MACS). Identical simulator inputs run once (a cache keyed by
the generated sm.yaml and the kernel point).

    python3 sweep.py --out results/sweep.jsonl --jobs 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import archs
import runner
import simconfig

SCRATCH = Path(os.environ.get("MEMARCH_SCRATCH", tempfile.gettempdir()))
GEMMS = ["square", "skinny", "kheavy"]
STAGE2 = GEMMS + archs.BG_WORKLOADS


def sim_key(sp: simconfig.SimPoint, kp: dict) -> str:
    return hashlib.sha1((simconfig.sm_yaml(sp) + json.dumps(kp, sort_keys=True)).encode()).hexdigest()


def run_one(job: tuple[str, simconfig.SimPoint, dict]) -> tuple[str, dict]:
    key, sp, kp = job
    d = Path(tempfile.mkdtemp(prefix="ma-", dir=SCRATCH))
    try:
        r = runner.run(sp, d, **kp)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    out = {"cycles": r.get("cycles"), "error": r.get("error")}
    ph = r.get("phase")
    if ph:
        out["culprits"] = ph["culprits"]
        out["busiest"] = ph["busiest"]
    return key, out


def run_all(jobs: dict, n: int) -> dict:
    """jobs: key -> (SimPoint, kernel point); returns key -> result."""
    items = [(k, sp, kp) for k, (sp, kp) in jobs.items()]
    out = {}
    t = time.time()
    with ProcessPoolExecutor(n) as ex:
        for i, (k, r) in enumerate(ex.map(run_one, items, chunksize=4)):
            out[k] = r
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(items)} runs, {time.time() - t:.0f}s", file=sys.stderr)
    return out


def util(res: dict, kp: dict, macs: int) -> float | None:
    m = archs.macs_of(kp)
    if not res or not res.get("cycles") or not m:
        return None
    return m / (res["cycles"] * macs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--options", default="ABC")
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp8"])
    ap.add_argument("--bound-min", type=float, default=0.6)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--stage2-min", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="designs per (option, P, dtype), cheapest first (0: all)")
    a = ap.parse_args(argv)

    t0 = time.time()
    designs, pruned = [], {}
    for opt in a.options:
        for macs in [None]:
            for dt in a.dtypes:
                macs = archs.MACS[dt]
                group = []
                for d in archs.enumerate_designs(opt, macs, dt):
                    ev = archs.evaluate(d)
                    why = ev.get("pruned", "")
                    if "area" not in ev or "macros" in why and "bandwidth" not in why:
                        pruned[why.split(" ")[0]] = pruned.get(why.split(" ")[0], 0) + 1
                        continue
                    if "bound" not in ev or min(ev["bound"]["attn"], ev["bound"]["gemm"]) < a.bound_min:
                        pruned["bandwidth"] = pruned.get("bandwidth", 0) + 1
                        continue
                    group.append(ev)
                group.sort(key=lambda e: e["area"]["total"])
                designs += group[: a.limit] if a.limit else group
    print(f"{len(designs)} designs to simulate; pruned {pruned}; {time.time() - t0:.0f}s", file=sys.stderr)

    # Stage 1: attention, and each kernel point's ideal ceiling.
    def jobs_for(evs, workloads):
        jobs, refs = {}, []
        for ev in evs:
            d = ev["design"]
            for wl in workloads:
                kp = archs.kernel_point(d, wl)
                for ideal in (False, True):
                    sp = archs.sim_point(d, wl, ideal=ideal)
                    k = sim_key(sp, kp)
                    jobs[k] = (sp, kp)
                    refs.append((ev, wl, ideal, k, kp))
        return jobs, refs

    results = {}
    jobs, refs1 = jobs_for(designs, ["attn"])
    print(f"stage 1: {len(jobs)} unique runs", file=sys.stderr)
    results.update(run_all(jobs, a.jobs))
    for ev, wl, ideal, k, kp in refs1:
        ev.setdefault("runs", {})[wl + ("_ideal" if ideal else "")] = {
            **results[k], "util": util(results[k], kp, ev["design"].macs), "kernel": kp}
    passing = [ev for ev in designs if (ev["runs"]["attn"].get("util") or 0) >= a.stage2_min]
    print(f"stage 1 done: {len(passing)} designs reach {a.stage2_min} on attention; {time.time() - t0:.0f}s",
          file=sys.stderr)

    # Stage 2: GEMMs and the background Muon workloads.
    jobs, refs2 = jobs_for(passing, STAGE2)
    jobs = {k: v for k, v in jobs.items() if k not in results}
    print(f"stage 2: {len(jobs)} unique runs", file=sys.stderr)
    results.update(run_all(jobs, a.jobs))
    for ev, wl, ideal, k, kp in refs2:
        ev["runs"][wl + ("_ideal" if ideal else "")] = {
            **results[k], "util": util(results[k], kp, ev["design"].macs), "kernel": kp}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for ev in designs:
            rec = {"design": asdict(ev["design"]), "label": ev["design"].label(), "area": ev["area"],
                   "bound": ev["bound"], "pruned": ev.get("pruned"), "runs": ev.get("runs", {})}
            f.write(json.dumps(rec) + "\n")
    meta = {"assumptions": archs.ASSUMPTIONS, "pruned_before_sim": pruned, "args": vars(a) | {"out": str(a.out)},
            "seconds": time.time() - t0}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"wrote {a.out}; {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
