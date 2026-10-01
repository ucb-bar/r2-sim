"""Sensitivity sweeps of the forward pass: runs the points of one or more
axes in parallel, collects each axis into one CSV, and plots the stages
against the parameter with the bounds as references.

Usage, from r2-sim/ in the workspace environment:

    PYTHONPATH=python python -m r2sim.sweep run AXIS [AXIS ...] [--out DIR] [--jobs 16]
    PYTHONPATH=python python -m r2sim.sweep collect AXIS [AXIS ...] [--out DIR]
    PYTHONPATH=python python -m r2sim.sweep plot [--out DIR] [--axes AXIS ...]

AXIS is a name from `AXES` (c_invoke, msg_latency, c_resolve, hbm, smem,
sm_count, acc_buffer, versions, root_versions). Every point is one
`r2sim run` of the forward pass; `BASE` holds the arguments common to all
points. A point that stops (a fatal error, or a builder panic such as a
window that needs more versioned ranges than an owner's table) is a row
with its status and no stage columns.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .data import load_run

R2SIM = "../build/target/release/r2sim"
DEFAULT_OUT = Path("../scratch/sweeps")

# Default point: 2p.yaml, mxfp4, 1 MiB accumulator buffer, seed 7.
BASE = [
    "--program", "forward",
    "--params", "configs/forward.toml",
    "--synthetic", "7",
    "--phase-stats",
    "--param", "weight_dtype=mxfp4",
]
BASE_CONFIG = "configs/2p.yaml"
BASE_ACC = ["--acc-buffer", "1048576"]

LAT = [8, 33, 138, 574, 2386, 10000]

# The color of the default point's configuration.
BASE_COLOR = style.config_color("2p.yaml", "mxfp4")


@dataclass
class Point:
    """One run of an axis: its x value (number or label), an optional
    series label, and the arguments beyond `BASE`."""

    x: float | str
    args: list[str]
    config: str = BASE_CONFIG
    series: str = ""
    acc: list[str] = field(default_factory=lambda: list(BASE_ACC))


def _every_neutrino(key: str, value: object) -> list[str]:
    """--set arguments that set `key` of every Neutrino (device, clusters,
    SMs)."""
    out = []
    for path in ("root", "root.*", "root.*.*"):
        out += ["--set", f"{path}.neutrino.{key}={value}"]
    return out


def _versions(v: int) -> list[str]:
    """--set arguments for a version table of `v` at the clusters and the
    SMs; the device keeps its own (1024 in 2p.yaml)."""
    out = []
    for path in ("root.*", "root.*.*"):
        out += ["--set", f"{path}.neutrino.versions={v}"]
    return out


# Per axis: the x label, "lines" (a latency: log-log lines; else stacked
# bars), the baseline x, the unit of x in labels, and the points.
AXES: dict[str, dict] = {
    "c_invoke": {
        "label": "Invoke cost c_invoke (cycles)",
        "lines": True,
        "baseline": 8,
        "points": [Point(v, _every_neutrino("c_invoke", v)) for v in LAT],
    },
    "msg_latency": {
        "label": "Remote resolution latency per message (cycles)",
        "lines": True,
        "baseline": 0,
        "points": [Point(0, [])] + [Point(v, ["--set", f"global.msg_latency={v}"]) for v in LAT],
    },
    "c_resolve": {
        "label": "Local resolution latency c_resolve (cycles)",
        "lines": True,
        "baseline": 0,
        "points": [Point(0, [])] + [Point(v, _every_neutrino("c_resolve", v)) for v in LAT],
    },
    "hbm": {
        "label": "HBM port (B/cycle)",
        "lines": False,
        "unit": "B/cycle",
        "baseline": 273,
        "points": [Point(v, ["--set", f"root.hbm.bw_bytes={v}"]) for v in (137, 273, 546, 1092)],
    },
    "smem": {
        "label": "SMEM per SM (MiB)",
        "lines": False,
        "unit": "MiB",
        "baseline": 1,
        "points": [
            Point(
                s,
                ["--set", f"root.*.*.smem.addr=[[0x0, {s << 20:#x}]]", "--set", f"root.*.*.smem.bw_bytes={p}"],
                series=f"{p} B/cycle",
            )
            for p in (256, 512)
            for s in (1, 2, 4)
        ],
    },
    "sm_count": {
        "label": "SMs",
        "lines": False,
        "baseline": 8,
        "points": [
            Point(4, [], config="configs/2p-4sm.yaml"),
            Point(8, []),
            Point(16, [], config="configs/2p-16sm.yaml"),
        ],
    },
    "acc_buffer": {
        "label": "Atlas accumulator buffer (KiB)",
        "lines": False,
        "unit": "KiB",
        "baseline": 1024,
        "points": [Point(k, [], acc=["--acc-buffer", str(k * 1024)]) for k in (0, 256, 512, 1024, 2048)],
    },
    "versions": {
        "label": "Version table per SM and cluster (entries)",
        "lines": False,
        "baseline": 256,
        "points": [Point(v, _versions(v)) for v in (64, 128, 256, 512)],
    },
    "root_versions": {
        "label": "Version table at the device (entries)",
        "lines": False,
        "baseline": 1024,
        "points": [Point(v, ["--set", f"root.neutrino.versions={v}"]) for v in (512, 1024, 2048)],
    },
}


def _name(p: Point) -> str:
    return f"{p.series.replace(' ', '').replace('/', '')}_{p.x}".strip("_").replace(" ", "")


def run_axes(axes: list[str], out: Path, jobs: int) -> list[Path]:
    """Runs every point of `axes` in one pool (at most `jobs` at a time) and
    writes `out/<axis>/points.csv` per axis; returns their paths."""

    def one(job: tuple[str, Point]) -> Path:
        axis, p = job
        d = out / axis
        stats = d / f"{_name(p)}.json"
        cmd = [R2SIM, "run", "--config", p.config, *BASE, *p.acc, *p.args, "--stats", str(stats)]
        with (d / f"{_name(p)}.log").open("w") as log:
            # A stopped run exits nonzero; its row records the status.
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        return stats

    jobs_list = [(a, p) for a in axes for p in AXES[a]["points"]]
    for a in axes:
        (out / a).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, jobs_list))
    return collect(axes, out)


def collect(axes: list[str], out: Path) -> list[Path]:
    """Writes `out/<axis>/points.csv` from the runs' JSON files."""
    paths = []
    for a in axes:
        rows = [_row(p, out / a / f"{_name(p)}.json") for p in AXES[a]["points"]]
        keys = list(dict.fromkeys(k for r in rows for k in r))
        path = out / a / "points.csv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        paths.append(path)
    return paths


def _stage_culprit(run: dict, start: int, end: int) -> str:
    """The resource with the most culprit time in [start, end), from the
    --phase-stats intervals."""
    acc: dict[str, float] = {}
    for ps in run.get("phase_stats", []):
        lo, hi = max(ps["start"], start), min(ps["end"], end)
        if hi <= lo:
            continue
        for name, share in ps["culprits"]:
            acc[name] = acc.get(name, 0.0) + share * (hi - lo)
    return max(acc, key=acc.get) if acc else ""


def _row(p: Point, stats: Path) -> dict:
    row = {"x": p.x, "series": p.series, "config": p.config}
    if not stats.exists():
        # A builder panic writes no stats; the log has the message.
        row["status"] = "failed (no stats; see the log)"
        return row
    run = load_run(stats)
    s = run["stats"]
    ok = s["fatal"] is None and not s["unfinished"]
    row["messages"] = run["messages"]
    row["vreset_messages"] = s["messages_by_kind"].get("VersionReset", 0)
    row["vreset_data"] = run["forward"].get("vreset_data", 0)
    row["vreset_signal"] = run["forward"].get("vreset_signal", 0)
    row["version_peak_max"] = max(s["version_peak"], default=0)
    row["status"] = "ok" if ok else f"stopped at {run['end_time']} ({s['unfinished'] or s['fatal']})"
    f = run["forward"]
    if not ok or not f["stages"]:
        return row
    for st in f["stages"] + [f["total"]]:
        n = st["name"]
        row[f"{n}_cycles"] = st["cycles"]
        row[f"{n}_hbm_bytes"] = st["hbm_bytes"]
        row[f"{n}_own"] = st["own_traffic_bound"]
        row[f"{n}_once"] = st["weights_once_bound"]
        row[f"{n}_roofline"] = st["roofline"]
        if n != "total":
            row[f"{n}_culprit"] = _stage_culprit(run, st["start"], st["end"])
    return row


def baseline_total(rows: list[dict], baseline: float) -> float:
    """The total cycles of the baseline point."""
    return next(float(r["total_cycles"]) for r in rows if float(r["x"]) == baseline)


def degradation_point(rows: list[dict], base: float, frac: float = 0.10) -> float | None:
    """The x (log-interpolated) where the total first exceeds the baseline
    total `base` by `frac`; None when it never does."""
    pts = sorted((float(r["x"]), float(r["total_cycles"])) for r in rows if float(r["x"]) > 0)
    limit = base * (1 + frac)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= limit < y1:
            t = (limit - y0) / (y1 - y0)
            return float(np.exp(np.log(x0) + t * (np.log(x1) - np.log(x0))))
    return None


def _latency_axis(ax: plt.Axes, axis: str, rows: list[dict]) -> None:
    """Log-log lines of the total and the stages against a latency, the
    bounds as horizontal references, and the +10 % point."""
    spec = AXES[axis]
    base = baseline_total(rows, spec["baseline"])
    pts = [r for r in rows if float(r["x"]) > 0]
    x = [float(r["x"]) for r in pts]
    ends = []
    for st, alpha in style.STAGE_ALPHA + (("total", 1.0),):
        y = [float(r[f"{st}_cycles"]) / 1e6 for r in pts]
        lw = 1.2 if st == "total" else 0.8
        ls = "-" if st == "total" else "--"
        ax.plot(x, y, ls, color=BASE_COLOR, alpha=alpha, linewidth=lw, marker="o" if st == "total" else None,
                markersize=2.5)
        ends.append([y[-1], st])
    # Direct labels at the right end, spread apart on the log axis.
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] * 1.25)
    for y, st in ends:
        ax.annotate(st, (x[-1], y), xytext=(3, 0), textcoords="offset points", fontsize=5, va="center")
    r0 = pts[0]
    for key, _, name in style.bound_styles(horizontal=False):
        v = float(r0[f"total_{key}"]) / 1e6
        ax.axhline(v, color="#999999", linewidth=0.4, linestyle=":")
        ax.annotate(name, (x[0], v), xytext=(0, 1.5), textcoords="offset points", fontsize=4.5, color="#555555")
    ax.axhline(base * 1.1 / 1e6, color="black", linewidth=0.5, linestyle="-.")
    d = degradation_point(rows, base)
    if d:
        ax.plot([d], [base * 1.1 / 1e6], "v", color="black", markersize=3)
        ax.annotate(f"+10 % at {d:.0f}", (d, base * 1.1 / 1e6), xytext=(3, 4), textcoords="offset points",
                    fontsize=5, ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(spec["label"])
    ax.set_ylabel("One step (M cycles)")


def _bar_axis(ax: plt.Axes, axis: str, rows: list[dict], failed: list[str]) -> None:
    """Stacked stage bars per point with the total's bounds as markers."""
    spec = AXES[axis]
    series = sorted({r["series"] for r in rows})
    colors = [BASE_COLOR, style.OKABE_ITO["vermillion"]]
    width = 0.8 / len(series)
    labels: list[str] = []
    for si, se in enumerate(series):
        part = [r for r in rows if r["series"] == se]
        for r in part:
            if r["x"] not in labels:
                labels.append(r["x"])
            x = labels.index(r["x"]) + (si - (len(series) - 1) / 2) * width
            bottom = 0.0
            for st, alpha in style.STAGE_ALPHA:
                h = float(r[f"{st}_cycles"]) / 1e6
                ax.bar(x, h, bottom=bottom, width=width * 0.8, color=colors[si], alpha=alpha, linewidth=0)
                bottom += h
            for key, marker, _ in style.bound_styles(horizontal=False):
                ax.plot(x + width * 0.5, float(r[f"total_{key}"]) / 1e6, marker, color="black",
                        markersize=2.5, markeredgewidth=0.5, fillstyle="none")
    if len(series) > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors[: len(series)]]
        ax.legend(handles, series, loc="upper right", handlelength=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(lb).replace("go-back-N", "go-back-\nN") for lb in labels])
    ax.set_xlabel(spec["label"] + ("\n(did not finish: " + ", ".join(failed) + ")" if failed else ""))
    ax.set_ylim(0, ax.get_ylim()[1] * (1.25 if len(series) > 1 else 1.0))
    ax.set_ylabel("One step (M cycles)")


def plot_axis(ax: plt.Axes, axis: str, out: Path) -> None:
    """One axis: lines for the latency axes, stacked bars for the others."""
    with (out / axis / "points.csv").open() as fh:
        allrows = list(csv.DictReader(fh))
    rows = [r for r in allrows if r["status"] == "ok"]
    unit = AXES[axis].get("unit", "")
    failed = [" ".join(filter(None, (r["x"], unit, r["series"]))) for r in allrows if r["status"] != "ok"]
    if AXES[axis]["lines"]:
        _latency_axis(ax, axis, rows)
    else:
        _bar_axis(ax, axis, rows, failed)


def _key(fig: plt.Figure) -> None:
    """The shared key of the stage shades and the bound glyphs."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR, alpha=a) for _, a in style.STAGE_ALPHA]
    handles += style.bound_handles(horizontal=False)
    names = [s for s, _ in style.STAGE_ALPHA] + [n for _, _, n in style.bound_styles(horizontal=False)]
    fig.legend(handles, names, loc="outside lower center", ncol=len(names), handlelength=0.9)


def plot_all(out: Path, axes: list[str]) -> list[Path]:
    """One figure per axis and a 2 x 4 overview (double column)."""
    files = []
    for a in axes:
        fig, ax = plt.subplots(figsize=(style.SINGLE_COL, 60 * style.MM), layout="constrained")
        plot_axis(ax, a, out)
        _key(fig)
        files += style.save(fig, out / "figures" / a)
    fig, grid = plt.subplots(2, 4, figsize=(style.DOUBLE_COL, 115 * style.MM), layout="constrained")
    for k, (a, ax) in enumerate(zip(axes, grid.flat)):
        plot_axis(ax, a, out)
        style.panel_label(ax, "abcdefgh"[k])
    for ax in list(grid.flat)[len(axes):]:
        ax.set_visible(False)
    _key(fig)
    files += style.save(fig, out / "figures" / "overview")
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("axes", nargs="+", choices=list(AXES))
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    r.add_argument("--jobs", type=int, default=16)
    c = sub.add_parser("collect")
    c.add_argument("axes", nargs="+", choices=list(AXES))
    c.add_argument("--out", type=Path, default=DEFAULT_OUT)
    pl = sub.add_parser("plot")
    pl.add_argument("--out", type=Path, default=DEFAULT_OUT)
    pl.add_argument("--axes", nargs="+", default=list(AXES))
    a = p.parse_args()
    if a.cmd == "run":
        for path in run_axes(a.axes, a.out, a.jobs):
            print(path)
    elif a.cmd == "collect":
        for path in collect(a.axes, a.out):
            print(path)
    else:
        for f in plot_all(a.out, a.axes):
            print(f)


if __name__ == "__main__":
    main()
