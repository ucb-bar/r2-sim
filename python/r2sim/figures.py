"""Turns r2sim runs into paper figures (PDF plus 300 dpi PNG).

Usage, from the repository root in the pixi environment:

    PYTHONPATH=python python -m r2sim.figures headline RUN.json ... --out DIR
    PYTHONPATH=python python -m r2sim.figures trace TRACE_DIR --out DIR

The default output folder is out/figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from . import plots, style
from .data import load_run, load_trace

DEFAULT_OUT = Path("out/figures")


def headline(runs: list[Path], out: Path) -> list[Path]:
    """Stage bars of forward-pass runs with their three bounds (double
    column)."""
    data = [load_run(p) for p in runs]
    fig, ax = plt.subplots(figsize=(style.DOUBLE_COL, (14 + 5 * len(data)) * style.MM))
    plots.stage_bars(ax, data)
    return style.save(fig, out / "headline")


def trace_figures(folder: Path, out: Path, window: tuple[int, int] | None = None) -> list[Path]:
    """The two trace figures of one --trace-out folder (two panels each):
    `utilization_gantt` and, when there are phases, `phases`."""
    tr = load_trace(folder)
    files = []
    # utilization_gantt. a: resources over time; b: Gantt of one worker SM.
    fig, (a, b) = plt.subplots(
        2, 1, figsize=(style.DOUBLE_COL, 125 * style.MM), height_ratios=(3, 1.4), layout="constrained"
    )
    im = plots.utilization_heatmap(a, tr)
    if window:
        plots.mark_window(a, window)
    fig.colorbar(im, ax=a, label="Busy fraction", fraction=0.03, pad=0.01)
    style.panel_label(a, "a")
    sms = sorted({str(n) for n in tr.tasks["node"].unique() if ".sm" in str(n)})
    plots.gantt(b, tr, node_prefix=sms[1] if len(sms) > 1 else "", window=window)
    style.panel_label(b, "b")
    files += style.save(fig, out / "utilization_gantt")
    # phases. a: culprit shares per phase; b: phases against their roofline.
    if len(tr.phases):
        fig, (c, d) = plt.subplots(1, 2, figsize=(style.DOUBLE_COL, 70 * style.MM))
        plots.culprit_shares(c, tr)
        style.panel_label(c, "a")
        if tr.phases["roofline"].notna().any():
            plots.phase_roofline(d, tr)
        style.panel_label(d, "b")
        fig.tight_layout()
        files += style.save(fig, out / "phases")
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("headline")
    h.add_argument("runs", nargs="+", type=Path)
    h.add_argument("--out", type=Path, default=DEFAULT_OUT)
    t = sub.add_parser("trace")
    t.add_argument("folder", type=Path)
    t.add_argument("--out", type=Path, default=DEFAULT_OUT)
    t.add_argument("--window", help="A:B cycles for the Gantt chart")
    a = p.parse_args()
    if a.cmd == "headline":
        files = headline(a.runs, a.out)
    else:
        w = tuple(int(x) for x in a.window.split(":")) if a.window else None
        files = trace_figures(a.folder, a.out, w)
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
