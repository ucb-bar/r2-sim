"""Turns the tracer files of an r2sim run (--trace-out) into a paper
figure (PDF plus 300 dpi PNG).

Usage, from the repository root in the pixi environment:

    PYTHONPATH=python python -m r2sim.figures TRACE_DIR --out DIR [--window A:B]

The default output folder is out/figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from . import plots, style
from .data import load_trace

DEFAULT_OUT = Path("out/figures")


def trace_figures(folder: Path, out: Path, window: tuple[int, int] | None = None) -> list[Path]:
    """The figure `utilization_gantt` of one --trace-out folder (two
    panels)."""
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
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window", help="A:B cycles for the Gantt chart")
    a = p.parse_args()
    w = tuple(int(x) for x in a.window.split(":")) if a.window else None
    files = trace_figures(a.folder, a.out, w)
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
