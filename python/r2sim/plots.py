"""Plots of r2sim runs. Each function draws into one matplotlib axis and
uses the figure style of `style.py`.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import style
from .data import Trace

# One color per task kind of the kernels program in Gantt charts
# (Okabe-Ito, fixed); other kinds are gray.
KIND_COLORS = {
    "load": style.OKABE_ITO["orange"],
    "store": style.OKABE_ITO["sky_blue"],
    "matmul": style.OKABE_ITO["blue"],
    "softmax": style.OKABE_ITO["bluish_green"],
    "rescale": style.OKABE_ITO["black"],
    "norm": style.OKABE_ITO["vermillion"],
    "eltwise": style.OKABE_ITO["purple"],
    "vec": style.OKABE_ITO["yellow"],
}


def resource_class(name: str) -> str:
    """The class of a resource name, for aggregating over SMs."""
    # A link direction is `<link>.ab` or `.ba`; by the naming convention
    # `<from>_to_<to>`, a control link starts at a control bus (`*cbus`).
    parts = name.split(".")
    if len(parts) > 1 and parts[-1] in ("ab", "ba"):
        return "control link" if parts[-2].split("_to_")[0].endswith("cbus") else "link"
    if any(part.endswith("BUS") for part in name.split(".")):
        return "bus"
    for suffix, cls in (
        ("smem.port", "SMEM port"),
        ("atlas.array", "Atlas"),
        ("muon.vector", "Muon"),
        ("hbm.port", "HBM port"),
        ("cmem.port", "cmem port"),
        ("scalar", "Neutrino scalar"),
        ("complete", "completion port"),
    ):
        if name.endswith(suffix):
            return cls
    return "other"


def _heatmap_rows(names: list[str], links: int) -> list[str | None]:
    """Row order of the heatmap: the busiest links and bus ports, memory ports,
    then per SM its SMEM port and Atlas array in pairs; None is a gap row
    between groups."""
    groups: list[list[str]] = [[], [], [], []]
    for n in names:
        cls = resource_class(n)
        if cls == "link":
            groups[0].append(n)
        elif cls == "bus":
            groups[1].append(n)
        elif cls in ("HBM port", "cmem port"):
            groups[2].append(n)
        elif cls in ("SMEM port", "Atlas"):
            groups[3].append(n)
    groups[0] = groups[0][:links]
    groups[1] = groups[1][:links]
    groups[3].sort(key=lambda n: (n.rsplit(".", 2)[0], resource_class(n) != "SMEM port"))
    rows: list[str | None] = []
    for g in groups:
        if g:
            if rows:
                rows.append(None)
            rows.extend(g if g is not groups[0] else sorted(g))
    return rows


def utilization_heatmap(ax: plt.Axes, trace: Trace, links: int = 6) -> plt.cm.ScalarMappable:
    """Busy fraction per resource (rows) over time (columns), grouped by
    class: the `links` busiest links and bus ports, memory ports, then each
    SM's SMEM port and Atlas array; thin gaps separate the groups."""
    df = trace.resources
    by_busy = df.groupby("resource")["busy"].sum().sort_values(ascending=False).index.tolist()
    rows = _heatmap_rows(by_busy, links)
    grid = df[df["resource"].isin([r for r in rows if r])].pivot_table(
        index="resource", columns="bin_start", values="busy", fill_value=0.0
    )
    data = np.full((len(rows), grid.shape[1]), np.nan)
    for k, r in enumerate(rows):
        if r is not None and r in grid.index:
            data[k] = grid.loc[r].to_numpy()
    bin_ = trace.bin
    t = grid.columns.to_numpy() / 1e6
    cmap = plt.get_cmap(style.CMAP).copy()
    cmap.set_bad("white")
    im = ax.imshow(
        np.ma.masked_invalid(data),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=(t[0], t[-1] + bin_ / 1e6, len(rows) - 0.5, -0.5),
    )
    ticks = [k for k, r in enumerate(rows) if r is not None]
    ax.set_yticks(ticks)
    ax.set_yticklabels([rows[k] for k in ticks])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Time (M cycles)")
    return im


def mark_window(ax: plt.Axes, window: tuple[int, int]) -> None:
    """A light band with a thin outline over the time `window` (cycles)."""
    ax.axvspan(
        window[0] / 1e6,
        window[1] / 1e6,
        facecolor="white",
        alpha=0.25,
        edgecolor="white",
        linewidth=0.6,
    )


def gantt(ax: plt.Axes, trace: Trace, node_prefix: str = "", window: tuple[int, int] | None = None) -> None:
    """Chunk intervals of the tasks of each PE (rows), colored by task
    kind. `node_prefix` and `window` (cycles) restrict what is drawn."""
    tasks = trace.tasks
    if node_prefix:
        tasks = tasks[tasks["node"].astype(str).str.startswith(node_prefix)]
    ch = trace.chunks.merge(tasks[["gid", "pe", "kind"]], on="gid")
    if window:
        ch = ch[(ch["end"] > window[0]) & (ch["start"] < window[1])]
    pes = sorted(ch["pe"].dropna().unique())
    row = {p: i for i, p in enumerate(pes)}
    for kind, g in ch.groupby("kind"):
        for pe, part in g.groupby("pe"):
            spans = list(zip(part["start"] / 1e6, (part["end"] - part["start"]) / 1e6))
            ax.broken_barh(spans, (row[pe] - 0.4, 0.8), facecolors=KIND_COLORS.get(kind, "#7f7f7f"), linewidth=0)
    ax.set_yticks(range(len(pes)))
    ax.set_yticklabels(pes)
    ax.set_xlabel("Time (M cycles)")
    kinds = [k for k in KIND_COLORS if k in set(ch["kind"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLORS[k]) for k in kinds]
    ax.legend(handles, kinds, ncol=len(kinds), loc="upper right", handlelength=0.8, borderaxespad=0.1)
    ax.set_ylim(-0.6, len(pes) - 0.4 + 0.8)
    if window:
        ax.set_xlim(window[0] / 1e6, window[1] / 1e6)
