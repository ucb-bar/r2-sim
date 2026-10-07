"""Plots of r2sim runs. Each function draws into one matplotlib axis and
uses the figure style of `style.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import style
from .data import Trace, acc_buffer, run_label

# One color per task kind in Gantt charts (Okabe-Ito, fixed).
KIND_COLORS = {
    "dma_w": style.OKABE_ITO["orange"],
    "dma_act": style.OKABE_ITO["sky_blue"],
    "qkv": style.OKABE_ITO["blue"],
    "attn": style.OKABE_ITO["bluish_green"],
    "wo": style.OKABE_ITO["blue"],
    "ffn_gate": style.OKABE_ITO["vermillion"],
    "ffn_up": style.OKABE_ITO["vermillion"],
    "ffn_down": style.OKABE_ITO["purple"],
    "vec": style.OKABE_ITO["yellow"],
    "router": style.OKABE_ITO["black"],
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


def _phase_labels(ph: pd.DataFrame) -> list[str]:
    """Phase names, with the stage only when there are several stages."""
    if ph["stage"].nunique() == 1:
        return list(ph["name"])
    return [f"{a} {b}" for a, b in zip(ph["stage"], ph["name"])]


def culprit_shares(ax: plt.Axes, trace: Trace, top: int = 5) -> None:
    """Per phase, the share of chunk time for which each resource class was
    the culprit (stacked bars; the `top` classes overall, the rest as
    other)."""
    ch = trace.chunks.dropna(subset=["culprit"])
    ph = trace.phases.reset_index(drop=True)
    rows = []
    for _, p in ph.iterrows():
        m = ch[(ch["start"] >= p["start"]) & (ch["start"] < p["end"])]
        d = (m["end"] - m["start"]).groupby(m["culprit"].map(resource_class)).sum()
        rows.append(d / max(d.sum(), 1))
    shares = pd.DataFrame(rows).fillna(0.0)
    keep = shares.sum().nlargest(top).index
    rest = shares.drop(columns=keep).sum(axis=1)
    shares = shares[list(keep)]
    if rest.max() > 1e-3:
        shares["rest"] = rest
    x = np.arange(len(shares))
    base = np.zeros(len(shares))
    colors = list(style.OKABE_ITO.values())[1:] + ["#bbbbbb"]
    for c, color in zip(shares.columns, colors):
        ax.bar(x, shares[c], bottom=base, color=color, width=0.8, label=c, linewidth=0)
        base += shares[c].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(_phase_labels(ph))
    ax.set_ylabel("Culprit share of chunk time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))


def phase_roofline(ax: plt.Axes, trace: Trace) -> None:
    """Per GEMM phase: measured cycles against the model's estimate and the
    roofline (log scale)."""
    ph = trace.phases.dropna(subset=["roofline"]).reset_index(drop=True)
    x = np.arange(len(ph))
    meas = (ph["end"] - ph["start"]) / 1e3
    ax.bar(x, meas, width=0.7, color=style.OKABE_ITO["blue"], linewidth=0)
    ax.plot(x, ph["estimate"] / 1e3, "_", color=style.OKABE_ITO["orange"], markersize=6, markeredgewidth=1.0)
    ax.plot(x, ph["roofline"] / 1e3, "_", color=style.OKABE_ITO["black"], markersize=6, markeredgewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(_phase_labels(ph))
    ax.set_ylabel("Phase time (k cycles)")
    # Direct labels on the last phase instead of a legend.
    last = len(ph) - 1
    for y, text, color in (
        (meas.iloc[last], "measured", style.OKABE_ITO["blue"]),
        (ph["estimate"].iloc[last] / 1e3, "estimate", style.OKABE_ITO["orange"]),
        (ph["roofline"].iloc[last] / 1e3, "roofline", style.OKABE_ITO["black"]),
    ):
        ax.annotate(text, (last + 0.45, y), fontsize=5, va="center", color="black")


def stage_bars(ax: plt.Axes, runs: list[dict]) -> None:
    """The headline: per run, the forward-pass stages as stacked bars (M
    cycles), with the three bounds of the total as markers."""
    y = np.arange(len(runs))
    for i, run in enumerate(runs):
        f = run["forward"]
        config, weight = run_label(run)
        color = style.config_color(config, weight)
        left = 0.0
        for st, (_, alpha) in zip(f["stages"], style.STAGE_ALPHA):
            w = st["cycles"] / 1e6
            ax.barh(i, w, left=left, height=0.55, color=color, alpha=alpha, linewidth=0)
            if i == 0:
                # Direct labels of the three stages, above the first bar.
                ax.annotate(st["name"], (left + w / 2, -0.45), fontsize=5, ha="center", va="bottom")
            left += w
        # The bounds of the total on a thin row just below the bar.
        tot = f["total"]
        bounds = [(tot[style.BOUND_KEYS[k]], m) for k, m, _ in style.bound_styles(horizontal=True)]
        ax.hlines(i + 0.4, 0, max(v for v, _ in bounds) / 1e6, color="#cccccc", linewidth=0.3)
        for v, marker in bounds:
            ax.plot(
                v / 1e6,
                i + 0.4,
                marker,
                color="black",
                markersize=2.5,
                markeredgewidth=0.5,
                fillstyle="none",
            )
    labels = []
    for run in runs:
        config, weight = run_label(run)
        acc = acc_buffer(run)
        labels.append(f"{config.removesuffix('.yaml')} {weight}" + (f" acc {acc >> 20} MiB" if acc else ""))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("One inference step (M cycles = ms at 1 GHz)")
    ax.axvline(100, color="#999999", linewidth=0.4, linestyle="--")
    names = [n for _, _, n in style.bound_styles(horizontal=True)]
    ax.legend(style.bound_handles(horizontal=True), names, loc="lower right", handletextpad=0.3)
    ax.annotate("100 ms control step", (100, -0.8), fontsize=5, ha="center", va="bottom", annotation_clip=False)
