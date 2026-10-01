"""One visual language for every r2sim paper figure.

Import this module before plotting: it sets the matplotlib defaults. Sizes
are final print sizes (Nature-style): single column 89 mm, double column
183 mm, text 5 to 7 pt, editable text in the PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE_COL = 89 * MM
DOUBLE_COL = 183 * MM

# Okabe-Ito qualitative palette.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# One fixed color per (configuration file, weight type), in every figure.
CONFIG_COLORS = {
    ("2p.yaml", "bf16"): OKABE_ITO["black"],
    ("2p.yaml", "mxfp4"): OKABE_ITO["blue"],
    ("2p-cmem64.yaml", "mxfp4"): OKABE_ITO["bluish_green"],
    ("2p-cmem-large.yaml", "mxfp4"): OKABE_ITO["orange"],
    ("2p-smem2m.yaml", "mxfp4"): OKABE_ITO["sky_blue"],
    ("2p-smem512.yaml", "mxfp4"): OKABE_ITO["vermillion"],
}

# Continuous data (utilization heatmaps).
CMAP = "viridis"

# Shades of the forward-pass stages in stacked bars.
STAGE_ALPHA = (("vision", 0.45), ("vlm", 0.7), ("action", 1.0))

# JSON keys of the three lower bounds of a stage, by short name.
BOUND_KEYS = {"own": "own_traffic_bound", "roofline": "roofline", "once": "weights_once_bound"}


def bound_styles(horizontal: bool) -> tuple[tuple[str, str, str], ...]:
    """(short name, marker, label) of the three bounds. The own-traffic
    tick crosses the value axis: `horizontal` is True when values run
    along x."""
    return (
        ("own", "|" if horizontal else "_", "own-traffic bound"),
        ("roofline", "x", "roofline"),
        ("once", "o", "weights-once bound"),
    )


def bound_handles(horizontal: bool) -> list[plt.Line2D]:
    """Legend handles of the three bounds."""
    return [
        plt.Line2D([], [], linestyle="none", marker=m, color="black", markersize=3, markeredgewidth=0.6,
                   fillstyle="none")
        for _, m, _ in bound_styles(horizontal)
    ]


def _font_family() -> list[str]:
    """Arial or Helvetica when installed, else the metric-compatible
    Liberation Sans, else DejaVu Sans or the default sans-serif."""
    have = {f.name for f in fm.fontManager.ttflist}
    wanted = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    return [f for f in wanted if f in have] or ["sans-serif"]


def apply() -> None:
    """Sets the matplotlib defaults of the figure style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _font_family(),
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 7,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=list(OKABE_ITO.values())[1:]),
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def config_color(config: str, weight: str) -> str:
    """The fixed color of a configuration, gray when it has none."""
    return CONFIG_COLORS.get((config, weight), "#7f7f7f")


def panel_label(ax: plt.Axes, letter: str) -> None:
    """A lowercase bold panel letter at the top left of `ax`."""
    ax.text(
        -0.12,
        1.04,
        letter.lower(),
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(fig: plt.Figure, path: Path) -> list[Path]:
    """Writes `path` as a vector PDF and a 300 dpi PNG; returns both."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".pdf"), path.with_suffix(".png")]
    fig.savefig(out[0], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out[1], bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


apply()
