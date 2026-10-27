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

# Continuous data (utilization heatmaps).
CMAP = "viridis"

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
