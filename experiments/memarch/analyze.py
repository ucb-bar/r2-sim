"""Analysis of a memarch sweep: per-workload Atlas utilization, winners,
one-knob sensitivity around each winner, and the Pareto plots.

    python3 analyze.py out/sweep2.jsonl          # tables on stdout, plots in figs/
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
WORKLOADS = ["attn", "square", "skinny", "kheavy", "transpose", "gather", "gather32"]
NAMES = {"attn": "Attention + softmax", "square": "GEMM 1024³", "skinny": "GEMM 64×4096×4096",
         "kheavy": "GEMM 256×256×8192", "transpose": "GEMM + Muon transpose",
         "gather": "GEMM + Muon gather (4 B)", "gather32": "GEMM + Muon gather (32 B)"}
TARGET = 0.95
# Reference palette (dataviz skill), categorical slots 1-3; text tokens.
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
OPTION = {"A": "A: TMEM-fed", "B": "B: unified smem", "C": "C: Blackwell-like"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KNOB_KEYS = ["banks", "subbanks", "width", "radix", "swizzle", "kc", "qr", "atlas_port", "n_acc", "partitions",
             "tmem_cols"]


def load(path: Path) -> list[dict]:
    out = []
    for line in open(path):
        r = json.loads(line)
        u = {w: (r["runs"].get(w) or {}).get("util") for w in WORKLOADS}
        if any(v is None for v in u.values()):
            continue
        out.append({"d": r["design"], "label": r["label"], "area": r["area"]["total"],
                    "macros": r["area"]["macro_count"], "u": u, "min": min(u.values()),
                    "culprits": {w: (r["runs"][w].get("culprits") or [])[:3] for w in WORKLOADS}})
    return out


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left frontier: the best utilization reachable at each area or less."""
    out, best = [], -1.0
    for a, u in sorted(points):
        if u > best:
            out.append((a, u))
            best = u
    return out


def cheapest(rows, key=lambda r: r["min"]):
    ok = [r for r in rows if key(r) >= TARGET]
    return min(ok, key=lambda r: r["area"]) if ok else None


def design_key(d: dict, skip: str) -> tuple:
    return tuple((k, d[k]) for k in sorted(d) if k != skip)


def sensitivity(rows: list[dict], win: dict) -> list[tuple[str, object, dict | None]]:
    """Every design that differs from the winner in exactly one knob."""
    out = []
    for knob in KNOB_KEYS:
        base = design_key(win["d"], knob)
        for r in rows:
            if r["d"][knob] != win["d"][knob] and design_key(r["d"], knob) == base:
                out.append((knob, r["d"][knob], r))
    return out


def fmt_u(u: float) -> str:
    return f"{100 * u:5.1f}"


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def panel(ax, groups: dict, util_of, title: str, label_lines: bool):
    """One Pareto panel: faint points per option, the frontier as a step line,
    and the 95% target."""
    style(ax)
    for opt in "ABC":
        rows = groups.get(opt, [])
        if not rows:
            continue
        xs = [r["area"] / 1e3 for r in rows]
        ys = [100 * util_of(r) for r in rows]
        ax.scatter(xs, ys, s=4, color=COLOR[opt], alpha=0.12, linewidths=0, rasterized=True)
        f = frontier(list(zip(xs, ys)))
        fx = [p[0] for p in f] + [max(xs)]
        fy = [p[1] for p in f] + [f[-1][1]]
        win = cheapest(rows, util_of)
        tag = f" (cheapest ≥95%: {win['area'] / 1e3:.0f}k µm²)" if win else " (none ≥95%)"
        ax.step(fx, fy, where="post", color=COLOR[opt], linewidth=2, label=OPTION[opt] + (tag if label_lines else ""))
        if win:
            ax.plot(win["area"] / 1e3, 100 * util_of(win), "o", ms=8, color=COLOR[opt],
                    markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.axhline(100 * TARGET, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.99, 100 * TARGET + 1, "95%", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(0, 102)
    ax.set_title(title, fontsize=11, color=INK, loc="left")


def plots(rows: list[dict], out: Path):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["d"]["dtype"]][r["d"]["option"]].append(r)
    xmax = max(r["area"] for r in rows) / 1e3

    # 1. The headline: worst workload per design, bf16 and fp8.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, dt in zip(axes, ["bf16", "fp8"]):
        p = {"bf16": 2048, "fp8": 4096}[dt]
        panel(ax, by[dt], lambda r: r["min"], f"{dt}, {p} MAC/cycle: worst of 7 workloads", True)
        ax.set_xlim(200, xmax * 1.02)
        ax.set_xlabel("Area (k µm²)", color=INK2, fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].set_ylabel("Atlas utilization (%)", color=INK2, fontsize=10)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out / f"pareto_min.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    # 2. Small multiples: every workload, bf16 and fp8.
    fig, axes = plt.subplots(2, len(WORKLOADS), figsize=(3.0 * len(WORKLOADS), 6.4), sharex=True, sharey=True)
    for i, dt in enumerate(["bf16", "fp8"]):
        for j, wl in enumerate(WORKLOADS):
            ax = axes[i][j]
            panel(ax, by[dt], lambda r, wl=wl: r["u"][wl], f"{NAMES[wl]}\n{dt}", False)
            ax.title.set_fontsize(9)
            ax.set_xlim(200, xmax * 1.02)
            if i == 1:
                ax.set_xlabel("Area (k µm²)", color=INK2, fontsize=9)
        axes[i][0].set_ylabel("Atlas util (%)", color=INK2, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out / f"pareto_workloads.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    path = Path((argv or sys.argv[1:] or ["out/sweep2.jsonl"])[0])
    rows = load(path)
    figs = HERE / "figs"
    figs.mkdir(exist_ok=True)
    print(f"{len(rows)} designs with every workload simulated")

    # Per-design CSV.
    with open(path.with_name(path.stem + "_util.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["option", "dtype", "label", "area_um2", "macros", *WORKLOADS, "min"])
        for r in rows:
            w.writerow([r["d"]["option"], r["d"]["dtype"], r["label"], round(r["area"]), r["macros"],
                        *[round(r["u"][x], 4) for x in WORKLOADS], round(r["min"], 4)])

    groups = defaultdict(list)
    for r in rows:
        groups[(r["d"]["dtype"], r["d"]["option"])].append(r)

    print("\n== Cheapest design per option reaching 95% on EVERY workload (utilization %)")
    print(f"{'dtype':5} {'opt':3} {'area':>8} {'mac':>4} " + " ".join(f"{w[:7]:>7}" for w in WORKLOADS) + "  config")
    winners = {}
    for (dt, opt) in sorted(groups):
        win = cheapest(groups[(dt, opt)])
        winners[(dt, opt)] = win
        if win is None:
            best = max(groups[(dt, opt)], key=lambda r: r["min"])
            print(f"{dt:5} {opt:3} {'none':>8}      best worst-workload {fmt_u(best['min'])}%: {best['label']}")
            continue
        print(f"{dt:5} {opt:3} {win['area'] / 1e3:7.0f}k {win['macros']:4} "
              + " ".join(f"{fmt_u(win['u'][w]):>7}" for w in WORKLOADS) + "  " + win["label"])

    print("\n== Cheapest design reaching 95% on each workload alone (area k µm²)")
    print(f"{'dtype':5} {'opt':3} " + " ".join(f"{w[:9]:>9}" for w in WORKLOADS))
    for (dt, opt) in sorted(groups):
        cells = []
        for wl in WORKLOADS:
            c = cheapest(groups[(dt, opt)], lambda r, wl=wl: r["u"][wl])
            cells.append(f"{c['area'] / 1e3:8.0f}k" if c else f"{'-':>9}")
        print(f"{dt:5} {opt:3} " + " ".join(cells))

    print("\n== One knob changed from each winner (area k µm², utilization % per workload)")
    for (dt, opt), win in sorted(winners.items()):
        if win is None:
            continue
        print(f"\n{dt} {opt}: {win['label']}")
        print(f"  {'knob':10} {'value':>6} {'area':>7} " + " ".join(f"{w[:7]:>7}" for w in WORKLOADS))
        print(f"  {'(winner)':10} {'':>6} {win['area'] / 1e3:6.0f}k " + " ".join(f"{fmt_u(win['u'][w]):>7}" for w in WORKLOADS))
        for knob, val, r in sensitivity(groups[(dt, opt)], win):
            v = "full" if val is None else val
            print(f"  {knob:10} {str(v):>6} {r['area'] / 1e3:6.0f}k " + " ".join(f"{fmt_u(r['u'][w]):>7}" for w in WORKLOADS))

    print("\n== Top stalls of each winner (share of the run)")
    for (dt, opt), win in sorted(winners.items()):
        if win is None:
            continue
        print(f"{dt} {opt}:")
        for wl in WORKLOADS:
            cs = ", ".join(f"{c[0].replace('root.sm0.', '')} {100 * c[1]:.0f}%" for c in win["culprits"][wl])
            print(f"  {wl:10} {cs}")

    plots(rows, figs)
    print(f"\nplots: {figs}/pareto_min.svg, {figs}/pareto_workloads.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
