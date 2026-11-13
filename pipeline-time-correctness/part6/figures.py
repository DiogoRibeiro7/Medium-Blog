"""Figures for Part 6, drawn from the two tables oos_numbers.py writes to results/.

Run: python figures.py  (after oos_numbers.py; writes figures/*.png)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def short(v: str) -> str:
    return (v.replace(", last-week fallback", " + last-week").replace(" + holidays + last-week", " + hol. + last-week")
             .replace("graded check", "graded"))


def fig1_in_against_out() -> None:
    df = pd.read_csv(HERE / "results" / "versions.csv").sort_values("out", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(df))
    for i, r in df.iterrows():
        ax.plot([100 * r["in"], 100 * r["out"]], [i, i], color=AXIS, linewidth=2, zorder=1)
    ax.scatter(100 * df["in"], y, s=42, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.6, zorder=2,
               label="2021-2023, in sample (822 mornings)")
    ax.scatter(100 * df["out"], y, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3,
               label="2024-2026, out of sample")
    ax.set_yticks(y, [short(v) for v in df.version], fontsize=8.5)
    ax.set_xlabel("mean absolute error of the nowcast against the final count (%)")
    ax.set_xscale("log")
    ticks = [1, 2, 3, 5, 10, 20, 30]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(1, max(35, 110 * df[["in", "out"]].max().max()))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Each version's error in sample and out of sample, ordered by the out-of-sample error")
    save(fig, "fig1_in_against_out.png")


def fig2_by_month() -> None:
    m = pd.read_csv(HERE / "results" / "oos_monthly.csv")
    m["month"] = pd.PeriodIndex(m.month, freq="M").to_timestamp()
    versions = list(dict.fromkeys(m.version))
    colors = [MUTED, BLUE, ORANGE, AQUA]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for v, c in zip(versions, colors):
        g = m[m.version == v]
        ax.plot(g.month, 100 * g.mae, color=c, marker="o", markersize=3.5, label=short(v))
    ax.set_ylabel("mean absolute error (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=8.5, ncols=2)
    ax.set_title("Out of sample, month by month: the nowcast's error and the size of the counts")
    g = m[m.version == versions[0]]
    bx.bar(g.month, g.final_7d_median, width=20, color=MUTED)
    bx.set_ylabel("median final\n7-day count")
    bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bx.grid(axis="x", visible=False)
    save(fig, "fig2_by_month.png")


if __name__ == "__main__":
    fig1_in_against_out()
    fig2_by_month()
