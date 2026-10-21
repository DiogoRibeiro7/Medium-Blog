"""Draw the follow-up article's figures from the outputs of ``industries.py``.

Run after ``industries.py``::

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import industries as ind

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
})


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG at article resolution."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig1_imputation() -> None:
    """Portugal's adjusted share with the self-employed imputed economy-wide or by industry."""
    data = pd.read_csv(RESULTS / "imputation_levels.csv")
    data = data[data["geo"] == "PT"].pivot(index="year", columns="rule", values="share")
    years = data.index.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    lines = (
        ("economy_per_person", ORANGE, "Economy-wide pay per employee"),
        ("a64_per_person", BLUE, "Own industry's pay per employee (61 industries)"),
        ("a10_per_hour", AQUA, "Own branch's pay per employee hour (10 branches)"),
        ("raw", MUTED, "Compensation of employees only"),
    )
    for key, colour, label in lines:
        ax.plot(years, data[key], color=colour, label=label)
    for end, x, ha in ((-1, years[-1] + 0.6, "left"), (0, years[0] - 0.6, "right")):
        values = {key: data[key].iloc[end] for key, _, _ in lines}
        placed: list[float] = []
        for key in sorted(values, key=values.get, reverse=True):
            y = values[key]
            if placed and placed[-1] - y < 0.9:
                y = placed[-1] - 0.9
            placed.append(y)
            ax.text(x, y, f"{values[key]:.1f}", color=INK_2, va="center", ha=ha, fontsize=9)
    ax.set_xlim(years[0] - 3, years[-1] + 3)
    ax.set_xticks([1995, 2000, 2005, 2010, 2015, 2020, 2024])
    ax.set_ylabel("Labour share of factor-cost value added, %")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig1_imputation_level")


def fig2_granularity() -> None:
    """Within and between parts of 2016-2024 at three levels of industry detail."""
    grid = pd.read_csv(RESULTS / "shift_share_grid.csv")
    grid = grid[(grid["start"] == 2016) & (grid["end"] == 2024) & (grid["numerator"] == "raw")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    titles = {"economy": "Whole economy", "market": "Market economy, excluding housing"}
    labels = {"a10": "10 branches", "a21": "21 sections", "a64": "61 industries"}
    for ax, scope in zip(axes, ("economy", "market")):
        sub = grid[grid["scope"] == scope].set_index("level").loc[list(labels)]
        x = np.arange(len(sub))
        ax.bar(x, sub["within"], color=BLUE, width=0.55, edgecolor=SURFACE, linewidth=2)
        ax.bar(x, sub["between"], bottom=sub["within"], color=ORANGE, width=0.55, edgecolor=SURFACE, linewidth=2)
        for xi, (w, b) in enumerate(zip(sub["within"], sub["between"])):
            ax.text(xi, w / 2, f"{w:.2f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            ax.text(xi, w + b / 2, f"{b:.2f}", ha="center", va="center", color=INK, fontsize=9, fontweight="bold")
            ax.text(xi, w + b + 0.08, f"{100 * w / (w + b):.0f}% within", ha="center", va="bottom", color=INK_2, fontsize=8.5)
        ax.set_xticks(x, [labels[k] for k in sub.index])
        ax.set_title(titles[scope])
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Change in labour share 2016 to 2024, pp")
    axes[0].set_ylim(0, 5.6)
    handles = [Patch(color=BLUE, label="Within industries"), Patch(color=ORANGE, label="Between industries")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_within_between_by_detail")


def fig3_sections() -> None:
    """Within and between contributions of each section to 2016-2024."""
    sec = pd.read_csv(RESULTS / "contributions_2016_2024_sections.csv", index_col=0)
    sec = sec.sort_values("total")
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(sec))
    h = 0.38
    ax.barh(y + h / 2, sec["within"], height=h, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.barh(y - h / 2, sec["between"], height=h, color=ORANGE, edgecolor=SURFACE, linewidth=1)
    ax.scatter(sec["total"], y, color=INK, s=22, zorder=4)
    ax.set_yticks(y, [f"{g}  {ind.SECTION_NAMES[g]}" for g in sec.index], fontsize=8.5)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Contribution to the change in the labour share 2016 to 2024, pp")
    handles = [Patch(color=BLUE, label="Within the section"), Patch(color=ORANGE, label="Change in the section's weight"),
               plt.Line2D([], [], color=INK, marker="o", linestyle="", markersize=5, label="Total")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig3_sections_2016_2024")


def fig4_pay() -> None:
    """Within-industry change against 2016 pay per employee, 61 industries."""
    data = pd.read_csv(RESULTS / "pay_and_share_2016_2024.csv")
    data = data[data["geo"] == "PT"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.grid(True, axis="both", color=GRID, linewidth=0.8)
    size = 600 * data["va_start"] / data["va_start"].max()
    ax.scatter(data["pay_start"], data["share_change"], s=size.clip(lower=12), color=BLUE, alpha=0.55,
               edgecolor=SURFACE, linewidth=1.5)
    ax.axhline(0, color=AXIS, linewidth=1)
    names = {"G46": ("wholesale", (10, 4)), "G47": ("retail", (10, 6)), "K64": ("banking", (-12, 4)),
             "J62_J63": ("IT services", (8, 4)), "C13-C15": ("textiles, clothing", (-10, 10)),
             "D": ("electricity", (-14, 10)), "I": ("hotels, restaurants", (-30, -26)),
             "O": ("public administration", (14, -4)), "L": ("real estate", (20, -24))}
    for code, (label, offset) in names.items():
        row = data[data["code"] == code]
        if row.empty:
            continue
        ax.annotate(label, (row["pay_start"].iloc[0], row["share_change"].iloc[0]), xytext=offset,
                    textcoords="offset points", color=INK_2, fontsize=8.5,
                    ha="right" if offset[0] < 0 else "left", va="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8, "shrinkA": 2, "shrinkB": 4})
    ax.set_xscale("log")
    ticks = [10, 15, 20, 30, 40, 50]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Compensation per employee in 2016, thousand euro (log scale)")
    ax.set_ylabel("Change in the industry's labour share 2016 to 2024, pp")
    ax.text(0.01, 0.98, "Bubble area: industry value added in 2016", transform=ax.transAxes, ha="left",
            va="top", color=INK_2, fontsize=8)
    fig.tight_layout()
    save(fig, "fig4_pay_and_share")


def main() -> None:
    """Draw every figure."""
    fig1_imputation()
    fig2_granularity()
    fig3_sections()
    fig4_pay()


if __name__ == "__main__":
    main()
