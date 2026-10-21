"""Draw the article's figures from the outputs of ``labour_share.py``.

Run after ``labour_share.py``::

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import labour_share as ls

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
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
})

ADJUSTED = [n for n in ls.NUMERATORS if n != "raw"]


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG at article resolution."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig1_shares() -> None:
    """Compensation-only share against the five adjusted shares, two denominators."""
    shares = pd.read_csv(RESULTS / "shares_long.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    panels = (
        ("market_prices", "GDP at market prices"),
        ("factor_cost_ex_government", "Factor cost, general government excluded"),
    )
    for ax, (den, title) in zip(axes, panels):
        sub = shares[shares["denominator"] == den]
        wide = sub.pivot(index="year", columns="numerator", values="share")
        years = wide.index.to_numpy()
        low, high = wide[ADJUSTED].min(axis=1), wide[ADJUSTED].max(axis=1)
        ax.fill_between(years, low, high, color=ORANGE, alpha=0.18, linewidth=0)
        ax.plot(years, wide["employment"], color=ORANGE)
        ax.plot(years, wide["raw"], color=BLUE)
        ax.text(years[-1] + 0.6, wide["employment"].iloc[-1], f"{wide['employment'].iloc[-1]:.1f}",
                color=INK_2, va="center", fontsize=9)
        ax.text(years[-1] + 0.6, wide["raw"].iloc[-1], f"{wide['raw'].iloc[-1]:.1f}",
                color=INK_2, va="center", fontsize=9)
        ax.text(years[0], wide["employment"].iloc[0] + 1.2, "self-employed counted\n(five rules, shaded)",
                color=INK_2, fontsize=8.5, va="bottom")
        ax.text(years[0], wide["raw"].iloc[0] - 1.2, "employees only", color=INK_2, fontsize=8.5, va="top")
        ax.set_title(title)
        ax.set_xlim(years[0] - 0.5, years[-1] + 2.5)
        ax.set_xticks([1995, 2000, 2005, 2010, 2015, 2020, 2024])
    axes[0].set_ylabel("Labour share, %")
    axes[0].set_ylim(42, 71)
    handles = [
        plt.Line2D([], [], color=BLUE, label="Compensation of employees only"),
        plt.Line2D([], [], color=ORANGE, label="Self-employed paid the average employee wage"),
        Patch(color=ORANGE, alpha=0.18, label="Range of the five rules for self-employed income"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig1_labour_share_definitions")


def fig2_alpha() -> None:
    """1995-2024 change as a function of the labour fraction of mixed income."""
    grid = pd.read_csv(RESULTS / "variants.csv").set_index(["numerator", "denominator"])
    pt = ls.load_portugal()
    c = ls.components(pt)
    implied = [c.self_employed_labour[r] / c.mixed_income for r in ("employment", "hours", "mixed_rest")]
    lo = min(float(s.min()) for s in implied)
    hi = max(1.0, max(float(s.max()) for s in implied))
    alpha = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(lo, hi, color=GRID, alpha=0.7, linewidth=0)
    ax.text((lo + hi) / 2, 3.95, "fractions implied by\nthe adjustment rules", ha="center", va="top",
            color=INK_2, fontsize=8.5)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    labels = {
        "market_prices": "GDP at market prices",
        "factor_cost": "Factor cost",
        "factor_cost_ex_government": "Factor cost, government excluded",
    }
    for den, colour in zip(ls.DENOMINATORS, (BLUE, ORANGE, AQUA)):
        d_raw = grid.loc[("raw", den), "change"]
        d_mi = grid.loc[("mixed_all", den), "change"] - d_raw
        change = d_raw + alpha * d_mi
        ax.plot(alpha, change, color=colour, label=labels[den])
        star = -d_raw / d_mi
        ax.plot([star], [0.0], marker="o", markersize=8, color=colour, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=5)
        ax.annotate(f"{star:.2f}", (star, 0.0), xytext=(-4, -14), textcoords="offset points",
                    ha="right", color=INK_2, fontsize=9)
        ax.text(1.015, change[-1], labels[den], color=INK_2, va="center", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-4.6, 4.2)
    ax.set_xlabel("Share of self-employed (mixed) income counted as labour")
    ax.set_ylabel("Change in labour share 1995 to 2024, pp")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_break_even_fraction")


def fig3_break_scan() -> None:
    """F statistic for a break at each year, with three 5% thresholds."""
    prof = pd.read_csv(RESULTS / "break_profile_raw_market.csv")
    n = ls.LAST_YEAR - ls.FIRST_YEAR + 1
    fixed = {"level": ls.stats.f.ppf(0.95, 1, n - 3), "level_slope": ls.stats.f.ppf(0.95, 2, n - 4)}
    titles = {"level": "Trend plus a level shift", "level_slope": "Trend plus a level shift and a slope change"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, spec in zip(axes, ("level", "level_slope")):
        sub = prof[prof["spec"] == spec]
        ax.plot(sub["year"], sub["f"], color=INK, linewidth=2.0, marker="o", markersize=4)
        peak = sub.loc[sub["f"].idxmax()]
        ax.annotate(f"{int(peak['year'])}", (peak["year"], peak["f"]), xytext=(0, 7),
                    textcoords="offset points", ha="center", color=INK, fontsize=9, fontweight="bold")
        lines = (
            (fixed[spec], MUTED, "5% threshold if the year were fixed in advance"),
            (sub["crit95_alt"].iloc[0], BLUE, "5% threshold after searching, low persistence"),
            (sub["crit95_null"].iloc[0], ORANGE, "5% threshold after searching, high persistence"),
        )
        for value, colour, label in lines:
            ax.axhline(value, color=colour, linewidth=1.5, linestyle=(0, (4, 3)), label=label)
        ax.set_title(titles[spec])
        ax.set_xticks([1999, 2004, 2008, 2012, 2016, 2020])
    axes[0].set_ylabel("F statistic for a break in that year")
    axes[0].set_ylim(0, 58)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.insert(0, plt.Line2D([], [], color=INK, marker="o", markersize=4, label="F statistic, compensation share of GDP"))
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig3_break_scan")


def fig4_government() -> None:
    """Decomposition of the fall in the factor-cost labour share around 2012."""
    pt = ls.load_portugal()
    c = ls.components(pt)
    windows = [(2010, 2012), (2010, 2013)]
    parts = [ls.government_shift_share(pt, c, a, b) for a, b in windows]
    keys = (
        ("within_government", BLUE, "Inside government"),
        ("between", ORANGE, "Smaller government"),
        ("within_rest", AQUA, "Rest of the economy"),
    )
    fig, ax = plt.subplots(figsize=(8, 2.9))
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for row, part in enumerate(parts):
        left = 0.0
        for key, colour, _ in keys:
            width = part[key]
            ax.barh(row, width, left=left, color=colour, height=0.5, edgecolor=SURFACE, linewidth=2)
            if abs(width) > 0.3:
                ax.text(left + width / 2, row, f"{width:.2f}", ha="center", va="center",
                        color="white" if colour == BLUE else INK, fontsize=9, fontweight="bold")
            left += width
        ax.text(left - 0.05, row, f"total {part['total']:.2f} pp", ha="right", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(windows)), [f"{a} to {b}" for a, b in windows])
    ax.invert_yaxis()
    ax.set_xlim(-3.4, 0.1)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    ax.set_xlabel("Contribution to the change in the factor-cost labour share, pp")
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=colour, label=label) for _, colour, label in keys]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig4_government_2012")


def main() -> None:
    """Draw every figure."""
    fig1_shares()
    fig2_alpha()
    fig3_break_scan()
    fig4_government()


if __name__ == "__main__":
    main()
