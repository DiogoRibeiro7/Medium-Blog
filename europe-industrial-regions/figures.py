"""Figures for the europe-industrial-regions article; reads data/raw and results/.

    python figures.py     # figures/fig1_*.png ... fig5_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import regions as R

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BINS = [0, 15, 20, 25, 30, 40, 50, 100]
# diverging pair for income relative to the 16-country level: red below, blue above, grey midpoint;
# the red arm matches the blue steps 200/400/600 in OKLCH lightness
DIV_LOW = ["#892b2a", "#d75853", "#f1aea8"]
DIV_HIGH = ["#9ec5f4", "#3987e5", "#184f95"]
DIV_BINS = [0, 60, 80, 95, 105, 125, 150, 1000]
SOURCE = "Data: Rosés-Wolf database on regional GDP, version 7 (2025). Analysis: Diogo Ribeiro."

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
})


def finish(fig, name: str, note: str = "") -> None:
    fig.text(0.01, 0.01, (note + "\n" if note else "") + SOURCE, fontsize=7.5, color=INK2,
             ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1() -> None:
    """The share of industry in employment by region, 1900, 1960 and 2015, with national borders,
    and the 1900 clusters that straddle a border outlined."""
    df = R.load_panel()
    g = R.load_map(pd.Series(sorted(df.code.unique())))
    g["country"] = df.drop_duplicates("code").set_index("code").country.reindex(g.index)
    nations = g.dissolve(by="country")
    core = pd.read_csv(RES / "core_1900.csv")
    multi = core[core.countries.str.contains(r"\+", regex=True)]
    cmap = ListedColormap(RAMP)
    norm = BoundaryNorm(BINS, cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2))
    for ax, year in zip(axes, [1900, 1960, 2015]):
        v = df[df.year == year].set_index("code").ind.reindex(g.index) * 100
        gg = g.assign(v=v)
        gg[gg.v.isna()].plot(ax=ax, color="#efeeea", edgecolor="white", linewidth=0.25)
        gg[gg.v.notna()].plot(ax=ax, column="v", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.25)
        nations.boundary.plot(ax=ax, color=INK, linewidth=0.9)
        if year == 1900:
            g.loc[multi.code].dissolve(by=multi.set_index("code").cluster.reindex(multi.code).values) \
                .boundary.plot(ax=ax, color=ORANGE, linewidth=2.2)
        ax.set_title(str(year), loc="left")
        ax.set_xlim(2.55e6, 5.55e6)
        ax.set_ylim(1.35e6, 5.25e6)
        ax.set_axis_off()
    handles = [Patch(facecolor=RAMP[i], edgecolor="none",
                     label=f"{BINS[i]}-{BINS[i + 1]}%" if BINS[i + 1] < 100 else f"{BINS[i]}% or more")
               for i in range(len(RAMP))]
    handles += [Patch(facecolor="#efeeea", edgecolor="none", label="no figure"),
                Line2D([0], [0], color=INK, linewidth=1, label="national border"),
                Line2D([0], [0], color=ORANGE, linewidth=2.2,
                       label="1900 industrial clusters that cross a border")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.06), title="share of industry and mining in employment",
               title_fontsize=9)
    fig.suptitle("Where Europe worked in industry: the clusters did not stop at borders",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    finish(fig, "fig1_industry_maps",
           "Regions are NUTS-2 units of 2010 (NUTS-1 for the United Kingdom). Clusters: land-connected groups "
           "among the 35 regions with the highest industry share in 1900.")


def fig2() -> None:
    """Share of regional variation that lies between countries, against random groupings of
    neighbouring regions."""
    t = pd.read_csv(RES / "variance_between_countries.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, var, title in [(axes[0], "ind", "Share of industry in employment"),
                           (axes[1], "lgdppc", "GDP per head (log)")]:
        d = t[t["var"] == var]
        ax.fill_between(d.year, 100 * d.contiguous_p05, 100 * d.contiguous_p95, color=BLUE, alpha=0.16,
                        linewidth=0, label="random groupings of neighbouring regions, 5-95%")
        ax.plot(d.year, 100 * d.contiguous_mean, color=BLUE, linewidth=1.4, linestyle=(0, (4, 2)),
                label="random groupings, mean")
        ax.plot(d.year, 100 * d.country, color=ORANGE, marker="o", ms=4.5, label="the 16 countries")
        ax.plot(d.year, 100 * d.shuffled_mean, color=INK2, linewidth=1.1, linestyle=":",
                label="random groups ignoring geography, mean")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 80)
        ax.set_xlabel("year")
    axes[0].set_ylabel("% of regional variation between groups")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.07))
    fig.suptitle("Countries explain the industrial map no better than arbitrary lines; they explained incomes before 1938",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    finish(fig, "fig2_countries_against_random_lines",
           "Population-weighted. 1,000 random partitions of the map into 16 contiguous groups of about the countries' sizes.")


def fig3() -> None:
    """Border premium: how much more neighbours differ across a border than inside a country."""
    t = pd.read_csv(RES / "border_premium.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, var, title, unit in [(axes[0], "ind", "Share of industry in employment", "percentage points"),
                                 (axes[1], "lgdppc", "GDP per head", "log points x 100")]:
        d = t[t["var"] == var]
        ax.fill_between(d.year, d.null_p05, d.null_p95, color=BLUE, alpha=0.16, linewidth=0,
                        label="same test with random lines, 5-95%")
        ax.errorbar(d.year, d.premium, yerr=1.96 * d.se, fmt="o", color=ORANGE, ms=5, elinewidth=1.3,
                    capsize=0, label="national borders, with 95% interval")
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_title(title, loc="left")
        ax.set_ylabel(f"extra difference, {unit}")
        ax.set_xlabel("year")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Across a border, neighbours differ a little more in industry and a lot more in income",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    finish(fig, "fig3_border_premium",
           "Pairs of regions sharing a land boundary; centroid distance held fixed; dyadic-robust standard errors.")


def fig4() -> None:
    """The 1900 industrial core against the other regions, 1900-2021."""
    t = pd.read_csv(RES / "core_vs_rest.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.plot(t.year, 100 * t.core, color=ORANGE, marker="o", ms=4.5, label="the 35 most industrial regions of 1900")
    ax.plot(t.year, 100 * t.rest, color=BLUE, marker="o", ms=4.5, label="the other 138 regions")
    ax.set_ylabel("share of industry in employment (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Industrial core and the rest", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax = axes[1]
    ax.plot(t.year, t.spearman, color=INK, marker="o", ms=4.5)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylim(-0.4, 1.05)
    ax.set_ylabel("rank correlation with 1900")
    ax.set_title("How much of the 1900 ranking survives", loc="left")
    for a in axes:
        a.set_xlabel("year")
    fig.suptitle("The industrial map of 1900 was gone by 2000",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    finish(fig, "fig4_core_reversal", "Population-weighted shares; Spearman rank correlation across regions.")


def fig5() -> None:
    """GDP per head by region as a share of the 16-country level: 1900 in 2011 and in 1990 prices,
    and 2021, with national borders."""
    df = R.load_panel()
    g = R.load_map(pd.Series(sorted(df.code.unique())))
    g["country"] = df.drop_duplicates("code").set_index("code").country.reindex(g.index)
    nations = g.dissolve(by="country")
    cols = DIV_LOW + ["#f0efec"] + DIV_HIGH
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(DIV_BINS, cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2))
    panels = [(1900, "gdp", "1900, in 2011 prices"), (1900, "gdp90", "1900, in 1990 prices"),
              (2021, "gdp", "2021, in 2011 prices")]
    for ax, (year, col, title) in zip(axes, panels):
        d = df[df.year == year].dropna(subset=[col, "pop"]).set_index("code")
        level = d[col].sum() / d["pop"].sum()
        v = (100 * d[col] / d["pop"] / level).reindex(g.index)
        gg = g.assign(v=v)
        if gg.v.isna().any():
            gg[gg.v.isna()].plot(ax=ax, facecolor=SURFACE, edgecolor="#a8a7a2", hatch="////", linewidth=0.25)
        gg[gg.v.notna()].plot(ax=ax, column="v", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.25)
        nations.boundary.plot(ax=ax, color=INK, linewidth=0.9)
        ax.set_title(title, loc="left")
        ax.set_xlim(2.55e6, 5.55e6)
        ax.set_ylim(1.35e6, 5.25e6)
        ax.set_axis_off()
    labels = ["below 60%", "60-80%", "80-95%", "95-105%", "105-125%", "125-150%", "150% or more"]
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(cols, labels)]
    handles += [Patch(facecolor=SURFACE, edgecolor="#a8a7a2", hatch="////", label="no figure"),
                Line2D([0], [0], color=INK, linewidth=1, label="national border")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.06), title="GDP per head, % of the level of the 16 countries together",
               title_fontsize=9)
    fig.suptitle("Income steps at national borders; how big some steps are depends on the prices used",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    finish(fig, "fig5_income_maps",
           "Each panel against its own year and price base. Within a country the two price bases give the same "
           "regional ratios; only the national levels move.")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
