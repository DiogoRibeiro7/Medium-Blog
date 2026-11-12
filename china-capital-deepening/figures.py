"""Draw the three figures for the article from ``results/``.

    python figures.py

Static PNGs for Medium (light surface), one fixed categorical order, direct labels where lines are few.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SOURCE = ("Sources: China Industrial Productivity (CIP) database 4.0, RIETI; OECD national accounts and Economic Outlook "
          "long-term baseline, snapshot 12 Nov 2026; author's calculations.")
NAMES = {"DEU": "Germany", "FRA": "France", "ITA": "Italy", "ESP": "Spain", "NLD": "Netherlands", "AUT": "Austria",
         "BEL": "Belgium", "CZE": "Czechia", "POL": "Poland", "PRT": "Portugal", "SWE": "Sweden", "JPN": "Japan"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def tidy(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name: str, title: str, subtitle: str, source: str = SOURCE) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, source, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1() -> None:
    c = pd.read_csv(RES / "cip_output_per_capital.csv", index_col=0)
    o = pd.read_csv(RES / "oecd_china_output_per_capital.csv", index_col=0).china_total_oecd
    e = pd.read_csv(RES / "europe_manufacturing_output_per_capital.csv", index_col=0).europe_manufacturing_median
    sh = pd.read_csv(RES / "cip_investment_shares.csv", index_col=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.14, wspace=0.18)
    years = range(1995, 2018)
    series = [("China, manufacturing (CIP)", 100 * c.china_manufacturing_cip / c.china_manufacturing_cip[2007], SLOTS[0], "-"),
              ("China, rest of the economy (CIP)", 100 * c.china_outside_manufacturing_cip / c.china_outside_manufacturing_cip[2007], SLOTS[1], "-"),
              ("China, whole economy (OECD)", 100 * o / o[2007], SLOTS[1], (0, (4, 2))),
              ("Europe, manufacturing, median of 11 (OECD)", 100 * e / e[2007], SLOTS[2], "-")]
    for label, s, col, ls in series:
        s = s.loc[1995:2024] if "OECD" in label else s.loc[list(years)]
        a1.plot(s.index, s.values, color=col, linewidth=2, linestyle=ls, label=label)
    a1.set_yscale("log")
    a1.set_yticks([25, 50, 100, 200], ["25", "50", "100", "200"])
    a1.minorticks_off()
    a1.axvline(2013, color=BASE, linewidth=1, linestyle=":")
    a1.text(2013.2, 205, "2013", color=MUTED, fontsize=8.5)
    a1.set_title("Output per unit of capital, 2007 = 100 (log scale)")
    a1.legend(loc="lower left", fontsize=8.3)
    tidy(a1)
    groups = [("Manufacturing", "Manufacturing", SLOTS[0]), ("Real estate", "Real estate", SLOTS[4]),
              ("Public administration and services", "Public admin.\nand services", SLOTS[6]),
              ("Transport and utilities", "Transport and\nutilities", SLOTS[3])]
    sh["Transport and utilities"] = sh["Transport"] + sh["Utilities"]
    ends = sorted(((100 * sh[n].loc[2017], n) for n, _, _ in groups))
    placed, last = {}, -99.0
    for y, n in ends:  # keep end labels at least 4.5 points apart
        y = max(y, last + 4.5)
        placed[n], last = y, y
    for name, label, col in groups:
        s = 100 * sh[name].loc[2000:2017]
        a2.plot(s.index, s.values, color=col, linewidth=2)
        a2.text(2017.6, placed[name], f"{label} {s.loc[2017]:.0f}%", va="center", fontsize=8, color=INK2)
    a2.set_xlim(2000, 2025.5)
    a2.set_xticks([2000, 2005, 2010, 2015])
    a2.set_ylim(0, 42)
    a2.set_title("Share of China's fixed capital formation, %")
    tidy(a2)
    finish(fig, "fig1_output_per_capital_and_where_investment_went",
           "Chinese manufacturing raised output per unit of capital while the rest of the economy did the opposite",
           "Manufacturing's share of China's investment fell from 20% in 2013 to 12% in 2017 in the CIP allocation; "
           "housing and public infrastructure took the rest")


def fig2() -> None:
    gdf = pd.read_csv(RES / "cip_share_grid.csv")
    sec = pd.read_csv(RES / "cip_sectors.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 6.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.14)
    for delta, col, lw in [(0.06, SLOTS[6], 1.6), (0.08, SLOTS[0], 2.4), (0.10, SLOTS[3], 1.6)]:
        s = gdf[gdf.delta == delta]
        ax.plot(100 * s.share, s.y_per_k_growth, color=col, linewidth=lw, label=f"depreciation {int(round(100 * delta))}% a year")
    ax.axhline(0, color=INK2, linewidth=1)
    marks = [(100 * 0.123, "CIP as published\n(average 2014-2017)", (10, 10)),
             (100 * sec.loc["Manufacturing", "inv_share_2013"], "manufacturing's\n2013 share", (-100, -34)),
             (33.8, "manufacturing's share of\nNBS fixed-asset investment\n(33.8% in 2013, 30.7% in 2017)", (-20, 42))]
    s8 = gdf[gdf.delta == 0.08].set_index("share").y_per_k_growth
    for x, lab, off in marks:
        y = float(np.interp(x / 100, s8.index, s8.values))
        ax.scatter([x], [y], s=60, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=8.3, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8, "shrinkA": 2, "shrinkB": 5})
    ax.set_xlabel("Manufacturing's share of China's fixed capital formation, held constant 2014-2017, %")
    ax.set_ylabel("Output per unit of capital, average annual growth 2013-2017, %")
    ax.legend(loc="lower left", fontsize=8.5)
    tidy(ax, "both")
    finish(fig, "fig2_break_even_investment_share",
           "The manufacturing result turns on one allocation: break-even near 18% of China's investment",
           "Chinese manufacturing's output per unit of capital, 2013-2017, if it had received a different share of national investment")


def fig3() -> None:
    t = pd.read_csv(RES / "oecd_manufacturing.csv", index_col=0)
    c = pd.read_csv(RES / "cip_manufacturing_periods.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.11)
    lim = (-1.0, 12.5)
    ax.plot(lim, lim, color=BASE, linewidth=1.2)
    ax.text(5.4, 4.75, "constant output per unit of capital", rotation=30, fontsize=8, color=MUTED, ha="center")
    sub = t[["kl_2013_2017", "yl_2013_2017"]].dropna()
    ax.scatter(sub.kl_2013_2017, sub.yl_2013_2017, s=40, color=SLOTS[2], edgecolor=SURFACE, linewidth=1.2, zorder=3,
               label="11 European countries and Japan, 2013-2017 (OECD; detail in the box)")
    box = (-0.6, 2.4, 1.0, 4.2)
    ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, edgecolor=MUTED,
                               linewidth=0.8, linestyle=(0, (3, 2))))
    china = [("2013-2017", "China 2013-2017 (CIP)", "o"), ("2007-2013", "China 2007-2013 (CIP)", "D"),
             ("2001-2007", "China 2001-2007 (CIP)", "s")]
    for period, lab, mk in china:
        r = c.loc[period]
        face = SLOTS[0] if period == "2013-2017" else "none"
        ax.scatter([r.k_per_worker], [r.y_per_worker], s=80, marker=mk, facecolor=face, edgecolor=SLOTS[0], linewidth=2,
                   zorder=4, label=lab)
        ax.annotate(lab.replace(" (CIP)", ""), (r.k_per_worker, r.y_per_worker), xytext=(9, -3),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlim(*lim)
    ax.set_ylim(-0.5, 15.5)
    ax.set_xlabel("Capital per worker, average annual growth, %")
    ax.set_ylabel("Value added per worker, average annual growth, %")
    ax.legend(loc="upper left", fontsize=8.3)
    tidy(ax, "both")
    ins = ax.inset_axes([0.60, 0.05, 0.38, 0.40])
    ins.set_facecolor(SURFACE)
    ins.plot([box[0], box[1]], [box[0], box[1]], color=BASE, linewidth=1.2)
    ins.scatter(sub.kl_2013_2017, sub.yl_2013_2017, s=36, color=SLOTS[2], edgecolor=SURFACE, linewidth=1.0, zorder=3)
    off = {"NLD": (-8, 6, "right"), "ESP": (-6, -9, "right"), "PRT": (6, -4, "left"), "ITA": (5, 3, "left"),
           "CZE": (6, -3, "left"), "POL": (6, 2, "left"), "DEU": (6, 2, "left"), "SWE": (-6, 5, "right"),
           "AUT": (6, -3, "left"), "BEL": (-6, -8, "right"), "FRA": (6, -6, "left"), "JPN": (-6, 4, "right")}
    for k, r in sub.iterrows():
        dx, dy, ha = off.get(k, (6, 0, "left"))
        ins.annotate(NAMES[k], (r.kl_2013_2017, r.yl_2013_2017), xytext=(dx, dy), textcoords="offset points",
                     fontsize=7.3, color=INK2, ha=ha, va="center")
    ins.set_xlim(box[0], box[1])
    ins.set_ylim(box[2], box[3])
    ins.tick_params(labelsize=7, length=0)
    for side in ins.spines.values():
        side.set_edgecolor(MUTED)
        side.set_linestyle((0, (3, 2)))
        side.set_linewidth(0.8)
    ins.grid(color=GRID, linewidth=0.6)
    ins.set_axisbelow(True)
    finish(fig, "fig3_manufacturing_capital_and_output_per_worker",
           "Manufacturing: capital per worker against output per worker",
           "Above the diagonal output per unit of capital rose: China was close to it in 2007-2013, "
           "well above it in 2013-2017, as was all of Europe")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3):
        f()
    print(f"figures written to {FIG}")
