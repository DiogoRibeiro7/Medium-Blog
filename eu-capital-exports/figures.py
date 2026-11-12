"""Draw the three figures for the article from ``results/``.

    python figures.py

Static PNGs for Medium (light surface), one fixed categorical order, direct labels on lines
and a legend on the dot plots.
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
SOURCE = ("Source: Eurostat (nasa_10_nf_tr, nama_10_gdp, bop_eu6_q, bop_iip6_q, nasa_10_f_bs), OECD (Table 2, "
          "T720R); snapshot 12 Nov 2026; author's calculations.")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def tidy(ax, grid_axis: str = "x") -> None:
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


def end_labels(ax, x, items, min_gap):
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for label, y in items:
        yy = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(yy)
        ax.text(x, yy, label, va="center", fontsize=8.5, color=INK2)


def fig1() -> None:
    si = pd.read_csv(RES / "saving_investment.csv")
    nl = pd.read_csv(RES / "sector_net_lending.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    fig.subplots_adjust(left=0.06, right=0.82, top=0.82, bottom=0.12, wspace=0.66)
    ax = axes[0]
    series = [("EU27", "saving", "EU27 saving", SLOTS[0], "-"), ("EU27", "investment", "EU27 investment", SLOTS[0], (0, (4, 2))),
              ("USA", "saving", "US saving", SLOTS[1], "-"), ("USA", "investment", "US investment", SLOTS[1], (0, (4, 2)))]
    items = []
    for area, col, lab, color, ls in series:
        s = si[(si.area == area) & (si.year >= 2002)].set_index("year")[col]
        ax.plot(s.index, s, color=color, linewidth=2, linestyle=ls)
        ax.scatter(s.index[-1], s.iloc[-1], s=32, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        items.append((f"{lab} {s.iloc[-1]:.1f}", s.iloc[-1]))
    end_labels(ax, 2025.9, items, 0.75)
    ax.set_xlim(2002, 2025)
    ax.set_ylim(12, 27)
    ax.set_title("Gross saving and gross capital formation, % of GDP")
    tidy(ax, "y")
    ax = axes[1]
    sectors = [("S14_S15", "Households"), ("S11", "Non-financial firms"), ("S12", "Financial firms"),
               ("S13", "Government"), ("S1", "Total economy")]
    items = []
    nl = nl.loc[2002:]
    for k, (col, lab) in enumerate(sectors):
        s = nl[col].dropna()
        ax.plot(s.index, s, color=SLOTS[k] if col != "S1" else INK, linewidth=2.6 if col == "S1" else 1.8)
        ax.scatter(s.index[-1], s.iloc[-1], s=32, color=SLOTS[k] if col != "S1" else INK, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        items.append((f"{lab} {s.iloc[-1]:.1f}", s.iloc[-1]))
    ax.axhline(0, color=INK2, linewidth=1)
    end_labels(ax, 2025.9, items, 0.75)
    ax.set_xlim(2002, 2025)
    ax.set_ylim(-7.5, 7.5)
    ax.set_title("EU27 net lending by sector, % of GDP")
    tidy(ax, "y")
    finish(fig, "fig1_saving_investment_and_sectors",
           "The EU saves more than it invests; the United States invests more than it saves",
           "EU27 (Eurostat) and United States (OECD), 2002–2025; net lending is the surplus lent to other sectors or abroad")


def fig2() -> None:
    t = pd.read_csv(RES / "candidate_measures.csv", index_col=0)
    order = ["non-EU purchases of EU securities", "household net lending (all destinations)",
             "gross purchases of non-EU securities", "of which by non-bank financial corporations",
             "gross purchases of US securities", "of which by non-financial corporations and households",
             "current account balance", "net lending (current + capital account)", "net financial account",
             "net portfolio investment"]
    labels = {"non-EU purchases of EU securities": "Non-EU purchases of EU securities (gross inflow)",
              "household net lending (all destinations)": "Household net lending, all destinations",
              "gross purchases of non-EU securities": "EU purchases of non-EU securities (gross)",
              "of which by non-bank financial corporations": "   of which funds, insurers, pension funds",
              "gross purchases of US securities": "EU purchases of US securities (gross)",
              "of which by non-financial corporations and households": "   of which households and firms directly",
              "current account balance": "Current account surplus",
              "net lending (current + capital account)": "Net lending to the rest of the world",
              "net financial account": "Net financial account",
              "net portfolio investment": "Net portfolio investment"}
    d = pd.DataFrame({"y2024": t.loc[2024, order], "avg": t[order].mean()}).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.0))
    fig.subplots_adjust(left=0.37, right=0.97, top=0.85, bottom=0.13)
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.hlines(i, min(r.y2024, r.avg), max(r.y2024, r.avg), color=BASE, linewidth=2)
        ax.text(max(r.y2024, r.avg) + 15, i, f"{r.y2024:,.0f}", va="center", fontsize=8, color=INK2)
    ax.scatter(d.y2024, y, s=60, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3, label="2024")
    ax.scatter(d.avg, y, s=64, facecolor="none", edgecolor=SLOTS[1], linewidth=2, zorder=4, label="Mean 2015–2025")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.axvline(300, color=INK, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.text(305, len(d) - 0.35, "€300 bn", fontsize=8.5, color=INK, va="bottom")
    ax.set_yticks(y, [labels[k] for k in d.index], fontsize=9)
    ax.set_xlabel("EUR billion per year; label shows 2024")
    ax.set_xlim(-250, 900)
    ax.legend(loc="lower right", fontsize=8.5)
    tidy(ax, "x")
    finish(fig, "fig2_ten_measures_of_savings_abroad", "Ten numbers that could be called 'savings sent abroad', EU27",
           "Balance of payments with non-EU residents and EU sector accounts; gross flows are purchases net of sales")


def fig3() -> None:
    pos = pd.read_csv(RES / "net_position.csv", index_col=0).loc[2014:]
    hh = pd.read_csv(RES / "household_assets.csv")
    fig = plt.figure(figsize=(10, 5.4))
    axes = [fig.add_axes([0.07, 0.2, 0.36, 0.6]), fig.add_axes([0.73, 0.2, 0.25, 0.6])]
    ax = axes[0]
    cols = [("direct investment", "Direct investment"), ("portfolio debt", "Portfolio debt"),
            ("other investment", "Other investment"), ("portfolio equity", "Portfolio equity"),
            ("total excl. reserves", "Total (excl. reserves)")]
    items = []
    for k, (c, lab) in enumerate(cols):
        s = pos[c].dropna() / 1e3
        color = SLOTS[k] if c != "total excl. reserves" else INK
        ax.plot(s.index, s, color=color, linewidth=2.6 if c == "total excl. reserves" else 1.8)
        ax.scatter(s.index[-1], s.iloc[-1], s=32, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        items.append((f"{lab} {s.iloc[-1]:.1f}", s.iloc[-1]))
    ax.axhline(0, color=INK2, linewidth=1)
    end_labels(ax, 2025.6, items, 0.42)
    ax.set_xlim(2014, 2025)
    ax.set_ylim(-4, 3.2)
    ax.set_title("EU27 net position vs non-EU, EUR trillion")
    tidy(ax, "y")
    ax = axes[1]
    cats = ["deposits", "debt securities", "listed and other equity", "investment fund shares", "insurance and pensions"]
    labs = ["Deposits", "Debt securities", "Equity", "Investment fund shares", "Insurance and pensions"]
    eu = hh[(hh.area == "EU27") & (hh.year == 2025)].iloc[0]
    us = hh[(hh.area == "US") & (hh.year == 2025)].iloc[0]
    y = np.arange(len(cats))[::-1]
    for yi, c in zip(y, cats):
        ax.hlines(yi, min(eu[c], us[c]) * 100, max(eu[c], us[c]) * 100, color=BASE, linewidth=2)
    ax.scatter([eu[c] * 100 for c in cats], y, s=60, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="EU27 households")
    ax.scatter([us[c] * 100 for c in cats], y, s=64, facecolor="none", edgecolor=SLOTS[1], linewidth=2, zorder=4,
               label="US households")
    ax.set_yticks(y, labs, fontsize=9)
    ax.set_xlim(0, 50)
    ax.set_title("Household assets, %, 2025")
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.1), ncol=1, fontsize=8.5)
    tidy(ax, "x")
    finish(fig, "fig3_positions_and_household_portfolios",
           "Stocks: a creditor in debt and direct investment, a debtor in equity",
           "EU27 international investment position by instrument, 2014–2025; household portfolios in the EU27 and the US")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3):
        f()
    print(f"figures written to {FIG}")
