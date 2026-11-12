"""Draw the four figures for the article from ``results/``.

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
SOURCE = ("Source: Eurostat (earn_mw_cur, earn_ses_mdeci, nama_10_fte, prc_hicp_aind, ilc_li01, ilc_iw01-07) and "
          "OECD TaxBEN (DF_IMW), snapshot 12 Nov 2026; author's calculations.")
NAMES = {"PT": "Portugal", "ES": "Spain", "EL": "Greece", "IT": "Italy", "FR": "France", "DE": "Germany",
         "BE": "Belgium", "NL": "Netherlands", "IE": "Ireland", "LU": "Luxembourg", "SI": "Slovenia", "MT": "Malta",
         "CY": "Cyprus", "SK": "Slovakia", "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania", "HR": "Croatia",
         "EU27_2020": "EU27"}

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
    """Place end-of-line labels, spreading them apart only as much as needed (in data units)."""
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for label, y in items:
        yy = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(yy)
        ax.text(x, yy, label, va="center", fontsize=8.5, color=INK2)


def bold_portugal(ax, codes) -> None:
    for t, c in zip(ax.get_yticklabels(), codes):
        if c == "PT":
            t.set_fontweight("bold")
            t.set_color(INK)


def fig1() -> None:
    d = pd.read_csv(RES / "real_growth_ses_2006_2022.csv").sort_values("median").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.84, bottom=0.17)
    y = np.arange(len(d))
    for i, r in d.iterrows():
        ax.hlines(i, 100 * r["median"], 100 * r.floor, color=BASE, linewidth=2)
    ax.scatter(100 * d.floor, y, s=60, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Statutory minimum wage")
    ax.scatter(100 * d["median"], y, s=64, facecolor="none", edgecolor=SLOTS[1], linewidth=2, zorder=4,
               label="Median full-time monthly earnings (SES)")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, d.name, fontsize=9)
    bold_portugal(ax, d.geo)
    ax.set_xlabel("Change 2006–2022 after national consumer prices (log points × 100)")
    ax.legend(loc="lower right", fontsize=8.5, handletextpad=0.4)
    tidy(ax, "x")
    finish(fig, "fig1_floor_and_median_real", "The floor and the middle in real terms, nine euro-area countries, 2006–2022",
           "In seven of nine the minimum wage rose more than the median; in Portugal the real median was flat until 2018")


def fig2() -> None:
    idx = pd.read_csv(RES / "portugal_real_indices.csv").set_index("year")
    iw = pd.read_csv(RES / "in_work_poverty_rates.csv").set_index("time")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))
    fig.subplots_adjust(left=0.07, right=0.86, top=0.82, bottom=0.14, wspace=0.55)
    ax = axes[0]
    series = [("minimum_real", "Minimum wage"), ("poverty_line_real", "Poverty line,\none adult"),
              ("fte_salary_real", "Average full-time\nsalary")]
    items = []
    for k, (col, lab) in enumerate(series):
        s = idx[col].dropna()
        ax.plot(s.index, s, color=SLOTS[k], linewidth=2)
        ax.scatter(s.index[-1], s.iloc[-1], s=36, color=SLOTS[k], edgecolor=SURFACE, linewidth=2, zorder=3)
        items.append((f"{lab} {s.iloc[-1]:.0f}", s.iloc[-1]))
    ax.axhline(100, color=INK2, linewidth=1)
    end_labels(ax, 2026.0, items, 9)
    ax.set_xlim(2006, 2025)
    ax.set_ylim(85, 170)
    ax.set_title("Portugal, real index, 2006 = 100")
    tidy(ax, "y")
    ax = axes[1]
    geos = ["PT", "ES", "EL", "IT", "EU27_2020"]
    items = []
    for k, g in enumerate(geos):
        s = iw[g].dropna()
        s = s[s.index >= 2005]
        ax.plot(s.index, s, color=SLOTS[k], linewidth=2.6 if g == "PT" else 1.8)
        ax.scatter(s.index[-1], s.iloc[-1], s=36, color=SLOTS[k], edgecolor=SURFACE, linewidth=2, zorder=3)
        items.append((f"{NAMES[g]} {s.iloc[-1]:.1f}", s.iloc[-1]))
    end_labels(ax, 2025.8, items, 0.75)
    ax.set_xlim(2005, 2025)
    ax.set_ylim(0, 16)
    ax.set_title("In-work at-risk-of-poverty rate, %")
    tidy(ax, "y")
    finish(fig, "fig2_floor_line_and_in_work_poverty",
           "A rising floor, a rising poverty line, and in-work poverty that moved little until 2024",
           "Portugal's real minimum wage and poverty line against the average salary; in-work poverty, employed 18–64")


def fig3() -> None:
    pr = pd.read_csv(RES / "minimum_wage_vs_poverty_line.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.0), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.12, wspace=0.62)
    ax = axes[0]
    pt = pr[pr.geo == "PT"].set_index("year")
    hh = [("single", "Single adult"), ("couple_2ch_two_earners", "Couple, two earners,\ntwo children"),
          ("single_parent", "Single parent,\ntwo children"), ("couple_one_earner", "Couple, one earner"),
          ("couple_2ch_one_earner", "Couple, one earner,\ntwo children")]
    items = []
    for k, (col, lab) in enumerate(hh):
        s = pt[col].dropna()
        ax.plot(s.index, s, color=SLOTS[k], linewidth=2)
        ax.scatter(s.index[-1], s.iloc[-1], s=36, color=SLOTS[k], edgecolor=SURFACE, linewidth=2, zorder=3)
        items.append((f"{lab} {s.iloc[-1]:.2f}", s.iloc[-1]))
    ax.axhline(1, color=INK, linewidth=1.2)
    ax.text(2003.3, 0.985, "poverty line", fontsize=8.5, color=INK2, va="top")
    end_labels(ax, 2024.8, items, 0.075)
    ax.set_xlim(2003, 2024)
    ax.set_ylim(0.5, 1.4)
    ax.set_title("Portugal, by income year")
    tidy(ax, "y")
    ax = axes[1]
    x = pr[pr.year == 2024].sort_values("single").reset_index(drop=True)
    y = np.arange(len(x))
    for i, r in x.iterrows():
        ax.hlines(i, r.couple_2ch_one_earner, r.single, color=BASE, linewidth=2)
    ax.scatter(x.single, y, s=52, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3, label="Single adult")
    ax.scatter(x.couple_2ch_two_earners, y, s=46, marker="s", color=SLOTS[1], edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="Couple, two earners, two children")
    ax.scatter(x.couple_2ch_one_earner, y, s=56, facecolor="none", edgecolor=SLOTS[4], linewidth=2, zorder=4,
               label="Couple, one earner, two children")
    ax.axvline(1, color=INK, linewidth=1.2)
    ax.set_yticks(y, [NAMES[g] for g in x.geo], fontsize=8.5)
    bold_portugal(ax, x.geo)
    ax.set_xlim(0.4, 1.45)
    ax.set_title("17 euro-area countries, income year 2024")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.06), fontsize=8, ncol=1, handletextpad=0.3)
    tidy(ax, "x")
    fig.subplots_adjust(bottom=0.2)
    finish(fig, "fig3_net_minimum_wage_vs_poverty_line",
           "Net income at a full-time minimum wage, divided by the 60% poverty line",
           "Above 1, the household is above the EU at-risk-of-poverty threshold; taxes, contributions and cash benefits included")


def fig4() -> None:
    d = pd.read_csv(RES / "in_work_poverty_groups_2025.csv")
    d = d.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.86, bottom=0.12)
    y = np.arange(len(d))
    for i, r in d.iterrows():
        ax.hlines(i, min(r.PT, r.EU27_2020), max(r.PT, r.EU27_2020), color=BASE, linewidth=2)
    ax.scatter(d.PT, y, s=60, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3, label="Portugal")
    ax.scatter(d.EU27_2020, y, s=64, facecolor="none", edgecolor=SLOTS[1], linewidth=2, zorder=4, label="EU27")
    for i, r in d.iterrows():
        ax.text(max(r.PT, r.EU27_2020) + 0.8, i, f"{r.PT:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [g[0].upper() + g[1:] for g in d.group], fontsize=9)
    for sep in (3.5, 7.5, 9.5, 11.5):
        ax.axhline(sep, color=GRID, linewidth=0.8)
    ax.set_xlabel("In-work at-risk-of-poverty rate, 2025 (%); label shows Portugal")
    ax.set_xlim(0, 42)
    ax.legend(loc="upper right", fontsize=8.5)
    tidy(ax, "x")
    finish(fig, "fig4_who_is_poor_in_work", "Who is poor in work, Portugal and the EU27, 2025",
           "Grouped by work intensity of the household, household type, working time, contract and employment status")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
    print(f"figures written to {FIG}")
