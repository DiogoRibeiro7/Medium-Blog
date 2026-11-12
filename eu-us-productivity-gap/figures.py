"""Draw the figures for the article from ``results/``.

    python figures.py

Static PNGs for Medium (light surface). Colours follow one fixed categorical order:
EU25 blue, North-West orange, South aqua, Centre-East yellow; the United States is
drawn in ink as the reference. Lines and dots are direct-labelled because aqua and
yellow sit below 3:1 contrast on the light surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
C = {"EU25": "#2a78d6", "North-West": "#eb6834", "South": "#1baf7a", "Centre-East": "#eda100",
     "Ireland and Luxembourg": "#898781", "United States": INK}
PART = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # components, same fixed order
NAMES = {"AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria", "HRV": "Croatia", "CZE": "Czechia", "DNK": "Denmark",
         "EST": "Estonia", "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece", "HUN": "Hungary",
         "IRL": "Ireland", "ITA": "Italy", "LVA": "Latvia", "LTU": "Lithuania", "LUX": "Luxembourg",
         "NLD": "Netherlands", "POL": "Poland", "PRT": "Portugal", "ROU": "Romania", "SVK": "Slovakia",
         "SVN": "Slovenia", "ESP": "Spain", "SWE": "Sweden", "USA": "United States"}
SOURCE = "Source: OECD Productivity Database and national accounts (snapshot 12 Nov 2026); author's calculations."

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def tidy(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name: str, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, SOURCE, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1() -> None:
    d = pd.read_csv(RES / "headline_definitions_2024.csv")
    def who(s: str) -> str:
        for g in ("North-West", "South", "Centre-East"):
            if g in s:
                return g
        return "EU25"
    d["agg"] = d.definition.map(who)
    d = d.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.43, right=0.95, top=0.84, bottom=0.12)
    ax.axvline(100, color=INK, linewidth=1.2)
    ax.text(99.4, len(d) - 0.45, "United States = 100", color=INK2, fontsize=8.5, va="center", ha="right")
    for i, r in d.iterrows():
        ax.hlines(i, 55, r.eu_relative_to_us, color=GRID, linewidth=1)
        ax.scatter(r.eu_relative_to_us, i, s=70, color=C[r["agg"]], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(r.eu_relative_to_us + 1.2, i, f"{r.eu_relative_to_us:.1f}", va="center", ha="left", fontsize=8.5, color=INK2)
    labels = [s.replace(", current PPPs", ",\ncurrent PPPs").replace(", constant 2020 PPPs", ",\nconstant 2020 PPPs") for s in d.definition]
    ax.set_yticks(range(len(d)), labels, fontsize=8.5)
    ax.set_xlim(55, 108)
    ax.set_xlabel("Europe relative to the United States, 2024 (US = 100)")
    tidy(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=C[g], label=g)
               for g in ("EU25", "North-West", "South", "Centre-East")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4, fontsize=8.5, handletextpad=0.3)
    finish(fig, "fig1_one_gap_ten_numbers", "One gap, ten numbers",
           "Europe's output relative to the US in 2024 under different definitions, aggregates and price bases")


def fig2() -> None:
    d = pd.read_csv(RES / "country_levels_2024.csv")
    order = ["North-West", "Ireland and Luxembourg", "South", "Centre-East", "United States"]
    d["g"] = pd.Categorical(d.group, order)
    d = d.sort_values(["g", "gdp_hour_rel"], ascending=[False, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 9.2))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.07)
    ax.axvline(100, color=INK, linewidth=1.2)
    y = np.arange(len(d))
    for i, r in d.iterrows():
        lo, hi = sorted([r.gdp_hour_rel, r.gni_hour_rel])
        ax.hlines(i, lo, hi, color=BASE, linewidth=2)
    ax.scatter(d.gdp_hour_rel, y, s=55, color=PART[0], edgecolor=SURFACE, linewidth=2, zorder=3, label="GDP per hour")
    ax.scatter(d.gni_hour_rel, y, s=60, facecolor="none", edgecolor=PART[1], linewidth=2, zorder=4, label="GNI per hour")
    for i, r in d.iterrows():
        if r.country in ("IRL", "LUX"):
            ax.text((r.gdp_hour_rel + r.gni_hour_rel) / 2, i + 0.42, f"GDP {r.gdp_hour_rel:.0f} → GNI {r.gni_hour_rel:.0f}",
                    va="center", ha="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [NAMES[c] for c in d.country], fontsize=8.5)
    # group brackets
    for g in order[:-1]:
        idx = d.index[d.group == g]
        ax.text(193, float(np.mean(idx)), g, va="center", ha="right", fontsize=8.5, color=MUTED, style="italic")
        ax.axhline(idx.max() + 0.5, color=GRID, linewidth=0.8)
    ax.set_xlim(40, 195)
    ax.set_xlabel("Per hour worked, current PPPs, United States = 100")
    tidy(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=9, handletextpad=0.3)
    finish(fig, "fig2_countries_gdp_gni", "Output per hour by country, 2024",
           "GDP per hour and gross national income per hour; the gap between the two is profit and interest paid to or received from abroad")


def fig3() -> None:
    c = pd.read_csv(RES / "country_levels_2024.csv").set_index("country")
    g = pd.read_csv(RES / "group_levels_2024.csv").set_index("group")
    rows = [("EU25", g.loc["EU25"]), ("North-West", g.loc["North-West"]), ("South", g.loc["South"]),
            ("Centre-East", g.loc["Centre-East"]), None]
    for k in ("DEU", "FRA", "NLD", "DNK", "ITA", "ESP", "PRT", "GRC", "POL", "CZE"):
        rows.append((NAMES[k], c.loc[k]))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=0.16, right=0.86, top=0.85, bottom=0.1)
    comps = [("lp_productivity", "Output per hour"), ("lp_hours", "Hours per worker"), ("lp_employment", "Employment rate")]
    labels = []
    for i, item in enumerate(rows):
        if item is None:
            labels.append("")
            continue
        name, r = item
        labels.append(name)
        pos, neg = 0.0, 0.0
        for j, (k, _) in enumerate(comps):
            v = r[k]
            left = pos if v >= 0 else neg + v
            ax.barh(i, abs(v), left=left, height=0.62, color=PART[j], edgecolor=SURFACE, linewidth=2)
            if v >= 0:
                pos += v
            else:
                neg += v
        ax.scatter(r["lp_total"], i, marker="D", s=40, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.text(min(neg, 0) - 1.5, i, f"{r['lp_total']:.0f}", va="center", ha="right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(range(len(rows)), labels, fontsize=9)
    ax.set_xlim(-86, 25)
    ax.set_xlabel("Log points relative to the United States (−10 ≈ 10% lower)")
    tidy(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PART[j], label=lab) for j, (_, lab) in enumerate(comps)]
    handles.append(plt.Line2D([], [], marker="D", linestyle="", color=INK, markersize=6, label="GDP per head (sum)"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4, fontsize=8.5, handletextpad=0.4)
    finish(fig, "fig3_per_capita_decomposition", "Why GDP per head is lower: output per hour, hours, or jobs",
           "GDP per head relative to the US split into output per hour, hours per worker and employment per head, 2024")


def fig4() -> None:
    p = pd.read_csv(RES / "time_paths.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.83, bottom=0.12, wspace=0.28)
    for ax, basis, title in ((axes[0], "const", "Constant 2020 PPPs (volume growth)"),
                             (axes[1], "cur", "Current PPPs (each year's price survey)")):
        ax.axhline(100, color=INK, linewidth=1.2)
        for g in ("North-West", "EU25", "South", "Centre-East"):
            ax.plot(p.year, p[f"{g}|{basis}"], color=C[g], linewidth=2)
            yv = p[f"{g}|{basis}"].iloc[-1]
            ax.scatter(p.year.iloc[-1], yv, s=40, color=C[g], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(p.year.iloc[-1] + 0.8, yv, f"{g} {yv:.0f}", va="center", fontsize=8.5, color=INK2)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(1995, 2024)
        ax.set_xticks([1995, 2000, 2005, 2010, 2015, 2020, 2024])
        tidy(ax, "y")
    axes[0].set_ylabel("GDP per hour, United States = 100")
    axes[0].set_ylim(25, 130)
    finish(fig, "fig4_two_price_bases", "Is Europe falling behind? It depends on the price basis",
           "GDP per hour relative to the US, 1995–2024; the two panels agree in 2020 by construction (Estonia excluded: no 1995 data)")


def fig5() -> None:
    d = pd.read_csv(RES / "growth_gap_cumulative.csv").rename(columns={"capital_deepening": "k"})
    d["country"] = d.country.map(NAMES)
    d = d.sort_values("lp").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.14, right=0.93, top=0.8, bottom=0.14)
    for i, r in d.iterrows():
        ax.barh(i, r.k, left=0, height=0.55, color=PART[0], edgecolor=SURFACE, linewidth=2)
        ax.barh(i, r.mfp, left=r.k, height=0.55, color=PART[1], edgecolor=SURFACE, linewidth=2)
        ax.text(r.k + r.mfp + 0.6, i, f"{r.lp:.0f} pp, of which {r.k:.0f} capital", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(d)), d.country)
    ax.set_xlim(0, 50)
    ax.set_xlabel("US labour productivity growth minus the country's, summed over 1998–2023 (percentage points)")
    tidy(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PART[0], label="Capital deepening (more capital per hour)"),
               plt.Rectangle((0, 0), 1, 1, color=PART[1], label="Multifactor productivity (the residual)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=8.5)
    finish(fig, "fig5_growth_accounting", "Where the US growth lead came from, 1998–2023",
           "Cumulative US advantage in growth of value added per hour, split by OECD growth accounting")


def fig6() -> None:
    d = pd.read_csv(RES / "income_split_2024.csv").set_index("group")
    order = ["Centre-East", "South", "EU25", "North-West", "United States"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.8, bottom=0.14)
    parts = [("Labour compensation", lambda r, n: r.lc_hour), ("Net taxes on production", lambda r, n: r.tax_hour),
             ("Depreciation", lambda r, n: r.cfc_hour), ("Net operating surplus and other", lambda r, n: r.net_rest_hour)]
    for i, n in enumerate(order):
        r = d.loc[n]
        left = 0.0
        for j, (_, f) in enumerate(parts):
            v = f(r, n)
            ax.barh(i, v, left=left, height=0.55, color=PART[j], edgecolor=SURFACE, linewidth=2)
            if v > 7:
                ax.text(left + v / 2, i, f"{v:.0f}", va="center", ha="center", fontsize=8.5,
                        color=INK if PART[j] == "#eda100" else "#ffffff")
            left += v
        ax.text(left + 1, i, f"{r.gdp_hour:.0f}", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, 112)
    ax.set_xlabel("GDP per hour worked, 2024, US dollars at current PPPs")
    tidy(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PART[j], label=lab) for j, (lab, _) in enumerate(parts)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4, fontsize=8.5)
    finish(fig, "fig6_income_side", "Who receives an hour's output",
           "GDP per hour split by income: labour includes an imputation for the self-employed; the US residual includes its statistical discrepancy")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print("figures written to", FIG)
