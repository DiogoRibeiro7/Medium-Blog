"""Draw the figures from the outputs of ``ceiling.py``.

Run after it::

    python ceiling.py
    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
CEILING = 2300.0
RMMG = {2017: 557, 2018: 580, 2019: 600, 2020: 635, 2021: 665, 2022: 705,
        2023: 760, 2024: 820, 2025: 870, 2026: 920}
INE = "INE, Estatísticas de Rendas da Habitação ao nível local and Censos 2021"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def finish(fig: plt.Figure, name: str, source: str, top: float = 0.94) -> None:
    fig.text(0.01, 0.01, f"Source: {source}; author's calculations.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, top))
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.985)


def end_label(ax, x, y, text, color, dy=0.0) -> None:
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=8.5, color=INK_2)
    ax.plot([x], [y], "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)


def fig01_area_needed() -> None:
    d = pd.read_csv(RESULTS / "area_needed.csv", dtype={"dtmn": str})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    suptitle(fig, "Floor area a new lease needed to reach EUR 2,300 a month, at local rents per m²")
    places = [("Lisboa", BLUE), ("Porto", ORANGE), ("Braga", AQUA)]
    for name, col in places:
        g = d[d["geodsg"] == name].sort_values("year")
        a.plot(g["year"], g["A_p75"], color=col)
        a.plot(g["year"], g["A_p50"], color=col, linestyle=(0, (3, 2)), linewidth=1.4)
        end_label(a, g["year"].iloc[-1], g["A_p75"].iloc[-1], f"{name}, Q3", col)
    a.set_title("At the 3rd quartile (solid) and median (dashed)", fontsize=9.5, fontweight="normal", color=INK_2)
    a.set_ylabel("m² needed to reach EUR 2,300")
    a.set_ylim(0, 480)
    a.set_xlim(2019.7, 2026.3)
    a.set_xticks(range(2020, 2026))
    for ref in (100, 120):
        a.axhline(ref, color=MUTED, linewidth=0.8, linestyle=":")
        a.text(2019.75, ref + 6, f"{ref} m²", fontsize=7.5, color=MUTED)
    g = d[(d["year"] == 2025) & d["A_p75"].notna()].sort_values("A_p75").reset_index(drop=True)
    b.scatter(np.arange(len(g)), g["A_p75"], s=9, color=BLUE, edgecolor="none")
    first = ", ".join(f"{r.geodsg} {r.A_p75:.0f}" for r in g.head(4).itertuples())
    b.annotate(f"{first} m²", (0, g["A_p75"].iloc[0]), xytext=(14, -4), textcoords="offset points",
               fontsize=8, color=INK_2)
    b.set_yscale("log")
    b.set_yticks([100, 150, 200, 300, 500, 800])
    b.yaxis.set_minor_formatter(NullFormatter())
    b.set_yticklabels(["100", "150", "200", "300", "500", "800"])
    b.axhline(150, color=MUTED, linewidth=0.8, linestyle=":")
    b.set_title(f"2025, at the 3rd quartile, {len(g)} municipalities ranked",
                fontsize=9.5, fontweight="normal", color=INK_2)
    b.set_xlabel("municipalities, from the dearest rent per m² to the cheapest")
    b.set_ylabel("m² needed (log scale)")
    b.set_xticks([])
    finish(fig, "fig01_area_needed.png", INE)


def fig02_rule_over_time() -> None:
    p = pd.read_csv(RESULTS / "panel_2017_2025.csv", dtype={"dtmn": str})
    nat = pd.read_csv(RESULTS / "national_series.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    suptitle(fig, "Square metres the moderate-rent ceiling buys at the median rent per m² of new leases")
    ax.plot(nat["year"], nat["m2_rule"], color=INK, linewidth=2.2)
    end_label(ax, nat["year"].iloc[-1], nat["m2_rule"].iloc[-1], "Portugal", INK)
    ax.plot(nat["year"], nat["m2_fixed"], color=INK, linewidth=1.2, linestyle=(0, (3, 2)))
    end_label(ax, nat["year"].iloc[-1], nat["m2_fixed"].iloc[-1], "Portugal, fixed EUR 2,300", INK, dy=8)
    for name, col, dy in (("Lisboa", BLUE, -12), ("Porto", ORANGE, 6), ("Cascais", PURPLE, -1)):
        g = p[p["geodsg"] == name].sort_values("year")
        ax.plot(g["year"], g["m2_rule"], color=col)
        end_label(ax, g["year"].iloc[-1], g["m2_rule"].iloc[-1], name, col, dy=dy)
    ax.axvspan(2016.6, 2019.5, color=GRID, alpha=0.45, linewidth=0)
    ax.text(2016.75, 60, "2021 methodology,\nspliced at 2020", fontsize=7.5, color=MUTED, va="top")
    ax.set_ylabel("m² (ceiling = 2.5 x that year's minimum wage)")
    ax.set_xlim(2016.6, 2026.6)
    ax.set_ylim(0, 540)
    ax.set_xticks(range(2017, 2026))
    finish(fig, "fig02_rule_over_time.png", INE + "; minimum wage, DGERT")


def fig03_share_above() -> None:
    s = pd.read_csv(RESULTS / "share_above.csv", dtype={"dtmn": str})
    s = s[s["leases"].notna()]
    cols = ["share_main", "share_beta_lo", "share_beta_hi", "share_area90", "share_area110", "share_sigma120", "share_beta0"]
    rows = []
    for y, g in s.groupby("year"):
        rows.append({"year": y, **{c: 100 * np.average(g[c], weights=g["leases"]) for c in cols}})
    t = pd.DataFrame(rows)
    band = t[["share_beta_lo", "share_beta_hi", "share_area90", "share_area110", "share_sigma120"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    suptitle(fig, "Modelled share of new leases with a monthly rent above EUR 2,300")
    ax.fill_between(t["year"], band.min(axis=1), band.max(axis=1), color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(t["year"], t["share_main"], color=BLUE)
    end_label(ax, t["year"].iloc[-1], t["share_main"].iloc[-1], "model", BLUE)
    ax.plot(t["year"], t["share_beta0"], color=MUTED, linewidth=1.3, linestyle=(0, (3, 2)))
    end_label(ax, t["year"].iloc[-1], t["share_beta0"].iloc[-1], "if rent per m² did not\nfall with size", MUTED)
    ax.plot([2024], [0.9], marker="D", color=ORANGE, markersize=7, markeredgecolor=SURFACE, linestyle="none")
    ax.annotate("UTAO: 0.9% of leases in the\n2024 tax base above EUR 2,300", (2024, 0.9), xytext=(-150, 30),
                textcoords="offset points", fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_ylabel("% of new leases")
    ax.set_xlim(2019.7, 2026.6)
    ax.set_ylim(0, 4.5)
    ax.set_xticks(range(2020, 2026))
    ax.text(2020.0, 4.25, "band: elasticity ±1.96 s.e., floor areas ±10%, dispersion +20%", fontsize=7.5, color=MUTED)
    finish(fig, "fig03_share_above.png", INE + "; UTAO, 13 February 2026")


def fig04_distribution() -> None:
    c = pd.read_csv(RESULTS / "national_cdf.csv")
    n = pd.read_csv(RESULTS / "national_distribution.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    suptitle(fig, "Modelled distribution of the monthly rents of new leases, Portugal")
    for y, col, at in (("2020", AQUA, 520), ("2025", BLUE, 1000)):
        ax.plot(c["rent"], 100 * (1 - c[y]), color=col)
        i = int(np.searchsorted(c["rent"], at))
        end_label(ax, c["rent"].iloc[i], 100 * (1 - c[y].iloc[i]), y, col, dy=8)
    ax.axvline(CEILING, color=ORANGE, linewidth=1.2)
    p = n.set_index("year")
    ax.text(CEILING * 1.04, 60, f"EUR 2,300\n{100 * (1 - p.loc[2025, 'pct_2300']):.1f}% of 2025 leases above", fontsize=8.5, color=INK_2)
    for y in (2025,):
        med = p.loc[y, "q50"]
        ax.axvline(med, color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(med * 1.03, 92, f"median {y}: EUR {med:,.0f}", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([200, 400, 700, 1000, 2300, 5000])
    ax.set_xticklabels(["200", "400", "700", "1,000", "2,300", "5,000"])
    ax.set_xlim(150, 6000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("monthly rent, EUR (log scale)")
    ax.set_ylabel("% of new leases with a higher rent")
    finish(fig, "fig04_distribution.png", INE)


def fig05_income() -> None:
    d = pd.read_csv(RESULTS / "income_ratios.csv", dtype={"dtmn": str})
    d = d[(d["year"] == 2024) & d["P80"].notna() & d["income_median"].notna()]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    suptitle(fig, "Gross declared income of fiscal households by municipality, 2024, against the ceiling")
    ax.scatter(d["income_median"], d["P80"], s=12, color=BLUE, edgecolor="none", alpha=0.8)
    ax.axhline(12 * CEILING, color=ORANGE, linewidth=1.2)
    ax.text(10100, 12 * CEILING + 900, "EUR 27,600: twelve months at EUR 2,300", fontsize=8.5, color=INK_2)
    ax.axhline(69000, color=PURPLE, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.text(10100, 69000 + 900, "EUR 69,000: the government's reference household (EUR 5,750 a month)", fontsize=8.5, color=INK_2)
    for name in ("Oeiras", "Lisboa", "Porto", "Odemira"):
        r = d[d["geodsg"] == name]
        if len(r):
            r = r.iloc[0]
            ax.annotate(name, (r["income_median"], r["P80"]), xytext=(6, 3), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("median income of a fiscal household, EUR a year")
    ax.set_ylabel("income at the 80th percentile, EUR a year")
    ax.set_xlim(10000, 23000)
    ax.set_ylim(15000, 75000)
    thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(thousands)
    ax.yaxis.set_major_formatter(thousands)
    finish(fig, "fig05_income.png", "INE, Estatísticas do Rendimento ao nível local (AT data)")


def fig06_by_type() -> None:
    t = pd.read_csv(RESULTS / "rent_by_type_2025.csv").set_index("geodsg")
    order = ["Cascais", "Lisboa", "Oeiras", "Almada", "Porto", "Sintra", "Funchal", "Coimbra", "Braga", "Portugal"]
    types = [("T0 ou T1", "T0-T1", AQUA), ("T2", "T2", BLUE), ("T3", "T3", ORANGE), ("T4 ou mais", "T4+", PURPLE)]
    fig, ax = plt.subplots(figsize=(9, 5.0))
    suptitle(fig, "Monthly rent of a typical new lease by dwelling type, 2025, at the local median rent per m²")
    for i, name in enumerate(order):
        if name not in t.index:
            continue
        vals = [t.loc[name, k] for k, _, _ in types if k in t.columns and pd.notna(t.loc[name, k])]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        for k, lab, col in types:
            if k in t.columns and pd.notna(t.loc[name, k]):
                ax.scatter(t.loc[name, k], i, s=46, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                           label=lab if i == 0 else None)
    ax.axvline(CEILING, color=INK, linewidth=1.2)
    ax.text(CEILING - 20, -0.55, "EUR 2,300", ha="right", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(300, 2500)
    ax.set_ylim(len(order) - 0.4, -0.9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("EUR a month (median rent per m² x Census median floor area of the type)")
    ax.legend(loc="lower right", bbox_to_anchor=(0.80, 0.02), ncol=2, fontsize=8.5, handletextpad=0.3, columnspacing=1.0)
    finish(fig, "fig06_by_type.png", INE)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fig01_area_needed()
    fig02_rule_over_time()
    fig03_share_above()
    fig04_distribution()
    fig05_income()
    fig06_by_type()


if __name__ == "__main__":
    main()
