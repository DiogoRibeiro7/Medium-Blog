"""Figures for the integrated assessment (reads integrated_panel.csv and results_integrated.json).

fig_integrated_coverage.png         Municipal coverage of each variable by year, with method breaks
fig_integrated_cross_section.png    Price growth against income, construction, population and rents (municipalities)
fig_integrated_specifications.png   Within-municipality estimates across specifications, with wild-bootstrap intervals
fig_integrated_influence.png        Stability of the main estimates to dropping regions and influential municipalities
fig_integrated_national.png         National annual growth: house prices against income, construction, population, rents

Needs matplotlib (the analysis itself does not). Run after integrated.py:
    python integrated_figures.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import integrated as I
from regional_figures import AQUA, BLUE, GREY, GRID, INK, INK2, MUTED, ORANGE

HERE = Path(__file__).parent
OUT = HERE / "figures"
VERDICT_COLOUR = {"robust": BLUE, "fragile": ORANGE, "not supported": MUTED}
SOURCE = "Source: INE (municipal statistics, NUTS 2024 by DICO); Eurostat. Diogo Ribeiro."


def read_panel() -> dict[tuple[str, int], dict[str, str]]:
    with (HERE / "integrated_panel.csv").open(newline="", encoding="utf-8") as fh:
        return {(r["dico"], int(r["year"])): r for r in csv.DictReader(fh)}


def num(panel: dict[tuple[str, int], dict[str, str]], u: str, t: int, col: str) -> float | None:
    cell = panel.get((u, t), {}).get(col, "")
    return float(cell) if cell else None


def lg(panel: dict[tuple[str, int], dict[str, str]], u: str, col: str, a: int, b: int) -> float | None:
    va, vb = num(panel, u, a, col), num(panel, u, b, col)
    return 100 * math.log(vb / va) if va and vb else None


def coverage_figure(results: dict[str, Any]) -> None:
    panel = read_panel()
    names = list(I.VARIABLES)
    years = I.YEARS
    grid = [[sum(1 for (u, t), r in panel.items() if t == y and r[n]) / 308 for y in years] for n in names]
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    cmap = LinearSegmentedColormap.from_list("cov", ["#ffffff", "#c9d8ee", BLUE])
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v > 0:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.6, color=INK if v < 0.8 else "white")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.4)
    ax.grid(False)
    ax.axvline(years.index(2021) - 0.5, color=ORANGE, lw=1.6)
    ax.text(years.index(2021) - 0.45, -0.75, "population: census-based | administrative-based", fontsize=7.2, color=ORANGE, ha="center")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_ticklabels(["0", "20", "40", "60", "80", "100"])
    cb.set_label("% of 308 municipalities", fontsize=8)
    ax.set_title("Municipal coverage of each variable (cells: % of municipalities with a published value)", pad=16)
    fig.text(0.01, 0.01, SOURCE + " Separate rows are separate methods; none is spliced.", fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_integrated_coverage.png")
    plt.close(fig)


def cross_section_figure(results: dict[str, Any]) -> None:
    panel = read_panel()
    units = sorted({u for u, _t in panel})
    nuts3 = {u: panel[(u, 2024)]["nuts3"] for u in units if (u, 2024) in panel}
    stats = results["cross_section_municipalities"]["changes_2019_2024"]

    def comp_rate(u: str) -> float | None:
        pop = num(panel, u, 2019, "population")
        vals = [num(panel, u, t, "completions") for t in range(2019, 2025)]
        return 1000 * sum(v for v in vals if v is not None) / 6 / pop if pop and all(v is not None for v in vals) else None

    panels = [
        ("price_growth_vs_income_growth", lambda u: lg(panel, u, "income", 2019, 2024), lambda u: lg(panel, u, "price_m2", 2019, 2024),
         "Income growth 2019-2024 (log points)"),
        ("price_growth_vs_completions_per_1000", comp_rate, lambda u: lg(panel, u, "price_m2", 2019, 2024),
         "Completions per 1,000 residents a year, 2019-2024"),
        ("price_growth_vs_population_growth_2021_2024", lambda u: lg(panel, u, "population", 2021, 2024), lambda u: lg(panel, u, "price_m2", 2019, 2024),
         "Population growth 2021-2024 (log points)"),
        ("price_growth_vs_rent_growth_2020_2024", lambda u: lg(panel, u, "rent_m2", 2020, 2024), lambda u: lg(panel, u, "price_m2", 2020, 2024),
         "New-lease rent growth 2020-2024 (log points)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))
    for ax, (key, fx, fy, xlabel) in zip(axes.flat, panels, strict=True):
        pts = [(fx(u), fy(u), nuts3.get(u)) for u in units]
        pts = [(x, y, r) for x, y, r in pts if x is not None and y is not None]
        for x, y, r in pts:
            ax.plot(x, y, "o", ms=3.4, color=ORANGE if r in I.METRO else GREY, alpha=0.85, mec="none")
        s = stats[key]
        ax.set_xlabel(xlabel, fontsize=8.6)
        ax.set_ylabel("Sale-price growth (log points)" if "rent" not in key else "Sale-price growth 2020-2024", fontsize=8.6)
        ax.set_title(f"Spearman {s['spearman']:+.2f} [{s['spearman_ci95_low']:+.2f}, {s['spearman_ci95_high']:+.2f}], n = {int(s['n'])}",
                     fontsize=9.2)
    axes.flat[0].plot([], [], "o", color=ORANGE, label="Lisbon, Setúbal and Porto metro municipalities")
    axes.flat[0].plot([], [], "o", color=GREY, label="other municipalities")
    axes.flat[0].legend(loc="upper right", fontsize=7.4)
    fig.suptitle("Municipal sale-price growth against income, construction, population and rents",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, SOURCE + " Intervals: 95%, bootstrap resampling whole NUTS 3 regions. Associations, not effects.",
             fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(OUT / "fig_integrated_cross_section.png")
    plt.close(fig)


def specifications_figure(results: dict[str, Any]) -> None:
    across = results["across_specifications"]
    titles = {"income": "Income (elasticity)", "completions_per_1000": "Completions per 1,000 residents (semi-elasticity)",
              "population": "Population (elasticity)", "rent": "New-lease rent (elasticity)"}
    fig, axes2 = plt.subplots(2, 2, figsize=(12.4, 9.6), gridspec_kw={"height_ratios": [1.8, 1]})
    axes = list(axes2.flat)
    for ax, (concept, title) in zip(axes, titles.items(), strict=True):
        rows = across[concept]["estimates"]
        for i, r in enumerate(rows):
            lo, hi = r["ci95_wild_bootstrap"]
            excl = lo is not None and (lo > 0 or hi < 0)
            colour = BLUE if excl else MUTED
            if lo is not None:
                ax.plot([lo, hi], [i, i], color=colour, lw=1.6)
            ax.plot(r["estimate"], i, "o", color=colour, ms=5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["spec"].replace("_", " ") + ("  [existing]" if "existing" in r["outcome"] else "")
                            for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9.2)
    axes[0].plot([], [], "o-", color=BLUE, label="95% interval excludes zero")
    axes[0].plot([], [], "o-", color=MUTED, label="includes zero")
    axes[0].legend(loc="upper right", fontsize=7.4)
    fig.suptitle("Within-municipality associations with house prices, across samples, definitions and transforms",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "A: 2022-method sales 2019-2024 (A2: 2021-2024); B: 2018-method sales 2016-2020; C: bank appraisals 2015-2024; "
             "D: new-lease rents 2020-2024; LD: long differences. Wild cluster bootstrap by NUTS 3. Diogo Ribeiro.",
             fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(OUT / "fig_integrated_specifications.png")
    plt.close(fig)


def influence_figure(results: dict[str, Any]) -> None:
    items = [("A1_twoway_income", "log_income", "Income\n(sales 2019-2024)"),
             ("A2_twoway_income_completions", "completions_per_1000", "Completions per 1,000\n(sales 2021-24; existing: 0.000)"),
             ("A2_twoway_full", "log_population", "Population\n(sales 2021-2024)"),
             ("D_twoway_price_on_rent", "log_rent_m2", "New-lease rent\n(2020-2024)")]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.0))
    for ax, (model, x, title) in zip(axes, items, strict=True):
        m = results["models"][model]["coefficients"][x]
        d = results["diagnostics"][model]
        v = results["verdicts"][f"{model}:{x}"]
        rows = [("main estimate", m["estimate"], m["ci95_cr1"]),
                ("leave one NUTS 3 out (range)", None, [d["leave_one_nuts3_out"][x]["min"], d["leave_one_nuts3_out"][x]["max"]]),
                ("without Lisbon/Porto metro", d["without_lisbon_and_porto_metro"][x]["estimate"], d["without_lisbon_and_porto_metro"][x]["ci95_cr1"]),
                ("without top 5% influential", d["without_top5pct_influential"][x]["estimate"], d["without_top5pct_influential"][x]["ci95_cr1"]),
                ("first differences", d["fe_vs_fd"][x]["fd"], None)]
        colour = VERDICT_COLOUR[v["verdict"]]
        for i, (_label, est, ci) in enumerate(rows):
            if ci is not None:
                ax.plot(ci, [i, i], color=colour, lw=2 if est is None else 1.4, alpha=0.9)
            if est is not None:
                ax.plot(est, i, "o", color=colour, ms=5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows] if ax is axes[0] else [""] * len(rows), fontsize=7.6)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{title}\nverdict: {v['verdict']}", fontsize=8.6)
    fig.suptitle("How stable are the main estimates?", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Two-way fixed effects (municipality and year); intervals CR1 by municipality. Colour: blue robust, "
             "orange fragile. Diogo Ribeiro.", fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "fig_integrated_influence.png")
    plt.close(fig)


def national_figure(results: dict[str, Any]) -> None:
    nat = I.national_series(I.DATA, I.DATE)
    g = {k: I.growth_rates(v, log=k != "mortgage_rate") for k, v in nat.items()}
    corr = results["national_time_series"]["2005_2024"]
    pairs = [("income_silc_income_year", "Median equivalised income (EU-SILC)"), ("completions", "Completed dwellings"),
             ("population_end_of_year", "Population"), ("rents_hicp", "Actual rentals (HICP price index)")]
    fig, axes2 = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True)
    axes = list(axes2.flat)
    years = [y for y in range(2009, 2025) if y in g["house_prices_hpi"]]
    for ax, (key, label) in zip(axes, pairs, strict=True):
        ax.plot(years, [g["house_prices_hpi"][y] for y in years], color=INK, lw=2, label="house prices (HPI)")
        oy = [y for y in years if y in g[key]]
        ax2 = ax.twinx() if key in ("completions", "population_end_of_year") else ax
        ax2.plot(oy, [g[key][y] for y in oy], color=BLUE if ax2 is ax else AQUA, lw=1.8, ls="--", label=label)
        if ax2 is not ax:
            ax2.tick_params(axis="y", colors=AQUA, labelsize=7.4)
            ax2.spines["right"].set_visible(True)
            ax2.grid(False)
        c = corr[key]
        ax.set_title(f"{label}\nr = {c['pearson']:+.2f}; effective n = {c['effective_n']:.0f}; 95% [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]",
                     fontsize=9)
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xticks([2010, 2015, 2020])
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("annual change (log points)")
    axes[0].legend(loc="upper left", fontsize=7.4)
    fig.suptitle("Portugal: annual growth of house prices and of possible correlates, 2009-2024",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Sources: Eurostat prc_hpi_a, ilc_di03 (income of the year before the survey), demo_pjan, prc_hicp_aind; "
             "INE/PORDATA completions. Completions and population on the right axis. Diogo Ribeiro.", fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "fig_integrated_national.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results = json.loads((HERE / "results_integrated.json").read_text(encoding="utf-8"))
    coverage_figure(results)
    cross_section_figure(results)
    specifications_figure(results)
    influence_figure(results)
    national_figure(results)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig_integrated_*.png"))))


if __name__ == "__main__":
    main()
