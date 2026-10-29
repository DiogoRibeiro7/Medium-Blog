"""Figures for the affordability panel (reads affordability_panel.csv and results_affordability.json).

fig_affordability_price_vs_income_growth.png  Price and income growth 2019-2024, NUTS 3 and municipalities
fig_affordability_m2_per_income_year.png      m2 per year of median income, NUTS 3, 2019 and 2024
fig_affordability_index_base_years.png        Price/income index for Lisbon, Porto and Portugal under three base years
fig_affordability_definitions.png             Price-minus-income growth by NUTS 2 under other price and income definitions

Needs matplotlib (the analysis itself does not). Run after affordability.py:
    python affordability_figures.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regional_figures import AQUA, BLUE, GREY, GRID, INK, INK2, MUTED, ORANGE

HERE = Path(__file__).parent
OUT = HERE / "figures"
START, END = 2019, 2024
HIGHLIGHT = {"1A0": ("Grande Lisboa", BLUE), "1B0": ("Península de Setúbal", AQUA),
             "11A": ("Área Metropolitana do Porto", ORANGE), "PT": ("Portugal", INK)}
SOURCE = ("Source: INE, Estatísticas de preços da habitação ao nível local (2022 method; 12 months to Q4) and "
          "Estatísticas do Rendimento ao nível local (IRS), NUTS 2024. Diogo Ribeiro.")
PROXY = "Proxy, not years to buy: ignores deposit, credit, taxes and dwelling size."
LABEL_OFFSET = {"1A0": (6, 5, "left"), "PT": (7, -10, "left")}

Panel = dict[str, dict[int, dict[str, str]]]


def read_panel() -> Panel:
    """region -> year -> row of affordability_panel.csv."""
    panel: Panel = defaultdict(dict)
    with (HERE / "affordability_panel.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            panel[row["region_code"]][int(row["year"])] = row
    return panel


def value(panel: Panel, region: str, year: int, column: str) -> float | None:
    """One cell as a float, or None when INE published no value."""
    cell = panel[region].get(year, {}).get(column, "")
    return float(cell) if cell else None


def annual(panel: Panel, region: str, column: str) -> float | None:
    """Compound annual growth START-END in percent."""
    a, b = value(panel, region, START, column), value(panel, region, END, column)
    return 100 * ((b / a) ** (1 / (END - START)) - 1) if a and b else None


def gap(panel: Panel, region: str, price_col: str, income_col: str) -> float | None:
    """Price minus income growth, START-END, in log points."""
    cells = [value(panel, region, y, c) for y in (START, END) for c in (price_col, income_col)]
    if any(c is None for c in cells):
        return None
    p0, y0, p1, y1 = (c for c in cells if c is not None)
    return 100 * (math.log(p1 / p0) - math.log(y1 / y0))


def growth_figure(panel: Panel, results: dict) -> None:
    regions = results["regions"]
    nuts3 = [r for r, v in regions.items() if v["level"] == "NUTS3"]
    munis = [r for r, v in regions.items() if v["level"] == "municipality"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
    for ax, units, title in ((axes[0], nuts3 + ["PT"], "NUTS 3 regions and Portugal"), (axes[1], munis, "Municipalities with both years published")):
        pts: list[tuple[str, float, float]] = []
        for r in units:
            x, y = annual(panel, r, "income_gross_per_household"), annual(panel, r, "price_m2_total")
            if x is not None and y is not None:
                pts.append((r, x, y))
        x_lo, x_hi = min(x for _r, x, _y in pts) - 0.6, max(x for _r, x, _y in pts) + 0.6
        y_lo, y_hi = min(min(y for _r, _x, y in pts), x_lo) - 1, max(y for _r, _x, y in pts) + 1.5
        ax.fill_between([x_lo, x_hi], [x_lo, x_hi], [y_hi, y_hi], color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.plot([x_lo, x_hi], [x_lo, x_hi], color=INK2, lw=0.9, ls="--", zorder=1)
        ax.text(x_lo + 0.1, y_hi - 0.3, "above the line: prices grew faster than incomes", fontsize=8, color=INK2, va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5}, zorder=5)
        for r, x, y in pts:
            if r in HIGHLIGHT:
                continue
            ax.plot(x, y, "o", color=MUTED if units is not munis else GREY, ms=5.5 if units is not munis else 3.2,
                    alpha=1 if units is not munis else 0.8, zorder=2, mec="none")
        for r, x, y in pts:
            if r in HIGHLIGHT:
                ax.plot(x, y, "o" if r != "PT" else "D", color=HIGHLIGHT[r][1], ms=8, zorder=4, mec="white", mew=0.8)
                dx, dy, ha = LABEL_OFFSET.get(r, (6, -3, "left"))
                ax.annotate(HIGHLIGHT[r][0], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                            color=INK2, ha=ha, zorder=6)
        if units is munis:
            for r in ("1A0", "1B0", "11A"):  # mark the regional value for reference
                x, y = annual(panel, r, "income_gross_per_household"), annual(panel, r, "price_m2_total")
                if x is not None and y is not None:
                    ax.plot(x, y, "o", color=HIGHLIGHT[r][1], ms=8, zorder=4, mec="white", mew=0.8)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(f"Median gross household income, annual change {START}-{END} (%)")
        ax.set_title(f"{title} (n = {len(pts)})")
    axes[0].set_ylabel(f"Median price per m² of dwellings sold, annual change {START}-{END} (%)")
    fig.suptitle(f"House prices and declared income, average annual change {START}-{END}",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, SOURCE + " Coloured dots on the right are the regional values.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "fig_affordability_price_vs_income_growth.png")
    plt.close(fig)


def m2_figure(panel: Panel, results: dict) -> None:
    regions = results["regions"]
    rows = [(r, v["name"], v[f"m2_per_income_year_{START}"], v[f"m2_per_income_year_{END}"])
            for r, v in regions.items() if v["level"] == "NUTS3" or r == "PT"]
    rows = [t for t in rows if t[2] is not None and t[3] is not None]
    rows.sort(key=lambda t: t[3])
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    for i, (r, _name, a, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color=GRID, lw=2.4, zorder=1)
        ax.plot(a, i, "o", color=MUTED, ms=6, zorder=2)
        ax.plot(b, i, "o", color=HIGHLIGHT[r][1] if r in HIGHLIGHT else BLUE, ms=7, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for _r, n, _a, _b in rows], fontsize=8.3)
    for tick, (r, *_rest) in zip(ax.get_yticklabels(), rows, strict=True):
        if r in HIGHLIGHT:
            tick.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=0)
    ax.set_xlabel("m² at the median price per m² paid for by one year of median gross household income")
    ax.plot([], [], "o", color=MUTED, label=str(START))
    ax.plot([], [], "o", color=BLUE, label=str(END))
    ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Square metres per year of median income, NUTS 3 regions, {START} and {END}",
                 x=0.01, y=0.99, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.948, "Floor area at the regional median price per m² that one year of median gross household "
             "income would pay for; lower = prices high relative to income", fontsize=8, color=INK2)
    fig.text(0.01, 0.01, SOURCE + "\n" + PROXY, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "fig_affordability_m2_per_income_year.png")
    plt.close(fig)


def index_figure(results: dict) -> None:
    paths = results["sensitivity_base_year"]["focus_index_paths"]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.6), sharey=True)
    styles = {"2019": (INK, "-"), "2020": (BLUE, "--"), "2021": (ORANGE, ":")}
    for ax, r in zip(axes, HIGHLIGHT, strict=True):
        for base, (colour, ls) in styles.items():
            series = paths[base][r]
            ax.plot([int(y) for y in series], list(series.values()), color=colour, ls=ls, lw=2, label=f"base {base} = 100")
        ax.axhline(100, color=INK2, lw=0.7)
        ax.set_title(HIGHLIGHT[r][0], fontsize=9.5)
        ax.set_xticks(range(START, END + 1, 1))
        ax.set_xticklabels([str(y)[2:] for y in range(START, END + 1)])
    axes[0].set_ylabel("Price index / income index × 100")
    axes[0].legend(loc="upper left", fontsize=7.6)
    fig.suptitle("Relative price-to-income index under three base years (a change, not a level)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, SOURCE, fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "fig_affordability_index_base_years.png")
    plt.close(fig)


def definitions_figure(panel: Panel, results: dict) -> None:
    regions = results["regions"]
    rows = [r for r, v in regions.items() if v["level"] == "NUTS2"] + ["PT"]
    head = {r: gap(panel, r, "price_m2_total", "income_gross_per_household") for r in rows}
    rows.sort(key=lambda r: head.get(r) or 0.0)
    variants = (
        ("Which price (income: gross per household)", "income", [("price_m2_total", "all dwellings", INK, "o"),
                                                                  ("price_m2_existing", "existing", BLUE, "s"),
                                                                  ("price_m2_new", "new", ORANGE, "^")]),
        ("Which income (price: all dwellings)", "price", [("income_gross_per_household", "gross per household", INK, "o"),
                                                          ("income_net_per_household", "net of IRS per household", BLUE, "s"),
                                                          ("income_gross_per_taxpayer", "gross per taxpayer", ORANGE, "^")]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, (title, fixed, options) in zip(axes, variants, strict=True):
        for k, (col, label, colour, marker) in enumerate(options):
            xs, ys = [], []
            for i, r in enumerate(rows):
                g = gap(panel, r, col, "income_gross_per_household") if fixed == "income" else gap(panel, r, "price_m2_total", col)
                if g is not None:
                    xs.append(g)
                    ys.append(i + (k - 1) * 0.18)
            ax.plot(xs, ys, marker, color=colour, ms=6, ls="none", label=label)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=9.6)
        ax.set_xlabel(f"Price growth minus income growth, {START}-{END} (log points)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([regions[r]["name"] for r in rows], fontsize=8.4)
    fig.suptitle("Is the price-income gap robust to the definitions? NUTS 2 regions",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, SOURCE + " Gaps right of zero: prices rose faster than incomes.", fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT / "fig_affordability_definitions.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panel = read_panel()
    results = json.loads((HERE / "results_affordability.json").read_text(encoding="utf-8"))
    growth_figure(panel, results)
    m2_figure(panel, results)
    index_figure(results)
    definitions_figure(panel, results)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig_affordability_*.png"))))


if __name__ == "__main__":
    main()
