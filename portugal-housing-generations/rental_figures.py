"""Figures for the rental-market analysis (reads results_rental.json).

fig_rental_three_measures.png      Stock rent index, new-lease rents and income, indexed
fig_rental_tenure.png              Population by tenure (EU-SILC) and market renters among tenants
fig_rental_overburden.png          Overburden by tenure and a composition/within-tenure decomposition
fig_rental_regional_growth.png     New-lease rent growth against income growth, NUTS 3, 2020-2024
fig_rental_census_contracts.png    Census 2021 tenancies by contract period, and rents by contract period

Needs matplotlib (the analysis itself does not). Run after rental.py:
    python rental_figures.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regional_figures import AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE

HERE = Path(__file__).parent
OUT = HERE / "figures"
PURPLE, YELLOW, NAVY = "#7b5cd6", "#d9a400", "#3b4a6b"
FOCUS = {"1A0": ("Grande Lisboa", BLUE), "1B0": ("Península de Setúbal", AQUA),
         "11A": ("Área Metropolitana do Porto", ORANGE), "PT": ("Portugal", INK)}
SILC = "Source: Eurostat, EU-SILC"
PERIODS = {"Antes de 1987": "before 1987", "Entre 1987 e 1990": "1987-1990", "Entre 1991 e 2005": "1991-2005",
           "Entre 2006 e 2011": "2006-2011", "Entre 2012 e 2016": "2012-2016", "Entre 2017 e 2021": "2017-2021"}


def read_panel() -> dict[str, dict[int, dict[str, str]]]:
    """region -> year -> row of rental_panel.csv."""
    panel: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with (HERE / "rental_panel.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            panel[row["region_code"]][int(row["year"])] = row
    return panel


def series(d: dict[str, float], start: int | None = None) -> tuple[list[int], list[float]]:
    """Years and values of a {year: value} mapping (string keys), optionally from ``start``."""
    pairs = sorted((int(y), v) for y, v in d.items() if v is not None and (start is None or int(y) >= start))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def footer(fig: plt.Figure, text: str, y: float = 0.01) -> None:
    fig.text(0.01, y, text, fontsize=6.9, color=MUTED)


def measures_figure(r: dict[str, Any], panel: dict[str, dict[int, dict[str, str]]]) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    stock = r["stock"]["hicp_actual_rentals_2015_100"]
    income = r["affordability"]["hicp_rent_over_silc_income"]["2015"]["income_year_aligned"]
    # Rebuild the income index from the rent index and the ratio: Y = 100 * P / ratio.
    inc_idx = {y: 100 * stock[y] / v for y, v in income.items() if y in stock}
    rel = r["stock"]["hicp_rentals_relative_to_all_items_2015_100"]
    all_items = {y: stock[y] * 100 / v for y, v in rel.items() if y in stock}
    for d, label, colour, ls in ((stock, "Actual rentals (HICP 04.1, price index)", BLUE, "-"),
                                 (inc_idx, "Median equivalised income (EU-SILC, income year)", INK, "--"),
                                 (all_items, "All consumer prices (HICP)", MUTED, ":")):
        x, y = series(d, 2004)
        a.plot(x, y, color=colour, ls=ls, lw=2.1, label=label)
    a.axhline(100, color=GRID, lw=1)
    a.set_title("HICP rent index against income, 2015 = 100")
    a.legend(loc="upper left", fontsize=7.8)
    a.set_xticks(range(2005, 2026, 5))
    ends: list[tuple[float, str]] = []
    for code, (label, colour) in FOCUS.items():
        rows = panel[code]
        xr = [y for y in sorted(rows) if rows[y]["new_lease_rent_m2"]]
        xi = [y for y in sorted(rows) if rows[y]["income_gross_per_household"]]
        r0, i0 = float(rows[2020]["new_lease_rent_m2"]), float(rows[2020]["income_gross_per_household"])
        yr = [100 * float(rows[y]["new_lease_rent_m2"]) / r0 for y in xr]
        b.plot(xr, yr, color=colour, lw=2.2, marker="o", ms=3.5)
        b.plot(xi, [100 * float(rows[y]["income_gross_per_household"]) / i0 for y in xi], color=colour, lw=1.4, ls="--")
        ends.append((yr[-1], label))
    placed: list[float] = []
    for y_end, label in sorted(ends, reverse=True):
        y_text = y_end
        while any(abs(y_text - q) < 3.2 for q in placed):
            y_text -= 1.0
        placed.append(y_text)
        b.annotate(label, (2025, y_end), xytext=(2025.15, y_text), textcoords="data", fontsize=7.6, color=INK2, va="center")
    b.plot([], [], color=INK2, lw=2.2, label="median rent per m², new leases")
    b.plot([], [], color=INK2, lw=1.4, ls="--", label="median gross household income (IRS)")
    b.legend(loc="upper left", fontsize=7.6)
    b.set_xlim(2019.6, 2027.6)
    b.set_xticks([2020, 2021, 2022, 2023, 2024, 2025])
    b.set_title("New leases and income, 2020 = 100")
    fig.suptitle("Two rent statistics: a price index for rented dwellings and the median of new contracts",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    footer(fig, "Sources: Eurostat HICP (prc_hicp_aind) and EU-SILC (ilc_di03; income of the year before the survey); "
                "INE, Estatísticas de rendas da habitação ao nível local (method 2026) and Estatísticas do Rendimento ao nível local. Diogo Ribeiro.")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(OUT / "fig_rental_three_measures.png")
    plt.close(fig)


def tenure_figure(r: dict[str, Any]) -> None:
    shares = r["tenure"]["eu_silc_population_shares_pct"]["TOTAL"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    years = sorted(int(y) for y in shares["RENT_MKT"])
    layers = [("OWN_NL", "owner, no mortgage", "#c9d8ee"), ("OWN_L", "owner with mortgage", "#8fb2df"),
              ("RENT_FR", "tenant, reduced rent or free", YELLOW), ("RENT_MKT", "tenant, market rent", ORANGE)]
    stack = [[shares[k][str(y)] for y in years] for k, _l, _c in layers]
    a.stackplot(years, *stack, colors=[c for _k, _l, c in layers], labels=[lab for _k, lab, _c in layers], lw=0)
    a.set_ylim(0, 100)
    a.set_ylabel("% of population")
    a.legend(loc="lower left", fontsize=7.6, ncol=2, framealpha=0.9, frameon=True)
    a.set_title("Population by tenure status")
    a.grid(axis="x", visible=False)
    x, y = series(shares["RENT_MKT"])
    b.plot(x, y, color=ORANGE, lw=2.2, label="tenants at market rent, % of population")
    x, y = series(r["tenure"]["market_rent_tenants_as_share_of_renters_pct"])
    b.plot(x, y, color=INK, lw=2.0, ls="--", label="market-rent tenants, % of all tenants")
    cen = r["tenure"]["census_rented_share_pct"]["PT"]
    cx, cy = series(cen)
    b.plot(cx, cy, "s", color=PURPLE, ms=6, label="rented dwellings, % of occupied (censuses)")
    b.set_ylim(0, 62)
    b.legend(loc="upper left", fontsize=7.4)
    b.set_title("Who rents at market prices")
    fig.suptitle("Tenure in Portugal, 2004-2025: the share renting at market prices rose after 2021",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    footer(fig, f"{SILC} (ilc_lvho02; persons, survey years; 2020-2021 affected by the pandemic survey); INE censuses 1991-2021 "
                "(dwellings of usual residence). Diogo Ribeiro.")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(OUT / "fig_rental_tenure.png")
    plt.close(fig)


def overburden_figure(r: dict[str, Any]) -> None:
    o = r["overburden"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
    for key, label, colour, lw in (("RENT_MKT", "tenant, market rent", ORANGE, 2.3), ("RENT_FR", "tenant, reduced rent or free", YELLOW, 1.8),
                                   ("OWN_L", "owner with mortgage", BLUE, 1.8), ("OWN_NL", "owner, no mortgage", "#8fb2df", 1.6),
                                   ("TOTAL", "all persons", INK, 2.0)):
        x, y = series(o["eu_silc_by_tenure_pct"][key])
        a.plot(x, y, color=colour, lw=lw, ls="--" if key == "TOTAL" else "-", label=label, marker="o", ms=2.5)
    a.set_ylabel("% living in overburdened households")
    a.set_xticks(range(2005, 2026, 5))
    a.legend(loc="upper left", fontsize=7.4)
    a.set_title("Housing-cost overburden rate by tenure")
    windows = list(o["shift_share"])
    xs = range(len(windows))
    comp = [o["shift_share"][w]["composition"] for w in windows]
    within = [o["shift_share"][w]["within_groups"] for w in windows]
    b.bar([x - 0.18 for x in xs], comp, width=0.36, color=ORANGE, label="tenure composition")
    b.bar([x + 0.18 for x in xs], within, width=0.36, color=BLUE, label="rates within each tenure")
    for pos, w in zip(xs, windows, strict=True):
        b.plot(pos, o["shift_share"][w]["change"], "D", color=INK, ms=6, zorder=4)
    b.plot([], [], "D", color=INK, label="total change")
    b.axhline(0, color=INK2, lw=0.8)
    b.set_xticks(list(xs))
    b.set_xticklabels([w.replace("_", "–") for w in windows])
    b.set_ylabel("percentage points")
    b.legend(loc="lower left", fontsize=7.4)
    b.grid(axis="x", visible=False)
    b.set_title("What moved the total (decomposition)")
    fig.suptitle("Overburden is concentrated among market tenants, and more people now rent at market prices",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    footer(fig, f"{SILC} (ilc_lvho07c, ilc_lvho02). Overburden: housing costs above 40% of disposable income. "
                "Kitagawa decomposition with mean weights; descriptive, not causal. Diogo Ribeiro.")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(OUT / "fig_rental_overburden.png")
    plt.close(fig)


def regional_figure(r: dict[str, Any]) -> None:
    regions = r["regions"]
    rows = [(k, v["name"], v["growth_2020_2024"]["price_annual_pct"], v["growth_2020_2024"]["income_annual_pct"])
            for k, v in regions.items() if v["level"] == "NUTS3" and v["growth_2020_2024"]]
    rows.append(("PT", "Portugal", regions["PT"]["growth_2020_2024"]["price_annual_pct"],
                 regions["PT"]["growth_2020_2024"]["income_annual_pct"]))
    rows.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(8.8, 7.8))
    for i, (code, _name, rent, inc) in enumerate(rows):
        ax.plot([inc, rent], [i, i], color=GRID, lw=2.4, zorder=1)
        ax.plot(inc, i, "o", color=MUTED, ms=6, zorder=2)
        ax.plot(rent, i, "o", color=FOCUS[code][1] if code in FOCUS else NAVY, ms=7, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for _c, n, _r, _i in rows], fontsize=8.2)
    for tick, (code, *_rest) in zip(ax.get_yticklabels(), rows, strict=True):
        if code in FOCUS:
            tick.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Average annual change, 2020-2024 (%)")
    ax.plot([], [], "o", color=NAVY, label="median rent per m² of new leases")
    ax.plot([], [], "o", color=MUTED, label="median gross household income")
    ax.legend(loc="lower right", fontsize=8.2)
    fig.suptitle("New-lease rents and income by NUTS 3 region, 2020-2024", x=0.01, y=0.99, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    n = r["new_leases"]["nuts3"]
    fig.text(0.01, 0.95, f"Rents rose faster than incomes in {n['price_outpaced_income']} of {n['regions']} regions; "
             "Lisbon, Setúbal, Porto and Portugal coloured", fontsize=8.2, color=INK2)
    footer(fig, "Source: INE, Estatísticas de rendas da habitação ao nível local (method 2026) and Estatísticas do Rendimento\n"
                "ao nível local (IRS), NUTS 2024. Medians of different populations. Diogo Ribeiro.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT / "fig_rental_regional_growth.png")
    plt.close(fig)


def census_figure(r: dict[str, Any]) -> None:
    t = r["tenure"]["census2021_tenancies"]
    codes = ["PT", "11", "19", "1D", "1A", "1B", "1C", "15", "20", "30", "11A"]
    codes = [c for c in codes if c in t]
    periods = list(t["PT"]["contract_period_pct"])
    colours = ["#3b4a6b", "#5f6f91", "#8fa1c2", "#c3cfe3", "#f2b48f", ORANGE]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, code in enumerate(codes):
        left = 0.0
        for p, colour in zip(periods, colours, strict=True):
            w = t[code]["contract_period_pct"][p]
            a.barh(i, w, left=left, color=colour, height=0.7, label=PERIODS[p] if i == 0 else None)
            left += w
    a.set_yticks(range(len(codes)))
    a.set_yticklabels([("Portugal" if c == "PT" else t[c]["name"]) + (" (NUTS 3)" if c == "11A" else "") for c in codes], fontsize=8.2)
    a.invert_yaxis()
    a.set_xlim(0, 100)
    a.set_xlabel("% of rented dwellings of usual residence")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.4, title="contract signed", title_fontsize=7.6)
    a.grid(axis="y", visible=False)
    a.set_title("Tenancies by date of contract, 2021")
    bands = r["tenure"]["census2021_rent_band_by_contract_period_portugal"]
    labels = [p for p in bands if p != "Total"]
    low = [bands[p]["share_below_100_eur_pct"] for p in labels]
    high = [bands[p]["share_500_eur_or_more_pct"] for p in labels]
    ys = range(len(labels))
    b.barh([y - 0.18 for y in ys], low, height=0.36, color="#5f6f91", label="rent below €100 a month")
    b.barh([y + 0.18 for y in ys], high, height=0.36, color=ORANGE, label="rent €500 a month or more")
    b.set_yticks(list(ys))
    b.set_yticklabels([PERIODS[p] for p in labels], fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("% of tenancies signed in each period")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=7.4)
    b.grid(axis="y", visible=False)
    b.set_title("Rents by contract date, Portugal")
    fig.suptitle("The stock of tenancies mixes old and new contracts with very different rents",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    footer(fig, "Source: INE, Censos 2021 (rented classic dwellings of usual residence), NUTS 2024. Diogo Ribeiro.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT / "fig_rental_census_contracts.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    r = json.loads((HERE / "results_rental.json").read_text(encoding="utf-8"))
    measures_figure(r, read_panel())
    tenure_figure(r)
    overburden_figure(r)
    regional_figure(r)
    census_figure(r)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig_rental_*.png"))))


if __name__ == "__main__":
    main()
