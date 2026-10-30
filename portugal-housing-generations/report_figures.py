"""Overview figure for the research report (reads results.json and results_rental.json).

fig_report_national_overview.png   Prices, rents, income and the cost of entry, Portugal, 2008-2025

Needs matplotlib. Run after analysis.py and rental.py:
    python report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regional_figures import BLUE, GRID, INK, MUTED, ORANGE

HERE = Path(__file__).parent
OUT = HERE / "figures"


def series(d: dict[str, float], start: int = 2008) -> tuple[list[int], list[float]]:
    pairs = sorted((int(y), v) for y, v in d.items() if v is not None and int(y) >= start)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    nat = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    rent = json.loads((HERE / "results_rental.json").read_text(encoding="utf-8"))
    hpi = nat["yearly_hpi"]
    income = {str(int(y) - 1): v for y, v in nat["yearly_median_equivalised_income_eur"].items()}  # income year
    base = income["2015"]
    income_idx = {y: 100 * v / base for y, v in income.items()}
    rents = rent["stock"]["hicp_actual_rentals_2015_100"]
    rel = rent["stock"]["hicp_rentals_relative_to_all_items_2015_100"]
    cpi = {y: rents[y] * 100 / v for y, v in rel.items() if y in rents}

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 0.75]})
    for d, label, colour, ls, lw in ((hpi, "House prices (Eurostat HPI)", INK, "-", 2.4),
                                     (income_idx, "Median equivalised income (EU-SILC, income year)", BLUE, "--", 2.0),
                                     (rents, "Actual rentals (HICP 04.1, price index)", ORANGE, "-", 2.0),
                                     (cpi, "All consumer prices (HICP)", MUTED, ":", 1.8)):
        x, y = series(d)
        a.plot(x, y, color=colour, ls=ls, lw=lw, label=label)
        a.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    a.axhline(100, color=GRID, lw=1)
    a.set_xlim(2007.5, 2026.4)
    a.set_title("Indices, 2015 = 100")
    a.set_xticks(range(2008, 2026, 2))
    a.legend(loc="upper left", fontsize=7.6)
    pay = nat["initial_payment_to_income_index_2015_100"]["survey_year_alignment"]
    pir = nat["relative_price_income_index_2015_100"]
    x, y = series(pay)
    b.plot(x, y, color=INK, lw=2.3, label="first mortgage payment / income\n(30-year loan)")
    b.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    x, y = series(pir)
    b.plot(x, y, color=BLUE, lw=1.8, ls="--", label="house prices / income")
    b.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    b.axhline(100, color=GRID, lw=1)
    b.set_xlim(2007.5, 2026.4)
    b.set_title("Cost of entering ownership, 2015 = 100")
    b.set_xticks(range(2008, 2026, 2))
    b.legend(loc="upper left", fontsize=7.6)
    rates = nat["mortgage_cost_of_borrowing_annual_mean_pct"]
    x, y = series(rates)
    c.bar(x, y, color=ORANGE, width=0.72)
    for yr in (2008, 2015, 2021, 2023, 2025):
        c.annotate(f"{rates[str(yr)]:.1f}", (yr, rates[str(yr)]), xytext=(0, 3), textcoords="offset points",
                   ha="center", fontsize=7.6, color=INK)
    c.set_title("Mortgage rate, new loans (%)")
    c.set_xticks(range(2008, 2026, 4))
    c.set_ylim(0, 6.4)
    fig.suptitle("Portugal: house prices outpaced incomes and consumer prices after 2015",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Sources: Eurostat prc_hpi_a, ilc_di03, prc_hicp_aind; ECB MIR (Banco de Portugal). Nominal; "
             "survey-year alignment in the middle panel. Indices, not levels. Diogo Ribeiro.", fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(OUT / "fig_report_national_overview.png")
    plt.close(fig)
    print("wrote fig_report_national_overview.png")


if __name__ == "__main__":
    main()
