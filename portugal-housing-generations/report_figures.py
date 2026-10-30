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

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    for d, label, colour, ls, lw in ((hpi, "House prices (Eurostat HPI)", INK, "-", 2.4),
                                     (income_idx, "Median equivalised income (EU-SILC, income year)", BLUE, "--", 2.0),
                                     (rents, "Rents paid by sitting tenants (HICP 04.1)", ORANGE, "-", 2.0),
                                     (cpi, "All consumer prices (HICP)", MUTED, ":", 1.8)):
        x, y = series(d)
        a.plot(x, y, color=colour, ls=ls, lw=lw, label=label)
        a.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=colour, va="center")
    a.axhline(100, color=GRID, lw=1)
    a.set_xlim(2007.5, 2026.4)
    a.set_title("Indices, 2015 = 100")
    a.set_xticks(range(2008, 2026, 2))
    a.legend(loc="upper left", fontsize=7.8)
    pay = nat["initial_payment_to_income_index_2015_100"]["survey_year_alignment"]
    pir = nat["relative_price_income_index_2015_100"]
    x, y = series(pay)
    b.plot(x, y, color=INK, lw=2.3, label="first mortgage payment / income (30-year loan)")
    x, y = series(pir)
    b.plot(x, y, color=BLUE, lw=1.8, ls="--", label="house prices / income")
    b.axhline(100, color=GRID, lw=1)
    b.set_ylabel("2015 = 100")
    b2 = b.twinx()
    rates = nat["mortgage_cost_of_borrowing_annual_mean_pct"]
    x, y = series(rates)
    b2.bar(x, y, color=ORANGE, alpha=0.25, width=0.7, label="mortgage rate on new loans (%)")
    b2.set_ylabel("% a year", color=ORANGE)
    b2.tick_params(axis="y", colors=ORANGE)
    b2.spines["right"].set_visible(True)
    b2.grid(False)
    b2.set_ylim(0, 12)
    b.set_zorder(b2.get_zorder() + 1)
    b.patch.set_visible(False)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6)
    b.set_title("The cost of entering ownership, 2015 = 100")
    b.set_xticks(range(2008, 2026, 2))
    fig.suptitle("Portugal: house prices outpaced incomes and consumer prices after 2015",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Sources: Eurostat prc_hpi_a, ilc_di03, prc_hicp_aind; ECB MIR (Banco de Portugal). Nominal; "
             "survey-year alignment in the right panel. Indices, not levels. Diogo Ribeiro.", fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(OUT / "fig_report_national_overview.png")
    plt.close(fig)
    print("wrote fig_report_national_overview.png")


if __name__ == "__main__":
    main()
