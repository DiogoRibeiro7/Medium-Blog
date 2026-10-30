"""Every number quoted in REPORT.md that is checked here is recomputed from the result files.

Each claim is (text, expected, decimals, value):
- ``text`` must appear in REPORT.md (whitespace-insensitive), so the test fails if the wording is edited
  without updating the claim;
- the expected number, formatted with ``decimals`` places, must appear inside ``text``;
- ``value()`` (read from the results, or computed from them) must round to ``expected``.

A further test checks that every figure and local link in the report exists.
"""

from __future__ import annotations

import csv
import json
import math
import re
import unittest
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
REPORT = HERE / "REPORT.md"


@cache
def results(name: str) -> Any:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def get(name: str, path: str) -> Any:
    """Value at a dotted path in a result file (path segments that are digits index lists)."""
    node = results(name)
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def nat(path: str) -> Callable[[], float]:
    return lambda: float(get("results.json", path))


def reg(path: str) -> Callable[[], float]:
    return lambda: float(get("results_regional.json", path))


def aff(path: str) -> Callable[[], float]:
    return lambda: float(get("results_affordability.json", path))


def ren(path: str) -> Callable[[], float]:
    return lambda: float(get("results_rental.json", path))


def itg(path: str) -> Callable[[], float]:
    return lambda: float(get("results_integrated.json", path))


def spec(regressor: str, name: str, field: int | None = None) -> Callable[[], float]:
    """Estimate (field None) or wild-bootstrap bound (0, 1) of one specification in across_specifications."""
    def value() -> float:
        for row in get("results_integrated.json", f"across_specifications.{regressor}.estimates"):
            if row["spec"] == name:
                return float(row["estimate"] if field is None else row["ci95_wild_bootstrap"][field])
        raise KeyError(name)
    return value


def hicp_all_items(year: str) -> float:
    """HICP all items (2015 = 100) recovered from the rent index and rents relative to all items."""
    stock = get("results_rental.json", "stock")
    return float(stock["hicp_actual_rentals_2015_100"][year]) * 100 / float(stock["hicp_rentals_relative_to_all_items_2015_100"][year])


def income_year(year: int) -> float:
    """EU-SILC median equivalised income of an income year (published with the next year's survey)."""
    return float(get("results.json", f"yearly_median_equivalised_income_eur.{year + 1}"))


def new_dwelling_premium_pct(region: str = "PT", year: str = "2024") -> float:
    """Median price per m2 of new dwellings over that of existing dwellings, minus one, in %."""
    with (HERE / "affordability_panel.csv").open(encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["region_code"] == region and r["year"] == year)
    return 100 * (float(row["price_m2_new"]) / float(row["price_m2_existing"]) - 1)


def m2(name: str, path: str, scale: float = 1.0) -> Callable[[], float]:
    return lambda: scale * float(get(name, path))


CXM = "cross_section_municipalities"
SP = f"{CXM}.changes_2019_2024"
SB = f"{CXM}.changes_2016_2020_2018_method"
SC = f"{CXM}.changes_2015_2024_appraisal"
LV = f"{CXM}.levels_2024"
IMP = "initial_payment_to_income_index_2015_100.survey_year_alignment"
DEC = "initial_payment_to_income_change_log_points.survey_year_alignment"
RG = "regions"
TS = "national_time_series"

CLAIMS: list[tuple[str, float, int, Callable[[], float]]] = [
    # Abstract and national results
    ("rose 164% between 2015 and 2025", 164, 0, nat("descriptives.growth_2015_2025_pct")),
    ("or 111% after consumer-price inflation", 111, 0,
     lambda: 100 * (get("results.json", "yearly_hpi.2025") / hicp_all_items("2025") - 1)),
    ("Consumer prices rose 25%", 25, 0, lambda: hicp_all_items("2025") - 100),
    ("Median equivalised income rose 65% between income years 2015 and 2024", 65, 0,
     lambda: 100 * (income_year(2024) / income_year(2015) - 1)),
    ("(index 153.9 in 2025, 2015 = 100)", 153.9, 1, nat("relative_price_income_index_2015_100.2025")),
    ("it was 136.3 in 2024", 136.3, 1,
     lambda: get("results.json", "yearly_hpi.2024") / (income_year(2024) / income_year(2015))),
    ("reached its highest value of the series in 2023 (183.1)", 183.1, 1, nat(f"{IMP}.2023")),
    ("It remained 66% above 2015 in 2025", 66, 0, lambda: get("results.json", f"{IMP}.2025") - 100),
    ("falls from 107.5 (2008) to 93.1 (2013)", 107.5, 1, nat("yearly_hpi.2008")),
    ("falls from 107.5 (2008) to 93.1 (2013)", 93.1, 1, nat("yearly_hpi.2013")),
    ("completions fell from 125,708 dwellings in 2002", 125708, 0, nat("completed_new_family_dwellings.2002")),
    ("to 7,148 in 2015", 7148, 0, nat("completed_new_family_dwellings.2015")),
    ("fell from 73.9 thousand (2005)", 73.9, 1, nat("dwellings_permitted_thousands.2005")),
    ("rose from 8.6 thousand to 40.9 thousand", 40.9, 1, nat("dwellings_permitted_thousands.2025")),
    ("fell from 2.38% (2015) to 0.83% (2021)", 2.38, 2, nat("mortgage_cost_of_borrowing_annual_mean_pct.2015")),
    ("fell from 2.38% (2015) to 0.83% (2021)", 0.83, 2, nat("mortgage_cost_of_borrowing_annual_mean_pct.2021")),
    ("rose to 4.13% (2023)", 4.13, 2, nat("mortgage_cost_of_borrowing_annual_mean_pct.2023")),
    ("was 2.99% in 2025", 2.99, 2, nat("mortgage_cost_of_borrowing_annual_mean_pct.2025")),
    ("barely moved (103.7 in 2021)", 103.7, 1, nat(f"{IMP}.2021")),
    ("offset 21.4 of the 52.4 log points", 21.4, 1, lambda: -get("results.json", f"{DEC}.2015-2021.financing_cost")),
    ("offset 21.4 of the 52.4 log points", 52.4, 1, nat(f"{DEC}.2015-2021.price")),
    ("financing-cost term (+43.5 log points)", 43.5, 1, nat(f"{DEC}.2021-2023.financing_cost")),
    ("price term (+19.8)", 19.8, 1, nat(f"{DEC}.2021-2023.price")),
    ("above 2008 (161.4)", 161.4, 1, nat(f"{IMP}.2008")),
    ("the index stood at 166.5", 166.5, 1, nat(f"{IMP}.2025")),
    ("owners fell from 74.8% of the population (2015) to 71.2% (2025)", 71.2, 1, nat("population_tenure_pct.OWN.2025")),
    ("owners fell from 74.8% of the population (2015)", 74.8, 1, nat("population_tenure_pct.OWN.2015")),
    ("rose from 12.7% to 16.3%", 12.7, 1, nat("population_tenure_pct.RENT_MKT.2015")),
    ("peaked at 9.2% in 2014", 9.2, 1, nat("housing_overburden_pct_by_age.TOTAL.2014")),
    ("rose from 62.9% (2015) to 72.3% (2021)", 72.3, 1, nat("parental_coresidence_pct_by_age.Y18-34.2021")),
    ("fell back to 61.0% (2025)", 61.0, 1, nat("parental_coresidence_pct_by_age.Y18-34.2025")),
    ("went from 0.688 (2015) to 2.540 (2025", 2.540, 3,
     nat("completed_new_dwellings_construction_intensity.2025.completed_per_1000_residents")),
    # Regions: population and construction
    ("Portugal's population fell 0.17% a year", 0.17, 2, lambda: -get("results_regional.json", f"{RG}.PT.population_growth_2011_2020.annual_pct")),
    ("Portugal grew 1.89% a year", 1.89, 2, reg(f"{RG}.PT.population_growth_2021_2025.annual_pct")),
    ("Algarve (3.29%)", 3.29, 2, reg(f"{RG}.150.population_growth_2021_2025.annual_pct")),
    ("Alentejo Litoral (3.09%)", 3.09, 2, reg(f"{RG}.1C1.population_growth_2021_2025.annual_pct")),
    ("Península de Setúbal (3.06%)", 3.06, 2, reg(f"{RG}.1B0.population_growth_2021_2025.annual_pct")),
    ("Grande Lisboa grew 2.54%", 2.54, 2, reg(f"{RG}.1A0.population_growth_2021_2025.annual_pct")),
    ("Porto metropolitan area 1.37%", 1.37, 2, reg(f"{RG}.11A.population_growth_2021_2025.annual_pct")),
    ("1.7–5.9% above Eurostat's", 5.9, 1,
     reg("national_cross_checks.population_ine_year_t_vs_eurostat_1_january_t_plus_1.2024.difference_pct")),
    ("1.7–5.9% above Eurostat's", 1.7, 1,
     reg("national_cross_checks.population_ine_year_t_vs_eurostat_1_january_t_plus_1.2021.difference_pct")),
    ("Portugal 2.07, Grande Lisboa 1.26", 2.07, 2, reg(f"{RG}.PT.completions_per_1000_mean_2021_2025")),
    ("Portugal 2.07, Grande Lisboa 1.26", 1.26, 2, reg(f"{RG}.1A0.completions_per_1000_mean_2021_2025")),
    ("Península de Setúbal 2.15", 2.15, 2, reg(f"{RG}.1B0.completions_per_1000_mean_2021_2025")),
    ("Área Metropolitana do Porto 2.71", 2.71, 2, reg(f"{RG}.11A.completions_per_1000_mean_2021_2025")),
    ("stayed between 0.28 and 0.38", 0.28, 2,
     lambda: min(get("results_regional.json", "dispersion_across_nuts3_cv.completions_per_1000").values())),
    ("stayed between 0.28 and 0.38", 0.38, 2,
     lambda: max(get("results_regional.json", "dispersion_across_nuts3_cv.completions_per_1000").values())),
    # Purchase affordability
    ("rose from €1,094 to €1,777 (10.2% a year)", 10.2, 1, aff(f"{RG}.PT.growth_2019_2024.price_annual_pct")),
    ("rose from €1,094 to €1,777", 1777, 0, aff(f"{RG}.PT.price_m2_2024")),
    ("household income rose 4.5% a year", 4.5, 1, aff(f"{RG}.PT.growth_2019_2024.income_annual_pct")),
    ("outpaced income by 26.5 log points", 26.5, 1, aff(f"{RG}.PT.growth_2019_2024.gap_log_points")),
    ("bought 11.4 m² at the median price in 2019 and 8.8 m² in 2024", 8.8, 1, aff(f"{RG}.PT.m2_per_income_year_2024")),
    ("all 9 NUTS 2 regions, 21 of 26 NUTS 3 regions and 220 of 283", 21, 0,
     aff("price_vs_income_growth.nuts3.price_outpaced_income")),
    ("all 9 NUTS 2 regions, 21 of 26 NUTS 3 regions and 220 of 283", 220, 0,
     aff("price_vs_income_growth.municipalities_with_both_end_years.price_outpaced_income")),
    ("| Grande Lisboa | 1,820 → 2,939 | 10.1% | 2.8% | 34.3 |", 34.3, 1, aff(f"{RG}.1A0.growth_2019_2024.gap_log_points")),
    ("| Península de Setúbal | 1,142 → 2,117 | 13.1% | 3.3% | 45.5 |", 45.5, 1, aff(f"{RG}.1B0.growth_2019_2024.gap_log_points")),
    ("| Área Metropolitana do Porto | 1,108 → 1,986 | 12.4% | 5.0% | 34.0 |", 34.0, 1, aff(f"{RG}.11A.growth_2019_2024.gap_log_points")),
    ("| Algarve | 1,696 → 2,752 | 10.2% | 4.1% | 28.2 | 6.7 → 5.0 |", 28.2, 1, aff(f"{RG}.150.growth_2019_2024.gap_log_points")),
    ("| Região Autónoma da Madeira | 1,214 → 2,395 | 14.6% | 5.0% | 43.8 |", 43.8, 1, aff(f"{RG}.300.growth_2019_2024.gap_log_points")),
    ("| Península de Setúbal | 1,142 → 2,117 | 13.1% |", 13.1, 1, aff(f"{RG}.1B0.growth_2019_2024.price_annual_pct")),
    ("| Grande Lisboa | 1,820 → 2,939 | 10.1% | 2.8% | 34.3 | 8.0 → 5.7 |", 5.7, 1, aff(f"{RG}.1A0.m2_per_income_year_2024")),
    ("rose from 0.39 to 0.50 for prices", 0.50, 2, aff("disparities.nuts3_price_m2.2024.cv")),
    ("fell from 0.088 to 0.068 for incomes", 0.068, 3, aff("disparities.nuts3_income.2024.cv")),
    ("appraisal growth correlates 0.82 with transaction growth", 0.82, 2,
     aff("sensitivity_transactions_vs_appraisal.pearson_annual_growth")),
    ("prices outpaced income in 24 of 25 NUTS 3 regions", 24, 0,
     aff("earlier_window_2018_method.nuts3_nuts2013.price_outpaced_income")),
    # Rental market
    ("rose from €5.79 to €9.29 per m² between 2020 and 2025", 9.29, 2, ren("resident_tenants.portugal.2025.all")),
    ("about 10.0% a year to 2024 against 5.5% for income", 10.0, 1,
     ren("affordability.new_lease_rent_vs_irs_income_portugal.price_annual_pct")),
    ("about 10.0% a year to 2024 against 5.5% for income", 5.5, 1,
     ren("affordability.new_lease_rent_vs_irs_income_portugal.income_annual_pct")),
    ("That is 16.6 log points faster than income", 16.6, 1,
     ren("affordability.new_lease_rent_vs_irs_income_portugal.gap_log_points")),
    ("rose 2.1% a year in 2011–2021 and 4.9% in 2021–2025", 2.1, 1, ren("stock.hicp_actual_rentals_annual_pct.2011_2021")),
    ("rose 2.1% a year in 2011–2021 and 4.9% in 2021–2025", 4.9, 1, ren("stock.hicp_actual_rentals_annual_pct.2021_2025")),
    ("it fell to 78.6 in 2024 (2015 = 100)", 78.6, 1,
     ren("affordability.hicp_rent_over_silc_income.2015.income_year_aligned.2024")),
    ("from €235 (2011) to €334 (2021), 3.6% a year", 3.6, 1, ren("census.PT.mean_rent_annual_pct_2011_2021")),
    ("from €235 (2011) to €334 (2021)", 235, 0, ren("census.PT.mean_monthly_rent.2011")),
    ("45% of tenancies had been signed in 2017–2021", 45, 0,
     ren("tenure.census2021_tenancies.PT.contract_period_pct.Entre 2017 e 2021")),
    ("and 16% before 1991", 16, 0, lambda: sum(
        get("results_rental.json", f"tenure.census2021_tenancies.PT.contract_period_pct.{p}")
        for p in ("Antes de 1987", "Entre 1987 e 1990"))),
    ("and 49% paid under €100 a month", 49, 0,
     ren("tenure.census2021_rent_band_by_contract_period_portugal.Antes de 1987.share_below_100_eur_pct")),
    ("and 30% paid €500 or more", 30, 0,
     ren("tenure.census2021_rent_band_by_contract_period_portugal.Entre 2017 e 2021.share_500_eur_or_more_pct")),
    ("in 24 of 26 NUTS 3 regions and 190 of 217 municipalities", 190, 0,
     ren("new_leases.municipalities_balanced.price_outpaced_income")),
    ("Alentejo Litoral (36.5 log points), Madeira (27.7) and Península de Setúbal (26.5)", 36.5, 1,
     ren(f"{RG}.1C1.growth_2020_2024.gap_log_points")),
    ("Alentejo Litoral (36.5 log points), Madeira (27.7) and Península de Setúbal (26.5)", 27.7, 1,
     ren(f"{RG}.300.growth_2020_2024.gap_log_points")),
    ("from 10.8% of the population (2004) to 13.2% (2019) and 16.3% (2025)", 13.2, 1,
     ren("tenure.eu_silc_population_shares_pct.TOTAL.RENT_MKT.2019")),
    ("57% of all tenants", 57, 0, ren("tenure.market_rent_tenants_as_share_of_renters_pct.2025")),
    ("93.1% of tenancies had no rental support", 93.1, 1, ren("tenure.census2021_tenancies.PT.support_pct.without_support")),
    ("market-rent tenants (27.2% in 2025)", 27.2, 1, ren("overburden.eu_silc_by_tenure_pct.RENT_MKT.2025")),
    ("highest in Grande Lisboa (10.0%) among NUTS 2 regions", 10.0, 1, ren("overburden.ine_nuts2_2024.1A.2025")),
    ("the national rate rose 0.53 points in 2019–2025", 0.53, 2, ren("overburden.shift_share.2019_2025.change")),
    ("accounts arithmetically for +0.72 points", 0.72, 2, ren("overburden.shift_share.2019_2025.composition")),
    ("rates within each tenure for −0.19", 0.19, 2, lambda: -get("results_rental.json", "overburden.shift_share.2019_2025.within_groups")),
    ("by a median of 0.28% across the 60 areas published in both versions in 2025, and by at most 1.24% (Lisboa)", 1.24, 2,
     lambda: -get("results_rental.json", "resident_tenants.largest_diff_2025.diff_pct")),
    ("The 2026 method is 3.2–6.3% higher", 6.3, 1, lambda: 100 * (
        get("results_rental.json", "methods_and_windows.portugal.2024.method_2026")
        / get("results_rental.json", "methods_and_windows.portugal.2024.method_2021") - 1)),
    # External figures quoted in the report (recorded in results.json, not computed from pinned data)
    ("1,104,881 (18.5%) were secondary residences", 18.5, 1, nat("descriptives.secondary_dwellings_2021_pct")),
    ("723,215 (12.1%) were vacant", 12.1, 1, nat("descriptives.vacant_dwellings_2021_pct")),
    ("about 60% for 1977–1986 and about 40% for 1987–1996", 60, 0, nat("descriptives.owner_1977_1986_pct")),
    ("about 60% for 1977–1986 and about 40% for 1987–1996", 40, 0, nat("descriptives.owner_1987_1996_pct")),
    # HICP interpretation and the completions composition check
    ("was 23.1% in 2015, 25.1% in 2024 and 24.0% in 2025", 23.1, 1,
     ren("affordability.share_of_rent_in_disposable_income_pct.TOTAL.2015")),
    ("was 23.1% in 2015, 25.1% in 2024 and 24.0% in 2025", 25.1, 1,
     ren("affordability.share_of_rent_in_disposable_income_pct.TOTAL.2024")),
    ("was 23.1% in 2015, 25.1% in 2024 and 24.0% in 2025", 24.0, 1,
     ren("affordability.share_of_rent_in_disposable_income_pct.TOTAL.2025")),
    ("the median price per m² of new dwellings was 27% above that of existing dwellings in Portugal", 27, 0,
     new_dwelling_premium_pct),
    # Integrated assessment
    ("| income growth | −0.23 [−0.38, −0.07] |", 0.23, 2, lambda: -get("results_integrated.json", f"{SP}.price_growth_vs_income_growth.spearman")),
    ("| income growth | −0.23 [−0.38, −0.07] | −0.10 [−0.28, 0.11] |", 0.10, 2,
     lambda: -get("results_integrated.json", f"{SB}.price_growth_vs_income_growth.spearman")),
    ("| −0.34 [−0.57, −0.04] |", 0.34, 2, lambda: -get("results_integrated.json", f"{SC}.appraisal_growth_vs_income_growth.spearman")),
    ("| population growth | +0.37 [0.19, 0.55] (2021–24)", 0.37, 2,
     itg(f"{SP}.price_growth_vs_population_growth_2021_2024.spearman")),
    ("+0.62 [0.42, 0.74] (2015–20)", 0.62, 2, itg(f"{SC}.appraisal_growth_vs_population_growth_2015_2020.spearman")),
    ("| completions per 1,000 | +0.19 [0.04, 0.33] |", 0.19, 2, itg(f"{SP}.price_growth_vs_completions_per_1000.spearman")),
    ("| new-lease rent growth | +0.38 [0.21, 0.52] (2020–24)", 0.38, 2, itg(f"{SP}.price_growth_vs_rent_growth_2020_2024.spearman")),
    ("log price correlates 0.92 [0.85, 0.95] with log new-lease rent", 0.92, 2, itg(f"{LV}.log_price_vs_log_rent.spearman")),
    ("log price correlates 0.92 [0.85, 0.95] with log new-lease rent", 0.85, 2, itg(f"{LV}.log_price_vs_log_rent.spearman_ci95_low")),
    ("*Population:* 0.72 [0.57, 0.82]", 0.72, 2, itg(f"{LV}.log_price_vs_population_growth_2021_2024.spearman")),
    ("*Construction:* 0.46 [0.30, 0.57]", 0.46, 2, itg(f"{LV}.log_price_vs_completions_per_1000_2021_2024.spearman")),
    ("*Income:* 0.45 [0.18, 0.66]", 0.45, 2, itg(f"{LV}.log_price_vs_log_income.spearman")),
    ("fell between 2020 and 2025 in 16 of 26 NUTS 3 regions", 16, 0, itg("gross_rental_yield_pct.nuts3_falling_2020_2025")),
    ("from 5.84% to 5.37% for Portugal", 5.37, 2, itg("gross_rental_yield_pct.by_region.PT.2025")),
    ("from 5.64% to 4.95% in Grande Lisboa", 4.95, 2, itg("gross_rental_yield_pct.by_region.1A0.2025")),
    ("−0.74 to +0.79 across 18 specifications (13 negative)", 0.74, 2, lambda: -spec("income", "A1_twoway_existing_prices")()),
    ("−0.74 to +0.79 across 18 specifications (13 negative)", 0.79, 2, spec("income", "A2_twoway_full")),
    ("−0.74 to +0.79 across 18 specifications (13 negative)", 13, 0, itg("across_specifications.income.negative")),
    ("+1.4% for all sales against 0.0% [−0.7%, 0.8%]", 1.4, 1, m2("results_integrated.json", "completions_composition_check.all_dwellings.estimate", 100)),
    ("+1.4% for all sales against 0.0% [−0.7%, 0.8%]", 0.8, 1,
     m2("results_integrated.json", "completions_composition_check.existing_dwellings.ci95_cr1.1", 100)),
    ("Bank appraisals (sample C, +1.9%)", 1.9, 1, lambda: 100 * spec("completions_per_1000", "C_twoway_income_completions")()),
    ("| 0.9 to 1.6, positive in all 4 specifications |", 1.6, 1, spec("population", "A2_twoway_full")),
    ("0.44 [0.29, 0.61] FE", 0.44, 2, spec("rent", "D_twoway_price_on_rent")),
    ("0.44 [0.29, 0.61] FE", 0.29, 2, spec("rent", "D_twoway_price_on_rent", 0)),
    ("0.31 [0.15, 0.48] FD", 0.31, 2, spec("rent", "D_fd_price_on_rent")),
    ("0.50 [0.26, 0.76] long difference", 0.50, 2, spec("rent", "LD_price_on_rent_2020_2024")),
    ("caution: 1.8% within variation", 1.8, 1, m2("results_integrated.json", "verdicts.D_twoway_price_on_rent:log_rent_m2.within_variance_share", 100)),
    ("In the appraisal sample the slope is −0.19", 0.19, 2,
     lambda: -get("results_integrated.json", "diagnostics.C_twoway_income_completions.wooldridge_fd_serial_correlation.slope")),
    ("(income −0.13 [−0.39, 0.13])", 0.13, 2, lambda: -spec("income", "C_fd_income_completions")()),
    ("growth in completions (r = 0.81)", 0.81, 2, itg(f"{TS}.2005_2024.completions.pearson")),
    ("median income (0.63)", 0.63, 2, itg(f"{TS}.2005_2024.income_silc_income_year.pearson")),
    ("and population (0.61) in 2009–2024", 0.61, 2, itg(f"{TS}.2005_2024.population_end_of_year.pearson")),
    ("except completions in 2013–2024 (0.83 [0.15, 0.98])", 0.83, 2, itg(f"{TS}.2013_2024.completions.pearson")),
    ("except completions in 2013–2024 (0.83 [0.15, 0.98])", 0.15, 2, itg(f"{TS}.2013_2024.completions.ci95.0")),
    # Samples
    ("| A | 2019–2024 | sale price (2022 method) | income | 272 |", 272, 0, itg("samples.A_transactions_2019_2024.municipalities")),
    ("| C | 2015–2024 | bank appraisal | income, completions per 1,000 | 179 |", 179, 0, itg("samples.C_appraisal_2015_2024.municipalities")),
    ("| D | 2020–2024 | sale price | new-lease rent | 215 |", 215, 0, itg("samples.D_rents_2020_2024.municipalities")),
]


def normalise(text: str) -> str:
    """Collapse whitespace and drop Markdown emphasis so wrapped lines still match."""
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


class ReportClaimTests(unittest.TestCase):
    def test_every_claim_is_in_the_report_and_matches_the_results(self) -> None:
        text = normalise(REPORT.read_text(encoding="utf-8"))
        for snippet, expected, decimals, value in CLAIMS:
            with self.subTest(snippet=snippet, expected=expected):
                self.assertIn(normalise(snippet), text)
                self.assertIn(f"{abs(expected):,.{decimals}f}", snippet)
                computed = value()
                self.assertTrue(math.isfinite(computed))
                self.assertLessEqual(abs(computed - expected), 0.5 * 10 ** -decimals + 1e-9,
                                     f"{snippet!r}: results give {computed}")

    def test_verdicts_quoted_in_the_report(self) -> None:
        verdicts = get("results_integrated.json", "verdicts")
        self.assertEqual(verdicts["D_twoway_price_on_rent:log_rent_m2"]["verdict"], "robust")
        self.assertEqual(verdicts["A2_twoway_income_completions:completions_per_1000"]["verdict"], "robust")
        self.assertEqual(verdicts["A2_twoway_full:log_population"]["verdict"], "fragile")
        self.assertTrue(all(v["verdict"] != "robust" for k, v in verdicts.items() if k.endswith(":log_income")))
        income = get("results_integrated.json", "across_specifications.income")
        self.assertLess(max(income["positive"], income["negative"]) / income["specifications"], 0.8)

    def test_bootstrap_p_values_agree_as_reported(self) -> None:
        pairs: list[tuple[float, float]] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if "p_value" in node and "p_value_restricted" in node:
                    pairs.append((node["p_value"], node["p_value_restricted"]))
                for child in node.values():
                    walk(child)

        walk(get("results_integrated.json", "models"))
        self.assertEqual(len(pairs), 32)
        self.assertLessEqual(max(abs(a - b) for a, b in pairs), 0.024 + 1e-9)
        self.assertFalse(any((a < 0.05) != (b < 0.05) for a, b in pairs))

    def test_figures_and_local_links_exist(self) -> None:
        raw = REPORT.read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", raw)
        local = [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))]
        self.assertGreaterEqual(len([t for t in local if t.startswith("figures/")]), 14)
        for target in local:
            with self.subTest(target=target):
                self.assertTrue((HERE / target).exists(), target)


if __name__ == "__main__":
    unittest.main()
