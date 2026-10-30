"""Rental market in Portugal: new-lease rents, the stock of tenancies, tenure and housing-cost burden.

Reads pinned INE containers (``fetch_regional.py --set rental``) and Eurostat
EU-SILC/HICP snapshots, checks every response against its recorded SHA-256, and
reports three different rent measures side by side without splicing them:

* new lease contracts registered with the tax authority (INE local rent
  statistics): the median rent per m2 of contracts signed in a year;
* the stock of tenancies: the mean monthly rent of all rented dwellings at each
  census (2001, 2011, 2021) and the HICP actual-rentals index, which follows the
  rents paid by sitting tenants;
* EU-SILC housing-cost burden and tenure, which describe households, not
  dwellings or contracts.

Nothing is downloaded here, no gap is filled, and nothing here identifies a cause
of the changes described.

Run:  python rental.py      # writes results_rental.json and rental_panel.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import affordability as F
import regional as R
from affordability_core import Panel, annual_growth, growth_comparison, median
from analysis import parse_income, verify_snapshot
from rental_core import jsonstat_select, relative_index, shift_share, years

HERE = Path(__file__).parent
DATA = HERE / "data"
SNAPSHOT_DATE = "2026-10-29"
FOCUS = {"1A0": "Grande Lisboa", "1B0": "Península de Setúbal", "11A": "Área Metropolitana do Porto", "PT": "Portugal"}
INE = {
    "0014698": "new_rent", "0014697": "new_contracts", "0014711": "new_rent_quartiles",
    "0012600": "new_rent_2021m", "0012601": "new_contracts_2021m",
    "0009631": "new_rent_2021m_nuts2013", "0009632": "new_contracts_2021m_nuts2013",
    "0014700": "new_rent_resident", "0014699": "new_contracts_resident",
    "0012521": "census21_rented_share", "0012522": "census21_mean_rent",
    "0006993": "census11_rented_share", "0006995": "census11_mean_rent",
    "0000977": "census91_01_rented_share", "0000777": "census01_mean_rent",
    "0012508": "census21_by_contract_period", "0012529": "census21_by_support",
    "0011522": "census21_unsupported_by_contract_period", "0012528": "census21_by_years_in_dwelling",
    "0011501": "census21_rent_band_by_contract_period",
    "0006260": "overburden_nuts2_a", "0014472": "overburden_nuts2_b",
}
EUROSTAT = ("ilc_lvho07c_PT", "ilc_lvho02_PT_by_poverty", "ilc_mded02_PT", "ilc_di09_PT", "ilc_lvho25_PT",
            "ilc_lvho07d_PT", "prc_hicp_aind_PT_rents")
TENURE = ("OWN_L", "OWN_NL", "RENT_MKT", "RENT_FR")
OLD_CENSUS = {"census91_01_rented_share", "census01_mean_rent"}  # rows for withdrawn parish codes are skipped
POSITIVE = {"new_rent", "new_rent_quartiles", "new_rent_2021m", "new_rent_2021m_nuts2013", "new_rent_resident"}  # rents per m2


def load_ine(folder: Path, date: str) -> dict[str, F.Series]:
    """Every pinned INE container of the rental set, by short name."""
    return {label: F.parse_series(folder / f"ine_{code}_{date}.jsonl.gz", allow_zero=label not in POSITIVE,
                                  skip_unknown_geo=label in OLD_CENSUS)
            for code, label in INE.items()}


def load_eurostat(folder: Path, date: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Every pinned Eurostat snapshot of the rental set (checked against its record) and the records."""
    docs, records = {}, {}
    for tag in EUROSTAT:
        path = folder / f"eurostat_{tag}_{date}.json"
        records[tag] = verify_snapshot(path, path.with_suffix(".meta.json"))
        docs[tag] = json.loads(path.read_text(encoding="utf-8"))
    return docs, records


def annual(doc: dict[str, Any], **fixed: str) -> dict[int, float]:
    """One annual Eurostat series as {year: value}."""
    values, _flags = jsonstat_select(doc, fixed)
    return years(values)


def flags(doc: dict[str, Any], **fixed: str) -> dict[int, str]:
    """Eurostat status flags of one annual series."""
    _values, f = jsonstat_select(doc, fixed)
    return {int(k): v for k, v in sorted(f.items())}


def round_series(series: dict[int, float], digits: int = 1) -> dict[str, float]:
    """JSON-friendly rounded series."""
    return {str(y): round(v, digits) for y, v in sorted(series.items())}


def ine_slice(series: F.Series, geo: R.Geography, dims: dict[str, str] | None = None) -> Panel:
    """NUTS 2024 values of one INE series (Portugal, NUTS 2, NUTS 3, municipalities)."""
    values, _missing = F.on_nuts2024(series, geo, dims)
    return values


def by_dico(series: F.Series, dims: dict[str, str] | None = None) -> dict[tuple[str, int], float]:
    """Municipal values of an INE series of any NUTS version, keyed by (DICO, year)."""
    source = F.as_source(series, dims)
    return {(d, y): v for d, code in R.municipalities(source).items() for (g, y), v in source.values.items() if g == code}


def unchanged_regions(series: F.Series, geo: R.Geography, dims: dict[str, str] | None = None) -> dict[str, str]:
    """Old NUTS 3 code -> NUTS 2024 code for regions with exactly the same municipalities."""
    groups = R.correspondence(F.as_source(series, dims), geo)
    return {o[0]: n[0] for o, n, _m in groups if len(o) == 1 and len(n) == 1}


def shares(counts: dict[str, float], total: float) -> dict[str, float]:
    """Counts as percentages of a total (rounded to 0.1)."""
    if total <= 0:
        raise ValueError("Non-positive total")
    return {k: round(100 * v / total, 1) for k, v in counts.items()}


def _band_amounts(label: str) -> list[float]:
    """Euro amounts in an INE rent-band label ('De 100 a 149,99 euros' -> [100.0, 149.99])."""
    return [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:,\d+)?", label)]


def band_lower_eur(label: str) -> float:
    """Lower bound of an INE rent band ('Menos de 20 euros' -> 0; '1000 euros ou mais' -> 1000)."""
    return 0.0 if label.startswith("Menos de") else _band_amounts(label)[0]


def band_upper_eur(label: str) -> float:
    """Upper bound of an INE rent band, rounded up to whole euros ('De 75 a 99,99 euros' -> 100; open band -> inf)."""
    if label.endswith("ou mais"):
        return math.inf
    return math.ceil(_band_amounts(label)[-1])


def band_median(counts: dict[str, float], order: list[str], labels: dict[str, str]) -> str:
    """Label of the band that contains the median of a banded distribution."""
    total = sum(counts[b] for b in order)
    if total <= 0:
        raise ValueError("Empty distribution")
    running = 0.0
    for band in order:
        running += counts[band]
        if running >= total / 2:
            return labels[band]
    raise ValueError("Empty distribution")


def main() -> None:  # noqa: C901 - one linear pipeline, kept in one place for auditability
    """Build, check and export the rental-market results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--date", default=SNAPSHOT_DATE)
    parser.add_argument("--results", type=Path, default=HERE / "results_rental.json")
    parser.add_argument("--panel", type=Path, default=HERE / "rental_panel.csv")
    args = parser.parse_args()
    src = load_ine(args.data, args.date)
    docs, records = load_eurostat(args.data, args.date)
    income_src = F.parse_series(args.data / f"ine_0012712_{args.date}.jsonl.gz")
    silc_income = parse_income(json.loads((args.data / f"eurostat_ilc_di03_PT_{args.date}.json").read_text(encoding="utf-8")))

    geo = R.nuts2024_geography(F.as_source(src["new_rent"]))
    if R.nuts2024_geography(F.as_source(income_src)).nuts3 != geo.nuts3:
        raise ValueError("Income and rent sources assign municipalities differently")
    order = src["new_rent"].order
    nuts2 = [c for c in order if c in geo.members2]
    nuts3 = [c for c in order if c in geo.members3]
    munis = [geo.municipality[c[-4:]] for c in order if src["new_rent"].levels[c] == F.MUNICIPALITY]
    level = {"PT": "country", **{r: "NUTS2" for r in nuts2}, **{r: "NUTS3" for r in nuts3}, **{m: "municipality" for m in munis}}
    units = ["PT", *nuts2, *nuts3, *munis]
    lisboa, porto = geo.municipality[R.LISBOA], geo.municipality["1312"]

    # ---- New lease contracts (INE, tax-authority records) ----
    rent, rent_na = F.on_nuts2024(src["new_rent"], geo)
    contracts = ine_slice(src["new_contracts"], geo)
    q1 = ine_slice(src["new_rent_quartiles"], geo, {"3": "1"})
    q3 = ine_slice(src["new_rent_quartiles"], geo, {"3": "3"})
    q2 = ine_slice(src["new_rent_quartiles"], geo, {"3": "2"})
    rent_2021m = ine_slice(src["new_rent_2021m"], geo)
    rent_res = ine_slice(src["new_rent_resident"], geo, {"3": "T"})
    contracts_res = ine_slice(src["new_contracts_resident"], geo, {"3": "T"})
    income = ine_slice(income_src, geo)
    rent_years = sorted({y for _r, y in rent})
    y0, y1 = rent_years[0], rent_years[-1]
    common = [y for y in rent_years if all(("PT", y) in s for s in (rent, income))]
    c0, c1 = common[0], common[-1]
    if (y0, y1, c0, c1) != (2020, 2025, 2020, 2024):
        raise ValueError(f"Unexpected windows: rents {y0}-{y1}, rents with income {c0}-{c1}")
    if any(q2[k] != rent[k] for k in q2.keys() & rent.keys()):
        raise ValueError("The published median and the second quartile differ")
    gaps = [(r, y) for r in ["PT", *nuts2, *nuts3] for y in rent_years if (r, y) not in rent]
    if gaps:
        raise ValueError(f"Regional new-lease medians missing: {gaps[:5]}")
    m2_month = {k: income[k] / 12 / rent[k] for k in rent.keys() & income.keys()}

    regions: dict[str, Any] = {}
    for r in units:
        e: dict[str, Any] = {"name": geo.names.get(r, r), "level": level[r]}
        for y in (y0, c1, y1):
            e[f"new_lease_rent_m2_{y}"] = rent.get((r, y))
        e[f"new_contracts_{y0}"], e[f"new_contracts_{y1}"] = contracts.get((r, y0)), contracts.get((r, y1))
        g = annual_growth(rent, r, y0, y1)
        e[f"rent_annual_pct_{y0}_{y1}"] = round(g, 2) if g is not None else None
        e[f"growth_{c0}_{c1}"] = (lambda c: {k: round(v, 2) for k, v in c.items()} if c else None)(
            growth_comparison(rent, income, r, c0, c1))
        for y in (c0, c1):
            e[f"m2_per_month_of_income_{y}"] = round(m2_month[(r, y)], 2) if (r, y) in m2_month else None
        for y in (y0, y1):
            e[f"quartile_ratio_q3_q1_{y}"] = round(q3[(r, y)] / q1[(r, y)], 3) if (r, y) in q1 and (r, y) in q3 else None
        regions[r] = e
    for r in units:
        own, nat = regions[r][f"new_lease_rent_m2_{y1}"], regions["PT"][f"new_lease_rent_m2_{y1}"]
        regions[r][f"new_lease_rent_m2_{y1}_relative_to_portugal"] = round(own / nat, 3) if own and nat else None

    def compare(units_: list[str]) -> dict[str, Any]:
        return F.compare_regions(rent, income, units_, c0, c1)

    balanced = [m for m in munis if all((m, y) in rent and (m, y) in income for y in common)]
    new_leases = {
        "window_with_income": [c0, c1], "window_rents": [y0, y1],
        "nuts3": compare(nuts3), "nuts2": compare(nuts2), "municipalities_with_both_end_years": compare(munis),
        "municipalities_balanced": compare(balanced),
        "dispersion_nuts3_rent_m2": {str(y): {k: round(v, 3) for k, v in F.dispersion([rent[(r, y)] for r in nuts3]).items()}
                                     for y in rent_years},
        "coverage_municipalities": {str(y): {"rent_published": sum(1 for m in munis if (m, y) in rent),
                                             "rent_unavailable": sum(1 for (r, yy) in rent_na if yy == y and level.get(r) == "municipality")}
                                    for y in rent_years},
        "balanced_municipalities": len(balanced),
    }

    # Method 2021 against method 2026 (2020-2024), and the earlier window under the 2021 method (NUTS 2013).
    shared = sorted({y for _r, y in rent_2021m} & set(rent_years))
    pairs3 = [(r, r) for r in nuts3]
    pairs_m = [(m, m) for m in munis]
    old = src["new_rent_2021m_nuts2013"]
    old_rent = F.as_source(old).values
    unchanged = unchanged_regions(old, geo)
    old_years = old.years()
    early = {
        "window": [old_years[0], 2019],
        "note": "2021 method on NUTS 2013, 2017-2019; reported beside the main window and never spliced onto it",
        "portugal_annual_pct": round(annual_growth(old_rent, "PT", old_years[0], 2019) or 0, 2),
        "area_metropolitana_de_lisboa_annual_pct": round(annual_growth(old_rent, "170", old_years[0], 2019) or 0, 2),
        "area_metropolitana_do_porto_annual_pct": round(annual_growth(old_rent, "11A", old_years[0], 2019) or 0, 2),
        "lisboa_municipality_annual_pct": round(annual_growth(by_dico(old), R.LISBOA, old_years[0], 2019) or 0, 2),
        "porto_municipality_annual_pct": round(annual_growth(by_dico(old), "1312", old_years[0], 2019) or 0, 2),
        "portugal_rent_m2": {str(y): old_rent[("PT", y)] for y in old_years if ("PT", y) in old_rent},
    }
    methods = {
        "method_2021_vs_2026_years": shared,
        "nuts3": F.method_overlap(rent_2021m, rent, pairs3, shared),
        "municipalities": F.method_overlap(rent_2021m, rent, pairs_m, shared),
        "portugal": {str(y): {"method_2021": rent_2021m[("PT", y)], "method_2026": rent[("PT", y)]} for y in shared},
        f"portugal_annual_pct_{shared[0]}_{shared[-1]}": {"method_2021": round(annual_growth(rent_2021m, "PT", shared[0], shared[-1]) or 0, 2),
                                                          "method_2026": round(annual_growth(rent, "PT", shared[0], shared[-1]) or 0, 2)},
        "method_2021_nuts2013_vs_nuts2024_unchanged_nuts3": F.method_overlap(old_rent, rent_2021m, list(unchanged.items()),
                                                                            sorted(set(old_years) & set(shared))),
    }

    # Tenants resident in Portugal against all tenants (12 months to Q4; regions and municipalities over 100,000).
    res_units = [r for r in units if (r, y1) in rent_res and (r, y1) in rent]
    res_diff = {r: 100 * (rent_res[(r, y1)] / rent[(r, y1)] - 1) for r in res_units}
    worst_res = max(res_diff, key=lambda r: abs(res_diff[r]))
    resident = {
        "units": len(res_units),
        "portugal": {str(y): {"all": rent.get(("PT", y)), "resident_tenants": rent_res.get(("PT", y)),
                              "contracts_all": contracts.get(("PT", y)), "contracts_resident": contracts_res.get(("PT", y))}
                     for y in rent_years},
        f"median_abs_diff_pct_{y1}": round(median(abs(v) for v in res_diff.values()), 2),
        f"largest_diff_{y1}": {"region": worst_res, "name": geo.names.get(worst_res, worst_res),
                               "diff_pct": round(res_diff[worst_res], 2)},
        "growth_all_vs_resident": {r: {"all": round(annual_growth(rent, r, y0, y1) or 0, 2),
                                       "resident": round(annual_growth(rent_res, r, y0, y1) or 0, 2)}
                                   for r in ["PT", "1A0", "1B0", "11A", lisboa, porto] if (r, y0) in rent_res},
    }

    # ---- Stock of tenancies: censuses and the HICP actual-rentals index ----
    def census_value(series: F.Series, dims: dict[str, str] | None, region: str, year: int) -> float | None:
        values = series.get(dims)
        if region in ("PT",):
            return values.get(("PT", year))
        dico_values = by_dico(series, dims)
        if region in (lisboa, porto):
            return dico_values.get((region[-4:], year))
        for oldc, newc in unchanged_regions(series, geo, dims).items():
            if newc == region:
                return values.get((oldc, year))
        return None

    census_units = ["PT", "1A0", "1B0", "11A", lisboa, porto]
    census: dict[str, dict[str, Any]] = {}
    for r in census_units:
        mean_rent = {2001: census_value(src["census01_mean_rent"], None, r, 2001),
                     2011: census_value(src["census11_mean_rent"], None, r, 2011),
                     2021: ine_slice(src["census21_mean_rent"], geo).get((r, 2021))}
        rented = {1991: census_value(src["census91_01_rented_share"], None, r, 1991),
                  2001: census_value(src["census91_01_rented_share"], None, r, 2001),
                  2011: census_value(src["census11_rented_share"], None, r, 2011),
                  2021: ine_slice(src["census21_rented_share"], geo).get((r, 2021))}
        census[r] = {"name": geo.names.get(r, r), "mean_monthly_rent": {str(k): v for k, v in mean_rent.items()},
                     "rented_share_pct": {str(k): v for k, v in rented.items()}}
        for a, b in ((2001, 2011), (2011, 2021)):
            ma, mb = mean_rent[a], mean_rent[b]
            census[r][f"mean_rent_annual_pct_{a}_{b}"] = round(100 * ((mb / ma) ** 0.1 - 1), 2) if ma and mb else None
    for r in ["PT", *nuts2, *nuts3]:
        regions[r]["census2021_mean_monthly_rent"] = ine_slice(src["census21_mean_rent"], geo).get((r, 2021))
        regions[r]["census2021_rented_share_pct"] = ine_slice(src["census21_rented_share"], geo).get((r, 2021))

    hicp = {c: annual(docs["prc_hicp_aind_PT_rents"], geo="PT", coicop=c) for c in ("CP041", "CP00")}
    hicp_ea = annual(docs["prc_hicp_aind_PT_rents"], geo="EA20", coicop="CP041")
    real_rent = {y: 100 * hicp["CP041"][y] / hicp["CP00"][y] for y in hicp["CP041"] if y in hicp["CP00"]}

    def pct(a: dict[int, float], s: int, e: int) -> float | None:
        return round(100 * ((a[e] / a[s]) ** (1 / (e - s)) - 1), 2) if s in a and e in a else None

    stock = {
        "hicp_actual_rentals_annual_pct": {f"{s}_{e}": pct(hicp["CP041"], s, e) for s, e in ((2001, 2011), (2011, 2021), (2015, 2025), (2020, 2025), (2021, 2025))},
        "hicp_all_items_annual_pct": {f"{s}_{e}": pct(hicp["CP00"], s, e) for s, e in ((2001, 2011), (2011, 2021), (2015, 2025), (2020, 2025))},
        "hicp_actual_rentals_euro_area_annual_pct": {f"{s}_{e}": pct(hicp_ea, s, e) for s, e in ((2015, 2025), (2020, 2025))},
        "hicp_actual_rentals_2015_100": round_series(hicp["CP041"], 2),
        "hicp_rentals_relative_to_all_items_2015_100": round_series(real_rent, 1),
        "new_lease_rent_m2_portugal_annual_pct_2020_2025": round(annual_growth(rent, "PT", 2020, 2025) or 0, 2),
        "census_mean_rent_portugal_annual_pct": {k: v for k, v in census["PT"].items() if k.startswith("mean_rent_annual")},
    }

    # ---- Rent against income ----
    # EU-SILC income of survey year t refers to calendar year t-1 (INE quality report); both alignments are shown.
    income_prev = {y - 1: v for y, v in silc_income.items()}
    rent_income = {}
    for base in (2010, 2015, 2019):
        rent_income[str(base)] = {
            "income_year_aligned": round_series(relative_index(hicp["CP041"], income_prev, base), 1),
            "survey_year_aligned": round_series(relative_index(hicp["CP041"], silc_income, base), 1),
        }
    tenure_income = {t: annual(docs["ilc_di09_PT"], tenure=t) for t in ("TOTAL", "OWN", "RENT")}
    rent_share = {r: annual(docs["ilc_mded02_PT"], rskpovth=r) for r in ("TOTAL", "A_60", "B_60")}
    affordability = {
        "hicp_rent_over_silc_income": rent_income,
        "new_lease_rent_vs_irs_income_portugal": regions["PT"][f"growth_{c0}_{c1}"],
        "share_of_rent_in_disposable_income_pct": {r: round_series(s, 1) for r, s in rent_share.items()},
        "median_equivalised_income_renters_over_owners": {str(y): round(tenure_income["RENT"][y] / tenure_income["OWN"][y], 3)
                                                          for y in tenure_income["RENT"] if y in tenure_income["OWN"]},
        "median_equivalised_income_by_tenure_eur": {t: round_series(s, 0) for t, s in tenure_income.items()},
    }

    # ---- Tenure: who pays market rents ----
    lvho02 = docs["ilc_lvho02_PT_by_poverty"]
    tenure_shares = {p: {t: annual(lvho02, rskpovth=p, tenure=t) for t in ("OWN", "OWN_L", "OWN_NL", "RENT", "RENT_MKT", "RENT_FR")}
                     for p in ("TOTAL", "A_60", "B_60")}
    for p, by_t in tenure_shares.items():
        for y in by_t["RENT"]:
            if all(y in by_t[t] for t in ("RENT_MKT", "RENT_FR", "OWN")) and abs(by_t["RENT_MKT"][y] + by_t["RENT_FR"][y] - by_t["RENT"][y]) > 0.21:
                raise ValueError(f"lvho02 {p} {y}: rental shares do not add up")
    market_among_renters = {str(y): round(100 * tenure_shares["TOTAL"]["RENT_MKT"][y] / tenure_shares["TOTAL"]["RENT"][y], 1)
                            for y in tenure_shares["TOTAL"]["RENT"] if tenure_shares["TOTAL"]["RENT"][y] > 0}

    support = ine_slice(src["census21_by_support"], geo, {"3": "T", "4": "T"})
    supported = {c: ine_slice(src["census21_by_support"], geo, {"3": c, "4": "T"}) for c in ("1", "11", "12", "13", "2")}
    periods = list(src["census21_by_contract_period"].dims["4"])
    period_counts = {p: ine_slice(src["census21_by_contract_period"], geo, {"3": "T", "4": p}) for p in periods}
    years_cats = list(src["census21_by_years_in_dwelling"].dims["4"])
    years_counts = {c: ine_slice(src["census21_by_years_in_dwelling"], geo, {"3": "T", "4": c}) for c in years_cats}
    unsupported_old = {p: by_dico(src["census21_unsupported_by_contract_period"], {"3": "2", "4": p}) for p in periods}
    unsupported_pt = {p: src["census21_unsupported_by_contract_period"].get({"3": "2", "4": p}).get(("PT", 2021)) for p in periods}
    tenancies: dict[str, Any] = {}
    for r in ["PT", *nuts2, *nuts3]:
        total = support[(r, 2021)]
        if abs(sum(period_counts[p][(r, 2021)] for p in periods if p != "T") - period_counts["T"][(r, 2021)]) > 0.5:
            raise ValueError(f"Contract periods do not add up for {r}")
        if abs(supported["1"][(r, 2021)] + supported["2"][(r, 2021)] - total) > 0.5 or period_counts["T"][(r, 2021)] != total:
            raise ValueError(f"Census 2021 tenancy totals disagree for {r}")
        tenancies[r] = {
            "name": geo.names.get(r, r), "tenancies": total,
            "support_pct": shares({"with_support": supported["1"][(r, 2021)], "social_or_supported_rent": supported["11"][(r, 2021)],
                                   "central_government_subsidy": supported["12"][(r, 2021)],
                                   "municipal_or_regional_subsidy": supported["13"][(r, 2021)],
                                   "without_support": supported["2"][(r, 2021)]}, total),
            "contract_period_pct": shares({src["census21_by_contract_period"].dims["4"][p]: period_counts[p][(r, 2021)]
                                           for p in periods if p != "T"}, total),
            "years_in_dwelling_pct": shares({src["census21_by_years_in_dwelling"].dims["4"][c]: years_counts[c][(r, 2021)]
                                             for c in years_cats if c != "T"}, years_counts["T"][(r, 2021)]),
        }
    pt_unsupported_total = unsupported_pt["T"] or 0
    band = src["census21_rent_band_by_contract_period"]
    band_codes = [c for c in band.dims["3"] if c != "T"]
    below_100 = [b for b in band_codes if band_upper_eur(band.dims["3"][b]) <= 100]
    from_500 = [b for b in band_codes if band_lower_eur(band.dims["3"][b]) >= 500]
    band_by_period = {}
    for p in periods:
        counts = {b: band.get({"3": b, "4": p}).get(("PT", 2021), 0.0) for b in band_codes}
        tot = band.get({"3": "T", "4": p}).get(("PT", 2021))
        if tot is None or abs(sum(counts.values()) - tot) > 0.5:
            raise ValueError(f"Rent bands do not add up for contract period {p}")
        band_by_period[band.dims["4"][p]] = {
            "tenancies": tot, "median_band": band_median(counts, band_codes, band.dims["3"]),
            "share_below_100_eur_pct": round(100 * sum(counts[b] for b in below_100) / tot, 1),
            "share_500_eur_or_more_pct": round(100 * sum(counts[b] for b in from_500) / tot, 1),
        }
    tenure = {
        "eu_silc_population_shares_pct": {p: {t: round_series(s, 1) for t, s in by_t.items()} for p, by_t in tenure_shares.items()},
        "market_rent_tenants_as_share_of_renters_pct": market_among_renters,
        "census_rented_share_pct": {r: census[r]["rented_share_pct"] for r in census_units},
        "census2021_tenancies": tenancies,
        "census2021_unsupported_tenancies_by_contract_period_portugal_pct": shares(
            {src["census21_unsupported_by_contract_period"].dims["4"][p]: unsupported_pt[p] or 0 for p in periods if p != "T"},
            pt_unsupported_total),
        "census2021_rent_band_by_contract_period_portugal": band_by_period,
        "census2021_unsupported_tenancies_lisboa_porto_municipalities_pct": {
            name: shares({src["census21_unsupported_by_contract_period"].dims["4"][p]: unsupported_old[p][(d, 2021)] for p in periods if p != "T"},
                         unsupported_old["T"][(d, 2021)]) for name, d in (("Lisboa", R.LISBOA), ("Porto", "1312"))},
    }

    # ---- Housing-cost overburden ----
    lvho07c = docs["ilc_lvho07c_PT"]
    over = {t: annual(lvho07c, tenure=t) for t in ("TOTAL", *TENURE)}
    shift = {}
    implied_check = {}
    for y in sorted(over["TOTAL"]):
        if all(y in over[t] and y in tenure_shares["TOTAL"][t] for t in TENURE):
            implied = sum(tenure_shares["TOTAL"][t][y] * over[t][y] for t in TENURE) / 100
            implied_check[str(y)] = {"published": over["TOTAL"][y], "implied_from_tenure": round(implied, 2)}
    worst = max(abs(v["published"] - v["implied_from_tenure"]) for v in implied_check.values())
    if worst > 0.35:
        raise ValueError(f"Overburden by tenure does not reproduce the total (max gap {worst:.2f} pp)")
    for a, b in ((2005, 2025), (2015, 2025), (2019, 2025)):
        sa = {t: tenure_shares["TOTAL"][t][a] for t in TENURE}
        sb = {t: tenure_shares["TOTAL"][t][b] for t in TENURE}
        ra = {t: over[t][a] for t in TENURE}
        rb = {t: over[t][b] for t in TENURE}
        shift[f"{a}_{b}"] = {k: round(v, 2) for k, v in shift_share(sa, ra, sb, rb).items()}
    urb = {c: annual(docs["ilc_lvho07d_PT"], deg_urb=c) for c in ("DEG1", "DEG2", "DEG3")}
    citizen = {c: annual(docs["ilc_lvho25_PT"], citizen=c) for c in ("NAT", "FOR", "EU27_2020_FOR", "NEU27_2020_FOR")}
    citizen_flags = {c: flags(docs["ilc_lvho25_PT"], citizen=c) for c in citizen}

    def ine_overburden(series: F.Series) -> dict[str, dict[str, float]]:
        values = series.get({"3": "T"})
        return {code: {str(y): v for (g, y), v in sorted(values.items()) if g == code}
                for code in series.order if any(g == code for g, _y in values)}

    overburden = {
        "eu_silc_by_tenure_pct": {t: round_series(s, 1) for t, s in over.items()},
        "total_implied_from_tenure_shares": implied_check,
        "shift_share": shift,
        "by_degree_of_urbanisation_pct": {c: round_series(s, 1) for c, s in urb.items()},
        "adults_by_citizenship_pct": {c: round_series(s, 1) for c, s in citizen.items()},
        "adults_by_citizenship_flags": {c: {str(y): f for y, f in fl.items()} for c, fl in citizen_flags.items()},
        "ine_nuts2_2013": ine_overburden(src["overburden_nuts2_a"]),
        "ine_nuts2_2024": ine_overburden(src["overburden_nuts2_b"]),
        "ine_nuts2_names": {**src["overburden_nuts2_a"].names, **src["overburden_nuts2_b"].names},
    }

    sources = {
        "ine": {label: {"indicator": s.code, "title": s.title, "unit": s.unit, "last_update": s.last_update,
                        "ine_note": s.note, "years": s.years(), "retrieved_first": min(s.retrieved),
                        "retrieved_last": max(s.retrieved), "cells_with_value": len(s.values),
                        "cells_unavailable": len(s.unavailable), "rows_for_withdrawn_codes": s.unknown_geo_rows}
                for label, s in src.items()},
        "eurostat": {tag: {"url": rec["url"], "retrieved_at_utc": rec["retrieved_at_utc"],
                           "label": docs[tag].get("label"), "updated": docs[tag].get("updated")} for tag, rec in records.items()},
    }
    results: dict[str, Any] = {
        "definitions": {
            "new_lease_rent_m2": "Median monthly rent per m2 of the new lease contracts for family dwellings registered with "
                                 "the tax authority in the year (INE, method 2026); contracts signed, not the rents of all tenants",
            "census_mean_rent": "Mean monthly rent of all rented classic dwellings of usual residence at the census date "
                                "(old and new contracts, social rents included)",
            "hicp_actual_rentals": "Eurostat HICP, COICOP 04.1 actual rentals for housing, 2015=100: price change of rents "
                                   "paid by tenants, a quality-adjusted index, not a rent level",
            "m2_per_month_of_income": "Median gross household income / 12 / median new-lease rent per m2: floor area that one "
                                      "month of median income would rent at the median new-lease rate. A ratio of medians from "
                                      "different populations; not the rent any household pays",
            "overburden": "Share of people living in households whose housing costs, net of housing allowances, exceed 40% "
                          "of disposable income, net of housing allowances (EU-SILC)",
            "asking_rents": "No official statistics of advertised rents exist; portal listings are not used",
        },
        "sources": sources,
        "new_leases": new_leases,
        "methods_and_windows": methods,
        "earlier_window_2017_2019": early,
        "resident_tenants": resident,
        "stock": stock,
        "census": census,
        "affordability": affordability,
        "tenure": tenure,
        "overburden": overburden,
        "regions": regions,
    }
    args.results.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    with args.panel.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["region_code", "region_name", "level", "year", "new_lease_rent_m2", "new_lease_rent_m2_q1",
                         "new_lease_rent_m2_q3", "new_contracts", "new_lease_rent_m2_method2021",
                         "new_lease_rent_m2_resident_tenants", "income_gross_per_household", "m2_per_month_of_income",
                         "rent_unavailable"])
        for r in units:
            for y in rent_years:
                def fmt(s: Panel, digits: int = 2) -> str:
                    return f"{s[(r, y)]:.{digits}f}" if (r, y) in s else ""
                writer.writerow([r, regions[r]["name"], level[r], y, fmt(rent), fmt(q1), fmt(q3), fmt(contracts, 0),
                                 fmt(rent_2021m), fmt(rent_res), fmt(income, 0), fmt(m2_month), rent_na.get((r, y), "")])
    print(json.dumps({"portugal": regions["PT"], "nuts3": new_leases["nuts3"], "shift_share": shift}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
