"""Reproduce selected descriptive claims about Portuguese housing affordability.

The only time-series input is the official Eurostat JSON-stat response for
`prc_hpi_a`, Portugal, total purchases, annual average index (2015=100).
No causal effects or electoral preferences are estimated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "prc_hpi_a?geo=PT&purchase=TOTAL&unit=I15_A_AVG"
)
INCOME_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "ilc_di03?geo=PT&age=TOTAL&sex=T&statinfo=MED_EI&unit=EUR"
)
OVERBURDEN_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "ilc_lvho07a?geo=PT&sex=T&rskpovth=TOTAL&unit=PC"
)
PARENTS_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "ilc_lvps08?geo=PT&sex=T&unit=PC"
)
LEAVING_AGE_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "yth_demo_030?geo=PT&sex=T"
)
TENURE_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "ilc_lvho02?geo=PT&rskpovth=TOTAL&hhcomp=TOTAL&unit=PC"
)
PERMITS_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "sts_cobp_a?geo=PT&indic_bt=BPRM_DW&cpa2_1=CPA_F41001_X_410014&s_adj=NSA&unit=THS"
)


def parse_hpi(document: dict[str, Any]) -> dict[int, float]:
    """Decode annual Portugal HPI from JSON-stat using dimension indexes.

    Raises:
        ValueError: If dimensions, filters, or numeric observations are invalid.
    """
    dimensions = document.get("id")
    sizes = document.get("size")
    if not isinstance(dimensions, list) or not isinstance(sizes, list):
        raise ValueError("Missing JSON-stat dimensions")
    if len(dimensions) != len(sizes) or not all(isinstance(n, int) and n > 0 for n in sizes):
        raise ValueError("Invalid JSON-stat dimension sizes")

    expected = {"geo": "PT", "purchase": "TOTAL", "unit": "I15_A_AVG", "freq": "A"}
    dimension_data = document.get("dimension", {})
    for dimension, code in expected.items():
        if dimension not in dimensions:
            raise ValueError(f"Missing dimension: {dimension}")
        indexes = dimension_data[dimension]["category"]["index"]
        if not isinstance(indexes, dict) or list(indexes) != [code]:
            raise ValueError(f"Unexpected selection for {dimension}: {indexes}")

    time_index = dimension_data["time"]["category"]["index"]
    if not isinstance(time_index, dict):
        raise ValueError("Invalid time dimension")
    values = document.get("value", {})
    if not isinstance(values, dict):
        raise ValueError("Invalid observations")
    result: dict[int, float] = {}
    for year, position in time_index.items():
        if not isinstance(position, int) or not isinstance(year, str) or not year.isdigit():
            raise ValueError("Invalid year index")
        coordinates = {name: 0 for name in dimensions}
        coordinates["time"] = position
        flat_index = 0
        for name, size in zip(dimensions, sizes, strict=True):
            flat_index = flat_index * size + coordinates[name]
        observation = values.get(str(flat_index))
        if observation is None:
            continue  # Eurostat can omit missing early observations.
        if not isinstance(observation, (float, int)) or isinstance(observation, bool):
            raise ValueError(f"Non-numeric observation in {year}")
        result[int(year)] = float(observation)
    if 2015 not in result or abs(result[2015] - 100.0) > 0.01:
        raise ValueError("Expected 2015=100 HPI series")
    return dict(sorted(result.items()))


def summarize(series: dict[int, float]) -> dict[str, float | int]:
    """Compute transparent arithmetic comparisons, not causal estimates."""
    if 2015 not in series or 2025 not in series or 2024 not in series:
        raise ValueError("Years 2015, 2024 and 2025 are required")
    return {
        "hpi_2015": series[2015],
        "hpi_2024": series[2024],
        "hpi_2025": series[2025],
        "growth_2015_2025_pct": round(100 * (series[2025] / series[2015] - 1), 2),
        "growth_2024_2025_pct": round(100 * (series[2025] / series[2024] - 1), 2),
        "owner_1977_1986_pct": 61,
        "owner_1987_1996_pct": 42,
        "owner_gap_percentage_points": 42 - 61,
        "vacant_dwellings_2021_pct": 12,
        "secondary_dwellings_2021_pct": 19,
    }


def parse_income(document: dict[str, Any], age: str = "TOTAL") -> dict[int, float]:
    """Read Eurostat annual median equivalised net income by age (EUR/person).

    Income is equivalised household income attributed to individuals in each
    age group, not the wages of those individuals or first-time buyers.
    """
    if age not in {"TOTAL", "Y18-24", "Y25-49"}:
        raise ValueError(f"Unsupported age category: {age}")
    expected = {"freq": "A", "geo": "PT", "age": age, "sex": "T", "statinfo": "MED_EI", "unit": "EUR"}
    dimensions = document.get("id")
    sizes = document.get("size")
    if not isinstance(dimensions, list) or not isinstance(sizes, list) or len(dimensions) != len(sizes):
        raise ValueError("Invalid income dimensions")
    for name, code in expected.items():
        if name not in dimensions or document["dimension"][name]["category"]["index"] != {code: 0}:
            raise ValueError(f"Unexpected income selection: {name}")
    if not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("Invalid income shape")
    if any(size != 1 for name, size in zip(dimensions, sizes, strict=True) if name != "time"):
        raise ValueError("Expected one income category per dimension")
    years = document["dimension"]["time"]["category"]["index"]
    observations = document.get("value", {})
    result: dict[int, float] = {}
    for year, position in years.items():
        value = observations.get(str(position))
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Invalid income for {year}")
        result[int(year)] = float(value)
    return dict(sorted(result.items()))


def relative_affordability(hpi: dict[int, float], income: dict[int, float], base_year: int = 2015) -> dict[int, float]:
    """Price-index / income-index proxy, normalized to 100 in `base_year`.

    This is not a transaction-price-to-income multiple, repayment burden, or
    causal estimate: it compares two national indices with different units.
    """
    if base_year not in hpi or base_year not in income or income[base_year] <= 0:
        raise ValueError("Missing or invalid common base year")
    common = sorted(hpi.keys() & income.keys())
    return {
        year: round(100 * (hpi[year] / hpi[base_year]) / (income[year] / income[base_year]), 2)
        for year in common if income[year] > 0
    }


def parse_housing_overburden(document: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Extract official 40%-threshold housing-cost overburden rates by age.

    Rates are shares of *people* living in households above the threshold,
    not average expenditure, house prices, or homeownership access.
    """
    dimensions = document.get("id")
    sizes = document.get("size")
    expected = {"freq": "A", "unit": "PC", "rskpovth": "TOTAL", "sex": "T", "geo": "PT"}
    if not isinstance(dimensions, list) or not isinstance(sizes, list) or len(dimensions) != len(sizes):
        raise ValueError("Invalid overburden dimensions")
    for name, code in expected.items():
        if name not in dimensions or document["dimension"][name]["category"]["index"] != {code: 0}:
            raise ValueError(f"Unexpected overburden selection: {name}")
    if not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("Invalid overburden dimension sizes")
    ages = document["dimension"]["age"]["category"]["index"]
    years = document["dimension"]["time"]["category"]["index"]
    values = document.get("value", {})
    if not isinstance(values, dict):
        raise ValueError("Invalid overburden values")
    result: dict[str, dict[int, float]] = {}
    for age in ("TOTAL", "Y18-24", "Y25-29"):
        if age not in ages:
            raise ValueError(f"Missing required age group: {age}")
        observations: dict[int, float] = {}
        for year, time_pos in years.items():
            coordinates = {name: 0 for name in dimensions}
            coordinates["age"] = ages[age]
            coordinates["time"] = time_pos
            flat_index = 0
            for name, size in zip(dimensions, sizes, strict=True):
                flat_index = flat_index * size + coordinates[name]
            value = values.get(str(flat_index))
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100:
                raise ValueError(f"Invalid overburden rate for {age}/{year}")
            observations[int(year)] = float(value)
        result[age] = dict(sorted(observations.items()))
    return result


def parse_parental_coresidence(document: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Read shares of Portuguese young adults living with parents by age.

    These are repeated survey cross-sections, not longitudinal transitions or
    measurements of why an individual continues to live with parents.
    """
    dimensions = document.get("id")
    sizes = document.get("size")
    expected = {"freq": "A", "sex": "T", "unit": "PC", "geo": "PT"}
    if not isinstance(dimensions, list) or not isinstance(sizes, list) or len(dimensions) != len(sizes):
        raise ValueError("Invalid co-residence dimensions")
    if not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("Invalid co-residence sizes")
    for name, code in expected.items():
        if name not in dimensions or document["dimension"][name]["category"]["index"] != {code: 0}:
            raise ValueError(f"Unexpected co-residence selection: {name}")
    ages = document["dimension"]["age"]["category"]["index"]
    years = document["dimension"]["time"]["category"]["index"]
    values = document.get("value", {})
    if not isinstance(values, dict):
        raise ValueError("Invalid co-residence observations")
    selected: dict[str, dict[int, float]] = {}
    for age in ("Y18-24", "Y18-34", "Y25-34"):
        if age not in ages:
            raise ValueError(f"Missing co-residence age: {age}")
        series: dict[int, float] = {}
        for year, time_pos in years.items():
            coordinates = {name: 0 for name in dimensions}
            coordinates["age"] = ages[age]
            coordinates["time"] = time_pos
            flat = 0
            for name, size in zip(dimensions, sizes, strict=True):
                flat = flat * size + coordinates[name]
            value = values.get(str(flat))
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100:
                raise ValueError(f"Invalid co-residence value: {age}/{year}")
            series[int(year)] = float(value)
        selected[age] = dict(sorted(series.items()))
    return selected


def parse_leaving_age(document: dict[str, Any]) -> dict[int, float]:
    """Read Eurostat's modelled age at 50% departure from parental home."""
    dimensions = document.get("id")
    sizes = document.get("size")
    if not isinstance(dimensions, list) or not isinstance(sizes, list) or len(dimensions) != len(sizes):
        raise ValueError("Invalid departure-age dimensions")
    expected = {"freq": "A", "unit": "AVG", "sex": "T", "geo": "PT"}
    for name, code in expected.items():
        if name not in dimensions or document["dimension"][name]["category"]["index"] != {code: 0}:
            raise ValueError(f"Unexpected departure-age selection: {name}")
    if any(size != 1 for name, size in zip(dimensions, sizes, strict=True) if name != "time"):
        raise ValueError("Departure-age data must have one selected category")
    years = document["dimension"]["time"]["category"]["index"]
    values = document.get("value", {})
    if not isinstance(values, dict):
        raise ValueError("Invalid departure-age observations")
    result: dict[int, float] = {}
    for year, position in years.items():
        value = values.get(str(position))
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 15 <= value <= 34:
            raise ValueError(f"Invalid departure-age value: {year}")
        result[int(year)] = float(value)
    return dict(sorted(result.items()))


def parse_tenure(document: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Extract person-weighted housing-tenure shares from Eurostat JSON-stat.

    Categories decompose population, not dwellings or property transactions.
    OWN_L/OWN_NL and RENT_MKT/RENT_FR are subsets of OWN and RENT.
    """
    dimensions = document.get("id")
    sizes = document.get("size")
    expected = {"freq": "A", "rskpovth": "TOTAL", "hhcomp": "TOTAL", "unit": "PC", "geo": "PT"}
    if not isinstance(dimensions, list) or not isinstance(sizes, list) or len(dimensions) != len(sizes):
        raise ValueError("Invalid tenure dimensions")
    if not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("Invalid tenure sizes")
    for name, code in expected.items():
        if name not in dimensions or document["dimension"][name]["category"]["index"] != {code: 0}:
            raise ValueError(f"Unexpected tenure selection: {name}")
    categories = document["dimension"]["tenure"]["category"]["index"]
    years = document["dimension"]["time"]["category"]["index"]
    values = document.get("value", {})
    if not isinstance(values, dict):
        raise ValueError("Invalid tenure observations")
    selected: dict[str, dict[int, float]] = {}
    for category in ("TOTAL", "OWN", "OWN_L", "OWN_NL", "RENT", "RENT_MKT", "RENT_FR"):
        if category not in categories:
            raise ValueError(f"Missing tenure category: {category}")
        series: dict[int, float] = {}
        for year, time_pos in years.items():
            coordinates = {name: 0 for name in dimensions}
            coordinates["tenure"] = categories[category]
            coordinates["time"] = time_pos
            flat = 0
            for name, size in zip(dimensions, sizes, strict=True):
                flat = flat * size + coordinates[name]
            value = values.get(str(flat))
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100:
                raise ValueError(f"Invalid tenure rate: {category}/{year}")
            series[int(year)] = float(value)
        selected[category] = dict(sorted(series.items()))
    for year in years:
        y = int(year)
        if all(y in selected[key] for key in selected):
            owner = selected["OWN"][y]
            renter = selected["RENT"][y]
            if abs(owner + renter - selected["TOTAL"][y]) > 0.21:
                raise ValueError(f"Tenure total does not balance for {year}")
            if abs(selected["OWN_L"][y] + selected["OWN_NL"][y] - owner) > 0.21:
                raise ValueError(f"Ownership composition does not balance for {year}")
            if abs(selected["RENT_MKT"][y] + selected["RENT_FR"][y] - renter) > 0.21:
                raise ValueError(f"Rental composition does not balance for {year}")
    return selected


def parse_dwellings_permitted(document: dict[str, Any]) -> dict[int, float]:
    """Read annual dwellings covered by building permits (thousands).

    This is authorised *potential* building activity, not completed homes,
    net additions to stock, permits for each building, or affordable supply.
    """
    dimensions = document.get("id")
    sizes = document.get("size")
    expected = {
        "freq": "A", "indic_bt": "BPRM_DW", "cpa2_1": "CPA_F41001_X_410014",
        "s_adj": "NSA", "unit": "THS", "geo": "PT",
    }
    if not isinstance(dimensions, list) or not isinstance(sizes, list) or len(dimensions) != len(sizes):
        raise ValueError("Invalid permits dimensions")
    if not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("Invalid permits dimension sizes")
    for name, code in expected.items():
        if name not in dimensions or document["dimension"][name]["category"]["index"] != {code: 0}:
            raise ValueError(f"Unexpected permits selection: {name}")
    if any(size != 1 for name, size in zip(dimensions, sizes, strict=True) if name != "time"):
        raise ValueError("Expected one permits category per non-time dimension")
    years = document["dimension"]["time"]["category"]["index"]
    values = document.get("value", {})
    if not isinstance(years, dict) or not isinstance(values, dict):
        raise ValueError("Invalid permits observations")
    result: dict[int, float] = {}
    for year, position in years.items():
        if not isinstance(year, str) or not year.isdigit() or not isinstance(position, int):
            raise ValueError("Invalid permits time index")
        value = values.get(str(position))
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid permits value for {year}")
        result[int(year)] = float(value)
    return dict(sorted(result.items()))


def main() -> None:
    """Read a pinned source snapshot and export auditable derived data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data" / "eurostat_prc_hpi_a_PT_2026-10-29.json")
    parser.add_argument("--income-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_2026-10-29.json")
    parser.add_argument("--income-18-24-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_Y18-24_2026-10-29.json")
    parser.add_argument("--income-25-49-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_Y25-49_2026-10-29.json")
    parser.add_argument("--overburden-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_ilc_lvho07a_PT_2026-10-29.json")
    parser.add_argument("--parents-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_ilc_lvps08_PT_2026-10-29.json")
    parser.add_argument("--leaving-age-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_yth_demo_030_PT_2026-10-29.json")
    parser.add_argument("--tenure-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_ilc_lvho02_PT_2026-10-29.json")
    parser.add_argument("--permits-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_sts_cobp_a_PT_dwellings_2026-10-29.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()
    raw: Any = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected JSON-stat object")
    series = parse_hpi(raw)
    raw_income: Any = json.loads(args.income_input.read_text(encoding="utf-8"))
    if not isinstance(raw_income, dict):
        raise ValueError("Expected income JSON-stat object")
    income = parse_income(raw_income)
    affordability = relative_affordability(series, income)
    by_age: dict[str, dict[str, Any]] = {}
    for label, path in {"Y18-24": args.income_18_24_input, "Y25-49": args.income_25_49_input}.items():
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON-stat object for {label}")
        age_income = parse_income(payload, age=label)
        by_age[label] = {
            "source_updated": payload.get("updated"),
            "income_eur": age_income,
            "relative_price_income_index_2015_100": relative_affordability(series, age_income),
        }
    raw_overburden: Any = json.loads(args.overburden_input.read_text(encoding="utf-8"))
    if not isinstance(raw_overburden, dict):
        raise ValueError("Expected overburden JSON-stat object")
    overburden = parse_housing_overburden(raw_overburden)
    raw_parents: Any = json.loads(args.parents_input.read_text(encoding="utf-8"))
    raw_leaving_age: Any = json.loads(args.leaving_age_input.read_text(encoding="utf-8"))
    if not isinstance(raw_parents, dict) or not isinstance(raw_leaving_age, dict):
        raise ValueError("Expected parental-home JSON-stat objects")
    raw_tenure: Any = json.loads(args.tenure_input.read_text(encoding="utf-8"))
    if not isinstance(raw_tenure, dict):
        raise ValueError("Expected tenure JSON-stat object")
    raw_permits: Any = json.loads(args.permits_input.read_text(encoding="utf-8"))
    if not isinstance(raw_permits, dict):
        raise ValueError("Expected permits JSON-stat object")
    permits = parse_dwellings_permitted(raw_permits)
    report = {
        "source_url": EUROSTAT_URL,
        "source_updated": raw.get("updated"),
        "income_source_url": INCOME_URL,
        "income_source_updated": raw_income.get("updated"),
        "yearly_hpi": series,
        "yearly_median_equivalised_income_eur": income,
        "relative_price_income_index_2015_100": affordability,
        "income_by_age": by_age,
        "housing_overburden_source_url": OVERBURDEN_URL,
        "housing_overburden_source_updated": raw_overburden.get("updated"),
        "housing_overburden_pct_by_age": overburden,
        "parental_coresidence_source_url": PARENTS_URL,
        "parental_coresidence_source_updated": raw_parents.get("updated"),
        "parental_coresidence_pct_by_age": parse_parental_coresidence(raw_parents),
        "leaving_parental_home_age_source_url": LEAVING_AGE_URL,
        "leaving_parental_home_age_source_updated": raw_leaving_age.get("updated"),
        "estimated_leaving_parental_home_age": parse_leaving_age(raw_leaving_age),
        "tenure_source_url": TENURE_URL,
        "tenure_source_updated": raw_tenure.get("updated"),
        "population_tenure_pct": parse_tenure(raw_tenure),
        "building_permits_source_url": PERMITS_URL,
        "building_permits_source_updated": raw_permits.get("updated"),
        "building_permits_status_flags": raw_permits.get("status", {}),
        "dwellings_permitted_thousands": permits,
        "dwellings_permitted_2015_2025_change_pct": round(100 * (permits[2025] / permits[2015] - 1), 2),
        "descriptives": summarize(series),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["descriptives"], indent=2))


if __name__ == "__main__":
    main()
