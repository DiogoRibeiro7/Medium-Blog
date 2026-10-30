"""Reproduce selected descriptive claims about Portuguese housing affordability.

Inputs are pinned snapshots of official series: Eurostat JSON-stat responses
(house prices, income, housing conditions, permits, households, population), an
INE/PORDATA transcription of completed dwellings, and the ECB/Banco de Portugal
cost of borrowing for house purchase. Nothing is downloaded here.
No causal effects or electoral preferences are estimated here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
COMPLETIONS_URL = (
    "https://www.pordata.pt/portugal/"
    "fogos+concluidos+em+construcoes+novas+para+habitacao+familiar+total+e+por+tipologia+do+fogo-189"
)
HOUSEHOLDS_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "lfst_hhnhtych?geo=PT&unit=THS_HH&agechild=TOTAL&n_child=TOTAL&phhcomp=TOTAL"
)
POPULATION_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "demo_pjan?geo=PT&age=TOTAL&sex=T"
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
        # External figures, quoted and not computed from the pinned data (see README, Sources 2 and 3).
        # Households whose reference person is aged 25-34, by birth cohort, Censos 1981-2021: "70%" (1967-76),
        # "60%" (1977-86) and "40%" (1987-96), as rounded in Costa, Farinha, Lourenço and Mesquita (2023).
        "owner_1977_1986_pct": 60,
        "owner_1987_1996_pct": 40,
        "owner_gap_percentage_points": 40 - 60,
        # INE, Censos 2021, indicator 0011495: 723,215 vacant and 1,104,881 secondary of 5,970,677 dwellings.
        "vacant_dwellings_2021_pct": round(100 * 723_215 / 5_970_677, 1),
        "secondary_dwellings_2021_pct": round(100 * 1_104_881 / 5_970_677, 1),
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


def parse_completed_dwellings(path: Path) -> tuple[dict[int, int], list[int]]:
    """Read INE/PORDATA annual new-build dwelling completions from CSV.

    A dwelling is a unit, not a building; only construction of new family
    dwellings is counted. Renovations and net additions to stock are excluded.
    Returns a year-to-count mapping and years marked preliminary by the source.
    """
    result: dict[int, int] = {}
    preliminary: list[int] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["year", "completed_dwellings", "preliminary"]:
            raise ValueError("Unexpected completions CSV schema")
        for row in reader:
            if not all(isinstance(row[k], str) for k in reader.fieldnames):
                raise ValueError("Incomplete completions CSV row")
            year_text = row["year"]
            count_text = row["completed_dwellings"]
            flag = row["preliminary"]
            if not year_text.isdigit() or not count_text.isdigit() or flag not in {"true", "false"}:
                raise ValueError("Invalid completions CSV value")
            year = int(year_text)
            count = int(count_text)
            if year in result or count <= 0:
                raise ValueError("Duplicate year or nonpositive completions count")
            result[year] = count
            if flag == "true":
                preliminary.append(year)
    if not result or sorted(result) != list(range(min(result), max(result) + 1)):
        raise ValueError("Completed-dwellings series has missing years")
    return dict(sorted(result.items())), sorted(preliminary)


def parse_demographic_totals(
    document: dict[str, Any], *, households: bool,
) -> tuple[dict[int, float], dict[int, str]]:
    """Validate a one-category Eurostat household or population time series.

    Household observations are in thousands; population observations are
    persons on 1 January. Break/status flags are keyed to calendar year.
    """
    expected = (
        {"freq": "A", "agechild": "TOTAL", "n_child": "TOTAL", "phhcomp": "TOTAL", "unit": "THS_HH", "geo": "PT"}
        if households else {"freq": "A", "unit": "NR", "age": "TOTAL", "sex": "T", "geo": "PT"}
    )
    dimensions = document.get("id")
    sizes = document.get("size")
    if not isinstance(dimensions, list) or not isinstance(sizes, list) or len(dimensions) != len(sizes):
        raise ValueError("Invalid demographic JSON-stat dimensions")
    if not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("Invalid demographic JSON-stat sizes")
    if set(dimensions) != set(expected) | {"time"}:
        raise ValueError("Unexpected demographic dimensions")
    for name, category in expected.items():
        if document["dimension"][name]["category"]["index"] != {category: 0}:
            raise ValueError(f"Unexpected demographic category: {name}")
    if any(size != 1 for name, size in zip(dimensions, sizes, strict=True) if name != "time"):
        raise ValueError("Expected single non-time demographic categories")
    years = document["dimension"]["time"]["category"]["index"]
    values = document.get("value", {})
    statuses = document.get("status", {})
    if not isinstance(years, dict) or not isinstance(values, dict) or not isinstance(statuses, dict):
        raise ValueError("Invalid demographic observations or flags")
    observations: dict[int, float] = {}
    flags: dict[int, str] = {}
    for year, position in years.items():
        if not isinstance(year, str) or not year.isdigit() or not isinstance(position, int):
            raise ValueError("Invalid demographic time index")
        if position < 0 or position >= sizes[dimensions.index("time")]:
            raise ValueError("Demographic time index outside shape")
        value = values.get(str(position))
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Invalid demographic observation in {year}")
        observations[int(year)] = float(value)
        if str(position) in statuses:
            flag = statuses[str(position)]
            if not isinstance(flag, str):
                raise ValueError("Invalid demographic observation flag")
            flags[int(year)] = flag
    return dict(sorted(observations.items())), dict(sorted(flags.items()))


def construction_intensity(
    completions: dict[int, int], population: dict[int, float], households: dict[int, float],
) -> dict[int, dict[str, float]]:
    """Calculate completed new dwellings per 1,000 people and households.

    Household totals are already in thousands, while population is in persons.
    These stock-standardized construction flows do not estimate housing need.
    """
    common = sorted(completions.keys() & population.keys() & households.keys())
    result: dict[int, dict[str, float]] = {}
    for year in common:
        people = population[year]
        thousand_households = households[year]
        if people <= 0 or thousand_households <= 0:
            raise ValueError(f"Invalid denominator for {year}")
        result[year] = {
            "completed_per_1000_residents": round(1000 * completions[year] / people, 3),
            "completed_per_1000_households": round(completions[year] / thousand_households, 3),
        }
    return result


MORTGAGE_SERIES_KEY = "MIR.M.PT.B.A2C.AM.R.A.2250.EUR.N"
MORTGAGE_RATE_URL = (
    "https://data-api.ecb.europa.eu/service/data/MIR/"
    "M.PT.B.A2C.AM.R.A.2250.EUR.N?format=csvdata"
)
_MORTGAGE_DIMENSIONS = {
    "FREQ": "M", "REF_AREA": "PT", "BS_REP_SECTOR": "B", "BS_ITEM": "A2C",
    "MATURITY_NOT_IRATE": "AM", "DATA_TYPE_MIR": "R", "AMOUNT_CAT": "A",
    "BS_COUNT_SECTOR": "2250", "CURRENCY_TRANS": "EUR", "IR_BUS_COV": "N",
}


def verify_snapshot(data_path: Path, record_path: Path) -> dict[str, Any]:
    """Check a pinned raw response against the retrieval record written when it was fetched.

    Returns the record. Raises ValueError if the size or SHA-256 digest differs,
    which catches edits, truncation and line-ending normalisation by git.
    """
    record: Any = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not {"url", "sha256", "bytes", "retrieved_at_utc"} <= set(record):
        raise ValueError("Incomplete retrieval record")
    body = data_path.read_bytes()
    if len(body) != record["bytes"] or hashlib.sha256(body).hexdigest() != record["sha256"]:
        raise ValueError(f"{data_path.name} does not match its retrieval record")
    return record


def parse_mortgage_rates(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Read the ECB cost-of-borrowing series for Portuguese house-purchase loans.

    Returns monthly observations keyed ``YYYY-MM`` (percent per annum) and any
    observation status other than ``A`` (normal). Months without a value are left
    out, never filled. Raises ValueError for another series, unit or frequency,
    duplicated or non-consecutive months, or implausible values.
    """
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"KEY", "TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "UNIT", "UNIT_MULT", *_MORTGAGE_DIMENSIONS}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("Unexpected cost-of-borrowing CSV schema")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty cost-of-borrowing series")
    rates: dict[str, float] = {}
    flags: dict[str, str] = {}
    periods: list[str] = []
    for row in rows:
        if row["KEY"] != MORTGAGE_SERIES_KEY or any(row[k] != v for k, v in _MORTGAGE_DIMENSIONS.items()):
            raise ValueError(f"Unexpected series in cost-of-borrowing file: {row['KEY']}")
        if row["UNIT"] != "PCPA" or row["UNIT_MULT"] != "0":
            raise ValueError("Cost of borrowing must be in percent per annum")
        period = row["TIME_PERIOD"]
        if len(period) != 7 or period[4] != "-" or not (period[:4].isdigit() and period[5:].isdigit()) \
                or not 1 <= int(period[5:]) <= 12:
            raise ValueError(f"Invalid month: {period}")
        if period in periods:
            raise ValueError(f"Duplicate month: {period}")
        periods.append(period)
        if row["OBS_STATUS"] not in {"", "A"}:
            flags[period] = row["OBS_STATUS"]
        text = row["OBS_VALUE"].strip()
        if not text:
            continue
        value = float(text)
        if not 0.0 <= value < 25.0:
            raise ValueError(f"Implausible cost of borrowing in {period}: {value}")
        rates[period] = value
    ordinal = [int(p[:4]) * 12 + int(p[5:]) - 1 for p in periods]
    if ordinal != list(range(ordinal[0], ordinal[0] + len(ordinal))):
        raise ValueError("Cost-of-borrowing months are not consecutive")
    return rates, flags


def annual_mean_rates(monthly: dict[str, float]) -> tuple[dict[int, float], dict[int, int]]:
    """Unweighted mean of the monthly rates for each calendar year with all 12 months.

    Years with fewer observed months are not averaged; they are returned
    separately with the number of months observed.
    """
    by_year: dict[int, list[float]] = {}
    for period, value in monthly.items():
        by_year.setdefault(int(period[:4]), []).append(value)
    complete = {year: round(sum(values) / 12, 4) for year, values in by_year.items() if len(values) == 12}
    incomplete = {year: len(values) for year, values in by_year.items() if len(values) != 12}
    return dict(sorted(complete.items())), dict(sorted(incomplete.items()))


def annuity_factor(rate_pct: float, term_years: int = 30) -> float:
    """Monthly instalment per euro borrowed on a level-payment loan at a fixed rate.

    ``rate_pct`` is an annual rate in percent, compounded monthly.
    """
    if term_years <= 0:
        raise ValueError("Loan term must be positive")
    if rate_pct < 0:
        raise ValueError("Negative rates are not supported")
    months = 12 * term_years
    monthly_rate = rate_pct / 1200
    if monthly_rate == 0:
        return 1 / months
    return monthly_rate / (1 - (1 + monthly_rate) ** -months)


def payment_to_income_index(
    hpi: dict[int, float], income: dict[int, float], rates: dict[int, float], *,
    base_year: int = 2015, term_years: int = 30, income_survey_offset: int = 0,
) -> dict[int, float]:
    """Initial mortgage instalment on an HPI-indexed loan relative to income, base year = 100.

    For calendar year t the index is 100 * (HPI_t / HPI_b) * (a(r_t) / a(r_b)) /
    (Y_{t+k} / Y_{b+k}), where a(r) is the annuity factor and k is
    ``income_survey_offset``: 0 pairs prices with the EU-SILC survey of the same
    year (as the existing price/income index does); 1 pairs them with the next
    survey, whose income refers to year t. Years missing any input are left out.
    The loan is assumed proportional to the price index (constant loan-to-value
    and dwelling quality); this is a descriptive index, not a repayment burden.
    """
    if income_survey_offset not in (0, 1):
        raise ValueError("income_survey_offset must be 0 or 1")
    k = income_survey_offset
    if base_year not in hpi or base_year not in rates or base_year + k not in income:
        raise ValueError("Missing base-year input")
    base_annuity = annuity_factor(rates[base_year], term_years)
    result: dict[int, float] = {}
    for year in sorted(hpi.keys() & rates.keys()):
        if year + k not in income:
            continue
        ratio = (hpi[year] / hpi[base_year]) * (annuity_factor(rates[year], term_years) / base_annuity)
        result[year] = round(100 * ratio / (income[year + k] / income[base_year + k]), 2)
    return result


def decompose_payment_change(
    hpi: dict[int, float], income: dict[int, float], rates: dict[int, float],
    start: int, end: int, *, term_years: int = 30, income_survey_offset: int = 0,
) -> dict[str, float]:
    """Split the log change of the payment-to-income index into price, financing and income terms.

    Values are in log points (100 * natural log ratio) and add up exactly:
    total = price + financing_cost - income. This is an accounting identity,
    not an attribution of causes.
    """
    k = income_survey_offset
    for name, series, years in (("HPI", hpi, (start, end)), ("rate", rates, (start, end)), ("income", income, (start + k, end + k))):
        if any(y not in series for y in years):
            raise ValueError(f"Missing {name} observation for {start}-{end}")
    price = 100 * math.log(hpi[end] / hpi[start])
    financing = 100 * math.log(annuity_factor(rates[end], term_years) / annuity_factor(rates[start], term_years))
    income_growth = 100 * math.log(income[end + k] / income[start + k])
    return {
        "price": round(price, 2),
        "financing_cost": round(financing, 2),
        "income": round(income_growth, 2),
        "total": round(price + financing - income_growth, 2),
    }


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
    parser.add_argument("--completions-input", type=Path, default=Path(__file__).parent / "data" / "ine_pordata_completed_new_dwellings_PT_2026-10-29.csv")
    parser.add_argument("--households-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_lfst_hhnhtych_PT_2026-10-29.json")
    parser.add_argument("--population-input", type=Path, default=Path(__file__).parent / "data" / "eurostat_demo_pjan_PT_2026-10-29.json")
    parser.add_argument("--mortgage-rate-input", type=Path, default=Path(__file__).parent / "data" / "ecb_mir_PT_house_purchase_cost_of_borrowing_2026-10-29.csv")
    parser.add_argument("--mortgage-rate-record", type=Path, default=Path(__file__).parent / "data" / "ecb_mir_PT_house_purchase_cost_of_borrowing_2026-10-29.meta.json")
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
    completed, preliminary_completed = parse_completed_dwellings(args.completions_input)
    raw_households: Any = json.loads(args.households_input.read_text(encoding="utf-8"))
    raw_population: Any = json.loads(args.population_input.read_text(encoding="utf-8"))
    if not isinstance(raw_households, dict) or not isinstance(raw_population, dict):
        raise ValueError("Expected demographic JSON-stat objects")
    households, household_flags = parse_demographic_totals(raw_households, households=True)
    population, population_flags = parse_demographic_totals(raw_population, households=False)
    rate_record = verify_snapshot(args.mortgage_rate_input, args.mortgage_rate_record)
    monthly_rates, rate_flags = parse_mortgage_rates(args.mortgage_rate_input)
    annual_rates, incomplete_rate_years = annual_mean_rates(monthly_rates)
    payment_index = {
        "survey_year_alignment": payment_to_income_index(series, income, annual_rates),
        "income_reference_year_alignment": payment_to_income_index(series, income, annual_rates, income_survey_offset=1),
    }
    term_sensitivity = {
        str(term): payment_to_income_index(series, income, annual_rates, term_years=term, income_survey_offset=1)
        for term in (25, 35, 40)
    }
    periods = {"survey_year_alignment": ((2008, 2015), (2015, 2021), (2021, 2023), (2023, 2025), (2015, 2025)),
               "income_reference_year_alignment": ((2008, 2015), (2015, 2021), (2021, 2023), (2023, 2024), (2015, 2024))}
    decomposition = {
        alignment: {
            f"{start}-{end}": decompose_payment_change(series, income, annual_rates, start, end,
                                                       income_survey_offset=int(alignment.startswith("income")))
            for start, end in spans
        }
        for alignment, spans in periods.items()
    }
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
        "completions_source_url": COMPLETIONS_URL,
        "completions_original_source": "INE, Estatísticas das obras concluídas; PORDATA annual table",
        "completed_new_family_dwellings": completed,
        "completed_new_family_dwellings_preliminary_years": preliminary_completed,
        "completed_dwellings_2015_2025_change_pct": round(100 * (completed[2025] / completed[2015] - 1), 2),
        "households_source_url": HOUSEHOLDS_URL,
        "households_source_updated": raw_households.get("updated"),
        "private_households_thousands": households,
        "private_households_status_flags": household_flags,
        "population_source_url": POPULATION_URL,
        "population_source_updated": raw_population.get("updated"),
        "population_on_january_1": population,
        "population_status_flags": population_flags,
        "completed_new_dwellings_construction_intensity": construction_intensity(completed, population, households),
        "mortgage_rate_source_url": MORTGAGE_RATE_URL,
        "mortgage_rate_series_key": MORTGAGE_SERIES_KEY,
        "mortgage_rate_retrieved_at_utc": rate_record["retrieved_at_utc"],
        "mortgage_rate_source_last_modified": rate_record.get("response_headers", {}).get("Last-Modified"),
        "mortgage_rate_sha256": rate_record["sha256"],
        "mortgage_rate_months_observed": [min(monthly_rates), max(monthly_rates)],
        "mortgage_rate_status_flags": rate_flags,
        "mortgage_cost_of_borrowing_annual_mean_pct": annual_rates,
        "mortgage_cost_of_borrowing_incomplete_years_months": incomplete_rate_years,
        "initial_payment_to_income_index_2015_100": payment_index,
        "initial_payment_to_income_index_term_sensitivity_2015_100": term_sensitivity,
        "initial_payment_to_income_change_log_points": decomposition,
        "descriptives": summarize(series),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["descriptives"], indent=2))


if __name__ == "__main__":
    main()
