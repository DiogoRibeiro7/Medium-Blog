"""Build auditable 2007–2019 U.S. crisis-and-recovery tables offline.

The input files are frozen public BLS API responses and FRED CSV exports.
The separate SCF table is a documented transcription from a Federal Reserve
accessible-data table. This module never downloads or extrapolates data.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from datetime import date
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
YEARS: tuple[int, ...] = tuple(range(2007, 2020))
RACES: tuple[str, ...] = ("White", "Black")

PANEL_FIELDS: tuple[str, ...] = (
    "year",
    "unemployment_pct",
    "white_unemployment_pct",
    "black_unemployment_pct",
    "black_white_unemployment_gap_pp",
    "payroll_employment_millions",
    "payroll_index_dec2007",
    "real_gdp_chained_billions",
    "real_gdp_index_2007q4",
    "fhfa_house_price_index",
    "fhfa_price_index_2007q4",
    "debt_service_pct",
    "real_median_household_income_2025_usd",
    "median_income_index_2007",
    "household_income_gini",
    "fed_funds_rate_pct",
    "fed_assets_trillions_usd",
)
RACE_FIELDS: tuple[str, ...] = (
    "observation_date",
    "white_unemployment_pct",
    "black_unemployment_pct",
    "gap_percentage_points",
)


def numeric(value: str | float | None, *, context: str) -> float:
    """Reject missing, non-numeric or non-finite observations."""
    if value is None or str(value).strip() in {"", ".", "NA", "NaN"}:
        raise ValueError(f"Missing observation: {context}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite observation: {context}")
    return result


def read_fred(path: Path, column: str) -> dict[date, float]:
    """Read dated FRED CSV values, retaining original frequencies and units."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "observation_date" not in reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"Incorrect columns in source snapshot: {path}")
        observed: dict[date, float] = {}
        for row in reader:
            day = date.fromisoformat(row["observation_date"])
            if day in observed:
                raise ValueError(f"Duplicate {column} observation on {day}")
            raw_value = row[column]
            if raw_value is not None and raw_value.strip() not in {"", "."}:
                observed[day] = numeric(raw_value, context=f"{column} on {day}")
    if not observed:
        raise ValueError(f"No observations in {path}")
    return observed


def monthly(source: Mapping[date, float], year: int, column: str) -> list[float]:
    """Require all twelve first-of-month observations, with no imputation."""
    expected = [date(year, month, 1) for month in range(1, 13)]
    missing = [day.isoformat() for day in expected if day not in source]
    if missing:
        raise ValueError(f"Missing monthly {column} observations: {missing}")
    return [source[d] for d in expected]


def quarterly(source: Mapping[date, float], year: int, column: str) -> list[float]:
    """Require four source quarterly observations."""
    expected = [date(year, month, 1) for month in (1, 4, 7, 10)]
    missing = [day.isoformat() for day in expected if day not in source]
    if missing:
        raise ValueError(f"Missing quarterly {column} observations: {missing}")
    return [source[d] for d in expected]


def annual(source: Mapping[date, float], year: int, column: str) -> float:
    """Require the one January-dated annual Census observation."""
    key = date(year, 1, 1)
    if key not in source:
        raise ValueError(f"Missing annual {column} observation in {year}")
    return source[key]


def read_bls_race() -> dict[str, dict[date, float]]:
    """Validate two BLS API responses with disjoint year windows.

    BLS period M13 (annual average) is not a monthly observation. The requested
    series are the seasonally adjusted CPS unemployment percentages for
    White (LNS14000003) and Black (LNS14000006) labor forces.
    """
    mapping = {"LNS14000003": "White", "LNS14000006": "Black"}
    values: dict[str, dict[date, float]] = {group: {} for group in RACES}
    for filename in ("bls_race_2007_2016.json", "bls_race_2017_2019.json"):
        payload = json.loads((RAW / filename).read_text(encoding="utf-8"))
        if payload.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(f"Unsuccessful BLS API source: {filename}")
        series = payload.get("Results", {}).get("series", [])
        if {item.get("seriesID") for item in series} != set(mapping):
            raise ValueError(f"Unexpected BLS series IDs: {filename}")
        for item in series:
            group = mapping[item["seriesID"]]
            for observation in item["data"]:
                period = observation["period"]
                if not (len(period) == 3 and period[0] == "M" and period[1:].isdigit()):
                    continue
                month = int(period[1:])
                if not 1 <= month <= 12:
                    continue
                day = date(int(observation["year"]), month, 1)
                if day in values[group]:
                    raise ValueError(f"Duplicate BLS {group} observation on {day}")
                value = numeric(observation["value"], context=f"BLS {group} on {day}")
                if not 0 <= value <= 30:
                    raise ValueError(f"Implausible BLS unemployment: {day}, {group}")
                values[group][day] = value
    expected = {date(year, month, 1) for year in YEARS for month in range(1, 13)}
    for group in RACES:
        if set(values[group]) != expected:
            raise ValueError(f"Invalid BLS {group} coverage: {len(values[group])} months")
    return values


def race_rows() -> list[dict[str, str | float]]:
    """Combine original API snapshots into an ordered monthly BLS table."""
    race = read_bls_race()
    return [
        {
            "observation_date": day.isoformat(),
            "white_unemployment_pct": race["White"][day],
            "black_unemployment_pct": race["Black"][day],
            "gap_percentage_points": race["Black"][day] - race["White"][day],
        }
        for day in sorted(race["White"])
    ]


def load_sources() -> dict[str, dict[date, float]]:
    """Read the explicitly enumerated source series, without network access."""
    specification = {
        "UNRATE": ("fred_UNRATE.csv", "UNRATE"),
        "PAYEMS": ("fred_PAYEMS.csv", "PAYEMS"),
        "FEDFUNDS": ("fred_FEDFUNDS.csv", "FEDFUNDS"),
        "GDPC1": ("fred_GDPC1.csv", "GDPC1"),
        "USSTHPI": ("fred_debt_housing.csv", "USSTHPI"),
        "TDSP": ("fred_debt_housing.csv", "TDSP"),
        "MEHOINUSA672N": ("fred_income_gini.csv", "MEHOINUSA672N"),
        "GINIALLRH": ("fred_income_gini.csv", "GINIALLRH"),
        "WALCL": ("fred_WALCL.csv", "WALCL"),
    }
    return {key: read_fred(RAW / filename, column) for key, (filename, column) in specification.items()}


def build_panel() -> list[dict[str, float | int]]:
    """Form annual descriptive observations using each series' original units."""
    source = load_sources()
    race = read_bls_race()
    gdp_base = source["GDPC1"][date(2007, 10, 1)]
    price_base = source["USSTHPI"][date(2007, 10, 1)]
    jobs_base = source["PAYEMS"][date(2007, 12, 1)]
    income_base = annual(source["MEHOINUSA672N"], 2007, "MEHOINUSA672N")
    if min(gdp_base, price_base, jobs_base, income_base) <= 0:
        raise ValueError("Positive index base required")
    result: list[dict[str, float | int]] = []
    for year in YEARS:
        white = fmean(monthly(race["White"], year, "BLS White"))
        black = fmean(monthly(race["Black"], year, "BLS Black"))
        payroll = fmean(monthly(source["PAYEMS"], year, "PAYEMS"))
        gdp = fmean(quarterly(source["GDPC1"], year, "GDPC1"))
        price = fmean(quarterly(source["USSTHPI"], year, "USSTHPI"))
        assets = [v for day, v in source["WALCL"].items() if day.year == year]
        if not 50 <= len(assets) <= 54:
            raise ValueError(f"Unexpected weekly Fed observation count in {year}: {len(assets)}")
        values: dict[str, float | int] = {
            "year": year,
            "unemployment_pct": fmean(monthly(source["UNRATE"], year, "UNRATE")),
            "white_unemployment_pct": white,
            "black_unemployment_pct": black,
            "black_white_unemployment_gap_pp": black - white,
            "payroll_employment_millions": payroll / 1000.0,
            "payroll_index_dec2007": 100 * payroll / jobs_base,
            "real_gdp_chained_billions": gdp,
            "real_gdp_index_2007q4": 100 * gdp / gdp_base,
            "fhfa_house_price_index": price,
            "fhfa_price_index_2007q4": 100 * price / price_base,
            "debt_service_pct": fmean(quarterly(source["TDSP"], year, "TDSP")),
            "real_median_household_income_2025_usd": annual(source["MEHOINUSA672N"], year, "MEHOINUSA672N"),
            "median_income_index_2007": 100 * annual(source["MEHOINUSA672N"], year, "MEHOINUSA672N") / income_base,
            "household_income_gini": annual(source["GINIALLRH"], year, "GINIALLRH"),
            "fed_funds_rate_pct": fmean(monthly(source["FEDFUNDS"], year, "FEDFUNDS")),
            "fed_assets_trillions_usd": fmean(assets) / 1_000_000.0,
        }
        if set(values) != set(PANEL_FIELDS):
            raise ValueError(f"Unexpected panel schema in {year}")
        if not 0.3 < values["household_income_gini"] < 0.7:
            raise ValueError(f"Implausible household Gini in {year}")
        if not 0 < values["debt_service_pct"] < 30:
            raise ValueError(f"Implausible debt-service burden in {year}")
        result.append(values)
    return result


def serialize(rows: Sequence[Mapping[str, float | str | int]], fields: Iterable[str]) -> str:
    """Format CSV deterministically, retaining ISO dates and six decimals."""
    fieldnames = tuple(fields)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: str(row[key]) if key in {"year", "observation_date"} else f"{float(row[key]):.6f}"
                for key in fieldnames
            }
        )
    return stream.getvalue()


def prepared_outputs() -> dict[Path, str]:
    """Return exact reproducible bytes for all derived source tables."""
    return {
        DATA / "annual_panel_2007_2019.csv": serialize(build_panel(), PANEL_FIELDS),
        DATA / "race_monthly_2007_2019.csv": serialize(race_rows(), RACE_FIELDS),
    }


def main() -> None:
    """Generate or audit source transformations without external requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed tables without writing")
    args = parser.parse_args()
    for path, expected in prepared_outputs().items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"Derived table drift: {path}")
            print(f"Verified {path.relative_to(ROOT)}")
        else:
            path.write_text(expected, encoding="utf-8")
            print(f"Generated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
