"""Build the frozen 1990–2007 annual panel from checked-in source snapshots.

There is deliberately no network access in this transformation. Each source's
publisher, measurement definition, frequency and download vintage are recorded
in ``data/sources.json``. Running with ``--check`` is a non-mutating audit.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
from statistics import fmean
from typing import Sequence

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
YEARS: tuple[int, ...] = tuple(range(1990, 2008))

PANEL_FIELDS: tuple[str, ...] = (
    "year",
    "manufacturing_employment_millions",
    "total_payroll_employment_millions",
    "manufacturing_payroll_share_pct",
    "net_exports_gdp_pct",
    "imports_gdp_pct",
    "exports_gdp_pct",
    "real_gdp_per_capita_2017_usd",
    "real_median_household_income_2025_usd",
    "nominal_median_household_income_usd",
    "household_income_gini",
    "union_membership_rate_pct",
    "union_members_thousands",
    "private_union_members_thousands",
    "public_union_members_thousands",
    "fhfa_house_price_index",
    "debt_to_gdp_pct",
    "mortgage_debt_to_gdp_pct",
    "mortgage_share_of_debt_pct",
    "fed_funds_rate_pct",
    "ten_year_treasury_yield_pct",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a FRED or BLS snapshot with explicit UTF-8 CSV semantics."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Absent or duplicated headers in {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Empty table: {path}")
    return rows


def numeric(value: str | None, *, label: str) -> float:
    """Require a finite, present source value and reject missing markers."""
    if value is None or value.strip() in {"", ".", "NA", "NaN"}:
        raise ValueError(f"Missing {label}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def observations(
    rows: Sequence[dict[str, str]],
    column: str,
    year: int,
    months: tuple[int, ...],
) -> list[float]:
    """Require exactly one observation for each expected month in a year."""
    selected: dict[int, float] = {}
    for row in rows:
        date = row.get("observation_date", "")
        if not date.startswith(f"{year}-"):
            continue
        month = int(date[5:7])
        if month not in months:
            continue
        if month in selected:
            raise ValueError(f"Duplicate {column} observation in {year}-{month:02d}")
        selected[month] = numeric(row.get(column), label=f"{column} {date}")
    if set(selected) != set(months):
        raise ValueError(f"Incomplete {column} {year}: expected {months}, got {tuple(selected)}")
    return [selected[month] for month in months]


def annual_mean(rows: Sequence[dict[str, str]], column: str, year: int, frequency: str) -> float:
    """Aggregate monthly/quarterly observations using explicit period counts."""
    month_map = {
        "monthly": tuple(range(1, 13)),
        "quarterly": (1, 4, 7, 10),
        "annual": (1,),
    }
    if frequency not in month_map:
        raise ValueError(f"Unsupported frequency: {frequency}")
    return fmean(observations(rows, column, year, month_map[frequency]))


def fourth_quarter(rows: Sequence[dict[str, str]], column: str, year: int) -> float:
    """Read the Q4 level, not an average of balance-sheet stocks."""
    return observations(rows, column, year, (10,))[0]


def union_by_year(rows: Sequence[dict[str, str]]) -> dict[int, dict[str, float]]:
    """Validate the independently rounded BLS membership transcription."""
    result: dict[int, dict[str, float]] = {}
    fields = (
        "union_membership_rate_pct",
        "union_members_thousands",
        "private_union_members_thousands",
        "public_union_members_thousands",
    )
    for row in rows:
        year = int(row["year"])
        if year in result:
            raise ValueError(f"Duplicate union observation: {year}")
        values = {key: numeric(row.get(key), label=f"BLS {key} {year}") for key in fields}
        if not 0.0 < values["union_membership_rate_pct"] < 100.0:
            raise ValueError(f"Invalid union rate in {year}")
        if any(values[key] <= 0.0 for key in fields[1:]):
            raise ValueError(f"Invalid union members in {year}")
        result[year] = values
    if tuple(sorted(result)) != YEARS:
        raise ValueError("BLS union membership years do not match 1990–2007")
    return result


def build_panel() -> list[dict[str, float | int]]:
    """Construct one annual record per year without imputing missing data.

    Household debt is a Q4 stock in millions. GDP is the Q4 nominal quarterly
    flow expressed at an annual rate in billions. Dividing the two is a
    stock-to-current-output ratio, not debt as a fraction of total income.
    """
    monthly = read_rows(DATA / "raw" / "fred_monthly.csv")
    rates = read_rows(DATA / "raw" / "fred_treasury_yield.csv")
    quarterly = read_rows(DATA / "raw" / "quarterly.csv")
    stocks = read_rows(DATA / "raw" / "quarterly,_end_of_period.csv")
    census = read_rows(DATA / "raw" / "fred_annual.csv")
    union = union_by_year(read_rows(DATA / "bls_union_1990_2007.csv"))

    panel: list[dict[str, float | int]] = []
    for year in YEARS:
        manuf = annual_mean(monthly, "MANEMP", year, "monthly") / 1000.0
        payroll = annual_mean(monthly, "PAYEMS", year, "monthly") / 1000.0
        gdp_mean = annual_mean(quarterly, "GDP", year, "quarterly")
        gdp_q4 = fourth_quarter(quarterly, "GDP", year)
        exports = annual_mean(quarterly, "EXPGS", year, "quarterly")
        imports = annual_mean(quarterly, "IMPGS", year, "quarterly")
        net_exports = annual_mean(quarterly, "NETEXP", year, "quarterly")
        if abs((exports - imports) - net_exports) > 0.01:
            raise ValueError(f"BEA export-import identity failed in {year}")

        debt_millions = fourth_quarter(stocks, "CMDEBT", year)
        mortgage_millions = fourth_quarter(stocks, "HHMSDODNS", year)
        if not 0.0 < mortgage_millions <= debt_millions:
            raise ValueError(f"Implausible household mortgage stock in {year}")

        values: dict[str, float | int] = {
            "year": year,
            "manufacturing_employment_millions": manuf,
            "total_payroll_employment_millions": payroll,
            "manufacturing_payroll_share_pct": 100.0 * manuf / payroll,
            "net_exports_gdp_pct": 100.0 * net_exports / gdp_mean,
            "imports_gdp_pct": 100.0 * imports / gdp_mean,
            "exports_gdp_pct": 100.0 * exports / gdp_mean,
            "real_gdp_per_capita_2017_usd": annual_mean(quarterly, "A939RX0Q048SBEA", year, "quarterly"),
            "real_median_household_income_2025_usd": annual_mean(census, "MEHOINUSA672N", year, "annual"),
            "nominal_median_household_income_usd": annual_mean(census, "MEHOINUSA646N", year, "annual"),
            "household_income_gini": annual_mean(census, "GINIALLRH", year, "annual"),
            "fhfa_house_price_index": annual_mean(quarterly, "USSTHPI", year, "quarterly"),
            "debt_to_gdp_pct": 100.0 * (debt_millions / 1000.0) / gdp_q4,
            "mortgage_debt_to_gdp_pct": 100.0 * (mortgage_millions / 1000.0) / gdp_q4,
            "mortgage_share_of_debt_pct": 100.0 * mortgage_millions / debt_millions,
            "fed_funds_rate_pct": annual_mean(monthly, "FEDFUNDS", year, "monthly"),
            "ten_year_treasury_yield_pct": annual_mean(rates, "GS10", year, "monthly"),
            **union[year],
        }
        if set(values) != set(PANEL_FIELDS):
            raise ValueError(f"Panel columns do not match schema in {year}")
        if not 0.2 < float(values["household_income_gini"]) < 0.7:
            raise ValueError(f"Implausible Gini coefficient in {year}")
        panel.append(values)
    return panel


def serialize_panel(rows: Sequence[dict[str, float | int]]) -> str:
    """Produce a byte-stable CSV with six-decimal rounded numeric fields."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=PANEL_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: str(row[key]) if key == "year" else f"{float(row[key]):.6f}" for key in PANEL_FIELDS})
    return stream.getvalue()


def main() -> None:
    """Regenerate or audit the checked-in annual panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the existing panel without writing")
    args = parser.parse_args()
    path = DATA / "annual_panel_1990_2007.csv"
    expected = serialize_panel(build_panel())
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"Panel drift detected: {path}; run prepare_data.py")
        print(f"Verified {path}: {len(YEARS)} annual observations")
    else:
        path.write_text(expected, encoding="utf-8")
        print(f"Created {path}: {len(YEARS)} annual observations")


if __name__ == "__main__":
    main()
