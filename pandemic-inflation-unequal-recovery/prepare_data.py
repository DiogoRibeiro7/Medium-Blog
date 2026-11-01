"""Offline preparation of pandemic, inflation and recovery evidence, 2019–2025.

BLS API observations and previously frozen public FRED CSV snapshots are the
only inputs. Analyses use current retrospective vintages, not 2020 real-time
vintages. No interpolation or network calls are made.
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
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
YEARS: tuple[int, ...] = tuple(range(2019, 2026))
MONTHS: tuple[date, ...] = tuple(date(y, m, 1) for y in YEARS for m in range(1, 13))
BLS_IDS: tuple[str, ...] = (
    "LNS14000000",  # Total unemployment rate, seasonally adjusted
    "LNS14000003",  # White unemployment rate, seasonally adjusted
    "LNS14000006",  # Black unemployment rate, seasonally adjusted
    "CUSR0000SA0",  # CPI-U all items, seasonally adjusted
    "CUUR0000SA0",  # CPI-U all items, not seasonally adjusted
    "CUSR0000SAH1",  # CPI-U shelter, seasonally adjusted
    "CES0500000003",  # Total private average hourly earnings, seasonally adjusted
)

MONTHLY_FIELDS: tuple[str, ...] = (
    "observation_date",
    "total_unemployment_pct",
    "white_unemployment_pct",
    "black_unemployment_pct",
    "black_white_gap_pp",
    "cpi_sa_index_dec2019",
    "cpi_nsa_12m_inflation_pct",
    "shelter_sa_index_dec2019",
    "nominal_hourly_earnings_usd",
    "nominal_earnings_index_dec2019",
    "real_earnings_index_dec2019",
)
PANEL_FIELDS: tuple[str, ...] = (
    "year",
    "unemployment_pct",
    "white_unemployment_pct",
    "black_unemployment_pct",
    "black_white_gap_pp",
    "payroll_employment_millions",
    "payroll_index_feb2020",
    "real_gdp_chained_billions",
    "real_gdp_index_2019q4",
    "cpi_price_index_dec2019",
    "headline_cpi_12m_dec_pct",
    "nominal_hourly_earnings_usd",
    "nominal_earnings_index_dec2019",
    "real_earnings_index_dec2019",
    "shelter_cpi_index_dec2019",
    "fhfa_house_price_index_2019q4",
    "real_median_household_income_2025_usd",
    "median_income_index_2019",
    "household_income_gini",
    "fed_funds_rate_pct",
    "fed_assets_trillions_usd",
)


def number(value: object, context: str) -> float:
    """Accept only finite source numbers, never missing observations."""
    if value is None or str(value).strip() in {"", ".", "NA", "NaN"}:
        raise ValueError(f"Missing {context}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {context}")
    return result


def read_bls() -> dict[str, dict[date, float | None]]:
    """Parse one official BLS API snapshot with full 2018–2025 months.

    The extra 2018 year is strictly for computing valid 2019 twelve-month
    inflation rates. BLS M13 annual observations, if present, are not months.
    """
    payload = json.loads((RAW / "bls_monthly_2018_2025.json").read_text(encoding="utf-8"))
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError("BLS API source is not successful")
    series = payload.get("Results", {}).get("series", [])
    if {item.get("seriesID") for item in series} != set(BLS_IDS):
        raise ValueError("BLS series identifiers differ from schema")
    out: dict[str, dict[date, float | None]] = {}
    expected = {date(y, m, 1) for y in range(2018, 2026) for m in range(1, 13)}
    for series_item in series:
        code = series_item["seriesID"]
        observations: dict[date, float | None] = {}
        for raw in series_item["data"]:
            period = raw["period"]
            if len(period) != 3 or period[0] != "M" or not period[1:].isdigit():
                continue
            month = int(period[1:])
            if not 1 <= month <= 12:
                continue
            day = date(int(raw["year"]), month, 1)
            if day in observations:
                raise ValueError(f"Repeated BLS observation: {code} {day}")
            # BLS explicitly reports unavailable October 2025 estimates as '-'.
            # Keep the observation date, but never fabricate a numerical value.
            missing_expected = day == date(2025, 10, 1) and code != "CES0500000003"
            if raw["value"] == "-":
                if not missing_expected:
                    raise ValueError(f"Unexpected missing BLS value: {code} {day}")
                observations[day] = None
                continue
            value = number(raw["value"], f"{code} {day}")
            if value < 0 or missing_expected:
                raise ValueError(f"Negative BLS source value: {code} {day}")
            observations[day] = value
        if set(observations) != expected:
            raise ValueError(f"Invalid BLS coverage: {code}; {len(observations)} observations")
        out[code] = observations
    return out


def observed(values: Mapping[date, float | None], day: date, label: str) -> float:
    """Return a compulsory valid value; used only for fixed baselines."""
    value = values.get(day)
    if value is None:
        raise ValueError(f"Missing baseline {label} on {day}")
    return value


def average_observed(values: Sequence[object], year: int, label: str) -> float:
    """Calculate descriptive means without imputing October 2025."""
    available = [float(v) for v in values if v is not None]
    minimum = 11 if year == 2025 else 12
    if len(available) < minimum:
        raise ValueError(f"Insufficient {label} in {year}: {len(available)}")
    return fmean(available)


def read_fred(path: Path, column: str) -> dict[date, float]:
    """Read full original FRED graph CSV, preserving period dates and units."""
    result: dict[date, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "observation_date" not in reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"Missing FRED column {column}: {path}")
        for item in reader:
            day = date.fromisoformat(item["observation_date"])
            raw = item[column]
            if day in result:
                raise ValueError(f"Duplicate FRED value: {column} {day}")
            if raw is not None and raw.strip() not in {"", "."}:
                result[day] = number(raw, f"{column} {day}")
    if not result:
        raise ValueError(f"Empty series: {column}")
    return result


def read_fred_sources() -> dict[str, dict[date, float]]:
    """Enumerate and read original BLS, BEA, Fed, FHFA and Census series."""
    specs = {
        "PAYEMS": ("fred_PAYEMS.csv", "PAYEMS"),
        "GDPC1": ("fred_GDPC1.csv", "GDPC1"),
        "FEDFUNDS": ("fred_FEDFUNDS.csv", "FEDFUNDS"),
        "WALCL": ("fred_WALCL.csv", "WALCL"),
        "USSTHPI": ("fred_debt_housing.csv", "USSTHPI"),
        "MEHOINUSA672N": ("fred_income_gini.csv", "MEHOINUSA672N"),
        "GINIALLRH": ("fred_income_gini.csv", "GINIALLRH"),
    }
    return {code: read_fred(RAW / name, field) for code, (name, field) in specs.items()}


def complete_months(values: Mapping[date, float], year: int, label: str) -> list[float]:
    """Require each month's first-day observation exactly once."""
    days = [date(year, m, 1) for m in range(1, 13)]
    if any(day not in values for day in days):
        raise ValueError(f"Missing monthly values for {label} in {year}")
    return [values[day] for day in days]


def complete_quarters(values: Mapping[date, float], year: int, label: str) -> list[float]:
    """Require the four quarterly timestamps in a calendar year."""
    days = [date(year, m, 1) for m in (1, 4, 7, 10)]
    if any(day not in values for day in days):
        raise ValueError(f"Missing quarterly values for {label} in {year}")
    return [values[day] for day in days]


def one_annual(values: Mapping[date, float], year: int, label: str) -> float:
    """Get the annual Census estimate, never a fabricated monthly series."""
    day = date(year, 1, 1)
    if day not in values:
        raise ValueError(f"Missing annual {label} in {year}")
    return values[day]


def monthly_rows() -> list[dict[str, str | float | None]]:
    """Create 84 dated rows, retaining the real October 2025 data gap."""
    s = read_bls()
    p0 = observed(s["CUSR0000SA0"], date(2019, 12, 1), "CPI")
    shelter0 = observed(s["CUSR0000SAH1"], date(2019, 12, 1), "shelter")
    wage0 = observed(s["CES0500000003"], date(2019, 12, 1), "wage")
    if min(p0, shelter0, wage0) <= 0:
        raise ValueError("Invalid positive price or wage index base")
    rows: list[dict[str, str | float | None]] = []
    for day in MONTHS:
        previous = date(day.year - 1, day.month, 1)
        price_sa = s["CUSR0000SA0"][day]
        wage = s["CES0500000003"][day]
        white = s["LNS14000003"][day]
        black = s["LNS14000006"][day]
        previous_price = s["CUUR0000SA0"][previous]
        current_price = s["CUUR0000SA0"][day]
        shelter = s["CUSR0000SAH1"][day]
        rows.append(
            {
                "observation_date": day.isoformat(),
                "total_unemployment_pct": s["LNS14000000"][day],
                "white_unemployment_pct": white,
                "black_unemployment_pct": black,
                "black_white_gap_pp": black - white if black is not None and white is not None else None,
                "cpi_sa_index_dec2019": 100 * price_sa / p0 if price_sa is not None else None,
                "cpi_nsa_12m_inflation_pct": 100 * (current_price / previous_price - 1) if current_price is not None and previous_price is not None else None,
                "shelter_sa_index_dec2019": 100 * shelter / shelter0 if shelter is not None else None,
                "nominal_hourly_earnings_usd": wage,
                "nominal_earnings_index_dec2019": 100 * wage / wage0 if wage is not None else None,
                "real_earnings_index_dec2019": 100 * (wage / price_sa) / (wage0 / p0) if wage is not None and price_sa is not None else None,
            }
        )
    return rows


def build_panel() -> list[dict[str, float | int]]:
    """Annual descriptive means, not annualized sums or policy effects."""
    bls = read_bls()
    fred = read_fred_sources()
    month = {date.fromisoformat(str(r["observation_date"])): r for r in monthly_rows()}
    gdp0 = fred["GDPC1"][date(2019, 10, 1)]
    jobs0 = fred["PAYEMS"][date(2020, 2, 1)]
    home0 = fred["USSTHPI"][date(2019, 10, 1)]
    income0 = one_annual(fred["MEHOINUSA672N"], 2019, "median income")
    if min(gdp0, jobs0, home0, income0) <= 0:
        raise ValueError("Invalid index benchmark")
    panel: list[dict[str, float | int]] = []
    for year in YEARS:
        monthly_data = [month[date(year, m, 1)] for m in range(1, 13)]
        weekly = [v for day, v in fred["WALCL"].items() if day.year == year]
        if not 50 <= len(weekly) <= 54:
            raise ValueError(f"Unexpected weekly Fed coverage: {year} {len(weekly)}")
        gdp = fmean(complete_quarters(fred["GDPC1"], year, "GDPC1"))
        house = fmean(complete_quarters(fred["USSTHPI"], year, "USSTHPI"))
        income = one_annual(fred["MEHOINUSA672N"], year, "median income")
        observations: dict[str, float | int] = {
            "year": year,
            "unemployment_pct": average_observed([r["total_unemployment_pct"] for r in monthly_data], year, "unemployment"),
            "white_unemployment_pct": average_observed([r["white_unemployment_pct"] for r in monthly_data], year, "White unemployment"),
            "black_unemployment_pct": average_observed([r["black_unemployment_pct"] for r in monthly_data], year, "Black unemployment"),
            "black_white_gap_pp": average_observed([r["black_white_gap_pp"] for r in monthly_data], year, "unemployment gap"),
            "payroll_employment_millions": fmean(complete_months(fred["PAYEMS"], year, "PAYEMS")) / 1000,
            "payroll_index_feb2020": 100 * fmean(complete_months(fred["PAYEMS"], year, "PAYEMS")) / jobs0,
            "real_gdp_chained_billions": gdp,
            "real_gdp_index_2019q4": 100 * gdp / gdp0,
            "cpi_price_index_dec2019": average_observed([r["cpi_sa_index_dec2019"] for r in monthly_data], year, "price level"),
            "headline_cpi_12m_dec_pct": 100 * (
                observed(bls["CUUR0000SA0"], date(year, 12, 1), "December CPI") / observed(bls["CUUR0000SA0"], date(year - 1, 12, 1), "previous December CPI") - 1
            ),
            "nominal_hourly_earnings_usd": average_observed([r["nominal_hourly_earnings_usd"] for r in monthly_data], year, "wage"),
            "nominal_earnings_index_dec2019": average_observed([r["nominal_earnings_index_dec2019"] for r in monthly_data], year, "nominal wage index"),
            "real_earnings_index_dec2019": average_observed([r["real_earnings_index_dec2019"] for r in monthly_data], year, "real wage index"),
            "shelter_cpi_index_dec2019": average_observed([r["shelter_sa_index_dec2019"] for r in monthly_data], year, "shelter index"),
            "fhfa_house_price_index_2019q4": 100 * house / home0,
            "real_median_household_income_2025_usd": income,
            "median_income_index_2019": 100 * income / income0,
            "household_income_gini": one_annual(fred["GINIALLRH"], year, "Gini"),
            "fed_funds_rate_pct": fmean(complete_months(fred["FEDFUNDS"], year, "FEDFUNDS")),
            "fed_assets_trillions_usd": fmean(weekly) / 1_000_000,
        }
        if set(observations) != set(PANEL_FIELDS):
            raise ValueError(f"Unexpected annual schema in {year}")
        if not 0.3 < observations["household_income_gini"] < 0.7:
            raise ValueError(f"Implausible Gini for {year}")
        panel.append(observations)
    return panel


def serialize(rows: Sequence[Mapping[str, str | int | float | None]], fields: Sequence[str]) -> str:
    """Byte-stable CSV with six decimals and strict column ordering."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {k: "" if row[k] is None else str(row[k]) if k in {"observation_date", "year"} else f"{float(row[k]):.6f}" for k in fields}
        )
    return buffer.getvalue()


def prepared_outputs() -> dict[Path, str]:
    """Exact text for checked-in BLS monthly and annual research tables."""
    return {
        DATA / "monthly_panel_2019_2025.csv": serialize(monthly_rows(), MONTHLY_FIELDS),
        DATA / "annual_panel_2019_2025.csv": serialize(build_panel(), PANEL_FIELDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="audit without modifying files")
    args = parser.parse_args()
    for path, expected in prepared_outputs().items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"Prepared table drift: {path}")
            print(f"Verified {path.relative_to(ROOT)}")
        else:
            path.write_text(expected, encoding="utf-8")
            print(f"Generated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
