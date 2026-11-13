"""Reproduce a descriptive evidence audit of wages, housing and distribution.

All observations come from frozen, source-attributed public research snapshots.
This module estimates no policy treatment effect. It requires Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
COUNTRIES: tuple[str, ...] = ("PT", "ES", "FR", "DE", "EU27_2020")
YEARS: range = range(2015, 2026)
LABOUR_VARIANTS: tuple[str, ...] = ("raw", "employment", "hours", "mixed_all")


class DataIntegrityError(ValueError):
    """Raised when a source snapshot fails the declared measurement contract."""


@dataclass(frozen=True)
class HousingObservation:
    """One country-year, using the annual mean of quarterly national HPI."""

    nominal_hpi: float
    standardised_price_income: float
    nominal_status: str
    income_status: str


@dataclass(frozen=True)
class SalaryObservation:
    """Eurostat `nama_10_fte` nominal annual salary per full-time equivalent."""

    euros_per_year: float
    status: str


def _rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read non-empty rows, rejecting schema drift and incomplete CSV records."""

    if not path.is_file():
        raise DataIntegrityError(f"Missing snapshot: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise DataIntegrityError(f"Unexpected columns in {path.name}")
        observations = list(reader)
    if not observations or any(None in record for record in observations):
        raise DataIntegrityError(f"Empty or malformed records in {path.name}")
    return observations


def _finite(value: str | None, *, column: str, positive: bool = False) -> float:
    """Convert a numeric field with an explicit finite-value constraint."""

    try:
        result = float(value) if value is not None else float("nan")
    except ValueError as exc:
        raise DataIntegrityError(f"Invalid {column}: {value!r}") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise DataIntegrityError(f"Invalid {column}: {value!r}")
    return result


def _year(raw: str | None) -> int:
    """Parse and validate a four-digit calendar year."""

    try:
        year = int(raw) if raw is not None else -1
    except ValueError as exc:
        raise DataIntegrityError(f"Invalid year: {raw!r}") from exc
    if not 1900 <= year <= 2100:
        raise DataIntegrityError(f"Out-of-range year: {year}")
    return year


def verify_snapshots(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Validate byte-for-byte SHA-256 digests and expected row counts."""

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("snapshots"), dict
    ):
        raise DataIntegrityError("Invalid manifest structure")
    for name, detail in manifest["snapshots"].items():
        path = data_dir / name
        if path.name != name or not path.is_file():
            raise DataIntegrityError(f"Missing or unsafe manifest filename: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != detail.get("sha256"):
            raise DataIntegrityError(f"Snapshot hash changed: {name}")
        if len(_rows(path, set())) != detail.get("rows"):
            raise DataIntegrityError(f"Snapshot row count changed: {name}")
    return manifest


def read_housing(
    data_dir: Path = DATA_DIR,
) -> dict[tuple[str, int], HousingObservation]:
    """Read the four-country and overlapping EU27 annual housing panel."""

    required = {
        "geo",
        "year",
        "nominal_hpi_quarter_mean",
        "standardised_price_income_ratio",
        "nominal_quarters_available",
        "nominal_status_flags",
        "price_income_status",
    }
    records: dict[tuple[str, int], HousingObservation] = {}
    for row in _rows(data_dir / "housing_indices_2015_2025.csv", required):
        key = (row["geo"], _year(row["year"]))
        if key in records or key[0] not in COUNTRIES or key[1] not in YEARS:
            raise DataIntegrityError(f"Duplicate or unexpected housing key: {key}")
        if row["nominal_quarters_available"] != "4":
            raise DataIntegrityError(f"Missing HPI quarter for {key}")
        records[key] = HousingObservation(
            nominal_hpi=_finite(
                row["nominal_hpi_quarter_mean"], column="HPI", positive=True
            ),
            standardised_price_income=_finite(
                row["standardised_price_income_ratio"],
                column="price/income",
                positive=True,
            ),
            nominal_status=row["nominal_status_flags"],
            income_status=row["price_income_status"],
        )
    _check_coverage(records, "housing")
    return records


def read_salaries(
    data_dir: Path = DATA_DIR,
) -> dict[tuple[str, int], SalaryObservation]:
    """Read nominal annual average employee salaries, not medians or household income."""

    records: dict[tuple[str, int], SalaryObservation] = {}
    for row in _rows(
        data_dir / "fte_salary_2015_2025.csv",
        {"dataset", "freq", "unit", "geo", "time", "value", "flag"},
    ):
        key = (row["geo"], _year(row["time"]))
        if (
            key in records
            or key[0] not in COUNTRIES
            or key[1] not in YEARS
            or (row["dataset"], row["freq"], row["unit"]) != ("nama_10_fte", "A", "EUR")
        ):
            raise DataIntegrityError(f"Duplicate or unexpected salary key: {key}")
        records[key] = SalaryObservation(
            euros_per_year=_finite(row["value"], column="FTE salary", positive=True),
            status=row["flag"],
        )
    _check_coverage(records, "salary")
    return records


def _check_coverage(records: dict[tuple[str, int], object], series_name: str) -> None:
    """Require a balanced 2015–2025 country-by-year panel."""

    expected = {(geo, year) for geo in COUNTRIES for year in YEARS}
    if set(records) != expected:
        raise DataIntegrityError(
            f"Incomplete {series_name} panel: missing={sorted(expected - set(records))}, "
            f"unexpected={sorted(set(records) - expected)}"
        )


def read_labour_shares(data_dir: Path = DATA_DIR) -> dict[tuple[int, str, str], float]:
    """Preserve all 18 original definitions; different definitions can reverse a trend."""

    result: dict[tuple[int, str, str], float] = {}
    for row in _rows(
        data_dir / "portugal_labour_shares_1995_2024.csv",
        {"year", "numerator", "denominator", "share"},
    ):
        key = (_year(row["year"]), row["numerator"], row["denominator"])
        if key in result or not 1995 <= key[0] <= 2024:
            raise DataIntegrityError(f"Duplicate or unexpected labour share: {key}")
        value = _finite(row["share"], column="labour share", positive=True)
        if value > 100:
            raise DataIntegrityError(f"Labour share exceeds 100%: {key}")
        result[key] = value
    numerators = {
        "raw",
        "employment",
        "hours",
        "mixed_all",
        "mixed_two_thirds",
        "mixed_rest",
    }
    denominators = {"market_prices", "factor_cost", "factor_cost_ex_government"}
    expected = {
        (year, num, den)
        for year in range(1995, 2025)
        for num in numerators
        for den in denominators
    }
    if set(result) != expected:
        raise DataIntegrityError(
            "Incomplete 18-definition Portuguese labour-share panel"
        )
    return result


def _read_us(
    path: Path, fields: tuple[str, ...], years: range
) -> dict[int, tuple[float, ...]]:
    """Read a complete annual U.S. historical series without interpolating values."""

    result: dict[int, tuple[float, ...]] = {}
    for row in _rows(path, {"year", *fields}):
        year = _year(row["year"])
        if year in result or year not in years:
            raise DataIntegrityError(f"Duplicate or unexpected U.S. year: {year}")
        result[year] = tuple(_finite(row[col], column=col) for col in fields)
    if set(result) != set(years):
        raise DataIntegrityError(f"Incomplete U.S. history in {path.name}")
    return result


def read_us_history(
    data_dir: Path = DATA_DIR,
) -> dict[str, dict[int, tuple[float, ...]]]:
    """Read U.S. top statutory rate, survey Gini and aggregate GDP/investment."""

    return {
        "tax": _read_us(
            data_dir / "us_top_income_tax_rate_1964_2018.csv",
            ("top_rate_pct",),
            range(1964, 2019),
        ),
        "gini": _read_us(
            data_dir / "us_household_gini_1967_2018.csv",
            ("household_money_income_gini",),
            range(1967, 2019),
        ),
        "macro": _read_us(
            data_dir / "us_growth_investment_1964_2018.csv",
            ("real_gdp_growth_pct", "private_investment_share_pct"),
            range(1964, 2019),
        ),
    }


def _pct_change(last: float, first: float) -> float:
    """Compute endpoint percentage change for strictly positive index levels."""

    if first <= 0 or last <= 0:
        raise DataIntegrityError(
            "Growth indices require strictly positive observations"
        )
    return 100.0 * (last / first - 1.0)


def build_analysis(
    data_dir: Path = DATA_DIR,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Compute the complete descriptives with no regression or treatment attribution."""

    verify_snapshots(data_dir)
    housing, salaries = read_housing(data_dir), read_salaries(data_dir)
    labour = read_labour_shares(data_dir)
    us = read_us_history(data_dir)
    panel: list[dict[str, str]] = []
    endpoints: dict[str, dict[str, float | str]] = {}

    for geo in COUNTRIES:
        base_hpi, base_sal = (
            housing[(geo, 2015)].nominal_hpi,
            salaries[(geo, 2015)].euros_per_year,
        )
        base_pi = housing[(geo, 2015)].standardised_price_income
        for year in YEARS:
            h, s = housing[(geo, year)], salaries[(geo, year)]
            price_index = h.nominal_hpi / base_hpi * 100.0
            salary_index = s.euros_per_year / base_sal * 100.0
            ratio_index = 100.0 * price_index / salary_index
            official_index = 100.0 * h.standardised_price_income / base_pi
            panel.append(
                {
                    "geo": geo,
                    "year": str(year),
                    "house_price_index_2015_100": f"{price_index:.6f}",
                    "nominal_fte_salary_index_2015_100": f"{salary_index:.6f}",
                    "house_price_to_fte_salary_index_2015_100": f"{ratio_index:.6f}",
                    "official_price_income_index_2015_100": f"{official_index:.6f}",
                    "nominal_hpi_status": h.nominal_status,
                    "official_price_income_status": h.income_status,
                    "fte_salary_status": s.status,
                }
            )
        end_h, end_s = housing[(geo, 2025)], salaries[(geo, 2025)]
        endpoints[geo] = {
            "nominal_house_price_growth_pct": round(
                _pct_change(end_h.nominal_hpi, base_hpi), 3
            ),
            "nominal_fte_salary_growth_pct": round(
                _pct_change(end_s.euros_per_year, base_sal), 3
            ),
            "house_price_to_fte_salary_growth_pct": round(
                _pct_change(
                    end_h.nominal_hpi / end_s.euros_per_year, base_hpi / base_sal
                ),
                3,
            ),
            "official_price_income_growth_pct": round(
                _pct_change(end_h.standardised_price_income, base_pi), 3
            ),
            "hpi_2025_status": end_h.nominal_status,
            "salary_2025_status": end_s.status,
        }

    shares: dict[str, dict[str, float]] = {}
    for name in LABOUR_VARIANTS:
        old, current = (
            labour[(1995, name, "market_prices")],
            labour[(2024, name, "market_prices")],
        )
        shares[name] = {
            "level_1995_pct": round(old, 3),
            "level_2024_pct": round(current, 3),
            "change_1995_2024_percentage_points": round(current - old, 3),
        }

    summary: dict[str, Any] = {
        "status": "descriptive_only_not_causal",
        "housing_wage": {
            "period": "2015-2025",
            "geographies": list(COUNTRIES),
            "aggregate_warning": "EU27 overlaps the member states; not an independent treatment unit.",
            "denominator_warning": "FTE salary is a national average, before tax; it is not the median wage or household disposable income.",
            "price_warning": "HPI is a national property transaction price index; it is not the price or mortgage burden of a specific home.",
            "endpoints": endpoints,
        },
        "portugal_labour_share": {
            "period": "1995-2024",
            "denominator": "GDP at market prices",
            "self_employment_treatments": shares,
            "inference_warning": "Do not infer a policy effect or a universally declining labour share from definition-sensitive endpoints.",
        },
        "us_context": {
            "period": "1967-2018 for Gini; 1964-2018 for tax and macro",
            "top_statutory_tax_rate_pct_1980": us["tax"][1980][0],
            "top_statutory_tax_rate_pct_1988": us["tax"][1988][0],
            "top_statutory_tax_rate_pct_2018": us["tax"][2018][0],
            "household_money_income_gini_1967": us["gini"][1967][0],
            "household_money_income_gini_1992": us["gini"][1992][0],
            "household_money_income_gini_1993": us["gini"][1993][0],
            "household_money_income_gini_2018": us["gini"][2018][0],
            "method_break": "1993 Current Population Survey redesign: Gini levels across the break are not strictly comparable.",
            "causal_warning": "Tax rate is statutory, not effective; co-movement with Gini or GDP cannot identify tax-reform effects.",
        },
    }
    return panel, summary


PANEL_COLUMNS = [
    "geo",
    "year",
    "house_price_index_2015_100",
    "nominal_fte_salary_index_2015_100",
    "house_price_to_fte_salary_index_2015_100",
    "official_price_income_index_2015_100",
    "nominal_hpi_status",
    "official_price_income_status",
    "fte_salary_status",
]


def serialize_panel(panel: Iterable[dict[str, str]]) -> str:
    """Build a deterministic CSV snapshot with a stable column order."""

    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PANEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(panel)
    return output.getvalue()


def write_or_check_results(
    *, check: bool = False, output_dir: Path = RESULTS_DIR, data_dir: Path = DATA_DIR
) -> dict[str, Any]:
    """Generate snapshots or compare them byte-for-byte with the committed results."""

    panel, summary = build_analysis(data_dir)
    files = {
        "housing_wage_panel.csv": serialize_panel(panel),
        "summary.json": json.dumps(
            summary, indent=2, ensure_ascii=False, sort_keys=True
        )
        + "\n",
    }
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        destination = output_dir / name
        if check:
            if (
                not destination.is_file()
                or destination.read_text(encoding="utf-8") != content
            ):
                raise DataIntegrityError(f"Reproduction mismatch: {destination}")
        else:
            destination.write_text(content, encoding="utf-8", newline="")
    return summary


def main() -> None:
    """Command-line entry point for the offline, deterministic audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify committed outputs without editing"
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR, help="results directory"
    )
    args = parser.parse_args()
    report = write_or_check_results(check=args.check, output_dir=args.output)
    print(
        json.dumps(
            {
                "status": "checked" if args.check else "written",
                "countries": len(report["housing_wage"]["endpoints"]),
                "destination": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
