"""Download official WDI macro series and audit their overlap with tax-cut data.

The first version is deliberately an ingestion and missingness audit, not an
adjusted causal estimator. A null observation stays null in every output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias
from urllib.request import Request, urlopen

from .prepare import PAPER_COUNTRIES

Key: TypeAlias = tuple[str, int]
START_YEAR = 1965
END_YEAR = 2015
MAX_RESPONSE_BYTES = 8_000_000


@dataclass(frozen=True, slots=True)
class Indicator:
    """An exact World Development Indicators series and output-column name."""

    column: str
    code: str
    name: str


INDICATORS: tuple[Indicator, ...] = (
    Indicator(
        "gdp_pc_growth_pct", "NY.GDP.PCAP.KD.ZG", "GDP per capita growth (annual %)"
    ),
    Indicator(
        "inflation_cpi_pct", "FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"
    ),
    Indicator("trade_open_pct_gdp", "NE.TRD.GNFS.ZS", "Trade (% of GDP)"),
    Indicator(
        "unemployment_pct_labor",
        "SL.UEM.TOTL.ZS",
        "Unemployment, total (% of total labor force) (modeled ILO estimate)",
    ),
)
CORE_COLUMNS: tuple[str, ...] = tuple(ind.column for ind in INDICATORS[:3])
ALL_COLUMNS: tuple[str, ...] = tuple(ind.column for ind in INDICATORS)


@dataclass(frozen=True, slots=True)
class SeriesAudit:
    """Verifiable metadata for one original API response, not its raw contents."""

    column: str
    indicator_code: str
    source_url: str
    source_sha256: str
    source_bytes: int
    n_api_rows: int
    n_observed_values: int
    n_missing_values: int
    missing_by_country: dict[str, int]


@dataclass(frozen=True, slots=True)
class PanelRow:
    """A validated source-panel row with its observed policy exposure."""

    country: str
    year: int
    treated: int
    top1_share_pct: float


def indicator_url(
    indicator: Indicator,
    *,
    countries: frozenset[str] = PAPER_COUNTRIES,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> str:
    """Request WDI source 2 for an explicit ISO3 and annual date interval."""
    if (
        not countries
        or not countries.issubset(PAPER_COUNTRIES)
        or not 1900 <= start_year <= end_year <= 2100
        or indicator not in INDICATORS
    ):
        raise ValueError("Invalid WDI indicator, paper-country sample or date interval")
    country_path = ";".join(sorted(countries))
    return (
        f"https://api.worldbank.org/v2/country/{country_path}/indicator/"
        f"{indicator.code}?format=json&source=2&date={start_year}:{end_year}"
        "&per_page=10000"
    )


def parse_wdi_response(
    content: bytes,
    *,
    indicator: Indicator,
    countries: frozenset[str] = PAPER_COUNTRIES,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> tuple[dict[Key, float], int]:
    """Validate API pagination, country/year identifiers, nulls and uniqueness.

    Missing API records and explicit JSON nulls both remain absent from the
    numeric mapping; they are never interpreted as measured zeros.
    """
    if not 1900 <= start_year <= end_year <= 2100 or not countries:
        raise ValueError("Invalid observation window or countries")
    try:
        document = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("World Bank did not return valid JSON") from exc
    if (
        not isinstance(document, list)
        or len(document) != 2
        or not isinstance(document[0], dict)
        or not isinstance(document[1], list)
    ):
        raise ValueError("Unexpected World Bank API envelope or API error")
    metadata, records = document
    try:
        page = int(metadata["page"])
        pages = int(metadata["pages"])
        total = int(metadata["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("World Bank response lacks valid pagination") from exc
    if page != 1 or pages != 1 or total != len(records):
        raise ValueError("World Bank response is truncated or paginated")

    values: dict[Key, float] = {}
    encountered: set[Key] = set()
    for row_index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not isinstance(
            record.get("indicator"), dict
        ):
            raise TypeError(f"Invalid WDI observation #{row_index}")
        if record["indicator"].get("id") != indicator.code:
            raise ValueError(f"Wrong WDI indicator at observation #{row_index}")
        country = record.get("countryiso3code")
        try:
            year = int(record["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid WDI year at observation #{row_index}") from exc
        if country not in countries or not start_year <= year <= end_year:
            raise ValueError(f"Unexpected WDI country/year: {country}, {year}")
        key = (country, year)
        if key in encountered:
            raise ValueError(f"Duplicated WDI country-year for {indicator.code}: {key}")
        encountered.add(key)
        measurement = record.get("value")
        if measurement is None:
            continue
        if isinstance(measurement, bool) or not isinstance(measurement, (float, int)):
            raise TypeError(f"Non-numeric WDI value for {indicator.code} at {key}")
        value = float(measurement)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite WDI value for {indicator.code} at {key}")
        if indicator.column == "unemployment_pct_labor" and not 0 <= value <= 100:
            raise ValueError(f"Unemployment percentage outside 0–100 at {key}")
        values[key] = value
    if not values:
        raise ValueError(f"WDI returned no observed values for {indicator.code}")
    return values, len(encountered)


def download_indicator(
    indicator: Indicator,
    *,
    countries: frozenset[str] = PAPER_COUNTRIES,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    timeout_seconds: int = 45,
) -> tuple[dict[Key, float], SeriesAudit]:
    """Download bounded official JSON, hash its bytes and audit all missing years."""
    url = indicator_url(
        indicator, countries=countries, start_year=start_year, end_year=end_year
    )
    request = Request(url, headers={"User-Agent": "research-audit/1.0 WDI public data"})
    with urlopen(request, timeout=timeout_seconds) as response:
        if "html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("World Bank served HTML instead of a JSON payload")
        content = response.read(MAX_RESPONSE_BYTES + 1)
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("WDI source payload exceeds the configured size limit")
    values, n_api_rows = parse_wdi_response(
        content,
        indicator=indicator,
        countries=countries,
        start_year=start_year,
        end_year=end_year,
    )
    missing_by_country = {
        country: sum(
            (country, year) not in values for year in range(start_year, end_year + 1)
        )
        for country in sorted(countries)
    }
    return values, SeriesAudit(
        column=indicator.column,
        indicator_code=indicator.code,
        source_url=url,
        source_sha256=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
        n_api_rows=n_api_rows,
        n_observed_values=len(values),
        n_missing_values=sum(missing_by_country.values()),
        missing_by_country=missing_by_country,
    )


def read_outcome_panel(path: Path) -> list[PanelRow]:
    """Read the already-audited WID/policy panel without editing its cohort."""
    required = {"country", "year", "treated", "top1_share_pct"}
    rows: list[PanelRow] = []
    encountered: set[Key] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Outcome panel lacks {sorted(required)}")
        for index, record in enumerate(reader, start=2):
            country = (record["country"] or "").strip()
            if country not in PAPER_COUNTRIES:
                raise ValueError(f"Invalid paper-sample country on row {index}")
            try:
                year = int(record["year"])
                treated = int(record["treated"])
                share = float(record["top1_share_pct"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid outcome-panel row {index}") from exc
            if (
                not START_YEAR <= year <= END_YEAR
                or record["treated"] not in {"0", "1"}
                or not math.isfinite(share)
                or not 1 <= share <= 100
            ):
                raise ValueError(f"Invalid outcome-panel values on row {index}")
            key = (country, year)
            if key in encountered:
                raise ValueError(f"Duplicated outcome-panel key: {key}")
            encountered.add(key)
            rows.append(PanelRow(country, year, treated, share))
    if not rows:
        raise ValueError("Outcome panel is empty")
    return sorted(rows, key=lambda item: (item.country, item.year))


def coverage_audit(
    panel: list[PanelRow],
    values: dict[str, dict[Key, float]],
    *,
    lag_years: int = 5,
) -> dict[str, object]:
    """Count complete cases, event-year loss and only pre-treatment histories."""
    if lag_years < 1 or set(values) != set(ALL_COLUMNS):
        raise ValueError("Expected all four WDI series and positive lag years")
    events = [item for item in panel if item.treated == 1]

    def complete(key: Key, columns: tuple[str, ...]) -> bool:
        return all(key in values[column] for column in columns)

    complete_core = [
        row for row in panel if complete((row.country, row.year), CORE_COLUMNS)
    ]
    complete_all = [
        row for row in panel if complete((row.country, row.year), ALL_COLUMNS)
    ]
    core_events = [
        row for row in events if complete((row.country, row.year), CORE_COLUMNS)
    ]
    # Lagged covariates are used exclusively before reform year t; availability
    # in year t or later is not a prerequisite for a valid matching history.
    lagged_events = [
        row
        for row in events
        if all(
            complete((row.country, year), CORE_COLUMNS)
            for year in range(row.year - lag_years, row.year)
        )
    ]
    return {
        "status": "coverage_audit_only_no_adjusted_effect_estimated",
        "n_original_outcome_panel_rows": len(panel),
        "n_original_treated_event_rows": len(events),
        "n_joint_rows_complete_core": len(complete_core),
        "n_joint_rows_complete_all_four": len(complete_all),
        "n_original_outcome_rows_missing_core": len(panel) - len(complete_core),
        "n_original_event_rows_missing_core_at_event_year": len(events)
        - len(core_events),
        "n_original_event_rows_with_complete_core_pre_treatment_lags": len(
            lagged_events
        ),
        "pre_treatment_lag_years": lag_years,
        "coverage_by_indicator_in_outcome_panel": {
            column: sum((row.country, row.year) in values[column] for row in panel)
            for column in ALL_COLUMNS
        },
        "joint_core_rows_by_country": dict(
            sorted(Counter(row.country for row in complete_core).items())
        ),
        "unemployment_is_exploratory_only": True,
        "notes": [
            "No panel rows have been removed or outcomes imputed.",
            "Complete-case counts are not matched event counts or causal estimates.",
            "The five pre-treatment lags need not be complete for every donor country.",
        ],
    }


def write_results(
    output_dir: Path,
    values: dict[str, dict[Key, float]],
    audits: list[SeriesAudit],
    *,
    start_year: int,
    end_year: int,
    panel: list[PanelRow] | None = None,
) -> None:
    """Write complete country/year grids, provenance and optional panel overlap."""
    if set(values) != set(ALL_COLUMNS) or len(audits) != len(INDICATORS):
        raise ValueError("Expected the four explicitly documented WDI series")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["country", "year", *ALL_COLUMNS]
    with (output_dir / "world_bank_controls.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for country in sorted(PAPER_COUNTRIES):
            for year in range(start_year, end_year + 1):
                key = (country, year)
                writer.writerow(
                    {
                        "country": country,
                        "year": year,
                        **{
                            column: format(values[column][key], ".15g")
                            if key in values[column]
                            else ""
                            for column in ALL_COLUMNS
                        },
                    }
                )
    manifest = {
        "status": "official_world_bank_wdi_live_data_not_frozen_vintage",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "publisher": "World Bank World Development Indicators (source=2)",
        "country_sample": sorted(PAPER_COUNTRIES),
        "study_years": [start_year, end_year],
        "core_matching_candidates": list(CORE_COLUMNS),
        "additional_coverage_only": ["unemployment_pct_labor"],
        "source_data_are_not_imputed": True,
        "series": [asdict(audit) for audit in audits],
    }
    (output_dir / "world_bank_provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if panel is None:
        return
    with (output_dir / "outcome_panel_with_macro.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["country", "year", "treated", "top1_share_pct", *ALL_COLUMNS],
        )
        writer.writeheader()
        for row in panel:
            key = (row.country, row.year)
            writer.writerow(
                {
                    "country": row.country,
                    "year": row.year,
                    "treated": row.treated,
                    "top1_share_pct": format(row.top1_share_pct, ".15g"),
                    **{
                        column: format(values[column][key], ".15g")
                        if key in values[column]
                        else ""
                        for column in ALL_COLUMNS
                    },
                }
            )
    (output_dir / "macro_overlap_audit.json").write_text(
        json.dumps(coverage_audit(panel, values), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Fetch four official annual WDI controls and report missingness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, help="Optional existing WID/tax-cut CSV")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    arguments = parser.parse_args()
    if not START_YEAR <= arguments.start_year <= arguments.end_year <= END_YEAR:
        parser.error("Year range must fall within the 1965–2015 study period")
    values: dict[str, dict[Key, float]] = {}
    audits: list[SeriesAudit] = []
    for indicator in INDICATORS:
        observations, audit = download_indicator(
            indicator, start_year=arguments.start_year, end_year=arguments.end_year
        )
        values[indicator.column] = observations
        audits.append(audit)
        print(
            f"WDI {indicator.code}: {audit.n_observed_values} observed; "
            f"{audit.n_missing_values} missing; SHA256={audit.source_sha256}"
        )
    panel = read_outcome_panel(arguments.panel) if arguments.panel else None
    write_results(
        arguments.output_dir,
        values,
        audits,
        start_year=arguments.start_year,
        end_year=arguments.end_year,
        panel=panel,
    )
    print(f"Saved validated WDI sources and coverage to {arguments.output_dir}")
    if panel is not None:
        print("Macro panel is a missingness audit, not an adjusted causal estimate")


if __name__ == "__main__":
    main()
