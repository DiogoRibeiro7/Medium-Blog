"""Ingest the World Bank's official WDI CSV archives and audit tax-cut coverage.

This module reports observable macroeconomic coverage only; it does not change
the study's existing event cohort, estimator or claims about causal effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypeAlias
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prepare import PAPER_COUNTRIES

Key: TypeAlias = tuple[str, int]
START_YEAR = 1965
END_YEAR = 2015
MAX_ARCHIVE_BYTES = 5_000_000
MAX_DATA_MEMBER_BYTES = 3_000_000


@dataclass(frozen=True, slots=True)
class Indicator:
    """A documented WDI series and stable local column name."""

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
CORE_COLUMNS: tuple[str, ...] = tuple(indicator.column for indicator in INDICATORS[:3])
ALL_COLUMNS: tuple[str, ...] = tuple(indicator.column for indicator in INDICATORS)


@dataclass(frozen=True, slots=True)
class SeriesAudit:
    """Provenance and coverage from one unmodified official WDI ZIP archive."""

    column: str
    indicator_code: str
    source_url: str
    source_sha256: str
    source_bytes: int
    source_csv_member: str
    source_last_updated: str
    n_paper_country_rows: int
    n_observed_values: int
    n_missing_values: int
    missing_by_country: dict[str, int]


@dataclass(frozen=True, slots=True)
class PanelRow:
    """One previously audited country-year, policy and income-share observation."""

    country: str
    year: int
    treated: int
    top1_share_pct: float


def archive_url(indicator: Indicator) -> str:
    """Use the World Bank's documented, stable ZIP/CSV bulk export endpoint."""
    if indicator not in INDICATORS:
        raise ValueError("Unsupported WDI indicator")
    return (
        f"https://api.worldbank.org/v2/en/indicator/{indicator.code}?downloadformat=csv"
    )


def parse_wdi_archive(
    content: bytes,
    *,
    indicator: Indicator,
    countries: frozenset[str] = PAPER_COUNTRIES,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> tuple[dict[Key, float], str, str, int]:
    """Validate the original ZIP/CSV and select only the declared study years.

    The archive contains country and indicator metadata CSVs alongside one
    source-2 measurement CSV. Never extract archive paths to the filesystem.
    Empty source cells are omitted; numeric zeros are genuine measurements.
    """
    if (
        indicator not in INDICATORS
        or not countries
        or not countries.issubset(PAPER_COUNTRIES)
        or not 1900 <= start_year <= end_year <= 2100
    ):
        raise ValueError("Invalid WDI indicator, countries or study interval")
    if not 0 < len(content) <= MAX_ARCHIVE_BYTES:
        raise ValueError("WDI source archive has an invalid size")
    try:
        with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
            matching = [
                member
                for member in archive.infolist()
                if member.filename.startswith(f"API_{indicator.code}_DS2_en_csv_")
                and member.filename.endswith(".csv")
                and not member.is_dir()
            ]
            if len(matching) != 1:
                raise ValueError("WDI ZIP lacks one unambiguous source-2 CSV")
            member = matching[0]
            if member.file_size > MAX_DATA_MEMBER_BYTES:
                raise ValueError("WDI uncompressed CSV exceeds the size limit")
            # zipfile checks CRC as it reads; no path is extracted or trusted.
            raw_csv = archive.read(member)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as exc:
        raise ValueError("Invalid or corrupt official WDI ZIP archive") from exc
    try:
        source = io.StringIO(raw_csv.decode("utf-8-sig"), newline="")
    except UnicodeDecodeError as exc:
        raise ValueError("WDI CSV is not valid UTF-8") from exc
    rows = csv.reader(source)
    try:
        source_line = next(rows)
        blank_one = next(rows)
        revision_line = next(rows)
        blank_two = next(rows)
        headings = next(rows)
    except StopIteration as exc:
        raise ValueError("WDI CSV is missing its published header") from exc
    if (
        source_line[:2] != ["Data Source", "World Development Indicators"]
        or blank_one
        or revision_line[:1] != ["Last Updated Date"]
        or len(revision_line) < 2
        or blank_two
        or headings[:4]
        != ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    ):
        raise ValueError("Unexpected WDI data-source metadata or CSV columns")
    last_updated = revision_line[1]
    try:
        date.fromisoformat(last_updated)
    except ValueError as exc:
        raise ValueError("Invalid WDI source last-updated date") from exc
    years = [str(year) for year in range(start_year, end_year + 1)]
    if len(headings) != len(set(headings)) or not set(years).issubset(headings):
        raise ValueError("WDI source CSV does not cover the requested annual window")

    values: dict[Key, float] = {}
    observed_countries: set[str] = set()
    for row_index, row in enumerate(
        csv.DictReader(source, fieldnames=headings), start=6
    ):
        country = row.get("Country Code")
        if country not in countries:
            continue  # Other World Bank countries and aggregates are not the paper sample.
        if country in observed_countries:
            raise ValueError(f"Duplicated WDI country row: {country}")
        observed_countries.add(country)
        if row.get("Indicator Code") != indicator.code:
            raise ValueError(f"Wrong WDI series for {country} on CSV row {row_index}")
        for year in range(start_year, end_year + 1):
            key = (country, year)
            original = row.get(str(year))
            if original is None:
                raise ValueError(f"Missing WDI year column at {key}")
            if not original.strip():
                continue
            try:
                value = float(original)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric WDI value at {key}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite WDI value at {key}")
            if indicator.column == "unemployment_pct_labor" and not 0 <= value <= 100:
                raise ValueError(f"Invalid unemployment percentage at {key}")
            values[key] = value
    if observed_countries != countries:
        raise ValueError(
            f"WDI source lacks paper countries: {sorted(countries - observed_countries)}"
        )
    if not values:
        raise ValueError(f"No observed WDI values for {indicator.code}")
    return values, member.filename, last_updated, len(observed_countries)


def download_indicator(
    indicator: Indicator,
    *,
    countries: frozenset[str] = PAPER_COUNTRIES,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    timeout_seconds: int = 30,
    attempts: int = 3,
) -> tuple[dict[Key, float], SeriesAudit]:
    """Fetch a bounded WDI ZIP with retries and a SHA-256 of its exact bytes."""
    if attempts < 1 or timeout_seconds < 1:
        raise ValueError("Retry count and timeout must be positive")
    url = archive_url(indicator)
    request = Request(url, headers={"User-Agent": "research-audit/1.0 WDI public data"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                if "html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("WDI endpoint served HTML instead of CSV ZIP")
                content = response.read(MAX_ARCHIVE_BYTES + 1)
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {429, 502, 503, 504}:
                raise
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"WDI {indicator.code}: exhausted {attempts} official ZIP requests"
                ) from exc
            time.sleep(min(2**attempt, 4))
    values, member, last_updated, n_country_rows = parse_wdi_archive(
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
        source_csv_member=member,
        source_last_updated=last_updated,
        n_paper_country_rows=n_country_rows,
        n_observed_values=len(values),
        n_missing_values=sum(missing_by_country.values()),
        missing_by_country=missing_by_country,
    )


def read_outcome_panel(path: Path) -> list[PanelRow]:
    """Read the existing WID/policy panel without editing its cohort."""
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
    """Count complete cases, event-year loss and pre-treatment macro histories."""
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
    # Matching uses only years t-1 through t-5, not the event year or future.
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
    """Persist a complete grid, original archive provenance and panel overlap."""
    if (
        set(values) != set(ALL_COLUMNS)
        or {audit.column for audit in audits} != set(ALL_COLUMNS)
        or len(audits) != len(INDICATORS)
    ):
        raise ValueError("Expected one audited WDI ZIP per declared indicator")
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
    manifest: dict[str, object] = {
        "status": "official_world_bank_wdi_live_data_not_frozen_vintage",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "publisher": "World Bank World Development Indicators (DS2)",
        "country_sample": sorted(PAPER_COUNTRIES),
        "study_years": [start_year, end_year],
        "core_matching_candidates": list(CORE_COLUMNS),
        "additional_coverage_only": ["unemployment_pct_labor"],
        "source_data_are_not_imputed": True,
        "series": [asdict(audit) for audit in sorted(audits, key=lambda a: a.column)],
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
    """Download four independent official WDI ZIP archives and audit coverage."""
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
    # Four source-independent downloads take less time than the unstable
    # multi-country JSON endpoint and retain exact original archive checksums.
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {
            executor.submit(
                download_indicator,
                indicator,
                start_year=arguments.start_year,
                end_year=arguments.end_year,
            ): indicator
            for indicator in INDICATORS
        }
        for task in as_completed(jobs):
            indicator = jobs[task]
            observations, audit = task.result()
            values[indicator.column] = observations
            audits.append(audit)
            print(
                f"WDI {indicator.code}: {audit.n_observed_values} observed; "
                f"{audit.n_missing_values} missing; SHA256={audit.source_sha256}",
                flush=True,
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
