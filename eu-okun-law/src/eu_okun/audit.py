"""Validate a frozen Eurostat snapshot and summarize modelling readiness."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from eu_okun.contract import EU27, SERIES, EurostatDataError, quarter_sequence


def _read_manifest(directory: Path) -> dict[str, Any]:
    """Validate provenance before examining any derived observations."""
    path = directory / "snapshot_manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise EurostatDataError("Expected an object in the snapshot manifest.")
    period = document.get("period")
    if not isinstance(period, dict):
        raise EurostatDataError("Missing period definition in the snapshot manifest.")
    start, end = period.get("start"), period.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        raise EurostatDataError("Invalid snapshot period.")
    quarter_sequence(start, end)
    if document.get("countries") != list(EU27):
        raise EurostatDataError("The snapshot country cohort is not the fixed EU-27.")
    sources = document.get("series")
    if not isinstance(sources, dict) or set(sources) != {spec.name for spec in SERIES}:
        raise EurostatDataError("The snapshot manifest lacks required source series.")
    for spec in SERIES:
        entry = sources[spec.name]
        if not isinstance(entry, dict) or entry.get("dataset") != spec.dataset:
            raise EurostatDataError(f"Unexpected provenance for {spec.dataset}.")
        expected = entry.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise EurostatDataError(f"Invalid SHA-256 provenance for {spec.dataset}.")
        path = directory / "raw" / f"{spec.dataset}.json"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise EurostatDataError(f"Raw Eurostat response changed: {spec.dataset}.")
    return document


def _read_rows(directory: Path, start: str, end: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Require exactly one row per EU-27 country and requested quarter."""
    path = directory / "processed" / "eu27_quarterly.csv"
    columns = {"geo", "quarter"}
    for spec in SERIES:
        columns.update((spec.column, f"{spec.name}_flag"))
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    all_quarters = set(quarter_sequence(start, end))
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != columns:
            raise EurostatDataError("Unexpected processed CSV columns.")
        for record in reader:
            country, quarter = record.get("geo"), record.get("quarter")
            if country not in EU27 or quarter not in all_quarters:
                raise EurostatDataError(f"Unrecognized country or quarter: {country}/{quarter}.")
            assert isinstance(country, str) and isinstance(quarter, str)
            key = (country, quarter)
            if key in observed:
                raise EurostatDataError(f"Duplicate country-quarter observation: {key}.")
            cell: dict[str, Any] = {}
            for spec in SERIES:
                raw = record.get(spec.column)
                if raw is None:
                    raise EurostatDataError(f"Missing CSV column {spec.column}.")
                if raw.strip():
                    try:
                        number = float(raw)
                    except ValueError as exc:
                        raise EurostatDataError(f"Non-numeric {spec.column}: {key}.") from exc
                    if not math.isfinite(number):
                        raise EurostatDataError(f"Non-finite {spec.column}: {key}.")
                    if spec.name != "gdp" and not 0 <= number <= 100:
                        raise EurostatDataError(f"Impossible percentage rate: {key}.")
                    cell[spec.name] = number
                else:
                    cell[spec.name] = None
                status = record.get(f"{spec.name}_flag")
                if status is None:
                    raise EurostatDataError(f"Missing quality status for {key}.")
                cell[f"{spec.name}_flag"] = status
            observed[key] = cell
    expected = len(EU27) * len(all_quarters)
    if len(observed) != expected:
        raise EurostatDataError(
            f"Country-quarter grid incomplete: found {len(observed)}, expected {expected}."
        )
    return observed


def audit_snapshot(directory: Path) -> dict[str, Any]:
    """Audit coverage, quality flags and eligible descriptive-regression rows.

    Counts below are diagnostics, not fitted model estimates. Lags use the
    preceding *calendar* quarter, never the previous nonmissing observation.
    """
    manifest = _read_manifest(directory)
    start = str(manifest["period"]["start"])
    end = str(manifest["period"]["end"])
    analysis_start = max(start, "2009-Q1")
    if analysis_start > end:
        raise EurostatDataError("Snapshot ends before the 2009 analysis period.")
    quarters = quarter_sequence(analysis_start, end)
    records = _read_rows(directory, start, end)
    country_reports: list[dict[str, Any]] = []
    all_flags: dict[str, Counter[str]] = {spec.name: Counter() for spec in SERIES}

    for country in EU27:
        series_coverage: dict[str, dict[str, Any]] = {}
        for spec in SERIES:
            present: list[str] = []
            statuses: Counter[str] = Counter()
            for quarter in quarters:
                row = records[(country, quarter)]
                if row[spec.name] is not None:
                    present.append(quarter)
                flag = row[f"{spec.name}_flag"]
                if flag:
                    statuses[flag] += 1
                    all_flags[spec.name][flag] += 1
            series_coverage[spec.name] = {
                "available": len(present),
                "missing": len(quarters) - len(present),
                "first": present[0] if present else None,
                "last": present[-1] if present else None,
                "flags": dict(sorted(statuses.items())),
            }

        complete = 0
        okun_pairs = 0
        participation_pairs = 0
        for quarter in quarters:
            row = records[(country, quarter)]
            current_complete = all(row[spec.name] is not None for spec in SERIES)
            complete += int(current_complete)
            # A lag must correspond to an actual adjacent quarter.
            previous_code = _previous_quarter(quarter)
            previous = records.get((country, previous_code))
            basic = (
                row["gdp"] is not None and row["unemployment"] is not None
                and previous is not None and previous["unemployment"] is not None
            )
            okun_pairs += int(basic)
            participation_pairs += int(
                basic and row["participation"] is not None and previous is not None
                and previous["participation"] is not None
            )
        country_reports.append({
            "geo": country,
            "series": series_coverage,
            "joint_complete_quarters": complete,
            "basic_okun_pairs": okun_pairs,
            "with_participation_pairs": participation_pairs,
        })

    return {
        "schema_version": 1,
        "analysis_period": {"start": analysis_start, "end": end},
        "quarters": len(quarters),
        "countries": len(EU27),
        "country_quarter_rows": len(records),
        "snapshot_manifest_sha256": hashlib.sha256(
            (directory / "snapshot_manifest.json").read_bytes()
        ).hexdigest(),
        "processed_csv_sha256": hashlib.sha256(
            (directory / "processed" / "eu27_quarterly.csv").read_bytes()
        ).hexdigest(),
        "quality_flags": {
            key: dict(sorted(value.items())) for key, value in all_flags.items()
        },
        "by_country": country_reports,
    }


def _previous_quarter(quarter: str) -> str:
    """Return the adjacent previous quarter, including year transitions."""
    year = int(quarter[:4])
    number = int(quarter[-1])
    return f"{year - 1}-Q4" if number == 1 else f"{year}-Q{number - 1}"
