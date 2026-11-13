"""Audit and join authors' tax-reform CSV with an independently sourced outcome panel.

No source CSV is silently scraped. Download it from the authors' cited URL,
inspect its headings, and specify the observed column names explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import TypeAlias

Key: TypeAlias = tuple[str, int]

PAPER_COUNTRIES: frozenset[str] = frozenset(
    {
        "AUS",
        "AUT",
        "BEL",
        "CAN",
        "DNK",
        "FIN",
        "FRA",
        "DEU",
        "IRL",
        "ITA",
        "JPN",
        "NLD",
        "NZL",
        "NOR",
        "SWE",
        "CHE",
        "GBR",
        "USA",
    }
)

# Labels are checked against a documented 18-country paper sample. This is not
# a generic fuzzy-match operation, which could silently misclassify countries.
ALIASES: dict[str, str] = {
    "AUSTRALIA": "AUS",
    "AU": "AUS",
    "AUSTRIA": "AUT",
    "AT": "AUT",
    "BELGIUM": "BEL",
    "BE": "BEL",
    "CANADA": "CAN",
    "CA": "CAN",
    "DENMARK": "DNK",
    "DK": "DNK",
    "FINLAND": "FIN",
    "FI": "FIN",
    "FRANCE": "FRA",
    "FR": "FRA",
    "GERMANY": "DEU",
    "DE": "DEU",
    "IRELAND": "IRL",
    "IE": "IRL",
    "ITALY": "ITA",
    "IT": "ITA",
    "JAPAN": "JPN",
    "JP": "JPN",
    "NETHERLANDS": "NLD",
    "THE NETHERLANDS": "NLD",
    "NL": "NLD",
    "NEW ZEALAND": "NZL",
    "NZ": "NZL",
    "NORWAY": "NOR",
    "NO": "NOR",
    "SWEDEN": "SWE",
    "SE": "SWE",
    "SWITZERLAND": "CHE",
    "CH": "CHE",
    "UNITED KINGDOM": "GBR",
    "UK": "GBR",
    "GB": "GBR",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    "US": "USA",
}


def normalize_country(raw: str) -> str | None:
    """Map known labels to ISO3 or return None for out-of-sample labels."""
    normalized = raw.strip().upper().replace(".", "")
    return normalized if normalized in PAPER_COUNTRIES else ALIASES.get(normalized)


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read a CSV after verifying its columns; retain source row numbers."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV headings in {path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path}: missing fields {sorted(missing)}, observed {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Source CSV is empty: {path}")
    return rows


def _key(
    row: dict[str, str], country_field: str, year_field: str, *, line: int
) -> Key | None:
    """Parse a country-year identifier and exclude verified out-of-sample ISO labels."""
    raw = row[country_field]
    if raw is None or not raw.strip():
        raise ValueError(f"Missing country at CSV row {line}")
    country = normalize_country(raw)
    try:
        year = int(row[year_field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid year at CSV row {line}") from exc
    if not 1900 <= year <= 2100:
        raise ValueError(f"Implausible year {year} at CSV row {line}")
    return (country, year) if country is not None else None


def prepare_panel(
    *,
    events_path: Path,
    outcomes_path: Path,
    event_mode: str,
    event_country: str,
    event_year: str,
    treatment_field: str,
    outcome_country: str,
    outcome_year: str,
    outcome_fields: tuple[str, ...],
    start_year: int = 1965,
    end_year: int = 2015,
    expected_events: int = 30,
    require_complete_event_panel: bool = True,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Build a canonical panel; never assume missing outcomes or policies are zero.

    In `positive-events` mode, absence of an event is explicitly interpreted as
    zero *only after* the authors' list has passed the expected-event count.
    In `binary-panel` mode, missing country-years are absent, not recoded.
    """
    if event_mode not in {"positive-events", "binary-panel"}:
        raise ValueError("event_mode must be positive-events or binary-panel")
    if not 1900 <= start_year <= end_year <= 2100 or expected_events < 1:
        raise ValueError("Invalid study years or expected event count")
    if not outcome_fields or len(outcome_fields) != len(set(outcome_fields)):
        raise ValueError("Provide one or more distinct outcome or covariate fields")
    if {"country", "year", "treated"}.intersection(outcome_fields):
        raise ValueError(
            "Outcome fields cannot replace canonical country, year or treated fields"
        )
    event_required = {event_country, event_year}
    if event_mode == "binary-panel":
        event_required.add(treatment_field)
    event_rows = _read_rows(events_path, event_required)
    event_map: dict[Key, int] = {}
    skipped_event_labels: Counter[str] = Counter()
    for line, row in enumerate(event_rows, start=2):
        key = _key(row, event_country, event_year, line=line)
        if key is None:
            skipped_event_labels[row[event_country].strip()] += 1
            continue
        if not start_year <= key[1] <= end_year:
            continue
        if key in event_map:
            raise ValueError(f"Duplicated tax-reform country-year: {key}")
        flag = (
            "1"
            if event_mode == "positive-events"
            else str(row[treatment_field]).strip()
        )
        if flag not in {"0", "1"}:
            raise ValueError(f"Nonbinary policy treatment at CSV row {line}: {flag!r}")
        event_map[key] = int(flag)
    covered_countries = {country for country, _ in event_map}
    missing_countries = PAPER_COUNTRIES.difference(covered_countries)
    if missing_countries:
        raise ValueError(
            f"No tax-reform data for paper-sample countries: {sorted(missing_countries)}"
        )
    event_count = sum(event_map.values())
    if event_count != expected_events:
        raise ValueError(
            f"Observed {event_count} paper-sample events, expected {expected_events}; "
            "verify source revision, treatment coding and country sample before proceeding"
        )
    expected_keys = {
        (country, year)
        for country in PAPER_COUNTRIES
        for year in range(start_year, end_year + 1)
    }
    if event_mode == "positive-events":
        # The positive-event list must be explicitly designated as exhaustive;
        # zero expansion is never applied to a partial binary panel.
        event_map = {key: event_map.get(key, 0) for key in expected_keys}
    elif require_complete_event_panel and (
        missing := expected_keys.difference(event_map)
    ):
        raise ValueError(
            f"Binary event panel is missing {len(missing)} country-years; "
            "use --allow-incomplete-events only after auditing the coverage"
        )

    outcome_rows = _read_rows(
        outcomes_path, {outcome_country, outcome_year, *outcome_fields}
    )
    observed_outcomes: dict[Key, dict[str, str]] = {}
    skipped_outcome_labels: Counter[str] = Counter()
    for line, row in enumerate(outcome_rows, start=2):
        key = _key(row, outcome_country, outcome_year, line=line)
        if key is None:
            skipped_outcome_labels[row[outcome_country].strip()] += 1
            continue
        if not start_year <= key[1] <= end_year:
            continue
        if key in observed_outcomes:
            raise ValueError(f"Duplicated outcome country-year: {key}")
        values: dict[str, str] = {}
        for field in outcome_fields:
            raw = row[field]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                # Missing measurements are excluded and audited, never set to 0.
                values = {}
                break
            if not math.isfinite(value):
                values = {}
                break
            values[field] = format(value, ".15g")
        observed_outcomes[key] = values

    joined = [
        {
            "country": country,
            "year": str(year),
            "treated": str(exposure),
            **observed_outcomes[(country, year)],
        }
        for (country, year), exposure in sorted(event_map.items())
        if observed_outcomes.get((country, year))
    ]
    if not joined:
        raise ValueError(
            "No complete country-year overlap between treatment and outcomes"
        )
    missing_outcome_keys = set(event_map).difference(
        key for key, values in observed_outcomes.items() if values
    )
    missing_event_keys = set(observed_outcomes).difference(event_map)
    audit: dict[str, object] = {
        "status": "prepared_real_data_requires_manual_source_provenance_review",
        "event_source": str(events_path),
        "outcome_source": str(outcomes_path),
        "event_mode": event_mode,
        "sample": "Hope-Limberg 2022 paper 18 OECD countries",
        "start_year": start_year,
        "end_year": end_year,
        "n_original_event_source_rows": len(event_rows),
        "n_valid_paper_sample_events": event_count,
        "n_complete_outcome_event_rows": len(joined),
        "n_lost_event_years_due_to_outcome_gaps": sum(
            event_map[key] for key in missing_outcome_keys
        ),
        "n_event_panel_keys_without_complete_outcomes": len(missing_outcome_keys),
        "n_outcome_keys_without_event_coverage": len(missing_event_keys),
        "n_event_panel_keys": len(event_map),
        "outcome_fields": list(outcome_fields),
        "skipped_event_country_labels": dict(skipped_event_labels),
        "skipped_outcome_country_labels": dict(skipped_outcome_labels),
    }
    return joined, audit


def main() -> None:
    """Convert verified author-source CSVs into the study's canonical schema."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument(
        "--event-mode", choices=("positive-events", "binary-panel"), required=True
    )
    parser.add_argument(
        "--event-country", required=True, help="Exact heading in author CSV"
    )
    parser.add_argument(
        "--event-year", required=True, help="Exact heading in author CSV"
    )
    parser.add_argument(
        "--event-treatment",
        default="treated",
        help="Exact heading for binary-panel mode",
    )
    parser.add_argument("--outcome-country", default="country")
    parser.add_argument("--outcome-year", default="year")
    parser.add_argument(
        "--fields", nargs="+", required=True, help="Outcome and covariate headings"
    )
    parser.add_argument("--start-year", type=int, default=1965)
    parser.add_argument("--end-year", type=int, default=2015)
    parser.add_argument("--expected-events", type=int, default=30)
    parser.add_argument("--allow-incomplete-events", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined, audit = prepare_panel(
        events_path=args.events,
        outcomes_path=args.outcomes,
        event_mode=args.event_mode,
        event_country=args.event_country,
        event_year=args.event_year,
        treatment_field=args.event_treatment,
        outcome_country=args.outcome_country,
        outcome_year=args.outcome_year,
        outcome_fields=tuple(args.fields),
        start_year=args.start_year,
        end_year=args.end_year,
        expected_events=args.expected_events,
        require_complete_event_panel=not args.allow_incomplete_events,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=["country", "year", "treated", *args.fields]
        )
        writer.writeheader()
        writer.writerows(joined)
    audit_file = args.output.with_suffix(".audit.json")
    audit_file.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(joined)} complete country-years; audit: {audit_file}")
    print("No imputation of missing WID/outcome values was performed")


if __name__ == "__main__":
    main()
