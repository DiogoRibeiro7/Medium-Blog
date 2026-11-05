"""Prepare a conservative ISO3 review ledger from Forbes country aggregates.

Run only after ``forbes_primary.py`` has successfully retrieved and audited
the annual person list. This module never downloads or retains person-level
records. An exact label match to the *secondary* transcription is only a
candidate mapping, not independently verified primary-source attribution.

Example::

    python forbes_reconciliation.py \
      --primary-dir results/forbes_primary_2026 \
      --secondary data/forbes_2026_citizenship_secondary.csv \
      --out-dir results/forbes_primary_2026/reconciliation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ISO3 = re.compile(r"[A-Z]{3}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
PRIMARY_FILE = "counts_citizenship.csv"


@dataclass(frozen=True)
class CountryCount:
    """One observed aggregate or secondary candidate, never an individual."""

    country: str
    count: int
    iso3: str | None = None


def _rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read a UTF-8 CSV and enforce the expected column contract."""
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError(f"Required aggregate CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing columns in {path.name}: {sorted(required)}")
        records = list(reader)
    if not records or any(None in record for record in records):
        raise ValueError(f"Empty or malformed aggregate CSV: {path}")
    return records


def _count(value: str, *, field: str, positive: bool = False) -> int:
    """Reject fractional, negative, blank and nonnumeric count strings."""
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)", value.strip()):
        raise ValueError(f"{field} must be a nonnegative integer")
    number = int(value)
    if positive and number == 0:
        raise ValueError(f"{field} must be positive for an observed aggregate")
    return number


def _secondary(path: Path) -> dict[str, CountryCount]:
    """Load existing *unverified* citizenship-to-ISO3 candidates by exact label."""
    required = {"country", "iso3", "billionaires", "billionaire_method", "forbes_year"}
    result: dict[str, CountryCount] = {}
    seen_iso3: set[str] = set()
    for record in _rows(path, required):
        label = record["country"].strip()
        code = record["iso3"].strip()
        if not label or not ISO3.fullmatch(code):
            raise ValueError("Secondary label or ISO3 is invalid")
        if label in result or code in seen_iso3:
            raise ValueError("Duplicate secondary country label or ISO3")
        if record["billionaire_method"] != "citizenship" or record["forbes_year"] != "2026":
            raise ValueError("Secondary source is not Forbes 2026 citizenship")
        result[label] = CountryCount(
            country=label,
            iso3=code,
            count=_count(record["billionaires"], field="secondary billionaires"),
        )
        seen_iso3.add(code)
    return result


def _primary(path: Path) -> dict[str, CountryCount]:
    """Read only explicitly citizenship-attributed *primary* aggregates."""
    required = {"country_label", "billionaires", "attribution", "source_field"}
    result: dict[str, CountryCount] = {}
    for record in _rows(path, required):
        label = record["country_label"].strip()
        if not label or label in result:
            raise ValueError("Blank or duplicate primary country label")
        if (
            record["attribution"] != "citizenship"
            or record["source_field"] != "countryOfCitizenship"
        ):
            raise ValueError("Primary geography must be countryOfCitizenship, not API country")
        result[label] = CountryCount(
            country=label,
            count=_count(record["billionaires"], field="primary billionaires", positive=True),
        )
    return result


def _provenance(audit_path: Path, aggregate_path: Path, *, expected_people: int) -> dict[str, Any]:
    """Bind aggregate CSV bytes to an actual non-fixture Forbes source audit."""
    if (
        not isinstance(expected_people, int)
        or isinstance(expected_people, bool)
        or expected_people < 1
    ):
        raise ValueError("Expected annual person count must be a positive integer")
    try:
        audit: Any = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict):
            raise ValueError("Audit JSON must contain an object")
        schema = audit["schema"]
        coverage = audit["coverage"]["citizenship"]
        mode = audit["source_retrieval"]["mode"]
        hash_record = audit["aggregated_output_sha256"][PRIMARY_FILE]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Incomplete or unreadable Forbes source audit") from error
    if mode not in ("single_response", "derived_from_paginated_responses"):
        raise ValueError("Source audit is a fixture or an unrecognized retrieval mode")
    if (
        schema.get("n_persons") != expected_people
        or schema.get("expected_count") != expected_people
    ):
        raise ValueError("Forbes source size is not the expected annual count")
    if not isinstance(hash_record, str) or not SHA256.fullmatch(hash_record):
        raise ValueError("Invalid aggregate provenance digest")
    if hashlib.sha256(aggregate_path.read_bytes()).hexdigest() != hash_record:
        raise ValueError("Primary aggregate changed since source audit")
    if (
        not isinstance(audit.get("source_sha256"), str)
        or not SHA256.fullmatch(audit["source_sha256"])
    ):
        raise ValueError("Invalid Forbes source digest")
    if coverage.get("field") != "countryOfCitizenship":
        raise ValueError("Audit geography is not citizenship")
    assigned, missing = coverage.get("assigned"), coverage.get("missing")
    if (
        not isinstance(assigned, int) or isinstance(assigned, bool)
        or not isinstance(missing, int) or isinstance(missing, bool)
        or assigned < 0 or missing < 0 or assigned + missing != expected_people
    ):
        raise ValueError("Citizenship coverage does not sum to annual persons")
    return audit


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a stable header even when no country falls into a category."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reconcile(
    primary_dir: Path,
    secondary_csv: Path,
    output_dir: Path,
    *,
    expected_people: int = 3428,
) -> dict[str, Any]:
    """Produce exact-label ISO3 *candidates* and explicit unmatched ledgers.

    No fuzzy matching, missing-as-zero conversion or country-of-residence
    inference is performed. Never treat these candidates as verified mappings
    or use this output alone to authorize an empirical regression.
    """
    if not all(isinstance(p, Path) for p in (primary_dir, secondary_csv, output_dir)):
        raise TypeError("Paths must be pathlib.Path instances")
    aggregate_path = primary_dir / PRIMARY_FILE
    source = _provenance(
        primary_dir / "audit.json", aggregate_path, expected_people=expected_people
    )
    primary = _primary(aggregate_path)
    candidates = _secondary(secondary_csv)
    coverage: Mapping[str, Any] = source["coverage"]["citizenship"]
    if sum(item.count for item in primary.values()) != coverage["assigned"]:
        raise ValueError("Primary country counts disagree with audited citizenship coverage")
    if len(primary) != coverage["distinct_labels"]:
        raise ValueError("Primary country labels disagree with audited coverage")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evidence: {output_dir}")

    matched: list[dict[str, Any]] = []
    unmatched_primary: list[dict[str, Any]] = []
    for label, observed in sorted(primary.items()):
        candidate = candidates.get(label)
        if candidate is None:
            unmatched_primary.append({
                "primary_label": label,
                "primary_count": observed.count,
                "reason": "no_exact_secondary_label; manual_geography_review_required",
            })
        else:
            matched.append({
                "primary_label": label,
                "iso3_candidate": candidate.iso3,
                "primary_count": observed.count,
                "secondary_count": candidate.count,
                "difference": observed.count - candidate.count,
                "review_status": "unverified_secondary_candidate",
            })
    unmatched_secondary = [
        {"secondary_label": label, "iso3_candidate": item.iso3, "secondary_count": item.count,
         "reason": "absent_as_exact_primary_label; not_a_verified_zero"}
        for label, item in sorted(candidates.items()) if label not in primary
    ]
    summary: dict[str, Any] = {
        "method": "citizenship_only",
        "source_person_count": expected_people,
        "source_sha256": source["source_sha256"],
        "source_retrieval_mode": source["source_retrieval"]["mode"],
        "primary_aggregate_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
        "secondary_transcription_sha256": hashlib.sha256(secondary_csv.read_bytes()).hexdigest(),
        "n_primary_country_labels": len(primary),
        "n_exact_label_candidates": len(matched),
        "n_unmapped_primary_labels": len(unmatched_primary),
        "n_secondary_labels_absent_from_primary": len(unmatched_secondary),
        "primary_people_with_exact_label_candidate": sum(r["primary_count"] for r in matched),
        "primary_people_with_unresolved_label": sum(r["primary_count"] for r in unmatched_primary),
        "citizenship_people_missing_from_source": coverage["missing"],
        "iso3_mappings_independently_verified": False,
        "historical_2026_vintage_verified": False,
        "residence_attribution_verified": False,
        "approved_for_publication_or_regression": False,
        "interpretation": (
            "Exact spelling matches to an incomplete secondary citizenship table are "
            "candidate ISO3 mappings only. Unmatched labels are unknown, not zero."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "exact_label_candidates.csv",
        [
            "primary_label", "iso3_candidate", "primary_count",
            "secondary_count", "difference", "review_status",
        ],
        matched,
    )
    _write_csv(
        output_dir / "unmapped_primary_labels.csv",
        ["primary_label", "primary_count", "reason"],
        unmatched_primary,
    )
    _write_csv(
        output_dir / "secondary_without_exact_primary_label.csv",
        ["secondary_label", "iso3_candidate", "secondary_count", "reason"],
        unmatched_secondary,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Run the audit-only reconciliation on an existing primary capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()
    report = reconcile(arguments.primary_dir, arguments.secondary, arguments.out_dir)
    print(
        "Exact-label ISO3 candidates: "
        f"{report['n_exact_label_candidates']}; "
        f"primary labels awaiting review: {report['n_unmapped_primary_labels']}. "
        "No primary geography mappings have been verified."
    )


if __name__ == "__main__":
    main()
