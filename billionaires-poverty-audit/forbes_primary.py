"""Audit Forbes 2026 geography without storing individual billionaire profiles.

Explicit countryOfCitizenship denotes citizenship. The separate country field
remains api_country, not assumed to be residence. Only country aggregates,
source provenance and a SHA-256 checksum of raw response bytes are saved.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from forbes_pages import retrieve_annual_list

FORBES_URL: str = (
    "https://www.forbes.com/forbesapi/person/billionaires/"
    "2026/position/true.json?limit=5000"
)
PUBLISHED_2026_COUNT: int = 3428
COUNTRY_FIELDS: dict[str, str] = {
    "citizenship": "countryOfCitizenship",
    "api_country": "country",
}


def parse_annual_list(
    raw: bytes, *, expected_count: int = PUBLISHED_2026_COUNT
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Fail closed on incorrect source count, schema or duplicate URIs."""
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer")
    try:
        doc: Any = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Forbes feed did not return valid JSON") from exc
    if not isinstance(doc, dict):
        raise ValueError("Forbes response must be a JSON object")
    parent = doc.get("personList")
    if isinstance(parent, dict):
        persons = parent.get("persons")
        path = "personList.persons"
    else:
        persons = doc.get("persons")
        path = "persons"
    if not isinstance(persons, list):
        raise ValueError(f"Forbes schema changed, no persons array: {sorted(doc)}")
    if len(persons) != expected_count:
        raise ValueError(
            f"Expected {expected_count} entries in annual list, got {len(persons)}; "
            "check the list vintage or pagination"
        )
    if not all(isinstance(person, dict) for person in persons):
        raise ValueError("Forbes list contains non-object rows")
    uris = [
        person["uri"].strip()
        for person in persons
        if isinstance(person.get("uri"), str) and person["uri"].strip()
    ]
    if len(uris) < 0.9 * expected_count:
        raise ValueError("Too few person URI identifiers to check uniqueness")
    if len(uris) != len(set(uris)):
        raise ValueError("Duplicate person URI identifiers")
    return persons, {
        "json_array_path": path,
        "n_persons": len(persons),
        "n_uris": len(uris),
        "expected_count": expected_count,
        "root_keys": sorted(doc),
    }


def aggregate_country_fields(
    persons: list[Mapping[str, Any]], *, min_citizenship_coverage: float = 0.95
) -> tuple[dict[str, Counter[str]], dict[str, dict[str, Any]]]:
    """Count nonempty country labels for each field, without imputation."""
    if not isinstance(persons, list) or not persons:
        raise ValueError("Nonempty Forbes person list required")
    if not 0 < min_citizenship_coverage <= 1:
        raise ValueError("Invalid citizenship coverage threshold")
    counts: dict[str, Counter[str]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for name, field in COUNTRY_FIELDS.items():
        counter: Counter[str] = Counter()
        for record in persons:
            if not isinstance(record, Mapping):
                raise ValueError("Expected mapping records")
            value: Any = record.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if not isinstance(value, str):
                raise ValueError(f"Country field {field} contains non-text data")
            counter[value.strip()] += 1
        assigned = sum(counter.values())
        counts[name] = counter
        coverage[name] = {
            "field": field,
            "assigned": assigned,
            "missing": len(persons) - assigned,
            "fraction_assigned": assigned / len(persons),
            "distinct_labels": len(counter),
        }
    if coverage["citizenship"]["fraction_assigned"] < min_citizenship_coverage:
        raise ValueError("Citizenship field is insufficiently complete")
    return counts, coverage


def compare_secondary(
    primary: Counter[str], secondary_csv: Path
) -> list[dict[str, Any]]:
    """Compare exact country names only; do not guess ISO mappings."""
    if not isinstance(secondary_csv, Path) or not secondary_csv.is_file():
        raise ValueError("Secondary transcription is missing")
    df = pd.read_csv(secondary_csv, dtype={"country": "string"})
    required = {"country", "billionaires", "billionaire_method", "iso3"}
    if not required.issubset(df):
        raise ValueError(f"Secondary columns missing: {sorted(required - set(df))}")
    if df["country"].isna().any() or df["country"].duplicated().any():
        raise ValueError("Country labels must be unique and nonempty")
    if not df["billionaire_method"].eq("citizenship").all():
        raise ValueError("Transcription must be citizenship-based")
    values = pd.to_numeric(df["billionaires"], errors="raise")
    if values.isna().any() or (values < 0).any() or not (values % 1 == 0).all():
        raise ValueError("Counts must be nonnegative integers")
    secondary = dict(zip(df["country"].astype(str), values.astype(int)))
    return [
        {
            "country_label": label,
            "forbes_count": None if label not in primary else int(primary[label]),
            "secondary_count": None if label not in secondary else secondary[label],
            "difference": (
                int(primary[label] - secondary[label])
                if label in primary and label in secondary else None
            ),
            "exact_name_match": label in primary and label in secondary,
        }
        for label in sorted(set(primary) | set(secondary))
    ]


def run_primary_audit(
    output_dir: Path,
    *,
    secondary_csv: Path | None = None,
    payload: bytes | None = None,
    source_url: str = FORBES_URL,
    expected_count: int = PUBLISHED_2026_COUNT,
) -> dict[str, Any]:
    """Validate one Forbes response and save aggregate-only evidence."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing audit: {output_dir}")
    if payload is not None and not isinstance(payload, bytes):
        raise TypeError("payload must be bytes or None")
    if payload is None:
        raw, source_provenance = retrieve_annual_list(
            source_url, expected_count=expected_count
        )
    else:
        raw = payload
        source_provenance = {
            "mode": "provided_test_fixture",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "raw_response": False,
        }
    persons, schema = parse_annual_list(raw, expected_count=expected_count)
    counts, coverage = aggregate_country_fields(persons)
    comparison = (
        compare_secondary(counts["citizenship"], secondary_csv)
        if secondary_csv is not None else []
    )
    audit = {
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "source_retrieval": source_provenance,
        "source_sha256_meaning": (
            "Raw response for single-response mode, derived in-memory JSON "
            "for paginated mode. Raw page hashes are in source_retrieval."
        ),
        "schema": schema,
        "coverage": coverage,
        "secondary_comparison_rows": len(comparison),
        "geography_warning": (
            "CountryOfCitizenship is citizenship; API country is ambiguous "
            "and MUST NOT be labelled residence without separate evidence."
        ),
        "research_status": "aggregate audit, not an exact chart replication",
        "privacy": "raw list and person-level fields are not saved",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, counter in counts.items():
        rows = [
            {
                "country_label": label,
                "billionaires": int(n),
                "attribution": name,
                "source_field": COUNTRY_FIELDS[name],
            }
            for label, n in sorted(counter.items())
        ]
        pd.DataFrame(
            rows,
            columns=["country_label", "billionaires", "attribution", "source_field"],
        ).to_csv(output_dir / f"counts_{name}.csv", index=False)
    if secondary_csv is not None:
        pd.DataFrame(comparison).to_csv(
            output_dir / "secondary_discrepancies.csv", index=False
        )
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    """Command-line entrypoint for the aggregate-only audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, default=None)
    args = parser.parse_args()
    result = run_primary_audit(args.out_dir, secondary_csv=args.secondary)
    print(
        f"Forbes 2026: {result['schema']['n_persons']} annual entries; "
        f"citizenship labels={result['coverage']['citizenship']['distinct_labels']}"
    )


if __name__ == "__main__":
    main()
