"""Validate the provenance and structure of the causal-inference reference ledger.

This is a local integrity check, not independent bibliographic verification.
Run: python causal_inference_statistics/validate_reference_ledger.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from urllib.parse import urlsplit

LEDGER = Path(__file__).resolve().parent / "reference_ledger.csv"
REQUIRED = frozenset(
    {"reference_id", "reference", "doi", "status", "verified_source", "verification_note"}
)
ALLOWED_STATUSES = frozenset({"pending", "verified"})


def valid_evidence_url(value: str) -> bool:
    """Require an absolute HTTP(S) URL with a hostname; reject placeholders."""
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return False
    try:
        parts = urlsplit(value)
        return parts.scheme in {"https", "http"} and bool(parts.hostname)
    except ValueError:
        return False


def validate_ledger(path: Path) -> list[str]:
    """Return actionable errors; do not modify the audited CSV."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    errors: list[str] = []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or set(reader.fieldnames) != REQUIRED:
            return [f"Expected columns {sorted(REQUIRED)}"]
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                errors.append(f"line {line}: malformed CSV row")
                continue
            ref_id = row["reference_id"].strip()
            status = row["status"].strip()
            evidence = row["verified_source"].strip()
            note = row["verification_note"].strip()
            if not ref_id or not row["reference"].strip():
                errors.append(f"line {line}: missing ID or citation")
            if ref_id in seen:
                errors.append(f"line {line}: duplicate reference_id {ref_id}")
            seen.add(ref_id)
            if status not in ALLOWED_STATUSES:
                errors.append(f"line {line}: invalid status {status!r}")
            elif status == "verified":
                if not valid_evidence_url(evidence):
                    errors.append(f"line {line}: verified reference lacks a valid evidence URL")
                if len(note) < 10:
                    errors.append(f"line {line}: verified reference lacks a meaningful verification note")
            elif evidence or note:
                errors.append(f"line {line}: pending reference must not retain verification evidence")
    return errors


def main() -> int:
    """Report the complete set of structural problems and exit nonzero on failure."""
    errors = validate_ledger(LEDGER)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Reference ledger structure and provenance fields: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
