"""Verify March 2026 Forbes country membership against November aggregates.

The source confirms at least one billionaire citizen in each named area.
It does NOT establish March country totals or residence attribution.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

STUDY = Path(__file__).resolve().parent
MARCH = STUDY / "data" / "forbes_march_2026_country_presence.csv"
NOVEMBER = STUDY / "evidence" / "forbes-2026-run-42099896845" / "counts_citizenship.csv"


class MembershipAudit(TypedDict):
    """Set-comparison results without inferred billionaire counts."""

    march_labels: int
    november_labels: int
    march_only: list[str]
    november_only: list[str]
    membership_matches: bool
    counts_verified: bool
    residence_verified: bool


def _labels(path: Path, field: str) -> list[str]:
    """Read the named column, rejecting empty and duplicate country names."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or field not in reader.fieldnames:
            raise ValueError(f"Missing {field} column: {path}")
        result = [row[field].strip() for row in reader]
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"Missing or repeated country label: {path}")
    return result


def audit_membership(march: Path = MARCH, november: Path = NOVEMBER) -> MembershipAudit:
    """Compare independent March editorial labels with live November labels."""
    if not isinstance(march, Path) or not isinstance(november, Path):
        raise TypeError("Input paths must be pathlib.Path")
    march_labels = set(_labels(march, "country_label"))
    november_labels = set(_labels(november, "country_label"))
    return {
        "march_labels": len(march_labels),
        "november_labels": len(november_labels),
        "march_only": sorted(march_labels - november_labels),
        "november_only": sorted(november_labels - march_labels),
        "membership_matches": march_labels == november_labels,
        "counts_verified": False,
        "residence_verified": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(audit_membership(), indent=2, sort_keys=True))
