"""Build an audit-only ISO3 review ledger from the frozen Forbes aggregates.

Every mapping remains provisional until independent evidence is recorded.
No person-level data are read, and this output is never an approved model input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import TypedDict

SOURCE = Path(__file__).resolve().parent
FROZEN = SOURCE / "evidence" / "forbes-2026-run-42099896845"
PRIMARY_SHA256 = "83879589e6a5ec70018b7a669e8514a3eb183d13"
SECONDARY_SHA256 = "2b80ea845ca2737134ced47cc0fbe38109094a80"

# Explicit label aliases/candidates; these are not independent source approvals.
ALIASES: dict[str, tuple[str, str, str]] = {
    "Eswatini (Swaziland)": ("SWZ", "Eswatini", "historical_alias"),
    "Afghanistan": ("AFG", "", "missing_from_secondary"),
    "Pakistan": ("PAK", "", "missing_from_secondary"),
    "Peru": ("PER", "", "missing_from_secondary"),
    "St. Kitts and Nevis": ("KNA", "", "abbreviated_label"),
}

# These classifications must be treated explicitly in any downstream join.
TERRITORIAL_LABELS = frozenset({"Guernsey", "Hong Kong", "Taiwan"})


class ReviewRow(TypedDict):
    """One provisional, non-approved ISO3 country mapping."""

    primary_label: str
    iso3_candidate: str
    primary_count: int
    secondary_count: str
    match_type: str
    territory_review: str
    review_status: str
    evidence_url: str
    reviewer_note: str


def read_csv(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    """Read immutable evidence only after verifying exact source bytes."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Frozen input digest mismatch: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_ledger() -> list[ReviewRow]:
    """Match exact names or documented aliases without authorizing regression."""
    primary = read_csv(FROZEN / "counts_citizenship.csv", PRIMARY_SHA256)
    secondary = read_csv(
        SOURCE / "data" / "forbes_2026_citizenship_secondary.csv",
        SECONDARY_SHA256,
    )
    lookup = {item["country"]: item for item in secondary}
    if len(lookup) != len(secondary):
        raise ValueError("Duplicate secondary country names")
    if len(primary) != 80 or sum(int(item["billionaires"]) for item in primary) != 3428:
        raise ValueError("Unexpected frozen Forbes primary population")
    if len({item["country_label"] for item in primary}) != len(primary):
        raise ValueError("Duplicate primary country labels")
    result: list[ReviewRow] = []
    for item in primary:
        label = item["country_label"]
        alias = ALIASES.get(label)
        if label in lookup:
            candidate = lookup[label]
            iso3 = candidate["iso3"]
            secondary_count = candidate["billionaires"]
            match_type = "exact_secondary_label"
        elif alias is not None:
            iso3, secondary_label, match_type = alias
            candidate = lookup.get(secondary_label)
            secondary_count = candidate["billionaires"] if candidate is not None else ""
        else:
            # A new unrecognized primary country cannot silently enter the panel.
            raise ValueError(f"Unmapped Forbes country label: {label}")
        if len(iso3) != 3 or not iso3.isascii() or not iso3.isupper():
            raise ValueError(f"Invalid ISO3 candidate for {label}")
        result.append({
            "primary_label": label,
            "iso3_candidate": iso3,
            "primary_count": int(item["billionaires"]),
            "secondary_count": secondary_count,
            "match_type": match_type,
            "territory_review": "required" if label in TERRITORIAL_LABELS else "standard",
            "review_status": "unverified_candidate",
            "evidence_url": "",
            "reviewer_note": "",
        })
    if len({row["iso3_candidate"] for row in result}) != len(result):
        raise ValueError("Country-to-ISO3 candidates are not one-to-one")
    return result


def main() -> None:
    """Write a reviewable CSV; reject accidental replacement of an existing ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing ledger: {output}")
    rows = build_ledger()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ReviewRow.__annotations__))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
