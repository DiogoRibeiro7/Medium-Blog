"""Cross-check frozen Forbes citizenship labels with UN area-code references.

This is an ISO3 *code* review, not verification of Forbes's original March
vintage, citizenship attribution policy, residence, or regression eligibility.
Only the frozen country aggregates and public UN geographical code metadata
are used. The emitted ledger retains an explicit publication veto.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
from typing import TypedDict, cast


SOURCE: Path = Path(__file__).resolve().parent
FROZEN: Path = SOURCE / "evidence" / "forbes-2026-run-42099896845"
REFERENCE: Path = SOURCE / "references" / "forbes_2026_un_country_code_review.csv"

# SHA-256 digests of actual file *bytes*, not Git's 40-digit blob object IDs.
PRIMARY_SHA256: str = "5936ce1c1f89d7bd7f2adf31ab0d7fc39c21d2fd42702e6296ef0e3ae8242696"
SECONDARY_SHA256: str = "0cb8973cf76e9194ec7a1d5cd6376193695a61317d7a3ab9f0ec096cb2fabe3b"
UN_REFERENCE_SHA256: str = "72f83caa7deaed60bed4072f0e0992658831860df78032c7bd7288bfdd377f56"

UNSD_URL: str = "https://unstats.un.org/unsd/methodology/m49/overview/"
UNPD_URL: str = (
    "https://population.un.org/dataportal/data/indicators/19/locations/158/"
    "start/1980/end/2026/metadata/metadatalocations"
)

# A secondary label or a missing-secondary status never independently verifies
# any country code. All final codes must also match the separate UN reference.
ALIASES: dict[str, tuple[str, str, str]] = {
    "Eswatini (Swaziland)": ("SWZ", "Eswatini", "historical_alias"),
    "Afghanistan": ("AFG", "", "missing_from_secondary"),
    "Pakistan": ("PAK", "", "missing_from_secondary"),
    "Peru": ("PER", "", "missing_from_secondary"),
    "St. Kitts and Nevis": ("KNA", "", "abbreviated_label"),
}
TERRITORIAL_LABELS: frozenset[str] = frozenset({"Guernsey", "Hong Kong", "Taiwan"})


class ReviewRow(TypedDict):
    """One geographical code cross-check with explicit research restrictions."""

    primary_label: str
    iso3_candidate: str
    m49_code: str
    un_geography_name: str
    reference_authority: str
    primary_count: int
    secondary_count: str
    difference_from_secondary: str
    match_type: str
    territory_review: str
    review_status: str
    evidence_url: str
    approval_status: str
    reviewer_note: str


def read_csv(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    """Reject modified input bytes, incomplete columns and null CSV cells."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected SHA-256 digest must contain 64 hex characters")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"Frozen input digest mismatch: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path.name}")
        rows = list(reader)
    if not rows or any(
        key is None or value is None
        for row in rows for key, value in row.items()
    ):
        raise ValueError(f"Malformed or empty CSV: {path.name}")
    return cast(list[dict[str, str]], rows)


def _reference_index() -> dict[str, dict[str, str]]:
    """Validate an 80-row external code check without using Forbes counts."""
    rows = read_csv(REFERENCE, UN_REFERENCE_SHA256)
    expected_columns = {
        "forbes_label", "iso3", "m49_code", "un_geography_name",
        "evidence_url", "reference", "name_review", "territory_review",
    }
    index: dict[str, dict[str, str]] = {}
    codes: set[str] = set()
    for row in rows:
        if not expected_columns.issubset(row):
            raise ValueError("UN code reference is missing required columns")
        label: str = row["forbes_label"]
        code: str = row["iso3"]
        un_name: str = row["un_geography_name"]
        if (
            not label or not un_name or label in index or code in codes
            or not re.fullmatch(r"[A-Z]{3}", code)
            or not re.fullmatch(r"[0-9]{3}", row["m49_code"])
        ):
            raise ValueError("Duplicate or invalid UN code reference")
        authority: str = row["reference"]
        if authority == "UNSD_M49":
            if row["evidence_url"] != UNSD_URL:
                raise ValueError("Unexpected UNSD evidence URL")
            if row["name_review"] not in ("exact_name", "reviewed_alias"):
                raise ValueError("UNSD name review status is invalid")
            if (row["name_review"] == "exact_name") != (un_name == label):
                raise ValueError("Exact-name or reviewed-alias status is incorrect")
        elif authority == "UNPD_location_metadata":
            if (
                label != "Taiwan" or code != "TWN" or row["m49_code"] != "158"
                or row["evidence_url"] != UNPD_URL
                or row["name_review"] != "reviewed_area_label"
            ):
                raise ValueError("Unexpected UN Population Division area mapping")
        else:
            raise ValueError("Unrecognized reference authority")
        expected_territory: str = "required" if label in TERRITORIAL_LABELS else "standard"
        if row["territory_review"] != expected_territory:
            raise ValueError("Territory-review status does not match the area label")
        index[label] = row
        codes.add(code)
    if len(index) != 80:
        raise ValueError("UN code crosswalk must cover precisely 80 Forbes labels")
    return index


def build_ledger() -> list[ReviewRow]:
    """Reconcile all 80 source labels; never authorize a statistical model."""
    primary = read_csv(FROZEN / "counts_citizenship.csv", PRIMARY_SHA256)
    secondary = read_csv(
        SOURCE / "data" / "forbes_2026_citizenship_secondary.csv",
        SECONDARY_SHA256,
    )
    references = _reference_index()
    lookup = {item["country"]: item for item in secondary}
    if len(lookup) != len(secondary):
        raise ValueError("Duplicate secondary country names")
    if len(primary) != 80 or sum(int(item["billionaires"]) for item in primary) != 3428:
        raise ValueError("Unexpected frozen Forbes primary population")
    if len({item["country_label"] for item in primary}) != len(primary):
        raise ValueError("Duplicate primary country labels")

    result: list[ReviewRow] = []
    for item in primary:
        label: str = item["country_label"]
        reference = references.get(label)
        if reference is None:
            raise ValueError(f"Missing UN code reference for Forbes label: {label}")
        source_code: str = reference["iso3"]
        alias = ALIASES.get(label)
        if label in lookup:
            candidate = lookup[label]
            candidate_code: str = candidate["iso3"]
            secondary_count = candidate["billionaires"]
            match_type = "exact_secondary_label"
        elif alias is not None:
            candidate_code, secondary_label, match_type = alias
            candidate = lookup.get(secondary_label)
            secondary_count = candidate["billionaires"] if candidate is not None else ""
        else:
            raise ValueError(f"Unmapped secondary Forbes country label: {label}")
        if candidate_code != source_code:
            raise ValueError(f"Secondary candidate disagrees with UN code for {label}")
        primary_count: int = int(item["billionaires"])
        if primary_count < 1 or (secondary_count and int(secondary_count) < 0):
            raise ValueError("Forbes country counts must be nonnegative integers")

        result.append({
            "primary_label": label,
            "iso3_candidate": source_code,
            "m49_code": reference["m49_code"],
            "un_geography_name": reference["un_geography_name"],
            "reference_authority": reference["reference"],
            "primary_count": primary_count,
            "secondary_count": secondary_count,
            "difference_from_secondary": (
                str(primary_count - int(secondary_count)) if secondary_count else ""
            ),
            "match_type": match_type,
            "territory_review": reference["territory_review"],
            "review_status": "iso3_code_crosschecked",
            "evidence_url": reference["evidence_url"],
            "approval_status": "not_approved_for_regression",
            "reviewer_note": (
                "Area definition requires a downstream compatibility review"
                if label in TERRITORIAL_LABELS else ""
            ),
        })
    if len({row["iso3_candidate"] for row in result}) != len(result):
        raise ValueError("Country-to-ISO3 crosswalk must remain one-to-one")
    return result


def main() -> None:
    """Write one fresh CSV review ledger, refusing evidence overwrite."""
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
