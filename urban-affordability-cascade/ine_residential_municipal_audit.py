"""Reconcile 17 official INE 0011631 municipality category slices with Porto archive.

The archived INE sources provide Census 2021 *destination-municipality*
residential migration categories. This audit never constructs a previous-
municipality origin-destination matrix, migration direction, or causal effect.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from extract_commuting import municipality_codes_from_panel
from ine_residential_categories_probe import municipality_candidate_url
from ine_residential_municipal_assemble import ASSEMBLY_PROTOCOL
from ine_residential_municipal_capture import (
    CAPTURE_PROTOCOL,
    CATEGORY_NAMES,
    parse_municipality,
    validate_metadata,
)

ROOT = Path(__file__).resolve().parent
REFERENCE_CSV = ROOT / "data" / "porto_census2021_residential_mobility.csv"
OUTPUT_CSV = ROOT / "data" / "amp_census2021_municipal_residential_categories.csv"
OUTPUT_PROVENANCE = (
    ROOT / "data" / "amp_census2021_municipal_residential_categories_provenance.json"
)

CATEGORY_FIELDS = {
    "T": "resident_total_2021",
    "1": "not_born_2019",
    "2": "same_residence",
    "3": "changed_residence",
    "31": "moved_within_same_parish",
    "32": "moved_from_other_parish_same_municipality",
    "33": "moved_from_other_municipality",
    "34": "moved_from_abroad",
}
CSV_FIELDS = (
    "dico", "municipality", *CATEGORY_FIELDS.values(), "eligible_2019_population",
    "moved_from_other_municipality_pct_eligible",
)


def _sha(raw: bytes) -> str:
    """Return the lowercase SHA-256 of the exact saved source bytes."""
    return hashlib.sha256(raw).hexdigest()


def load_reference(path: Path = REFERENCE_CSV) -> dict[str, dict[str, str]]:
    """Read the already frozen official Porto/AMP migration-category totals."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {"1312", "11A"}
    selected = [row for row in rows if row["geography_code"] in expected]
    lookup = {row["geography_code"]: row for row in selected}
    if len(lookup) != 2 or len(selected) != 2:
        raise ValueError("Porto and AMP official destination margins are required")
    return lookup


def reconcile_categories(
    observed: Mapping[str, Mapping[str, int]],
    names: Mapping[str, str],
    reference: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str | int]]:
    """Reject mismatch with separately captured Porto and AMP destination totals."""
    if len(names) != 17 or set(observed) != set(names):
        raise ValueError("Exactly 17 AMP municipality category slices are required")
    for code, counts in observed.items():
        if set(counts) != set(CATEGORY_NAMES) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError(f"Invalid category grid or noninteger count for {code}")
        if counts["T"] != counts["1"] + counts["2"] + counts["3"]:
            raise ValueError(f"Incomplete total population partition for {code}")
        if counts["3"] != sum(counts[sub] for sub in ("31", "32", "33", "34")):
            raise ValueError(f"Incomplete residence-mover category partition for {code}")

    checked = {
        "1312": observed["1312"],
        "11A": {category: sum(observed[code][category] for code in names)
                for category in CATEGORY_NAMES},
    }
    for geography, counts in checked.items():
        original = reference.get(geography)
        if not isinstance(original, dict):
            raise ValueError(f"Missing official independent {geography} reference")
        for category, field in CATEGORY_FIELDS.items():
            if category == "3":
                # The frozen archive has four disjoint mover groups rather
                # than the source's parent category '3' as a standalone field.
                expected = sum(
                    int(original[name]) for name in (
                        "moved_within_same_parish",
                        "moved_from_other_parish_same_municipality",
                        "moved_from_other_municipality",
                        "moved_from_abroad",
                    )
                )
            else:
                expected = int(original[field])
            if counts[category] != expected:
                raise ValueError(
                    f"INE {geography} category {category} ({field}) "
                    f"disagrees with frozen official archive: {counts[category]} != {expected}"
                )

    rows: list[dict[str, str | int]] = []
    for code in sorted(names):
        counts = observed[code]
        eligible = counts["T"] - counts["1"]
        if eligible <= 0:
            raise ValueError("Census population eligible for prior residence must be positive")
        row: dict[str, str | int] = {"dico": code, "municipality": names[code]}
        row.update({field: counts[category] for category, field in CATEGORY_FIELDS.items()})
        row.update({
            "eligible_2019_population": eligible,
            "moved_from_other_municipality_pct_eligible": (
                f"{100 * counts['33'] / eligible:.6f}"
            ),
        })
        rows.append(row)
    return rows


def audit_archive(
    archive_directory: Path,
    *,
    reference_csv: Path = REFERENCE_CSV,
) -> tuple[list[dict[str, str | int]], dict[str, Any]]:
    """Check exact captures and derive a complete non-OD municipal category panel."""
    names = municipality_codes_from_panel()
    compressed_path = archive_directory / "ine_0011631_metadata.json.gz"
    compressed = compressed_path.read_bytes()
    metadata_bytes = gzip.decompress(compressed)
    metadata = validate_metadata(metadata_bytes, names)
    manifest_bytes = (archive_directory / "municipal_capture_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    codes = sorted(names)
    if (
        not isinstance(manifest, dict)
        or manifest.get("protocol") != CAPTURE_PROTOCOL
        or manifest.get("requested_codes") != codes
        or manifest.get("requested_count") != 17
        or manifest.get("validated_count") != 17
        or manifest.get("full_residential_origin_destination_matrix_observed") is not False
        or manifest.get("source_metadata") != metadata
        or manifest.get("metadata_source_sha256") != _sha(metadata_bytes)
        or manifest.get("housing_displacement_causality_established") is not False
    ):
        raise ValueError("Archived INE 0011631 metadata or manifest scope changed")
    captured = manifest.get("items")
    if not isinstance(captured, list) or len(captured) != 17:
        raise ValueError("Archived source manifest lacks all 17 municipality entries")

    attempts: dict[str, dict[str, Any]] = {}
    if "archive_assembly_protocol" in manifest:
        if manifest["archive_assembly_protocol"] != ASSEMBLY_PROTOCOL:
            raise ValueError("Unexpected INE archive assembly protocol")
        receipts = manifest.get("source_attempt_receipts")
        if not isinstance(receipts, list) or not 1 <= len(receipts) <= 10:
            raise ValueError("Missing original INE source attempt receipts")
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise ValueError("Invalid source attempt provenance")
            attempt = receipt.get("attempt")
            if not isinstance(attempt, str) or attempt in attempts or not attempt.isascii() or not attempt.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Invalid or repeated source attempt identifier")
            evidence = archive_directory / "evidence"
            original_manifest = (evidence / f"{attempt}_municipal_capture_manifest.json").read_bytes()
            original_probe = (evidence / f"{attempt}_probe_manifest.json").read_bytes()
            original_meta = gzip.decompress(
                (evidence / f"{attempt}_metadata.json.gz").read_bytes()
            )
            if (
                _sha(original_manifest) != receipt.get("municipal_manifest_sha256")
                or _sha(original_probe) != receipt.get("probe_manifest_sha256")
                or _sha(original_meta) != receipt.get("metadata_source_sha256")
            ):
                raise ValueError("Changed original source attempt receipt")
            original = json.loads(original_manifest)
            probe = json.loads(original_probe)
            if (
                not isinstance(probe, dict)
                or probe.get("protocol") != "ine_0011631_2021_porto_category_source_probe_v1"
                or probe.get("indicator_id") != "0011631"
                or probe.get("requested_kinds") != ["metadata", "porto_dim2_candidate"]
                or probe.get("source_semantics_verified") is not False
                or probe.get("directed_residential_origin_destination_counts_reconciled") is not False
                or not isinstance(probe.get("items"), list)
                or len(probe["items"]) != 2
                or any(not isinstance(item, dict) for item in probe["items"])
                or probe["items"][0].get("source_sha256") != _sha(original_meta)
                or probe["items"][1].get("source_sha256") != _sha(
                    (evidence / f"{attempt}_porto_candidate.json").read_bytes()
                )
            ):
                raise ValueError("Original candidate source receipt changed")
            if (
                not isinstance(original, dict)
                or original.get("protocol") != CAPTURE_PROTOCOL
                or original.get("metadata_source_sha256") != _sha(original_meta)
                or original.get("source_metadata") != validate_metadata(original_meta, names)
                or original.get("full_residential_origin_destination_matrix_observed") is not False
            ):
                raise ValueError("Invalid original source attempt scope")
            original_items = original.get("items")
            if not isinstance(original_items, list) or sorted(
                item.get("code") for item in original_items if isinstance(item, dict)
            ) != original.get("requested_codes"):
                raise ValueError("Incomplete original source attempt membership")
            originals = {
                item["code"]: item for item in original_items
                if isinstance(item, dict) and item.get("status") == "validated_2021_migration_categories"
            }
            if sorted(originals) != receipt.get("validated_codes"):
                raise ValueError("Original source attempt validated membership changed")
            attempts[attempt] = originals

    observed: dict[str, dict[str, int]] = {}
    raw_records: list[dict[str, str | int]] = []
    for code, item in zip(codes, captured):
        if not isinstance(item, dict):
            raise ValueError("Invalid archived capture item")
        filename = f"ine_0011631_dim2_{code}_2021.json"
        if (
            item.get("code") != code
            or item.get("municipality") != names[code]
            or item.get("status") != "validated_2021_migration_categories"
            or item.get("source_url") != municipality_candidate_url(code)
            or item.get("response_url") != municipality_candidate_url(code)
            or item.get("reference_period") != "2021"
            or item.get("filename") != filename
            or item.get("municipal_origin_identified") is not False
        ):
            raise ValueError(f"Archive has wrong municipality identity for {code}")
        if attempts:
            selected_attempt = item.get("selected_from_attempt")
            if selected_attempt not in attempts or attempts[selected_attempt].get(code) != {
                key: value for key, value in item.items() if key != "selected_from_attempt"
            }:
                raise ValueError(f"Original source attempt does not support {code}")
        raw = (archive_directory / filename).read_bytes()
        digest = _sha(raw)
        if digest != item.get("source_sha256") or len(raw) != item.get("source_bytes"):
            raise ValueError(f"Raw INE source/manifest SHA or byte size changed for {code}")
        counts = parse_municipality(raw, code, names)
        if item.get("both_sexes_counts") != counts:
            raise ValueError(f"Archived {code} category totals changed")
        observed[code] = counts
        raw_records.append({
            "dico": code, "sha256": digest, "bytes": len(raw),
            "filename": filename,
        })

    reference = load_reference(reference_csv)
    rows = reconcile_categories(observed, names, reference)
    provenance = {
        "evidence_class": "2021_census_municipal_destination_migration_categories_not_od",
        "indicator_id": "0011631",
        "period": "2021",
        "geography": "17 AMP municipalities, 2021 Census NUTS 2013 municipality DICO",
        "unit": "persons",
        "source_metadata_sha256": _sha(metadata_bytes),
        "metadata_gzip_sha256": _sha(compressed),
        "municipal_capture_manifest_sha256": _sha(manifest_bytes),
        "independent_porto_archive_csv_sha256": _sha(reference_csv.read_bytes()),
        "municipalities": len(rows),
        "category_cells_per_municipality": 24,
        "porto_previous_other_portuguese_municipality": observed["1312"]["33"],
        "amp_previous_other_portuguese_municipality": sum(
            observed[code]["33"] for code in names
        ),
        "source_data_is_named_origin_destination": False,
        "residential_displacement_or_housing_causality_established": False,
        "source_attempt_count": len(attempts) if attempts else 1,
        "raw_source_files": raw_records,
    }
    return rows, provenance


def run(
    capture_directory: Path,
    output_csv: Path = OUTPUT_CSV,
    output_provenance: Path = OUTPUT_PROVENANCE,
) -> dict[str, Any]:
    """Write deterministic derived CSV and checksum-bearing source provenance."""
    rows, provenance = audit_archive(capture_directory)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw_csv = buffer.getvalue().encode("utf-8")
    provenance["derived_csv_sha256"] = _sha(raw_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_provenance.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_bytes(raw_csv)
    output_provenance.write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return provenance


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline source-verification and derived-margin extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-provenance", type=Path, default=OUTPUT_PROVENANCE)
    args = parser.parse_args(argv)
    result = run(args.capture_directory, args.output_csv, args.output_provenance)
    print(f"INE 0011631: {result['municipalities']} AMP destination category rows; "
          f"Porto {result['porto_previous_other_portuguese_municipality']}, "
          f"AMP {result['amp_previous_other_portuguese_municipality']} prior other-municipality counts")


if __name__ == "__main__":
    main()
