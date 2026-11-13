"""Audit one captured INE 0012340 municipality slice against frozen AMP pairs.

This offline comparison is deliberately *partial*: a single Dim2 selection
can check 16 AMP interactions and the municipality's zero diagonal, but not
the symmetry of the full 17-by-17 matrix or its 136 independent pairs.
INE 0012340 measures 2021 work/study interactions, not residential moves.

Source bytes and the original capture manifest are never modified. Their
digests establish provenance continuity, not independent authentication of
the publisher, nor a causal interpretation of the indicator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from extract_commuting import OUTPUT as FROZEN_PAIRS, municipality_codes_from_panel
from ine_commuting_capture import _parse_json
from ine_commuting_partitioned_capture import candidate_slice_url, require_slice_url
from ine_commuting_reconciliation import (
    INDICATOR_ID,
    PERIOD,
    _decode_object,
    _digest,
    _int_count,
    read_frozen_pairs,
)

CAPTURE_PROTOCOL = "ine_0012340_2021_dico_partition_candidate_v1"
EXPECTED_UNIT = "Número (N.º)"


def validate_manifest(
    manifest_bytes: bytes,
    source_bytes: bytes,
    municipality_code: str,
    codes: Mapping[str, str],
) -> dict[str, Any]:
    """Check a captured Dim2 receipt, file size, URL and byte-exact SHA-256."""
    if municipality_code not in codes:
        raise ValueError("Requested code is not an AMP municipality")
    manifest = _decode_object(manifest_bytes, "partition capture manifest")
    if (manifest.get("capture_protocol") != CAPTURE_PROTOCOL
            or manifest.get("dimension") != "Dim2"
            or manifest.get("source_data_is_residential_migration") is not False
            or manifest.get("all_136_pairs_reconciled") is not False
            or manifest.get("metadata_dimensions_checked_against_panel") is not True):
        raise ValueError("Partition capture manifest protocol or evidence scope changed")
    metadata = manifest.get("source_metadata")
    if (not isinstance(metadata, dict)
            or metadata.get("indicator") != INDICATOR_ID
            or metadata.get("period") != PERIOD
            or metadata.get("metadata_unit") != EXPECTED_UNIT
            or metadata.get("geographic_dimensions") != [2, 3]
            or metadata.get("amp_municipalities_validated_in_both_dimensions") != len(codes)
            or metadata.get("metadata_sha256") != manifest.get("source_metadata_sha256")):
        raise ValueError("Partition capture metadata does not validate the AMP panel")

    requested = manifest.get("requested_codes")
    items = manifest.get("items")
    if (not isinstance(requested, list)
            or any(not isinstance(code, str) or code not in codes for code in requested)
            or municipality_code not in requested
            or len(requested) != len(set(requested))
            or not isinstance(items, list) or len(items) != len(requested)
            or manifest.get("capture_count") != len(requested)):
        raise ValueError("Partition manifest contains an invalid requested-code set")
    if any(not isinstance(item, dict) or item.get("code") != code
           for code, item in zip(requested, items)):
        raise ValueError("Partition manifest item codes do not match the requested order")
    if (sum(item.get("status") == "captured_unverified" for item in items)
            != manifest.get("completed_json_captures")):
        raise ValueError("Partition manifest success count is inconsistent")

    entry = items[requested.index(municipality_code)]
    if (entry.get("status") != "captured_unverified"
            or entry.get("requested_dico") != municipality_code
            or entry.get("requested_dimension") != "Dim2"
            or entry.get("municipality") != codes[municipality_code]
            or entry.get("filter_effect_verified") is not False
            or entry.get("municipal_pair_counts_reconciled") is not False
            or entry.get("observed_period_keys") != [PERIOD]
            or entry.get("filename") != f"ine_0012340_dim2_{municipality_code}_candidate.json"):
        raise ValueError("Source slice does not have the expected unverified capture record")
    expected_url = candidate_slice_url(municipality_code, dimension="Dim2")
    for key in ("candidate_url", "response_url"):
        url = entry.get(key)
        require_slice_url(url, dico=municipality_code, dimension="Dim2")
        if url != expected_url:
            raise ValueError(f"Partition manifest {key} is not the expected request URL")
    if (entry.get("sha256") != _digest(source_bytes)
            or entry.get("size_bytes") != len(source_bytes)):
        raise ValueError("Captured source bytes do not match manifest SHA-256/size")
    return entry


def parse_dim2_slice(
    source_bytes: bytes,
    municipality_code: str,
    codes: Mapping[str, str],
) -> tuple[dict[str, int], int]:
    """Require an effective Dim2 filter and all 17 local Dim3 counterparts.

    Other Dim3 records may be other municipalities or aggregate geographies;
    they are checked for structural uniqueness but never added to AMP totals.
    """
    decoded = _parse_json(source_bytes)
    if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], dict):
        raise ValueError("INE payload must be one indicator in a JSON array")
    indicator = decoded[0]
    if indicator.get("IndicadorCod") != INDICATOR_ID:
        raise ValueError("Source indicator is not INE 0012340")
    data = indicator.get("Dados")
    if not isinstance(data, dict) or set(data) != {PERIOD} or not isinstance(data[PERIOD], list):
        raise ValueError("Source slice must have exactly the 2021 reference period")
    success = indicator.get("Sucesso")
    if (not isinstance(success, dict)
            or success.get("Verdadeiro") != [{"Msg": "OK"}]):
        raise ValueError("INE source did not report success")

    observed: dict[str, int] = {}
    seen: set[str] = set()
    for index, row in enumerate(data[PERIOD], start=1):
        if not isinstance(row, dict) or not all(
            field in row for field in ("geocod", "geodsg", "dim_3", "valor")
        ):
            raise ValueError(f"INE slice row {index} lacks the source fields")
        if row["geocod"] != municipality_code or row["geodsg"] != codes[municipality_code]:
            raise ValueError("Dim2 filter is ineffective or geography changed")
        counterpart = row["dim_3"]
        if not isinstance(counterpart, str) or not counterpart or counterpart in seen:
            raise ValueError("Duplicate or invalid Dim3 geographic code")
        seen.add(counterpart)
        count = _int_count(row["valor"])
        if counterpart in codes:
            if row.get("dim_3_t") != codes[counterpart]:
                raise ValueError("Official AMP counterpart code/name is inconsistent")
            observed[counterpart] = count

    if set(observed) != set(codes):
        raise ValueError("Incomplete AMP Dim2 slice: not all 17 counterparts are present")
    if observed[municipality_code] != 0:
        raise ValueError("Same-municipality interaction diagonal is not zero")
    return observed, len(seen)


def reconcile_slice(
    municipality_code: str,
    official: Mapping[str, int],
    frozen: Mapping[tuple[str, str], int],
    codes: Mapping[str, str],
) -> dict[str, Any]:
    """Return all 16 numerical comparisons, without asserting full symmetry."""
    if set(official) != set(codes) or municipality_code not in codes:
        raise ValueError("A complete 17-code official source slice is required")
    rows: list[dict[str, str | int | bool]] = []
    for counterpart in sorted(set(codes) - {municipality_code}):
        pair = tuple(sorted((municipality_code, counterpart)))
        original = frozen[pair]
        current = official[counterpart]
        rows.append({
            "counterpart_code": counterpart,
            "counterpart_municipality": codes[counterpart],
            "ine_interactions": current,
            "frozen_third_party_interactions": original,
            "difference": current - original,
            "match": current == original,
        })
    matched = sum(row["match"] is True for row in rows)
    return {
        "status": "partial_match" if matched == len(rows) else "partial_discrepancies_detected",
        "compared_amp_off_diagonal_pairs": len(rows),
        "matched_pairs": matched,
        "mismatched_pairs": len(rows) - matched,
        "comparison_rows": rows,
    }


def run(
    source_json: Path,
    manifest_json: Path,
    municipality_code: str,
    output_json: Path,
) -> dict[str, Any]:
    """Produce an auditable partial source comparison without any HTTP calls."""
    source_bytes = source_json.read_bytes()
    manifest_bytes = manifest_json.read_bytes()
    codes = municipality_codes_from_panel()
    entry = validate_manifest(manifest_bytes, source_bytes, municipality_code, codes)
    official, source_records = parse_dim2_slice(source_bytes, municipality_code, codes)
    frozen = read_frozen_pairs()
    report: dict[str, Any] = {
        "evidence_class": "single_municipality_ine_dim2_source_slice_reconciliation",
        "indicator_id": INDICATOR_ID,
        "reference_period": PERIOD,
        "municipality_code": municipality_code,
        "municipality": codes[municipality_code],
        "capture_url": entry["response_url"],
        "source_sha256": _digest(source_bytes),
        "capture_manifest_sha256": _digest(manifest_bytes),
        "frozen_third_party_csv_sha256": _digest(FROZEN_PAIRS.read_bytes()),
        "source_records": source_records,
        "amp_geographies_present": len(official),
        "diagonal_count": official[municipality_code],
        "slice_filter_effect_observed": True,
        "full_17_municipality_matrix_reconciled": False,
        "source_authenticity": "capture manifest records an INE HTTPS URL; this offline audit cannot authenticate the response independently",
        "data_semantics": "2021 work/study interactions, not residential relocation or causal displacement",
        **reconcile_slice(municipality_code, official, frozen, codes),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Read the pinned source and provenance, writing a partial audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.source_json, args.manifest_json, args.municipality_code, args.output_json)
    print(f"INE 0012340 {report['municipality']}: "
          f"{report['matched_pairs']}/{report['compared_amp_off_diagonal_pairs']} "
          "AMP interactions match the frozen source; full matrix remains unverified.")
    if report["mismatched_pairs"]:
        parser.exit(1, "Partial source reconciliation identified discrepancies.\n")


if __name__ == "__main__":
    main()
