"""Check an operator-supplied INE 2021 commuting JSON export against frozen pairs.

The public INE indicator 0012340 describes municipal *work/study*
interactions, not changes of residence. This script never downloads data or
replaces the pinned analysis matrix. It compares a checksum-matched, externally
retrieved JSON response after an explicitly reviewed geographic-code mapping.

A digest verifies file integrity, not that the operator really obtained the
file from INE. Source authenticity and the meaning of INE dimensions must be
checked independently against official metadata before interpreting a match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit

from extract_commuting import (
    FIELDS,
    INDICATOR_URL,
    OUTPUT as FROZEN_PAIRS,
    PROVENANCE as FROZEN_PROVENANCE,
    municipality_codes_from_panel,
)

ROOT = Path(__file__).resolve().parent
INDICATOR_ID = "0012340"
PERIOD = "2021"
SOURCE_GEOGRAPHY = "2021 Census, NUTS 2024 municipal geography"
SOURCE_RESPONSE_FIELDS = ("geocod", "dim_3", "valor")
MAP_FIELDS = ("source_code", "dico")
HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
Row = dict[str, str | int]


def _digest(data: bytes) -> str:
    """Return SHA-256 of the original bytes, without normalising the source."""
    return hashlib.sha256(data).hexdigest()


def _decode_object(data: bytes, label: str) -> dict[str, Any]:
    """Decode an object and reject duplicate JSON keys or malformed UTF-8."""
    def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"Duplicate JSON key in {label}: {key!r}")
            output[key] = value
        return output

    try:
        result = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON in {label}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object in {label}")
    return result


def _source_url_is_exact_indicator(url: str) -> bool:
    """Accept only the explicit INE census API request, not an arbitrary URL."""
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != "www.ine.pt"
            or parts.path != "/ine/json_indicador/pindica.jsp"
            or parts.fragment or parts.username or parts.password):
        return False
    params = parse_qs(parts.query, strict_parsing=True)
    return params == {
        "op": ["2"],
        "varcd": [INDICATOR_ID],
        "Dim1": ["S7A2021"],
        "lang": ["PT"],
    }


def read_receipt(path: Path, source_bytes: bytes) -> dict[str, Any]:
    """Validate a local download receipt, period, declared API URL and digest."""
    record = _decode_object(path.read_bytes(), "receipt")
    expected = {
        "indicator_id": INDICATOR_ID,
        "reference_period": PERIOD,
        "geography": SOURCE_GEOGRAPHY,
        "primary_dimension_field": SOURCE_RESPONSE_FIELDS[0],
        "secondary_dimension_field": SOURCE_RESPONSE_FIELDS[1],
        "count_field": SOURCE_RESPONSE_FIELDS[2],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Wrong INE receipt field {key}; inspect official indicator metadata")
    if not _source_url_is_exact_indicator(record.get("source_url")):
        raise ValueError("INE source URL must select only 0012340 and the 2021 period")
    digest = record.get("response_sha256")
    if not isinstance(digest, str) or HEX_DIGEST.fullmatch(digest) is None:
        raise ValueError("INE response_sha256 must be lowercase 64-character SHA-256")
    if digest != _digest(source_bytes):
        raise ValueError("INE response SHA-256 mismatch")
    captured = record.get("downloaded_at_utc")
    if not isinstance(captured, str):
        raise ValueError("Download receipt must include downloaded_at_utc")
    try:
        timestamp = datetime.fromisoformat(captured.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid UTC download timestamp") from exc
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("Download timestamp must specify UTC")
    return record


def read_code_map(path: Path, expected_codes: Mapping[str, str]) -> dict[str, str]:
    """Require a reviewed, one-to-one mapping from source codes to AMP DICO."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(MAP_FIELDS):
            raise ValueError("Geographic code mapping must use source_code,dico")
        for row in reader:
            raw, dico = row["source_code"], row["dico"]
            if (not raw or raw != raw.strip() or not dico or
                    dico not in expected_codes or raw in mapping):
                raise ValueError("Invalid, duplicated or unknown municipality source-code mapping")
            mapping[raw] = dico
    if len(mapping) != len(expected_codes) or set(mapping.values()) != set(expected_codes):
        raise ValueError("Source mapping must cover exactly the 17 AMP DICO codes once")
    return mapping


def _int_count(value: object) -> int:
    """Read canonical nonnegative integer counts; do not coerce null or float."""
    if isinstance(value, bool):
        raise ValueError("Boolean count is invalid")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"0|[1-9]\d*", value):
        return int(value)
    raise ValueError("Official interaction count must be a nonnegative integer")


def parse_ine_matrix(raw_bytes: bytes, mapping: Mapping[str, str]) -> dict[tuple[str, str], int]:
    """Parse a 17x17 off-diagonal symmetric view of one INE JSON period.

    A row with a geographic code outside the reviewed 17-municipality map
    is excluded *only after* validating the structural fields. Duplicated
    local pairs, nonzero diagonals, suppressed values and asymmetry fail closed.
    """
    try:
        value = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid official JSON response") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("INE API response must be a single-indicator JSON array")
    indicator = value[0]
    for field in ("IndicadorCod", "IndicadorId", "varcd"):
        if field in indicator and str(indicator[field]) != INDICATOR_ID:
            raise ValueError("INE payload indicator code is not 0012340")
    data = indicator.get("Dados")
    if not isinstance(data, dict) or set(data) != {PERIOD} or not isinstance(data[PERIOD], list):
        raise ValueError("INE payload must contain exactly one 2021 data period")
    pairs: dict[tuple[str, str], int] = {}
    diagonals: set[str] = set()
    for offset, row in enumerate(data[PERIOD], start=1):
        if not isinstance(row, dict) or any(k not in row for k in SOURCE_RESPONSE_FIELDS):
            raise ValueError(f"INE data row {offset} lacks declared dimension/value fields")
        first, second = row["geocod"], row["dim_3"]
        if not isinstance(first, str) or not isinstance(second, str):
            raise ValueError("INE geographic dimension codes must be strings")
        if first not in mapping or second not in mapping:
            continue
        a, b = mapping[first], mapping[second]
        count = _int_count(row["valor"])
        if a == b:
            if a in diagonals:
                raise ValueError("Duplicated INE AMP diagonal; unresolved indicator dimensions")
            if count != 0:
                raise ValueError("INE same-municipality interactions are nonzero; semantics differ")
            diagonals.add(a)
            continue
        key = (a, b)
        if key in pairs:
            raise ValueError(f"Duplicated INE AMP pair {key}; unresolved indicator dimensions")
        pairs[key] = count
    if diagonals != set(mapping.values()):
        raise ValueError("Incomplete INE AMP matrix: missing same-municipality diagonals")
    expected = {(a, b) for a in mapping.values() for b in mapping.values() if a != b}
    if set(pairs) != expected:
        raise ValueError(
            f"Incomplete INE AMP pair matrix: {len(expected - set(pairs))} missing, "
            f"{len(set(pairs) - expected)} unexpected"
        )
    if any(pairs[a, b] != pairs[b, a] for a, b in pairs):
        raise ValueError("INE matrix is asymmetric; cannot compare it as undirected interactions")
    return pairs


def read_frozen_pairs(
    path: Path = FROZEN_PAIRS,
    provenance: Path = FROZEN_PROVENANCE,
) -> dict[tuple[str, str], int]:
    """Load the pinned workbook derivative and verify its original checksum."""
    manifest = _decode_object(provenance.read_bytes(), "frozen commuting provenance")
    if _digest(path.read_bytes()) != manifest.get("derived_csv_sha256"):
        raise ValueError("Frozen commuting pair CSV failed its recorded SHA-256")
    codes = municipality_codes_from_panel()
    result: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Frozen commuting CSV schema has changed")
        for row in reader:
            a, b = row["municipality_a_code"], row["municipality_b_code"]
            if (a not in codes or b not in codes or a >= b or
                    row["municipality_a"] != codes[a] or row["municipality_b"] != codes[b]
                    or (a, b) in result):
                raise ValueError("Frozen commuting pair identity is invalid or duplicated")
            result[a, b] = _int_count(row["commuting_interactions_2021"])
    expected = {(a, b) for i, a in enumerate(sorted(codes))
                for b in sorted(codes)[i + 1:]}
    if set(result) != expected:
        raise ValueError("Frozen commuting panel is missing canonical pairs")
    return result


def compare_pairs(
    official: Mapping[tuple[str, str], int],
    frozen: Mapping[tuple[str, str], int],
    codes: Mapping[str, str],
) -> dict[str, Any]:
    """Return transparent row-level differences, never auto-replacing source data."""
    if set(official) != {(a, b) for a, b in official}:
        raise ValueError("Unexpected official matrix shape")
    expected = {(a, b) for a in codes for b in codes if a != b}
    if set(official) != expected:
        raise ValueError("Official comparison must contain 272 directed AMP pair views")
    expected_frozen = {(a, b) for i, a in enumerate(sorted(codes))
                       for b in sorted(codes)[i + 1:]}
    if set(frozen) != expected_frozen:
        raise ValueError("Frozen comparison must contain 136 undirected AMP pairs")
    differences: list[Row] = []
    for a, b in sorted(expected_frozen):
        current, original = official[a, b], frozen[a, b]
        if current != original:
            differences.append({
                "municipality_a_code": a,
                "municipality_a": codes[a],
                "municipality_b_code": b,
                "municipality_b": codes[b],
                "ine_interactions": current,
                "third_party_interactions": original,
                "ine_minus_third_party": current - original,
            })
    return {
        "status": "match" if not differences else "discrepancies_detected",
        "compared_undirected_pairs": len(expected_frozen),
        "matched_pairs": len(expected_frozen) - len(differences),
        "mismatched_pairs": len(differences),
        "mismatches": differences,
    }


def run(
    source_json: Path,
    receipt_json: Path,
    code_map_csv: Path,
    output_json: Path,
) -> dict[str, Any]:
    """Audit provided files, write a report, and preserve all upstream originals."""
    raw_bytes = source_json.read_bytes()
    receipt = read_receipt(receipt_json, raw_bytes)
    codes = municipality_codes_from_panel()
    mapping = read_code_map(code_map_csv, codes)
    official = parse_ine_matrix(raw_bytes, mapping)
    frozen = read_frozen_pairs()
    comparison = compare_pairs(official, frozen, codes)
    report = {
        "evidence_class": "operator_supplied_independent_ine_json_pairwise_reconciliation",
        "reported_source": "INE Censos 2021 municipal work/study interactions",
        "indicator_id": INDICATOR_ID,
        "indicator_url": INDICATOR_URL,
        "reference_period": PERIOD,
        "declared_download_url": receipt["source_url"],
        "declared_downloaded_at_utc": receipt["downloaded_at_utc"],
        "official_response_sha256": _digest(raw_bytes),
        "code_map_sha256": _digest(code_map_csv.read_bytes()),
        "frozen_third_party_csv_sha256": _digest(FROZEN_PAIRS.read_bytes()),
        "geographic_units": len(codes),
        "directed_views_validated": len(official),
        "source_authenticity": "operator-attested only; not independently verified by this offline program",
        "data_semantics": "symmetric undirected 2021 work/study interactions; not residential relocations",
        "publication_rule": "do not alter or promote the frozen matrix without separate raw source and definition review",
        **comparison,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Validate a supplied INE response and produce a non-destructive audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--receipt-json", type=Path, required=True)
    parser.add_argument("--code-map-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.source_json, args.receipt_json,
                 args.code_map_csv, args.output_json)
    print(f"Compared {result['compared_undirected_pairs']} AMP pairs: "
          f"{result['matched_pairs']} matches, {result['mismatched_pairs']} mismatches")
    if result["mismatched_pairs"]:
        parser.exit(status=1, message="Official/third-party commuting counts disagree; review audit JSON.\n")


if __name__ == "__main__":
    main()
