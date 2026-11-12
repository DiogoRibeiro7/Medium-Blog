"""Review integrity of an INE capture artifact without asserting source semantics.

This is a deterministic, offline handoff between the manual 0012340 source
capture and the separately reviewed pairwise reconciliation. Exact response
bytes and SHA-256 are checked against the capture manifest. This does NOT
authenticate the claimed INE publisher, validate Census dimensions, or
establish a match to the frozen work/study interaction matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from ine_commuting_capture import (
    INDICATOR,
    MAX_BYTES,
    SOURCES,
    _endpoint_kind,
    _parse_json,
    _schema_outline,
)

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
CAPTURE_STATUS = {"json_saved_schema_unreviewed", "capture_failed"}


def _check_utc_timestamp(timestamp: object) -> None:
    """Require an explicit UTC offset on the original capture timestamp."""
    if not isinstance(timestamp, str):
        raise ValueError("Capture timestamp must be an ISO 8601 UTC string")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Capture timestamp is not ISO 8601") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("Capture timestamp must specify UTC")


def _read_manifest(directory: Path) -> dict[str, Any]:
    """Load a supported capture manifest and reject unsupported trust claims."""
    manifest_path = directory / "capture_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Capture manifest is missing or is a symlink")
    try:
        manifest = _parse_json(manifest_path.read_bytes())
    except OSError as exc:
        raise ValueError("Cannot read capture manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Capture manifest must be one JSON object")
    if type(manifest.get("capture_version")) is not int or manifest["capture_version"] != 1:
        raise ValueError("Unsupported capture manifest version")
    if manifest.get("indicator_id") != INDICATOR:
        raise ValueError("Capture manifest identifies the wrong indicator")
    if (manifest.get("semantics_verified") is not False
            or manifest.get("frozen_pairs_reconciled") is not False
            or manifest.get("candidate_data_endpoint_not_validated") is not True):
        raise ValueError("Capture manifest must retain all unverified-source flags")
    if manifest.get("candidate_period") != "2021":
        raise ValueError("Capture manifest has an unexpected candidate period")
    _check_utc_timestamp(manifest.get("captured_at_utc"))
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError("Capture manifest must contain exactly the two expected sources")
    return manifest


def review_capture(directory: Path) -> dict[str, Any]:
    """Validate both artifact entries and summarize structural observations.

    Captured JSON is *source-shape evidence only*. Never use this function
    to automatically create a verified INE reconciliation receipt.
    """
    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    if not directory.is_dir():
        raise ValueError("Capture directory does not exist")
    manifest = _read_manifest(directory)
    source_results: dict[str, dict[str, Any]] = {}
    saved_count = 0
    for kind, expected_url in SOURCES.items():
        entry = manifest["sources"][kind]
        if not isinstance(entry, dict):
            raise ValueError(f"{kind}: source entry must be a JSON object")
        status = entry.get("status")
        if status not in CAPTURE_STATUS:
            raise ValueError(f"{kind}: unsupported capture source status")
        if entry.get("requested_url") != expected_url:
            raise ValueError(f"{kind}: requested official endpoint has changed")
        filename = f"ine_{INDICATOR}_{kind}.json"
        source_file = directory / filename
        if status == "capture_failed":
            if source_file.exists() or source_file.is_symlink():
                raise ValueError(f"{kind}: failed capture must not include source bytes")
            if any(field in entry for field in ("sha256", "bytes", "filename", "response_url")):
                raise ValueError(f"{kind}: failed capture claims successful download fields")
            if not isinstance(entry.get("error_type"), str) or not isinstance(
                entry.get("error"), str
            ):
                raise ValueError(f"{kind}: failed capture lacks a diagnostic")
            source_results[kind] = {
                "status": "unavailable",
                "failure_type": entry["error_type"],
                "source_semantics_verified": False,
            }
            continue

        if entry.get("filename") != filename:
            raise ValueError(f"{kind}: source filename is not the expected fixed basename")
        response_url = entry.get("response_url")
        try:
            if _endpoint_kind(response_url) != kind:
                raise ValueError(f"{kind}: unexpected endpoint kind")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{kind}: response URL failed official endpoint check") from exc
        if not source_file.is_file() or source_file.is_symlink():
            raise ValueError(f"{kind}: captured JSON is missing or is a symlink")
        data = source_file.read_bytes()
        if not data or len(data) > MAX_BYTES:
            raise ValueError(f"{kind}: captured JSON has an invalid size")
        claimed_size = entry.get("bytes")
        if type(claimed_size) is not int or claimed_size != len(data):
            raise ValueError(f"{kind}: manifest byte count does not match source file")
        claimed_digest = entry.get("sha256")
        if (not isinstance(claimed_digest, str)
                or not SHA256_PATTERN.fullmatch(claimed_digest)
                or hashlib.sha256(data).hexdigest() != claimed_digest):
            raise ValueError(f"{kind}: source SHA-256 digest mismatch")
        decoded = _parse_json(data)
        outline = _schema_outline(decoded)
        if outline != entry.get("schema_outline"):
            raise ValueError(f"{kind}: structural fingerprint differs from manifest")
        source_results[kind] = {
            "status": "integrity_checked_semantics_unreviewed",
            "filename": filename,
            "sha256": claimed_digest,
            "bytes": len(data),
            "response_url": response_url,
            "schema_outline": outline,
            "source_semantics_verified": False,
        }
        saved_count += 1

    all_captured = saved_count == len(SOURCES)
    expected_status = "captured_unreconciled" if all_captured else "capture_incomplete"
    if manifest.get("status") != expected_status:
        raise ValueError("Overall capture status conflicts with source entries")

    return {
        "evidence_class": "offline_capture_artifact_integrity_review",
        "indicator_id": INDICATOR,
        "claimed_capture_timestamp_utc": manifest["captured_at_utc"],
        "status": "integrity_checked_unreconciled" if all_captured else "partial_capture_integrity_checked",
        "saved_source_files": saved_count,
        "expected_source_files": len(SOURCES),
        "source_semantics_verified": False,
        "frozen_pairs_reconciled": False,
        "INE_publisher_authenticated_independently": False,
        "sources": source_results,
        "next_gate": (
            "Review actual INE metadata, date/period and full geographic dimension definitions; "
            "verify source authenticity and map all 17 DICO municipalities; only then "
            "attempt independent 136-pair comparison"
        ),
    }


def run(directory: Path, output_json: Path) -> dict[str, Any]:
    """Save a deterministic summary outside the frozen research input set."""
    if not isinstance(output_json, Path):
        raise TypeError("output_json must be a Path")
    result = review_capture(directory)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Verify a downloaded/extracted capture artifact with no network traffic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.capture_dir, args.output_json)
    print(
        f"INE {INDICATOR} capture: {result['saved_source_files']}/"
        f"{result['expected_source_files']} response files passed integrity checks; "
        "source semantics remain unverified"
    )


if __name__ == "__main__":
    main()
