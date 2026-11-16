"""Assemble a fully evidenced 17-AMP INE category archive from bounded attempts.

INE can time out on individual read-only municipality queries. This utility
combines only *validated* official source responses from separately archived
GitHub Actions attempts. It retains each original receipt and response byte,
checks cross-attempt category consistency, and never invents missing counts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Sequence

from extract_commuting import municipality_codes_from_panel
from ine_residential_categories_probe import municipality_candidate_url
from ine_residential_municipal_capture import (
    CAPTURE_PROTOCOL,
    parse_municipality,
    validate_metadata,
)

ASSEMBLY_PROTOCOL = "ine_0011631_amp17_cross_attempt_source_assembly_v1"
PROBE_PROTOCOL = "ine_0011631_2021_porto_category_source_probe_v1"
SOURCE_KINDS = ("metadata", "porto_dim2_candidate")


def _sha(raw: bytes) -> str:
    """Hash original evidence bytes without canonicalizing their JSON."""
    return hashlib.sha256(raw).hexdigest()


def _stable_gzip(raw: bytes) -> bytes:
    """Compress source bytes with a fixed gzip header and no filename."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(raw)
    return buffer.getvalue()


def _checked_json(raw: bytes) -> dict[str, Any]:
    """Require a JSON object as an archived evidence receipt."""
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON evidence manifest object")
    return result


def _checked_attempt(
    name: str,
    candidate_dir: Path,
    municipal_dir: Path,
    names: dict[str, str],
) -> dict[str, Any]:
    """Verify exact metadata, candidate and all captured municipality receipts."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Attempt name must be a short, path-safe identifier")
    candidate_raw = (candidate_dir / "probe_manifest.json").read_bytes()
    probe = _checked_json(candidate_raw)
    metadata_raw = (candidate_dir / "ine_0011631_metadata.json").read_bytes()
    metadata = validate_metadata(metadata_raw, names)
    if (
        probe.get("protocol") != PROBE_PROTOCOL
        or probe.get("indicator_id") != "0011631"
        or probe.get("requested_kinds") != list(SOURCE_KINDS)
        or probe.get("source_semantics_verified") is not False
        or probe.get("directed_residential_origin_destination_counts_reconciled") is not False
    ):
        raise ValueError(f"Changed candidate scope in attempt {name}")
    probe_items = probe.get("items")
    if not isinstance(probe_items, list) or len(probe_items) != 2:
        raise ValueError(f"Incomplete official candidate receipts in attempt {name}")
    for kind, item, filename in zip(
        SOURCE_KINDS, probe_items,
        ("ine_0011631_metadata.json", "ine_0011631_porto_candidate.json"),
    ):
        if (
            not isinstance(item, dict)
            or item.get("kind") != kind
            or item.get("status") != "captured_unverified"
            or item.get("filename") != filename
        ):
            raise ValueError(f"Candidate source {kind} unavailable in attempt {name}")
        raw = (candidate_dir / filename).read_bytes()
        if _sha(raw) != item.get("source_sha256") or len(raw) != item.get("source_bytes"):
            raise ValueError(f"Candidate source checksum mismatch in attempt {name}")

    municipal_raw = (municipal_dir / "municipal_capture_manifest.json").read_bytes()
    municipal = _checked_json(municipal_raw)
    codes = municipal.get("requested_codes")
    items = municipal.get("items")
    if (
        municipal.get("protocol") != CAPTURE_PROTOCOL
        or not isinstance(codes, list)
        or not codes
        or codes != sorted(set(codes))
        or not set(codes) <= set(names)
        or municipal.get("requested_count") != len(codes)
        or municipal.get("metadata_source_sha256") != _sha(metadata_raw)
        or municipal.get("source_metadata") != metadata
        or municipal.get("full_residential_origin_destination_matrix_observed") is not False
        or municipal.get("housing_displacement_causality_established") is not False
        or not isinstance(items, list)
        or len(items) != len(codes)
    ):
        raise ValueError(f"Changed municipal source scope in attempt {name}")

    verified: dict[str, tuple[dict[str, Any], bytes, dict[str, int]]] = {}
    for code, item in zip(codes, items):
        if (
            not isinstance(item, dict)
            or item.get("code") != code
            or item.get("municipality") != names[code]
            or item.get("source_url") != municipality_candidate_url(code)
        ):
            raise ValueError(f"Wrong source municipality or URL in attempt {name}")
        if item.get("status") == "failed":
            # An unvalidated response is retained in the original Actions
            # artifact; never silently convert it into a numeric observation.
            continue
        filename = f"ine_0011631_dim2_{code}_2021.json"
        if (
            item.get("status") != "validated_2021_migration_categories"
            or item.get("filename") != filename
            or item.get("response_url") != municipality_candidate_url(code)
            or item.get("reference_period") != "2021"
            or item.get("municipal_origin_identified") is not False
        ):
            raise ValueError(f"Invalid source classification in attempt {name}")
        raw = (municipal_dir / filename).read_bytes()
        if _sha(raw) != item.get("source_sha256") or len(raw) != item.get("source_bytes"):
            raise ValueError(f"Original source checksum mismatch for {name}:{code}")
        counts = parse_municipality(raw, code, names)
        if counts != item.get("both_sexes_counts"):
            raise ValueError(f"Original source totals changed for {name}:{code}")
        verified[code] = (item, raw, counts)
    if municipal.get("validated_count") != len(verified):
        raise ValueError(f"Wrong verified source count in attempt {name}")
    return {
        "name": name,
        "metadata_raw": metadata_raw,
        "probe_raw": candidate_raw,
        "porto_candidate_raw": (candidate_dir / "ine_0011631_porto_candidate.json").read_bytes(),
        "municipal_raw": municipal_raw,
        "metadata": metadata,
        "municipal": municipal,
        "verified": verified,
    }


def assemble(
    attempts: Sequence[tuple[str, Path, Path]],
    destination: Path,
) -> dict[str, Any]:
    """Write complete auditable source set; fail if any AMP source is missing."""
    if not attempts or len(attempts) > 10 or not isinstance(destination, Path):
        raise ValueError("Supply 1–10 bounded attempts and an output directory")
    names = municipality_codes_from_panel()
    if len({name for name, _, _ in attempts}) != len(attempts):
        raise ValueError("Source attempt names must be unique")
    sources = [
        _checked_attempt(name, candidate, municipal, names)
        for name, candidate, municipal in attempts
    ]
    selected: dict[str, tuple[str, dict[str, Any], bytes]] = {}
    observed: dict[str, dict[str, int]] = {}
    for source in sources:
        for code, (item, raw, counts) in source["verified"].items():
            if code in observed and observed[code] != counts:
                raise ValueError(f"Official category counts disagree between attempts for {code}")
            observed[code] = counts
            selected.setdefault(code, (source["name"], item, raw))
    missing = sorted(set(names) - set(selected))
    if missing:
        raise ValueError(f"Missing validated INE municipality categories: {', '.join(missing)}")

    # Assemble only after the complete source set passed exact-byte checks.
    destination.mkdir(parents=True, exist_ok=True)
    evidence = destination / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    metadata_raw = sources[0]["metadata_raw"]
    (destination / "ine_0011631_metadata.json.gz").write_bytes(
        _stable_gzip(metadata_raw)
    )
    receipt_records: list[dict[str, Any]] = []
    for source in sources:
        name = source["name"]
        (evidence / f"{name}_probe_manifest.json").write_bytes(source["probe_raw"])
        (evidence / f"{name}_municipal_capture_manifest.json").write_bytes(
            source["municipal_raw"]
        )
        (evidence / f"{name}_metadata.json.gz").write_bytes(
            _stable_gzip(source["metadata_raw"])
        )
        (evidence / f"{name}_porto_candidate.json").write_bytes(
            source["porto_candidate_raw"]
        )
        receipt_records.append({
            "attempt": name,
            "probe_manifest_sha256": _sha(source["probe_raw"]),
            "municipal_manifest_sha256": _sha(source["municipal_raw"]),
            "metadata_source_sha256": _sha(source["metadata_raw"]),
            "validated_codes": sorted(source["verified"]),
        })

    items: list[dict[str, Any]] = []
    for code in sorted(names):
        name, item, raw = selected[code]
        (destination / item["filename"]).write_bytes(raw)
        items.append({**item, "selected_from_attempt": name})
    composite: dict[str, Any] = {
        "protocol": CAPTURE_PROTOCOL,
        "archive_assembly_protocol": ASSEMBLY_PROTOCOL,
        "captured_at_utc": sources[0]["municipal"].get("captured_at_utc"),
        "source_metadata": sources[0]["metadata"],
        "metadata_source_sha256": _sha(metadata_raw),
        "requested_codes": sorted(names),
        "requested_count": 17,
        "validated_count": 17,
        "data_semantics": "2021 Census residence/migration categories, not named origin-to-destination flows",
        "full_residential_origin_destination_matrix_observed": False,
        "housing_displacement_causality_established": False,
        "source_attempt_receipts": receipt_records,
        "selection_rule": "first validated exact source per municipality; cross-attempt totals must agree",
        "items": items,
    }
    (destination / "municipal_capture_manifest.json").write_text(
        json.dumps(composite, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return composite


def main(argv: Sequence[str] | None = None) -> None:
    """Assemble source artifacts downloaded from approved read-only Actions runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt", nargs=3, action="append", required=True,
        metavar=("RUN_ATTEMPT", "CANDIDATE_DIR", "MUNICIPAL_DIR"),
        help="One originally archived GitHub run attempt and its two artifact directories",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    attempts = [(name, Path(candidate), Path(municipal))
                for name, candidate, municipal in args.attempt]
    manifest = assemble(attempts, args.output)
    print(f"Archived {manifest['validated_count']}/17 verified residential category destinations "
          f"from {len(attempts)} official source attempts; no named OD observed")


if __name__ == "__main__":
    main()
