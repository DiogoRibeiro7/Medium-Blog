"""Audit a future INE Census 2021 *residential* municipality OD release.

No directed residential origin-destination data are bundled with this study.
This module validates an explicitly supplied, privacy-safe *aggregate* table
against the observed Census 2021 Porto and AMP destination margins. A user's
source metadata and checksum are necessary but do not independently certify
that the release originated from INE.

Input CSV: origin_dico,destination_dico,count,disclosure_status
Origin: one of the 17 AMP DICO municipalities or REST_PT (other Portugal).
Destination: one of the same 17 AMP municipalities. A complete 18x17 grid
contains structural-zero diagonal cells, observed nonnegative counts and/or
explicitly suppressed cells with blank count. No personal records are accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from extract_commuting import municipality_codes_from_panel
from residential_identifiability import DestinationMargins, load_margins


OUTSIDE_AMP = "REST_PT"
CSV_FIELDS = ("origin_dico", "destination_dico", "count", "disclosure_status")
DISCRETE_STATUSES = frozenset({"observed", "suppressed", "structural_zero"})
CONTRACT_VERSION = "census2021_residential_od_amp_v1"
REFERENCE_START = "2019-12-31"
REFERENCE_END = "2021-04-19"


@dataclass(frozen=True, slots=True)
class ODCell:
    """One normalized, validated *aggregate* origin-destination cell."""

    origin: str
    destination: str
    count: int | None
    status: str


@dataclass(frozen=True, slots=True)
class GroupBounds:
    """Sharp count bounds within a group with an observed destination margin."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower < 0 or self.upper < self.lower:
            raise ValueError("Invalid nonnegative partial-identification interval")


def amp_codes() -> tuple[str, ...]:
    """Read exact 17 stable DICO identifiers from the pinned 2019–2025 panel."""
    codes = tuple(sorted(municipality_codes_from_panel()))
    if len(codes) != 17 or "1312" not in codes:
        raise ValueError("Incomplete 17-municipality AMP geography")
    return codes


def read_cells(path: Path, *, municipality_codes: Sequence[str]) -> list[ODCell]:
    """Require a complete canonical grid, reject missing and invented values."""
    destination_codes = set(municipality_codes)
    origin_codes = destination_codes | {OUTSIDE_AMP}
    expected_pairs = {(o, d) for o in origin_codes for d in destination_codes}
    cells: dict[tuple[str, str], ODCell] = {}

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("Unexpected directed Census OD CSV schema; personal fields forbidden")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed or extra CSV fields at line {line_number}")
            origin, dest = row["origin_dico"], row["destination_dico"]
            raw_count, status = row["count"], row["disclosure_status"]
            if (origin, dest) not in expected_pairs:
                raise ValueError(f"Unknown origin/destination DICO pair at line {line_number}")
            if (origin, dest) in cells:
                raise ValueError(f"Duplicate residential OD cell at line {line_number}")
            if status not in DISCRETE_STATUSES:
                raise ValueError(f"Unknown disclosure status at line {line_number}")
            if origin == dest:
                if status != "structural_zero" or raw_count != "0":
                    raise ValueError("Same-municipality OD diagonal must be structural zero")
                count: int | None = 0
            elif status == "structural_zero":
                raise ValueError("Off-diagonal cells must never be declared structural zero")
            elif status == "suppressed":
                if raw_count != "":
                    raise ValueError("Suppressed cell must have an empty count, not zero")
                count = None
            else:
                if not raw_count or not re.fullmatch(r"0|[1-9][0-9]*", raw_count):
                    raise ValueError(f"Invalid published nonnegative integer at line {line_number}")
                count = int(raw_count)
            cells[(origin, dest)] = ODCell(origin, dest, count, status)

    if set(cells) != expected_pairs:
        missing = len(expected_pairs - set(cells))
        raise ValueError(f"Incomplete 18x17 residential OD grid: {missing} missing cells")
    return [cells[key] for key in sorted(cells)]


def read_metadata(path: Path, *, source_bytes: bytes) -> dict[str, object]:
    """Check declared source identity, universe and pinned release bytes."""
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Source metadata must be a JSON object")
    required = {
        "contract_version": CONTRACT_VERSION,
        "publisher": "INE",
        "reference_start": REFERENCE_START,
        "census_reference": REFERENCE_END,
        "population_universe": "2021 usual residents with previous residence in another Portuguese municipality",
        "origin_scope": "AMP17_PLUS_REST_PT",
        "destination_scope": "AMP17",
        "unit": "persons",
        "domestic_previous_residence_only": True,
    }
    for field, expected in required.items():
        if type(meta.get(field)) is not type(expected) or meta.get(field) != expected:
            raise ValueError(f"Incorrect source metadata field: {field}")
    for field in ("source_title", "source_citation", "license", "disclosure_policy"):
        if not isinstance(meta.get(field), str) or not meta[field].strip():
            raise ValueError(f"Missing source metadata: {field}")
    digest = hashlib.sha256(source_bytes).hexdigest()
    if meta.get("source_csv_sha256") != digest:
        raise ValueError("Supplied directed OD CSV does not match its claimed SHA-256")
    return meta


def _bounds_for_group(
    rows: Sequence[ODCell], *, margin: int, target_origin: set[str],
) -> GroupBounds:
    """Bound a subset's count given source cells and a known destination sum.

    Residual mass must occupy suppressed cells. If suppression affects only
    target (or only other) origins, its destination total is still identified.
    """
    known_total = sum(c.count for c in rows if c.count is not None)
    if known_total > margin:
        raise ValueError("Observed OD cells exceed an official destination margin")
    unknown = [c for c in rows if c.count is None]
    if not unknown and known_total != margin:
        raise ValueError("Complete published OD grid does not reconcile with the Census margin")
    residual = margin - known_total
    known_target = sum(c.count for c in rows if c.count is not None and c.origin in target_origin)
    suppressed_target = any(c.origin in target_origin for c in unknown)
    suppressed_other = any(c.origin not in target_origin for c in unknown)
    lower = known_target + (residual if suppressed_target and not suppressed_other else 0)
    upper = known_target + (residual if suppressed_target else 0)
    return GroupBounds(lower, upper)


def audit(
    cells: Sequence[ODCell], *, margins: DestinationMargins,
    codes: Sequence[str], source_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Bound Porto↔other AMP exchange, respecting explicit cell suppressions.

    Without named origin information the earlier sharp rectangle remains the
    baseline. A disclosed OD release can tighten it; point identification
    requires adequate cells, but never identifies a *reason* for relocating.
    """
    porto = "1312"
    other_amp = set(codes) - {porto}
    porto_dest = [c for c in cells if c.destination == porto]
    rest_dest = [c for c in cells if c.destination != porto]
    if len(porto_dest) != 18 or len(rest_dest) != 18 * 16:
        raise ValueError("Incomplete municipality destination grouping")

    x = _bounds_for_group(rest_dest, margin=margins.rest_amp, target_origin={porto})
    y = _bounds_for_group(porto_dest, margin=margins.porto, target_origin=other_amp)
    # Independent destination partitions make these net bounds sharp under
    # the published Porto and AMP margins and the declared suppression mask.
    net_lower, net_upper = x.lower - y.upper, x.upper - y.lower
    total_suppressed = sum(c.status == "suppressed" for c in cells)
    return {
        "evidence_class": "aggregate_residential_od_intake_validation_not_causal_inference",
        "source": {
            "publisher": source_metadata["publisher"],
            "title": source_metadata["source_title"],
            "citation": source_metadata["source_citation"],
            "source_csv_sha256": source_metadata["source_csv_sha256"],
            "license": source_metadata["license"],
        },
        "reference_start": REFERENCE_START,
        "census_reference": REFERENCE_END,
        "municipalities": len(codes),
        "cells": len(cells),
        "suppressed_cells": total_suppressed,
        "official_destination_margins": {
            "porto_other_municipality": margins.porto,
            "amp_other_municipality": margins.amp,
            "rest_amp_other_municipality": margins.rest_amp,
        },
        "porto_to_other_amp_bounds": asdict(x),
        "other_amp_to_porto_bounds": asdict(y),
        "net_toward_other_amp_bounds": {"lower": net_lower, "upper": net_upper},
        "direction_point_identified": x.lower == x.upper and y.lower == y.upper,
        "caveats": [
            "A verified official source and correct disclosure treatment are required; claimed metadata alone cannot prove origin authenticity.",
            "Cells count individuals resident in AMP in 2021, not departures to other regions or subsequent moves.",
            "This is not evidence that housing costs caused a move; no price, tenure or reason variables are present.",
            "Suppressed cells remain unknown, not zero. Bounds use the two published aggregate destination margins only.",
        ],
    }


def run(source_csv: Path, metadata_json: Path, output_json: Path) -> dict[str, object]:
    """Validate a supplied release and save only derived, aggregate diagnostics."""
    source_bytes = source_csv.read_bytes()
    metadata = read_metadata(metadata_json, source_bytes=source_bytes)
    codes = amp_codes()
    cells = read_cells(source_csv, municipality_codes=codes)
    result = audit(cells, margins=load_margins(), codes=codes, source_metadata=metadata)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True, help="INE-authorized aggregate OD CSV")
    parser.add_argument("--metadata-json", type=Path, required=True, help="Companion source and release contract")
    parser.add_argument("--output-json", type=Path, required=True, help="Destination of the derived audit")
    args = parser.parse_args(argv)
    result = run(args.source_csv, args.metadata_json, args.output_json)
    print("Validated source counts, suppressed cells and conditional residential flow bounds")
    print(json.dumps(result["net_toward_other_amp_bounds"]))


if __name__ == "__main__":
    main()
