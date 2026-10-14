"""Unit tests of the frozen-snapshot audit invariants."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from eu_okun.audit import audit_snapshot
from eu_okun.contract import EU27, SERIES, EurostatDataError


def fixture_snapshot(tmp_path: Path) -> Path:
    """Build a complete, small and traceable EU-27 fixture."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    sources = {}
    for spec in SERIES:
        content = b"{}"
        (raw / f"{spec.dataset}.json").write_bytes(content)
        sources[spec.name] = {
            "dataset": spec.dataset, "sha256": hashlib.sha256(content).hexdigest()
        }
    (tmp_path / "snapshot_manifest.json").write_text(json.dumps({
        "countries": list(EU27), "period": {"start": "2009-Q1", "end": "2009-Q2"},
        "series": sources,
    }))
    columns = ["geo", "quarter"]
    for spec in SERIES:
        columns.extend([spec.column, f"{spec.name}_flag"])
    with (processed / "eu27_quarterly.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for geo in EU27:
            for quarter in ("2009-Q1", "2009-Q2"):
                row = {"geo": geo, "quarter": quarter}
                for spec in SERIES:
                    row[spec.column] = 1.0
                    row[f"{spec.name}_flag"] = ""
                writer.writerow(row)
    return tmp_path


def test_complete_panel_and_consecutive_pairs(tmp_path: Path) -> None:
    report = audit_snapshot(fixture_snapshot(tmp_path))
    assert report["country_quarter_rows"] == 54
    assert len(report["by_country"]) == 27
    assert report["by_country"][0]["basic_okun_pairs"] == 1
    assert report["by_country"][0]["with_participation_pairs"] == 1


def test_tampered_raw_snapshot_is_rejected(tmp_path: Path) -> None:
    directory = fixture_snapshot(tmp_path)
    (directory / "raw" / "namq_10_gdp.json").write_bytes(b"changed")
    with pytest.raises(EurostatDataError, match="changed"):
        audit_snapshot(directory)


def test_missing_country_quarter_row_is_rejected(tmp_path: Path) -> None:
    directory = fixture_snapshot(tmp_path)
    path = directory / "processed" / "eu27_quarterly.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(EurostatDataError, match="incomplete"):
        audit_snapshot(directory)
