"""Frozen inputs of ``AGGREGATE_PROTOCOL.md``: unemployed persons and unemployment rates
(for the labour-force weights and the official EU rate) and GDP growth (for the official
EU growth rate), for the 27 countries and ``EU27_2020``.

The raw JSON-stat responses are stored byte for byte with SHA-256 hashes, and the
processed table is re-derived from them by the audit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eu_okun.acquisition import fetch_dataset
from eu_okun.contract import EU27, EurostatDataError, Observation, SeriesSpec, quarter_sequence
from eu_okun.heterogeneity_data import _read, _sha256, _write
from eu_okun.jsonstat import decode_series

START, END = "2008-Q1", "2026-Q2"
EU = "EU27_2020"
GEOS: tuple[str, ...] = (*EU27, EU)
QUARTERLY_FILE = "quarterly.csv"

SERIES: dict[str, SeriesSpec] = {
    "unemployed_ths": SeriesSpec("unemployed_ths", "une_rt_q", "unemployed_ths", {
        "freq": "Q", "age": "Y15-74", "sex": "T", "unit": "THS_PER", "s_adj": "SA"}),
    "rate": SeriesSpec("rate", "une_rt_q", "rate", {
        "freq": "Q", "age": "Y15-74", "sex": "T", "unit": "PC_ACT", "s_adj": "SA"}),
    "gdp": SeriesSpec("gdp", "namq_10_gdp", "gdp", {
        "freq": "Q", "unit": "CLV_PCH_PRE", "s_adj": "SCA", "na_item": "B1GQ"}),
}


def _decode(payload: dict[str, Any], spec: SeriesSpec) -> dict[tuple[str, str], Observation]:
    geo = payload.get("dimension", {}).get("geo", {}).get("category", {}).get("index", {})
    missing = [g for g in GEOS if g not in geo]
    if missing:
        raise EurostatDataError(f"{spec.dataset} does not publish {', '.join(missing)}.")
    return decode_series(payload, spec, countries=list(GEOS))


def _rows(cells: dict[str, dict[tuple[str, str], Observation]]) -> list[dict[str, str]]:
    rows = []
    for geo in GEOS:
        for q in quarter_sequence(START, END):
            row = {"geo": geo, "quarter": q}
            for key in SERIES:
                cell = cells[key].get((geo, q), Observation(None, ""))
                row[key] = "" if cell.value is None else repr(cell.value)
                row[f"{key}_flag"] = cell.status
            rows.append(row)
    return rows


def fetch_eu_aggregate(output_dir: Path) -> dict[str, Any]:
    """Download every series, freeze the raw responses and write the processed CSV."""
    raw_dir, processed = output_dir / "raw", output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    files, cells = [], {}
    for key, spec in SERIES.items():
        raw, payload, url = fetch_dataset(spec, START, END)
        path = raw_dir / f"{key}.json"
        path.write_bytes(raw)
        cells[key] = _decode(payload, spec)
        files.append({"key": key, "dataset": spec.dataset, "url": url,
                      "updated": str(payload.get("updated", "")), "sha256": _sha256(path)})
    _write(processed / QUARTERLY_FILE, _rows(cells))
    manifest = {
        "description": "Inputs of AGGREGATE_PROTOCOL.md",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "period": {"start": START, "end": END},
        "geos": list(GEOS),
        "files": files,
        "processed": {QUARTERLY_FILE: _sha256(processed / QUARTERLY_FILE)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    return manifest


def audit_eu_aggregate(directory: Path) -> dict[str, Any]:
    """Check raw hashes and re-derive the processed CSV from the raw responses."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text(
        encoding="utf-8"))
    for entry in manifest["files"]:
        if _sha256(directory / "raw" / f"{entry['key']}.json") != entry["sha256"]:
            raise EurostatDataError(f"{entry['key']}.json: SHA-256 mismatch.")
    if _sha256(directory / "processed" / QUARTERLY_FILE) != manifest["processed"][
            QUARTERLY_FILE]:
        raise EurostatDataError(f"{QUARTERLY_FILE} does not match its manifest hash.")
    cells = {key: _decode(json.loads((directory / "raw" / f"{key}.json").read_bytes()), spec)
             for key, spec in SERIES.items()}
    if _read(directory / "processed" / QUARTERLY_FILE) != _rows(cells):
        raise EurostatDataError(f"{QUARTERLY_FILE} does not re-derive from the raw responses.")
    return manifest


def load_eu_aggregate(directory: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    """Values by (geo, quarter); missing cells are None."""
    return {(row["geo"], row["quarter"]): {key: float(row[key]) if row[key] else None
                                           for key in SERIES}
            for row in _read(directory / "processed" / QUARTERLY_FILE)}
