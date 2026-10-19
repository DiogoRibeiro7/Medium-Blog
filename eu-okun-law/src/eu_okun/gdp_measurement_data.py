"""Frozen inputs of ``GDP_MEASUREMENT_PROTOCOL.md``: value added and hours, in total and
for finance and insurance (NACE K), from the Eurostat quarterly national accounts.

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
from eu_okun.heterogeneity_data import _read, _sha256, _write, decode_available

START, END = "2008-Q1", "2026-Q2"
QUARTERLY_FILE = "quarterly.csv"


def _va(key: str, unit: str, nace: str) -> SeriesSpec:
    return SeriesSpec(key, "namq_10_a10", key, {
        "freq": "Q", "unit": unit, "s_adj": "SCA", "na_item": "B1G", "nace_r2": nace})


def _hours(key: str, nace: str) -> SeriesSpec:
    return SeriesSpec(key, "namq_10_a10_e", key, {
        "freq": "Q", "unit": "THS_HW", "s_adj": "SCA", "na_item": "EMP_DC", "nace_r2": nace})


SERIES: dict[str, SeriesSpec] = {
    "va_clv": _va("va_clv", "CLV20_MEUR", "TOTAL"),
    "va_clv_k": _va("va_clv_k", "CLV20_MEUR", "K"),
    "va_cp": _va("va_cp", "CP_MEUR", "TOTAL"),
    "va_cp_k": _va("va_cp_k", "CP_MEUR", "K"),
    "hours": _hours("hours", "TOTAL"),
    "hours_k": _hours("hours_k", "K"),
}


def _rows(cells: dict[str, dict[tuple[str, str], Observation]]) -> list[dict[str, str]]:
    rows = []
    for geo in EU27:
        for q in quarter_sequence(START, END):
            row = {"geo": geo, "quarter": q}
            for key in SERIES:
                cell = cells[key].get((geo, q), Observation(None, ""))
                row[key] = "" if cell.value is None else repr(cell.value)
                row[f"{key}_flag"] = cell.status
            rows.append(row)
    return rows


def fetch_gdp_measurement(output_dir: Path) -> dict[str, Any]:
    """Download every series, freeze the raw responses and write the processed CSV."""
    raw_dir, processed = output_dir / "raw", output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    files, cells = [], {}
    for key, spec in SERIES.items():
        raw, payload, url = fetch_dataset(spec, START, END)
        path = raw_dir / f"{key}.json"
        path.write_bytes(raw)
        cells[key] = decode_available(payload, spec)
        files.append({"key": key, "dataset": spec.dataset, "url": url,
                      "updated": str(payload.get("updated", "")), "sha256": _sha256(path),
                      "countries": sorted({g for g, _ in cells[key]})})
    _write(processed / QUARTERLY_FILE, _rows(cells))
    manifest = {
        "description": "Inputs of GDP_MEASUREMENT_PROTOCOL.md",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "period": {"start": START, "end": END},
        "files": files,
        "processed": {QUARTERLY_FILE: _sha256(processed / QUARTERLY_FILE)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    return manifest


def audit_gdp_measurement(directory: Path) -> dict[str, Any]:
    """Check raw hashes and re-derive the processed CSV from the raw responses."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text(
        encoding="utf-8"))
    for entry in manifest["files"]:
        if _sha256(directory / "raw" / f"{entry['key']}.json") != entry["sha256"]:
            raise EurostatDataError(f"{entry['key']}.json: SHA-256 mismatch.")
    if _sha256(directory / "processed" / QUARTERLY_FILE) != manifest["processed"][
            QUARTERLY_FILE]:
        raise EurostatDataError(f"{QUARTERLY_FILE} does not match its manifest hash.")
    cells = {key: decode_available(json.loads((directory / "raw" / f"{key}.json").read_bytes()),
                                   spec) for key, spec in SERIES.items()}
    if _read(directory / "processed" / QUARTERLY_FILE) != _rows(cells):
        raise EurostatDataError(f"{QUARTERLY_FILE} does not re-derive from the raw responses.")
    return manifest


def load_gdp_measurement(directory: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    """Values by (country, quarter); missing cells are None."""
    return {(row["geo"], row["quarter"]): {key: float(row[key]) if row[key] else None
                                           for key in SERIES}
            for row in _read(directory / "processed" / QUARTERLY_FILE)}
