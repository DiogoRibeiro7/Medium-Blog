"""Network acquisition, complete panel assembly and provenance snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from eu_okun.contract import (
    EU27,
    SERIES,
    EurostatDataError,
    Observation,
    SeriesSpec,
    quarter_sequence,
    request_url,
)
from eu_okun.jsonstat import decode_series


def fetch_dataset(spec: SeriesSpec, start: str, end: str) -> tuple[bytes, dict[str, Any], str]:
    """Download original Eurostat JSON-stat bytes (no transformations)."""
    url = request_url(spec, start, end)
    request = Request(url, headers={"User-Agent": "Medium-Blog/eu-okun-law"})
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Eurostat download failed for {spec.dataset}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EurostatDataError(f"Invalid JSON from {spec.dataset}.") from exc
    if not isinstance(payload, dict):
        raise EurostatDataError(f"Invalid JSON root in {spec.dataset}.")
    return raw, payload, url


def build_panel(
    observations: Mapping[str, Mapping[tuple[str, str], Observation]],
    start: str, end: str, countries: Sequence[str] = EU27,
) -> list[dict[str, str | float | None]]:
    """Left join every series onto the full country-by-quarter grid."""
    if set(observations) != {spec.name for spec in SERIES}:
        raise EurostatDataError("All three indicators are required for panel construction.")
    rows: list[dict[str, str | float | None]] = []
    for country in countries:
        for quarter in quarter_sequence(start, end):
            row: dict[str, str | float | None] = {"geo": country, "quarter": quarter}
            for spec in SERIES:
                point = observations[spec.name].get((country, quarter), Observation(None, ""))
                row[spec.column], row[f"{spec.name}_flag"] = point.value, point.status
            rows.append(row)
    return rows


def save_snapshot(
    destination: Path, start: str = "2008-Q1", end: str = "2026-Q2",
) -> dict[str, Any]:
    """Validate all sources, then persist CSV, original bytes and manifest."""
    quarter_sequence(start, end)
    series: dict[str, dict[tuple[str, str], Observation]] = {}
    downloaded: dict[str, bytes] = {}
    provenance: dict[str, dict[str, str | int]] = {}
    for spec in SERIES:
        raw, payload, url = fetch_dataset(spec, start, end)
        cells = decode_series(payload, spec)
        series[spec.name] = cells
        downloaded[spec.name] = raw
        provenance[spec.name] = {
            "dataset": spec.dataset, "url": url,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "updated": str(payload.get("updated", "")),
            "nonmissing_cells": sum(cell.value is not None for cell in cells.values()),
            "flagged_cells": sum(bool(cell.status) for cell in cells.values()),
        }
    rows = build_panel(series, start, end)
    fields = ["geo", "quarter"]
    for spec in SERIES:
        fields.extend([spec.column, f"{spec.name}_flag"])
    manifest: dict[str, Any] = {
        "source": "Eurostat Statistics API (JSON-stat 2.0)",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "period": {"start": start, "end": end, "analysis_start": "2009-Q1"},
        "countries": list(EU27), "country_quarter_rows": len(rows),
        "series": provenance,
        "missing_cells": {
            spec.name: sum(row[spec.column] is None for row in rows) for spec in SERIES
        },
        "note": "Observations can be revised; flagged and missing values are not imputed.",
    }

    raw_dir, processed = destination / "raw", destination / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    for spec in SERIES:
        (raw_dir / f"{spec.dataset}.json").write_bytes(downloaded[spec.name])
    with (processed / "eu27_quarterly.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
