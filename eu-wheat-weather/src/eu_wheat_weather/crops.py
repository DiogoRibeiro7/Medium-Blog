"""Eurostat crop statistics for common wheat by NUTS-2 region (METHODS.md, section 3.1).

``fetch_crops`` downloads the raw JSON-stat response; ``wheat_table`` turns it into one
row per region and year, with area, production, the yield computed from them and the
Eurostat flags. ``save_snapshot`` freezes the raw bytes and the table with SHA-256
hashes; ``audit_snapshot`` re-derives the table from the raw bytes and checks both.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eu_wheat_weather.contract import (
    AREA,
    CROP,
    CROPS_DATASET,
    EU27,
    EUROSTAT_API,
    FIRST_YEAR,
    LAST_YEAR,
    MEASURES,
    PRODUCTION,
    DataError,
)
from eu_wheat_weather.jsonstat import decode

RAW_NAME = f"raw/{CROPS_DATASET}_{CROP}_nuts2.json"
TABLE_NAME = "processed/wheat_nuts2.csv"
COLUMNS = ("geo", "year", "area_kha", "production_kt", "yield_t_ha", "area_flag",
           "production_flag")


def crops_url() -> str:
    query = [("format", "JSON"), ("lang", "EN"), ("crops", CROP),
             *(("strucpro", m) for m in MEASURES), ("geoLevel", "nuts2")]
    return f"{EUROSTAT_API}{CROPS_DATASET}?{urllib.parse.urlencode(query)}"


def fetch_crops(timeout: float = 300.0) -> tuple[bytes, str]:
    """The raw response and the URL it came from."""
    url = crops_url()
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read(), url


@dataclass(frozen=True)
class WheatRow:
    """One EU-27 NUTS-2 region and harvest year."""

    geo: str
    year: int
    area_kha: float | None
    production_kt: float | None
    area_flag: str
    production_flag: str

    @property
    def yield_t_ha(self) -> float | None:
        """Production over area; undefined without both, or without a positive area."""
        if self.area_kha is None or self.production_kt is None or self.area_kha <= 0:
            return None
        return self.production_kt / self.area_kha


def is_eu_nuts2(code: str) -> bool:
    """A NUTS-2 code of an EU-27 country (extra-regio codes ``..ZZ`` excluded)."""
    return len(code) == 4 and code[:2] in EU27 and not code.endswith("ZZ")


def wheat_table(payload: dict[str, Any]) -> list[WheatRow]:
    """One row per EU-27 NUTS-2 region and year in the study period with any cell."""
    cells = decode(payload, {"freq": "A", "crops": CROP, "strucpro": MEASURES})
    by_key: dict[tuple[str, int], dict[str, tuple[float | None, str]]] = {}
    for cell in cells:
        geo, period = cell.codes["geo"], cell.codes["time"]
        if not period.isdigit():
            raise DataError(f"Unexpected period {period}.")
        year = int(period)
        if not is_eu_nuts2(geo) or not FIRST_YEAR <= year <= LAST_YEAR:
            continue
        by_key.setdefault((geo, year), {})[cell.codes["strucpro"]] = (cell.value, cell.flag)
    rows = []
    for (geo, year), measures in sorted(by_key.items()):
        area, area_flag = measures.get(AREA, (None, ""))
        production, production_flag = measures.get(PRODUCTION, (None, ""))
        rows.append(WheatRow(geo, year, area, production, area_flag, production_flag))
    return rows


def _number(x: float | None) -> str:
    return "" if x is None else repr(x)


def table_text(rows: list[WheatRow]) -> str:
    """The processed table as CSV text (LF line endings, full float precision)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(COLUMNS)
    for r in rows:
        writer.writerow([r.geo, r.year, _number(r.area_kha), _number(r.production_kt),
                         _number(r.yield_t_ha), r.area_flag, r.production_flag])
    return out.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_snapshot(directory: Path, raw: bytes, url: str,
                  retrieved: datetime | None = None) -> dict[str, Any]:
    """Write the raw response, the processed table and a manifest with their hashes."""
    payload = json.loads(raw.decode("utf-8"))
    rows = wheat_table(payload)
    table = table_text(rows).encode("utf-8")
    files = {RAW_NAME: raw, TABLE_NAME: table}
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {
        "dataset": CROPS_DATASET, "crop": CROP, "measures": list(MEASURES), "url": url,
        "retrieved_utc": (retrieved or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eurostat_updated": payload.get("updated"),
        "files": {name: sha256(data) for name, data in files.items()},
        "summary": summary(rows),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    return manifest


def summary(rows: list[WheatRow]) -> dict[str, int]:
    """Counts that make a snapshot recognisable; no yield level is summarised."""
    with_yield = [r for r in rows if r.yield_t_ha is not None]
    return {"rows": len(rows), "rows_with_yield": len(with_yield),
            "regions_with_yield": len({r.geo for r in with_yield}),
            "first_year": min((r.year for r in rows), default=0),
            "last_year": max((r.year for r in rows), default=0)}


def audit_snapshot(directory: Path) -> dict[str, Any]:
    """Check every file against the manifest and re-derive the table from the raw bytes."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.exists():
            raise DataError(f"Missing snapshot file {name}.")
        if sha256(path.read_bytes()) != expected:
            raise DataError(f"{name} does not match its recorded SHA-256.")
    raw = (directory / RAW_NAME).read_bytes()
    rows = wheat_table(json.loads(raw.decode("utf-8")))
    if table_text(rows).encode("utf-8") != (directory / TABLE_NAME).read_bytes():
        raise DataError("The processed table does not re-derive from the raw response.")
    counts = summary(rows)
    if counts != manifest["summary"]:
        raise DataError("The manifest summary does not match the table.")
    return dict(counts)
