"""Eurostat crop statistics by region: common wheat (METHODS.md, section 3.1) and grain
maize (section 9). The crop is a parameter; it defaults to wheat.

``fetch_crops`` downloads the raw JSON-stat responses (NUTS-2 and NUTS-1);
``wheat_table`` turns them into one row per region and year, with area, production, the
yield computed from them and the Eurostat flags. ``save_snapshot`` freezes the raw bytes
and the table with SHA-256 hashes; ``audit_snapshot`` re-derives the table from the raw
bytes and checks both.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eu_wheat_weather.contract import (
    AREA,
    CROP,
    CROP_NAMES,
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

# NUTS-1 is fetched because some regions are published only there (Brandenburg,
# Sachsen-Anhalt); a NUTS-1 region with a single NUTS-2 region is that region
# (METHODS.md, change log).
LEVELS: tuple[str, ...] = ("nuts2", "nuts1")
TABLE_NAME = "processed/wheat_nuts2.csv"


def table_name(crop: str = CROP) -> str:
    """processed/wheat_nuts2.csv for wheat, processed/maize_nuts2.csv for maize."""
    return f"processed/{CROP_NAMES[crop]}_nuts2.csv"


def snapshot_crop(directory: Path) -> str:
    """The crop a snapshot holds, from its manifest."""
    return str(json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["crop"])
COLUMNS = ("geo", "year", "area_kha", "production_kt", "yield_t_ha", "area_flag",
           "production_flag")


def raw_name(level: str, crop: str = CROP) -> str:
    return f"raw/{CROPS_DATASET}_{crop}_{level}.json"


def crops_url(level: str = "nuts2", crop: str = CROP) -> str:
    query = [("format", "JSON"), ("lang", "EN"), ("crops", crop),
             *(("strucpro", m) for m in MEASURES), ("geoLevel", level)]
    return f"{EUROSTAT_API}{CROPS_DATASET}?{urllib.parse.urlencode(query)}"


def fetch_crops(levels: tuple[str, ...] = LEVELS, timeout: float = 300.0,
                crop: str = CROP) -> dict[str, tuple[bytes, str]]:
    """For each NUTS level, the raw response and the URL it came from."""
    out = {}
    for level in levels:
        url = crops_url(level, crop)
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            out[level] = (response.read(), url)
    return out


@dataclass(frozen=True)
class WheatRow:
    """One EU-27 region (NUTS-2, or NUTS-1 where published there) and harvest year."""

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


def is_eu_region(code: str) -> bool:
    """A NUTS-2 or NUTS-1 code of an EU-27 country, extra-regio codes excluded."""
    if code[:2] not in EU27:
        return False
    if len(code) == 4:
        return not code.endswith("ZZ")
    return len(code) == 3 and not code.endswith("Z")


def wheat_table(payloads: Sequence[Mapping[str, Any]], crop: str = CROP) -> list[WheatRow]:
    """One row per EU-27 region (NUTS-2 or NUTS-1) and year in the study period, for
    ``crop`` (wheat unless stated)."""
    by_key: dict[tuple[str, int], dict[str, tuple[float | None, str]]] = {}
    seen: set[str] = set()
    for payload in payloads:
        cells = decode(payload, {"freq": "A", "crops": crop, "strucpro": MEASURES})
        codes = {c.codes["geo"] for c in cells}
        if codes & seen:
            raise DataError(f"Regions in more than one response: {sorted(codes & seen)}.")
        seen |= codes
        for cell in cells:
            geo, period = cell.codes["geo"], cell.codes["time"]
            if not period.isdigit():
                raise DataError(f"Unexpected period {period}.")
            year = int(period)
            if not is_eu_region(geo) or not FIRST_YEAR <= year <= LAST_YEAR:
                continue
            by_key.setdefault((geo, year), {})[cell.codes["strucpro"]] = (cell.value,
                                                                           cell.flag)
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


def save_snapshot(directory: Path, responses: Mapping[str, tuple[bytes, str]],
                  retrieved: datetime | None = None, crop: str = CROP) -> dict[str, Any]:
    """Write the raw responses (by NUTS level), the table and a manifest with hashes."""
    payloads = {level: json.loads(raw.decode("utf-8")) for level, (raw, _) in responses.items()}
    rows = wheat_table(list(payloads.values()), crop)
    table = table_text(rows).encode("utf-8")
    files = {raw_name(level, crop): raw for level, (raw, _) in responses.items()}
    files[table_name(crop)] = table
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {
        "dataset": CROPS_DATASET, "crop": crop, "measures": list(MEASURES),
        "urls": {level: url for level, (_, url) in responses.items()},
        "retrieved_utc": (retrieved or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eurostat_updated": {level: p.get("updated") for level, p in payloads.items()},
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
    raws = [name for name in manifest["files"] if name.startswith("raw/")]
    crop = manifest["crop"]
    rows = wheat_table([json.loads((directory / name).read_bytes().decode("utf-8"))
                        for name in raws], crop)
    if table_text(rows).encode("utf-8") != (directory / table_name(crop)).read_bytes():
        raise DataError("The processed table does not re-derive from the raw response.")
    counts = summary(rows)
    if counts != manifest["summary"]:
        raise DataError("The manifest summary does not match the table.")
    return dict(counts)
