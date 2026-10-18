"""Monthly Eurostat outcomes (``MONTHLY_IDENTIFICATION_PROTOCOL.md``, section 3).

Two indicators: the seasonally adjusted monthly unemployment rate (``une_rt_m``) and
seasonally and calendar adjusted industrial production (``sts_inpr_m``, NACE B-D,
2021 = 100). The original JSON-stat bytes are frozen with SHA-256 hashes, decoded
defensively (missing cells and flags kept, nothing imputed) and written as one
country-month CSV; the audit re-derives that CSV from the raw bytes.

How each country produces its monthly unemployment figure is recorded in
``UNEMPLOYMENT_METHOD``. Trends, centred moving averages and interpolation between
quarterly survey results put information from later months into a month's value, which
breaks the outcome-timing assumption A7; only group D enters the primary sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from eu_okun.contract import API_BASE, EU27, EurostatDataError, Observation, SeriesSpec
from eu_okun.jsonstat import decode_series

MONTH_RE = re.compile(r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])$")

MONTHLY_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("unemployment", "une_rt_m", "unemployment_rate_pct", {
        "freq": "M", "s_adj": "SA", "age": "TOTAL", "sex": "T", "unit": "PC_ACT",
    }),
    SeriesSpec("ip", "sts_inpr_m", "industrial_production_index", {
        "freq": "M", "indic_bt": "PRD", "nace_r2": "B-D", "s_adj": "SCA", "unit": "I21",
    }),
)

METHOD_SOURCE = "https://ec.europa.eu/eurostat/cache/metadata/en/une_rt_m_esms.htm"
METHOD_READ = "2026-10-18"
GROUPS: dict[str, str] = {
    "D": "monthly Labour Force Survey estimates, seasonally adjusted",
    "T": "trend published instead of seasonally adjusted data",
    "A": "listed as direct, but the last of three months is nowcast (ambiguous)",
    "M": "three-month moving average published under the middle month",
    "I": "quarterly survey results interpolated with registered unemployment",
    "B": "benchmarking to moving annual averages of survey results",
    "U": "method not stated",
}
UNEMPLOYMENT_METHOD: dict[str, str] = {
    **{g: "D" for g in ("AT", "CZ", "EL", "FI", "HU", "IT", "RO", "SE")},
    **{g: "T" for g in ("DE", "NL")},
    "DK": "A",
    **{g: "M" for g in ("EE", "PT")},
    **{g: "I" for g in ("BE", "BG", "CY", "ES", "FR", "LT", "LV", "PL", "SK")},
    **{g: "B" for g in ("HR", "LU")},
    **{g: "U" for g in ("IE", "MT", "SI")},
}
PRIMARY_GROUP = "D"
PROCESSED_FILE = "eu27_monthly.csv"
MANIFEST_FILE = "snapshot_manifest.json"

MonthlyPanel = dict[tuple[str, str], dict[str, float | None]]


def month_sequence(start: str, end: str) -> list[str]:
    """Every month from start to end inclusive, as YYYY-MM."""
    if not MONTH_RE.fullmatch(start) or not MONTH_RE.fullmatch(end):
        raise ValueError("Months must have the format YYYY-MM.")
    first = int(start[:4]) * 12 + int(start[5:]) - 1
    last = int(end[:4]) * 12 + int(end[5:]) - 1
    if last < first:
        raise ValueError("End month cannot precede start month.")
    return [f"{m // 12:04d}-{m % 12 + 1:02d}" for m in range(first, last + 1)]


def primary_countries() -> tuple[str, ...]:
    """Countries whose monthly unemployment satisfies A7 by construction (group D)."""
    return tuple(g for g in EU27 if UNEMPLOYMENT_METHOD[g] == PRIMARY_GROUP)


def monthly_url(spec: SeriesSpec, start: str) -> str:
    """Filtered Eurostat query from ``start`` to the latest published month."""
    month_sequence(start, start)
    params = {"lang": "EN", **spec.filters, "sinceTimePeriod": start}
    return f"{API_BASE}/{spec.dataset}?{urlencode(params)}"


def _download(url: str) -> tuple[bytes, dict[str, Any]]:
    request = Request(url, headers={"User-Agent": "Medium-Blog/eu-okun-law"})
    try:
        with urlopen(request, timeout=120) as response:
            raw = bytes(response.read())
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Eurostat download failed for {url}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EurostatDataError(f"Invalid JSON from {url}.") from exc
    if not isinstance(payload, dict):
        raise EurostatDataError(f"Invalid JSON root from {url}.")
    return raw, payload


def decode_monthly(payload: Mapping[str, Any], spec: SeriesSpec,
                   countries: Sequence[str] = EU27) -> dict[tuple[str, str], Observation]:
    """Country-month cells of one series (flags kept, gaps kept, nothing imputed)."""
    return decode_series(payload, spec, countries, periods=MONTH_RE)


def last_complete_month(cells: Mapping[str, Mapping[tuple[str, str], Observation]],
                        countries: Sequence[str] = EU27) -> str:
    """The last month in which every series has a value for every country."""
    months = sorted({m for series in cells.values() for (_, m) in series})
    for month in reversed(months):
        if all(series.get((g, month), Observation(None, "")).value is not None
               for series in cells.values() for g in countries):
            return month
    raise EurostatDataError("No month has every series for every country.")


def _rows(cells: Mapping[str, Mapping[tuple[str, str], Observation]], start: str, end: str,
          countries: Sequence[str]) -> list[dict[str, str]]:
    rows = []
    for geo in countries:
        for month in month_sequence(start, end):
            row = {"geo": geo, "month": month}
            for spec in MONTHLY_SERIES:
                cell = cells[spec.name].get((geo, month), Observation(None, ""))
                row[spec.column] = "" if cell.value is None else repr(cell.value)
                row[f"{spec.name}_flag"] = cell.status
            rows.append(row)
    return rows


def _columns() -> list[str]:
    columns = ["geo", "month"]
    for spec in MONTHLY_SERIES:
        columns += [spec.column, f"{spec.name}_flag"]
    return columns


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_columns(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_monthly_snapshot(destination: Path, start: str = "2000-01",
                          end: str | None = None) -> dict[str, Any]:
    """Download both series, cut at the last complete month, freeze with a manifest."""
    month_sequence(start, start)
    raw_bytes: dict[str, bytes] = {}
    cells: dict[str, dict[tuple[str, str], Observation]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for spec in MONTHLY_SERIES:
        url = monthly_url(spec, start)
        raw, payload = _download(url)
        raw_bytes[spec.dataset] = raw
        cells[spec.name] = decode_monthly(payload, spec)
        sources[spec.name] = {"dataset": spec.dataset, "url": url,
                              "sha256": hashlib.sha256(raw).hexdigest(),
                              "updated": str(payload.get("updated", "")),
                              "filters": dict(spec.filters)}
    end = end or last_complete_month(cells)
    rows = _rows(cells, start, end, EU27)
    for spec in MONTHLY_SERIES:
        flags = Counter(r[f"{spec.name}_flag"] for r in rows if r[f"{spec.name}_flag"])
        sources[spec.name]["missing_cells"] = sum(r[spec.column] == "" for r in rows)
        sources[spec.name]["flags"] = dict(sorted(flags.items()))

    raw_dir, processed = destination / "raw", destination / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    for dataset, raw in raw_bytes.items():
        (raw_dir / f"{dataset}.json").write_bytes(raw)
    _write_csv(processed / PROCESSED_FILE, rows)
    manifest: dict[str, Any] = {
        "source": "Eurostat Statistics API (JSON-stat 2.0)",
        "protocol": "MONTHLY_IDENTIFICATION_PROTOCOL.md",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "period": {"start": start, "end": end},
        "countries": list(EU27),
        "country_month_rows": len(rows),
        "series": sources,
        "processed_csv_sha256": _sha256(processed / PROCESSED_FILE),
        "unemployment_method": {"source": METHOD_SOURCE, "read": METHOD_READ,
                                "groups": GROUPS, "countries": UNEMPLOYMENT_METHOD},
        "note": ("Observations can be revised; flagged and missing values are not imputed. "
                 "The end month is the last one with both series for all 27 countries."),
    }
    (destination / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True)
                                             + "\n", encoding="utf-8", newline="\n")
    return manifest


def read_monthly_rows(directory: Path) -> list[dict[str, str]]:
    """The processed country-month CSV as strings, exactly as stored."""
    with (directory / "processed" / PROCESSED_FILE).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != _columns():
            raise EurostatDataError("Unexpected monthly CSV columns.")
        return list(reader)


def audit_monthly_snapshot(directory: Path) -> dict[str, Any]:
    """Verify hashes, the complete grid, plausible values and re-derivation from raw."""
    manifest: dict[str, Any] = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("countries") != list(EU27):
        raise EurostatDataError("The monthly snapshot cohort is not the fixed EU-27.")
    if manifest.get("unemployment_method", {}).get("countries") != UNEMPLOYMENT_METHOD:
        raise EurostatDataError("The snapshot's country classification differs from the code.")
    start, end = manifest["period"]["start"], manifest["period"]["end"]
    cells: dict[str, dict[tuple[str, str], Observation]] = {}
    for spec in MONTHLY_SERIES:
        entry = manifest["series"][spec.name]
        path = directory / "raw" / f"{spec.dataset}.json"
        if entry.get("dataset") != spec.dataset or _sha256(path) != entry.get("sha256"):
            raise EurostatDataError(f"Raw monthly response changed: {spec.dataset}.")
        cells[spec.name] = decode_monthly(json.loads(path.read_bytes()), spec)
    processed = directory / "processed" / PROCESSED_FILE
    if _sha256(processed) != manifest.get("processed_csv_sha256"):
        raise EurostatDataError("Processed monthly CSV does not match its manifest hash.")
    stored = read_monthly_rows(directory)
    if stored != _rows(cells, start, end, EU27):
        raise EurostatDataError("Processed monthly CSV does not re-derive from the raw files.")
    for row in stored:
        u, ip = row["unemployment_rate_pct"], row["industrial_production_index"]
        if u and not 0 <= float(u) <= 100:
            raise EurostatDataError(f"Implausible unemployment rate {row['geo']}/{row['month']}.")
        if ip and not (math.isfinite(float(ip)) and float(ip) > 0):
            raise EurostatDataError(f"Non-positive production index {row['geo']}/{row['month']}.")
    return manifest


def load_monthly_panel(directory: Path) -> MonthlyPanel:
    """Unemployment rate and production index by (country, month); None where missing."""
    panel: MonthlyPanel = {}
    for row in read_monthly_rows(directory):
        panel[(row["geo"], row["month"])] = {
            "unemployment": float(row["unemployment_rate_pct"])
            if row["unemployment_rate_pct"] else None,
            "ip": float(row["industrial_production_index"])
            if row["industrial_production_index"] else None,
        }
    return panel


def flag_counts(directory: Path, first: str, countries: Sequence[str]) -> dict[str, Counter[str]]:
    """Flags by series from ``first`` onward for the given countries (reported, not used)."""
    counts: dict[str, Counter[str]] = {spec.name: Counter() for spec in MONTHLY_SERIES}
    for row in read_monthly_rows(directory):
        if row["geo"] in countries and row["month"] >= first:
            for spec in MONTHLY_SERIES:
                if row[f"{spec.name}_flag"]:
                    counts[spec.name][row[f"{spec.name}_flag"]] += 1
    return counts
