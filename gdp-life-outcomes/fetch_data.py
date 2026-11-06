"""Capture a reproducible World Bank WDI country panel for the GDP/welfare study.

The API is used exclusively for *source capture*. Subsequent analysis and CI
read the archived snapshot, so the study cannot change silently when WDI is
revised. No credential is needed. Aggregates are excluded using the country
metadata endpoint, rather than an unreliable country-code blacklist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "https://api.worldbank.org/v2"
INDICATORS: dict[str, str] = {
    "NY.GDP.PCAP.PP.KD": "GDP per capita, constant 2021 international dollars",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total, years",
    "SH.DYN.MORT": "Under-five mortality rate, per 1,000 live births",
    "EG.ELC.ACCS.ZS": "Access to electricity, percent of population",
    "SH.H2O.SMDW.ZS": "Safely managed drinking water, percent of population",
    "SH.XPD.GHED.GD.ZS": "Domestic general government health expenditure, percent of GDP",
    "SI.POV.GINI": "Gini index, World Bank survey estimate",
    "SP.POP.TOTL": "Population, total",
}
PANEL_FIELDS = (
    "country_code", "country_name", "region", "income_group", "year",
    "indicator_id", "value",
)


@dataclass(frozen=True)
class ApiResponse:
    """A validated single-page WDI API response with provenance."""

    rows: list[dict[str, Any]]
    last_updated: str
    canonical_sha256: str
    total: int


def _request_json(url: str, *, timeout: float = 45.0, attempts: int = 3) -> Any:
    """Read JSON with bounded retries for temporary network/server failures."""

    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": "gdp-welfare-research/1.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Unreachable retry state")


def read_api_page(url: str) -> ApiResponse:
    """Reject incomplete or malformed responses, including silently paged data."""

    payload = _request_json(url)
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError(f"Unexpected API envelope for {url}")
    meta, rows = payload
    if not isinstance(meta, dict) or not isinstance(rows, list):
        raise ValueError("Unexpected API metadata or row list")
    if int(meta.get("pages", 0)) != 1:
        raise ValueError(f"WDI response has {meta.get('pages')} pages: {url}")
    total = int(meta.get("total", -1))
    if total != len(rows):
        raise ValueError(f"WDI response truncated: {len(rows)} of {total} rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("WDI rows must all be JSON objects")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return ApiResponse(
        rows=rows,
        last_updated=str(meta.get("lastupdated", "unknown")),
        canonical_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        total=total,
    )


def _countries() -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    """Return country metadata without World Bank regional aggregates."""

    url = f"{API_BASE}/country?format=json&per_page=500"
    result = read_api_page(url)
    countries: dict[str, dict[str, str]] = {}
    for row in result.rows:
        code = str(row.get("id", ""))
        region = row.get("region", {})
        income = row.get("incomeLevel", {})
        if not isinstance(region, dict) or region.get("id") == "NA":
            continue
        if len(code) != 3 or not code.isalpha():
            continue
        countries[code] = {
            "country_name": str(row.get("name", "")),
            "region": str(region.get("value", "")),
            "income_group": str(income.get("value", "")) if isinstance(income, dict) else "",
        }
    if len(countries) < 180:
        raise ValueError(f"Unexpectedly few countries ({len(countries)})")
    return countries, {
        "url": url, "rows": result.total,
        "sha256": result.canonical_sha256,
        "last_updated": result.last_updated,
        "country_count": len(countries),
    }


def _required_year(indicator: str, year: int, *, start_year: int, end_year: int) -> bool:
    """Keep only observations needed for the prespecified comparisons.

    Endpoint comparisons require both years; delayed survey series require
    an explicitly documented lookback period. Trimming does not invent data.
    """

    if indicator in {"NY.GDP.PCAP.PP.KD", "SP.DYN.LE00.IN", "SH.DYN.MORT"}:
        return year in {start_year, end_year}
    if indicator == "SI.POV.GINI":
        return end_year - 5 <= year <= end_year
    if indicator == "SH.XPD.GHED.GD.ZS":
        return end_year - 3 <= year <= end_year
    return year == end_year


def capture(output_dir: Path, start_year: int, end_year: int) -> None:
    """Capture a stable panel CSV and a structured, source-specific manifest."""

    if not 1960 <= start_year <= end_year <= 2026:
        raise ValueError("Invalid WDI year range")
    output_dir.mkdir(parents=True, exist_ok=True)
    countries, country_provenance = _countries()
    data: list[dict[str, object]] = []
    audit: dict[str, dict[str, object]] = {}
    keys: set[tuple[str, int, str]] = set()

    for indicator, label in INDICATORS.items():
        url = (
            f"{API_BASE}/country/all/indicator/{indicator}?format=json"
            f"&date={start_year}:{end_year}&per_page=20000"
        )
        result = read_api_page(url)
        kept = 0
        for row in result.rows:
            code = str(row.get("countryiso3code", ""))
            if code not in countries or row.get("value") is None:
                continue
            indicator_object = row.get("indicator", {})
            if not isinstance(indicator_object, dict) or indicator_object.get("id") != indicator:
                raise ValueError(f"Unexpected indicator in {url}")
            year = int(str(row["date"]))
            if not _required_year(indicator, year, start_year=start_year, end_year=end_year):
                continue
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {indicator} in {code}, {year}")
            key = (code, year, indicator)
            if key in keys:
                raise ValueError(f"Duplicate country-year-indicator: {key}")
            keys.add(key)
            data.append({
                "country_code": code, **countries[code], "year": year,
                "indicator_id": indicator, "value": f"{value:.14g}",
            })
            kept += 1
        audit[indicator] = {
            "description": label,
            "url": url,
            "raw_rows": result.total,
            "country_observations": kept,
            "source_last_updated": result.last_updated,
            "api_response_canonical_sha256": result.canonical_sha256,
        }
        print(f"{indicator}: {kept} genuine-country observations", flush=True)

    data.sort(key=lambda r: (str(r["country_code"]), int(r["year"]), str(r["indicator_id"])))
    if not data:
        raise ValueError("No observations were captured")
    panel_file = output_dir / "wdi_panel.csv"
    with panel_file.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=PANEL_FIELDS)
        writer.writeheader()
        writer.writerows(data)
    file_sha256 = hashlib.sha256(panel_file.read_bytes()).hexdigest()
    manifest = {
        "source": "World Bank World Development Indicators API v2",
        "source_documentation": "https://datahelpdesk.worldbank.org/knowledgebase/articles/889392",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "years": {"first": start_year, "last": end_year},
        "country_metadata": country_provenance,
        "indicators": audit,
        "panel_rows": len(data),
        "panel_sha256": file_sha256,
        "notes": [
            "World Bank aggregates removed by official region metadata (region id NA).",
            "Only endpoint years are archived for GDP, life expectancy and under-five mortality.",
            "Only the last six years of Gini and four years of health-spending observations are retained.",
            "Other indicators retain the final-year observation only, matching the prespecified analysis.",
            "Panel observations use the indicator-specific source vintage documented above.",
            "Missing observations are omitted, never imputed or filled forward.",
            "Latest available values may differ from the Our World in Data 2026 graphic.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(data)} rows with SHA-256 {file_sha256}")


def main() -> None:
    """Run the source capture from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()
    capture(args.output_dir, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
