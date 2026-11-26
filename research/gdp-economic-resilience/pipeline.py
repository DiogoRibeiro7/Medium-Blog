"""Reproduce the first GDP resilience results using immutable registry snapshots.

Run from any working directory. Only local, checksum-verified source bytes are
read by the analysis. The optional fetch step retrieves pinned GitHub files,
never the live upstream World Bank API.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMIT = "ce3cc0995411c124534408e2086772275ee2c7c8"
REPO = "DiogoRibeiro7/data"
WDI = "world-bank-wdi-gdp-resilience-2026-11-26"
CWON = "world-bank-cwon-2024-gdp-resilience-2026-11-26"
EU27 = tuple(sorted("""AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN
IRL ITA LVA LTU LUX MLT NLD POL PRT ROU SVK SVN ESP SWE""".split()))
COUNTRIES = tuple(sorted((*EU27, "USA", "GBR", "CHE", "JPN", "CHN", "IND")))
GDP = "NY.GDP.MKTP.CD"
PPP = "NY.GDP.MKTP.PP.CD"
REAL = "NY.GDP.MKTP.KD"
POP = "SP.POP.TOTL"
AGR = "NV.AGR.TOTL.ZS"
IND = "NV.IND.TOTL.ZS"
MAN = "NV.IND.MANF.ZS"
SRV = "NV.SRV.TOTL.ZS"
WEALTH = "NW.TOW.PC"
WDI_CODES = (GDP, PPP, REAL, POP, AGR, IND, MAN, SRV)
CWON_CODES = ("NW.HCA.PC", "NW.NCA.SSOI.PC", "NW.NCA.TOTL.PC",
              "NW.NFA.PC", "NW.NFL.PC", "NW.PCA.PC", WEALTH)
SOURCES = {WDI: ("2", 1995, 2025, WDI_CODES),
           CWON: ("59", 1995, 2020, CWON_CODES)}
Key = tuple[str, str, int]
Value = float | None
Record = dict[str, Any]


@dataclass(frozen=True)
class Snapshot:
    """Retain explicit nulls and distinguish them from absent source rows."""

    data: dict[Key, Value]

    def value(self, code: str, country: str, year: int) -> Value:
        """Retrieve a numeric cell, or None when unavailable."""
        return self.data.get((code, country, year))


def checksum(raw: bytes) -> str:
    """Hash original file bytes without JSON reserialization."""
    return hashlib.sha256(raw).hexdigest()


def read_contract(slug: str) -> dict[str, Any]:
    """Accept only the expected registry revision and normalized raw paths."""
    contract = json.loads((ROOT / "data_sources" / f"{slug}.json").read_text(encoding="utf-8"))
    if (contract["schema_version"] != 1 or contract["layer"] != "canonical"
            or contract["commit"] != COMMIT or contract["repository"] != REPO
            or contract["dataset"] != slug):
        raise ValueError(f"Unexpected data contract: {slug}")
    paths: set[str] = set()
    for item in contract["files"]:
        path = item["path"]
        expected_url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/datasets/{slug}/{path}"
        if (not re.fullmatch(r"raw/[A-Za-z0-9._-]+\.json", path)
                or path in paths or item["url"] != expected_url
                or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
                or type(item["bytes"]) is not int or item["bytes"] <= 0):
            raise ValueError(f"Invalid pinned file: {path}")
        paths.add(path)
    return contract


def verify_bytes(raw: bytes, item: Record) -> None:
    """Fail closed on any byte-count or checksum discrepancy."""
    if len(raw) != item["bytes"] or checksum(raw) != item["sha256"]:
        raise ValueError(f"Checksum or length mismatch: {item['path']}")


def download(url: str) -> bytes:
    """Fetch a pinned registry URL with bounded retries."""
    for attempt in range(4):
        try:
            req = Request(url, headers={"User-Agent": "gdp-resilience-reproduction"})
            with urlopen(req, timeout=90) as response:
                return response.read()
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable retry state")


def verified_files(slug: str, data_dir: Path, fetch: bool = False) -> dict[str, bytes]:
    """Verify every registered file, optionally downloading missing files."""
    files = {}
    for item in read_contract(slug)["files"]:
        path = data_dir / slug / item["path"]
        if path.is_symlink():
            raise ValueError(f"Refusing symlink: {path}")
        if path.exists():
            raw = path.read_bytes()
            verify_bytes(raw, item)
        elif fetch:
            raw = download(item["url"])
            verify_bytes(raw, item)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(raw)
            temporary.replace(path)
        else:
            raise FileNotFoundError(f"{path}; run with --fetch first")
        files[item["path"]] = raw
    return files


def decode_rows(files: dict[str, bytes], slug: str) -> Snapshot:
    """Validate page identity, expected indicators, keys and manifest coverage."""
    source, start, end, indicators = SOURCES[slug]
    manifest = json.loads(files["raw/acquisition-manifest.json"])
    if (str(manifest["source_id"]) != source or manifest["start"] != start
            or manifest["end"] != end or tuple(sorted(manifest["countries"])) != COUNTRIES
            or set(manifest["coverage"]) != set(indicators)):
        raise ValueError(f"Unexpected manifest scope: {slug}")
    data: dict[Key, Value] = {}
    for code in indicators:
        metadata = json.loads(files[f"raw/{code}.metadata.json"])[1]
        if len(metadata) != 1 or metadata[0]["id"] != code or str(metadata[0]["source"]["id"]) != source:
            raise ValueError(f"Unexpected indicator metadata: {code}")
        pages = {
            int(match.group(1)): raw
            for path, raw in files.items()
            if (match := re.fullmatch(r"raw/" + re.escape(code) + r"\.page-(\d+)\.json", path))
        }
        if not pages or sorted(pages) != list(range(1, len(pages) + 1)):
            raise ValueError(f"Missing or noncontiguous pages: {code}")
        expected_total: int | None = None
        for page, raw in sorted(pages.items()):
            header, rows = json.loads(raw)
            total = int(header["total"])
            if (str(header["sourceid"]) != source or int(header["page"]) != page
                    or int(header["pages"]) != len(pages)
                    or (expected_total is not None and total != expected_total)):
                raise ValueError(f"Unexpected page metadata: {code}/{page}")
            expected_total = total
            for row in rows:
                country = row["countryiso3code"]
                year = int(row["date"])
                key = (code, country, year)
                value = row["value"]
                if (row["indicator"]["id"] != code or country not in COUNTRIES
                        or not start <= year <= end or key in data):
                    raise ValueError(f"Unexpected or duplicate observation: {key}")
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError(f"Nonfinite or nonnumeric observation: {key}")
                data[key] = None if value is None else float(value)
        values = [v for k, v in data.items() if k[0] == code]
        stats = manifest["coverage"][code]
        observed = sum(v is not None for v in values)
        actual = (len(values), observed, len(values) - observed,
                  len(COUNTRIES) * (end - start + 1) - len(values))
        expected = tuple(stats[k] for k in ("records", "observed", "explicit_nulls", "absent_cells"))
        if actual != expected or len(values) != expected_total:
            raise ValueError(f"Observation counts differ from reviewed manifest: {code}")
    return Snapshot(data)


def complete_sum(values: list[Value]) -> Value:
    """Do not construct a partial-country aggregate."""
    if not values or any(v is None for v in values):
        return None
    return math.fsum(v for v in values if v is not None)


def ratio(numerator: Value, denominator: Value) -> Value:
    """Return None for undefined or missing ratios."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def growth(start: Value, end: Value) -> Value:
    """Percentage change requires a positive baseline; no imputation."""
    relative = ratio(end, start)
    return None if relative is None else 100 * (relative - 1)


def concentration(shares: list[Value]) -> tuple[Value, Value, Value]:
    """Return observed share sum, normalized HHI, and effective sector count.

    The shares use GDP as denominator. Normalize only over a complete,
    non-overlapping set of nonnegative sector GVA shares. Retain the original
    sum so unallocated GDP is not implicitly treated as sectoral output.
    """
    total = complete_sum(shares)
    if total is None or total <= 0 or any(v < 0 for v in shares if v is not None):
        return total, None, None
    hhi = math.fsum((v / total) ** 2 for v in shares if v is not None)
    return total, hhi, 1 / hhi


def gdp_rankings(snapshot: Snapshot, year: int = 2025) -> list[Record]:
    """Keep country ranks distinct from the fixed-membership EU comparison."""
    groups = {"countries": {c: (c,) for c in COUNTRIES},
              "fixed_eu27_comparators": {"EU27": EU27, **{c: (c,) for c in COUNTRIES if c not in EU27}}}
    results: list[Record] = []
    for group, members in groups.items():
        rows = []
        for entity, countries in members.items():
            totals = {code: complete_sum([snapshot.value(code, c, year) for c in countries])
                      for code in (GDP, PPP, POP)}
            rows.append({"group": group, "entity": entity, "year": year,
                         "members": len(countries), "nominal_usd": totals[GDP],
                         "ppp_international_dollars": totals[PPP], "population": totals[POP],
                         "nominal_usd_per_capita": ratio(totals[GDP], totals[POP]),
                         "ppp_per_capita": ratio(totals[PPP], totals[POP])})
        for metric in ("nominal_usd", "ppp_international_dollars"):
            ordered = sorted((row for row in rows if row[metric] is not None),
                             key=lambda row: (-row[metric], row["entity"]))
            for rank, row in enumerate(ordered, 1):
                row[f"{metric}_rank"] = rank
        for row in rows:
            for metric in ("nominal_usd", "ppp_international_dollars"):
                row.setdefault(f"{metric}_rank", None)
        results.extend(sorted(rows, key=lambda row: row["entity"]))
    return results


def sector_rows(snapshot: Snapshot) -> list[Record]:
    """Compute three- and four-sector concentration without double-counting manufacturing."""
    results = []
    for country in COUNTRIES:
        for year in range(1995, 2026):
            agr, ind, man, srv = (snapshot.value(code, country, year) for code in (AGR, IND, MAN, SRV))
            nonmanufacturing = None if ind is None or man is None else ind - man
            total, hhi, effective = concentration([agr, ind, srv])
            _, hhi4, effective4 = concentration([agr, nonmanufacturing, man, srv])
            results.append({"country": country, "year": year,
                            "agriculture_pct_gdp": agr, "industry_pct_gdp": ind,
                            "manufacturing_pct_gdp": man, "nonmanufacturing_industry_pct_gdp": nonmanufacturing,
                            "services_pct_gdp": srv, "sector_sum_pct_gdp": total,
                            "unallocated_pct_gdp": None if total is None else 100 - total,
                            "hhi_three_broad_sectors": hhi, "effective_three_sectors": effective,
                            "hhi_four_sectors": hhi4, "effective_four_sectors": effective4})
    return results


def wealth_rows(wdi: Snapshot, cwon: Snapshot) -> list[Record]:
    """Compare within-country endpoint growth, not incompatible dollar levels."""
    windows = ((1996, 2020), (1995, 2020), (2000, 2020), (1996, 2019))
    results = []
    for start, end in windows:
        for country in COUNTRIES:
            g0, g1 = (ratio(wdi.value(REAL, country, year), wdi.value(POP, country, year))
                      for year in (start, end))
            w0, w1 = (cwon.value(WEALTH, country, year) for year in (start, end))
            gdp_change, wealth_change = growth(g0, g1), growth(w0, w1)
            comparable = gdp_change is not None and wealth_change is not None
            results.append({"country": country, "start": start, "end": end,
                            "real_gdp_pc_start": g0, "real_gdp_pc_end": g1,
                            "real_wealth_pc_start": w0, "real_wealth_pc_end": w1,
                            "real_gdp_pc_growth_pct": gdp_change, "real_wealth_pc_growth_pct": wealth_change,
                            "growth_gap_percentage_points": gdp_change - wealth_change if comparable else None,
                            "gdp_up_wealth_down": gdp_change > 0 and wealth_change < 0 if comparable else None,
                            "comparable": comparable,
                            "bulgaria_1996_baseline_flag": country == "BGR" and start == 1996})
    return results


def coverage_rows(snapshot: Snapshot, slug: str) -> list[Record]:
    """Report absent and explicit null observations separately by indicator and year."""
    _, start, end, codes = SOURCES[slug]
    results = []
    for code in sorted(codes):
        for year in range(start, end + 1):
            present = [(code, c, year) for c in COUNTRIES if (code, c, year) in snapshot.data]
            observed = sum(snapshot.data[k] is not None for k in present)
            results.append({"dataset": slug, "indicator": code, "year": year, "observed": observed,
                            "explicit_nulls": len(present) - observed, "absent": len(COUNTRIES) - len(present)})
    return results


def write_csv(path: Path, rows: list[Record]) -> None:
    """Write deterministic CSV; missing values are blank rather than zero."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir: Path, output: Path, fetch: bool = False) -> None:
    """Generate reproducible tables from both verified snapshots."""
    wdi = decode_rows(verified_files(WDI, data_dir, fetch), WDI)
    cwon = decode_rows(verified_files(CWON, data_dir, fetch), CWON)
    rankings = gdp_rankings(wdi)
    sectors = sector_rows(wdi)
    wealth = wealth_rows(wdi, cwon)
    coverage = coverage_rows(wdi, WDI) + coverage_rows(cwon, CWON)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"gdp_rankings_2025.csv": rankings, "sector_composition.csv": sectors,
              "gdp_wealth_endpoint_growth.csv": wealth, "coverage.csv": coverage}
    for filename, rows in tables.items():
        write_csv(output / filename, rows)
    primary = [r for r in wealth if (r["start"], r["end"]) == (1996, 2020)]
    comparable = [r for r in primary if r["comparable"]]
    summary = {
        "registry_commit": COMMIT,
        "requested_countries": len(COUNTRIES),
        "wdi_non_null": sum(v is not None for v in wdi.data.values()),
        "cwon_non_null": sum(v is not None for v in cwon.data.values()),
        "primary_wealth_window": [1996, 2020],
        "comparable_wealth_countries": len(comparable),
        "excluded_wealth_countries": [r["country"] for r in primary if not r["comparable"]],
        "gdp_up_wealth_down": [r["country"] for r in comparable if r["gdp_up_wealth_down"]],
        "gdp_up_wealth_down_excluding_bulgaria": [r["country"] for r in comparable if r["gdp_up_wealth_down"] and r["country"] != "BGR"],
        "broad_sector_balanced_years": [year for year in range(1995, 2026)
                                       if sum(r["year"] == year and r["hhi_three_broad_sectors"] is not None for r in sectors) == len(COUNTRIES)],
        "tables": {name: len(rows) for name, rows in tables.items()},
        "scope": "Descriptive baseline; no causal inference or shock simulations.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Command-line entry point for offline analysis and optional fetching."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--fetch", action="store_true", help="Download missing pinned registry files only.")
    args = parser.parse_args()
    run(args.data_dir, args.output, args.fetch)


if __name__ == "__main__":
    main()
