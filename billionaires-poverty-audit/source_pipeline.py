"""Build a country panel from provenance-audited official-data extracts.

No observational country values are hard-coded. Missing billionaires are not
interpreted as zero, and OWID observation years are not mislabelled as the
underlying household survey dates.

The build command runs offline. The separate fetch command downloads only
public OWID/WDI inputs and records retrieval metadata. Country-level Forbes
counts must be independently reviewed and supplied as a CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping
from resilient_fetch import fetch_bytes

import pandas as pd

Attribution = Literal["residence", "citizenship"]

OWID_URL = (
    "https://ourworldindata.org/grapher/"
    "poverty-share-on-less-than-30-per-day.csv?v=1&csvType=full&useColumnShortNames=false"
)
WDI_GDP_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.KD"
    "?format=json&date=2020:2025&per_page=20000"
)
WDI_POP_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"
    "?format=json&date=2020:2025&per_page=20000"
)
SOURCE_URLS: Mapping[str, str] = {
    "owid_30usd_2021_ppp.csv": OWID_URL,
    "wdi_gdp_ppp_constant.json": WDI_GDP_URL,
    "wdi_population.json": WDI_POP_URL,
}
ISO3_PATTERN = re.compile(r"[A-Z]{3}\Z")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of the exact bytes of one local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_public_inputs(output_dir: Path) -> dict[str, Any]:
    """Download public sources and create a manifest, refusing overwrite.

    No individual billionaire profiles are fetched, stored or redistributed.
    """
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(timezone.utc).isoformat()
    records: dict[str, dict[str, Any]] = {}
    for filename, url in SOURCE_URLS.items():
        content = fetch_bytes(url, name=filename)
        target = output_dir / filename
        if target.exists():
            raise FileExistsError(f"Refuse to overwrite a snapshot: {target}")
        target.write_bytes(content)
        records[filename] = {
            "source_url": url,
            "bytes": len(content),
            "sha256": _sha256(target),
        }
    manifest: dict[str, Any] = {
        "retrieved_at_utc": retrieved,
        "files": records,
        "note": "Live OWID/WDI sources may be revised; exact bytes and retrieval date are captured.",
    }
    (output_dir / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def read_owid_poverty(
    path: Path, *, last_year: int = 2025, first_year: int = 2020
) -> pd.DataFrame:
    """Parse OWID Grapher data and select latest observation per ISO3.

    OWID Year is the dataset observation year, *not necessarily* the survey
    interview year. The latter must be independently recovered from PIP.
    """
    if first_year > last_year:
        raise ValueError("first_year must not exceed last_year")
    raw = pd.read_csv(path)
    required = {"Entity", "Code", "Year"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"OWID missing required fields: {sorted(missing)}")
    value_cols = [name for name in raw.columns if name not in required]
    if len(value_cols) != 1:
        raise ValueError(f"Expected one OWID poverty indicator, got {value_cols}")
    raw["Code"] = raw["Code"].astype("string").str.strip()
    raw = raw.loc[raw["Code"].str.fullmatch(ISO3_PATTERN.pattern).fillna(False)].copy()
    raw["Year"] = pd.to_numeric(raw["Year"], errors="raise")
    raw["poverty_pct"] = pd.to_numeric(raw[value_cols[0]], errors="coerce")
    raw = raw.loc[
        raw["Year"].between(first_year, last_year)
        & raw["poverty_pct"].notna()
    ].copy()
    if not raw["poverty_pct"].between(0.0, 100.0).all():
        raise ValueError("OWID poverty percentage outside [0, 100]")
    if raw.duplicated(["Code", "Year"]).any():
        raise ValueError("OWID country-year duplicates must be resolved explicitly")
    chosen = raw.sort_values(["Code", "Year"]).drop_duplicates("Code", keep="last")
    out = chosen.rename(
        columns={"Code": "iso3", "Year": "poverty_year", "Entity": "owid_entity"}
    )[["iso3", "owid_entity", "poverty_year", "poverty_pct"]].reset_index(drop=True)
    if out.empty:
        raise ValueError("No countries with OWID observations in date window")
    return out


def read_wdi_indicator(
    path: Path,
    *,
    indicator: str,
    field: str,
    last_year: int = 2025,
    first_year: int = 2020,
) -> pd.DataFrame:
    """Parse World Bank JSON and reject incomplete pagination."""
    response: Any = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(response, list) and len(response) == 2):
        raise ValueError("World Bank JSON must be [metadata, observations]")
    metadata, observations = response
    if not isinstance(metadata, dict) or not isinstance(observations, list):
        raise ValueError("Malformed World Bank metadata or observations")
    if int(metadata.get("pages", 0)) != 1:
        raise ValueError("World Bank API response is paginated or incomplete")
    values: list[dict[str, Any]] = []
    for record in observations:
        if not isinstance(record, dict) or record.get("value") is None:
            continue
        code = str(record.get("countryiso3code", "")).strip()
        if ISO3_PATTERN.fullmatch(code) is None:
            continue
        indicator_info = record.get("indicator")
        if not isinstance(indicator_info, dict) or indicator_info.get("id") != indicator:
            raise ValueError(f"Unexpected WDI indicator for {code}")
        year = int(record["date"])
        if not first_year <= year <= last_year:
            continue
        value = float(record["value"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Non-positive or non-finite WDI {field}: {code}")
        values.append({"iso3": code, field: value, field + "_year": year})
    data = pd.DataFrame(values)
    if data.empty:
        raise ValueError("No WDI observations found")
    if data.duplicated(["iso3", field + "_year"]).any():
        raise ValueError("World Bank country-year duplicates")
    return (
        data.sort_values(["iso3", field + "_year"])
        .drop_duplicates("iso3", keep="last")
        .reset_index(drop=True)
    )


def read_forbes_aggregate(
    path: Path, *, year: int = 2026
) -> tuple[pd.DataFrame, Attribution]:
    """Validate reviewed country totals without inferring absent zeros."""
    data = pd.read_csv(
        path,
        dtype={
            "iso3": "string",
            "country": "string",
            "billionaire_method": "string",
        },
    )
    required = {
        "iso3", "country", "billionaires", "billionaire_method", "forbes_year"
    }
    if required - set(data.columns):
        raise ValueError(f"Forbes aggregate missing: {sorted(required - set(data.columns))}")
    data = data[list(required)].copy()
    data["iso3"] = data["iso3"].str.strip()
    if not data["iso3"].str.fullmatch(ISO3_PATTERN.pattern).fillna(False).all():
        raise ValueError("Invalid Forbes ISO3 code")
    if data["iso3"].duplicated().any():
        raise ValueError("Duplicate Forbes ISO3 code")
    counts = pd.to_numeric(data["billionaires"], errors="raise")
    if counts.isna().any() or not (counts >= 0).all() or not (counts % 1 == 0).all():
        raise ValueError("Forbes counts must be nonnegative integers")
    data["billionaires"] = counts.astype("int64")
    years = pd.to_numeric(data["forbes_year"], errors="raise")
    if not (years == year).all():
        raise ValueError("Unexpected Forbes vintage")
    methods = set(data["billionaire_method"].dropna().tolist())
    if len(methods) != 1 or not methods.issubset({"residence", "citizenship"}):
        raise ValueError("Forbes rows must use one documented attribution method")
    if data["country"].isna().any() or data["country"].str.strip().eq("").any():
        raise ValueError("Forbes country labels required")
    method: Attribution = "residence" if "residence" in methods else "citizenship"
    return data, method


def assemble_panel(
    *,
    poverty: pd.DataFrame,
    gdp: pd.DataFrame,
    population: pd.DataFrame,
    forbes: pd.DataFrame,
    method: Attribution,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join on explicit ISO3 identifiers and report unmatched countries."""
    if method not in ("residence", "citizenship"):
        raise ValueError("Invalid attribution method")
    # Left joins reveal mismatches; missing counts must never become zeros.
    merged = forbes.merge(poverty, on="iso3", how="left", validate="one_to_one")
    merged = merged.merge(gdp, on="iso3", how="left", validate="one_to_one")
    merged = merged.merge(population, on="iso3", how="left", validate="one_to_one")
    required_fields = ["poverty_pct", "gdp_ppp_per_capita", "population"]
    unmatched = merged.loc[
        merged[required_fields].isna().any(axis=1), "iso3"
    ].tolist()
    matched = merged.dropna(subset=required_fields).copy()
    matched["billionaire_method"] = method
    if matched.empty:
        raise ValueError("No matched countries")
    if not (matched["population"] > 0).all():
        raise ValueError("Population must be positive")
    if (
        matched["poverty_year"].isna().any()
        or matched["gdp_ppp_per_capita_year"].isna().any()
    ):
        raise ValueError("Matched source year is missing")
    out = matched.rename(
        columns={"gdp_ppp_per_capita_year": "gdp_year"}
    )[
        ["country", "iso3", "billionaires", "population", "poverty_pct",
         "gdp_ppp_per_capita", "poverty_year", "gdp_year", "billionaire_method",
         "population_year", "forbes_year", "owid_entity"]
    ].sort_values("iso3").reset_index(drop=True)
    for field in (
        "population", "poverty_year", "gdp_year", "population_year", "forbes_year"
    ):
        out[field] = out[field].astype("int64")
    coverage = {
        "n_forbes_country_rows": int(len(forbes)),
        "n_matched_rows": int(len(out)),
        "unmatched_iso3": sorted(unmatched),
        "zero_counts_explicitly_listed": int((out["billionaires"] == 0).sum()),
        "billionaire_method": method,
        "poverty_year_note": "OWID observation year; actual survey date may differ",
    }
    return out, coverage


def build_from_paths(
    *,
    poverty_csv: Path,
    gdp_json: Path,
    population_json: Path,
    forbes_csv: Path,
    output: Path,
    manifest: Path,
    last_year: int = 2025,
    first_year: int = 2020,
) -> dict[str, Any]:
    """Produce a deterministic panel and SHA-256 provenance report."""
    for path in (poverty_csv, gdp_json, population_json, forbes_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Required source file missing: {path}")
    poverty = read_owid_poverty(
        poverty_csv, last_year=last_year, first_year=first_year
    )
    gdp = read_wdi_indicator(
        gdp_json, indicator="NY.GDP.PCAP.PP.KD", field="gdp_ppp_per_capita",
        last_year=last_year, first_year=first_year
    )
    population = read_wdi_indicator(
        population_json, indicator="SP.POP.TOTL", field="population",
        last_year=last_year, first_year=first_year
    )
    forbes, method = read_forbes_aggregate(forbes_csv)
    panel, coverage = assemble_panel(
        poverty=poverty, gdp=gdp, population=population,
        forbes=forbes, method=method
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False, float_format="%.12g")
    report = {
        "method": "observational country-level matched panel, not causal",
        "coverage": coverage,
        "input_sha256": {
            str(path): _sha256(path)
            for path in (poverty_csv, gdp_json, population_json, forbes_csv)
        },
        "panel_sha256": _sha256(output),
        "selection_years": [first_year, last_year],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Expose separate public fetch and offline assembly commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    fetch = subs.add_parser("fetch", help="Download public official inputs")
    fetch.add_argument("--output-dir", required=True, type=Path)
    build = subs.add_parser("build", help="Assemble offline country panel")
    for arg in (
        "poverty-csv", "gdp-json", "population-json",
        "forbes-csv", "output", "manifest"
    ):
        build.add_argument("--" + arg, required=True, type=Path)
    build.add_argument("--first-year", type=int, default=2020)
    build.add_argument("--last-year", type=int, default=2025)
    args = parser.parse_args()
    if args.command == "fetch":
        result = fetch_public_inputs(args.output_dir)
    else:
        result = build_from_paths(
            poverty_csv=args.poverty_csv, gdp_json=args.gdp_json,
            population_json=args.population_json, forbes_csv=args.forbes_csv,
            output=args.output, manifest=args.manifest,
            last_year=args.last_year, first_year=args.first_year,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
