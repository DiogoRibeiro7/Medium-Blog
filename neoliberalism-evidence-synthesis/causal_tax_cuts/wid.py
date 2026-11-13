"""Download and validate top-1% pre-tax income shares from WID's bulk CSVs.

The WID official CSV format stores shares as fractions of one. This module
selects a single explicit distributional concept and converts it to percentage
points, never filling missing country-years with zero or interpolated values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

# These are the paper's 18 OECD countries, mapped to WID's two-letter IDs.
# Portugal is intentionally excluded: it is not one of the 18 paper countries.
ISO3_TO_WID: dict[str, str] = {
    "AUS": "AU",
    "AUT": "AT",
    "BEL": "BE",
    "CAN": "CA",
    "CHE": "CH",
    "DEU": "DE",
    "DNK": "DK",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "IRL": "IE",
    "ITA": "IT",
    "JPN": "JP",
    "NLD": "NL",
    "NOR": "NO",
    "NZL": "NZ",
    "SWE": "SE",
    "USA": "US",
}
WID_VARIABLE = "sptinc992j"
WID_PERCENTILE = "p99p100"
START_YEAR = 1965
END_YEAR = 2015
MAX_SOURCE_BYTES = 160_000_000


@dataclass(frozen=True, slots=True)
class TopShare:
    """A country-year with a top-1% share in percentage points."""

    country: str
    year: int
    top1_share_pct: float


@dataclass(frozen=True, slots=True)
class DownloadAudit:
    """Per-country provenance and scope of a downloaded WID bulk CSV."""

    country: str
    source_url: str
    sha256: str
    downloaded_bytes: int
    n_observations: int
    missing_years: tuple[int, ...]


def source_url(country: str) -> str:
    """Build the fixed official WID location only for supported paper countries."""
    if country not in ISO3_TO_WID:
        raise ValueError(f"Unsupported country for paper sample: {country}")
    return f"https://wid.world/bulk_download/WID_data_{ISO3_TO_WID[country]}.csv"


def parse_wid_csv(
    rows: Iterable[dict[str, str]],
    *,
    country: str,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> list[TopShare]:
    """Filter the unique WID series; reject ambiguous units and duplicates."""
    if country not in ISO3_TO_WID or start_year > end_year:
        raise ValueError("Invalid paper-sample country or study year interval")
    expected_iso2 = ISO3_TO_WID[country]
    parsed: dict[int, TopShare] = {}
    for row_index, row in enumerate(rows, start=2):
        if (
            row.get("variable") != WID_VARIABLE
            or row.get("percentile") != WID_PERCENTILE
        ):
            continue
        if row.get("country") != expected_iso2:
            raise ValueError(
                f"WID country mismatch in selected series, row {row_index}: {row.get('country')}"
            )
        try:
            year = int(row["year"])
            fraction = float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid selected WID row {row_index}") from exc
        if not start_year <= year <= end_year:
            continue
        if not math.isfinite(fraction) or not 0.01 <= fraction <= 1.0:
            raise ValueError(
                f"Invalid top-1% WID share (expected a fraction of 1) "
                f"in {country} {year}: {fraction}"
            )
        if year in parsed:
            raise ValueError(f"Duplicate WID top-1% value for {country}, {year}")
        parsed[year] = TopShare(country, year, fraction * 100.0)
    if not parsed:
        raise ValueError(
            f"No WID {WID_VARIABLE}/{WID_PERCENTILE} values in {country} "
            f"during {start_year}–{end_year}"
        )
    return [parsed[year] for year in sorted(parsed)]


def download_country(
    country: str,
    *,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    timeout_seconds: int = 90,
) -> tuple[list[TopShare], DownloadAudit]:
    """Stream an official CSV into a bounded spool; hash every original byte."""
    url = source_url(country)
    request = Request(url, headers={"User-Agent": "research-audit/1.0 WID public data"})
    digest = hashlib.sha256()
    total_bytes = 0
    with tempfile.SpooledTemporaryFile(max_size=8_000_000, mode="w+b") as cache:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                raise ValueError(f"Expected a CSV, received HTML for {country}")
            while chunk := response.read(128 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_SOURCE_BYTES:
                    raise ValueError(
                        f"WID source exceeds {MAX_SOURCE_BYTES} bytes for {country}"
                    )
                digest.update(chunk)
                cache.write(chunk)
        cache.seek(0)
        with io.TextIOWrapper(cache, encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter=";")
            needed = {"country", "variable", "percentile", "year", "value"}
            if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
                raise ValueError(
                    f"Unexpected WID headers for {country}: {reader.fieldnames}"
                )
            selected = parse_wid_csv(
                reader, country=country, start_year=start_year, end_year=end_year
            )
    observed_years = {item.year for item in selected}
    missing = tuple(
        year for year in range(start_year, end_year + 1) if year not in observed_years
    )
    return selected, DownloadAudit(
        country=country,
        source_url=url,
        sha256=digest.hexdigest(),
        downloaded_bytes=total_bytes,
        n_observations=len(selected),
        missing_years=missing,
    )


def write_results(
    destination: Path,
    records: list[TopShare],
    audits: list[DownloadAudit],
    *,
    start_year: int,
    end_year: int,
) -> None:
    """Write compact derived series and a machine-readable provenance manifest."""
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "wid_top1_pretax.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(
            target, fieldnames=["country", "year", "top1_share_pct"]
        )
        writer.writeheader()
        for item in sorted(records, key=lambda entry: (entry.country, entry.year)):
            writer.writerow(
                {
                    "country": item.country,
                    "year": item.year,
                    "top1_share_pct": f"{item.top1_share_pct:.12g}",
                }
            )
    manifest: dict[str, object] = {
        "status": "downloaded_official_wid_bulk_source",
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "source": "World Inequality Database (WID.world) country bulk CSV",
        "series": {"variable": WID_VARIABLE, "percentile": WID_PERCENTILE},
        "measurement": "top 1% pre-tax national income, equal-split adults, percent",
        "conversion": "100 times the original WID fraction of 1",
        "sample_years": [start_year, end_year],
        "extrapolation_flag": "WID source metadata not yet assessed",
        "countries": [asdict(audit) for audit in audits],
        "n_rows": len(records),
    }
    (destination / "wid_top1_provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run a documented, explicit real-data acquisition for selected countries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--countries",
        nargs="+",
        default=sorted(ISO3_TO_WID),
        help="ISO3 paper-country codes; default is all 18",
    )
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    countries = sorted(set(args.countries))
    if len(countries) != len(args.countries):
        parser.error("Duplicate country codes provided")
    if args.start_year > args.end_year:
        parser.error("Start year must precede end year")
    series: list[TopShare] = []
    audits: list[DownloadAudit] = []
    for country in countries:
        observations, audit = download_country(
            country, start_year=args.start_year, end_year=args.end_year
        )
        series.extend(observations)
        audits.append(audit)
        print(
            f"WID {country}: {audit.n_observations} observed years; "
            f"{len(audit.missing_years)} missing; SHA256={audit.sha256}"
        )
    write_results(
        args.output_dir,
        series,
        audits,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"Saved {len(series)} country-year observations to {args.output_dir}")


if __name__ == "__main__":
    main()
