"""Extract World Inequality Database wealth and income shares from the bulk file.

Second networked step of the billionaire-poverty follow-up series (Part 2).
The GitHub Actions workflow downloads the WID bulk archive
(``https://wid.world/bulk_download/wid_all_data.zip``) with curl; this script
reads it without unpacking everything, keeps a small set of national series
and writes:

``data/wid_series.csv.gz``
    Long table: country (WID two-letter code), variable (full WID code, e.g.
    ``shweal992j``), percentile, year, value. Only the variables in
    ``VARIABLES`` and percentiles in ``PERCENTILES``, years from ``MIN_YEAR``.

``data/wid_metadata.csv.gz``
    The WID metadata rows for those variables (source, method, imputation
    notes), one row per country and variable.

``data/wid_manifest.json``
    Archive SHA-256 and size, member names, retrieval time, row counts and the
    SHA-256 of each output table.

Usage (GitHub Actions)::

    curl -L --retry 5 -o wid_all_data.zip https://wid.world/bulk_download/wid_all_data.zip
    python fetch_wid.py --zip wid_all_data.zip --output-dir data
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

WID_URL = "https://wid.world/bulk_download/wid_all_data.zip"
MIN_YEAR = 1980
# Full WID codes: 6-letter concept, 3-digit age group, 1-letter population unit.
VARIABLES: dict[str, str] = {
    "shweal992j": "net personal wealth share, adults, equal-split",
    "ghweal992j": "net personal wealth Gini, adults, equal-split",
    "ahweal992j": "average net personal wealth, adults, equal-split (local currency, constant prices)",
    "sptinc992j": "pre-tax national income share, adults, equal-split",
    "gptinc992j": "pre-tax national income Gini, adults, equal-split",
    "anninc992i": "average national income per adult (local currency, constant prices)",
    "whweal992i": "net personal wealth to net national income ratio",
    "xlcusp999i": "PPP conversion factor, local currency per US dollar",
    "npopul992i": "adult population",
}
PERCENTILES: tuple[str, ...] = (
    "p0p100", "p0p50", "p50p90", "p90p100", "p99p100", "p99.9p100", "p99.99p100",
)
DATA_MEMBER = re.compile(r"(?:^|/)WID_data_([A-Z]{2})\.csv$")
META_MEMBER = re.compile(r"(?:^|/)WID_metadata_([A-Z]{2})\.csv$")
FULL_CODE = re.compile(r"^[a-z]{6}\d{3}[a-z]$")


def full_codes(frame: pd.DataFrame) -> pd.Series:
    """Return full WID codes whether the file stores them whole or split.

    Some WID exports put the full code (``shweal992j``) in ``variable``; others
    put the six-letter concept there and the age and population unit in
    separate ``age`` and ``pop`` columns.
    """
    var = frame["variable"].astype(str).str.strip()
    if var.str.match(FULL_CODE).all():
        return var
    if {"age", "pop"} <= set(frame.columns):
        age = frame["age"].astype(str).str.strip().str.zfill(3)
        return var + age + frame["pop"].astype(str).str.strip()
    raise ValueError("cannot build full WID variable codes: no age/pop columns")


def parse_data_member(raw: bytes, country: str) -> pd.DataFrame:
    """Parse one ``WID_data_XX.csv`` member and keep the selected series."""
    frame = pd.read_csv(io.BytesIO(raw), sep=";", dtype=str, keep_default_na=False)
    required = {"country", "variable", "percentile", "year", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"WID_data_{country}.csv lacks columns {sorted(missing)}")
    frame = frame.assign(variable=full_codes(frame))
    keep = frame["variable"].isin(VARIABLES) & frame["percentile"].isin(PERCENTILES)
    out = frame.loc[keep, ["country", "variable", "percentile", "year", "value"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out = out[out["year"] >= MIN_YEAR]
    out["value"] = pd.to_numeric(out["value"].replace("", pd.NA), errors="coerce")
    out = out.dropna(subset=["value"])
    if (out["country"] != country).any():
        raise ValueError(f"WID_data_{country}.csv contains rows for other countries")
    if out.duplicated(["country", "variable", "percentile", "year"]).any():
        raise ValueError(f"duplicate series rows in WID_data_{country}.csv")
    shares = out[out["variable"].str.startswith("s")]
    if ((shares["value"] < -0.5) | (shares["value"] > 1.0 + 1e-9)).any():
        raise ValueError(f"share outside [-0.5, 1] in WID_data_{country}.csv")
    return out


def parse_meta_member(raw: bytes) -> pd.DataFrame:
    """Parse one ``WID_metadata_XX.csv`` member and keep the selected variables."""
    frame = pd.read_csv(io.BytesIO(raw), sep=";", dtype=str, keep_default_na=False)
    if "variable" not in frame.columns:
        raise ValueError("WID metadata lacks a variable column")
    frame = frame.assign(variable=full_codes(frame))
    return frame[frame["variable"].isin(VARIABLES)].copy()


def _write_gz(frame: pd.DataFrame, path: Path) -> str:
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(zip_path: Path, output_dir: Path) -> dict[str, Any]:
    """Read the bulk archive, validate, and write the tables and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with zip_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    series, metadata, members = [], [], []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for name in sorted(names):
            if match := DATA_MEMBER.search(name):
                part = parse_data_member(archive.read(name), match.group(1))
                if len(part):
                    series.append(part)
                members.append(name)
            elif META_MEMBER.search(name):
                part = parse_meta_member(archive.read(name))
                if len(part):
                    metadata.append(part)
        other = [n for n in names if not DATA_MEMBER.search(n) and not META_MEMBER.search(n)]
    if not series:
        raise ValueError("no WID_data_XX.csv members with the selected series")
    table = pd.concat(series, ignore_index=True).sort_values(
        ["country", "variable", "percentile", "year"]).reset_index(drop=True)
    meta = pd.concat(metadata, ignore_index=True) if metadata else pd.DataFrame()
    us = table[(table.country == "US") & (table.variable == "shweal992j")
               & (table.percentile == "p99p100")]
    if us.empty:
        raise ValueError("US top 1% wealth share missing: the archive layout has changed")
    manifest: dict[str, Any] = {
        "source_url": WID_URL,
        "processed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive_bytes": zip_path.stat().st_size,
        "archive_sha256": digest.hexdigest(),
        "n_data_members": len(members),
        "other_members": other[:50],
        "variables": VARIABLES,
        "percentiles": list(PERCENTILES),
        "min_year": MIN_YEAR,
        "n_rows": int(len(table)),
        "n_countries": int(table.country.nunique()),
        "latest_year_by_variable": {v: int(g.year.max()) for v, g in table.groupby("variable")},
        "metadata_columns": list(meta.columns),
        "wid_series_sha256": _write_gz(table, output_dir / "wid_series.csv.gz"),
        "wid_metadata_sha256": _write_gz(meta, output_dir / "wid_metadata.csv.gz"),
    }
    (output_dir / "wid_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                  encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True, help="path to wid_all_data.zip")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--retrieved-utc", default=None,
                        help="download time recorded by the workflow")
    parser.add_argument("--last-modified", default=None,
                        help="Last-Modified header of the archive, if the server sent one")
    args = parser.parse_args(argv)
    manifest = extract(args.zip, args.output_dir)
    manifest["retrieved_utc"] = args.retrieved_utc
    manifest["last_modified"] = args.last_modified
    (args.output_dir / "wid_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                       encoding="utf-8")
    print(json.dumps(manifest, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
