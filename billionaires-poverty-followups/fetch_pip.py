"""Download World Bank PIP survey estimates across a grid of poverty lines.

This is the only networked step of the billionaire-poverty follow-up series.
It queries the public Poverty and Inequality Platform API (2021 PPP, survey
years only, no gap filling) once per poverty line, keeps the exact response
bytes for hashing, and writes two compressed tables:

``data/pip_surveys.csv.gz``
    One row per survey estimate (country, reporting year, welfare type,
    reporting level) with survey metadata: mean, median, Gini, decile shares,
    coverage, comparability spell and welfare reference time.

``data/pip_headcounts.csv.gz``
    Long table of headcount and poverty gap for every survey estimate at
    every poverty line in ``POVERTY_LINES``.

``data/pip_manifest.json`` records the API release, URLs, retrieval time and
SHA-256 of each raw response. Run in GitHub Actions (the analysis workspace
has no access to api.worldbank.org)::

    python fetch_pip.py --output-dir data
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "billionaires-poverty-audit"))
from resilient_fetch import fetch_bytes  # noqa: E402

API = "https://api.worldbank.org/pip/v1"
PPP_VERSION = "2021"
# 2021-PPP dollars per person per day. Includes the World Bank's $3.00, $4.20
# and $8.30 lines and the $30 line used by the first article.
POVERTY_LINES: tuple[float, ...] = (
    2.0, 3.0, 4.2, 5.0, 6.0, 8.3, 10.0, 12.0, 15.0, 17.5, 20.0, 25.0,
    30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 85.0, 100.0,
)
KEY = ["country_code", "reporting_year", "welfare_type", "reporting_level"]
SURVEY_COLUMNS = KEY + [
    "country_name", "region_code", "survey_acronym", "survey_coverage", "survey_year",
    "welfare_time", "survey_comparability", "comparable_spell", "distribution_type",
    "is_interpolated", "estimation_type", "mean", "median", "gini", "mld", "polarization",
    "decile1", "decile2", "decile3", "decile4", "decile5", "decile6", "decile7",
    "decile8", "decile9", "decile10", "ppp", "cpi", "reporting_pop", "reporting_gdp",
    "reporting_pce",
]
LINE_COLUMNS = KEY + ["poverty_line", "headcount", "poverty_gap", "poverty_severity", "watts"]


def pip_url(poverty_line: float) -> str:
    """URL of the survey-year PIP estimates at one poverty line."""
    return (f"{API}/pip?country=all&year=all&povline={poverty_line:g}&fill_gaps=false"
            f"&ppp_version={PPP_VERSION}&format=csv")


def parse_pip_csv(raw: bytes, poverty_line: float) -> pd.DataFrame:
    """Parse one PIP CSV response and check that it answers the request."""
    frame = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in KEY + ["poverty_line", "headcount", "mean", "median"] if c not in frame]
    if missing:
        raise ValueError(f"PIP response at {poverty_line:g} lacks columns {missing}")
    if frame.empty:
        raise ValueError(f"PIP response at {poverty_line:g} is empty")
    lines = frame.poverty_line.dropna().unique()
    if len(lines) != 1 or abs(float(lines[0]) - poverty_line) > 1e-9:
        raise ValueError(f"PIP response at {poverty_line:g} reports poverty lines {lines}")
    if not frame.headcount.dropna().between(0, 1).all():
        raise ValueError("headcount outside [0, 1]")
    if frame.duplicated(KEY).any():
        raise ValueError(f"duplicate survey keys at {poverty_line:g}")
    return frame


def build_tables(responses: dict[float, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine per-line responses into a survey table and a long headcount table."""
    first = responses[min(responses)]
    surveys = first[[c for c in SURVEY_COLUMNS if c in first]].copy()
    keys = set(map(tuple, surveys[KEY].astype(str).to_numpy()))
    long = []
    for line, frame in sorted(responses.items()):
        other = set(map(tuple, frame[KEY].astype(str).to_numpy()))
        if other != keys:
            raise ValueError(f"survey set at {line:g} differs from the set at {min(responses):g}")
        long.append(frame[[c for c in LINE_COLUMNS if c in frame]])
    headcounts = pd.concat(long, ignore_index=True).sort_values(KEY + ["poverty_line"])
    # Headcounts must be non-decreasing in the poverty line within each survey.
    diffs = headcounts.groupby(KEY, sort=False).headcount.diff().dropna()
    if (diffs < -1e-9).any():
        raise ValueError("headcount decreases as the poverty line rises")
    return surveys.sort_values(KEY).reset_index(drop=True), headcounts.reset_index(drop=True)


def _write_gz(frame: pd.DataFrame, path: Path) -> str:
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(output_dir: Path, fetcher: Callable[..., bytes] = fetch_bytes,
          lines: Sequence[float] = POVERTY_LINES) -> dict[str, Any]:
    """Download every line, validate, and write tables plus a manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_pip"
    raw_dir.mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api": API, "ppp_version": PPP_VERSION, "fill_gaps": False, "responses": [],
    }
    for name, url in (("versions", f"{API}/versions?format=json"), ("pip_info", f"{API}/pip-info")):
        try:
            manifest[name] = json.loads(fetcher(url, name=name, max_attempts=3).decode("utf-8"))
        except Exception as exc:  # metadata is informative, not required
            manifest[name] = {"error": repr(exc)}
    responses = {}
    for line in lines:
        url = pip_url(line)
        raw = fetcher(url, name=f"pip_{line:g}", timeout_seconds=240, max_attempts=4, wait_seconds=10)
        (raw_dir / f"pip_{line:g}.csv.gz").write_bytes(gzip.compress(raw, mtime=0))
        responses[line] = parse_pip_csv(raw, line)
        manifest["responses"].append({"poverty_line": line, "url": url, "bytes": len(raw),
                                      "sha256": hashlib.sha256(raw).hexdigest(),
                                      "rows": int(len(responses[line]))})
        print(f"$ {line:g}: {len(responses[line])} survey estimates", flush=True)
    surveys, headcounts = build_tables(responses)
    manifest["pip_surveys_sha256"] = _write_gz(surveys, output_dir / "pip_surveys.csv.gz")
    manifest["pip_headcounts_sha256"] = _write_gz(headcounts, output_dir / "pip_headcounts.csv.gz")
    manifest["n_survey_estimates"] = int(len(surveys))
    manifest["n_countries"] = int(surveys.country_code.nunique())
    manifest["welfare_type_counts"] = {k: int(v) for k, v in surveys.welfare_type.value_counts().items()}
    (output_dir / "pip_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    manifest = fetch(args.output_dir)
    print(json.dumps({k: v for k, v in manifest.items() if k != "responses"}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
