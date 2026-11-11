"""Country totals of the Forbes billionaire lists, year by year (Part 3).

For each year the script requests Forbes's annual list endpoint
``/forbesapi/person/billionaires/{year}/position/true.json`` and keeps only
country aggregates: the number of people and their summed net worth by
``countryOfCitizenship`` and by ``country`` (read as residence in Part 2).
No person-level record is written. To detect an endpoint that ignores the year
and returns the current list, each year's set of person identifiers is
compared with the latest year's in memory, and only the overlap share is kept.

Outputs:

``data/forbes_history_counts.csv.gz``
    year, field (citizenship | residence), country_label, n, worth_musd.

``data/forbes_history_manifest.json``
    Per year: URL, bytes, SHA-256 of the raw response, number of people, unique
    identifiers, overlap with the latest year, record keys, error if any.

Run in GitHub Actions (the analysis workspace cannot reach forbes.com)::

    python fetch_forbes_history.py --first 1997 --last 2026 --output-dir data
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "billionaires-poverty-audit"))
from forbes_pages import extract_people  # noqa: E402
from resilient_fetch import fetch_bytes  # noqa: E402

URL = "https://www.forbes.com/forbesapi/person/billionaires/{year}/position/true.json?limit=5000"
FIELDS = {"citizenship": "countryOfCitizenship", "residence": "country"}


def _label(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _worth(person: dict[str, Any]) -> float | None:
    for key in ("finalWorth", "netWorth", "worth"):
        value = person.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def aggregate(people: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Country totals for one year, plus key-level diagnostics (no records)."""
    rows = []
    info: dict[str, Any] = {"n_persons": len(people)}
    for field, key in FIELDS.items():
        n: Counter[str] = Counter()
        worth: defaultdict[str, float] = defaultdict(float)
        missing = 0
        for person in people:
            label = _label(person.get(key))
            if label is None:
                missing += 1
                continue
            n[label] += 1
            w = _worth(person)
            if w is not None:
                worth[label] += w
        info[f"{field}_missing"] = missing
        info[f"{field}_labels"] = len(n)
        rows += [{"field": field, "country_label": c, "n": k, "worth_musd": worth[c]} for c, k in sorted(n.items())]
    keys: Counter[str] = Counter(k for person in people for k in person)
    info["record_keys"] = sorted(keys)
    years = Counter(person.get("year") for person in people if isinstance(person.get("year"), int))
    info["year_field_values"] = {str(k): v for k, v in years.most_common(5)}
    info["worth_missing"] = sum(_worth(person) is None for person in people)
    return rows, info


def fetch(first: int, last: int, output_dir: Path, fetcher: Callable[..., bytes] = fetch_bytes,
          pause: float = 3.0) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "url_pattern": URL, "years": {}}
    rows: list[dict[str, Any]] = []
    uris: dict[int, set[str]] = {}
    for year in range(last, first - 1, -1):
        url = URL.format(year=year)
        entry: dict[str, Any] = {"url": url}
        try:
            raw = fetcher(url, name=f"forbes_{year}", timeout_seconds=180, max_attempts=3, wait_seconds=15)
            entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            people, path = extract_people(json.loads(raw))
            entry["json_path"] = path
            year_rows, info = aggregate(people)
            entry.update(info)
            ids = {p["uri"].strip() for p in people if isinstance(p.get("uri"), str) and p["uri"].strip()}
            entry["n_unique_uri"] = len(ids)
            uris[year] = ids
            rows += [{"year": year, **r} for r in year_rows]
        except Exception as exc:  # one failed year should not stop the others
            entry["error"] = repr(exc)[:500]
        manifest["years"][str(year)] = entry
        print(f"{year}: {entry.get('n_persons', 'ERROR')} people"
              f"{'' if 'error' not in entry else ' ' + entry['error'][:160]}", flush=True)
        time.sleep(pause)
    if uris:
        latest = max(uris)
        for year, ids in uris.items():
            manifest["years"][str(year)]["overlap_with_latest"] = (
                len(ids & uris[latest]) / len(ids) if ids else None)
        manifest["latest_year"] = latest
    table = pd.DataFrame(rows, columns=["year", "field", "country_label", "n", "worth_musd"])
    data = table.to_csv(index=False, lineterminator="\n").encode("utf-8")
    (output_dir / "forbes_history_counts.csv.gz").write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    manifest["counts_sha256"] = hashlib.sha256((output_dir / "forbes_history_counts.csv.gz").read_bytes()).hexdigest()
    manifest["n_years_ok"] = int(len(uris))
    (output_dir / "forbes_history_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=int, default=1997)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    manifest = fetch(args.first, args.last, args.output_dir)
    summary = {y: {k: e.get(k) for k in ("n_persons", "overlap_with_latest", "citizenship_labels", "error")}
               for y, e in manifest["years"].items()}
    print(json.dumps(summary, indent=1))
    return 0 if manifest["n_years_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
