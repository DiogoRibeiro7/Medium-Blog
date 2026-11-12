"""Tenure, previous change and rounding of fortunes near $1 billion (Part 5).

Part 4 found that fortunes just above $1 billion leave the Forbes list more
often than a model built on the year's changes of large fortunes predicts, and
conjectured that they are more volatile. Two other explanations need data that
``fetch_forbes_flows.py`` does not keep: people just above the line may have
arrived there recently on a lucky year and fall back (selection on a past
shock), and net worth is rounded, which adds proportionally more noise to small
fortunes. This script requests the same annual lists, links them in memory with
the rules of ``fetch_forbes_flows.py`` and writes only aggregate counts to a new
file, leaving Part 4's data untouched. No person-level record is written.

For a person on list t:

* ``tenure`` is ``incumbent`` (on list t-1), ``returning`` (on an earlier list
  but not t-1), ``new`` (on no earlier list since the first year fetched) or
  ``unknown`` (t is the first year);
* ``prev`` buckets the change in log10 net worth from t-1 to t for incumbents
  (``none`` otherwise);
* the outcome is ``stay`` (on list t+1, with the change in log10 net worth in
  bins of 0.02) or ``exit``.

Output ``forbes_tenure.csv.gz``: year, table, band_lo_musd, tenure, prev, key, n.
Tables: ``stock`` (people on list t), ``exit`` (not on t+1), ``logchange``
(stayers, ``key`` = bin lower edge x 1000), ``rounding`` (``key`` = ``m100`` if
net worth is a multiple of $100 million, ``m10`` if of $10 million, else
``other``). Bands are those of ``FLOW_EDGES``.

Manifest ``forbes_tenure_manifest.json``: per year URL, bytes and SHA-256 of the
raw response, and whether it is byte-identical to the response Part 4 used.

Run in GitHub Actions (the analysis workspace cannot reach forbes.com)::

    python fetch_forbes_tenure.py --first 2010 --last 2025 --output-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fetch_forbes_flows import (FLOW_EDGES, LOG_BIN, LOG_RANGE, URL, Person, _keys, _write, band, extract_people,
                                fetch_bytes, link)

PREV_EDGES = [-math.inf, -0.1, -0.03, 0.03, 0.1]
PREV_LABELS = ["fell >0.1", "fell 0.03-0.1", "flat", "rose 0.03-0.1", "rose >0.1"]
COLUMNS = ["year", "table", "band_lo_musd", "tenure", "prev", "key", "n"]


def _prev_bucket(change: float) -> str:
    k = max(i for i, e in enumerate(PREV_EDGES) if change >= e)
    return PREV_LABELS[k]


def _bin(change: float) -> int:
    clipped = max(-LOG_RANGE, min(LOG_RANGE, change))
    return int(round(math.floor(round(clipped / LOG_BIN, 9)) * LOG_BIN * 1000))


def _rounding(worth: float) -> str:
    if abs(worth / 100 - round(worth / 100)) < 1e-9:
        return "m100"
    if abs(worth / 10 - round(worth / 10)) < 1e-9:
        return "m10"
    return "other"


def aggregate(lists: dict[int, list[Person]]) -> list[dict[str, Any]]:
    """All tenure tables from linked yearly lists (pure function, used by the tests)."""
    years = sorted(lists)
    links = {y: link(lists[y], lists[y + 1])[0] for y in years if y + 1 in lists}
    inverse = {y + 1: {j: i for i, j in m.items()} for y, m in links.items()}
    rows: Counter[tuple[int, str, int, str, str, str]] = Counter()
    for year in years:
        earlier = set()
        for y in years:
            if y < year - 1:
                for p in lists[y]:
                    earlier |= _keys(p)
        prev_map = inverse.get(year, {})
        for i, person in enumerate(lists[year]):
            if person.worth is None:
                continue
            b = band(person.worth, FLOW_EDGES)
            if year == years[0] or (year - 1) not in lists:
                tenure, prev = "unknown", "none"
            elif i in prev_map:
                tenure = "incumbent"
                before = lists[year - 1][prev_map[i]]
                prev = (_prev_bucket(math.log10(person.worth / before.worth))
                        if before.worth and before.worth > 0 and person.worth > 0 else "none")
            elif _keys(person) & earlier:
                tenure, prev = "returning", "none"
            else:
                tenure, prev = "new", "none"
            rows[(year, "rounding", b, "all", "none", _rounding(person.worth))] += 1
            if year not in links:
                continue
            rows[(year, "stock", b, tenure, prev, "")] += 1
            if i in links[year]:
                after = lists[year + 1][links[year][i]]
                if after.worth and after.worth > 0 and person.worth > 0:
                    key = str(_bin(math.log10(after.worth / person.worth)))
                    rows[(year, "logchange", b, tenure, prev, key)] += 1
            else:
                rows[(year, "exit", b, tenure, prev, "")] += 1
    return [dict(zip(COLUMNS, k + (n,))) for k, n in sorted(rows.items())]


def fetch(first: int, last: int, output_dir: Path, fetcher: Callable[..., bytes] = fetch_bytes,
          pause: float = 3.0) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = {}
    old = output_dir / "forbes_flows_manifest.json"
    if old.exists():
        previous = {y: e.get("sha256") for y, e in json.loads(old.read_text(encoding="utf-8"))["years"].items()}
    manifest: dict[str, Any] = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "url_pattern": URL, "flow_edges_musd": FLOW_EDGES, "logchange_bin": LOG_BIN,
                                "prev_edges": [e for e in PREV_EDGES if math.isfinite(e)], "prev_labels": PREV_LABELS,
                                "years": {}}
    lists: dict[int, list[Person]] = {}
    for year in range(first, last + 1):
        url = URL.format(year=year)
        entry: dict[str, Any] = {"url": url}
        try:
            raw = fetcher(url, name=f"forbes_{year}", timeout_seconds=180, max_attempts=3, wait_seconds=15)
            sha = hashlib.sha256(raw).hexdigest()
            entry.update(bytes=len(raw), sha256=sha, same_as_part4=previous.get(str(year)) == sha)
            records, _ = extract_people(json.loads(raw))
            lists[year] = [Person(r, year) for r in records]
            entry["n_persons"] = len(lists[year])
        except Exception as exc:  # one failed year should not stop the others
            entry["error"] = repr(exc)[:500]
        manifest["years"][str(year)] = entry
        print(f"{year}: {entry.get('n_persons', 'ERROR')} people, same as Part 4: {entry.get('same_as_part4')}",
              flush=True)
        time.sleep(pause)
    manifest["tenure_sha256"] = _write(aggregate(lists), COLUMNS, output_dir / "forbes_tenure.csv.gz")
    manifest["n_years_ok"] = len(lists)
    (output_dir / "forbes_tenure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=int, default=2010)
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    manifest = fetch(args.first, args.last, args.output_dir)
    return 0 if manifest["n_years_ok"] >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
