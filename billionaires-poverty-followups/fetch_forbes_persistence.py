"""Each fortune's own past volatility and its departures from the Forbes list (Part 6).

Part 5 conjectured that volatility is a stable property of individual fortunes,
and that a fortune's own past volatility predicts its chance of leaving the
list better than its distance from $1 billion. Testing that needs, for every
person on list t, a measure of volatility over several earlier lists, which no
earlier file keeps. This script requests the same annual lists, links them in
memory with the rules of ``fetch_forbes_flows.py`` and writes only counts, and
sums of squared changes, by cell. No person-level record or identifier is
written, and the files of Parts 4 and 5 are left untouched.

For a person on list t the links are followed back for up to four lists. Let
c1 be the change in log10 net worth from t-1 to t, c2 the change from t-2 to
t-1, and so on.

* ``hist``: the number of consecutive earlier changes found (0 to 4); 0 for
  anyone not on list t-1.
* ``prev``: the bucket of c1, with Part 5's labels; ``none`` if hist = 0.
* ``vol``: the root mean square of c2..c_hist, the earlier changes without the
  latest one, in classes whose lower edges are ``VOL_EDGES`` (stored x1000);
  -1 if hist < 2. The latest change is kept apart because Part 5 found that a
  recent rise is followed by a fall back, which is selection, not volatility.
* ``trend``: the mean of c2..c_hist: ``down`` (below -0.03), ``flat`` or ``up``
  (above 0.03); ``none`` if hist < 2.
* ``age``: ``<60``, ``60-79``, ``80+`` or ``unknown`` on list t.
* ``tenure``: as in Part 5 (incumbent, returning, new, unknown).

The outcome is ``exit`` (not on list t+1) or a stay, with the change c0 from t
to t+1 in bins of 0.02.

Output ``forbes_persistence.csv.gz`` with columns year, table, band_lo_musd,
age, tenure, hist, prev, vol, trend, key, n, ms_sum, c1sq_sum:

* ``stock`` and ``exit``: every dimension above; ``ms_sum`` is the sum over the
  cell's people of the mean square of c2..c_hist and ``c1sq_sum`` the sum of
  c1 squared.
* ``logchange``: stayers by year, band, tenure, hist, prev and vol, with
  ``key`` = lower edge of the c0 bin x1000 (age and trend are ``all``).
* ``industry_stock`` and ``industry_exit``: by year, band, hist and vol, with
  ``key`` = Forbes's ``category`` label (the other dimensions are ``all``).

Bands are those of ``FLOW_EDGES``. Manifest ``forbes_persistence_manifest.json``:
per year the URL, bytes and SHA-256 of the raw response, whether it is
byte-identical to the response Part 5 used, and counts by hist and industry.

Run in GitHub Actions (the analysis workspace cannot reach forbes.com)::

    python fetch_forbes_persistence.py --first 2010 --last 2025 --output-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fetch_forbes_flows import FLOW_EDGES, LOG_BIN, URL, Person, _keys, _write, band, extract_people, fetch_bytes, link
from fetch_forbes_tenure import PREV_EDGES, PREV_LABELS, _bin, _prev_bucket

MAX_HIST = 4
VOL_EDGES = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22, 0.30]
TREND_CUT = 0.03
COLUMNS = ["year", "table", "band_lo_musd", "age", "tenure", "hist", "prev", "vol", "trend", "key", "n", "ms_sum",
           "c1sq_sum"]


def vol_class(rms: float) -> int:
    k = max(i for i, e in enumerate(VOL_EDGES) if rms >= e - 1e-12)
    return int(round(VOL_EDGES[k] * 1000))


def age_group(age: int | None) -> str:
    if age is None:
        return "unknown"
    return "<60" if age < 60 else "60-79" if age < 80 else "80+"


def trend_label(mean: float) -> str:
    return "down" if mean < -TREND_CUT else "up" if mean > TREND_CUT else "flat"


def industry(record: dict[str, Any]) -> str:
    value = record.get("category")
    if not (isinstance(value, str) and value.strip()):
        listed = record.get("industries")
        value = listed[0] if isinstance(listed, list) and listed and isinstance(listed[0], str) else None
    return " ".join(value.split()) if isinstance(value, str) and value.strip() else "unknown"


def history(lists: dict[int, list[Person]], inverse: dict[int, dict[int, int]], year: int, i: int) -> list[float]:
    """Consecutive earlier changes c1, c2, ... (most recent first) for person i on list ``year``."""
    changes: list[float] = []
    y, idx = year, i
    while len(changes) < MAX_HIST and idx in inverse.get(y, {}):
        j = inverse[y][idx]
        now, before = lists[y][idx].worth, lists[y - 1][j].worth
        if not (now and before and now > 0 and before > 0):
            break
        changes.append(math.log10(now / before))
        y, idx = y - 1, j
    return changes


def aggregate(lists: dict[int, list[Person]], industries: dict[int, list[str]] | None = None) -> list[dict[str, Any]]:
    """All persistence tables from linked yearly lists (pure function, used by the tests)."""
    years = sorted(lists)
    links = {y: link(lists[y], lists[y + 1])[0] for y in years if y + 1 in lists}
    inverse = {y + 1: {j: i for i, j in m.items()} for y, m in links.items()}
    counts: Counter[tuple] = Counter()
    ms: defaultdict[tuple, float] = defaultdict(float)
    c1sq: defaultdict[tuple, float] = defaultdict(float)
    for year in years:
        if year not in links:
            continue
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
                tenure = "unknown"
            elif i in prev_map:
                tenure = "incumbent"
            elif _keys(person) & earlier:
                tenure = "returning"
            else:
                tenure = "new"
            changes = history(lists, inverse, year, i)
            hist = len(changes)
            prev = _prev_bucket(changes[0]) if hist else "none"
            earlier_changes = changes[1:]
            if earlier_changes:
                mean_sq = sum(c * c for c in earlier_changes) / len(earlier_changes)
                vol = vol_class(math.sqrt(mean_sq))
                trend = trend_label(sum(earlier_changes) / len(earlier_changes))
            else:
                mean_sq, vol, trend = 0.0, -1, "none"
            first_sq = changes[0] ** 2 if hist else 0.0
            age = age_group(person.age)
            stayed = i in links[year]
            for table in ("stock",) + (() if stayed else ("exit",)):
                k = (year, table, b, age, tenure, hist, prev, vol, trend, "")
                counts[k] += 1
                ms[k] += mean_sq
                c1sq[k] += first_sq
            if stayed:
                after = lists[year + 1][links[year][i]]
                if after.worth and after.worth > 0 and person.worth > 0:
                    k = (year, "logchange", b, "all", tenure, hist, prev, vol, "all",
                         str(_bin(math.log10(after.worth / person.worth))))
                    counts[k] += 1
                    ms[k] += mean_sq
                    c1sq[k] += first_sq
            label = industries[year][i] if industries and year in industries else "unknown"
            for table in ("industry_stock",) + (() if stayed else ("industry_exit",)):
                k = (year, table, b, "all", "all", hist, "all", vol, "all", label)
                counts[k] += 1
                ms[k] += mean_sq
                c1sq[k] += first_sq
    return [dict(zip(COLUMNS, k + (n, round(ms[k], 8), round(c1sq[k], 8)))) for k, n in sorted(counts.items(),
            key=lambda kv: tuple(str(x) for x in kv[0]))]


def fetch(first: int, last: int, output_dir: Path, fetcher: Callable[..., bytes] = fetch_bytes,
          pause: float = 3.0) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = {}
    old = output_dir / "forbes_tenure_manifest.json"
    if old.exists():
        previous = {y: e.get("sha256") for y, e in json.loads(old.read_text(encoding="utf-8"))["years"].items()}
    manifest: dict[str, Any] = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "url_pattern": URL, "flow_edges_musd": FLOW_EDGES, "logchange_bin": LOG_BIN,
                                "prev_edges": [e for e in PREV_EDGES if math.isfinite(e)], "prev_labels": PREV_LABELS,
                                "vol_edges": VOL_EDGES, "trend_cut": TREND_CUT, "max_hist": MAX_HIST, "years": {}}
    lists: dict[int, list[Person]] = {}
    labels: dict[int, list[str]] = {}
    for year in range(first, last + 1):
        url = URL.format(year=year)
        entry: dict[str, Any] = {"url": url}
        try:
            raw = fetcher(url, name=f"forbes_{year}", timeout_seconds=180, max_attempts=3, wait_seconds=15)
            sha = hashlib.sha256(raw).hexdigest()
            entry.update(bytes=len(raw), sha256=sha, same_as_part5=previous.get(str(year)) == sha)
            records, _ = extract_people(json.loads(raw))
            lists[year] = [Person(r, year) for r in records]
            labels[year] = [industry(r) for r in records]
            entry["n_persons"] = len(lists[year])
            entry["industry_counts"] = dict(sorted(Counter(labels[year]).items()))
        except Exception as exc:  # one failed year should not stop the others
            entry["error"] = repr(exc)[:500]
        manifest["years"][str(year)] = entry
        print(f"{year}: {entry.get('n_persons', 'ERROR')} people, same as Part 5: {entry.get('same_as_part5')}",
              flush=True)
        time.sleep(pause)
    rows = aggregate(lists, labels)
    by_hist: Counter[tuple[int, int]] = Counter()
    for r in rows:
        if r["table"] == "stock":
            by_hist[(r["year"], r["hist"])] += r["n"]
    for year, entry in manifest["years"].items():
        entry["stock_by_hist"] = {str(h): n for (y, h), n in sorted(by_hist.items()) if y == int(year)}
    manifest["persistence_sha256"] = _write(rows, COLUMNS, output_dir / "forbes_persistence.csv.gz")
    manifest["n_years_ok"] = len(lists)
    (output_dir / "forbes_persistence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                                 encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=int, default=2010)
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    manifest = fetch(args.first, args.last, args.output_dir)
    return 0 if manifest["n_years_ok"] >= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
