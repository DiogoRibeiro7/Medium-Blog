"""Count agent-handled 311 calls per clock hour, 2016-2019, on the NYC Open Data portal (PLAN.md, "Data").

python fetch.py --sources sources

Writes sources/hourly_calls.csv, sources/FILES (SHA-256) and sources/REVISION (the dataset's update time). If the
counts are already committed, they are verified and kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "wewp-mm3p"
BASE = "https://data.cityofnewyork.us"
YEARS = range(2016, 2020)


def get(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise SystemExit(f"{e.code} for {url}: {e.read().decode('utf-8', 'replace')[:2000]}") from e
            if i == tries - 1:
                raise
            time.sleep(10 * 2**i)
        except Exception as e:  # noqa: BLE001 - retry any transient failure, then give up
            if i == tries - 1:
                raise
            wait = 10 * 2**i
            print(f"  retry in {wait}s after {type(e).__name__}: {e}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


TRUNC = (
    "date_trunc_ymdh(date_time)",
    "date_trunc_ymdh(date_time, 'America/New_York')",
    "date_trunc_ymdh(date_time, 'UTC')",
)


def url(params: dict) -> str:
    return f"{BASE}/resource/{DATASET}.csv?" + urllib.parse.urlencode(params)


def month_query(year: int, month: int, trunc: str) -> str:
    start = f"{year}-{month:02d}-01T00:00:00"
    end = f"{year + (month == 12)}-{month % 12 + 1:02d}-01T00:00:00"
    return url(
        {
            "$select": f"{trunc} AS clock_hour, count(*) AS calls",
            "$where": f"date_time >= '{start}' AND date_time < '{end}'",
            "$group": trunc,
            "$order": trunc,
            "$limit": "5000",
        }
    )


def clock_hour(text: str) -> int | None:
    """Hour of the day from the TIME field, '14:05:09' or '02:05:09 PM'."""
    t = text.strip().upper()
    try:
        h = int(t.split(":")[0])
    except ValueError:
        return None
    if t.endswith("PM") and h != 12:
        h += 12
    if t.endswith("AM") and h == 12:
        h = 0
    return h


def choose_trunc() -> tuple[str, float, int]:
    """The first truncation the portal accepts whose hour matches the TIME field on sampled calls."""
    for trunc in TRUNC:
        rows = []
        for month in ("01", "07"):  # winter and summer time
            q = url(
                {
                    "$select": f"{trunc} AS clock_hour, time",
                    "$where": f"date_time >= '2016-{month}-12T00:00:00' AND date_time < '2016-{month}-13T00:00:00'",
                    "$limit": "300",
                }
            )
            try:
                rows += list(csv.DictReader(io.StringIO(get(q).decode("utf-8"))))
            except SystemExit as e:
                print(f"{trunc}: rejected ({str(e)[:160]})")
                rows = None
                break
        if not rows:
            continue
        pairs = [(int(r["clock_hour"][11:13]), clock_hour(r["time"])) for r in rows if r.get("time")]
        agree = sum(a == b for a, b in pairs) / max(len(pairs), 1)
        print(f"{trunc}: hour matches TIME on {agree:.3f} of {len(pairs)} sampled calls")
        if agree >= 0.99:
            return trunc, agree, len(pairs)
    raise SystemExit("no truncation of date_time matches the TIME field")


def missing_query(year: int) -> str:
    return url(
        {
            "$select": "count(*) AS n",
            "$where": f"date_time IS NULL AND date >= '{year}-01-01T00:00:00' AND date < '{year + 1}-01-01T00:00:00'",
        }
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    a = ap.parse_args()
    a.sources.mkdir(parents=True, exist_ok=True)
    out = a.sources / "hourly_calls.csv"
    files = a.sources / "FILES"
    if out.exists() and files.exists():
        name, digest = files.read_text(encoding="utf-8").split()[:2]
        if digest != sha256(out):
            raise SystemExit("sources/hourly_calls.csv does not match sources/FILES")
        print("using the committed counts")
        return
    meta = json.loads(get(f"{BASE}/api/views/{DATASET}.json"))
    trunc, agree, sampled = choose_trunc()
    rows = []
    for y in YEARS:
        for m in range(1, 13):
            body = get(month_query(y, m, trunc)).decode("utf-8")
            part = list(csv.DictReader(io.StringIO(body)))
            rows += [(r["clock_hour"][:13].replace("T", " "), int(r["calls"])) for r in part]
            print(f"{y}-{m:02d}: {len(part)} hours, {sum(int(r['calls']) for r in part)} calls")
    missing = {}
    for y in YEARS:
        body = get(missing_query(y)).decode("utf-8")
        missing[y] = int(next(csv.DictReader(io.StringIO(body)))["n"])
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hour", "calls"])
        w.writerows(rows)
    files.write_text(f"hourly_calls.csv {sha256(out)}\n", encoding="utf-8")
    (a.sources / "REVISION").write_text(
        json.dumps(
            {
                "rowsUpdatedAt": meta.get("rowsUpdatedAt"),
                "truncation": trunc,
                "hour_matches_time_field": agree,
                "sampled_calls": sampled,
                "missing_date_time": missing,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(rows)} hours written; rows without date_time by year: {missing}")


if __name__ == "__main__":
    main()
