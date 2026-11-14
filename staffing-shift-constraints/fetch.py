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


def url(params: dict) -> str:
    return f"{BASE}/resource/{DATASET}.csv?" + urllib.parse.urlencode(params)


def hour_pattern(h: int) -> str:
    """SoQL condition on the TIME text ('12:00:16 AM') for clock hour h, with or without a leading zero."""
    h12 = 12 if h % 12 == 0 else h % 12
    half = "AM" if h < 12 else "PM"
    pats = {f"{h12:02d}:%{half}", f"{h12}:%{half}"}
    return "(" + " OR ".join(f"time like '{p}'" for p in sorted(pats)) + ")"


def year_where(year: int) -> str:
    return f"date >= '{year}-01-01T00:00:00' AND date < '{year + 1}-01-01T00:00:00'"


def by_date(where: str) -> dict[str, int]:
    q = url({"$select": "date, count(*) AS calls", "$where": where, "$group": "date", "$limit": "5000"})
    rows = csv.DictReader(io.StringIO(get(q).decode("utf-8")))
    return {r["date"][:10]: int(r["calls"]) for r in rows}


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
    sample = get(url({"$select": "date, time, date_time", "$limit": "3", "$order": "date_time"})).decode("utf-8")
    print("field formats (3 calls):", sample.replace("\n", " | "))
    rows, check = [], {}
    for y in YEARS:
        total = by_date(year_where(y))
        per_hour = {}
        for h in range(24):
            for day, n in by_date(f"{year_where(y)} AND {hour_pattern(h)}").items():
                per_hour[(day, h)] = n
        summed = {}
        for (day, _h), n in per_hour.items():
            summed[day] = summed.get(day, 0) + n
        check[y] = {"calls": sum(total.values()), "matched_by_hour": sum(summed.values())}
        rows += [(f"{day} {h:02d}", n) for (day, h), n in sorted(per_hour.items())]
        print(f"{y}: {check[y]}")
        if abs(check[y]["calls"] - check[y]["matched_by_hour"]) > 0.005 * check[y]["calls"]:
            raise SystemExit(f"{y}: calls by hour do not add up to the calls by date: {check[y]}")
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
                "hour_from": "DATE field and the hour of the TIME field",
                "calls_by_year_and_matched_by_hour": check,
                "missing_date_time": missing,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(rows)} hours written; rows without date_time by year: {missing}")


if __name__ == "__main__":
    main()
