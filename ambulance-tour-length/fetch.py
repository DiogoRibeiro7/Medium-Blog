"""Download FDNY EMS incidents assigned in the test year and write hourly busy time (PLAN.md, "Data").

python fetch.py --sources sources

Writes sources/schema.json (column list and update time), sources/ems_hourly.csv (busy unit-minutes and incidents
assigned per local clock hour and borough, over the test weeks), sources/FILES (SHA-256) and sources/REVISION
(download counts). If the hourly file is already committed, it is verified and kept. Only hourly aggregates are
written; no incident record is stored.
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

import numpy as np
import pandas as pd

DATASET = "76xm-jjuj"
BASE = "https://data.cityofnewyork.us"
FIELDS = ("incident_id", "incident_datetime", "first_assignment_datetime", "incident_close_datetime", "borough")
FROM = "2018-12-30T00:00:00"
TO = "2019-12-30T00:00:00"
GRID_START = pd.Timestamp("2018-12-31 00:00")
GRID_HOURS = 52 * 168
ORIGIN = pd.Timestamp("2018-12-30 00:00")
MAX_HOURS = 8.0
PAGE = 250_000


def get(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
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


def resource(params: dict) -> str:
    return f"{BASE}/resource/{DATASET}.csv?" + urllib.parse.urlencode(params)


def where() -> str:
    return f"first_assignment_datetime >= '{FROM}' AND first_assignment_datetime < '{TO}'"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def busy_by_hour(start: np.ndarray, end: np.ndarray, n_hours: int) -> np.ndarray:
    """Exact busy seconds in each hour of a grid, from intervals [start, end) given in seconds from the grid start.

    Intervals are clipped to the grid. The first and last hour of an interval get their partial overlap; the hours in
    between get 3,600 seconds each, through a difference array.
    """
    lo, hi = 0.0, n_hours * 3600.0
    s = np.clip(start, lo, hi)
    e = np.clip(end, lo, hi)
    keep = e > s
    s, e = s[keep], e[keep]
    out = np.zeros(n_hours + 1)
    hs = np.floor(s / 3600).astype(np.int64)
    he = np.floor((e - 1e-9) / 3600).astype(np.int64)
    same = hs == he
    np.add.at(out, hs[same], e[same] - s[same])
    d = ~same
    np.add.at(out, hs[d], (hs[d] + 1) * 3600 - s[d])
    np.add.at(out, he[d], e[d] - he[d] * 3600)
    diff = np.zeros(n_hours + 2)
    np.add.at(diff, hs[d] + 1, 3600.0)
    np.add.at(diff, he[d], -3600.0)
    out += np.cumsum(diff)[: n_hours + 1]
    return out[:n_hours]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    a = ap.parse_args()
    a.sources.mkdir(parents=True, exist_ok=True)
    out = a.sources / "ems_hourly.csv"
    files = a.sources / "FILES"
    if out.exists() and files.exists():
        want = dict(line.split() for line in files.read_text().splitlines() if line.strip())
        for name, digest in want.items():
            if sha256(a.sources / name) != digest:
                raise SystemExit(f"{name}: SHA-256 differs from sources/FILES")
        print("committed sources verified; nothing downloaded")
        return

    meta = json.loads(get(f"{BASE}/api/views/{DATASET}.json"))
    cols = [c.get("fieldName") for c in meta.get("columns", [])]
    (a.sources / "schema.json").write_text(
        json.dumps({"columns": cols, "rowsUpdatedAt": meta.get("rowsUpdatedAt")}, indent=1) + "\n"
    )
    missing = [f for f in FIELDS if f not in cols]
    if missing:
        print("columns of the dataset:", cols)
        raise SystemExit(f"fields missing from the dataset: {missing}")

    counted = get(resource({"$select": "count(*) AS n", "$where": where()})).decode()
    total = int(next(csv.DictReader(io.StringIO(counted)))["n"])
    print(f"incidents assigned in the window: {total}")
    frames = []
    offset = 0
    while True:
        q = resource(
            {
                "$select": ",".join(FIELDS),
                "$where": where(),
                "$order": "incident_id",
                "$limit": str(PAGE),
                "$offset": str(offset),
            }
        )
        df = pd.read_csv(io.BytesIO(get(q)), dtype=str)
        frames.append(df)
        print(f"  page at {offset}: {len(df)} rows")
        if len(df) < PAGE:
            break
        offset += PAGE
    df = pd.concat(frames, ignore_index=True)
    rows = len(df)
    dup = int(df["incident_id"].duplicated().sum())
    df = df.drop_duplicates("incident_id")
    assign = pd.to_datetime(df["first_assignment_datetime"], errors="coerce")
    close = pd.to_datetime(df["incident_close_datetime"], errors="coerce")
    dur_h = (close - assign).dt.total_seconds() / 3600
    no_time = assign.isna() | close.isna()
    nonpos = ~no_time & (dur_h <= 0)
    long = ~no_time & (dur_h > MAX_HOURS)
    keep = ~(no_time | nonpos | long)
    df, assign, close = df[keep], assign[keep], close[keep]
    borough = df["borough"].fillna("UNKNOWN").str.strip().str.upper().to_numpy()

    start = ((assign - GRID_START).dt.total_seconds()).to_numpy()
    end = ((close - GRID_START).dt.total_seconds()).to_numpy()
    ahour = np.floor(start / 3600).astype(np.int64)
    g = pd.date_range(GRID_START, periods=GRID_HOURS, freq="h")
    records = []
    for b in sorted(set(borough)):
        m = borough == b
        busy = busy_by_hour(start[m], end[m], GRID_HOURS) / 60.0
        inside = (ahour[m] >= 0) & (ahour[m] < GRID_HOURS)
        assigned = np.bincount(ahour[m][inside], minlength=GRID_HOURS)
        for i in range(GRID_HOURS):
            if busy[i] > 0 or assigned[i] > 0:
                records.append((g[i].strftime("%Y-%m-%d %H"), b, f"{busy[i]:.4f}", int(assigned[i])))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["hour", "borough", "busy_minutes", "assigned"])
        w.writerows(records)
    other = a.sources / "nyc311_hourly_calls.csv"
    files.write_text(f"ems_hourly.csv {sha256(out)}\nnyc311_hourly_calls.csv {sha256(other)}\n")
    rev = {
        "rowsUpdatedAt": meta.get("rowsUpdatedAt"),
        "window_first_assignment": [FROM, TO],
        "count_query": total,
        "rows_downloaded": rows,
        "duplicate_ids_dropped": dup,
        "excluded_missing_time": int(no_time.sum()),
        "excluded_close_not_after_assignment": int(nonpos.sum()),
        "excluded_longer_than_8h": int(long.sum()),
        "kept": int(keep.sum()),
        "boroughs": {b: int((borough == b).sum()) for b in sorted(set(borough))},
        "median_busy_minutes": float(np.median(dur_h[keep]) * 60),
    }
    (a.sources / "REVISION").write_text(json.dumps(rev, indent=1) + "\n")
    if rows != total:
        raise SystemExit(f"downloaded {rows} rows but the count query reported {total}")
    print(json.dumps(rev, indent=1))


if __name__ == "__main__":
    main()
