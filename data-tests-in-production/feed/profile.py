"""Download one monthly yellow-taxi file and reduce it to a profile.

A profile is everything the checks in ``feed/checks.py`` need, computed once from the rows:
the schema as delivered, row counts, null counts, quantiles, category counts, trips per day
and per hour, pickups per zone, duplicates, and the rows whose total does not reconcile with
its components. Profiles are small JSON files, so the checks never touch the raw data.

The same function profiles a *faulted* copy of a month: ``feed/faults.py`` rewrites the rows
with one known defect, and the profile of the result is what a monitor would have seen.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import duckdb

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"

# columns as named in the TLC data dictionary (yellow, March 2025); files differ in case
NUMERIC = ("passenger_count", "trip_distance", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
           "improvement_surcharge", "total_amount", "congestion_surcharge", "airport_fee", "cbd_congestion_fee")
CATEGORIES = ("vendorid", "ratecodeid", "payment_type", "passenger_count", "store_and_fwd_flag")
COMPONENTS = ("fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
              "congestion_surcharge", "airport_fee", "cbd_congestion_fee")
QS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def download(month: str, dest: Path, tries: int = 5) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"yellow_tripdata_{month}.parquet"
    if path.exists():
        return path
    req = urllib.request.Request(URL.format(month=month), headers={"User-Agent": "data-tests-in-production/0.1"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return path
        except Exception:  # network: retry with backoff
            if path.exists():
                path.unlink()
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _cols(con, rel: str) -> dict[str, tuple[str, str]]:
    """lower-case name -> (name as delivered, DuckDB type)"""
    rows = con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
    return {r[0].lower(): (r[0], r[1]) for r in rows}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def profile(con: duckdb.DuckDBPyConnection, rel: str, month: str) -> dict:
    """Profile the rows of relation ``rel`` delivered as the file for ``month`` (YYYY-MM)."""
    cols = _cols(con, rel)
    c = {k: _q(v[0]) for k, v in cols.items()}
    first = f"DATE '{month}-01'"
    nxt = f"({first} + INTERVAL 1 MONTH)"
    pu, do = c.get("tpep_pickup_datetime"), c.get("tpep_dropoff_datetime")
    out: dict = {"month": month, "schema": [[v[0], v[1]] for v in cols.values()]}
    out["n_rows"] = con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]

    # numeric columns: nulls, negatives, zeros, range, mean, quantiles
    num = {}
    for k in NUMERIC:
        if k not in c:
            continue
        x = f"CAST({c[k]} AS DOUBLE)"
        qs = ", ".join(f"quantile_cont({x}, {q})" for q in QS)
        r = con.execute(f"SELECT count(*) - count({x}), sum(CASE WHEN {x} < 0 THEN 1 ELSE 0 END), "
                        f"sum(CASE WHEN {x} = 0 THEN 1 ELSE 0 END), min({x}), max({x}), avg({x}), {qs} "
                        f"FROM {rel}").fetchone()
        num[k] = dict(null=r[0], neg=r[1] or 0, zero=r[2] or 0, min=r[3], max=r[4], mean=r[5],
                      q={str(q): v for q, v in zip(QS, r[6:], strict=True)})
    out["numeric"] = num

    # categories (value -> count), nulls under the key "null"
    cats = {}
    for k in CATEGORIES:
        if k in c:
            e = c[k] if k == "store_and_fwd_flag" else f"CAST(round(CAST({c[k]} AS DOUBLE)) AS BIGINT)"
            rows = con.execute(f"SELECT CAST({e} AS VARCHAR) AS v, count(*) FROM {rel} GROUP BY 1").fetchall()
            cats[k] = {("null" if v is None else str(v)): n for v, n in rows}
    out["categories"] = cats

    # time: trips per pickup day near the month, outside the month, per hour, durations
    if pu and do:
        rows = con.execute(f"SELECT CAST(CAST({pu} AS DATE) AS VARCHAR), count(*) FROM {rel} "
                           f"WHERE {pu} >= {first} - INTERVAL 7 DAY AND {pu} < {nxt} + INTERVAL 7 DAY "
                           f"GROUP BY 1 ORDER BY 1").fetchall()
        out["by_day"] = {d: n for d, n in rows}
        r = con.execute(f"SELECT sum(CASE WHEN {pu} < {first} THEN 1 ELSE 0 END), "
                        f"sum(CASE WHEN {pu} >= {nxt} THEN 1 ELSE 0 END), "
                        f"sum(CASE WHEN {pu} IS NULL THEN 1 ELSE 0 END), "
                        f"sum(CASE WHEN {do} < {pu} THEN 1 ELSE 0 END), "
                        f"sum(CASE WHEN {do} - {pu} > INTERVAL 1 DAY THEN 1 ELSE 0 END), "
                        f"quantile_cont(epoch({do}) - epoch({pu}), 0.5), "
                        f"quantile_cont(epoch({do}) - epoch({pu}), 0.95) FROM {rel}").fetchone()
        out["time"] = dict(before_month=r[0] or 0, after_month=r[1] or 0, null_pickup=r[2] or 0,
                           dropoff_before_pickup=r[3] or 0, over_24h=r[4] or 0, dur_q50=r[5], dur_q95=r[6])
        rows = con.execute(f"SELECT hour({pu}), count(*) FROM {rel} WHERE {pu} >= {first} AND {pu} < {nxt} "
                           f"GROUP BY 1").fetchall()
        out["by_hour"] = {str(h): n for h, n in rows}

    # pickups per zone
    if "pulocationid" in c:
        e = f"CAST(round(CAST({c['pulocationid']} AS DOUBLE)) AS BIGINT)"
        rows = con.execute(f"SELECT CAST({e} AS VARCHAR), count(*) FROM {rel} GROUP BY 1").fetchall()
        out["pu_zone"] = {("null" if z is None else z): n for z, n in rows}

    # derived: fare per mile, tip share on card payments
    if {"fare_amount", "trip_distance"} <= c.keys():
        f, d = f"CAST({c['fare_amount']} AS DOUBLE)", f"CAST({c['trip_distance']} AS DOUBLE)"
        r = con.execute(f"SELECT quantile_cont({f} / {d}, 0.5), quantile_cont({f} / {d}, 0.95) FROM {rel} "
                        f"WHERE {d} >= 0.5 AND {f} > 0").fetchone()
        out["fare_per_mile"] = dict(q50=r[0], q95=r[1])
    if {"tip_amount", "fare_amount", "payment_type"} <= c.keys():
        r = con.execute(f"SELECT avg(CAST({c['tip_amount']} AS DOUBLE) / CAST({c['fare_amount']} AS DOUBLE)), "
                        f"avg(CASE WHEN CAST({c['tip_amount']} AS DOUBLE) > 0 THEN 1.0 ELSE 0.0 END) FROM {rel} "
                        f"WHERE CAST({c['payment_type']} AS VARCHAR) = '1' AND CAST({c['fare_amount']} AS DOUBLE) > 0"
                        ).fetchone()
        out["card_tips"] = dict(mean_share=r[0], share_tipped=r[1])

    # reconciliation: total against the sum of the components present in the file
    if "total_amount" in c:
        parts = " + ".join(f"coalesce(CAST({c[k]} AS DOUBLE), 0)" for k in COMPONENTS if k in c)
        r = con.execute(f"SELECT sum(CASE WHEN abs(CAST({c['total_amount']} AS DOUBLE) - ({parts})) > 0.011 "
                        f"THEN 1 ELSE 0 END) FROM {rel}").fetchone()
        out["recon_mismatch"] = r[0] or 0

    # exact duplicate rows
    distinct = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {rel})").fetchone()[0]
    out["duplicates"] = out["n_rows"] - distinct
    return out


def profile_file(path: Path, month: str) -> dict:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")  # timestamps in the files are naive local times; keep them as delivered
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path}')")
    p = profile(con, "t", month)
    p["source"] = dict(url=URL.format(month=month), bytes=path.stat().st_size, sha256=sha256(path))
    return p


def save(p: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, default=float, separators=(",", ":")), encoding="utf-8")
