"""Reduce one monthly file of a feed to the counts its contract's rules need.

    python -m feed.profile --feed fhvhv --month 2024-01

One DuckDB pass over the file counts, per field of the contract, the rows that are empty, negative, outside the
listed codes or outside the zone lookup; per ordered pair of timestamps the rows out of order; the rows picked
up outside the month; for yellow the rows whose total is not the sum of its charges; and, in a second pass, the
exact duplicate rows (count minus distinct 64-bit row hashes). Fields are matched by lower-case name and cast,
so a renamed case or a changed type is recorded in ``schema`` without stopping the profile. For every code
rule the five most common offending values are kept, so an alarm can be read without the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb

from feed.contracts import CONTRACTS, FEEDS, ZONES, Contract
from feed.sources import UA, url


def download(feed: str, month: str, dest: Path, tries: int = 5) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{feed}_{month}.parquet"
    if path.exists():
        return path
    for i in range(tries):
        try:
            req = urllib.request.Request(url(feed, month), headers=UA)
            part = path.with_suffix(".part")
            with urllib.request.urlopen(req, timeout=600) as r, open(part, "wb") as fh:
                expected = int(r.headers.get("Content-Length") or -1)
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            got = part.stat().st_size
            if expected >= 0 and got != expected:  # the connection closed early: try again
                print(f"{feed} {month}: got {got:,} of {expected:,} bytes, retrying", flush=True)
                part.unlink()
                time.sleep(2**i)
                continue
            part.rename(path)
            return path
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            time.sleep(2**i)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2**i)
    raise RuntimeError(f"download failed: {feed} {month}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def lit(v) -> str:
    return f"'{v}'" if isinstance(v, str) else repr(float(v))


def next_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y + (m == 12):04d}-{m % 12 + 1:02d}"


def profile(con: duckdb.DuckDBPyConnection, rel: str, month: str, c: Contract) -> dict:
    described = con.execute(f"DESCRIBE {rel}").fetchall()
    schema = [[r[0], r[1]] for r in described]
    present = {r[0].lower(): r[0] for r in described}
    col = {k: q(present[k]) for k in present}

    def num(k: str) -> str:
        return f"TRY_CAST({col[k]} AS DOUBLE)"

    def ts(k: str) -> str:
        return f"TRY_CAST({col[k]} AS TIMESTAMP)"

    exprs: dict[str, str] = {"n_rows": "count(*)"}
    for f in c.fields:
        k = f.name.lower()
        if k not in col:
            continue
        exprs[f"null|{f.name}"] = f"count(*) - count({col[k]})"
        if f.nonneg:
            exprs[f"neg|{f.name}"] = f"count(*) FILTER (WHERE {num(k)} < 0)"
        if f.codes:
            vals = ", ".join(lit(v) for v in f.codes)
            x = f"TRIM(CAST({col[k]} AS VARCHAR))" if isinstance(f.codes[0], str) else num(k)
            exprs[f"code|{f.name}"] = f"count(*) FILTER (WHERE {col[k]} IS NOT NULL AND {x} NOT IN ({vals}))"
        if f.zone:
            exprs[f"zone|{f.name}"] = (
                f"count(*) FILTER (WHERE {col[k]} IS NOT NULL AND NOT ({num(k)} BETWEEN {ZONES[0]} AND {ZONES[1]}))"
            )
    for a, b in c.order:
        if a.lower() in col and b.lower() in col:
            exprs[f"order|{a}<={b}"] = f"count(*) FILTER (WHERE {ts(a.lower())} > {ts(b.lower())})"
    pu = ts(c.pickup.lower()) if c.pickup.lower() in col else None
    first, nxt = f"TIMESTAMP '{month}-01'", f"TIMESTAMP '{next_month(month)}-01'"
    if pu:
        exprs["window|outside"] = f"count(*) FILTER (WHERE {pu} < {first} OR {pu} >= {nxt})"
    if c.total and c.total[0].lower() in col:
        parts = " + ".join(f"coalesce({num(k.lower())}, 0)" for k in c.total[1] if k.lower() in col)
        exprs["total|mismatch"] = f"count(*) FILTER (WHERE abs({num(c.total[0].lower())} - ({parts})) > 0.011)"
    keys = list(exprs)
    row = con.execute(f"SELECT {', '.join(exprs[k] for k in keys)} FROM {rel}").fetchone()
    counts = {k: int(v or 0) for k, v in zip(keys, row, strict=True)}

    allcols = ", ".join(col[k] for k in col)
    counts["unique|duplicates"] = int(
        con.execute(f"SELECT count(*) - count(DISTINCT hash({allcols})) FROM {rel}").fetchone()[0]
    )

    offending = {}
    for f in c.fields:
        k = f.name.lower()
        if (
            k not in col
            or not (f.codes or f.zone)
            or not (counts.get(f"code|{f.name}") or counts.get(f"zone|{f.name}"))
        ):
            continue
        cond = []
        if f.codes:
            vals = ", ".join(lit(v) for v in f.codes)
            x = f"TRIM(CAST({col[k]} AS VARCHAR))" if isinstance(f.codes[0], str) else num(k)
            cond.append(f"{x} NOT IN ({vals})")
        if f.zone:
            cond.append(f"NOT ({num(k)} BETWEEN {ZONES[0]} AND {ZONES[1]})")
        top = con.execute(
            f"SELECT CAST({col[k]} AS VARCHAR), count(*) FROM {rel} WHERE {col[k]} IS NOT NULL AND "
            f"({' OR '.join(cond)}) GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 5"
        ).fetchall()
        offending[f.name] = [[v, int(n)] for v, n in top]

    return {
        "feed": c.feed,
        "month": month,
        "schema": schema,
        "n_rows": counts.pop("n_rows"),
        "counts": counts,
        "offending": offending,
    }


def profile_file(path: Path, month: str, feed: str) -> dict:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path}')")
    if feed in CONTRACTS:
        p = profile(con, "t", month, CONTRACTS[feed])
    else:  # no contract: the row count and the schema only
        schema = [[r[0], r[1]] for r in con.execute("DESCRIBE t").fetchall()]
        n = int(con.execute("SELECT count(*) FROM t").fetchone()[0])
        p = {"feed": feed, "month": month, "schema": schema, "n_rows": n, "counts": {}, "offending": {}}
    p["source"] = {"url": url(feed, month), "bytes": path.stat().st_size, "sha256": sha256(path)}
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True, choices=FEEDS)
    ap.add_argument("--month", required=True)
    ap.add_argument("--cache", default="/tmp/tlc")
    a = ap.parse_args()
    path = download(a.feed, a.month, Path(a.cache))
    print(json.dumps(profile_file(path, a.month, a.feed), indent=1))


if __name__ == "__main__":
    main()
