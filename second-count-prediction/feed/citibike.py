"""Count the Citi Bike trip files by month of trip start.

    python -m feed.citibike --year 2024 --bucket sources/citibike_bucket.json --out counts --cache /tmp/cb

For every New York key of the year in the public bucket (yearly archives and monthly files; Jersey City keys
start with "JC-" and are left out), the file is downloaded, every CSV in it, and in zips inside it, is read
with DuckDB, and the rows are counted by the month of their start time. The counts are kept per key, so that
evaluate.py can apply the rule of PLAN.md when a month is covered by more than one key.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import duckdb

from feed.http import download

BUCKET = "https://s3.amazonaws.com/tripdata"
KEY = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})?[-_][^/]*citibike[^/]*tripdata[^/]*\.zip$", re.I)
START_COLUMNS = ("started_at", "starttime", "start time", "start_time")

MONTH_SQL = r"""
CASE
  WHEN regexp_matches(c, '^\s*"?\d{4}-\d{2}-\d{2}') THEN substr(trim(c, ' "'), 1, 7)
  WHEN regexp_matches(c, '^\s*"?\d{1,2}/\d{1,2}/\d{4}')
    THEN regexp_extract(c, '(\d{4})', 1) || '-' || lpad(regexp_extract(c, '^\s*"?(\d{1,2})/', 1), 2, '0')
END
"""


def year_keys(bucket: list[dict], year: int) -> list[dict]:
    out = []
    for k in bucket:
        m = KEY.match(k["key"])
        if m and int(m["y"]) == year and not k["key"].upper().startswith("JC"):
            out.append({**k, "monthly": m["m"] is not None, "file_month": f"{m['y']}-{m['m']}" if m["m"] else None})
    return out


def start_column(con: duckdb.DuckDBPyConnection, path: Path) -> str | None:
    cols = [
        r[0]
        for r in con.execute(
            "DESCRIBE SELECT * FROM read_csv(?, all_varchar=true, header=true)", [str(path)]
        ).fetchall()
    ]
    for c in cols:
        if c.strip().strip('"').lower() in START_COLUMNS:
            return c
    return None


def count_csv(con: duckdb.DuckDBPyConnection, path: Path) -> dict:
    """Rows by month of trip start; rows whose start time has no recognisable date are counted apart."""
    with open(path, "rb") as fh:
        lines = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 24), b""))
    col = start_column(con, path)
    if col is None:
        return {"months": {}, "rows": 0, "unparsed": 0, "lines": lines, "error": "no start-time column"}
    q = f"""
        SELECT m, count(*) FROM (
          SELECT {MONTH_SQL} AS m
          FROM (SELECT "{col}" AS c FROM read_csv(?, all_varchar=true, header=true, ignore_errors=true,
                                                  null_padding=true))
        ) GROUP BY m
    """
    months, unparsed = {}, 0
    for m, n in con.execute(q, [str(path)]).fetchall():
        if m is None:
            unparsed += n
        else:
            months[m] = n
    return {"months": months, "rows": sum(months.values()) + unparsed, "unparsed": unparsed, "lines": lines}


def walk_zip(con: duckdb.DuckDBPyConnection, zpath: Path, work: Path, seen: set[str], out: dict) -> None:
    """Count every CSV in a zip, descending into zips inside it; a CSV name seen twice is counted once."""
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            name = info.filename
            base = name.rsplit("/", 1)[-1]
            if info.is_dir() or "__MACOSX" in name or base.startswith("._") or not base:
                continue
            if base.lower().endswith(".zip"):
                inner = Path(zf.extract(info, work))
                walk_zip(con, inner, work, seen, out)
                inner.unlink()
            elif base.lower().endswith(".csv"):
                if base in seen:
                    out["duplicate_names"].append(name)
                    continue
                seen.add(base)
                p = Path(zf.extract(info, work))
                r = count_csv(con, p)
                p.unlink()
                out["inner"][name] = {k: v for k, v in r.items() if k != "months"}
                for m, n in r["months"].items():
                    out["months"][m] = out["months"].get(m, 0) + n
                out["rows"] += r["rows"]
                out["unparsed"] += r["unparsed"]


def count_key(con: duckdb.DuckDBPyConnection, key: dict, cache: Path) -> dict:
    zpath = download(f"{BUCKET}/{key['key']}", cache / key["key"])
    out = {
        "size": key["size"],
        "monthly": key["monthly"],
        "file_month": key["file_month"],
        "months": {},
        "rows": 0,
        "unparsed": 0,
        "inner": {},
        "duplicate_names": [],
    }
    work = Path(tempfile.mkdtemp(dir=cache))
    try:
        walk_zip(con, zpath, work, set(), out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        zpath.unlink(missing_ok=True)
    out["months"] = dict(sorted(out["months"].items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--bucket", type=Path, default=Path("sources/citibike_bucket.json"))
    ap.add_argument("--out", type=Path, default=Path("counts"))
    ap.add_argument("--cache", type=Path, default=Path("/tmp/citibike"))
    a = ap.parse_args()
    bucket = json.loads(a.bucket.read_text(encoding="utf-8"))
    dest = a.out / "citibike" / f"{a.year}.json"
    done = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {"year": a.year, "keys": {}}
    a.cache.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    failures = []
    for k in year_keys(bucket, a.year):
        prev = done["keys"].get(k["key"])
        if prev and prev.get("size") == k["size"]:
            continue
        try:
            done["keys"][k["key"]] = count_key(con, k, a.cache)
        except RuntimeError as e:
            failures.append(f"{k['key']}: {e}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(done, indent=1), encoding="utf-8")
    print(f"{a.year}: {len(done['keys'])} keys counted, {len(failures)} failed")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
