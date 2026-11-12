"""Profile the real monthly files and their faulted copies.

    python -m feed.run --months 2023-01:2023-12 --faults-from 2023-01 --out profiles

For every month in the range: download the file, profile it as delivered, and, for months from
``--faults-from`` on, profile each of the fourteen faulted copies (``feed/faults.py``). Files
are deleted once profiled, except the previous month, which ``stale_copy`` needs.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import duckdb

from feed.faults import FAULTS, build
from feed.profile import download, profile, profile_file, save


def months(spec: str) -> list[str]:
    a, b = spec.split(":")
    y, m = map(int, a.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= b:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def previous(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y - 1:04d}-12" if m == 1 else f"{y:04d}-{m - 1:02d}"


def faulted(path: Path, prev: Path, month: str) -> dict[str, dict]:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path}')")
    con.execute(f"CREATE VIEW prev AS SELECT * FROM read_parquet('{prev}')")
    names = [r[0] for r in con.execute("DESCRIBE SELECT * FROM t").fetchall()]
    out = {}
    for name in FAULTS:
        con.execute(f"CREATE OR REPLACE VIEW f AS {build(name, names, month)}")
        out[name] = profile(con, "f", month)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True, help="YYYY-MM:YYYY-MM")
    ap.add_argument("--faults-from", default="9999-99")
    ap.add_argument("--out", default="profiles")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    cache = Path(a.cache) if a.cache else Path(tempfile.mkdtemp())
    keep: set[Path] = set()
    for month in months(a.months):
        path = download(month, cache)
        p = profile_file(path, month)
        save(p, out / "real" / f"{month}.json")
        print(f"{month}: {p['n_rows']:,} rows, {len(p['schema'])} columns", flush=True)
        if month >= a.faults_from:
            prev = download(previous(month), cache)
            for name, fp in faulted(path, prev, month).items():
                save(fp, out / "fault" / name / f"{month}.json")
            keep = {path}
        for f in cache.glob("*.parquet"):
            if f != path and f not in keep:
                f.unlink()


if __name__ == "__main__":
    main()
