"""Snapshot the sources the experiments depend on, without downloading the trip files.

    python -m feed.sources --out sources

- the Parquet footer of every monthly file of the four TLC feeds (yellow, green, FHV, high-volume FHV):
  row count and schema, read with HTTP range requests (a few kilobytes per file);
- the TLC monthly data report (average trips per day by licence class), the outside count;
- the text of the four data dictionaries.

Everything is written under ``--out`` with the retrieval date and a SHA-256, so a later revision of a source
shows up as a changed hash.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

UA = {"User-Agent": "data-tests-limits/0.1 (research; github.com/DiogoRibeiro7/Medium-Blog)"}
BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
PREFIX = {"yellow": "yellow", "green": "green", "fhv": "fhv", "fhvhv": "fhvhv"}
DICTIONARY = {
    "yellow": "https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf",
    "green": "https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf",
    "fhv": "https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_fhv.pdf",
    "fhvhv": "https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf",
}
REPORT = [
    "https://www.nyc.gov/assets/tlc/downloads/csv/data_reports_monthly.csv",
    "https://data.cityofnewyork.us/api/views/v6kb-cqej/rows.csv?accessType=DOWNLOAD",
]


def url(feed: str, month: str) -> str:
    return f"{BASE}/{PREFIX[feed]}_tripdata_{month}.parquet"


def months(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def get(u: str, headers: dict | None = None, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(u, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            time.sleep(2**i)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2**i)
    raise RuntimeError(f"failed: {u}")


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file that fetches only the byte ranges it is asked for."""

    def __init__(self, u: str):
        self.u = u
        req = urllib.request.Request(u, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = get(self.u, {"Range": f"bytes={self.pos}-{end}"})
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def footer(feed: str, month: str) -> dict | None:
    u = url(feed, month)
    try:
        f = RangeFile(u)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    md = pq.ParquetFile(f).metadata
    schema = md.schema.to_arrow_schema()
    return {
        "feed": feed,
        "month": month,
        "url": u,
        "bytes": f.size,
        "rows": md.num_rows,
        "row_groups": md.num_row_groups,
        "created_by": md.created_by,
        "schema": [[fld.name, str(fld.type)] for fld in schema],
    }


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sources")
    ap.add_argument("--first", default="2019-01")
    ap.add_argument("--last", default="2026-09")
    ap.add_argument(
        "--metadata-only",
        action="store_true",
        help="dictionaries and the report's columns only: what is needed to write the plan, no values",
    )
    a = ap.parse_args()
    out = Path(a.out)
    (out / "dictionaries").mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    log = {"retrieved": today, "files": {}}

    # dictionaries
    from pypdf import PdfReader

    for feed, u in DICTIONARY.items():
        try:
            b = get(u)
        except Exception as e:  # noqa: BLE001
            print(f"dictionary {feed} not available from this runner: {e}", flush=True)
            continue
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(b)).pages)
        (out / "dictionaries" / f"{feed}.txt").write_text(text, encoding="utf-8")
        log["files"][f"dictionaries/{feed}.txt"] = {"source": u, "pdf_sha256": sha(b), "pdf_bytes": len(b)}
        print(f"--- dictionary {feed} ({len(b):,} bytes)\n{text}\n", flush=True)

    # outside count
    for u in REPORT:
        try:
            b = get(u)
        except Exception as e:  # noqa: BLE001
            print(f"report not available at {u}: {e}", flush=True)
            continue
        (out / "data_reports_monthly.csv").write_bytes(b)
        log["files"]["data_reports_monthly.csv"] = {"source": u, "sha256": sha(b), "bytes": len(b)}
        rows = list(csv.reader(io.StringIO(b.decode("utf-8-sig"))))
        print(f"--- report {u}: {len(rows) - 1} rows; header {rows[0]}", flush=True)
        if a.metadata_only:
            col = {k: i for i, k in enumerate(rows[0])}
            for k in rows[0]:
                vals = sorted({r[col[k]] for r in rows[1:] if len(r) == len(rows[0])})
                if len(vals) <= 12:
                    print(f"    {k}: {vals}", flush=True)
                else:
                    print(
                        f"    {k}: {len(vals)} values, e.g. {vals[0]!r} .. {vals[-1]!r}"
                        if "month" in k.lower() or "date" in k.lower()
                        else f"    {k}: {len(vals)} values",
                        flush=True,
                    )
            (out / "data_reports_monthly.csv").unlink()
            log["files"].pop("data_reports_monthly.csv")
        else:
            for r in rows[1:13]:
                print("   ", r, flush=True)
        break

    if a.metadata_only:
        (out / "SOURCES.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        return

    # footers
    footers = []
    for feed in PREFIX:
        for m in months(a.first, a.last):
            r = footer(feed, m)
            if r is None:
                print(f"{feed} {m}: missing", flush=True)
                continue
            footers.append(r)
            print(f"{feed} {m}: {r['rows']:,} rows, {len(r['schema'])} columns", flush=True)
    with (out / "footers.jsonl").open("w", encoding="utf-8") as fh:
        for r in footers:
            fh.write(json.dumps(r) + "\n")
    log["files"]["footers.jsonl"] = {"source": BASE, "files": len(footers)}
    (out / "SOURCES.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    # schema history, printed for the log
    for feed in PREFIX:
        prev = None
        for r in (x for x in footers if x["feed"] == feed):
            s = [tuple(c) for c in r["schema"]]
            if s != prev:
                print(f"{feed} {r['month']}: {s}", flush=True)
            prev = s


if __name__ == "__main__":
    main()
