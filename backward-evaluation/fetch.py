"""Download the monthly US series used in the real-data part of the study.

    python fetch.py --out data

Series (FRED, latest vintage, no API key):

- INDPRO   Industrial Production: Total Index, seasonally adjusted (Federal Reserve, G.17)
- CPIAUCSL Consumer Price Index for All Urban Consumers: All Items, seasonally adjusted (BLS)
- UNRATE   Unemployment rate, seasonally adjusted (BLS)

Each series is read from the public ``fredgraph.csv`` endpoint. If that fails, the
same mnemonic is taken from the current FRED-MD file (McCracken and Ng), which carries
the three series under the same names.

Writes ``data/fred_monthly.csv`` (date plus one column per series) and appends one line
per request to ``data/fetch_log.txt`` (UTC time, URL, SHA-256 of the response, rows).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SERIES = ("INDPRO", "CPIAUCSL", "UNRATE")
FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_MD = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
USER_AGENT = "backward-evaluation/1.0 (research code)"


def get(url: str, tries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}: {last}")


def log(out: Path, url: str, body: bytes | None, rows: int | str) -> None:
    sha = hashlib.sha256(body).hexdigest() if body is not None else "-"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{stamp}\t{url}\t{sha}\t{rows}\n")


def from_fredgraph(sid: str, out: Path) -> pd.Series:
    url = FREDGRAPH.format(sid=sid)
    body = get(url)
    df = pd.read_csv(io.BytesIO(body))
    date_col = df.columns[0]
    s = pd.to_numeric(df[sid], errors="coerce")
    s.index = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    s = s.dropna()
    log(out, url, body, len(s))
    return s.rename(sid)


def from_fred_md(out: Path) -> pd.DataFrame:
    body = get(FRED_MD)
    df = pd.read_csv(io.BytesIO(body))
    df = df[df["sasdate"].astype(str).str.match(r"\d+/\d+/\d{4}")]  # drop the transform-code row
    idx = pd.to_datetime(df["sasdate"], format="%m/%d/%Y").dt.to_period("M").dt.to_timestamp()
    res = df[[c for c in SERIES if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    res.index = idx
    log(out, FRED_MD, body, len(res))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols: dict[str, pd.Series] = {}
    missing: list[str] = []
    for sid in SERIES:
        try:
            cols[sid] = from_fredgraph(sid, out)
        except Exception as e:  # noqa: BLE001 - logged and retried from FRED-MD
            log(out, FREDGRAPH.format(sid=sid), None, f"ERROR {e}")
            missing.append(sid)
    if missing:
        md = from_fred_md(out)
        for sid in missing:
            if sid in md.columns:
                cols[sid] = md[sid].dropna().rename(sid)
    if set(cols) != set(SERIES):
        print(f"missing series: {sorted(set(SERIES) - set(cols))}", file=sys.stderr)
        return 1

    panel = pd.concat(cols.values(), axis=1).sort_index()
    panel.index.name = "date"
    panel.to_csv(out / "fred_monthly.csv", float_format="%.6g")
    print(f"wrote {out / 'fred_monthly.csv'}: {len(panel)} months, "
          f"{panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
