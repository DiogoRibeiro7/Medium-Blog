"""Download the eight test series named in PREREGISTRATION.md, plus the two in-sample series.

    python fetch.py --out data

Series (latest vintage, no API key): UNRATE, PAYEMS, HOUST, W875RX1, DPCERA3M086SBEA,
WPSFD49207, GS10, M2SL (test) and INDPRO, CPIAUCSL (in-sample).

Sources are tried in order for each series, and the one that answered is logged:

1. FRED ``fredgraph.csv`` (Federal Reserve Bank of St. Louis);
2. the current FRED-MD file (McCracken and Ng), which carries the same mnemonics;
3. for the two in-sample series only, DBnomics mirrors of the primary publishers.

Writes ``data/fred_monthly.csv`` (date plus one column per series) and appends one line
per request to ``data/fetch_log.txt`` (UTC time, URL, SHA-256 of the response, rows or
error).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SERIES = ("UNRATE", "PAYEMS", "HOUST", "W875RX1", "DPCERA3M086SBEA", "WPSFD49207", "GS10", "M2SL",
          "INDPRO", "CPIAUCSL")
FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_MD = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
DBNOMICS = {
    "INDPRO": "FED/G17_IP_MAJOR_INDUSTRY_GROUPS/IP.B50001.S",
    "CPIAUCSL": "BLS/cu/CUSR0000SA0",
}
DBNOMICS_URL = "https://api.db.nomics.world/v22/series/{code}?observations=1&format=json"
HEADERS = {"User-Agent": "backward-evaluation-pooling/1.0 (research code)", "Accept": "text/csv,application/json,*/*"}


def get(url: str, tries: int = 2, timeout: int = 45) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} tries: {last}")


def log(out: Path, url: str, body: bytes | None, note: object) -> None:
    sha = hashlib.sha256(body).hexdigest() if body is not None else "-"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{stamp}\t{url}\t{sha}\t{note}\n")


def monthly_index(dates: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp())


def from_fredgraph(sid: str, out: Path) -> pd.Series:
    url = FREDGRAPH.format(sid=sid)
    body = get(url)
    df = pd.read_csv(io.BytesIO(body))
    s = pd.Series(pd.to_numeric(df[sid], errors="coerce").to_numpy(), index=monthly_index(df.iloc[:, 0]))
    s = s.dropna()
    log(out, url, body, f"{len(s)} rows")
    return s.rename(sid)


def from_fred_md(out: Path) -> pd.DataFrame:
    body = get(FRED_MD)
    df = pd.read_csv(io.BytesIO(body))
    df = df[df["sasdate"].astype(str).str.match(r"\d+/\d+/\d{4}")]  # drop the transform-code row
    idx = monthly_index(pd.to_datetime(df["sasdate"], format="%m/%d/%Y"))
    res = df[[c for c in SERIES if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    res.index = idx
    log(out, FRED_MD, body, f"{len(res)} rows")
    return res


def from_dbnomics(sid: str, out: Path) -> pd.Series:
    url = DBNOMICS_URL.format(code=DBNOMICS[sid])
    body = get(url)
    doc = json.loads(body)["series"]["docs"][0]
    s = pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").to_numpy(),
                  index=monthly_index(pd.Series(doc["period"])))
    s = s.dropna()
    log(out, url, body, f"{len(s)} rows")
    return s.rename(sid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols: dict[str, pd.Series] = {}
    source: dict[str, str] = {}
    for sid in SERIES:
        try:
            cols[sid] = from_fredgraph(sid, out)
            source[sid] = "FRED"
        except Exception as e:  # noqa: BLE001 - logged, next source tried
            log(out, FREDGRAPH.format(sid=sid), None, f"ERROR {e}")
    missing = [s for s in SERIES if s not in cols]
    if missing:
        try:
            md = from_fred_md(out)
            for sid in missing:
                if sid in md.columns and md[sid].notna().sum() > 100:
                    cols[sid] = md[sid].dropna().rename(sid)
                    source[sid] = "FRED-MD"
        except Exception as e:  # noqa: BLE001
            log(out, FRED_MD, None, f"ERROR {e}")
    for sid in [s for s in SERIES if s not in cols and s in DBNOMICS]:
        try:
            cols[sid] = from_dbnomics(sid, out)
            source[sid] = "DBnomics"
        except Exception as e:  # noqa: BLE001
            log(out, DBNOMICS_URL.format(code=DBNOMICS[sid]), None, f"ERROR {e}")

    if set(cols) != set(SERIES):
        print(f"missing series: {sorted(set(SERIES) - set(cols))}", file=sys.stderr)
        return 1

    panel = pd.concat(cols.values(), axis=1).sort_index()
    panel.index.name = "date"
    panel.to_csv(out / "fred_monthly.csv", float_format="%.6g")
    (out / "sources.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'fred_monthly.csv'}: {len(panel)} months, "
          f"{panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}; sources {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
