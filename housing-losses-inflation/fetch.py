"""Download the public series for the housing-losses-inflation analysis.

    python fetch.py --out data
    python fetch.py --out data --jst-file path/to/JSTdatasetR6.csv   # use a local copy of JST R6

Sources
-------
- Jorda-Schularick-Taylor Macrohistory Database, release 6 (1870-2020), macrohistory.net.
  Only the housing, price-level and rate columns used here are kept (``data/jst_housing.csv``).
  The JST data are licensed CC BY-NC-SA 4.0; the extract keeps that licence.
- OECD Analytical house price indicators (OECD.ECO.MPD, DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES),
  quarterly, seasonally adjusted, 2015 = 100: nominal house prices (HPI), real house prices
  (RHP, deflated by the private consumption deflator), rent prices (RPI) and the price-to-rent
  ratio (HPI_RPI), 1970-Q1 onwards (``data/oecd_house_prices.csv``).

``data/fetch_log.txt`` records URL (or local path), HTTP status, size and SHA-256 of every
response. The JST download is best effort: macrohistory.net sometimes refuses scripted
requests, and when it does the committed extract is kept and the failure is logged. The OECD
request is required.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "housing-losses-inflation/1.0 (research companion)"}

JST_URLS = [
    "https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.xlsx",
    "https://www.macrohistory.net/app/download/9834512049/JSTdatasetR6.xlsx",
    "https://data.macrohistory.net/JSTdatasetR6.xlsx",
]
JST_COLUMNS = [
    "year", "country", "iso", "cpi", "hpnom", "housing_capgain", "housing_capgain_ipolated",
    "housing_rent_rtn", "housing_rent_yd", "housing_tr", "rent_ipolated", "stir", "ltrate",
]

OECD_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/.Q.HPI+RHP+RPI+HPI_RPI.IX"
    "?startPeriod=1970-Q1&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)


def _get(url: str, timeout: int = 180) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            if r.headers.get("Content-Encoding", "").lower() == "gzip" or body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # network refused, timeout, TLS
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        return 0, b""


def _log(lines: list[str], source: str, status: int | str, body: bytes) -> None:
    sha = hashlib.sha256(body).hexdigest() if body else "-"
    lines.append(f"{source}\t{status}\t{len(body)}\t{sha}")


def _tidy_jst(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in JST_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"JST file lacks columns {missing}")
    out = raw[JST_COLUMNS].copy()
    out["year"] = out["year"].astype(int)
    num = [c for c in JST_COLUMNS if c not in ("year", "country", "iso")]
    out[num] = out[num].astype(float)
    return out.sort_values(["iso", "year"]).reset_index(drop=True)


def fetch_jst(out: Path, log: list[str], local: Path | None) -> bool:
    if local is not None:
        body = local.read_bytes()
        _log(log, f"local:{local.name}", "file", body)
        raw = pd.read_csv(io.BytesIO(body)) if local.suffix == ".csv" else pd.read_excel(io.BytesIO(body))
        _tidy_jst(raw).to_csv(out / "jst_housing.csv", index=False, float_format="%.12g")
        return True
    for url in JST_URLS:
        status, body = _get(url)
        _log(log, url, status, body)
        # the download link is named .xlsx but has served a Stata file; accept either
        kind = "xlsx" if body[:2] == b"PK" else "dta" if body[:11] == b"<stata_dta>" else None
        if status == 200 and kind:
            try:
                raw = (pd.read_excel(io.BytesIO(body)) if kind == "xlsx"
                       else pd.read_stata(io.BytesIO(body)))
                tidy = _tidy_jst(raw)
            except Exception as e:  # unexpected layout: keep the committed extract
                log.append(f"# {url}: could not parse ({type(e).__name__}: {e})")
                continue
            tidy.to_csv(out / "jst_housing.csv", index=False, float_format="%.12g")
            return True
        if status == 200:
            log.append(f"# {url}: neither xlsx nor Stata, first bytes {body[:16]!r}")
    return False


def fetch_oecd(out: Path, log: list[str]) -> bool:
    status, body = _get(OECD_URL, timeout=300)
    _log(log, OECD_URL, status, body)
    if status != 200 or not body:
        return False
    raw = pd.read_csv(io.BytesIO(body))
    tidy = raw[["REF_AREA", "Reference area", "MEASURE", "TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"REF_AREA": "iso", "Reference area": "area", "MEASURE": "measure",
                 "TIME_PERIOD": "quarter", "OBS_VALUE": "value"}
    )
    tidy["value"] = tidy["value"].astype(float)
    tidy = tidy.sort_values(["iso", "measure", "quarter"]).reset_index(drop=True)
    tidy.to_csv(out / "oecd_house_prices.csv", index=False, float_format="%.12g")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--jst-file", default=None, help="local copy of JSTdatasetR6 (.csv or .xlsx)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log = [f"# fetched {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}",
           "source\tstatus\tbytes\tsha256"]
    ok_oecd = fetch_oecd(out, log)
    ok_jst = fetch_jst(out, log, Path(args.jst_file) if args.jst_file else None)
    if not ok_jst:
        log.append("# JST: no official download succeeded; the committed data/jst_housing.csv is kept")
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok_oecd else 1


if __name__ == "__main__":
    sys.exit(main())
