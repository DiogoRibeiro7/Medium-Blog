"""Download the inputs for the uk-repeat-sales-rigidity analysis.

    python fetch.py --out data                 # UK monthly CPI only (small, committed)
    python fetch.py --out data --ppd DIR       # also HM Land Registry Price Paid Data into DIR

Sources
-------
- OECD Consumer price indices (OECD.SDD.TPS, DSD_PRICES@DF_PRICES_ALL): United Kingdom, monthly
  CPI, 2015 = 100, January 1994 onwards -> ``data/uk_cpi_monthly.csv``.
- HM Land Registry Price Paid Data, complete file (every residential sale in England and Wales
  lodged for registration since January 1995, about 5.3 GB). It is downloaded to a scratch
  directory, never committed: it carries addresses. ``build.py`` turns it into the aggregate
  tables committed under ``data/``.

Contains HM Land Registry data (c) Crown copyright and database right 2026. This data is
licensed under the Open Government Licence v3.0.

``data/fetch_log.txt`` records each URL, HTTP status, size and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "uk-repeat-sales-rigidity/1.0 (research companion)"}

CPI_URL = (
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/"
    "GBR.M.N.CPI.IX._T.N._Z?startPeriod=1994-01&format=csvfilewithlabels"
)
PPD_URLS = [
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv",
    "https://price-paid-data.publicdata.landregistry.gov.uk/pp-complete.csv",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_cpi(out: Path, log: list[str]) -> bool:
    req = urllib.request.Request(CPI_URL, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            status, body = r.status, r.read()
    except Exception as e:
        log.append(f"{CPI_URL}\terror {type(e).__name__}\t0\t-")
        return False
    log.append(f"{CPI_URL}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}")
    raw = pd.read_csv(io.BytesIO(body))
    cpi = (raw[["TIME_PERIOD", "OBS_VALUE"]].rename(columns={"TIME_PERIOD": "month", "OBS_VALUE": "cpi"})
           .assign(month=lambda d: d.month + "-01").sort_values("month"))
    cpi.to_csv(out / "uk_cpi_monthly.csv", index=False, float_format="%.10g")
    return True


def fetch_ppd(dest: Path, log: list[str]) -> Path | None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "pp-complete.csv"
    for url in PPD_URLS:
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=600) as r, target.open("wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
                status = r.status
        except Exception as e:
            log.append(f"{url}\terror {type(e).__name__}: {e}\t0\t-")
            continue
        size = target.stat().st_size
        log.append(f"{url}\t{status}\t{size}\t{_sha256_file(target)}")
        if status == 200 and size > 1_000_000_000:
            return target
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--ppd", default=None, help="scratch directory for the Price Paid Data file")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetched {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}",
           "source\tstatus\tbytes\tsha256"]
    ok = fetch_cpi(out, log)
    if args.ppd:
        ok = ok and fetch_ppd(Path(args.ppd), log) is not None
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
