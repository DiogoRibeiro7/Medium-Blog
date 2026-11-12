"""Download the OECD series used to decompose the EU-US productivity gap.

    python fetch.py --out data

Three requests to the SDMX API of the OECD Data Explorer:

* ``DSD_PDB@DF_PDB`` (OECD Productivity Database), total economy: GDP and gross value added
  in national currency and in US dollars at PPP (current prices and chain-linked volumes,
  reference year 2020), hours worked, employment, population, labour compensation
  (employees plus an imputation for the self-employed), and the growth-accounting
  contributions of capital deepening (ICT and non-ICT) and multifactor productivity.
* ``DSD_NAMAIN10@DF_TABLE2`` (national accounts, Table 2): gross national income (B5G),
  net national income (B5N), consumption of fixed capital (P51C) and GDP (B1GQ) in
  national currency at current prices, used for the GNI/GDP and NNI/GDP ratios.
* ``DSD_NAMAIN10@DF_TABLE1_INCOME`` (national accounts, Table 1, income approach):
  compensation of employees (D1), gross operating surplus and mixed income (B2A3G),
  taxes less subsidies on production and imports (D2X3) and GDP, national currency,
  current prices. Used to split GDP per hour into labour income, net taxes and the rest
  on the same basis for every country.

Raw responses are kept gzipped in ``data/raw/``, tidy tables in ``data/*.csv``, and
``data/fetch_log.txt`` records the URL, HTTP status, size and SHA-256 of every response.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OECD = "https://sdmx.oecd.org/public/rest/data"
UA = {"User-Agent": "eu-us-productivity-gap/1.0 (research companion)"}

PDB_MEASURES = (
    "GDP", "GDPHRS", "GDPPOP", "GDPEMP", "HRSAV", "HRSPOP", "HRSTO", "EMP", "POP",
    "GVA", "GVAHRS", "LCTOT", "MFPH", "KSERHRS", "MFP_PCCONLP", "KSERHRS_PCCONLP",
)

REQUESTS = {
    # name -> (agency, dataflow, key, extra query)
    "oecd_pdb": (
        "OECD.SDD.TPS",
        "DSD_PDB@DF_PDB",
        ".A." + "+".join(PDB_MEASURES) + "._T......",
        "startPeriod=1990",
    ),
    "oecd_gni": (
        "OECD.SDD.NAD",
        "DSD_NAMAIN10@DF_TABLE2",
        "A..S1.S1.B5G+B5N+P51C+B1GQ....XDC.V..",
        "startPeriod=1990",
    ),
    "oecd_income": (
        "OECD.SDD.NAD",
        "DSD_NAMAIN10@DF_TABLE1_INCOME",
        "A..S1.S1.D1+D2X3+B2A3G+B1GQ.._T+_Z..XDC.V..",
        "startPeriod=1995",
    ),
}

KEEP = [
    "REF_AREA", "Reference area", "MEASURE", "TRANSACTION",
    "ACTIVITY", "UNIT_MEASURE", "PRICE_BASE", "TRANSFORMATION", "ASSET_CODE", "CONVERSION_TYPE",
    "TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "UNIT_MULT", "BASE_PER",
]


def get(url: str, tries: int = 4) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                last = e
            else:
                return e.code, e.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    out: Path = args.out
    (out / "raw").mkdir(parents=True, exist_ok=True)

    log = [f"# fetched {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
           "# name\tstatus\tbytes\tsha256\turl"]
    failures = 0
    for name, (agency, dataflow, key, query) in REQUESTS.items():
        url = (f"{OECD}/{agency},{dataflow},/{key}?{query}"
               "&dimensionAtObservation=AllDimensions&format=csvfilewithlabels")
        try:
            status, body = get(url)
        except RuntimeError as e:
            log.append(f"{name}\tERROR\t{e}")
            failures += 1
            continue
        log.append(f"{name}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
        if status != 200 or len(body) < 500:
            failures += 1
            continue
        (out / "raw" / f"{name}.csv.gz").write_bytes(gzip.compress(body, mtime=0))
        df = pd.read_csv(io.BytesIO(body), low_memory=False)
        df = df[[c for c in KEEP if c in df.columns]]
        df.to_csv(out / f"{name}.csv", index=False)
        time.sleep(6)  # the OECD API rate-limits bursts of requests

    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
