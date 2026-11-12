"""Download the OECD series for the industry decomposition of the EU-US operating surplus gap.

    python fetch.py --out data

Two requests to the SDMX API of the OECD Data Explorer:

* ``DSD_STAN@DF_STAN_2025`` (STAN Database for Structural Analysis, 2025 edition): value added,
  compensation of employees, gross and net operating surplus and mixed income, consumption of fixed
  capital, other taxes less subsidies on production, employment, employees and the self-employed
  (persons and hours), by ISIC Rev. 4 industry, for the United States and the eight North-West
  European countries of Part 1 of the series (Austria, Belgium, Denmark, Finland, France, Germany,
  the Netherlands, Sweden), current prices, from 2015.
* ``DSD_NAMAIN10@DF_TABLE4`` (national accounts): GDP purchasing power parities and average
  exchange rates against the US dollar.

Raw responses are kept gzipped in ``data/raw/``, tidy tables in ``data/*.csv``, and
``data/fetch_log.txt`` records URL, HTTP status, size and SHA-256 of every response.
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

UA = {"User-Agent": "eu-us-surplus-by-industry/1.0 (research companion)"}
BASE = "https://sdmx.oecd.org/public/rest/data"
AREAS = "USA+AUT+BEL+DNK+FIN+FRA+DEU+NLD+SWE"
MEASURES = "B1G+D1+B2A3G+B2A3N+P51C+D29X39+EMP+SAL+SELF"

REQUESTS = {
    # FREQ.REF_AREA.ACTIVITY.MEASURE.PRICE_BASE.UNIT_MEASURE
    "oecd_stan": ("OECD.STI.PIE,DSD_STAN@DF_STAN_2025,1.0", f"A.{AREAS}..{MEASURES}..", "startPeriod=2015"),
    # FREQ.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.
    # PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
    "oecd_ppp": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE4,", f"A.{AREAS}...PPP_B1GQ+EXC_A.......", "startPeriod=2015"),
}
KEEP = ["REF_AREA", "ACTIVITY", "MEASURE", "TRANSACTION", "PRICE_BASE", "UNIT_MEASURE", "TIME_PERIOD", "OBS_VALUE",
        "UNIT_MULT"]


def get(url: str, tries: int = 4) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=600) as r:
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
    for name, (flow, key, query) in REQUESTS.items():
        url = f"{BASE}/{flow}/{key}?{query}&dimensionAtObservation=AllDimensions&format=csv"
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
        if "PRICE_BASE" in df.columns:
            df = df[df.PRICE_BASE.isin(["V", "_Z"])]
        df.to_csv(out / f"{name}.csv", index=False)
        time.sleep(6)  # the OECD API rate-limits bursts of requests
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
