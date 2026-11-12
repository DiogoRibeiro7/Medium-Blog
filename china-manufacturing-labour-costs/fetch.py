"""Download the OECD series for the China manufacturing labour-cost analysis.

    python fetch.py --out data

Three requests to the SDMX API of the OECD Data Explorer:

* ``DSD_TIM_2025@DF_TIM_2025`` (Trade in Employment, 2025 edition, built on the OECD
  Inter-Country Input-Output tables): employment (persons engaged), compensation of
  employees (US dollars at market exchange rates, and as a share of value added) and value
  added per person engaged, for total manufacturing and its 2-digit industries, 1995-2022.
* ``DSD_TIVA_MAINLV@DF_MAINLV`` (Trade in Value Added, 2025 edition, principal indicators):
  value added in US dollars by country and industry, including the world total.
* ``DSD_NAMAIN10@DF_TABLE4`` (national accounts): average exchange rates and GDP PPPs
  against the US dollar, used to split China's dollar wage growth into yuan wage growth
  and exchange-rate movement.

Raw responses are kept gzipped in ``data/raw/``, tidy tables in ``data/*.csv``, and
``data/fetch_log.txt`` records URL, HTTP status, size and SHA-256 of every response.
Published IFR robot statistics are not downloadable; they are kept by hand in
``data/ifr_published.csv`` with the press release each value comes from.
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

UA = {"User-Agent": "china-manufacturing-labour-costs/1.0 (research companion)"}

INDUSTRIES = ("C", "C10T12", "C13T15", "C16", "C17_18", "C19", "C20", "C21", "C22", "C23", "C24", "C25",
              "C26", "C27", "C28", "C29", "C30", "C31T33", "_T")

REQUESTS = {
    # name -> (base, agency,dataflow,version, key, query)
    "oecd_tim": ("https://sdmx.oecd.org/public/rest/data", "OECD.STI.PIE,DSD_TIM_2025@DF_TIM_2025,",
                 "EMPN+LABR+VALU_EMPN.." + "+".join(INDUSTRIES) + ".W..A", ""),
    "oecd_tiva": ("https://sdmx.oecd.org/sti-public/rest/data", "OECD.STI.PIE,DSD_TIVA_MAINLV@DF_MAINLV,1.1",
                  "VALU..C+C13T15+C26+C29+_T.W.USD.A", ""),
    "oecd_fx": ("https://sdmx.oecd.org/public/rest/data", "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE4,",
                "A.CHN+USA+MEX+IND+DEU+PRT+POL+ROU+BGR+HUN+CZE+ESP+ITA+FRA+JPN+KOR+IDN+TUR+BRA..........",
                "startPeriod=1995"),
}

KEEP = ["REF_AREA", "MEASURE", "TRANSACTION", "ACTIVITY", "COUNTERPART_AREA", "UNIT_MEASURE", "TIME_PERIOD",
        "OBS_VALUE", "OBS_STATUS", "UNIT_MULT"]

# economies kept in the tidy tables (the raw responses keep everything)
AREAS = {
    "CHN", "VNM", "IND", "IDN", "THA", "MYS", "PHL", "BGD", "KHM", "PAK", "MMR", "LAO", "TWN", "KOR", "JPN",
    "MEX", "BRA", "TUR", "MAR", "USA", "CAN",
    "DEU", "FRA", "ITA", "ESP", "PRT", "GRC", "NLD", "BEL", "AUT", "SWE", "FIN", "DNK", "IRL",
    "POL", "CZE", "SVK", "HUN", "ROU", "BGR", "SVN", "HRV", "LTU", "LVA", "EST",
    "W", "EU27_2020",
}


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
    for name, (base, flow, key, query) in REQUESTS.items():
        q = (query + "&" if query else "") + "dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
        url = f"{base}/{flow}/{key}?{q}"
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
        df = df[df.REF_AREA.isin(AREAS)]
        df.to_csv(out / f"{name}.csv", index=False)
        time.sleep(6)  # the OECD API rate-limits bursts of requests
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
