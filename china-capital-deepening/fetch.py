"""Download the data for the test of capital deepening in Chinese manufacturing.

    python fetch.py --out data

Two sources, both public:

* OECD Data Explorer (SDMX API):
  - ``DSD_EO_LTB@DF_EO_LTB`` (Economic Outlook long-term baseline): productive capital stock in volume,
    GDP in volume and potential employment for China and 13 comparison economies, 1990-2025;
  - ``DSD_NAMAIN10@DF_TABLE1_EXPENDITURE``: GDP at current prices and in volume and gross fixed capital
    formation at current prices, for the investment rate and the incremental capital-output ratio;
  - ``DSD_NAMAIN10@DF_TABLE9A``, ``@DF_TABLE6`` and ``@DF_TABLE7``: net fixed assets, value added and
    employment (persons and hours) of manufacturing (ISIC Rev. 4 section C) in OECD countries.
* China Industrial Productivity (CIP) database, 4.0 (2023 release), RIETI: value added, its distribution,
  producer prices, investment, capital stock, employment and hours for 37 Chinese industries.

Raw OECD responses are kept gzipped in ``data/raw/``, the CIP workbooks as published in ``data/raw/cip/``,
tidy OECD tables in ``data/*.csv``; ``data/fetch_log.txt`` records URL, HTTP status, size and SHA-256 of
every response. Figures published by China's National Bureau of Statistics that are used for the years after
the CIP database ends are not downloaded: they are recorded by hand, with the URL of each release, in
``data/nbs_published.csv``.
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

UA = {"User-Agent": "china-capital-deepening/1.0 (research companion)"}
OECD = "https://sdmx.oecd.org/public/rest/data"
CIP = "https://www.rieti.go.jp/en/database/CIP2023/data"

ECONOMIES = "CHN+USA+JPN+KOR+DEU+FRA+ITA+ESP+NLD+POL+CZE+MEX+IND+IDN"
MANUF = "DEU+FRA+ITA+ESP+NLD+AUT+BEL+CZE+POL+PRT+SWE+JPN+USA"

# name: (dataflow, key, query)
OECD_REQUESTS = {
    # REF_AREA.MEASURE.SCENARIO.FREQ; history is identical across scenarios
    "oecd_ltb": ("OECD.ECO.MAD,DSD_EO_LTB@DF_EO_LTB,", f"{ECONOMIES}.KTPV_AV+GDPV+ETPT.BAU1.A",
                 "startPeriod=1990&endPeriod=2025"),
    # FREQ.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.
    # PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
    "oecd_t1": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_EXPENDITURE,", f"A.{ECONOMIES}...B1GQ+P51G....XDC.V+LR..",
                "startPeriod=1980"),
    "oecd_t9a": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE9A,", f"A.{MANUF}...LE.N11N.C...L+V..", "startPeriod=1990"),
    "oecd_t6": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE6,", f"A.{MANUF}...B1G._Z.C...L+V..", "startPeriod=1990"),
    "oecd_t7": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE7,", f"A.{MANUF}...EMP+SAL._Z.C.....", "startPeriod=1990"),
}
KEEP = ["REF_AREA", "MEASURE", "SCENARIO", "TRANSACTION", "INSTR_ASSET", "ACTIVITY", "UNIT_MEASURE",
        "PRICE_BASE", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT"]

# CIP 4.0 workbooks: value added (1.2), its distribution (1.4), producer prices (1.6), investment (2.1),
# capital stock (2.2), capital input (2.3), employment (3.1), hours (3.2), growth accounts (4), classification (5)
CIP_FILES = ["1.2", "1.4", "1.6", "2.1", "2.2", "2.3", "3.1", "3.2", "4", "5"]


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
    ap.add_argument("--only", choices=["oecd", "cip"], help="fetch one source only")
    args = ap.parse_args()
    out: Path = args.out
    (out / "raw" / "cip").mkdir(parents=True, exist_ok=True)
    log = [f"# fetched {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
           "# name\tstatus\tbytes\tsha256\turl"]
    failures = 0
    if args.only in (None, "oecd"):
        for name, (flow, key, query) in OECD_REQUESTS.items():
            url = f"{OECD}/{flow}/{key}?{query}&dimensionAtObservation=AllDimensions&format=csv"
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
            df[[c for c in KEEP if c in df.columns]].to_csv(out / f"{name}.csv", index=False)
            time.sleep(6)  # the OECD API rate-limits bursts of requests
    if args.only in (None, "cip"):
        for f in CIP_FILES:
            name = f"CIP_4.0_(2023)_{f}.xlsx"
            url = f"{CIP}/{name}"
            try:
                status, body = get(url)
            except RuntimeError as e:
                log.append(f"cip_{f}\tERROR\t{e}")
                failures += 1
                continue
            log.append(f"cip_{f}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
            if status != 200 or not body.startswith(b"PK"):
                failures += 1
                continue
            (out / "raw" / "cip" / name).write_bytes(body)
    log_path = out / "fetch_log.txt"
    if args.only and log_path.exists():  # keep the other source's lines
        kept = [ln for ln in log_path.read_text().splitlines()[2:]
                if not ln.startswith("cip_" if args.only == "cip" else "oecd_")]
        log += kept
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
