"""Download the OECD and Eurostat series used to test the ICT-deflator explanation of the EU-US price drift.

    python fetch.py --out data

OECD Data Explorer (SDMX API), national accounts (DSD_NAMAIN10):

* ``DF_TABLE1_EXPENDITURE``: GDP (B1GQ), actual individual consumption (P41) and household final
  consumption expenditure (P3, sector S14), national currency, current prices and chain-linked volumes,
  every economy the OECD publishes.
* ``DF_TABLE4``: purchasing power parities for GDP, household final consumption and actual individual
  consumption, national currency per US dollar, and exchange rates.
* ``DF_TABLE5_T501``: household consumption by purpose (COICOP), current prices and volumes, for the United
  States, the EU members that report it and Japan. Used for the consumer prices of ICT goods and services.
* ``DF_TABLE6``: gross value added by industry (ISIC Rev. 4), current prices and volumes. Used for the value
  added deflators of the ICT-producing industries (C26, J) and of everything else.
* ``DF_TABLE1_EXPENDITURE_GFCF_ASSET``: investment in ICT equipment, software and in total, current prices
  and volumes. Used to size the investment channel.

Eurostat (``prc_ppp_ind``): PPPs relative to the EU27 for GDP, actual individual consumption and household
consumption, as a cross-check of the OECD PPPs for the United States and Japan.

Raw responses are kept gzipped in ``data/raw/``, tidy tables in ``data/*.csv``, and ``data/fetch_log.txt``
records the URL, HTTP status, size and SHA-256 of every response.
"""

from __future__ import annotations

import argparse
import gzip
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

UA = {"User-Agent": "eu-us-ppp-drift/1.0 (research companion)"}
OECD = "https://sdmx.oecd.org/public/rest/data"
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

EU = ["AUT", "BEL", "BGR", "HRV", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
      "LTU", "LUX", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]
DETAIL = "+".join(["USA", "JPN", "KOR", "CAN", "GBR", *EU])

# FREQ.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.
# PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
OECD_REQUESTS = {
    "oecd_t1": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_EXPENDITURE,", "A..S1+S14..B1GQ+P3+P41....XDC.V+LR..",
                "startPeriod=1995"),
    "oecd_t4": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE4,", "A.............", "startPeriod=1995"),
    "oecd_t5": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE5_T501,", f"A.{DETAIL}.......XDC.V+LR..", "startPeriod=1995"),
    "oecd_t6": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE6,", f"A.{DETAIL}...B1G....XDC.V+LR..", "startPeriod=1995"),
    "oecd_gfcf": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_EXPENDITURE_GFCF_ASSET,",
                  f"A.{DETAIL}...P51G.N11G+N1132G+N11321G+N11322G+N1173G...XDC.V+LR..", "startPeriod=1995"),
}
KEEP = ["REF_AREA", "SECTOR", "TRANSACTION", "INSTR_ASSET", "ACTIVITY", "EXPENDITURE", "UNIT_MEASURE",
        "PRICE_BASE", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT"]

ES_GEO = ["US", "JP", "EU27_2020", "BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "LV",
          "LT", "LU", "HU", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE"]
ES_REQUEST = ("prc_ppp_ind", "format=JSON&na_item=PPP_EU27_2020&ppp_cat=GDP&ppp_cat=A01&ppp_cat=E011&"
              + "&".join(f"geo={g}" for g in ES_GEO))


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


def eurostat_table(body: bytes) -> pd.DataFrame:
    """Flatten a Eurostat JSON-stat response into one row per observation."""
    d = json.loads(body)
    dims, sizes = d["id"], d["size"]
    labels = [{i: k for k, i in d["dimension"][dim]["category"]["index"].items()} for dim in dims]
    rows = []
    for flat, value in d["value"].items():
        k, idx = int(flat), []
        for s in reversed(sizes):
            idx.append(k % s)
            k //= s
        idx.reverse()
        rows.append([labels[i][j] for i, j in enumerate(idx)] + [value])
    return pd.DataFrame(rows, columns=dims + ["value"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    out: Path = args.out
    (out / "raw").mkdir(parents=True, exist_ok=True)
    log = [f"# fetched {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
           "# name\tstatus\tbytes\tsha256\turl"]
    failures = 0
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
    table, query = ES_REQUEST
    url = f"{EUROSTAT}/{table}?{query}"
    try:
        status, body = get(url)
        log.append(f"eurostat_ppp\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
        if status == 200:
            (out / "raw" / "eurostat_ppp.json.gz").write_bytes(gzip.compress(body, mtime=0))
            eurostat_table(body).to_csv(out / "eurostat_ppp.csv", index=False)
        else:
            failures += 1
    except RuntimeError as e:
        log.append(f"eurostat_ppp\tERROR\t{e}")
        failures += 1
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
