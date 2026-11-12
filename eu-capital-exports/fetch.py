"""Download the Eurostat and OECD series for the analysis of the EU's savings invested abroad.

    python fetch.py --out data

Eurostat (JSON-stat dissemination API), all for the EU27 as one economy unless stated:

* ``bop_eu6_q``: EU balance of payments with the rest of the world and by partner (United States,
  United Kingdom, Switzerland, offshore centres, China, Japan), quarterly, million euro:
  current, capital and financial accounts by functional category.
* ``bop_eu6_q``: portfolio investment assets and liabilities by resident sector (households,
  investment funds, insurers and pension funds, banks, government, others).
* ``bop_iip6_q``: international investment position at the end of each year, assets,
  liabilities and net, with revaluations and other changes, by partner.
* ``nasa_10_nf_tr``: gross saving, gross capital formation, consumption of fixed capital and net
  lending by institutional sector, EU27, euro area and six member states.
* ``nama_10_gdp``: GDP at current prices.
* ``nasa_10_f_bs``: financial assets of households by instrument.

OECD (SDMX API of the OECD Data Explorer):

* ``DSD_NAMAIN10@DF_TABLE2``: national saving, gross capital formation, consumption of fixed
  capital and net lending, United States, euro area, EU27, Japan, United Kingdom.
* ``DSD_NASEC20@DF_T720R_A``: financial assets of households by instrument, United States and
  Germany (as a cross-check of the Eurostat figures).

Raw responses are kept gzipped in ``data/raw/``, tidy tables in ``data/*.csv``, and
``data/fetch_log.txt`` records URL, HTTP status, size and SHA-256 of every response.
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
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "eu-capital-exports/1.0 (research companion)"}
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
OECD = "https://sdmx.oecd.org/public/rest/data"

PARTNERS = ["EXT_EU27_2020", "US", "UK", "CH", "OFFSHO", "CN_X_HK", "JP"]
BOP_ITEMS = ["CA", "CKA", "KA", "GS", "G", "S", "IN1", "IN2", "D4P__F", "FA", "FA__D__F", "FA__P__F", "FA__P__F5",
             "FA__P__F3", "FA__F__F7", "FA__O__F", "FA__R__F", "EO"]
IIP_ITEMS = ["FA", "FA__D__F", "FA__P__F", "FA__P__F5", "FA__P__F3", "FA__F__F7", "FA__O__F", "FA__R__F"]
IIP_FLOWS = ["A_LE", "L_LE", "N_LE", "A_K7A", "A_K7B", "A_KA", "L_K7A", "L_K7B", "L_KA", "N_K7A", "N_K7B", "N_KA"]
SECTORS = ["S1", "S11", "S121", "S12T", "S124", "S12Q", "S12O", "S13", "S1M", "S1V", "S1P", "S12M"]
GEOS_NA = ["EU27_2020", "EA20", "DE", "FR", "IT", "ES", "NL", "PT"]


def multi(name: str, values: list[str]) -> list[tuple[str, str]]:
    return [(name, v) for v in values]


# name -> (dataset, filters)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "bop_flows": ("bop_eu6_q", [("geo", "EU27_2020"), ("currency", "MIO_EUR"), ("sector10", "S1"),
                                ("sectpart", "S1"), ("s_adj", "NSA"), ("freq", "Q")]
                  + multi("partner", PARTNERS) + multi("stk_flow", ["BAL", "NET", "ASS", "LIAB", "CRE", "DEB"])
                  + multi("bop_item", BOP_ITEMS)),
    "bop_portfolio_sectors": ("bop_eu6_q", [("geo", "EU27_2020"), ("currency", "MIO_EUR"), ("sectpart", "S1"),
                                            ("s_adj", "NSA"), ("freq", "Q")]
                              + multi("partner", ["EXT_EU27_2020", "US"]) + multi("stk_flow", ["ASS", "LIAB"])
                              + multi("bop_item", ["FA__P__F", "FA__P__F5", "FA__P__F3"]) + multi("sector10", SECTORS)),
    "iip": ("bop_iip6_q", [("geo", "EU27_2020"), ("currency", "MIO_EUR"), ("sector10", "S1"), ("sectpart", "S1"),
                           ("freq", "A")]
            + multi("partner", ["EXT_EU27_2020", "US", "UK", "CH", "OFFSHO"]) + multi("stk_flow", IIP_FLOWS)
            + multi("bop_item", IIP_ITEMS)),
    "sector_accounts": ("nasa_10_nf_tr", [("unit", "CP_MEUR")] + multi("geo", GEOS_NA)
                        + multi("sector", ["S1", "S11", "S12", "S13", "S14_S15"])
                        + multi("na_item", ["B8G", "B8N", "P5G", "P51G", "P51C", "B9", "B6G"])),
    "gdp": ("nama_10_gdp", [("unit", "CP_MEUR")] + multi("geo", GEOS_NA) + multi("na_item", ["B1GQ", "P51G", "P5G"])),
    "household_assets": ("nasa_10_f_bs", [("unit", "MIO_EUR"), ("co_nco", "NCO"), ("sector", "S14_S15"),
                                          ("finpos", "ASS")] + multi("geo", GEOS_NA)
                         + multi("na_item", ["F", "F2", "F3", "F5", "F51", "F52", "F6"])),
}

OECD_REQUESTS: dict[str, tuple[str, str, str]] = {
    # FREQ.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.
    # PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
    "oecd_saving_investment": ("OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE2,",
                               "A.USA+EA20+EU27_2020+JPN+GBR.S1.S1.B1GQ+B8N+P51C+P5+B9....XDC.V..",
                               "startPeriod=1995"),
    # FREQ.ADJUSTMENT.REF_AREA.COUNTERPART_AREA.SECTOR.COUNTERPART_SECTOR.CONSOLIDATION.ACCOUNTING_ENTRY.
    # TRANSACTION.INSTR_ASSET.MATURITY.UNIT_MEASURE.CURRENCY_DENOM.VALUATION.PRICE_BASE.TRANSFORMATION.
    # TABLE_IDENTIFIER.DEBT_BREAKDOWN
    "oecd_household_assets": ("OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.1",
                              "A..USA+DEU..S1M..N.A.LE.F+F2+F3+F5+F51+F52+F6..XDC......", "startPeriod=2005"),
}
OECD_KEEP = ["REF_AREA", "SECTOR", "TRANSACTION", "INSTR_ASSET", "UNIT_MEASURE", "PRICE_BASE", "TIME_PERIOD",
             "OBS_VALUE", "UNIT_MULT"]


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


def jsonstat_to_long(doc: dict) -> pd.DataFrame:
    """Flatten a JSON-stat 2.0 dataset into one row per cell."""
    ids: list[str] = doc["id"]
    sizes: list[int] = doc["size"]
    cats = []
    for d in ids:
        index = doc["dimension"][d]["category"]["index"]
        codes = index if isinstance(index, list) else [c for c, _ in sorted(index.items(), key=lambda kv: kv[1])]
        cats.append(codes)
    values = doc.get("value", {})
    status = doc.get("status", {})
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    if isinstance(status, list):
        status = {str(i): s for i, s in enumerate(status) if s}
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    rows = []
    for key in sorted(values, key=int):
        k = int(key)
        row = {d: codes[(k // stride) % size] for d, codes, stride, size in zip(ids, cats, strides, sizes)}
        row["value"] = values[key]
        row["flag"] = status.get(key, "")
        rows.append(row)
    return pd.DataFrame(rows)


def record(out: Path, log: list[str], name: str, url: str, status: int, body: bytes, ext: str) -> bool:
    log.append(f"{name}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
    if status != 200 or len(body) < 200:
        return False
    (out / "raw" / f"{name}.{ext}.gz").write_bytes(gzip.compress(body, mtime=0))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    out: Path = args.out
    (out / "raw").mkdir(parents=True, exist_ok=True)
    log = [f"# fetched {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
           "# name\tstatus\tbytes\tsha256\turl"]
    failures = 0
    for name, (dataset, filters) in EUROSTAT_REQUESTS.items():
        params = [("lang", "en"), ("format", "JSON")] + filters
        url = f"{EUROSTAT}/{dataset}?{urllib.parse.urlencode(params)}"
        try:
            status, body = get(url)
        except RuntimeError as e:
            log.append(f"{name}\tERROR\t{e}")
            failures += 1
            continue
        if not record(out, log, name, url, status, body, "json"):
            failures += 1
            continue
        df = jsonstat_to_long(json.loads(body))
        df.insert(0, "dataset", dataset)
        df.drop(columns=[c for c in ("flag",) if c in df.columns and (df[c] == "").all()], inplace=True)
        df.to_csv(out / f"{name}.csv", index=False)
    for name, (flow, key, query) in OECD_REQUESTS.items():
        url = f"{OECD}/{flow}/{key}?{query}&dimensionAtObservation=AllDimensions&format=csv"
        try:
            status, body = get(url)
        except RuntimeError as e:
            log.append(f"{name}\tERROR\t{e}")
            failures += 1
            continue
        if not record(out, log, name, url, status, body, "csv"):
            failures += 1
            continue
        df = pd.read_csv(io.BytesIO(body), low_memory=False)
        df = df[[c for c in OECD_KEEP if c in df.columns]]
        df.to_csv(out / f"{name}.csv", index=False)
        time.sleep(6)  # the OECD API rate-limits bursts of requests
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
