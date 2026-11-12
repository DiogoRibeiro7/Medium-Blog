"""Download the Eurostat and OECD series for the minimum-wage and in-work-poverty analysis.

    python fetch.py --out data

Eurostat datasets come from the JSON-stat dissemination API, one request per dataset for
all member states at once:

* ``earn_mw_cur``: statutory monthly minimum wages in euro and national currency, twice a year, 1999-2026.
* ``prc_hicp_aind``: harmonised index of consumer prices, annual average, all items.
* ``earn_ses_mdeci`` (2006), ``earn_ses10_mdeci``, ``earn_ses14_mdeci``, ``earn_ses18_mdeci``,
  ``earn_ses22_mdeci``: 10th percentile, median and 90th percentile of monthly earnings in
  the Structure of Earnings Survey.
* ``nama_10_fte``: average full-time adjusted salary per employee (national accounts).
* ``ilc_li01``: at-risk-of-poverty thresholds by household type (EU-SILC).
* ``ilc_iw01``, ``ilc_iw02``, ``ilc_iw03``, ``ilc_iw05``, ``ilc_iw07``: in-work at-risk-of-poverty
  rates by activity status, household type, work intensity, contract and working time.
* ``ilc_li02``: at-risk-of-poverty rate of the whole population (context).

OECD datasets come from the SDMX API of the OECD Data Explorer (TaxBEN model):

* ``DSD_TAXBEN_IMW@DF_IMW``: annual gross earnings and net family income of minimum-wage
  earners by household type.
* ``DSD_TAXBEN_HOURSPOV@DF_HOURSPOV``: weekly hours of work needed to escape poverty (50% of
  median disposable income) for a family claiming minimum-income benefits.

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

UA = {"User-Agent": "minimum-wage-in-work-poverty/1.0 (research companion)"}
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
OECD = "https://sdmx.oecd.org/public/rest/data"

GEOS = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT",
        "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK", "EU27_2020"]

# name -> (dataset, fixed filters)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "minimum_wage": ("earn_mw_cur", [("currency", "EUR"), ("currency", "NAC")]),
    "hicp": ("prc_hicp_aind", [("unit", "INX_A_AVG"), ("coicop", "CP00")]),
    "ses_2006": ("earn_ses_mdeci", [("unit", "EUR"), ("unit", "NAC"), ("worktime", "FT")]),
    "ses_2010": ("earn_ses10_mdeci", [("unit", "EUR"), ("unit", "NAC"), ("worktime", "FT")]),
    "ses_2014": ("earn_ses14_mdeci", [("unit", "EUR"), ("unit", "NAC"), ("worktime", "FT")]),
    "ses_2018": ("earn_ses18_mdeci", [("unit", "EUR"), ("unit", "NAC"), ("worktime", "FT")]),
    "ses_2022": ("earn_ses22_mdeci", [("unit", "EUR"), ("unit", "NAC"), ("worktime", "FT")]),
    "fte_salary": ("nama_10_fte", [("unit", "EUR"), ("unit", "NAC")]),
    "poverty_thresholds": ("ilc_li01", [("unit", "EUR")]),
    "iw_status": ("ilc_iw01", [("sex", "T"), ("age", "Y18-64")]),
    "iw_household": ("ilc_iw02", []),
    "iw_intensity": ("ilc_iw03", []),
    "iw_contract": ("ilc_iw05", [("sex", "T")]),
    "iw_worktime": ("ilc_iw07", []),
    "arop": ("ilc_li02", [("statinfo", "MED_EI"), ("rskpovth", "B_60"), ("unit", "PC"), ("sex", "T"), ("age", "TOTAL")]),
}

OECD_AREAS = "PRT+ESP+GRC+ITA+FRA+DEU+POL+ROU+BGR+HUN+CZE+SVK+SVN+HRV+LTU+LVA+EST+BEL+NLD+IRL+LUX+MLT+CYP+OECD"
OECD_REQUESTS: dict[str, tuple[str, str]] = {
    # dims: REF_AREA.MEASURE.UNIT_MEASURE.HOUSEHOLD_TYPE.AGE_CHILDREN.INCOME_PART.SOC_ASS_BENEFIT.
    #       HOUSE_BENEFIT.STAT_REF_POP.METHOD_MIN_WAGE.FREQ
    "oecd_imw": ("OECD.ELS.JAI,DSD_TAXBEN_IMW@DF_IMW,1.0",
                 f"{OECD_AREAS}.NFI+GFE.XDC.....NO.AE_FTE.MW_WA_YEAR.A"),
    # dims: REF_AREA.MEASURE.UNIT_MEASURE.INCOME_CURR.HOUSEHOLD_TYPE.AGE_CHILDREN.HOUSE_BENEFIT.
    #       TEMP_INTOWORK_BENEFIT.FREQ
    "oecd_hours_poverty": ("OECD.ELS.JAI,DSD_TAXBEN_HOURSPOV@DF_HOURSPOV,1.0", "...MINW+AW67...NO.NO.A"),
}
OECD_KEEP = ["REF_AREA", "MEASURE", "UNIT_MEASURE", "HOUSEHOLD_TYPE", "AGE_CHILDREN", "INCOME_PART",
             "SOC_ASS_BENEFIT", "HOUSE_BENEFIT", "INCOME_CURR", "TEMP_INTOWORK_BENEFIT", "TIME_PERIOD", "OBS_VALUE"]


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
        params = [("lang", "en"), ("format", "JSON")] + filters + [("geo", g) for g in GEOS]
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
        df.to_csv(out / f"{name}.csv", index=False)
    for name, (flow, key) in OECD_REQUESTS.items():
        url = f"{OECD}/{flow}/{key}?dimensionAtObservation=AllDimensions&format=csv"
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
