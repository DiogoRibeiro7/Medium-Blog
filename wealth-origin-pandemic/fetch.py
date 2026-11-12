"""Download the pandemic-exposure series for the wealth-origin-pandemic analysis.

    python fetch.py --out data

Sources
-------
- Oxford COVID-19 Government Response Tracker (OxCGRT), national compact file from the
  OxCGRT/covid-policy-dataset repository: daily stringency, containment and economic-support
  indices and cumulative confirmed deaths, averaged (indices) or taken at year end (deaths) to
  country x year for 2020-2022.
- International Labour Organization (ILOSTAT): employment by economic activity, total for both
  sexes, 2014-2019: survey-based counts by ISIC Rev. 4 section and the ILO modelled estimates by
  detailed activity. Several published endpoints are tried in turn; the log records which one
  answered.
- World Bank WDI: international tourism receipts (current US$ and % of exports), travel credits
  in the balance of payments (% of service exports) and service exports (current US$), services
  value added (% of GDP), services employment (% of employment), population.

The Forbes, Penn World Table and WDI GDP tables are copied from the wealth-origin-sectors
snapshot (see data/PROVENANCE.txt), so they are not downloaded again.

The workspace where the analysis is written cannot reach most of these hosts, so
``.github/workflows/wealth-origin-pandemic-data.yml`` runs this script on GitHub's runners and
commits the tidy tables; ``data/fetch_log.txt`` records URL, status, size and SHA-256 of every
response.
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

UA = {"User-Agent": "wealth-origin-pandemic/1.0 (research companion; dfr@esmad.ipp.pt)"}

OXCGRT = ("https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/"
          "OxCGRT_compact_national_v1.csv")
OX_INDICES = ["StringencyIndex_Average", "ContainmentHealthIndex_Average", "EconomicSupportIndex"]

ILO_BASE = [
    "https://rplumber.ilo.org/data/indicator/?id={id}&lang=en&type=code&format=.csv.gz&channel=ilostat",
    "https://rplumber.ilo.org/data/indicator/?id={id}&type=code&format=.csv",
    "https://www.ilo.org/ilostat-files/WEB_bulk_download/indicator/{id}.csv.gz",
]
# survey-based employment by ISIC Rev. 4 section, and ILO modelled estimates by detailed activity
ILO_SERIES = {
    "EMP_TEMP_SEX_ECO_NB_A": ("ilostat_emp_isic4.csv",
                              {"ECO_ISIC4_G", "ECO_ISIC4_H", "ECO_ISIC4_I", "ECO_ISIC4_R", "ECO_ISIC4_S", "ECO_ISIC4_TOTAL"}),
    "EMP_2EMP_SEX_ECO_NB_A": ("ilostat_emp_modelled.csv", None),
}

WDI = "https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=5000&date=2010:2023"
WDI_CODES = {
    "ST.INT.RCPT.CD": "tourism_receipts_usd",
    "ST.INT.RCPT.XP.ZS": "tourism_receipts_pct_exports",
    "NV.SRV.TOTL.ZS": "services_va_pct_gdp",
    "SL.SRV.EMPL.ZS": "services_emp_pct",
    "SP.POP.TOTL": "population",
    "BX.GSR.TRVL.ZS": "travel_pct_service_exports",
    "BX.GSR.NFSV.CD": "service_exports_usd",
}

LOG: list[str] = []


def log(url: str, status: int | str, payload: bytes | None = None) -> None:
    size = sha = None
    if payload is not None:
        size = len(payload)
        sha = hashlib.sha256(payload).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOG.append(f"{stamp}\t{status}\t{size}\t{sha}\t{url}")
    print(LOG[-1], flush=True)


def get(url: str, tries: int = 4, timeout: int = 600) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                log(url, r.status, body)
                return r.status, body
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                last = e
            else:
                body = e.read()
                log(url, e.code, body)
                return e.code, body
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(10 * (attempt + 1))
    log(url, f"failed: {last}")
    raise RuntimeError(f"{url}: {last}")


def fetch_oxcgrt(out: Path) -> None:
    status, body = get(OXCGRT)
    if status != 200:
        raise RuntimeError(f"OxCGRT returned {status}")
    cols = ["CountryCode", "Jurisdiction", "Date", "ConfirmedDeaths"] + OX_INDICES
    d = pd.read_csv(io.BytesIO(body), usecols=cols, low_memory=False)
    d = d[d["Jurisdiction"] == "NAT_TOTAL"].copy()
    d["year"] = d["Date"] // 10000
    d = d[d["year"].between(2020, 2022)]
    g = d.groupby(["CountryCode", "year"])
    out_df = g[OX_INDICES].mean()
    out_df["deaths_cum_year_end"] = g["ConfirmedDeaths"].max()
    out_df["days"] = g.size()
    out_df = out_df.reset_index().rename(columns={"CountryCode": "iso3"})
    out_df.to_csv(out / "oxcgrt_country_year.csv", index=False, float_format="%.4f")
    print(f"OxCGRT: {out_df['iso3'].nunique()} countries", flush=True)


def _read_any(body: bytes) -> pd.DataFrame:
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return pd.read_csv(io.BytesIO(body), low_memory=False)


def fetch_ilostat(out: Path) -> None:
    failed = []
    for ilo_id, (fname, keep_classes) in ILO_SERIES.items():
        errors = []
        for base in ILO_BASE:
            url = base.format(id=ilo_id)
            try:
                status, body = get(url, tries=2)
            except RuntimeError as e:
                errors.append(str(e))
                continue
            if status != 200 or len(body) < 1000:
                errors.append(f"{url}: status {status}, {len(body)} bytes")
                continue
            try:
                d = _read_any(body)
            except Exception as e:  # noqa: BLE001 - try the next endpoint
                errors.append(f"{url}: unreadable ({e})")
                continue
            need = {"ref_area", "sex", "classif1", "time", "obs_value"}
            if not need <= set(d.columns):
                errors.append(f"{url}: columns {list(d.columns)[:12]}")
                continue
            d = d[(d["sex"] == "SEX_T") & d["time"].between(2014, 2019)]
            d = d[d["classif1"].isin(keep_classes)] if keep_classes else d[d["classif1"].str.startswith("ECO_")]
            keep = [c for c in ["ref_area", "source", "classif1", "time", "obs_value"] if c in d.columns]
            d[keep].rename(columns={"ref_area": "iso3"}).to_csv(out / fname, index=False)
            print(f"ILOSTAT {ilo_id}: {d['ref_area'].nunique()} areas from {url}", flush=True)
            break
        else:
            failed.append(f"{ilo_id}: " + " | ".join(errors))
    if failed:
        raise RuntimeError("; ".join(failed))


def fetch_wdi(out: Path) -> None:
    """One request per indicator and page; the API answers 502 now and then, so retry patiently."""
    rows, failed = [], []
    for code, name in WDI_CODES.items():
        page, pages = 1, 1
        try:
            while page <= pages:
                status, body = get(WDI.format(code=code) + f"&page={page}", tries=8)
                if status != 200:
                    raise RuntimeError(f"WDI {code} returned {status}")
                meta, data = json.loads(body)
                pages = int(meta.get("pages", 1))
                rows += [{"iso3": d["countryiso3code"], "year": int(d["date"]), "indicator": name, "value": d["value"]}
                         for d in (data or []) if d["countryiso3code"] and d["value"] is not None]
                page += 1
        except RuntimeError as e:
            failed.append(f"{code}: {e}")
        time.sleep(2)
    pd.DataFrame(rows).to_csv(out / "wdi_exposure.csv", index=False)
    if failed:
        raise RuntimeError("; ".join(failed))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--only", choices=["oxcgrt", "ilostat", "wdi"], nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    steps = {"oxcgrt": fetch_oxcgrt, "ilostat": fetch_ilostat, "wdi": fetch_wdi}
    failed = []
    for name, fn in steps.items():
        if args.only and name not in args.only:
            continue
        try:
            fn(args.out)
        except Exception as e:  # keep going so the log shows every failure
            print(f"{name} failed: {e}", file=sys.stderr, flush=True)
            LOG.append(f"# {name} failed: {e}")
            failed.append(name)
    with open(args.out / "fetch_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
