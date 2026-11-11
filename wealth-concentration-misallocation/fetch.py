"""Download the public series for the wealth-concentration-misallocation analysis.

    python fetch.py --out data

Sources
-------
- World Inequality Database (WID.world), full bulk download: top 1%, top 10% and bottom 50%
  shares of net personal wealth (shweal992j) and of pre-tax national income (sptinc992j),
  adults, equal-split; plus the WID metadata rows for those two variables.
- Penn World Table 11.0 (Feenstra, Inklaar and Timmer), read through the DataverseNL API
  (doi:10.34894/FABVLR): real GDP, population, TFP, human capital, investment share.
- World Bank WDI: GDP in current US dollars (NY.GDP.MKTP.CD) and the country list, used for
  billionaire wealth / GDP and for the ISO2 -> ISO3 mapping.
- Billionaire Characteristics Database (Freund and Oliver 2016, PIIE WP 16-1), as
  redistributed by the CORGIS project: Forbes billionaires in 1996, 2001 and 2014 with the
  source of the fortune (inherited, founder, finance, privatised/resources, executive).

The workspace where the analysis was written cannot reach WID, DataverseNL or the World
Bank API, so ``.github/workflows/wealth-concentration-misallocation-data.yml`` runs this
script on GitHub's runners and commits the tidy tables. Raw downloads are not committed;
``data/fetch_log.txt`` records URL, HTTP status, size and SHA-256 of every response.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "wealth-concentration-misallocation/1.0 (research companion; dfr@esmad.ipp.pt)"}

WID_BULK = "https://wid.world/bulk_download/wid_all_data.zip"
WID_VARIABLES = {"shweal992j", "sptinc992j"}
WID_PERCENTILES = {"p99p100", "p90p100", "p0p50"}

PWT_DOI = "doi:10.34894/FABVLR"
PWT_LISTING = f"https://dataverse.nl/api/datasets/:persistentId/?persistentId={PWT_DOI}"
PWT_FILE = "https://dataverse.nl/api/access/datafile/{id}"
PWT_COLUMNS = [
    "countrycode", "country", "year", "rgdpna", "rgdpo", "rgdpe", "pop", "emp", "hc",
    "rtfpna", "csh_i", "cgdpo", "rnna", "labsh",
]

WDI_GDP = (
    "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"
    "?format=json&per_page=20000&date=1990:2024"
)
WDI_COUNTRIES = "https://api.worldbank.org/v2/country?format=json&per_page=400"

BILLIONAIRES = (
    "https://raw.githubusercontent.com/corgis-edu/corgis/master/website/datasets/csv/"
    "billionaires/billionaires.csv"
)

LOG: list[str] = []


def log(url: str, status: int | str, payload: bytes | None = None, size: int | None = None,
        sha: str | None = None) -> None:
    if payload is not None:
        size = len(payload)
        sha = hashlib.sha256(payload).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOG.append(f"{stamp}\t{status}\t{size}\t{sha}\t{url}")
    print(LOG[-1], flush=True)


def get(url: str, tries: int = 4) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
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
        time.sleep(5 * (attempt + 1))
    log(url, f"failed: {last}")
    raise RuntimeError(f"{url}: {last}")


def stream_to_file(url: str, dest: Path, tries: int = 3) -> None:
    """Download a large file to disk, logging its size and SHA-256."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            h = hashlib.sha256()
            size = 0
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
                log(url, r.status, size=size, sha=h.hexdigest())
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(10 * (attempt + 1))
    log(url, f"failed: {last}")
    raise RuntimeError(f"{url}: {last}")


def fetch_wid(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wid_all_data.zip"
        stream_to_file(WID_BULK, path)
        rows, meta = [], []
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            countries = [n for n in names if n.endswith("WID_countries.csv")]
            if countries:
                pd.read_csv(io.BytesIO(z.read(countries[0])), sep=";", keep_default_na=False) \
                    .to_csv(out / "wid_countries.csv", index=False)
            for name in names:
                stem = Path(name).name
                if stem.startswith("WID_data_") and stem.endswith(".csv"):
                    code = stem[len("WID_data_"):-4]
                    if len(code) != 2:  # sub-national areas look like US-CA
                        continue
                    df = pd.read_csv(io.BytesIO(z.read(name)), sep=";", keep_default_na=False,
                                     dtype={"country": str, "variable": str, "percentile": str})
                    df = df[df["variable"].isin(WID_VARIABLES) & df["percentile"].isin(WID_PERCENTILES)]
                    if len(df):
                        rows.append(df)
                elif stem.startswith("WID_metadata_") and stem.endswith(".csv"):
                    code = stem[len("WID_metadata_"):-4]
                    if len(code) != 2:
                        continue
                    md = pd.read_csv(io.BytesIO(z.read(name)), sep=";", keep_default_na=False, dtype=str)
                    md = md[md["variable"].isin(WID_VARIABLES)]
                    if len(md):
                        meta.append(md)
    shares = pd.concat(rows, ignore_index=True)
    shares = shares[(shares["year"] >= 1980)]
    shares.sort_values(["country", "variable", "percentile", "year"]).to_csv(out / "wid_shares.csv", index=False)
    pd.concat(meta, ignore_index=True).to_csv(out / "wid_metadata.csv", index=False)
    print(f"WID: {shares['country'].nunique()} areas, {len(shares)} rows", flush=True)


def fetch_pwt(out: Path) -> None:
    status, body = get(PWT_LISTING)
    if status != 200:
        raise RuntimeError(f"PWT listing returned {status}")
    files = json.loads(body)["data"]["latestVersion"]["files"]
    target = None
    for f in files:
        label = f.get("label") or f.get("dataFile", {}).get("filename", "")
        if label.lower() in {"pwt110.dta", "pwt110.tab"}:
            target = f
            break
    if target is None:
        raise RuntimeError("pwt110.dta not found: " + ", ".join(f.get("label", "?") for f in files))
    file_id = target["dataFile"]["id"]
    # original=true returns the deposited Stata file rather than Dataverse's tabular export
    status, body = get(PWT_FILE.format(id=file_id) + "?format=original")
    if status != 200:
        raise RuntimeError(f"PWT download returned {status}")
    pwt = pd.read_stata(io.BytesIO(body), convert_categoricals=False)
    keep = [c for c in PWT_COLUMNS if c in pwt.columns]
    pwt = pwt[keep]
    pwt = pwt[pwt["year"] >= 1980]
    pwt.to_csv(out / "pwt110_panel.csv", index=False)
    print(f"PWT: {pwt['countrycode'].nunique()} countries, years {pwt['year'].min()}-{pwt['year'].max()}", flush=True)


def fetch_wdi(out: Path) -> None:
    status, body = get(WDI_COUNTRIES)
    meta, countries = json.loads(body)
    rows = [{
        "iso3": c["id"], "iso2": c["iso2Code"], "name": c["name"],
        "region": c["region"]["value"], "income_group": c["incomeLevel"]["value"],
    } for c in countries]
    pd.DataFrame(rows).to_csv(out / "wdi_countries.csv", index=False)

    status, body = get(WDI_GDP)
    meta, data = json.loads(body)
    if meta.get("pages", 1) > 1:
        raise RuntimeError("WDI GDP response is paginated; raise per_page")
    gdp = pd.DataFrame([{
        "iso3": d["countryiso3code"], "year": int(d["date"]), "gdp_usd": d["value"],
    } for d in data if d["countryiso3code"]])
    gdp.dropna().to_csv(out / "wdi_gdp_usd.csv", index=False)


def fetch_billionaires(out: Path) -> None:
    status, body = get(BILLIONAIRES)
    if status != 200:
        raise RuntimeError(f"billionaires returned {status}")
    (out / "billionaires_piie.csv").write_bytes(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--only", choices=["wid", "pwt", "wdi", "billionaires"], nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    steps = {"billionaires": fetch_billionaires, "wdi": fetch_wdi, "pwt": fetch_pwt, "wid": fetch_wid}
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
