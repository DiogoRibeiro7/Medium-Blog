"""Download the public series for the wealth-origin-sectors analysis.

    python fetch.py --out data

Sources
-------
- Forbes annual billionaire lists 2011-2026 through the Forbes person-list API, aggregated to
  citizenship x year x industry x self-made flag before anything is written (counts and summed
  net worth only; the first listed industry is used). ``data/forbes_coverage.csv`` records, per
  year, the list size and the share of people with an industry, with several industries, and
  with the self-made flag filled. Person-level rows are never saved.
- Penn World Table 11.0 (Feenstra, Inklaar and Timmer), through the DataverseNL API.
- World Bank WDI: GDP in current US dollars and the country list.

The workspace where the analysis is written cannot reach these hosts, so
``.github/workflows/wealth-origin-sectors-data.yml`` runs this script on GitHub's runners and
commits the tidy tables; ``data/fetch_log.txt`` records URL, status, size and SHA-256 of every
response.
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

UA = {"User-Agent": "wealth-origin-sectors/1.0 (research companion; dfr@esmad.ipp.pt)"}


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


FORBES = "https://www.forbes.com/forbesapi/person/billionaires/{year}/position/true.json?limit=5000"
FORBES_YEARS = range(2011, 2027)


def _persons(doc: dict) -> list[dict]:
    node = doc.get("personList", {})
    rows = node.get("personsLists") if isinstance(node, dict) else None
    if not isinstance(rows, list):
        raise ValueError("personList.personsLists not found")
    return [r for r in rows if isinstance(r, dict)]


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in {"true", "false"}:
        return v.lower() == "true"
    return None


def _industry(p: dict) -> tuple[str | None, int]:
    v = p.get("industries")
    if isinstance(v, list):
        items = [x.strip() for x in v if isinstance(x, str) and x.strip()]
        return (items[0] if items else None), len(items)
    if isinstance(v, str) and v.strip():
        return v.strip(), 1
    return None, 0


def fetch_forbes(out: Path) -> None:
    """Forbes annual lists aggregated to citizenship x year x industry x self-made flag in memory.

    Only counts and summed net worth are written. The first listed industry is used; the
    coverage report records how many people have none, one or several industries.
    """
    agg_rows, cov_rows = [], []
    for year in FORBES_YEARS:
        url = FORBES.format(year=year)
        try:
            status, body = get(url)
        except RuntimeError as e:
            cov_rows.append({"year": year, "status": f"failed: {e}"})
            continue
        if status != 200:
            cov_rows.append({"year": year, "status": status})
            time.sleep(3)
            continue
        try:
            persons = _persons(json.loads(body))
        except (ValueError, json.JSONDecodeError) as e:
            cov_rows.append({"year": year, "status": f"unparsed: {e}"})
            continue
        ind = [_industry(p) for p in persons]
        df = pd.DataFrame({
            "citizenship": [p.get("countryOfCitizenship") for p in persons],
            "worth_m": [p.get("finalWorth") if isinstance(p.get("finalWorth"), (int, float)) else None for p in persons],
            "self_made": [_bool(p.get("selfMade")) for p in persons],
            "industry": [i[0] for i in ind],
            "n_industries": [i[1] for i in ind],
        })
        n = len(df)
        cov_rows.append({
            "year": year, "status": status, "n_persons": n,
            "share_industry": float((df["n_industries"] > 0).mean()) if n else 0.0,
            "share_multi_industry": float((df["n_industries"] > 1).mean()) if n else 0.0,
            "share_selfmade_flag": float(df["self_made"].notna().mean()) if n else 0.0,
            "n_distinct_industries": int(df["industry"].nunique()),
        })
        df = df.dropna(subset=["citizenship", "worth_m"])
        df["citizenship"] = df["citizenship"].str.strip()
        df["industry"] = df["industry"].fillna("(none)")
        df["origin"] = df["self_made"].map({True: "self_made", False: "inherited"}).fillna("missing")
        g = df.groupby(["citizenship", "industry", "origin"]).agg(n=("worth_m", "size"), worth_m=("worth_m", "sum"))
        for (c, i, o), r in g.iterrows():
            agg_rows.append({"year": year, "citizenship": c, "industry": i, "origin": o,
                             "n": int(r["n"]), "worth_m": float(r["worth_m"])})
        time.sleep(3)
    pd.DataFrame(cov_rows).to_csv(out / "forbes_coverage.csv", index=False)
    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(out / "forbes_country_industry_year.csv", index=False, float_format="%.3f")
    if not any(r.get("status") == 200 for r in cov_rows):
        raise RuntimeError("no Forbes year could be fetched")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--only", choices=["pwt", "wdi", "forbes"], nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    steps = {"forbes": fetch_forbes, "wdi": fetch_wdi, "pwt": fetch_pwt}
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
