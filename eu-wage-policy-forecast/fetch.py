"""Download the Eurostat and OECD series for the eu-wage-policy-forecast analysis.

    python fetch.py --out data

Eurostat datasets come from the JSON-stat dissemination API, one request per dataset for
all member states at once. OECD datasets come from the SDMX API of the OECD Data Explorer
(CSV); if that fails, the same dataflow is read from DBnomics, which mirrors it.

Each request carries a list of filter sets (Eurostat) or endpoints (OECD) tried in order;
the first accepted one is kept. Tidy long tables go to ``data/<name>.csv`` and
``data/fetch_log.txt`` records URL, HTTP status and SHA-256 of every response. Optional
datasets may fail without failing the run.
"""

from __future__ import annotations

import argparse
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

EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
EUROSTAT_TOC = "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en"
OECD = "https://sdmx.oecd.org/public/rest/data"
DBNOMICS = "https://api.db.nomics.world/v22/series/OECD"
UA = {"User-Agent": "eu-wage-policy-forecast/1.0 (research companion)"}

EU27 = (
    "BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV",
    "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE",
)
GEOS = EU27 + ("EU27_2020",)

Filters = list[tuple[str, str]]

# name -> (dataset, filter sets tried in order, required)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[Filters], bool]] = {
    "minimum_wage_eur": ("earn_mw_cur", [[("currency", "EUR")]], True),
    "minimum_wage_ratio": ("earn_mw_avgr2", [[]], True),
    "national_accounts_cp": (
        "nama_10_gdp",
        [[("unit", "CP_MEUR"), ("na_item", "D1"), ("na_item", "D11")]],
        True,
    ),
    "employment_persons": (
        "nama_10_a10_e",
        [[("unit", "THS_PER"), ("nace_r2", "TOTAL"), ("na_item", "EMP_DC"), ("na_item", "SAL_DC")]],
        True,
    ),
    # Structure of Earnings Survey: monthly earnings by decile, one table per wave
    "ses_deciles_2006": ("earn_ses_mdeci", [[]], True),
    "ses_deciles_2010": ("earn_ses10_mdeci", [[]], True),
    "ses_deciles_2014": ("earn_ses14_mdeci", [[]], True),
    "ses_deciles_2018": ("earn_ses18_mdeci", [[]], True),
    "ses_deciles_2022": ("earn_ses22_mdeci", [[]], True),
    # Business statistics by size class: employment in enterprises by number of persons employed.
    # The pay-transparency duties start at 250, 150 and 50 (Portuguese draft) employees.
    "sbs_size_class": (
        "sbs_sc_ovw",
        [[("nace_r2", "B-S_X_O_S94"), ("lastTimePeriod", "3")],
         [("nace_r2", "B-N_S95_X_K"), ("lastTimePeriod", "3")],
         [("lastTimePeriod", "2")]],
        False,
    ),
    "sbs_size_class_old": (
        "sbs_sc_sca_r2",
        [[("nace_r2", "B-N_S95_X_K"), ("lastTimePeriod", "3")], [("lastTimePeriod", "2")]],
        False,
    ),
    # Unadjusted gender pay gap (hourly earnings, SES-based, annual)
    "gender_pay_gap": ("earn_gr_gpgr2", [[("nace_r2", "B-S_X_O")], []], False),
}

# name -> (agency, dataflow, required)
OECD_REQUESTS: dict[str, tuple[str, str, bool]] = {
    "oecd_min_to_average": ("OECD.ELS.SAE", "DSD_EARNINGS@MIN2AVE", True),
}

# EU member states in ISO3 (the OECD code), plus two outside comparators
OECD_AREAS = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL",
    "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR", "USA",
}


def get(url: str, tries: int = 4, headers: dict | None = None) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=240) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                last = e
            elif 400 <= e.code < 500:
                return e.code, e.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def eurostat_url(dataset: str, filters: Filters) -> str:
    params = [("lang", "en"), ("format", "JSON")] + list(filters) + [("geo", g) for g in GEOS]
    return f"{EUROSTAT}/{dataset}?{urllib.parse.urlencode(params)}"


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
    for key, v in values.items():
        k = int(key)
        row = {d: codes[(k // stride) % size] for d, codes, stride, size in zip(ids, cats, strides, sizes)}
        row["value"] = v
        row["flag"] = status.get(key, "")
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_eurostat(out: Path, log: list[str], only: list[str] | None) -> int:
    failures = 0
    for name, (dataset, filter_sets, required) in EUROSTAT_REQUESTS.items():
        if only and name not in only:
            continue
        ok = False
        for filters in filter_sets:
            url = eurostat_url(dataset, filters)
            try:
                status, body = get(url)
            except RuntimeError as e:
                log.append(f"{name}\tERROR\t{e}")
                continue
            log.append(f"{name}\t{status}\t{json.dumps(filters)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
            if status != 200:
                continue
            doc = json.loads(body)
            if not doc.get("value"):
                log.append(f"{name}\tEMPTY")
                continue
            df = jsonstat_to_long(doc)
            df.insert(0, "dataset", dataset)
            df["dataset_updated"] = doc.get("updated", "")
            df.to_csv(out / f"{name}.csv", index=False)
            ok = True
            break
        if not ok:
            log.append(f"{name}\tFAILED\t{'required' if required else 'optional'}")
            failures += int(required)
    return failures


def dbnomics_series(dataflow: str, log: list[str], name: str) -> pd.DataFrame | None:
    """Read every series of an OECD dataflow mirrored by DBnomics, page by page."""
    rows = []
    offset, limit = 0, 1000
    while True:
        url = f"{DBNOMICS}/{urllib.parse.quote(dataflow)}?observations=1&format=json&limit={limit}&offset={offset}"
        try:
            status, body = get(url)
        except RuntimeError as e:
            log.append(f"{name}\tDBNOMICS_ERROR\t{e}")
            return None
        log.append(f"{name}\tdbnomics\t{status}\t{hashlib.sha256(body).hexdigest()}\t{url}")
        if status != 200:
            return None
        doc = json.loads(body)
        docs = doc.get("series", {}).get("docs", [])
        for s in docs:
            dims = s.get("dimensions", {})
            for period, value in zip(s.get("period", []), s.get("value", [])):
                if value in (None, "NA"):
                    continue
                rows.append({**{k.upper(): v for k, v in dims.items()}, "TIME_PERIOD": period, "OBS_VALUE": value,
                             "series_code": s.get("series_code", "")})
        total = doc.get("series", {}).get("num_found", 0)
        offset += limit
        if offset >= total or not docs:
            break
    return pd.DataFrame(rows) if rows else None


def fetch_oecd(out: Path, log: list[str], only: list[str] | None) -> int:
    failures = 0
    for name, (agency, dataflow, required) in OECD_REQUESTS.items():
        if only and name not in only:
            continue
        ok = False
        for version in ("1.0", ""):
            url = f"{OECD}/{agency},{dataflow},{version}/all?dimensionAtObservation=AllDimensions&format=csvfile"
            try:
                status, body = get(url)
            except RuntimeError as e:
                log.append(f"{name}\tERROR\t{e}")
                continue
            log.append(f"{name}\t{status}\t{hashlib.sha256(body).hexdigest()}\t{url}")
            if status != 200 or len(body) < 200:
                continue
            df = pd.read_csv(io.BytesIO(body), low_memory=False)
            df = df[df["REF_AREA"].isin(OECD_AREAS)]
            df.insert(0, "source", f"OECD {agency} {dataflow}")
            df.to_csv(out / f"{name}.csv", index=False)
            ok = True
            break
        if not ok:
            df = dbnomics_series(dataflow, log, name)
            if df is not None:
                df = df[df["REF_AREA"].isin(OECD_AREAS)] if "REF_AREA" in df else df
                df.insert(0, "source", f"DBnomics mirror of OECD {dataflow}")
                df.to_csv(out / f"{name}.csv", index=False)
                ok = True
        if not ok:
            log.append(f"{name}\tFAILED\t{'required' if required else 'optional'}")
            failures += int(required)
        time.sleep(6)  # the OECD API rate-limits bursts of requests
    return failures


def fetch_toc(out: Path, log: list[str]) -> None:
    """Keep the Eurostat catalogue lines on earnings, minimum wages and size classes, to document dataset choices."""
    try:
        status, body = get(EUROSTAT_TOC)
    except RuntimeError as e:
        log.append(f"eurostat_toc\tERROR\t{e}")
        return
    log.append(f"eurostat_toc\t{status}\t{hashlib.sha256(body).hexdigest()}\t{EUROSTAT_TOC}")
    if status != 200:
        return
    keep = [ln for ln in body.decode("utf-8", "replace").splitlines()
            if any(k in ln.lower() for k in ("minimum wage", "earnings", "pay gap", "size class"))]
    (out / "eurostat_toc_earnings.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    failures = fetch_eurostat(out, log, args.only) + fetch_oecd(out, log, args.only)
    if not args.only:
        fetch_toc(out, log)
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
