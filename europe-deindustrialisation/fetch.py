"""Download the Eurostat series for the analysis of Europe's deindustrialisation.

    python fetch.py --out data [--only name ...]

One request per dataset. National series are requested for the 27 member states and the
EU aggregate; regional series without a geography filter (every NUTS level comes back, and
the analysis keeps NUTS 2). Each request carries filter sets tried in order, narrowest first;
the first accepted one is kept. A tidy long table goes to ``data/<name>.csv``, and
``data/fetch_log.txt`` records URL, HTTP status, filter set and SHA-256 of every response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
AGENT = "europe-deindustrialisation/1.0"

EU27 = (
    "BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV",
    "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE",
)
NATIONAL = EU27 + ("EU27_2020",)

Filters = list[tuple[str, str]]

# name -> (dataset, filter sets, geographies or None for all)
REQUESTS: dict[str, tuple[str, list[Filters], tuple[str, ...] | None]] = {
    # national accounts by industry
    "gva_a64_current": (
        "nama_10_a64",
        [[("unit", "CP_MEUR"), ("na_item", "B1G"), ("nace_r2", "TOTAL"), ("nace_r2", "B-E"), ("nace_r2", "C"), ("nace_r2", "F")]],
        NATIONAL,
    ),
    "gva_a64_volume": (
        "nama_10_a64",
        [
            [("unit", "CLV20_MEUR"), ("na_item", "B1G"), ("nace_r2", "TOTAL"), ("nace_r2", "B-E"), ("nace_r2", "C"), ("nace_r2", "F")],
            [("unit", "CLV15_MEUR"), ("na_item", "B1G"), ("nace_r2", "TOTAL"), ("nace_r2", "B-E"), ("nace_r2", "C"), ("nace_r2", "F")],
        ],
        NATIONAL,
    ),
    "employment_a64": (
        "nama_10_a64_e",
        [[("unit", "THS_PER"), ("na_item", "EMP_DC"), ("nace_r2", "TOTAL"), ("nace_r2", "B-E"), ("nace_r2", "C"), ("nace_r2", "F")]],
        NATIONAL,
    ),
    "gfcf_by_industry": (
        "nama_10_a64_p5",
        [
            [("unit", "CLV20_MEUR"), ("asset10", "N11G"), ("nace_r2", "TOTAL"), ("nace_r2", "C")],
            [("unit", "CP_MEUR"), ("asset10", "N11G"), ("nace_r2", "TOTAL"), ("nace_r2", "C")],
        ],
        NATIONAL,
    ),
    "gdp_trade": (
        "nama_10_gdp",
        [[("unit", "CP_MEUR"), ("na_item", "B1GQ"), ("na_item", "P61"), ("na_item", "P71"), ("na_item", "P62"), ("na_item", "P72")]],
        NATIONAL,
    ),
    "niip": ("tipsii10", [[("unit", "PC_GDP")], []], NATIONAL),
    "current_account": ("tipsbp10", [[("unit", "PC_GDP")], []], NATIONAL),
    "industrial_production": (
        "sts_inpr_a",
        [
            [("nace_r2", "C"), ("unit", "I21"), ("s_adj", "CA")],
            [("nace_r2", "C"), ("unit", "I15"), ("s_adj", "CA")],
            [("nace_r2", "C"), ("unit", "I21")],
            [("nace_r2", "C")],
        ],
        NATIONAL,
    ),
    # energy
    "energy_dependency": ("nrg_ind_id", [[("unit", "PC")], []], NATIONAL),
    "gas_imports_by_partner": (
        "nrg_ti_gas",
        [[("siec", "G3000"), ("unit", "TJ_GCV")], [("siec", "G3000"), ("unit", "MIO_M3")], [("siec", "G3000")]],
        ("EU27_2020",),
    ),
    "gas_supply": ("nrg_cb_gas", [[("siec", "G3000"), ("unit", "TJ_GCV")], [("siec", "G3000")]], ("EU27_2020",)),
    "oil_imports_by_partner": ("nrg_ti_oil", [[("siec", "O4100_TOT"), ("unit", "THS_T")], [("siec", "O4100_TOT")]], ("EU27_2020",)),
    "coal_imports_by_partner": ("nrg_ti_sff", [[("siec", "C0100"), ("unit", "THS_T")], [("siec", "C0100")]], ("EU27_2020",)),
    # regions (NUTS 2 kept downstream)
    "regional_employment": (
        "nama_10r_3empers",
        [
            [("wstatus", "EMP"), ("unit", "THS"), ("nace_r2", "TOTAL"), ("nace_r2", "B-E"), ("nace_r2", "C")],
            [("wstatus", "EMP"), ("nace_r2", "TOTAL"), ("nace_r2", "B-E"), ("nace_r2", "C")],
        ],
        None,
    ),
    "regional_gdp_per_head": (
        "nama_10r_2gdp",
        [[("unit", "PPS_EU27_2020_HAB")], [("unit", "EUR_HAB")]],
        None,
    ),
    "regional_population": ("demo_r_d2jan", [[("sex", "T"), ("age", "TOTAL"), ("unit", "NR")]], None),
    "regional_unemployment": (
        "lfst_r_lfu3rt",
        [[("sex", "T"), ("age", "Y15-74"), ("isced11", "TOTAL"), ("unit", "PC")], [("sex", "T"), ("age", "Y15-74"), ("unit", "PC")]],
        None,
    ),
    "regional_employment_rate": (
        "lfst_r_lfe2emprt",
        [[("sex", "T"), ("age", "Y20-64"), ("unit", "PC")]],
        None,
    ),
}


def url_for(dataset: str, filters: Filters, geos: tuple[str, ...] | None) -> str:
    params = [("lang", "en"), ("format", "JSON")] + list(filters)
    if geos:
        params += [("geo", g) for g in geos]
    return f"{API}/{dataset}?{urllib.parse.urlencode(params)}"


def get(url: str, tries: int = 3, timeout: int = 300) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return e.code, e.read()
            last = e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(5 * (attempt + 1))
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
    for key, v in values.items():
        k = int(key)
        row = {d: codes[(k // stride) % size] for d, codes, stride, size in zip(ids, cats, strides, sizes)}
        row["value"] = v
        row["flag"] = status.get(key, "")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    failures = 0
    for name, (dataset, filter_sets, geos) in REQUESTS.items():
        if args.only and name not in args.only:
            continue
        ok = False
        for filters in filter_sets:
            url = url_for(dataset, filters, geos)
            try:
                status, body = get(url)
            except RuntimeError as e:
                log.append(f"{name}\tERROR\t{e}")
                continue
            log.append(f"{name}\t{status}\t{json.dumps(filters)}\t{hashlib.sha256(body).hexdigest()}\t{len(body)} bytes\t{url}")
            if status != 200:
                log.append(f"{name}\tBODY\t{body[:300]!r}")
                continue
            doc = json.loads(body)
            if not doc.get("value"):
                log.append(f"{name}\tEMPTY")
                continue
            df = jsonstat_to_long(doc)
            df.insert(0, "dataset", dataset)
            df["dataset_updated"] = doc.get("updated", "")
            df.to_csv(out / f"{name}.csv", index=False)
            log.append(f"{name}\tROWS\t{len(df)}")
            ok = True
            break
        if not ok:
            failures += 1
            log.append(f"{name}\tFAILED")
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
