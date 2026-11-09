"""Download the public series used to measure Portugal's housebuilding sector, layer by layer.

    python fetch.py --out data

Two sources, both open and without tokens:

- Eurostat statistics API (JSON-stat 2.0). Each request carries a list of filter sets tried in
  order; the first one the API accepts is kept.
- INE (Statistics Portugal) indicator API, ``pindica.jsp?op=2``, one indicator per request, all
  periods (``Dim1=T``), Portugal only (``Dim2=PT``) and, where the indicator has an activity
  dimension, only the CAE codes listed below.

Raw responses go to ``data/raw/`` unchanged, tidy long tables to ``data/<name>.csv``, and
``data/fetch_log.txt`` records the URL, HTTP status, filters and SHA-256 of every response.

Eurostat datasets:

- ``nama_10_a64_e``  employment in construction, national accounts (persons, employees, self-employed)
- ``nama_10_a64``    compensation of employees, construction and all industries
- ``lfsa_egan2``     employed persons by age, construction (section F) and all industries (2008-)
- ``lfsa_egana2d``   employed persons by age, NACE Rev. 1.1 division 45 (1992-2008)
- ``lfsa_eisn2``     employed persons in construction by occupation (ISCO-08)
- ``edat_lfs_9910``  employees in construction by educational attainment
- ``educ_uoe_grad02`` graduates in architecture and construction (ISCED-F 073) by level
- ``lc_lci_r2_a``    labour cost index, construction and the business economy
- ``sbs_sc_con_r2``  construction enterprises by employment size class, NACE classes (2008-2020)
- ``sbs_sc_ovw``     the same from 2021 onwards (new SBS regulation)
- ``lfsa_egana``     employed persons by age, all industries (1992-2008), the benchmark before 2008

INE indicators:

- ``0008466`` / ``0013838``  enterprises by CAE Rev. 3 subclass (2008-2023 / 2023-2024)
- ``0008467`` / ``0013839``  persons employed by subclass
- ``0008475`` / ``0013847``  production (EUR) by subclass
- ``0013236``  dwellings licensed in new construction for family housing, by promoter (2002-)
- ``0013249``  dwellings completed in new construction for family housing, by promoter (2011-)
- ``0013235``  dwellings licensed in new construction, by typology (total used)
- ``0013244``  apartment buildings licensed in new construction for family housing
- ``0013245``  single-family houses (moradias) licensed in new construction
- ``0011748``  construction cost index for new housing, by factor (labour, materials), monthly
- ``0012787``  housing transactions (number), by category (new, existing), annual (2009-)
- ``0012778``  dwellings completed in new construction for family housing, quarterly (2004-)
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

EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
INE = "https://www.ine.pt/ine/json_indicador/pindica.jsp"
UA = {"User-Agent": "portugal-building-ecosystem/1.0"}

# CAE Rev. 3 subclasses kept from the INE business statistics (SCIE).
CAE_CODES = (
    # developers and building contractors
    "41100", "41200",
    # specialised trades
    "43110", "43120", "43210", "43221", "43222", "43290", "43310", "43320", "43330", "43340",
    "43390", "43910", "43991", "43992",
    # real estate: buying and selling own property, renting, agencies, finders, valuers
    "68100", "68200", "68311", "68312", "68313",
    # architects and engineers
    "71110", "71120",
    # aggregates, kept to check that the subclasses add up
    "41", "43", "68", "F", "L",
)

# name -> (dataset, list of filter sets tried in order)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[dict[str, str]]]] = {
    "na_employment": (
        "nama_10_a64_e",
        [{"nace_r2": "F", "unit": "THS_PER"}, {"nace_r2": "F"}],
    ),
    "na_compensation": (
        "nama_10_a64",
        [{"nace_r2": "F,TOTAL", "unit": "CP_MEUR", "na_item": "D1"}, {"nace_r2": "F,TOTAL", "na_item": "D1"}],
    ),
    "na_employees_total": (
        "nama_10_a64_e",
        [{"nace_r2": "TOTAL", "unit": "THS_PER", "na_item": "SAL_DC"}],
    ),
    "lfs_age_nace2": (
        "lfsa_egan2",
        [{"nace_r2": "F,TOTAL", "sex": "T", "unit": "THS_PER"}, {"nace_r2": "F,TOTAL", "sex": "T"}],
    ),
    "lfs_age_nace1": (
        "lfsa_egana2d",
        [{"nace_r1": "F45", "sex": "T", "unit": "THS_PER"}, {"nace_r1": "F45", "sex": "T"}],
    ),
    "lfs_occupation": (
        "lfsa_eisn2",
        [{"nace_r2": "F", "sex": "T", "age": "Y_GE15", "unit": "THS_PER"}, {"nace_r2": "F", "sex": "T"}],
    ),
    "lfs_education": (
        "edat_lfs_9910",
        [{"nace_r2": "F,TOTAL", "sex": "T", "age": "Y15-64"}, {"nace_r2": "F", "sex": "T"}],
    ),
    "graduates_construction": (
        "educ_uoe_grad02",
        [{"iscedf13": "F073", "sex": "T", "unit": "NR"}, {"iscedf13": "F073", "sex": "T"}],
    ),
    "labour_cost_index": (
        "lc_lci_r2_a",
        [{"nace_r2": "F,B-S", "unit": "I20", "lcstruct": "D1_D4_MD5"}, {"nace_r2": "F,B-S", "unit": "I20"}],
    ),
    "sbs_size_construction": (
        "sbs_sc_con_r2",
        [{"nace_r2": "F411,F412,F43", "indic_sb": "V11110,V16110"}, {"nace_r2": "F411,F412,F43"}],
    ),
    "sbs_size_construction_2021": (
        "sbs_sc_ovw",
        [{"nace_r2": "F411,F412,F43", "indic_sbs": "ENT_NR,EMP_NR"}, {"nace_r2": "F412,F43", "indic_sbs": "ENT_NR,EMP_NR"}],
    ),
    "lfs_age_total_nace1": (
        "lfsa_egana",
        [{"nace_r1": "TOTAL", "sex": "T", "unit": "THS_PER"}, {"nace_r1": "TOTAL", "sex": "T"}],
    ),
}

# name -> (indicator, has an activity dimension)
INE_REQUESTS: dict[str, tuple[str, bool]] = {
    "ine_firms_2008": ("0008466", True),
    "ine_firms_2023": ("0013838", True),
    "ine_persons_2008": ("0008467", True),
    "ine_persons_2023": ("0013839", True),
    "ine_production_2008": ("0008475", True),
    "ine_production_2023": ("0013847", True),
    "ine_licensed_by_promoter": ("0013236", False),
    "ine_completed_by_promoter": ("0013249", False),
    "ine_licensed_dwellings": ("0013235", False),
    "ine_licensed_apartment_buildings": ("0013244", False),
    "ine_licensed_houses": ("0013245", False),
    "ine_cost_index": ("0011748", False),
    "ine_transactions": ("0012787", False),
    "ine_completed_quarterly": ("0012778", False),
}


def eurostat_url(dataset: str, filters: dict[str, str]) -> str:
    params = [("geo", "PT"), ("lang", "en"), ("format", "JSON")]
    for key, value in sorted(filters.items()):
        for v in value.split(","):
            params.append((key, v))
    return f"{EUROSTAT}/{dataset}?{urllib.parse.urlencode(params)}"


def ine_url(indicator: str, activity: bool) -> str:
    params = [("op", "2"), ("varcd", indicator), ("Dim1", "T"), ("Dim2", "PT"), ("lang", "PT")]
    if activity:
        params.append(("Dim3", ",".join(CAE_CODES)))
    return f"{INE}?{urllib.parse.urlencode(params, safe=',')}"


def get(url: str, tries: int = 4) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return e.code, e.read()
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(3 * (attempt + 1))
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


def ine_to_long(doc: list) -> pd.DataFrame:
    """Flatten an INE ``op=2`` response into one row per period and category."""
    rows = []
    for period, cells in doc[0].get("Dados", {}).items():
        for c in cells:
            row = {"period": period, "geo": c.get("geocod", "")}
            for k, v in c.items():
                if k.startswith("dim_") and not k.endswith("_t"):
                    row[k] = v
                    row[k + "_label"] = c.get(k + "_t", "")
            raw = c.get("valor")
            row["value"] = float(raw) if raw not in (None, "", "x", "-") else None
            row["flag"] = c.get("sinal_conv", "")
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", help="request names to run (default: all)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    log = [f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    failures = 0

    for name, (dataset, filter_sets) in EUROSTAT_REQUESTS.items():
        if args.only and name not in args.only:
            continue
        ok = False
        for filters in filter_sets:
            url = eurostat_url(dataset, filters)
            try:
                status, body = get(url)
            except RuntimeError as e:
                log.append(f"{name}\tERROR\t{e}")
                continue
            log.append(f"{name}\t{status}\t{json.dumps(filters, sort_keys=True)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
            if status != 200:
                continue
            doc = json.loads(body)
            if not doc.get("value"):
                log.append(f"{name}\tEMPTY\t{json.dumps(filters, sort_keys=True)}")
                continue
            (out / "raw" / f"{name}.json").write_bytes(body)
            df = jsonstat_to_long(doc)
            df.insert(0, "dataset", dataset)
            df["dataset_updated"] = doc.get("updated", "")
            df.to_csv(out / f"{name}.csv", index=False)
            ok = True
            break
        if not ok:
            failures += 1
            log.append(f"{name}\tFAILED\tno filter set accepted")

    for name, (indicator, activity) in INE_REQUESTS.items():
        if args.only and name not in args.only:
            continue
        url = ine_url(indicator, activity)
        try:
            status, body = get(url)
        except RuntimeError as e:
            failures += 1
            log.append(f"{name}\tERROR\t{e}")
            continue
        log.append(f"{name}\t{status}\t{indicator}\t{hashlib.sha256(body).hexdigest()}\t{url}")
        doc = json.loads(body) if status == 200 else None
        if not doc or "Verdadeiro" not in doc[0].get("Sucesso", {}):
            failures += 1
            log.append(f"{name}\tFAILED\t{indicator}")
            continue
        (out / "raw" / f"{name}.json").write_bytes(body)
        df = ine_to_long(doc)
        df.insert(0, "indicator", indicator)
        df["indicator_label"] = doc[0].get("IndicadorDsg", "")
        df["last_update"] = doc[0].get("DataUltimoAtualizacao", "")
        df.to_csv(out / f"{name}.csv", index=False)

    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
