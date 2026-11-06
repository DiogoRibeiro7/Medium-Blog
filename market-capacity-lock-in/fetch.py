"""Download the Eurostat series that are not already pinned in ``data/archived``.

    python fetch.py --out data

Each request asks the Eurostat statistics API (JSON-stat 2.0) for one dataset and one
country. Because some dimension codes differ between datasets, every request carries a
list of filter sets, tried in order from the narrowest to the widest; the first one the
API accepts is kept. The raw response goes to ``data/raw/<name>_<geo>.json`` unchanged,
a tidy long table to ``data/<name>.csv``, and ``data/fetch_log.txt`` records the URL,
HTTP status, filter set used and SHA-256 of every response.

Datasets:

- ``ef_m_farmang``  farm managers by age (Farm Structure Survey / Census of Agriculture)
- ``aact_ali01``    agricultural labour input, thousand annual work units
- ``aact_eaa05``    economic accounts for agriculture, volume and price indices
- ``sts_copr_a``    production in construction, annual
- ``ilc_di12``      Gini coefficient of equivalised disposable income
- ``ilc_di03``      mean and median equivalised net income
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

FARM_GEOS = ("PT", "EU27_2020", "ES", "IT", "EL", "FR", "PL", "AT")

# name -> (dataset, geos, list of filter sets tried in order)
REQUESTS: dict[str, tuple[str, tuple[str, ...], list[dict[str, str]]]] = {
    "farm_managers_age": (
        "ef_m_farmang",
        FARM_GEOS,
        [
            {"sex": "T", "so_eur": "TOTAL", "agrarea": "TOTAL", "farmtype": "TOTAL"},
            {"sex": "T", "so_eur": "TOTAL", "agrarea": "TOTAL"},
            {"sex": "T"},
            {},
        ],
    ),
    "agri_labour_input": ("aact_ali01", ("PT", "EU27_2020", "ES"), [{}]),
    "agri_accounts_indices": (
        "aact_eaa05",
        ("PT",),
        [{"itm_newa": "10000"}, {}],
    ),
    "construction_production": (
        "sts_copr_a",
        ("PT",),
        [
            {"indic_bt": "PRD", "nace_r2": "F", "s_adj": "CA", "unit": "I21"},
            {"indic_bt": "PRD", "nace_r2": "F", "unit": "I21"},
            {"indic_bt": "PRD", "nace_r2": "F"},
            {"nace_r2": "F"},
            {},
        ],
    ),
    "gini": ("ilc_di12", ("PT",), [{}]),
    "income": (
        "ilc_di03",
        ("PT",),
        [{"age": "TOTAL", "sex": "T", "indic_il": "MED_E"}, {"age": "TOTAL", "sex": "T"}, {}],
    ),
}


def url_for(dataset: str, geo: str, filters: dict[str, str]) -> str:
    params = [("geo", geo), ("lang", "en"), ("format", "JSON")]
    params += sorted(filters.items())
    return f"{API}/{dataset}?{urllib.parse.urlencode(params)}"


def get(url: str, tries: int = 3) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "market-capacity-lock-in/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            # 4xx: the filter set is not valid for this dataset, do not retry it
            if 400 <= e.code < 500:
                return e.code, e.read()
            last = e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def jsonstat_to_long(doc: dict) -> pd.DataFrame:
    """Flatten a JSON-stat 2.0 dataset into one row per cell."""
    ids: list[str] = doc["id"]
    sizes: list[int] = doc["size"]
    cats = []
    for d in ids:
        index = doc["dimension"][d]["category"]["index"]
        if isinstance(index, list):
            codes = index
        else:
            codes = [c for c, _ in sorted(index.items(), key=lambda kv: kv[1])]
        cats.append(codes)
    values = doc.get("value", {})
    status = doc.get("status", {})
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    if isinstance(status, list):
        status = {str(i): s for i, s in enumerate(status) if s}
    rows = []
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    for key, v in values.items():
        k = int(key)
        row = {}
        for d, codes, stride, size in zip(ids, cats, strides, sizes):
            row[d] = codes[(k // stride) % size]
        row["value"] = v
        row["flag"] = status.get(key, "")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", help="request names to run (default: all)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    log_lines = [f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    failures = 0
    for name, (dataset, geos, filter_sets) in REQUESTS.items():
        if args.only and name not in args.only:
            continue
        frames = []
        for geo in geos:
            ok = False
            for filters in filter_sets:
                url = url_for(dataset, geo, filters)
                try:
                    status, body = get(url)
                except RuntimeError as e:
                    log_lines.append(f"{name}\t{geo}\tERROR\t{e}")
                    continue
                digest = hashlib.sha256(body).hexdigest()
                log_lines.append(f"{name}\t{geo}\t{status}\t{json.dumps(filters, sort_keys=True)}\t{digest}\t{url}")
                if status != 200:
                    continue
                doc = json.loads(body)
                if not doc.get("value"):
                    log_lines.append(f"{name}\t{geo}\tEMPTY\t{json.dumps(filters, sort_keys=True)}")
                    continue
                (out / "raw" / f"{name}_{geo}.json").write_bytes(body)
                df = jsonstat_to_long(doc)
                df.insert(0, "dataset", dataset)
                df["dataset_updated"] = doc.get("updated", "")
                frames.append(df)
                ok = True
                break
            if not ok:
                failures += 1
                log_lines.append(f"{name}\t{geo}\tFAILED\tno filter set accepted")
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    (out / "fetch_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
