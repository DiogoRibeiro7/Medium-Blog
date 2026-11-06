"""Download the Eurostat industry series for the recovery-after-busts analysis.

    python fetch.py --out data

One request per dataset, for all member states at once where the API allows it.
Each request carries a list of filter sets tried in order (narrowest first); the
first accepted one is kept. Raw responses go to ``data/raw/<name>.json``, a tidy long
table to ``data/<name>.csv``, and ``data/fetch_log.txt`` records URL, HTTP status,
filter set and SHA-256 of every response.
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

EU27 = (
    "BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV",
    "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE",
)
GEOS = EU27 + ("EU27_2020",)

Filters = list[tuple[str, str]]

# name -> (dataset, list of filter sets); every filter set gets the geo list appended
REQUESTS: dict[str, tuple[str, list[Filters]]] = {
    "employment_a64": (
        "nama_10_a64_e",
        [[("unit", "THS_PER"), ("na_item", "EMP_DC")], [("unit", "THS_PER")]],
    ),
    "gva_a64_volume": (
        "nama_10_a64",
        [[("unit", "CLV20_MEUR"), ("na_item", "B1G")], [("unit", "CLV15_MEUR"), ("na_item", "B1G")], [("na_item", "B1G")]],
    ),
    "gdp_volume": (
        "nama_10_gdp",
        [[("unit", "CLV20_MEUR"), ("na_item", "B1GQ")], [("unit", "CLV15_MEUR"), ("na_item", "B1GQ")]],
    ),
    "house_prices": ("prc_hpi_a", [[("purchase", "TOTAL"), ("unit", "I15_A_AVG")], [("unit", "I15_A_AVG")]]),
}


def url_for(dataset: str, filters: Filters) -> str:
    params = [("lang", "en"), ("format", "JSON")] + list(filters) + [("geo", g) for g in GEOS]
    return f"{API}/{dataset}?{urllib.parse.urlencode(params)}"


def get(url: str, tries: int = 3) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "recovery-after-busts/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return e.code, e.read()
            last = e
        except (urllib.error.URLError, TimeoutError) as e:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    log = [f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    failures = 0
    for name, (dataset, filter_sets) in REQUESTS.items():
        if args.only and name not in args.only:
            continue
        ok = False
        for filters in filter_sets:
            url = url_for(dataset, filters)
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
            (out / "raw" / f"{name}.json").write_bytes(body)
            df = jsonstat_to_long(doc)
            df.insert(0, "dataset", dataset)
            df["dataset_updated"] = doc.get("updated", "")
            df.to_csv(out / f"{name}.csv", index=False)
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
