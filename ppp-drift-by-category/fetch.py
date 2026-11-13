"""Download the Eurostat series for the category-level test of the PPP drift inside Europe.

    python fetch.py --out data

Eurostat (JSON-stat dissemination API):

* ``prc_ppp_ind``: purchasing power parities (EU27 = 1), price level indices (EU27 = 100), nominal
  expenditure in national currency and real expenditure in PPS, for household consumption, its twelve
  COICOP divisions, selected groups, and consumer goods (non-durable, semi-durable, durable) and
  consumer services, every European economy in the Eurostat-OECD programme plus the US and Japan.
* ``nama_10_co3_p3``: household final consumption expenditure by purpose (COICOP), implicit deflators
  (2020 = 100, national currency and euro) and current prices, for the same divisions and groups.
* ``nama_10_fcs``: household final consumption expenditure by durability, current prices and
  chain-linked volumes, for the goods and services split.

One request per category keeps each response under the API's synchronous size limit. Raw responses
are kept gzipped in ``data/raw/``, tidy tables in ``data/*.csv``, and ``data/fetch_log.txt`` records
URL, HTTP status, size and SHA-256 of every response.
"""

from __future__ import annotations

import argparse
import gzip
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

UA = {"User-Agent": "ppp-drift-by-category/1.0 (research companion)"}
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT",
        "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
OTHER_EUROPE = ["CH", "NO", "IS", "UK", "TR", "RS", "MK", "AL", "ME", "BA"]
GEOS_PPP = EU27 + OTHER_EUROPE + ["US", "JP", "EU27_2020"]
GEOS_NA = EU27 + ["CH", "NO", "IS", "RS", "AL", "BA", "EU27_2020"]

# PPP analytical category -> COICOP code in the national accounts
CATEGORIES = {
    "E011": "TOTAL",
    "A0101": "CP01", "A010101": "CP011", "A010102": "CP012",
    "A0102": "CP02", "A010201": "CP021", "A010202": "CP022",
    "A0103": "CP03", "A010301": "CP031", "A010302": "CP032",
    "A0104": "CP04", "A010405": "CP045",
    "A0105": "CP05", "A010501": "CP051", "A010503": "CP053",
    "A0106": "CP06",
    "A0107": "CP07", "A010701": "CP071", "A010703": "CP073",
    "A0108": "CP08",
    "A0109": "CP09", "A010901": "CP091",
    "A0110": "CP10",
    "A0111": "CP11",
    "A0112": "CP12",
}
GOODS_SERVICES = ["P0101", "P010101", "P010102", "P010103", "P0201"]
EXTRA_COICOP = ["CP041", "CP042", "CP111", "CP112", "CP121"]
FCS_ITEMS = ["P31_S14", "P31_S14_DC", "P311_S14", "P312_S14", "P313_S14", "P314_S14"]


def multi(name: str, values: list[str]) -> list[tuple[str, str]]:
    return [(name, v) for v in values]


def _requests() -> dict[str, tuple[str, list[tuple[str, str]]]]:
    req: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for cat in list(CATEGORIES) + GOODS_SERVICES:
        req[f"ppp_{cat}"] = ("prc_ppp_ind", [("ppp_cat", cat)] + multi("geo", GEOS_PPP)
                             + multi("na_item", ["PPP_EU27_2020", "PLI_EU27_2020", "EXP_NAC", "EXP_PPS_EU27_2020"]))
    for coicop in list(CATEGORIES.values()) + EXTRA_COICOP:
        req[f"defl_{coicop}"] = ("nama_10_co3_p3", [("coicop", coicop), ("sinceTimePeriod", "1995")]
                                 + multi("geo", GEOS_NA) + multi("unit", ["PD20_NAC", "PD20_EUR", "CP_MNAC"]))
    for item in FCS_ITEMS:
        req[f"fcs_{item}"] = ("nama_10_fcs", [("na_item", item), ("sinceTimePeriod", "1995")] + multi("geo", GEOS_NA)
                              + multi("unit", ["CP_MNAC", "CLV20_MNAC", "CP_MEUR", "CLV20_MEUR"]))
    return req


EUROSTAT_REQUESTS = _requests()
TABLES = {
    "ppp": [n for n in EUROSTAT_REQUESTS if n.startswith("ppp_")],
    "deflators": [n for n in EUROSTAT_REQUESTS if n.startswith("defl_")],
    "durability": [n for n in EUROSTAT_REQUESTS if n.startswith("fcs_")],
}


def get(url: str, tries: int = 5) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
            if b'"ASYNCHRONOUS_RESPONSE' in body[:400]:
                last = RuntimeError("asynchronous response")
                time.sleep(30 * (attempt + 1))
                continue
            return 200, body
        except urllib.error.HTTPError as e:
            if e.code in (413, 429) or e.code >= 500:
                last = e
            else:
                return e.code, e.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(20 * (attempt + 1))
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


def record(out: Path, log: list[str], name: str, url: str, status: int, body: bytes) -> bool:
    log.append(f"{name}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
    if status != 200 or len(body) < 200:
        return False
    (out / "raw" / f"{name}.json.gz").write_bytes(gzip.compress(body, mtime=0))
    return True


def tidy(out: Path) -> None:
    """Rebuild the CSV tables from the raw responses (no network)."""
    for table, names in TABLES.items():
        frames = []
        for name in names:
            path = out / "raw" / f"{name}.json.gz"
            if not path.exists():
                continue
            doc = json.loads(gzip.decompress(path.read_bytes()))
            if "value" not in doc:
                continue
            df = jsonstat_to_long(doc)
            df.insert(0, "dataset", EUROSTAT_REQUESTS[name][0])
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df = df.drop(columns=[c for c in ("flag",) if c in df.columns and (df[c].fillna("") == "").all()])
        df.to_csv(out / f"{table}.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--tidy-only", action="store_true", help="rebuild the CSV files from data/raw")
    args = ap.parse_args()
    out: Path = args.out
    (out / "raw").mkdir(parents=True, exist_ok=True)
    if args.tidy_only:
        tidy(out)
        return 0
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
        if not record(out, log, name, url, status, body):
            failures += 1
        print(log[-1][:110], flush=True)
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    tidy(out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
