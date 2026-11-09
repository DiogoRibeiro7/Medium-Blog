"""Download the public series behind the German public-investment analysis.

    python fetch.py --out data [--only name ...]

Eurostat (JSON-stat API), annual unless stated:

- gov_10a_main: gross fixed capital formation (P.51G), consumption of fixed capital (P.51C),
  investment grants paid (D.92),
  interest payable (D.41), net lending (B.9), total expenditure and revenue, for general
  government and its subsectors (central, state, local, social security);
- gov_10dd_edpt1: Maastricht debt; nama_10_gdp: GDP at current prices and chain-linked volumes;
- nama_10_pe: population;
- nama_10_nfa_bs: net fixed assets of general government (balance sheets for non-financial assets);
- gov_10a_exp: gross fixed capital formation of general government by function (COFOG);
- gov_10q_ggnfa and namq_10_gdp: quarterly investment, interest, net lending and GDP from 2015.

Each request carries filter sets tried in order, narrowest first. Every response is logged with
URL, HTTP status and SHA-256 in ``data/fetch_log.txt``.
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
AGENT = "german-public-investment/1.0 (research; python urllib)"

GEOS = ("DE", "NL", "AT", "FI", "DK", "SE", "BE", "FR", "IT", "ES", "PT", "IE", "EL")
Filters = list[tuple[str, str]]
SECTORS = [("sector", s) for s in ("S13", "S1311", "S1312", "S1313", "S1314")]
ITEMS = [("na_item", i) for i in ("P51G", "P51C", "D92PAY", "D41PAY", "B9", "TE", "TR")]

# name -> (dataset, filter sets, extra query parameters)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[Filters], Filters]] = {
    "gov_main": ("gov_10a_main", [[("unit", "MIO_EUR")] + SECTORS + ITEMS, [("unit", "MIO_EUR"), ("sector", "S13")] + ITEMS], []),
    "debt": ("gov_10dd_edpt1", [[("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "GD")]], []),
    "gdp": ("nama_10_gdp", [[("unit", "CP_MEUR"), ("unit", "CLV15_MEUR"), ("na_item", "B1GQ")], [("unit", "CP_MEUR"), ("na_item", "B1GQ")]], []),
    "population": ("nama_10_pe", [[("unit", "THS_PER"), ("na_item", "POP_NC")], [("unit", "THS_PER")]], []),
    "fixed_assets": (
        "nama_10_nfa_bs",
        [[("unit", "CRC_MEUR"), ("sector", "S13"), ("asset10", "N11N")], [("unit", "CRC_MEUR"), ("sector", "S13")], [("sector", "S13")]],
        [],
    ),
    "cofog_investment": ("gov_10a_exp", [[("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "P51G")]], []),
    "gov_quarterly": (
        "gov_10q_ggnfa",
        [[("unit", "MIO_EUR"), ("sector", "S13"), ("s_adj", "NSA"), ("na_item", "P51G"), ("na_item", "D41PAY"), ("na_item", "B9")], [("unit", "MIO_EUR"), ("sector", "S13")]],
        [("sinceTimePeriod", "2015-Q1")],
    ),
    "gdp_quarterly": ("namq_10_gdp", [[("unit", "CP_MEUR"), ("s_adj", "NSA"), ("na_item", "B1GQ")]], [("sinceTimePeriod", "2015-Q1")]),
}


def get(url: str, tries: int = 4, timeout: int = 300) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                return e.code, e.read()
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url[:200]}: {last}")


def jsonstat_to_long(doc: dict) -> pd.DataFrame:
    ids, sizes = doc["id"], doc["size"]
    cats = []
    for d in ids:
        index = doc["dimension"][d]["category"]["index"]
        cats.append(index if isinstance(index, list) else [c for c, _ in sorted(index.items(), key=lambda kv: kv[1])])
    values = doc.get("value", {})
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    rows = []
    for key, v in values.items():
        k = int(key)
        row = {d: codes[(k // s) % n] for d, codes, s, n in zip(ids, cats, strides, sizes)}
        row["value"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def log_response(log: list[str], tag: str, url: str, status: int, body: bytes) -> None:
    log.append(f"{tag}\t{status}\t{hashlib.sha256(body).hexdigest()}\t{len(body)} bytes\t{url[:300]}")


def eurostat(name: str, dataset: str, filter_sets: list[Filters], geos: tuple[str, ...], extra: Filters, out: Path, log: list[str]) -> bool:
    for filters in filter_sets:
        params = [("format", "JSON"), ("lang", "EN")] + filters + extra + [("geo", g) for g in geos]
        url = f"{EUROSTAT}/{dataset}?{urllib.parse.urlencode(params)}"
        try:
            status, body = get(url)
        except RuntimeError as e:
            log.append(f"eurostat {name}\tERROR\t{e}")
            continue
        log_response(log, f"eurostat {name}", url, status, body)
        if status != 200:
            log.append(f"eurostat {name}\tBODY\t{body[:300]!r}")
            continue
        try:
            doc = json.loads(body)
        except json.JSONDecodeError:
            log.append(f"eurostat {name}\tNOT JSON")
            continue
        if not doc.get("value"):
            log.append(f"eurostat {name}\tEMPTY with {filters}")
            continue
        d = jsonstat_to_long(doc)
        path = out / (f"{name}.csv.gz" if len(d) > 50_000 else f"{name}.csv")
        d.to_csv(path, index=False, **({"compression": {"method": "gzip", "mtime": 0}} if path.suffix == ".gz" else {}))
        last = sorted(d["time"].unique())[-1] if "time" in d else "?"
        log.append(f"eurostat {name}\tROWS\t{len(d)}\tcolumns {list(d.columns)}\tlast period {last}\tfilters {filters}")
        return True
    log.append(f"eurostat {name}\tFAILED")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    want = set(args.only) if args.only else None
    ok = True
    for name, (ds, fs, extra) in EUROSTAT_REQUESTS.items():
        if want and name not in want:
            continue
        ok &= eurostat(name, ds, fs, GEOS, extra, out, log)
        time.sleep(1)
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
