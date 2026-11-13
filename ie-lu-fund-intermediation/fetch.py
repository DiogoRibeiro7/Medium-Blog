"""Download the Eurostat series for the test of fund intermediation through Ireland and Luxembourg.

    python fetch.py --out data

Eurostat (JSON-stat dissemination API), million euro:

* ``bop_eu6_q``: portfolio investment of the EU27 as one economy against non-EU residents, the United
  States and the United Kingdom, quarterly flows: assets (purchases of foreign securities net of
  sales) by resident sector, and liabilities (non-EU purchases of EU securities).
* ``bop_c6_q``: portfolio investment of each of the 27 member states, quarterly flows, by partner:
  the whole rest of the world, other EU members, non-EU, the United States, the United Kingdom,
  Ireland and Luxembourg; equity, investment fund shares and debt securities.
* ``bop_iip6_q``: the matching positions at the end of each year, for each member state and for
  the EU27.
* ``nasa_10_f_bs``: financial balance sheets of Ireland, Luxembourg and the EU27 by institutional
  sector, to show which sectors hold the securities and issue the fund shares.

One request per member state keeps each response under the API's synchronous size limit. Raw
responses are kept gzipped in ``data/raw/``, tidy tables in ``data/*.csv``, and
``data/fetch_log.txt`` records URL, HTTP status, size and SHA-256 of every response.
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

UA = {"User-Agent": "ie-lu-fund-intermediation/1.0 (research companion)"}
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

MEMBERS = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT",
           "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
PARTNERS = ["WRL_REST", "EU27_2020", "EXT_EU27_2020", "US", "UK", "IE", "LU"]
PI_ITEMS = ["FA__P__F", "FA__P__F5", "FA__P__F51", "FA__P__F52", "FA__P__F3"]
EU_SECTORS = ["S1", "S121", "S12T", "S123", "S12M", "S13", "S1V", "S1P"]
FA_SECTORS = ["S1", "S12", "S121", "S122", "S123", "S124", "S125", "S127", "S128_S129", "S11", "S14_S15", "S13",
              "S2"]


def multi(name: str, values: list[str]) -> list[tuple[str, str]]:
    return [(name, v) for v in values]


def _requests() -> dict[str, tuple[str, list[tuple[str, str]]]]:
    """name -> (dataset, filters)."""
    common = [("currency", "MIO_EUR"), ("sector10", "S1"), ("sectpart", "S1")]
    req: dict[str, tuple[str, list[tuple[str, str]]]] = {
        "eu_flows": ("bop_eu6_q", [("geo", "EU27_2020"), ("currency", "MIO_EUR"), ("sectpart", "S1"),
                                   ("s_adj", "NSA"), ("freq", "Q")]
                     + multi("partner", ["EXT_EU27_2020", "US", "UK"]) + multi("stk_flow", ["ASS", "LIAB"])
                     + multi("bop_item", ["FA__P__F", "FA__P__F5", "FA__P__F3"]) + multi("sector10", EU_SECTORS)),
        "eu_positions": ("bop_iip6_q", [("geo", "EU27_2020"), ("freq", "A")] + common
                         + multi("partner", ["EXT_EU27_2020", "US", "UK"]) + multi("stk_flow", ["A_LE", "L_LE"])
                         + multi("bop_item", PI_ITEMS)),
        "fin_accounts": ("nasa_10_f_bs", [("unit", "MIO_EUR"), ("co_nco", "NCO")]
                         + multi("geo", ["IE", "LU", "EU27_2020"]) + multi("finpos", ["ASS", "LIAB"])
                         + multi("sector", FA_SECTORS) + multi("na_item", ["F3", "F5", "F51", "F52"])),
    }
    for g in MEMBERS:
        req[f"flows_{g}"] = ("bop_c6_q", [("geo", g), ("freq", "Q")] + common + multi("partner", PARTNERS)
                             + multi("stk_flow", ["ASS", "LIAB"]) + multi("bop_item", PI_ITEMS))
        req[f"positions_{g}"] = ("bop_iip6_q", [("geo", g), ("freq", "A")] + common + multi("partner", PARTNERS)
                                 + multi("stk_flow", ["A_LE", "L_LE"]) + multi("bop_item", PI_ITEMS))
    return req


EUROSTAT_REQUESTS = _requests()

# Tidy tables: which raw responses are stacked into which CSV.
TABLES = {
    "eu_flows": ["eu_flows"],
    "eu_positions": ["eu_positions"],
    "fin_accounts": ["fin_accounts"],
    "member_flows": [f"flows_{g}" for g in MEMBERS],
    "member_positions": [f"positions_{g}" for g in MEMBERS],
}


def get(url: str, tries: int = 5) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
            # Eurostat answers 200 with an error document when a request is queued (413 inside).
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
            df = jsonstat_to_long(json.loads(gzip.decompress(path.read_bytes())))
            df.insert(0, "dataset", EUROSTAT_REQUESTS[name][0])
            frames.append(df)
        if not frames:
            continue
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
        print(log[-1][:120], flush=True)
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    tidy(out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
