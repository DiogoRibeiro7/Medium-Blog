"""Download 2025 EU trade at the CN 8-digit level, by supplier (Comext DS-045409).

    python fetch_cn8.py --out data [--limit N]

1. Totals: EU-27 imports and exports in 2025 with partners outside the EU (EXT_EU27_2020) and
   inside it (INT_EU27_2020), every product; 8-digit CN codes are kept.
2. Imports by partner for every CN code that appears in the CN-Prodcom correspondence, in
   chunks of codes. Partner-level rows are kept (compressed), so that imports can be summed
   exactly to Prodcom codes and supplier shares recomputed.

Writes ``data/cn8_totals_2025.csv`` and ``data/cn8_imports_by_partner_2025.csv.gz`` and appends to
``data/fetch_log.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_prodcom import get, jsonstat_to_long

API = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/DS-045409"
YEAR = 2025
CHUNK = 40


def request(params: list[tuple[str, str]], log: list[str], tag: str) -> pd.DataFrame | None:
    base = [("format", "JSON"), ("lang", "en"), ("freq", "A"), ("indicators", "VALUE_IN_EUROS"), ("time", str(YEAR)), ("reporter", "EU27_2020")]
    url = f"{API}?{urllib.parse.urlencode(base + params)}"
    try:
        status, body = get(url)
    except RuntimeError as e:
        log.append(f"cn8 {tag}\tERROR\t{e}")
        return None
    log.append(f"cn8 {tag}\t{status}\t{hashlib.sha256(body).hexdigest()}\t{len(body)} bytes")
    if status != 200:
        log.append(f"cn8 {tag}\tBODY\t{body[:300]!r}")
        return None
    try:
        doc = json.loads(body)
    except json.JSONDecodeError:
        log.append(f"cn8 {tag}\tNOT JSON\t{body[:200]!r}")
        return None
    if not doc.get("value"):
        return pd.DataFrame()
    return jsonstat_to_long(doc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    out = Path(args.out)
    log = [f"# fetch_cn8.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]

    totals = []
    for flow in ("1", "2"):
        for partner in ("EXT_EU27_2020", "INT_EU27_2020"):
            d = request([("partner", partner), ("flow", flow)], log, f"totals flow {flow} {partner}")
            if d is None or not len(d):
                log.append(f"cn8 totals flow {flow} {partner}\tFAILED")
                continue
            totals.append(d)
    if totals:
        t = pd.concat(totals, ignore_index=True)
        t = t[t["product"].astype(str).str.fullmatch(r"\d{8}")]
        t["col"] = t.flow.astype(str).map({"1": "imports", "2": "exports"}) + "_" + t.partner.map({"EXT_EU27_2020": "extra", "INT_EU27_2020": "intra"})
        tw = t.pivot_table(index="product", columns="col", values="value", aggfunc="sum").reset_index()
        tw.to_csv(out / "cn8_totals_2025.csv", index=False)
        log.append(f"cn8 totals\tROWS\t{len(tw)}")

    corr = pd.read_csv(out / "cn_prodcom_2025_raw.csv", header=None, dtype=str)
    corr.columns = corr.iloc[0]
    codes = sorted(corr.iloc[1:]["Commodity code"].dropna().str.replace(r"\D", "", regex=True).unique())
    if args.limit:
        codes = codes[: args.limit]
    parts, failed = [], []
    queue = [codes[i : i + CHUNK] for i in range(0, len(codes), CHUNK)]
    while queue:
        chunk = queue.pop(0)
        d = request([("flow", "1")] + [("product", c) for c in chunk], log, f"partners {chunk[0]}..{chunk[-1]} ({len(chunk)})")
        if d is None:
            if len(chunk) > 1:
                h = len(chunk) // 2
                queue[:0] = [chunk[:h], chunk[h:]]
            else:
                failed.append(chunk[0])
            continue
        if len(d):
            parts.append(d[["product", "partner", "value"]])
        time.sleep(0.3)
    if parts:
        p = pd.concat(parts, ignore_index=True)
        p.to_csv(out / "cn8_imports_by_partner_2025.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        log.append(f"cn8 partners\tROWS\t{len(p)}\tcodes {p['product'].nunique()}\tfailed {len(failed)}")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log[-5:]))
    return 1 if (not totals or len(failed) > len(codes) // 10) else 0


if __name__ == "__main__":
    sys.exit(main())
