"""Download what the Commission's dependency screen needs, for every HS 6-digit product (Comext).

    python fetch_hs6.py --out data [--limit N]

Dataset DS-045409. For the years in YEARS:

1. Totals by product: EU-27 imports and exports, from outside the EU (partner EXT_EU27_2020) and
   from inside it (INT_EU27_2020). Four requests, one per flow and partner, all products.
2. Imports by supplier: for every HS 6-digit product, imports from every partner, requested in
   chunks of products. The partner detail is reduced here to one row per product and year
   (extra-EU imports, Herfindahl-Hirschman index of extra-EU suppliers, largest supplier and its
   share, and the shares of selected partners), so that the committed snapshot stays small.

Writes ``data/hs6_totals.csv`` and ``data/hs6_suppliers.csv`` and appends to ``data/fetch_log.txt``.
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

from fetch import jsonstat_to_long
from fetch_comext import get

API = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/DS-045409"
YEARS = (2002, 2005, 2008, 2012, 2015, 2019, 2021, 2022, 2023, 2024, 2025)
REPORTERS = ("EU27_2020", "EU")
EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "GR", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"}
PARTNERS = ("CN", "RU", "US", "GB", "CH", "NO", "TR", "JP", "KR", "IN", "TW", "VN", "BR", "UA", "BY")
CHUNK = 40


def url(params: list[tuple[str, str]], years: tuple[int, ...] = YEARS) -> str:
    base = [("format", "JSON"), ("lang", "en"), ("freq", "A"), ("indicators", "VALUE_IN_EUROS")]
    return f"{API}?{urllib.parse.urlencode(base + params + [('time', str(y)) for y in years])}"


def request(params: list[tuple[str, str]], log: list[str], tag: str, years: tuple[int, ...] = YEARS) -> pd.DataFrame | None:
    u = url(params, years)
    try:
        status, body = get(u, timeout=300)
    except RuntimeError as e:
        log.append(f"hs6 {tag}\tERROR\t{e}")
        return None
    log.append(f"hs6 {tag}\t{status}\t{hashlib.sha256(body).hexdigest()}\t{len(body)} bytes\t{u[:400]}")
    if status != 200:
        log.append(f"hs6 {tag}\tBODY\t{body[:300]!r}")
        return None
    doc = json.loads(body)
    if not doc.get("value"):
        log.append(f"hs6 {tag}\tEMPTY")
        return pd.DataFrame()
    return jsonstat_to_long(doc)


def extra_eu(p: pd.Series) -> pd.Series:
    return (p.str.len() == 2) & ~p.isin(EU27) & ~p.str.startswith("Q")


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    d = d[extra_eu(d.partner) & (d.value > 0)]
    rows = []
    for (prod, year), g in d.groupby(["product", "time"]):
        tot = g.value.sum()
        sh = (g.set_index("partner").value / tot).sort_values(ascending=False)
        row = {
            "product": prod,
            "year": int(year),
            "extra_imports": float(tot),
            "hhi": float((sh**2).sum()),
            "top_partner": sh.index[0],
            "top_share": float(sh.iloc[0]),
            "suppliers": int((sh >= 0.001).sum()),
        }
        for p in PARTNERS:
            row[f"share_{p}"] = float(sh.get(p, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=None, help="only the first N products (for testing)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetch_hs6.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]

    # 1. totals by product
    reporter, totals = None, []
    for flow in ("1", "2"):
        for partner in ("EXT_EU27_2020", "INT_EU27_2020"):
            got = None
            for rep in ([reporter] if reporter else REPORTERS):
                got = request([("reporter", rep), ("partner", partner), ("flow", flow)], log, f"totals flow {flow} {partner} {rep}")
                if got is not None and len(got):
                    reporter = rep
                    break
            if got is None or not len(got):
                # the all-years request may be too large: fall back to one request per year
                rep = reporter or REPORTERS[0]
                per_year = [request([("reporter", rep), ("partner", partner), ("flow", flow)], log, f"totals flow {flow} {partner} {y}", (y,)) for y in YEARS]
                per_year = [x for x in per_year if x is not None and len(x)]
                if not per_year:
                    log.append(f"hs6 totals flow {flow} {partner}\tFAILED")
                    continue
                reporter = rep
                got = pd.concat(per_year, ignore_index=True)
            totals.append(got)
    if not totals:
        (out / "fetch_log.txt").open("a", encoding="utf-8").write("\n".join(log) + "\n")
        print("\n".join(log))
        return 1
    t = pd.concat(totals, ignore_index=True)
    t = t[t["product"].astype(str).str.fullmatch(r"\d{6}")]
    t["col"] = t.flow.astype(str).map({"1": "imports", "2": "exports"}) + "_" + t.partner.map({"EXT_EU27_2020": "extra", "INT_EU27_2020": "intra"})
    tw = t.pivot_table(index=["product", "time"], columns="col", values="value", aggfunc="sum").reset_index().rename(columns={"time": "year"})
    tw.to_csv(out / "hs6_totals.csv", index=False)
    log.append(f"hs6 totals\tROWS\t{len(tw)}\tproducts {tw['product'].nunique()}")

    # 2. imports by supplier, in chunks of products
    products = sorted(tw.loc[tw.get("imports_extra", pd.Series(dtype=float)).fillna(0) > 0, "product"].unique())
    if args.limit:
        products = products[: args.limit]
    parts, failed = [], []
    queue = [products[i : i + CHUNK] for i in range(0, len(products), CHUNK)]
    while queue:
        chunk = queue.pop(0)
        got = request([("reporter", reporter), ("flow", "1")] + [("product", p) for p in chunk], log, f"suppliers {chunk[0]}..{chunk[-1]} ({len(chunk)})")
        if got is None:
            if len(chunk) > 1:
                half = len(chunk) // 2
                queue[:0] = [chunk[:half], chunk[half:]]
            else:
                failed.append(chunk[0])
            continue
        if len(got):
            got["value"] = pd.to_numeric(got.value, errors="coerce")
            parts.append(summarise(got))
        time.sleep(0.3)
    s = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    s.to_csv(out / "hs6_suppliers.csv", index=False, float_format="%.6g")
    log.append(f"hs6 suppliers\tROWS\t{len(s)}\tproducts {s['product'].nunique() if len(s) else 0}\tfailed {len(failed)}")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log[-6:]))
    return 1 if len(failed) > len(products) // 10 else 0


if __name__ == "__main__":
    sys.exit(main())
