"""Download an annual HS 6-digit panel of EU imports by supplier, in value and quantity (Comext).

    python fetch_panel.py --out data/parts --years 2002-2007 [--limit N]

Dataset DS-045409 ("EU trade since 1988 by HS2-4-6 and CN8"), reporter EU27_2020. For every year
in the range:

1. Totals by product: EU-27 imports and exports with partners outside the EU (EXT_EU27_2020) and
   inside it (INT_EU27_2020), every HS 6-digit code, one request per year, flow and partner.
2. Imports by supplier: for every HS 6-digit code with imports from outside the EU, imports from
   every partner in euros and in 100 kg, requested in chunks of codes. The partner detail is
   reduced here so that the committed snapshot stays small:
   - ``summary``: one row per product and year with extra-EU imports (value and quantity), the
     Herfindahl-Hirschman index of extra-EU suppliers computed on all partners and the number
     of suppliers with at least 0.1% of the value;
   - ``partners``: the ten largest extra-EU suppliers of each product and year (value and
     quantity) and one row ``ZZ`` with the remainder.

The workflow runs one range of years per job and stitches the parts together. Codes are kept as
reported: the Harmonized System is revised every five years (2007, 2012, 2017, 2022), so a code
can start or stop in the panel; ``panel.py`` deals with that.
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

API = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/DS-045409"
AGENT = "dependency-early-warning/1.0"
REPORTER = "EU27_2020"
EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "GR", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"}
INDICATORS = ("VALUE_IN_EUROS", "QUANTITY_IN_100KG")
CHUNK = 20
TOP = 10


def get(url: str, tries: int = 4, timeout: int = 300) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
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
    """Flatten a JSON-stat 2.0 dataset into one row per cell."""
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


def request(params: list[tuple[str, str]], years: list[int], indicators: tuple[str, ...], log: list[str], tag: str) -> pd.DataFrame | None:
    base = [("format", "JSON"), ("lang", "en"), ("freq", "A"), ("reporter", REPORTER)]
    base += [("indicators", i) for i in indicators] + [("time", str(y)) for y in years]
    u = f"{API}?{urllib.parse.urlencode(base + params)}"
    try:
        status, body = get(u)
    except RuntimeError as e:
        log.append(f"panel {tag}\tERROR\t{e}")
        return None
    log.append(f"panel {tag}\t{status}\t{hashlib.sha256(body).hexdigest()}\t{len(body)} bytes")
    if status != 200:
        log.append(f"panel {tag}\tBODY\t{body[:300]!r}")
        return None
    try:
        doc = json.loads(body)
    except json.JSONDecodeError:
        log.append(f"panel {tag}\tNOT JSON\t{body[:200]!r}")
        return None
    if "value" not in doc:
        log.append(f"panel {tag}\tNO VALUES\t{json.dumps(doc)[:300]}")
        return None
    if not doc["value"]:
        return pd.DataFrame()
    d = jsonstat_to_long(doc)
    ind = next((c for c in d.columns if c.startswith("indicator")), None)
    d = d.rename(columns={ind: "indicator"}) if ind else d.assign(indicator=indicators[0])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["time"] = d["time"].astype(int)
    return d


def extra_eu(p: pd.Series) -> pd.Series:
    return (p.str.len() == 2) & ~p.isin(EU27) & ~p.str.startswith("Q")


def reduce(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partner-level rows of one chunk -> (summary rows, top-ten partner rows)."""
    d = d[extra_eu(d["partner"].astype(str))]
    w = d.pivot_table(index=["product", "time", "partner"], columns="indicator", values="value", aggfunc="sum").reset_index()
    w = w.rename(columns={"time": "year", "VALUE_IN_EUROS": "value", "QUANTITY_IN_100KG": "quantity"})
    if "quantity" not in w:
        w["quantity"] = float("nan")
    w = w[w["value"].fillna(0) > 0]
    summary, partners = [], []
    for (prod, year), g in w.groupby(["product", "year"], sort=True):
        g = g.sort_values("value", ascending=False)
        tot = g["value"].sum()
        sh = g["value"] / tot
        summary.append({
            "product": prod,
            "year": int(year),
            "value": float(tot),
            "quantity": float(g["quantity"].sum(min_count=1)),
            "hhi": float((sh**2).sum()),
            "suppliers": int((sh >= 0.001).sum()),
        })
        top = g.head(TOP)[["product", "year", "partner", "value", "quantity"]]
        partners.append(top)
        rest = g.iloc[TOP:]
        if len(rest):
            partners.append(pd.DataFrame([{"product": prod, "year": int(year), "partner": "ZZ", "value": float(rest["value"].sum()), "quantity": float(rest["quantity"].sum(min_count=1))}]))
    return pd.DataFrame(summary), (pd.concat(partners, ignore_index=True) if partners else pd.DataFrame())


def parse_years(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/parts")
    ap.add_argument("--years", required=True, help="a range such as 2002-2007")
    ap.add_argument("--limit", type=int, default=None, help="only the first N products (for testing)")
    args = ap.parse_args(argv)
    years = parse_years(args.years)
    tag = f"{years[0]}_{years[-1]}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetch_panel.py --years {args.years} run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]

    # 1. totals, one request per year, flow and partner aggregate
    totals = []
    for y in years:
        for flow in ("1", "2"):
            for partner in ("EXT_EU27_2020", "INT_EU27_2020"):
                d = request([("partner", partner), ("flow", flow)], [y], ("VALUE_IN_EUROS",), log, f"totals {y} flow {flow} {partner}")
                if d is None or not len(d):
                    log.append(f"panel totals {y} flow {flow} {partner}\tFAILED")
                    continue
                totals.append(d)
                time.sleep(0.5)
    if not totals:
        (out / f"log_{tag}.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        print("\n".join(log))
        return 1
    t = pd.concat(totals, ignore_index=True)
    t = t[t["product"].astype(str).str.fullmatch(r"\d{6}")]
    t["col"] = t["flow"].astype(str).map({"1": "imports", "2": "exports"}) + "_" + t["partner"].map({"EXT_EU27_2020": "extra", "INT_EU27_2020": "intra"})
    tw = t.pivot_table(index=["product", "time"], columns="col", values="value", aggfunc="sum").reset_index().rename(columns={"time": "year"})
    tw.to_csv(out / f"totals_{tag}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    log.append(f"panel totals\tROWS\t{len(tw)}\tproducts {tw['product'].nunique()}")

    # 2. imports by supplier, value and quantity, in chunks of products
    products = sorted(tw.loc[tw.get("imports_extra", pd.Series(0, index=tw.index)).fillna(0) > 0, "product"].unique())
    if args.limit:
        products = products[: args.limit]
    summaries, partners, failed = [], [], []
    queue = [products[i : i + CHUNK] for i in range(0, len(products), CHUNK)]
    while queue:
        chunk = queue.pop(0)
        d = request([("flow", "1")] + [("product", p) for p in chunk], years, INDICATORS, log, f"suppliers {chunk[0]}..{chunk[-1]} ({len(chunk)})")
        if d is None:
            if len(chunk) > 1:
                h = len(chunk) // 2
                queue[:0] = [chunk[:h], chunk[h:]]
            else:
                failed.append(chunk[0])
            continue
        if len(d):
            s, p = reduce(d)
            summaries.append(s)
            partners.append(p)
        time.sleep(0.3)
    s = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    p = pd.concat(partners, ignore_index=True) if partners else pd.DataFrame()
    s.to_csv(out / f"summary_{tag}.csv.gz", index=False, float_format="%.8g", compression={"method": "gzip", "mtime": 0})
    p.to_csv(out / f"partners_{tag}.csv.gz", index=False, float_format="%.8g", compression={"method": "gzip", "mtime": 0})
    log.append(f"panel suppliers\tROWS\t{len(s)} summary, {len(p)} partner rows\tproducts {s['product'].nunique() if len(s) else 0}\tfailed {len(failed)}: {' '.join(failed[:50])}")
    (out / f"log_{tag}.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log[-4:]))
    return 1 if len(failed) > max(1, len(products) // 50) else 0


if __name__ == "__main__":
    sys.exit(main())
