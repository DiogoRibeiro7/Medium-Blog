"""Download the public series behind the euro-crisis ledger.

    python fetch.py --out data [--only name ...]

Sources:

1. Eurostat (JSON-stat API): monthly long-term government bond yields (EMU convergence
   criterion), general government debt, interest payable and receivable, GDP, and the
   supplementary tables on intergovernmental lending and on support to financial
   institutions. Each request carries filter sets tried in order, narrowest first.
2. Eurostat's table of contents, searched for the supplementary tables (``eurostat_toc.txt``).
3. FRED (OECD Main Economic Indicators): monthly 10-year government bond yields of advanced
   economies outside the euro area, for the safe-asset counterfactual.
4. BIS consolidated banking statistics (WS_CBS_PUB): foreign claims of banks, by nationality,
   on the five programme countries and Italy.

Every response is logged with URL, HTTP status and SHA-256 in ``data/fetch_log.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
TOC = "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
BIS = (
    "https://stats.bis.org/api/v1/data/WS_CBS_PUB/{key}?startPeriod=2005-Q1&format=csv",
    "https://data.bis.org/api/v1/data/BIS,WS_CBS_PUB,1.0/{key}?startPeriod=2005-Q1&format=csv",
)
AGENT = "euro-crisis-ledger/1.0 (research; python urllib)"

EURO = ("BE", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV", "LT", "LU", "MT", "NL", "AT", "PT", "SI", "SK", "FI")
GEOS = EURO + ("DK", "SE", "EA20", "EA19", "EU27_2020")
Filters = list[tuple[str, str]]

# name -> (dataset, filter sets, geographies)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[Filters], tuple[str, ...] | None]] = {
    "yields_monthly": ("irt_lt_mcby_m", [[], [("int_rt", "MCBY")]], GEOS),
    "debt": ("gov_10dd_edpt1", [[("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "GD")], [("unit", "MIO_EUR"), ("sector", "S13")]], GEOS),
    "gov_main": (
        "gov_10a_main",
        [
            [("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "D41PAY"), ("na_item", "D41REC"), ("na_item", "B9"), ("na_item", "D4PAY"), ("na_item", "D4REC")],
            [("unit", "MIO_EUR"), ("sector", "S13")],
        ],
        GEOS,
    ),
    "gdp": ("nama_10_gdp", [[("unit", "CP_MEUR"), ("na_item", "B1GQ")]], GEOS),
    "debt_structure": ("gov_10dd_ggd", [[("unit", "MIO_EUR"), ("sector", "S13")], []], ("EL", "IE", "PT", "CY", "ES", "DE", "FR", "NL", "FI", "AT", "IT")),
}
# filled in from the table of contents if these codes exist; tried as given otherwise
SUPPLEMENTARY = ("gov_10dd_igl", "gov_dd_igl", "gov_10dd_ilg", "gov_10dd_fins", "gov_dd_fi", "gov_10dd_fi")

FRED_SERIES = {c: f"IRLTLT01{c}M156N" for c in ("US", "GB", "CA", "SE", "NO", "CH", "JP", "AU", "NZ", "DK", "DE", "FR", "NL", "IT", "ES")}
FRED_SERIES["USD_PER_EUR"] = "EXUSEU"

# BIS consolidated banking statistics. Key order: FREQ.L_MEASURE.L_REP_CTY.CBS_BANK_TYPE.CBS_BASIS.
# L_POSITION.L_INSTR.REM_MATURITY.CURR_TYPE_BOOK.L_CP_SECTOR.L_CP_COUNTRY. The analysis uses basis U
# (ultimate risk), position C (foreign claims), sector O (official sector) and all sectors (A).
BIS_KEYS = (
    "Q.S..4R...A.A.TO1..GR+IE+PT+CY+ES+IT",
    "Q.S..4B...A.A.TO1..GR+IE+PT+CY+ES+IT",
)


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


def eurostat(name: str, dataset: str, filter_sets: list[Filters], geos: tuple[str, ...] | None, out: Path, log: list[str]) -> bool:
    for filters in filter_sets:
        params = [("format", "JSON"), ("lang", "EN")] + filters + ([("geo", g) for g in geos] if geos else [])
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
        d.to_csv(out / f"{name}.csv", index=False)
        log.append(f"eurostat {name}\tROWS\t{len(d)}\tcolumns {list(d.columns)}\tfilters {filters}")
        return True
    log.append(f"eurostat {name}\tFAILED")
    return False


def toc(out: Path, log: list[str]) -> list[str]:
    try:
        status, body = get(TOC)
    except RuntimeError as e:
        log.append(f"toc\tERROR\t{e}")
        return []
    log_response(log, "toc", TOC, status, body)
    if status != 200:
        return []
    text = body.decode("utf-8", errors="replace")
    keep = [ln for ln in text.splitlines() if any(k in ln.lower() for k in ("intergovernmental", "financial crisis", "financial institutions", "government debt", "contingent liabilities"))]
    (out / "eurostat_toc.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
    codes = []
    for ln in keep:
        for part in ln.split("\t"):
            p = part.strip().strip('"')
            if p.startswith("gov_") and " " not in p:
                codes.append(p)
    log.append(f"toc\tMATCHES\t{len(keep)} lines\tcodes {sorted(set(codes))}")
    return sorted(set(codes))


def fred(out: Path, log: list[str]) -> bool:
    parts = []
    for c, sid in FRED_SERIES.items():
        url = FRED.format(sid=sid)
        try:
            status, body = get(url)
        except RuntimeError as e:
            log.append(f"fred {sid}\tERROR\t{e}")
            continue
        log_response(log, f"fred {sid}", url, status, body)
        if status != 200:
            continue
        d = pd.read_csv(io.BytesIO(body))
        d.columns = ["date", "value"]
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d["country"] = c
        parts.append(d.dropna())
        time.sleep(1)
    if not parts:
        return False
    f = pd.concat(parts, ignore_index=True)
    f.to_csv(out / "fred_long_rates.csv", index=False)
    log.append(f"fred\tROWS\t{len(f)}\tcountries {sorted(f.country.unique())}")
    return True


def bis(out: Path, log: list[str]) -> bool:
    for key in BIS_KEYS:
        for base in BIS:
            url = base.format(key=key)
            try:
                status, body = get(url, timeout=600)
            except RuntimeError as e:
                log.append(f"bis {key}\tERROR\t{e}")
                continue
            log_response(log, f"bis {key}", url, status, body)
            if status != 200 or len(body) < 200:
                log.append(f"bis {key}\tBODY\t{body[:300]!r}")
                continue
            d = pd.read_csv(io.BytesIO(body), dtype=str)
            d.to_csv(out / "bis_cbs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            log.append(f"bis\tROWS\t{len(d)}\tcolumns {list(d.columns)[:20]}")
            return True
    log.append("bis\tFAILED")
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
    for name, (ds, fs, geos) in EUROSTAT_REQUESTS.items():
        if want and name not in want:
            continue
        ok &= eurostat(name, ds, fs, geos, out, log)
        time.sleep(1)
    if not want or "supplementary" in want:
        codes = toc(out, log)
        for code in list(dict.fromkeys([c for c in codes if any(k in c for k in ("igl", "fi", "ilg"))] + list(SUPPLEMENTARY))):
            eurostat(f"supp_{code}", code, [[("unit", "MIO_EUR")], []], None, out, log)
            time.sleep(1)
    if not want or "fred" in want:
        ok &= fred(out, log)
    if not want or "bis" in want:
        bis(out, log)
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
