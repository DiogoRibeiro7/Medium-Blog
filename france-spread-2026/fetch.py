"""Download the public series behind the France 2026 / euro crisis 2011 comparison.

    python fetch.py --out data [--only name ...]

Sources:

1. Eurostat (JSON-stat API): monthly long-term government bond yields (EMU
   convergence criterion), annual and quarterly government interest payable, net lending and
   debt, and annual and quarterly GDP at current prices. Each request carries filter sets tried
   in order, narrowest first. (Eurostat's daily yield table, irt_lt_mcby_d, is not disseminated
   through the API.)
2. ECB Data Portal: monthly convergence-criterion yields (dataset IRS), usually published before
   Eurostat's copy, and a probe of the daily financial-market series (dataset FM) for France.
3. Daily probes (kept when they answer, never required): Bundesbank daily German 10-year
   yields and the Banque de France's daily French 10-year constant-maturity rate (TEC10), with
   a search of the Webstat catalogue. Stooq's free CSV downloads (``--only stooq``) now
   answer 404 to scripted requests.

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
AGENT = "france-spread-2026/1.0 (research; python urllib)"

SPREAD = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
FISCAL = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
Filters = list[tuple[str, str]]

# name -> (dataset, filter sets, geographies, extra query parameters)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[Filters], tuple[str, ...], Filters]] = {
    "yields_monthly": ("irt_lt_mcby_m", [[], [("int_rt", "MCBY")]], SPREAD, []),
    "gov_main": (
        "gov_10a_main",
        [
            [("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "D41PAY"), ("na_item", "B9"), ("na_item", "TE"), ("na_item", "TR")],
            [("unit", "MIO_EUR"), ("sector", "S13")],
        ],
        FISCAL,
        [],
    ),
    "debt": ("gov_10dd_edpt1", [[("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "GD")], [("unit", "MIO_EUR"), ("sector", "S13")]], FISCAL, []),
    "gdp": ("nama_10_gdp", [[("unit", "CP_MEUR"), ("na_item", "B1GQ")]], FISCAL, []),
    "gov_quarterly": (
        "gov_10q_ggnfa",
        [
            [("unit", "MIO_EUR"), ("sector", "S13"), ("s_adj", "NSA"), ("na_item", "D41PAY"), ("na_item", "B9")],
            [("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "D41PAY"), ("na_item", "B9")],
            [("unit", "MIO_EUR"), ("sector", "S13")],
        ],
        FISCAL,
        [("sinceTimePeriod", "2019-Q1")],
    ),
    "debt_quarterly": ("gov_10q_ggdebt", [[("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "GD")]], FISCAL, [("sinceTimePeriod", "2019-Q1")]),
    "gdp_quarterly": (
        "namq_10_gdp",
        [[("unit", "CP_MEUR"), ("s_adj", "NSA"), ("na_item", "B1GQ")], [("unit", "CP_MEUR"), ("na_item", "B1GQ")]],
        FISCAL,
        [("sinceTimePeriod", "2019-Q1")],
    ),
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


ECB_CODES = {"EL": "GR"}
STOOQ = "https://stooq.com/q/d/l/?s={sym}&i=d"
STOOQ_SYMBOLS = {c: f"10{ECB_CODES.get(c, c).lower()}y.b" for c in SPREAD}


def ecb_monthly(out: Path, log: list[str]) -> bool:
    """ECB convergence-criterion yields (dataset IRS), often published before Eurostat's copy."""
    areas = "+".join(ECB_CODES.get(c, c) for c in SPREAD)
    url = f"https://data-api.ecb.europa.eu/service/data/IRS/M.{areas}.L.L40.CI.0000.EUR.N.Z?startPeriod=1995-01&format=csvdata"
    try:
        status, body = get(url, timeout=600)
    except RuntimeError as e:
        log.append(f"ecb irs\tERROR\t{e}")
        return False
    log_response(log, "ecb irs", url, status, body)
    if status != 200 or len(body) < 200:
        log.append(f"ecb irs\tBODY\t{body[:300]!r}")
        return False
    d = pd.read_csv(io.BytesIO(body), dtype=str)
    keep = [c for c in ("KEY", "REF_AREA", "TIME_PERIOD", "OBS_VALUE") if c in d]
    d = d[keep]
    d.to_csv(out / "ecb_yields_monthly.csv", index=False)
    log.append(f"ecb irs\tROWS\t{len(d)}\tlast period {d['TIME_PERIOD'].max() if 'TIME_PERIOD' in d else '?'}")
    return True


def ecb_probe(out: Path, log: list[str]) -> None:
    """Record which daily FM series exist for France, to find a daily 10-year yield if one is published."""
    lines = []
    for key in ("D.FR.....", "B.FR.....", "D.U2.EUR.4F.BB.."):
        url = f"https://data-api.ecb.europa.eu/service/data/FM/{key}?lastNObservations=1&format=csvdata"
        try:
            status, body = get(url, tries=2, timeout=300)
        except RuntimeError as e:
            log.append(f"ecb probe {key}\tERROR\t{e}")
            continue
        log_response(log, f"ecb probe {key}", url, status, body)
        if status == 200 and len(body) > 100:
            p = pd.read_csv(io.BytesIO(body), dtype=str)
            cols = [c for c in ("KEY", "TITLE", "TIME_PERIOD") if c in p]
            lines += [" | ".join(str(r[c]) for c in cols) for _, r in p.iterrows()]
    (out / "ecb_keys.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.append(f"ecb probe\tSERIES\t{len(lines)}")


def stooq(out: Path, log: list[str]) -> bool:
    """Daily 10-year benchmark yields from Stooq's free CSV downloads."""
    parts = []
    for c, sym in STOOQ_SYMBOLS.items():
        url = STOOQ.format(sym=sym)
        try:
            status, body = get(url, tries=2, timeout=120)
        except RuntimeError as e:
            log.append(f"stooq {sym}\tERROR\t{e}")
            continue
        log_response(log, f"stooq {sym}", url, status, body)
        head = body[:80].decode("utf-8", errors="replace")
        if status != 200 or not head.lower().startswith("date"):
            log.append(f"stooq {sym}\tBODY\t{head!r}")
            continue
        d = pd.read_csv(io.BytesIO(body))
        d.columns = [x.lower() for x in d.columns]
        d = d[["date", "close"]].rename(columns={"close": "value"})
        d["geo"] = c
        parts.append(d)
        time.sleep(2)
    if not parts:
        log.append("stooq\tFAILED")
        return False
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(out / "yields_daily.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    log.append(f"stooq\tROWS\t{len(d)}\tcountries {sorted(d.geo.unique())}\tlast {d.date.max()}")
    return len(parts) >= 4


DAILY_PROBES = {
    # Bundesbank: yields on listed Federal securities, residual maturity 10 years, daily
    "bbk_de10": "https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A?format=csv&lang=en",
    # Banque de France Webstat: French 10-year constant-maturity rate (TEC10), daily
    "bdf_fr10": "https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets/observations/exports/csv?where=series_key%3D%22FM.D.FR.EUR.FR2.BB.FRMOYTEC10.HSTA%22",
    # Banque de France Webstat catalogue, searched for the 10-year constant-maturity rate
    "bdf_catalog": "https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets?limit=100&where=search(%22TEC%22)",
    "bdf_catalog_all": "https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets?limit=100&select=dataset_id",
}


def daily_probes(out: Path, log: list[str]) -> None:
    """Try the daily sources; keep whatever answers, never fail the run on them."""
    for name, url in DAILY_PROBES.items():
        try:
            status, body = get(url, tries=1, timeout=90)
        except RuntimeError as e:
            log.append(f"probe {name}\tERROR\t{e}")
            continue
        log_response(log, f"probe {name}", url, status, body)
        if status == 200 and len(body) > 300:
            (out / f"probe_{name}.csv.gz").write_bytes(__import__("gzip").compress(body, mtime=0))
            log.append(f"probe {name}\tKEPT\t{body[:160]!r}")
        else:
            log.append(f"probe {name}\tBODY\t{body[:200]!r}")


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
    for name, (ds, fs, geos, extra) in EUROSTAT_REQUESTS.items():
        if want and name not in want:
            continue
        ok &= eurostat(name, ds, fs, geos, extra, out, log)
        time.sleep(1)
    if not want or "ecb" in want:
        ok &= ecb_monthly(out, log)
        ecb_probe(out, log)
    if want and "stooq" in want:
        ok &= stooq(out, log)
    if not want or "probes" in want:
        daily_probes(out, log)
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
