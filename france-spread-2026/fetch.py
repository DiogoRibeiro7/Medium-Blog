"""Download the public series behind the France 2026 / euro crisis 2011 comparison.

    python fetch.py --out data [--only name ...]

Sources:

1. Eurostat (JSON-stat API): daily and monthly long-term government bond yields (EMU
   convergence criterion), annual and quarterly government interest payable, net lending and
   debt, and annual and quarterly GDP at current prices. Each request carries filter sets tried
   in order, narrowest first.
2. ECB Data Portal: daily 10-year benchmark yields, requested only as a fallback and as a probe
   of the available keys if the Eurostat daily table is missing.

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
ECB = "https://data-api.ecb.europa.eu/service/data/FM/{key}?startPeriod={start}&format=csvdata"
AGENT = "france-spread-2026/1.0 (research; python urllib)"

SPREAD = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
FISCAL = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
Filters = list[tuple[str, str]]

# name -> (dataset, filter sets, geographies, extra query parameters)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[Filters], tuple[str, ...], Filters]] = {
    "yields_daily": ("irt_lt_mcby_d", [[], [("int_rt", "MCBY")]], SPREAD, [("sinceTimePeriod", "2007-01-01")]),
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


def ecb_daily(out: Path, log: list[str]) -> bool:
    """Fallback: ECB benchmark yields. First probe which daily yield keys exist, then fetch them."""
    url = ECB.format(key="D......YLD", start="2026-09-01")
    try:
        status, body = get(url, timeout=600)
    except RuntimeError as e:
        log.append(f"ecb probe\tERROR\t{e}")
        return False
    log_response(log, "ecb probe", url, status, body)
    if status != 200 or len(body) < 200:
        log.append(f"ecb probe\tBODY\t{body[:300]!r}")
        return False
    probe = pd.read_csv(io.BytesIO(body), dtype=str)
    keys = sorted(probe["KEY"].unique()) if "KEY" in probe else []
    (out / "ecb_keys.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")
    want = [k for k in keys if "10Y" in k and any(f".{c}." in k for c in SPREAD)]
    log.append(f"ecb probe\tKEYS\t{len(keys)}\t10-year keys {want[:20]}")
    if not want:
        return False
    parts = []
    for k in want:
        short = k.split(".", 1)[1]
        u = ECB.format(key=short, start="2007-01-01")
        try:
            st, b = get(u, timeout=600)
        except RuntimeError as e:
            log.append(f"ecb {short}\tERROR\t{e}")
            continue
        log_response(log, f"ecb {short}", u, st, b)
        if st == 200 and len(b) > 200:
            parts.append(pd.read_csv(io.BytesIO(b), dtype=str)[["KEY", "TIME_PERIOD", "OBS_VALUE"]])
    if not parts:
        return False
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(out / "ecb_yields_daily.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    log.append(f"ecb daily\tROWS\t{len(d)}")
    return True


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
    daily_ok = None
    for name, (ds, fs, geos, extra) in EUROSTAT_REQUESTS.items():
        if want and name not in want:
            continue
        r = eurostat(name, ds, fs, geos, extra, out, log)
        if name == "yields_daily":
            daily_ok = r
        else:
            ok &= r
        time.sleep(1)
    if (want and "ecb" in want) or daily_ok is False:
        daily_ok = ecb_daily(out, log) or bool(daily_ok)
    if daily_ok is False:
        ok = False
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
