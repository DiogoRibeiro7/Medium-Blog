"""Download the public series behind the Portugal 2010-2012 analysis.

    python fetch.py --out data [--only name ...]

Sources:

1. Eurostat (JSON-stat API): monthly long-term government bond yields (EMU convergence
   criterion); quarterly government debt, net lending and interest; quarterly GDP at current
   prices and real growth; the quarterly current account and net international investment
   position; annual government deficit and debt in % of GDP; annual real GDP growth; private
   sector debt; unemployment.
2. ECB Data Portal: monthly convergence-criterion yields (IRS); TARGET balances (TGB); three-month
   Euribor and the main refinancing rate (FM); interest rates on new and outstanding bank loans to
   households for house purchase and to firms (MIR); Portuguese banks' housing loans (BSI).

Every response is logged with URL, HTTP status and SHA-256 in ``data/fetch_log.txt``. Requests
marked optional are logged on failure but do not fail the run.
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
ECB_API = "https://data-api.ecb.europa.eu/service/data"
AGENT = "portugal-2010-contagion/1.0 (research; python urllib)"

EURO = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
Filters = list[tuple[str, str]]
SINCE_Q = [("sinceTimePeriod", "1999-Q1")]
SINCE_A = [("sinceTimePeriod", "1995")]

# name -> (dataset, filter sets tried in order, geographies, extra query parameters, required)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[Filters], tuple[str, ...], Filters, bool]] = {
    "yields_monthly": ("irt_lt_mcby_m", [[], [("int_rt", "MCBY")]], EURO, [], True),
    "gov_quarterly": (
        "gov_10q_ggnfa",
        [
            [("unit", "MIO_EUR"), ("sector", "S13"), ("s_adj", "NSA"), ("na_item", "D41PAY"), ("na_item", "B9")],
            [("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "D41PAY"), ("na_item", "B9")],
        ],
        EURO, SINCE_Q, True,
    ),
    "debt_quarterly": ("gov_10q_ggdebt", [[("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "GD")]], EURO, SINCE_Q, True),
    "gdp_quarterly": (
        "namq_10_gdp",
        [[("unit", "CP_MEUR"), ("s_adj", "NSA"), ("na_item", "B1GQ")], [("unit", "CP_MEUR"), ("na_item", "B1GQ")]],
        EURO, SINCE_Q, True,
    ),
    "real_growth_quarterly": (
        "namq_10_gdp",
        [
            [("unit", "CLV_PCH_SM"), ("s_adj", "SCA"), ("na_item", "B1GQ")],
            [("unit", "CLV_PCH_SM"), ("s_adj", "NSA"), ("na_item", "B1GQ")],
            [("unit", "CLV_PCH_SM"), ("na_item", "B1GQ")],
        ],
        EURO, SINCE_Q, True,
    ),
    "current_account": (
        "bop_gdp6_q",
        [
            [("unit", "PC_GDP"), ("bop_item", "CA"), ("stk_flow", "BAL"), ("partner", "WRL_REST"), ("s_adj", "NSA")],
            [("unit", "PC_GDP"), ("bop_item", "CA"), ("stk_flow", "BAL"), ("partner", "WRL_REST")],
        ],
        EURO, SINCE_Q, False,
    ),
    "niip_quarterly": (
        "bop_iip6_q",
        [
            [("unit", "PC_GDP"), ("bop_item", "FA"), ("stk_flow", "N_LE"), ("partner", "WRL_REST"), ("sector10", "S1"), ("sectpart", "S1")],
            [("unit", "PC_GDP"), ("bop_item", "FA"), ("stk_flow", "N_LE"), ("partner", "WRL_REST")],
        ],
        EURO, SINCE_Q, False,
    ),
    "niip_annual": ("tipsii10", [[("unit", "PC_GDP")], []], EURO, SINCE_A, False),
    "private_debt": ("tipspd10", [[("unit", "PC_GDP")], []], EURO, SINCE_A, False),
    "deficit_debt_annual": (
        "gov_10dd_edpt1",
        [[("unit", "PC_GDP"), ("sector", "S13"), ("na_item", "B9"), ("na_item", "GD")]],
        EURO, SINCE_A, True,
    ),
    "growth_annual": ("nama_10_gdp", [[("unit", "CLV_PCH_PRE"), ("na_item", "B1GQ")]], EURO, SINCE_A, True),
    "unemployment": ("une_rt_a", [[("age", "Y15-74"), ("sex", "T"), ("unit", "PC_ACT")]], EURO, SINCE_A, False),
}

ECB_CODES = {"EL": "GR"}
ECB_AREAS = "+".join(ECB_CODES.get(c, c) for c in EURO)

# name -> (dataflow/key, start period, required)
ECB_SERIES: dict[str, tuple[str, str, bool]] = {
    "ecb_yields_monthly": (f"IRS/M.{ECB_AREAS}.L.L40.CI.0000.EUR.N.Z", "1995-01", True),
    "target_balances": (f"TGB/M.{ECB_AREAS}.N.A094T.U2.EUR.E", "2008-01", False),
    "euribor_3m": ("FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", "2005-01", False),
    "mro_rate": ("FM/B.U2.EUR.4F.KR.MRR_FR.LEV", "2005-01-01", False),
    "loan_rates": ("MIR/M.PT+ES+IT+IE+GR+DE.B.A2C+A2A.A.R.A.2250+2240.EUR.N+O", "2005-01", False),
    "housing_loans_pt": ("BSI/M.PT.N.A.A22.A.1.U2.2250.Z01.E", "2005-01", False),
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


def save(d: pd.DataFrame, out: Path, name: str) -> Path:
    path = out / (f"{name}.csv.gz" if len(d) > 50_000 else f"{name}.csv")
    d.to_csv(path, index=False, **({"compression": {"method": "gzip", "mtime": 0}} if path.suffix == ".gz" else {}))
    return path


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
        save(d, out, name)
        last = sorted(d["time"].unique())[-1] if "time" in d else "?"
        log.append(f"eurostat {name}\tROWS\t{len(d)}\tcolumns {list(d.columns)}\tlast period {last}\tfilters {filters}")
        return True
    log.append(f"eurostat {name}\tFAILED")
    return False


def ecb_series(name: str, key: str, start: str, out: Path, log: list[str]) -> bool:
    url = f"{ECB_API}/{key}?startPeriod={start}&format=csvdata"
    try:
        status, body = get(url, timeout=600)
    except RuntimeError as e:
        log.append(f"ecb {name}\tERROR\t{e}")
        return False
    log_response(log, f"ecb {name}", url, status, body)
    if status != 200 or len(body) < 200:
        log.append(f"ecb {name}\tBODY\t{body[:300]!r}")
        return False
    d = pd.read_csv(io.BytesIO(body), dtype=str)
    keep = [c for c in ("KEY", "REF_AREA", "COUNTERPART_AREA", "COUNTERPART_SECTOR", "TIME_PERIOD", "OBS_VALUE", "TITLE") if c in d]
    d = d[keep]
    save(d, out, name)
    log.append(f"ecb {name}\tROWS\t{len(d)}\tseries {d['KEY'].nunique() if 'KEY' in d else '?'}\tlast period {d['TIME_PERIOD'].max() if 'TIME_PERIOD' in d else '?'}")
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
    for name, (ds, fs, geos, extra, required) in EUROSTAT_REQUESTS.items():
        if want and name not in want:
            continue
        got = eurostat(name, ds, fs, geos, extra, out, log)
        ok &= got or not required
        time.sleep(1)
    for name, (key, start, required) in ECB_SERIES.items():
        if want and name not in want:
            continue
        got = ecb_series(name, key, start, out, log)
        ok &= got or not required
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
