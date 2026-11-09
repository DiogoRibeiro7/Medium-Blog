"""Download the public series behind the Portugal debt-management analysis.

    python fetch.py --out data [--only name ...]

Sources:

1. Eurostat (JSON-stat API): monthly long-term government bond yields (EMU convergence
   criterion) for Portugal and Germany; quarterly government interest payable and debt.
2. ECB Data Portal: three-month Euribor; euro-area AAA government yield curve (spot rates).
3. FRED: three-month rates for the dollar, yen, pound and renminbi legs of the SDR interest rate
   (the IMF's own pages reject automated requests, so the rate is rebuilt from its basket).
4. Banco de Portugal BPstat API: every dataset of the budget-outturn, interest-rate,
   money-market, secondary-market, general-government financing and debt domains (raw
   JSON-stat in ``data/bpstat``).

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
BPSTAT = "https://bpstat.bportugal.pt/data/v1"
AGENT = "portugal-debt-management/1.0 (research; python urllib)"

Filters = list[tuple[str, str]]
EUROSTAT_REQUESTS: dict[str, tuple[str, list[Filters], tuple[str, ...], Filters, bool]] = {
    "yields_monthly": ("irt_lt_mcby_m", [[], [("int_rt", "MCBY")]], ("PT", "DE"), [], True),
    "gov_quarterly": (
        "gov_10q_ggnfa",
        [[("unit", "MIO_EUR"), ("sector", "S13"), ("s_adj", "NSA"), ("na_item", "D41PAY")],
         [("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "D41PAY")]],
        ("PT",), [("sinceTimePeriod", "2005-Q1")], True,
    ),
    "debt_quarterly": ("gov_10q_ggdebt", [[("unit", "MIO_EUR"), ("sector", "S13")]], ("PT",), [("sinceTimePeriod", "2005-Q1")], True),
}

ECB_SERIES: dict[str, tuple[str, str, bool]] = {
    "euribor_3m": ("FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", "2005-01", True),
    "euribor_12m": ("FM/M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA", "2005-01", False),
    # euro-area AAA government spot rates, daily (the euro leg of the SDR interest rate is the 3-month rate)
    "aaa_curve_daily": ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M+SR_1Y+SR_5Y+SR_10Y", "2010-01-01", False),
    "exr_monthly": ("EXR/M.USD+JPY+GBP+CNY.EUR.SP00.A", "2010-01", True),
}

# FRED (fredgraph CSV, no key): the other legs of the SDR interest rate
FRED_SERIES = {
    "fred_rates": ("TB3MS", "IR3TIB01JPM156N", "IR3TTS01GBM156N", "IR3TIB01CNM156N", "INTGSTJPM193N", "IR3TIB01GBM156N"),
}

# BPstat domains whose datasets are downloaded in full (JSON-stat): budget outturn, interest
# rates, money market, secondary market, general government financing and debt
BPSTAT_DOMAINS = (10, 21, 22, 26, 27, 28)



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




def fred(name: str, ids: tuple[str, ...], out: Path, log: list[str]) -> bool:
    frames = []
    for sid in ids:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            status, body = get(url, tries=2, timeout=120)
        except RuntimeError as e:
            log.append(f"fred {sid}\tERROR\t{e}")
            continue
        log_response(log, f"fred {sid}", url, status, body)
        if status != 200 or not body.lstrip().startswith(b"observation_date") and not body.lstrip().startswith(b"DATE"):
            log.append(f"fred {sid}\tBODY\t{body[:200]!r}")
            continue
        d = pd.read_csv(io.BytesIO(body))
        d.columns = ["date", "value"]
        d["series"] = sid
        frames.append(d)
        time.sleep(1)
    if not frames:
        return False
    d = pd.concat(frames, ignore_index=True)
    save(d, out, name)
    log.append(f"fred {name}\tROWS\t{len(d)}\tseries {d['series'].nunique()}")
    return True


def bpstat_datasets(out: Path, log: list[str]) -> None:
    """Every dataset of the chosen BPstat domains, raw JSON-stat plus a long CSV with series labels."""
    bp = out / "bpstat"
    bp.mkdir(exist_ok=True)
    for did in BPSTAT_DOMAINS:
        url = f"{BPSTAT}/domains/{did}/datasets/?lang=EN"
        try:
            status, body = get(url, tries=2, timeout=120)
        except RuntimeError as e:
            log.append(f"bpstat datasets {did}\tERROR\t{e}")
            continue
        log_response(log, f"bpstat datasets {did}", url, status, body)
        if status != 200:
            continue
        for item in json.loads(body).get("link", {}).get("item", []):
            xid = item.get("extension", {}).get("id")
            if not xid:
                continue
            u = f"{BPSTAT}/domains/{did}/datasets/{xid}/?lang=EN"
            try:
                status, body = get(u, tries=3, timeout=300)
            except RuntimeError as e:
                log.append(f"bpstat dataset {did}/{xid}\tERROR\t{e}")
                continue
            log_response(log, f"bpstat dataset {did}/{xid}", u, status, body)
            if status != 200:
                continue
            (bp / f"{did}_{xid}.json").write_bytes(body)
            time.sleep(0.5)


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
    for name, ids in FRED_SERIES.items():
        if want and name not in want:
            continue
        fred(name, ids, out, log)
    if not want or "bpstat" in want:
        try:
            bpstat_datasets(out, log)
        except Exception as e:  # noqa: BLE001
            log.append(f"bpstat\tERROR\t{type(e).__name__}: {str(e)[:300]}")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
