"""Download the public series behind the Portugal debt-management analysis.

    python fetch.py --out data [--only name ...]

Sources:

1. Eurostat (JSON-stat API): monthly long-term government bond yields (EMU convergence
   criterion) for Portugal and Germany; quarterly government interest payable and debt.
2. ECB Data Portal: three-month Euribor; euro-area AAA government yield curve (spot rates).
3. IMF: the weekly SDR interest rate and SDR exchange rates (IMF SDR pages), and Portugal's
   transactions with the Fund (member key 810): purchases, repurchases and charges paid.
4. Banco de Portugal BPstat API: the State's direct debt by instrument and creditor, retail
   savings instruments, and Portuguese government yields.
5. IGCP: pages listing statistics and bulletins (discovery).

Every response is logged with URL, HTTP status and SHA-256 in ``data/fetch_log.txt``. Requests
marked optional are logged on failure but do not fail the run. The ``discovery`` group saves raw
catalogues (BPstat domains, datasets and series; IMF and IGCP pages) under ``data/discovery`` so
that series identifiers can be chosen and then fixed in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
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
    "aaa_curve": ("YC/M.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y+SR_2Y+SR_3Y+SR_5Y+SR_7Y+SR_10Y+SR_15Y+SR_20Y+SR_30Y", "2005-01", False),
}

# name -> (url, required)
RAW: dict[str, tuple[str, bool]] = {
    "imf_sdr_rate_page": ("https://www.imf.org/external/np/fin/data/sdr_ir.aspx", False),
    "imf_pt_transactions": ("https://www.imf.org/external/np/fin/tad/extrans1.aspx?memberKey1=810&endDate=2099-12-31&finposition_flag=YES", False),
    "imf_pt_transactions_tsv": ("https://www.imf.org/external/np/fin/tad/extrans1.aspx?memberKey1=810&endDate=2099-12-31&finposition_flag=YES&tsvflag=Y", False),
    "imf_pt_position_2014": ("https://www.imf.org/external/np/fin/tad/exfin2.aspx?memberKey1=810&date1key=2014-04-30", False),
    "imf_pt_position_2016": ("https://www.imf.org/external/np/fin/tad/exfin2.aspx?memberKey1=810&date1key=2016-12-31", False),
    "igcp_home_pt": ("https://www.igcp.pt/pt/", False),
    "igcp_home_en": ("https://www.igcp.pt/en/", False),
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




def raw(name: str, url: str, out: Path, log: list[str]) -> bool:
    try:
        status, body = get(url, tries=2, timeout=120)
    except RuntimeError as e:
        log.append(f"raw {name}\tERROR\t{e}")
        return False
    log_response(log, f"raw {name}", url, status, body)
    if status != 200:
        return False
    ext = ".html" if body.lstrip()[:1] == b"<" else ".txt"
    (out / f"{name}{ext}").write_bytes(body)
    return True


def imf_sdr(out: Path, log: list[str]) -> None:
    """SDR interest and exchange rates through the imf-reader package (it parses the IMF pages)."""
    try:
        from imf_reader import sdr  # type: ignore
    except ImportError as e:
        log.append(f"imf sdr\tIMPORT\t{e}")
        return
    for name, fn in (("sdr_interest", lambda: sdr.fetch_interest_rates()),
                     ("sdr_exchange_usd", lambda: sdr.fetch_exchange_rates("USD")),
                     ("sdr_exchange_sdr", lambda: sdr.fetch_exchange_rates())):
        try:
            d = fn()
            save(d, out, name)
            log.append(f"imf {name}\tROWS\t{len(d)}\tcolumns {list(d.columns)}")
        except Exception as e:  # noqa: BLE001
            log.append(f"imf {name}\tERROR\t{type(e).__name__}: {str(e)[:200]}")


def bpstat_discovery(out: Path, log: list[str]) -> None:
    disc = out / "discovery"
    disc.mkdir(exist_ok=True)
    status, body = get(f"{BPSTAT}/domains/?lang=EN", tries=2, timeout=120)
    log_response(log, "bpstat domains", f"{BPSTAT}/domains/?lang=EN", status, body)
    if status != 200:
        return
    (disc / "bpstat_domains.json").write_bytes(body)
    domains = json.loads(body)
    items = domains if isinstance(domains, list) else domains.get("results", domains.get("items", []))
    keep = re.compile(r"government|public|debt|interest|securit|financial market|savings|treasury", re.I)
    for d in items:
        label = " ".join(str(d.get(k, "")) for k in ("label", "name", "description"))
        if not keep.search(label):
            continue
        did = d.get("id")
        url = f"{BPSTAT}/domains/{did}/datasets/?lang=EN"
        status, body = get(url, tries=2, timeout=120)
        log_response(log, f"bpstat datasets {did}", url, status, body)
        if status != 200:
            continue
        (disc / f"bpstat_domain_{did}_datasets.json").write_bytes(body)
        ds = json.loads(body)
        ds_items = ds if isinstance(ds, list) else ds.get("results", ds.get("items", ds.get("link", {}).get("item", [])))
        for x in ds_items if isinstance(ds_items, list) else []:
            xid = x.get("id") or str(x.get("href", "")).rstrip("/").split("/")[-1]
            for kind, u in (("series", f"{BPSTAT}/series/?lang=EN&domain_id={did}&dataset_id={xid}&page_size=500"),
                            ("dataset", f"{BPSTAT}/domains/{did}/datasets/{xid}/?lang=EN&last_n=1")):
                try:
                    status, body = get(u, tries=2, timeout=120)
                except RuntimeError as e:
                    log.append(f"bpstat {kind} {did}/{xid}\tERROR\t{e}")
                    continue
                log_response(log, f"bpstat {kind} {did}/{xid}", u, status, body)
                if status == 200:
                    (disc / f"bpstat_{kind}_{did}_{xid}.json").write_bytes(body)
                time.sleep(0.3)


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
    disc = out / "discovery"
    disc.mkdir(exist_ok=True)
    for name, (url, required) in RAW.items():
        if want and name not in want:
            continue
        got = raw(name, url, disc, log)
        ok &= got or not required
    if not want or "imf_sdr" in want:
        imf_sdr(out, log)
    if not want or "bpstat_discovery" in want:
        try:
            bpstat_discovery(out, log)
        except Exception as e:  # noqa: BLE001
            log.append(f"bpstat discovery\tERROR\t{type(e).__name__}: {str(e)[:300]}")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
