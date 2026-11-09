"""Download the public series behind the ECB backstop and France analysis.

    python fetch.py --out data [--only name ...]

Sources:

1. Eurostat (JSON-stat API): monthly long-term government bond yields (EMU convergence
   criterion); quarterly government debt, net lending and interest; quarterly GDP at current
   prices and real growth; the quarterly current account; annual government accounts and GDP;
   gross national income for the EU-27.
2. ECB Data Portal: monthly convergence-criterion yields (IRS), TARGET balances of the national
   central banks (TGB) and government debt by holding sector (GFS).
3. ECB website: the history of public-sector holdings by jurisdiction under the PEPP and the PSPP
   (CSV files linked from the programme pages; links are read from the pages, not hard-coded).
4. EBA: the 2025 EU-wide transparency exercise, sovereign exposures and capital bank by bank (the files are
   located from links on the exercise page; the exercise was discontinued after June 2025, so
   June 2025 is the last reference date).

Every response is logged with URL, HTTP status and SHA-256 in ``data/fetch_log.txt``. Requests for
the channel data (ECB website files, GFS, EBA) are optional: a failure is logged but does not fail
the run, and the analysis reports what is missing.
"""

from __future__ import annotations

import argparse
import gzip
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
AGENT = "ecb-france-backstop/1.0 (research; python urllib)"

EURO = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
EU27 = ("AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK", "EU27_2020")
Filters = list[tuple[str, str]]
SINCE_Q = [("sinceTimePeriod", "1999-Q1")]

# name -> (dataset, filter sets tried in order, geographies, extra query parameters, required)
EUROSTAT_REQUESTS: dict[str, tuple[str, list[Filters], tuple[str, ...], Filters, bool]] = {
    "yields_monthly": ("irt_lt_mcby_m", [[], [("int_rt", "MCBY")]], EURO, [], True),
    "gov_main": (
        "gov_10a_main",
        [
            [("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "D41PAY"), ("na_item", "B9"), ("na_item", "TE"), ("na_item", "TR")],
            [("unit", "MIO_EUR"), ("sector", "S13")],
        ],
        EURO, [], True,
    ),
    "debt": ("gov_10dd_edpt1", [[("unit", "MIO_EUR"), ("sector", "S13"), ("na_item", "GD")], [("unit", "MIO_EUR"), ("sector", "S13")]], EURO, [], True),
    "gdp": ("nama_10_gdp", [[("unit", "CP_MEUR"), ("na_item", "B1GQ")]], EURO, [], True),
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
    "gni": (
        "nasa_10_nf_tr",
        [
            [("unit", "CP_MEUR"), ("na_item", "B5GQ"), ("sector", "S1"), ("direct", "PAID")],
            [("unit", "CP_MEUR"), ("na_item", "B5GQ"), ("sector", "S1"), ("direct", "RECV")],
            [("unit", "CP_MEUR"), ("na_item", "B5GQ"), ("sector", "S1")],
        ],
        EU27, [("sinceTimePeriod", "2015")], False,
    ),
    "gdp_eu27": ("nama_10_gdp", [[("unit", "CP_MEUR"), ("na_item", "B1GQ")]], EU27, [("sinceTimePeriod", "2015")], False),
}

ECB_CODES = {"EL": "GR"}
ECB_AREAS = "+".join(ECB_CODES.get(c, c) for c in EURO)

# name -> (dataflow/key, start period, required)
ECB_SERIES: dict[str, tuple[str, str, bool]] = {
    "ecb_yields_monthly": (f"IRS/M.{ECB_AREAS}.L.L40.CI.0000.EUR.N.Z", "1995-01", True),
    "target_balances": (f"TGB/M.{ECB_AREAS}.N.A094T.U2.EUR.E", "2008-01", False),
    "gfs_debt_holders": (f"GFS/Q.N.FR+IT+ES+DE+BE..S13..C.L.LE.GD.T._Z.XDC._T.F.V.N._T", "2008-Q1", False),
}

PAGES = {
    "pepp": "https://www.ecb.europa.eu/mopo/implement/pepp/html/index.en.html",
    "app": "https://www.ecb.europa.eu/mopo/implement/app/html/index.en.html",
    "eba": "https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/eu-wide-transparency-exercise",
    "eba_node": "https://www.eba.europa.eu/node/18852",
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


def links(page: str, log: list[str], tag: str) -> list[str]:
    try:
        status, body = get(page)
    except RuntimeError as e:
        log.append(f"{tag} page\tERROR\t{e}")
        return []
    log_response(log, f"{tag} page", page, status, body)
    if status != 200:
        return []
    html = body.decode("utf-8", errors="replace")
    found = sorted(set(re.findall(r'href=["\']([^"\']+\.(?:csv|xlsx|xlsb|zip)(?:\?[^"\']*)?)["\']', html, flags=re.I)))
    full = [urllib.parse.urljoin(page, h) for h in found]
    for u in full:
        log.append(f"{tag} link\t{u[:300]}")
    return full


def download(url: str, dest: Path, log: list[str], tag: str) -> bool:
    try:
        status, body = get(url, timeout=600)
    except RuntimeError as e:
        log.append(f"{tag}\tERROR\t{e}")
        return False
    log_response(log, tag, url, status, body)
    if status != 200 or len(body) < 100:
        return False
    if len(body) > 2_000_000 and not dest.name.endswith(".gz"):
        dest = dest.with_name(dest.name + ".gz")
        body = gzip.compress(body, mtime=0)
    dest.write_bytes(body)
    log.append(f"{tag}\tSAVED\t{dest.name}\t{len(body)} bytes")
    return True


def ecb_programmes(out: Path, log: list[str]) -> bool:
    """PEPP and PSPP public-sector holdings by jurisdiction, from the CSVs linked on the ECB pages."""
    ok = False
    for prog in ("pepp", "app"):
        for u in links(PAGES[prog], log, f"ecb {prog}"):
            fname = Path(urllib.parse.urlparse(u).path).name
            low = fname.lower()
            if ("public_sector" in low and "breakdown" in low) or ("pspp" in low and "breakdown" in low) or ("pspp" in low and "jurisdiction" in low):
                ok |= download(u, out / f"ecb_{fname}", log, f"ecb {prog} file")
    return ok


def eba_sovereign(out: Path, log: list[str]) -> bool:
    """EBA 2025 transparency exercise: sovereign exposures and capital (bank by bank), located from page links."""
    seen: list[str] = []
    for tag in ("eba", "eba_node"):
        seen += links(PAGES[tag], log, tag)
    ok = False
    for u in dict.fromkeys(seen):
        fname = Path(urllib.parse.urlparse(u).path).name
        low = fname.lower()
        if ("sov" in low or "oth" in low or "metadata" in low) and low.endswith((".csv", ".xlsx", ".zip")):
            ok |= download(u, out / f"eba_{fname}", log, "eba file")
    return ok


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
    if not want or "programmes" in want:
        ecb_programmes(out, log)
    if not want or "eba" in want:
        eba_sovereign(out, log)
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
