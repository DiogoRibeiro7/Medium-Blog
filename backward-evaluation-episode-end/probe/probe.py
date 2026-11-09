"""Availability probe for candidate international test series (run before the pre-registration).

    python probe/probe.py --out probe

For each candidate (series, source) pair this records only metadata: whether the source
answered, the first and last month, the number of monthly observations, the median spacing
between observations in days, and the SHA-256 of the response. **No values are written.**
The probe exists so that the pre-registration can name series that are actually available,
without anyone looking at the data the test will use.

Output: ``probe/probe_log.tsv`` (one line per candidate).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HEADERS = {"User-Agent": "backward-evaluation-episode-end/1.0 (research code)",
           "Accept": "text/csv,application/json,*/*"}
FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
DBNOMICS = "https://api.db.nomics.world/v22/series/{code}?observations=1&format=json"
STATCAN = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"
ONS = "https://www.ons.gov.uk/generator?format=csv&uri={uri}"

# (label, source, identifier)
CANDIDATES = [
    # Japan
    ("Japan industrial production", "fred", "JPNPROINDMISMEI"),
    ("Japan industrial production", "dbnomics", "IMF/IFS/M.JP.AIP_IX"),
    ("Japan industrial production", "dbnomics", "OECD/MEI/JPN.PRINTO01.IXOBSA.M"),
    ("Japan CPI", "fred", "JPNCPIALLMINMEI"),
    ("Japan CPI", "fred", "CPALTT01JPM661N"),
    ("Japan CPI", "dbnomics", "IMF/IFS/M.JP.PCPI_IX"),
    ("Japan unemployment rate", "fred", "LRUNTTTTJPM156S"),
    ("Japan unemployment rate", "fred", "LRHUTTTTJPM156S"),
    ("Japan 10-year yield", "fred", "IRLTLT01JPM156N"),
    ("Yen per US dollar", "fred", "EXJPUS"),
    # United Kingdom
    ("UK unemployment rate", "fred", "LRHUTTTTGBM156S"),
    ("UK unemployment rate", "fred", "LMUNRRTTGBM156S"),
    ("UK unemployment rate", "ons", "/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms"),
    ("UK industrial production", "fred", "GBRPROINDMISMEI"),
    ("UK industrial production", "ons", "/economy/economicoutputandproductivity/output/timeseries/k222/diop"),
    ("UK CPI", "fred", "GBRCPIALLMINMEI"),
    ("UK CPI", "fred", "CPALTT01GBM661N"),
    ("UK CPI", "ons", "/economy/inflationandpriceindices/timeseries/d7bt/mm23"),
    ("UK 10-year yield", "fred", "IRLTLT01GBM156N"),
    ("US dollars per pound", "fred", "EXUSUK"),
    # Canada
    ("Canada unemployment rate", "fred", "LRUNTTTTCAM156S"),
    ("Canada unemployment rate", "fred", "LRHUTTTTCAM156S"),
    ("Canada unemployment rate", "statcan", "2062815"),
    ("Canada industrial production", "fred", "CANPROINDMISMEI"),
    ("Canada CPI", "fred", "CANCPIALLMINMEI"),
    ("Canada CPI", "fred", "CPALTT01CAM661N"),
    ("Canada CPI", "statcan", "41690973"),
    ("Canada 10-year yield", "fred", "IRLTLT01CAM156N"),
    ("Canadian dollars per US dollar", "fred", "EXCAUS"),
    # Germany and the euro area
    ("Germany industrial production", "fred", "DEUPROINDMISMEI"),
    ("Germany industrial production", "dbnomics", "IMF/IFS/M.DE.AIP_IX"),
    ("Germany unemployment rate", "fred", "LRHUTTTTDEM156S"),
    ("Germany CPI", "fred", "DEUCPIALLMINMEI"),
    ("Germany CPI", "fred", "CPALTT01DEM661N"),
    ("Germany 10-year yield", "fred", "IRLTLT01DEM156N"),
    ("Euro area unemployment rate", "fred", "LRHUTTTTEZM156S"),
    ("Euro area HICP", "fred", "CP0000EZ19M086NEST"),
    # Others with long monthly histories
    ("France industrial production", "fred", "FRAPROINDMISMEI"),
    ("France CPI", "fred", "FRACPIALLMINMEI"),
    ("Italy industrial production", "fred", "ITAPROINDMISMEI"),
    ("Swiss francs per US dollar", "fred", "EXSZUS"),
    ("Australia unemployment rate", "fred", "LRUNTTTTAUM156S"),
    ("Korea industrial production", "fred", "KORPROINDMISMEI"),
]


def get(url: str, data: bytes | None = None, tries: int = 2, timeout: int = 45) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            headers = dict(HEADERS)
            if data is not None:
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} tries: {last}")


def dates_fred(sid: str) -> tuple[str, bytes, pd.DatetimeIndex]:
    url = FREDGRAPH.format(sid=sid)
    body = get(url)
    df = pd.read_csv(io.BytesIO(body))
    ok = pd.to_numeric(df.iloc[:, 1], errors="coerce").notna()
    return url, body, pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0][ok]))


def dates_dbnomics(code: str) -> tuple[str, bytes, pd.DatetimeIndex]:
    url = DBNOMICS.format(code=code)
    body = get(url)
    doc = json.loads(body)["series"]["docs"][0]
    per = pd.Series(doc["period"])
    ok = pd.to_numeric(pd.Series(doc["value"]), errors="coerce").notna()
    return url, body, pd.DatetimeIndex(pd.to_datetime(per[ok]))


def dates_statcan(vector: str) -> tuple[str, bytes, pd.DatetimeIndex]:
    payload = json.dumps([{"vectorId": int(vector), "latestN": 1500}]).encode()
    body = get(STATCAN, data=payload)
    pts = json.loads(body)[0]["object"]["vectorDataPoint"]
    ok = [p["refPer"] for p in pts if p.get("value") is not None]
    return f"{STATCAN} vector {vector}", body, pd.DatetimeIndex(pd.to_datetime(ok))


def dates_ons(uri: str) -> tuple[str, bytes, pd.DatetimeIndex]:
    url = ONS.format(uri=uri)
    body = get(url)
    lines = body.decode("utf-8", errors="replace").splitlines()
    months = []
    for ln in lines:
        head = ln.split(",")[0].strip('"')
        parts = head.split()
        if len(parts) == 2 and parts[0].isdigit() and len(parts[1]) == 3:
            months.append(f"{parts[0]} {parts[1]}")
    return url, body, pd.DatetimeIndex(pd.to_datetime(months, format="%Y %b"))


FETCH = {"fred": dates_fred, "dbnomics": dates_dbnomics, "statcan": dates_statcan, "ons": dates_ons}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, src, ident in CANDIDATES:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            url, body, d = FETCH[src](ident)
            d = d.sort_values()
            spacing = float(pd.Series(d).diff().dt.days.median()) if len(d) > 1 else float("nan")
            rows.append([stamp, label, src, ident, "ok", f"{d.min():%Y-%m}", f"{d.max():%Y-%m}", len(d),
                         f"{spacing:.0f}", hashlib.sha256(body).hexdigest()])
        except Exception as e:  # noqa: BLE001 - every failure is recorded and the probe moves on
            rows.append([stamp, label, src, ident, f"error: {str(e)[:120]}", "", "", 0, "", ""])
    cols = ["utc", "label", "source", "id", "status", "first", "last", "n", "median_days", "sha256"]
    pd.DataFrame(rows, columns=cols).to_csv(out / "probe_log.tsv", sep="\t", index=False)
    ok = sum(r[4] == "ok" for r in rows)
    print(f"{ok} of {len(rows)} candidates answered; wrote {out / 'probe_log.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
