"""Download the ten test series named in PREREGISTRATION.md.

    python fetch.py --out data

Each series has a primary source and at most one fallback, tried in that order; the source that
answered is recorded. Writes ``data/international_monthly.csv`` (date plus one column per series,
levels as published), ``data/sources.json`` and one line per request in ``data/fetch_log.txt``
(UTC time, URL, SHA-256 of the response, rows or error).

The development set (ten US series) is read from ``../backward-evaluation-pooling/data``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
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

# key: [(source, identifier), ...] in the order of PREREGISTRATION.md
SERIES = {
    "JP_UNEMP": [("fred", "LRHUTTTTJPM156S"), ("fred", "LRUNTTTTJPM156S")],
    "CA_UNEMP": [("fred", "LRHUTTTTCAM156S"), ("statcan", "2062815")],
    "UK_IP": [("ons", "/economy/economicoutputandproductivity/output/timeseries/k222/diop"),
              ("fred", "GBRPROINDMISMEI")],
    "DE_IP": [("fred", "DEUPROINDMISMEI")],
    "JP_IP": [("fred", "JPNPROINDMISMEI"), ("dbnomics", "OECD/MEI/JPN.PRINTO01.IXOBSA.M")],
    "UK_10Y": [("fred", "IRLTLT01GBM156N")],
    "DE_10Y": [("fred", "IRLTLT01DEM156N")],
    "CA_10Y": [("fred", "IRLTLT01CAM156N")],
    "JPY_USD": [("fred", "EXJPUS")],
    "CHF_USD": [("fred", "EXSZUS")],
}


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


def log(out: Path, url: str, body: bytes | None, note: object) -> None:
    sha = hashlib.sha256(body).hexdigest() if body is not None else "-"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{stamp}\t{url}\t{sha}\t{note}\n")


def monthly(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(idx)).dt.to_period("M").dt.to_timestamp())


def from_fred(sid: str) -> tuple[str, bytes, pd.Series]:
    url = FREDGRAPH.format(sid=sid)
    body = get(url)
    df = pd.read_csv(io.BytesIO(body))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=monthly(df.iloc[:, 0]))
    return url, body, s


def from_dbnomics(code: str) -> tuple[str, bytes, pd.Series]:
    url = DBNOMICS.format(code=code)
    body = get(url)
    doc = json.loads(body)["series"]["docs"][0]
    s = pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").to_numpy(), index=monthly(doc["period"]))
    return url, body, s


def from_statcan(vector: str) -> tuple[str, bytes, pd.Series]:
    payload = json.dumps([{"vectorId": int(vector), "latestN": 1500}]).encode()
    body = get(STATCAN, data=payload)
    pts = json.loads(body)[0]["object"]["vectorDataPoint"]
    s = pd.Series([p.get("value") for p in pts], index=monthly([p["refPer"] for p in pts]), dtype=float)
    return f"{STATCAN} vector {vector}", body, s


def from_ons(uri: str) -> tuple[str, bytes, pd.Series]:
    url = ONS.format(uri=uri)
    body = get(url)
    dates, vals = [], []
    for ln in body.decode("utf-8", errors="replace").splitlines():
        cells = [c.strip().strip('"') for c in ln.split(",")]
        if len(cells) < 2:
            continue
        parts = cells[0].split()
        if len(parts) == 2 and parts[0].isdigit() and len(parts[1]) == 3 and not parts[1].startswith("Q"):
            dates.append(f"{parts[0]} {parts[1].title()}")
            vals.append(cells[1])
    idx = pd.DatetimeIndex(pd.to_datetime(dates, format="%Y %b"))
    return url, body, pd.Series(pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(), index=idx)


FETCH = {"fred": from_fred, "dbnomics": from_dbnomics, "statcan": from_statcan, "ons": from_ons}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols: dict[str, pd.Series] = {}
    source: dict[str, str] = {}
    for key, options in SERIES.items():
        for src, ident in options:
            try:
                url, body, s = FETCH[src](ident)
                s = s.dropna().sort_index()
                s = s[~s.index.duplicated(keep="last")]
                if len(s) < 120:
                    raise ValueError(f"only {len(s)} rows")
                log(out, url, body, f"{key}: {len(s)} rows")
                cols[key] = s.rename(key)
                source[key] = f"{src}:{ident}"
                break
            except Exception as e:  # noqa: BLE001 - logged, next source tried
                log(out, f"{src}:{ident}", None, f"{key}: ERROR {e}")
    missing = sorted(set(SERIES) - set(cols))
    panel = pd.concat(cols.values(), axis=1).sort_index()
    panel.index.name = "date"
    panel.to_csv(out / "international_monthly.csv", float_format="%.6g")
    (out / "sources.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'international_monthly.csv'}: {len(panel)} months; sources {source}")
    if missing:
        print(f"missing series (dropped as pre-registered): {missing}", file=sys.stderr)
    return 0 if len(cols) >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
