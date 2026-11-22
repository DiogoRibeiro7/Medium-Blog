"""Download the INE series used to place the EUR 2,300 "moderate rent" ceiling against real rents.

    python fetch.py --out data

One source, open and without a token: the INE (Statistics Portugal) indicator API,
``pindica.jsp?op=2``, all periods (``Dim1=T``). Tables published down to parish (freguesia)
level are requested only for Portugal, its NUTS regions and its 308 municipalities, in chunks of
geography codes (the API rejects long ``Dim2`` lists); the codes come from the first table.

Raw responses go to ``data/raw/`` unchanged (except the Census cross-tab, about 12 MB, whose raw
responses are not kept); the API returns records in a varying order, so raw files differ from one
refresh to the next even when no value changes. Tidy long tables go to ``data/<name>.csv`` sorted by
period, geography and category, so their diffs show only real changes. ``data/fetch_log.txt``
records the time, HTTP status, SHA-256 and URL of every response.

INE indicators:

- ``0014711``  rents of new residential leases, EUR per m2, 1st quartile, median and 3rd quartile,
               by municipality, annual, Metodologia 2026 (2020-2025)
- ``0013041``  the same, Metodologia 2021 (2020-2024)
- ``0009631``  median rent per m2 of new leases, Metodologia 2021, NUTS 2013 (2017-2023)
- ``0009632``  number of new leases, Metodologia 2021, NUTS 2013 (2017-2023)
- ``0014697``  number of new leases, annual, Metodologia 2026 (2020-2025)
- ``0014706``  median rent per m2 of new leases by dwelling type, quarterly, municipalities with
               more than 100,000 inhabitants, Metodologia 2026
- ``0014705``  number of new leases by dwelling type, same coverage
- ``0012712``  median gross declared income per fiscal household (EUR), annual
- ``0012673``  fiscal households by bracket of gross declared income, annual
- ``0012714``  gross declared income of fiscal households at the quintile boundaries (P20, P40, P60,
               P80), annual
- ``0012522``  Census 2021: mean monthly rent of rented main-residence dwellings (EUR)
- ``0012526``  Census 2021: rented main-residence dwellings by floor-area class and monthly-rent
               class (all socioeconomic groups)
- ``0014625``  Census 2021: main-residence dwellings by number of rooms and floor-area class
               (Portugal only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

INE = "https://www.ine.pt/ine/json_indicador/pindica.jsp"
UA = {"User-Agent": "moderate-rent-ceiling/1.0"}
CHUNK = 40  # geography codes per request

# name -> (indicator, geography scope, extra filters, keep raw): scope "nuts2024" = Portugal, NUTS
# regions and municipalities with NUTS 2024 codes; "nuts2013" = the same with NUTS 2013 codes;
# "census" = Portugal and municipalities with 4-digit codes; "pt" = Portugal only; None = every
# geography (tables that stop at municipalities).
SERIES: dict[str, tuple[str, str | None, dict[str, str], bool]] = {
    "rent_quartiles_2021": ("0013041", None, {}, True),
    "new_leases_nuts2013": ("0009632", "nuts2013", {}, True),
    "new_leases_2026": ("0014697", "nuts2024", {}, True),
    "rent_by_type_quarterly": ("0014706", None, {}, True),
    "leases_by_type_quarterly": ("0014705", None, {}, True),
    "income_median_household": ("0012712", "nuts2024", {}, True),
    "income_brackets_household": ("0012673", "nuts2024", {}, True),
    "income_quintiles_household": ("0012714", "nuts2024", {}, True),
    "census_mean_rent": ("0012522", "census", {}, True),
    "census_rooms_area": ("0014625", "pt", {}, True),
    "census_area_rent": ("0012526", "census", {"Dim4": "T"}, False),
}
TRANSIENT = ("jdbc", "conexao", "Connection", "timeout")


def url_for(indicator: str, filters: dict[str, str]) -> str:
    params = [("op", "2"), ("varcd", indicator), ("Dim1", "T")]
    params += sorted(filters.items())
    params.append(("lang", "PT"))
    return f"{INE}?{urllib.parse.urlencode(params, safe=',')}"


def get(url: str, tries: int = 6) -> tuple[int, bytes]:
    """GET with retries on network errors and on the API's transient database failures."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
            head = body[:2000].decode("utf-8", "replace")
            if '"Falso"' in head and any(t in head for t in TRANSIENT) and attempt < tries - 1:
                last = RuntimeError(head[:300])
                time.sleep(10 * (attempt + 1))
                continue
            return 200, body
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return e.code, e.read()
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def tidy(doc: list) -> pd.DataFrame:
    """One row per cell: period, geocode, name, the category columns, the value and its flag."""
    rows = []
    for period, cells in doc[0].get("Dados", {}).items():
        for c in cells:
            row = {"period": period, "geocod": c.get("geocod"), "geodsg": c.get("geodsg")}
            row.update({k: v for k, v in c.items() if k.startswith("dim_")})
            v = c.get("valor")
            row["value"] = float(v) if v not in (None, "", "x", "-") else None
            row["flag"] = c.get("sinal_conv", "")
            rows.append(row)
    return pd.DataFrame(rows)


SORT_KEYS = ["period", "geocod", "dim_3", "dim_4", "dim_5"]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a tidy table in a fixed row order: the API returns records in a varying order, and
    sorting keeps refreshed snapshots comparable line by line."""
    keys = [k for k in SORT_KEYS if k in df.columns]
    df.astype({k: str for k in keys}).sort_values(keys, kind="mergesort").to_csv(path, index=False)


class Fetcher:
    def __init__(self, out: Path):
        self.out = out
        self.log: list[str] = []
        self.failed = False

    def one(self, name: str, indicator: str, filters: dict[str, str], keep_raw: bool = True) -> pd.DataFrame | None:
        url = url_for(indicator, filters)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            status, body = get(url)
        except RuntimeError as e:
            self.failed = True
            self.log.append(f"{stamp}  ERR  {name}  {e}")
            return None
        sha = hashlib.sha256(body).hexdigest()
        if keep_raw:
            (self.out / "raw" / f"{name}.json").write_bytes(body)
        note = "" if keep_raw else " (raw not kept)"
        self.log.append(f"{stamp}  {status}  {sha}  {name}{note}  {url}")
        if status != 200:
            self.failed = True
            return None
        doc = json.loads(body)
        if not doc[0].get("Sucesso", {}).get("Verdadeiro"):
            self.failed = True
            self.log.append(f"    API reported failure: {doc[0].get('Sucesso')}")
            return None
        return tidy(doc)

    def chunked(self, name: str, indicator: str, codes: list[str], extra: dict[str, str],
                keep_raw: bool) -> pd.DataFrame | None:
        parts = []
        for i in range(0, len(codes), CHUNK):
            filters = {"Dim2": ",".join(codes[i:i + CHUNK]), **extra}
            df = self.one(f"{name}_part{i // CHUNK:02d}", indicator, filters, keep_raw)
            if df is None:
                return None
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def write_log(self) -> None:
        (self.out / "fetch_log.txt").write_text("\n".join(self.log) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out = Path(ap.parse_args().out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    f = Fetcher(out)

    quart = f.one("rent_quartiles_2026", "0014711", {})
    if quart is None:
        f.write_log()
        raise SystemExit("the quartile table failed; the other requests depend on its geography codes")
    write_csv(quart, out / "rent_quartiles_2026.csv")
    print(f"rent_quartiles_2026: {len(quart):,} rows")

    # The NUTS 2013 median table goes down to parishes: its full response (about 0.7 MB) also gives
    # the NUTS 2013 municipality codes; only municipalities and above are kept in the tidy table.
    old = f.one("median_nuts2013", "0009631", {})
    if old is not None:
        old = old[old["geocod"].astype(str).str.len() <= 7].reset_index(drop=True)
        write_csv(old, out / "median_nuts2013.csv")
        print(f"median_nuts2013: {len(old):,} rows")

    codes = sorted(set(quart["geocod"].astype(str)))
    scopes = {
        "nuts2024": codes,
        "nuts2013": sorted(set(old["geocod"].astype(str))) if old is not None else [],
        "census": ["PT"] + sorted({c[-4:] for c in codes if len(c) == 7}),
        "pt": ["PT"],
    }

    for name, (indicator, scope, extra, keep_raw) in SERIES.items():
        if scope is None:
            df = f.one(name, indicator, extra, keep_raw)
        else:
            df = f.chunked(name, indicator, scopes[scope], extra, keep_raw)
        if df is not None:
            write_csv(df, out / f"{name}.csv")
            print(f"{name}: {len(df):,} rows")

    f.write_log()
    if f.failed:
        raise SystemExit("at least one request failed; see data/fetch_log.txt")


if __name__ == "__main__":
    main()
