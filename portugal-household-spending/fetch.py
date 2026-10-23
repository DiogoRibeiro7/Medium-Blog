"""Download the Eurostat series used by ``spending.py`` for Portugal.

Each dataset is requested from the Eurostat statistics API (JSON-stat) with the
filters below and saved in long format, one CSV per dataset::

    python fetch.py --out data

Each value keeps its Eurostat status flag (p = provisional, e = estimated, u =
low reliability, ...) in a separate column. Failed requests are logged to
``fetch_log.txt`` and the rest of the download continues.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from itertools import product
from pathlib import Path

import pandas as pd

API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
HERE = Path(__file__).resolve().parent

# dataset -> filters (lists are sent as repeated parameters)
QUERIES: dict[str, dict[str, str | list[str]]] = {
    # Household final consumption by COICOP purpose, current prices, annual.
    "nama_10_co3_p3": {"geo": "PT", "unit": "CP_MEUR"},
    # Household key indicators: saving rate, investment rate, ...
    "nasa_10_ki": {"geo": "PT"},
    # Household budget survey: structure of consumption by COICOP (per mille).
    "hbs_str_t211": {"geo": "PT"},
    # Household budget survey: structure by income quintile.
    "hbs_str_t223": {"geo": "PT"},
    # Household budget survey: mean consumption expenditure by income quintile.
    "hbs_exp_t133": {"geo": "PT"},
    # Experimental: median saving rate by income quintile.
    "icw_sr_03": {"geo": "PT"},
    # HICP annual average indices by COICOP.
    "prc_hicp_aind": {"geo": "PT", "unit": "INX_A_AVG"},
}


def parse_jsonstat(payload: dict) -> pd.DataFrame:
    """Turn a Eurostat JSON-stat 2.0 response into a long DataFrame."""
    ids, sizes = payload["id"], payload["size"]
    cats = []
    for dim in ids:
        index = payload["dimension"][dim]["category"]["index"]
        ordered = sorted(index, key=index.get) if isinstance(index, dict) else list(index)
        cats.append(ordered)
    values = payload.get("value", {})
    status = payload.get("status", {})
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    records = []
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    for pos in set(values) | set(status):
        n = int(pos)
        coords = [(n // strides[i]) % sizes[i] for i in range(len(sizes))]
        rec = {dim: cats[i][coords[i]] for i, dim in enumerate(ids)}
        rec["value"] = values.get(pos)
        rec["flag"] = status.get(pos, "")
        records.append(rec)
    frame = pd.DataFrame.from_records(records, columns=ids + ["value", "flag"])
    frame = frame.rename(columns={"time": "year"})
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    dims = [c for c in frame.columns if c not in ("year", "value", "flag", "freq")]
    return frame.drop(columns=[c for c in ("freq",) if c in frame]).sort_values(dims + ["year"]).reset_index(drop=True)


def download(dataset: str, filters: dict, attempts: int = 4) -> pd.DataFrame:
    params = [("format", "JSON"), ("lang", "EN")]
    for key, value in filters.items():
        for v in value if isinstance(value, list) else [value]:
            params.append((key, v))
    url = f"{API}/{dataset}?{urllib.parse.urlencode(params)}"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 (fixed https host)
                return parse_jsonstat(json.loads(response.read().decode("utf-8")))
        except (urllib.error.URLError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                raise
            if attempt == attempts:
                raise
            time.sleep(20 * attempt)
    raise RuntimeError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log, failures = [], 0
    for dataset, filters in QUERIES.items():
        try:
            frame = download(dataset, filters)
        except Exception as error:  # noqa: BLE001 (logged and reported)
            failures += 1
            detail = error.read().decode("utf-8", "replace")[:300] if isinstance(error, urllib.error.HTTPError) else ""
            log.append(f"FAIL {dataset} {filters}: {error!r} {detail}".strip())
            continue
        frame.to_csv(args.out / f"{dataset}.csv", index=False)
        dims = [c for c in frame.columns if c not in ("year", "value", "flag")]
        summary = ", ".join(f"{d} {frame[d].nunique()}" for d in dims)
        log.append(f"{dataset}: {len(frame)} rows; {summary}; years {frame['year'].min()}-{frame['year'].max()}")
    (args.out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
