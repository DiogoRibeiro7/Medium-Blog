"""Download the Eurostat price series used by ``ivazero.py``.

Monthly HICP indices by ECOICOP subclass, the same indices at constant tax
rates, and the annual item weights, for every EU member state. Each request
covers one dataset and one country; only the food codes (CP01...) and the
all-items index (CP00) are kept. Values are saved in long format with their
Eurostat status flag::

    python fetch.py --out data

Failed requests are logged to ``fetch_log.txt`` and the rest continue.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
HERE = Path(__file__).resolve().parent

GEOS = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE",
        "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
SINCE_MONTH = "2018-01"
SINCE_YEAR = "2018"

# output file -> (dataset, filters, time filter)
QUERIES: dict[str, tuple[str, dict[str, str], str]] = {
    # HICP, monthly index, 2015 = 100.
    "prc_hicp_midx": ("prc_hicp_midx", {"unit": "I15"}, SINCE_MONTH),
    # HICP at constant tax rates, monthly index, 2015 = 100.
    "prc_hicp_cind": ("prc_hicp_cind", {"unit": "I15"}, SINCE_MONTH),
    # HICP item weights, per mille, annual.
    "prc_hicp_inw": ("prc_hicp_inw", {}, SINCE_YEAR),
}


def keep(code: str) -> bool:
    return code == "CP00" or code.startswith("CP01")


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
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    records = []
    for pos in set(values) | set(status):
        n = int(pos)
        rec = {dim: cats[i][(n // strides[i]) % sizes[i]] for i, dim in enumerate(ids)}
        rec["value"] = values.get(pos)
        rec["flag"] = status.get(pos, "")
        records.append(rec)
    frame = pd.DataFrame.from_records(records, columns=ids + ["value", "flag"])
    frame = frame.rename(columns={"time": "period"})
    frame = frame.drop(columns=[c for c in ("freq",) if c in frame])
    dims = [c for c in frame.columns if c not in ("period", "value", "flag")]
    return frame.sort_values(dims + ["period"]).reset_index(drop=True)


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
    for name, (dataset, filters, since) in QUERIES.items():
        frames = []
        # A small request for one country and one period lists the codes; only the food codes are then requested.
        try:
            probe = download(dataset, {**filters, "geo": "PT", "time": "2023-01" if "-" in since else "2023"})
            codes = sorted(c for c in probe["coicop"].unique() if keep(c))
        except Exception as error:  # noqa: BLE001 (logged and reported)
            failures += 1
            log.append(f"FAIL {name} code list: {error!r}")
            continue
        for geo in GEOS:
            query = {**filters, "geo": geo, "coicop": codes, "sinceTimePeriod": since}
            try:
                frame = download(dataset, query)
            except Exception as error:  # noqa: BLE001 (logged and reported)
                failures += 1
                detail = error.read().decode("utf-8", "replace")[:200] if isinstance(error, urllib.error.HTTPError) else ""
                log.append(f"FAIL {name} {geo}: {error!r} {detail}".strip())
                continue
            frames.append(frame[frame["coicop"].map(keep)])
        if not frames:
            continue
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(args.out / f"{name}.csv", index=False)
        log.append(f"{name}: {len(out)} rows; geo {out['geo'].nunique()}, coicop {out['coicop'].nunique()}; "
                   f"periods {out['period'].min()}-{out['period'].max()}")
    (args.out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
