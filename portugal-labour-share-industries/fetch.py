"""Download the Eurostat industry accounts used by ``industries.py``.

Every branch of the A10 and A64 breakdowns is downloaded (the industry
dimension is left empty in the query), for Portugal, Spain, Italy and Greece::

    python fetch.py --out data

Values are current prices (million euro) or thousands of persons and hours.
Each value keeps its Eurostat flag (p = provisional, e = estimated,
c = confidential, ...) in a separate column.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
START = 1995
GEOS = "PT+ES+IT+EL"
HERE = Path(__file__).resolve().parent

# dataset -> list of (file stem, SDMX key). The nace_r2 position is empty,
# which asks for every industry code in the dataset.
QUERIES = {
    "nama_10_a10": [("nama_10_a10", f"A.CP_MEUR..B1G+D1+D11+B2A3G+D29X39.{GEOS}")],
    "nama_10_a10_e": [("nama_10_a10_e", f"A.THS_PER+THS_HW..EMP_DC+SAL_DC.{GEOS}")],
    "nama_10_a64": [("nama_10_a64", f"A.CP_MEUR..B1G+D1+D11+B2A3G+D29X39.{GEOS}")],
    "nama_10_a64_e": [("nama_10_a64_e", f"A.THS_PER+THS_HW..EMP_DC+SAL_DC.{GEOS}")],
}

CELL = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)?\s*([a-z]*)\s*$")


def parse_tsv(text: str) -> pd.DataFrame:
    """Parse an Eurostat SDMX-TSV response into long format.

    The first header cell lists the dimensions (``freq,unit,nace_r2,...\\TIME_PERIOD``),
    the other header cells are periods. A value cell is a number optionally
    followed by flags, or ``:`` (missing) optionally followed by flags.

    Returns
    -------
    pd.DataFrame
        One column per dimension, plus ``year`` (int), ``value`` (float, NaN
        when missing) and ``flag`` (string, empty when none).
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty response.")
    header = lines[0].split("\t")
    dims = header[0].split("\\")[0].split(",")
    periods = [int(cell.strip()) for cell in header[1:]]
    records = []
    for line in lines[1:]:
        cells = line.split("\t")
        keys = cells[0].split(",")
        if len(keys) != len(dims):
            raise ValueError(f"Malformed row: {line[:80]!r}")
        for period, cell in zip(periods, cells[1:]):
            raw = cell.strip()
            if raw.startswith(":"):
                value, flag = np.nan, raw[1:].strip()
            else:
                match = CELL.match(raw)
                if not match or match.group(1) is None:
                    raise ValueError(f"Unreadable cell {cell!r} in {cells[0]}")
                value, flag = float(match.group(1)), match.group(2)
            records.append({**dict(zip(dims, keys)), "year": period, "value": value, "flag": flag})
    return pd.DataFrame.from_records(records)


def download(dataset: str, key: str, attempts: int = 4) -> pd.DataFrame:
    """Download one dataset slice from Eurostat, retrying transient failures."""
    url = f"{API}/{dataset}/{key}/?format=TSV&startPeriod={START}"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 (fixed https host)
                return parse_tsv(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts:
                raise
            wait = 20 * attempt
            print(f"{dataset}: {error}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset, slices in QUERIES.items():
        for stem, key in slices:
            frame = download(dataset, key)
            dims = [c for c in frame.columns if c not in ("year", "value", "flag", "freq")]
            frame = frame.drop(columns="freq").sort_values(dims + ["year"]).reset_index(drop=True)
            frame.to_csv(args.out / f"{stem}.csv", index=False)
            items = sorted(frame["na_item"].unique())
            print(f"{stem}: {len(frame)} rows, items {items}, "
                  f"{frame['nace_r2'].nunique()} industry codes, geos {sorted(frame['geo'].unique())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
