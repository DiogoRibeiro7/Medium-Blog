"""Download the Eurostat industry accounts used by ``industries.py``.

Every branch of the A10 and A64 breakdowns is downloaded, for Portugal,
Spain, Italy and Greece::

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
HERE = Path(__file__).resolve().parent

GEOS = ("PT", "ES", "IT", "EL")

# One small request per dataset, country and item; the industry position is
# left empty, which asks for every industry code in the dataset.
# Items each accounts dataset publishes (B.2a3g is not published by industry;
# D.29 - D.39 only at the A64 level).
ACCOUNT_ITEMS = {"nama_10_a10": ("B1G", "D1", "D11"), "nama_10_a64": ("B1G", "D1", "D11", "D29X39")}
EMPLOYMENT_ITEMS = ("EMP_DC", "SAL_DC")
EMPLOYMENT_UNITS = ("THS_PER", "THS_HW")


def queries() -> dict[str, list[str]]:
    """Return the SDMX keys to download, grouped by dataset."""
    out: dict[str, list[str]] = {}
    for dataset in ("nama_10_a10", "nama_10_a64"):
        out[dataset] = [f"A.CP_MEUR..{item}.{geo}" for geo in GEOS for item in ACCOUNT_ITEMS[dataset]]
    for dataset in ("nama_10_a10_e", "nama_10_a64_e"):
        out[dataset] = [
            f"A.{unit}..{item}.{geo}" for geo in GEOS for unit in EMPLOYMENT_UNITS for item in EMPLOYMENT_ITEMS
        ]
    return out


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
    """Command-line entry point.

    Failed requests are logged to ``fetch_log.txt`` in the output folder and
    the rest of the download continues, so a partial snapshot and its log are
    always written. The exit code is 1 if any request failed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    failures = 0
    for dataset, keys in queries().items():
        frames = []
        for key in keys:
            try:
                frames.append(download(dataset, key))
            except Exception as error:  # noqa: BLE001 (logged and reported)
                failures += 1
                detail = error.read().decode("utf-8", "replace")[:300] if isinstance(error, urllib.error.HTTPError) else ""
                log.append(f"FAIL {dataset} {key}: {error!r} {detail}".strip())
        if not frames:
            log.append(f"{dataset}: no data")
            continue
        frame = pd.concat(frames, ignore_index=True).drop(columns="freq")
        dims = [c for c in frame.columns if c not in ("year", "value", "flag")]
        frame = frame.sort_values(dims + ["year"]).reset_index(drop=True)
        frame.to_csv(args.out / f"{dataset}.csv", index=False)
        log.append(
            f"{dataset}: {len(frame)} rows, items {sorted(frame['na_item'].unique())}, "
            f"units {sorted(frame['unit'].unique())}, {frame['nace_r2'].nunique()} industry codes, "
            f"geos {sorted(frame['geo'].unique())}, years {frame['year'].min()}-{frame['year'].max()}"
        )
    (args.out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
