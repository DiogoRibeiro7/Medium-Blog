"""Download the Eurostat series used by ``labour_share.py``.

Rebuilds the four CSV files in ``data/`` from the Eurostat SDMX 2.1 API, or
compares a fresh download with the committed snapshot::

    python fetch.py --out data              # overwrite the snapshot
    python fetch.py --out live --compare    # download to live/ and report revisions

Values are current prices (million euro) or thousands of persons and hours.
Flags (p = provisional, e = estimated, ...) are dropped.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
START = 1995
HERE = Path(__file__).resolve().parent

QUERIES = {
    "nama_10_gdp": "A.CP_MEUR.B1GQ+D1+B2A3G+D2X3.PT+ES+IT+EL",
    "nama_10_a10_e": "A.THS_PER+THS_HW.TOTAL.EMP_DC+SAL_DC.PT",
    "gov_10a_main": "A.MIO_EUR.S13.D1PAY+P51C+P1+B1G+D29PAY+D39REC.PT",
    "nasa_10_nf_tr": "A.CP_MEUR.PAID.B2A3G+B2G+B3G.S1+S14_S15.PT",
}

NUMBER = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)")


def parse_tsv(text: str) -> pd.DataFrame:
    """Parse an Eurostat SDMX-TSV response into long format.

    The first header cell lists the dimensions (``freq,unit,na_item,geo\\TIME_PERIOD``),
    the other header cells are periods, and each value cell is a number
    optionally followed by a space and flags, or ``:`` when missing.

    Returns
    -------
    pd.DataFrame
        One column per dimension, plus ``year`` (int) and ``value`` (float, NaN
        when missing).
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
            match = NUMBER.match(cell)
            value = float(match.group(1)) if match else np.nan
            records.append({**dict(zip(dims, keys)), "year": period, "value": value})
    return pd.DataFrame.from_records(records)


def download(dataset: str, key: str) -> pd.DataFrame:
    """Download one dataset slice from Eurostat."""
    url = f"{API}/{dataset}/{key}/?format=TSV&startPeriod={START}"
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed https host)
        return parse_tsv(response.read().decode("utf-8"))


def build_files(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Reshape the downloaded slices into the snapshot files."""
    gdp = frames["nama_10_gdp"].pivot_table(index=["geo", "year"], columns="na_item", values="value")
    gdp = gdp.reset_index()[["geo", "year", "B1GQ", "D1", "B2A3G", "D2X3"]]
    order = {"PT": 0, "ES": 1, "IT": 2, "EL": 3}
    gdp = gdp.sort_values(["geo", "year"], key=lambda s: s.map(order) if s.name == "geo" else s)

    emp = frames["nama_10_a10_e"].copy()
    emp["column"] = emp["na_item"] + "_" + emp["unit"]
    emp = emp.pivot_table(index="year", columns="column", values="value").reset_index()
    emp = emp[["year", "EMP_DC_THS_PER", "SAL_DC_THS_PER", "EMP_DC_THS_HW", "SAL_DC_THS_HW"]]

    gov = frames["gov_10a_main"].pivot_table(index="year", columns="na_item", values="value", dropna=False)
    gov = gov.reindex(columns=["B1G", "D1PAY", "D29PAY", "D39REC", "P1", "P51C"])
    gov[["D29PAY", "D39REC"]] = gov[["D29PAY", "D39REC"]].fillna(0.0)
    gov = gov.reset_index()

    nasa = frames["nasa_10_nf_tr"].copy()
    nasa["column"] = nasa["na_item"] + "_" + nasa["sector"]
    nasa = nasa.pivot_table(index="year", columns="column", values="value").reset_index()
    nasa = nasa[["year", "B2A3G_S1", "B2G_S1", "B3G_S1", "B2A3G_S14_S15", "B2G_S14_S15", "B3G_S14_S15"]]

    return {
        "nama_10_gdp.csv": gdp,
        "nama_10_a10_e_PT.csv": emp,
        "gov_10a_main_PT.csv": gov,
        "nasa_10_nf_tr_PT.csv": nasa,
    }


def compare(live: Path, snapshot: Path) -> None:
    """Print the largest absolute revision per file and column."""
    for name in sorted(p.name for p in snapshot.glob("*.csv")):
        old = pd.read_csv(snapshot / name)
        new = pd.read_csv(live / name)
        keys = [k for k in ("geo", "year") if k in old.columns]
        merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
        print(f"{name}: rows only in snapshot {int((merged['_merge'] == 'left_only').sum())}, "
              f"only in live {int((merged['_merge'] == 'right_only').sum())}")
        for col in [c for c in old.columns if c not in keys]:
            diff = (merged[f"{col}_new"] - merged[f"{col}_old"]).abs()
            if diff.max() > 0:
                where = merged.loc[diff.idxmax(), keys].to_dict()
                print(f"  {col}: max revision {diff.max():.1f} at {where}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "data")
    parser.add_argument("--compare", action="store_true", help="compare --out with data/ after downloading")
    args = parser.parse_args(argv)
    frames = {dataset: download(dataset, key) for dataset, key in QUERIES.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in build_files(frames).items():
        frame.to_csv(args.out / name, index=False)
    if args.compare:
        compare(args.out, HERE / "data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
