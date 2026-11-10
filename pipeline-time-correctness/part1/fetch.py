"""Download the two historical datasets used in Part 1 and store them as CSV.

Both come from R packages on CRAN, read from their GitHub mirrors at pinned commits:

* Taylor and Ashe (1983) run-off triangle, as ``GenInsLong`` in the ChainLadder
  package (the data Mack, 1993, uses in his Tables 1-3).
* Hospitalization and reporting dates of 630 HUS cases from the 2011 STEC O104:H4
  outbreak in Germany, as ``husO104Hosp`` in the surveillance package (the
  reporting triangle of Höhle and an der Heiden, 2014, web appendix Fig. 1).

The CSV files are committed, so the analysis does not need the network or pyreadr.
Run this only to rebuild them: ``python fetch.py`` (needs ``pip install pyreadr``).
"""
from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SOURCES = {
    "taylor_ashe.csv": (
        "https://raw.githubusercontent.com/mages/ChainLadder/"
        "41f4e949e5c3bffd7a18af2c0eaa98d6bae2da2f/data/GenInsLong.RData",
        "GenInsLong",
    ),
    "hus_o104_2011.csv": (
        "https://raw.githubusercontent.com/cran/surveillance/"
        "8e07579097e463f828b48ff7fb34b2945d1fb374/data/husO104Hosp.RData",
        "husO104Hosp",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import pyreadr  # only needed to rebuild the CSV files

    DATA.mkdir(exist_ok=True)
    log = []
    with tempfile.TemporaryDirectory() as tmp:
        for csv_name, (url, obj) in SOURCES.items():
            raw = Path(tmp) / Path(url).name
            urllib.request.urlretrieve(url, raw)
            frame = pyreadr.read_r(str(raw))[obj]
            if obj == "GenInsLong":
                frame = frame.rename(columns={"accyear": "accident_year", "devyear": "development_year",
                                              "incurred claims": "cumulative_claims"})
                frame = frame.astype({"accident_year": int, "development_year": int, "cumulative_claims": int})
                frame = frame.sort_values(["accident_year", "development_year"])
            else:
                frame = frame.rename(columns={"dHosp": "hospitalized", "dReport": "reported"})
                frame = frame.sort_values(["hospitalized", "reported"])
            out = DATA / csv_name
            frame.to_csv(out, index=False, lineterminator="\n")
            log.append(f"{csv_name}\n  source {url}\n  source sha256 {sha256(raw)}\n"
                       f"  rows {len(frame)}\n  csv sha256 {sha256(out)}")
    (DATA / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
