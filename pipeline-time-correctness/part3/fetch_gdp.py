"""Download quarterly real-GDP vintages for the US, the euro area, Japan and Switzerland.

Source: the `gdp` data set of the rOpenSci R package reviser (Burri and Wegmueller), file
data/gdp.rda in ropensci/reviser on GitHub, pinned to REVISER_COMMIT. The package documents it
as real GDP from the real-time database of Indergand and Leist (2014): one vintage per quarter
from 2002 Q4 to 2024 Q4, observations from 1980 Q1, sources SECO (Switzerland), FRED and OECD
(US), Eurostat and OECD (euro area), Cabinet Office and OECD (Japan).

Writes data/gdp_vintages.csv (id, time, pub_date, value) and its SHA-256 to data/gdp_fetch_log.txt.
Needs pyreadr (pip install pyreadr) to read the .rda file; nothing else in Part 3 does.

Usage: python fetch_gdp.py
"""
from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

REVISER_COMMIT = "2a96be4be728c7b8c4210456379e1e3779818db1"
URL = f"https://raw.githubusercontent.com/ropensci/reviser/{REVISER_COMMIT}/data/gdp.rda"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main() -> None:
    import pyreadr

    DATA.mkdir(exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=60).read()
    sha = hashlib.sha256(raw).hexdigest()
    with tempfile.NamedTemporaryFile(suffix=".rda") as fh:
        fh.write(raw)
        fh.flush()
        gdp = pyreadr.read_r(fh.name)["gdp"]
    gdp = gdp[["id", "time", "pub_date", "value"]].sort_values(["id", "pub_date", "time"])
    gdp.to_csv(DATA / "gdp_vintages.csv", index=False, float_format="%.6f")
    (DATA / "gdp_fetch_log.txt").write_text(f"{URL}\nsha256 {sha}\nrows {len(gdp)}\n")
    print(f"{len(gdp)} rows, sha256 {sha}")


if __name__ == "__main__":
    main()
