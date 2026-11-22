"""Rebuild Portugal's annual general government series, 1994-2025, from the AMECO bulk file.

Output: data/portugal.csv (not committed; rebuilt at each run). AMECO also carries the
Commission's forecasts for 2026-2028; they are dropped here.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ZIP = HERE / "raw" / "ameco0.zip"
OUT = HERE / "data" / "portugal.csv"

COUNTRY = "PRT"
# AMECO codes: country.trn.agg.unit.ref.variable
VARS = {
    "debt": "1.0.319.0.UDGG",  # gross debt, EDP, % of GDP
    "pb": "1.0.319.0.UBLGIE",  # net lending excluding interest, EDP, % of GDP
    "bal": "1.0.319.0.UBLGE",  # net lending, EDP, % of GDP
    "interest": "1.0.319.0.UYIGE",  # interest, EDP, % of GDP
    "revenue": "1.0.319.0.URTG",  # total revenue, % of GDP
    "primary_exp": "1.0.319.0.UUTGI",  # total expenditure excluding interest, % of GDP
    "gap": "1.0.0.0.AVGDGP",  # output gap, % of potential GDP
    "gdp_nom": "1.0.0.0.UVGD",  # GDP, current prices, national currency bn
    "gdp_real": "1.1.0.0.OVGD",  # GDP, 2020 reference levels
    "deflator": "3.1.0.0.PVGD",  # GDP deflator, 2020 = 100
    "ltr": "1.1.0.0.ILN",  # nominal long-term interest rate, %
}
LAST_ACTUAL = 2025


def read_ameco(zip_path: Path = ZIP) -> pd.DataFrame:
    want = {f"{COUNTRY}.{v}": k for k, v in VARS.items()}
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            with z.open(name) as fh:
                txt = io.TextIOWrapper(fh, encoding="latin-1")
                header = txt.readline().rstrip("\n;").split(";")
                years = header[5:]
                for line in txt:
                    parts = line.rstrip("\n;").split(";")
                    if parts[0] not in want:
                        continue
                    k = want[parts[0]]
                    for y, x in zip(years, parts[5:], strict=False):
                        x = x.strip()
                        if x not in ("NA", ""):
                            rows.append((int(y), k, float(x)))
    df = pd.DataFrame(rows, columns=["year", "var", "value"])
    return df.pivot_table(index="year", columns="var", values="value").reset_index()


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    p = read_ameco()
    p = p[(p.year >= 1994) & (p.year <= LAST_ACTUAL)]
    missing = sorted(set(VARS) - set(p.columns))
    if missing:
        raise SystemExit(f"AMECO series missing: {missing}")
    p.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(HERE)}: {p.year.min()}-{p.year.max()}, {len(VARS)} series")
