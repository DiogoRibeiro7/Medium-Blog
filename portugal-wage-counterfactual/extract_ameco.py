"""Extract the Portuguese AMECO series used here from the CRAN package ``ameco``.

The European Commission's AMECO servers are not reachable from every
environment, so the series are taken from the data shipped in the CRAN package
``ameco`` 0.2.9 (spring 2018 vintage, mirrored at https://github.com/cran/ameco)
and committed as ``data/ameco_prt.csv``. Rerun only to rebuild that file::

    git clone --depth 1 https://github.com/cran/ameco.git /tmp/ameco
    python -m pip install pyreadr
    python extract_ameco.py /tmp/ameco/data/ameco.RData
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyreadr

CODES = {
    "PRT.1.0.0.0.HWCDW": "comp_per_emp",       # nominal compensation per employee, 1000 EUR-PTE
    "PRT.1.0.0.0.UWCD": "comp_total",          # compensation of employees, bn EUR-PTE
    "PRT.1.0.0.0.NWTD": "employees",           # employees, domestic, 1000 persons
    "PRT.1.0.0.0.NETD": "employment",          # employment, domestic, 1000 persons
    "PRT.1.0.0.0.NLHT": "hours",               # total hours worked, millions
    "PRT.1.1.0.0.OVGD": "gdp_real",            # GDP at 2010 reference levels, bn EUR-PTE
    "PRT.1.1.0.0.RVGDE": "gdp_real_per_emp",   # GDP at 2010 reference levels per person employed
    "PRT.3.1.0.0.PVGD": "gdp_deflator",        # 2010 = 100
    "PRT.3.0.0.0.ZCPIN": "cpi",                # national CPI, 2010 = 100
    "PRT.1.0.0.0.ALCD0": "adj_wage_share",     # adjusted wage share, % of GDP at market prices
}

HERE = Path(__file__).resolve().parent


def main(rdata: str) -> None:
    df = list(pyreadr.read_r(rdata).values())[0]
    df = df[df["code"].isin(CODES)]
    wide = df.pivot_table(index="year", columns="code", values="value").rename(columns=CODES)
    wide.index = wide.index.astype(int)
    wide = wide.loc[1960:2018, list(CODES.values())]
    wide.to_csv(HERE / "data" / "ameco_prt.csv", float_format="%.6g")


if __name__ == "__main__":
    main(sys.argv[1])
