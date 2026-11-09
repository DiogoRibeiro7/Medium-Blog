"""Series definitions shared by premise.py and forward.py (PREREGISTRATION.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
START = "1960-01-01"
FIRST_ORIGIN = "1985-01-01"
COVID = ("2020-03-01", "2020-12-01")

# id: (label, transformation)
TEST = {
    "UNRATE": ("Unemployment rate, monthly change (pp)", "diff"),
    "PAYEMS": ("Nonfarm payrolls, monthly growth (%)", "dlog100"),
    "HOUST": ("Housing starts, monthly growth (%)", "dlog100"),
    "W875RX1": ("Real personal income ex transfers, monthly growth (%)", "dlog100"),
    "DPCERA3M086SBEA": ("Real consumer spending, monthly growth (%)", "dlog100"),
    "WPSFD49207": ("PPI finished goods inflation (% a year)", "dlog1200"),
    "GS10": ("10-year Treasury yield, monthly change (pp)", "diff"),
    "M2SL": ("M2 money stock, monthly growth (%)", "dlog100"),
}
IN_SAMPLE = {
    "INDPRO": ("Industrial production, monthly growth (%)", "dlog100"),
    "CPIAUCSL": ("CPI inflation (% a year)", "dlog1200"),
}
SHORT = {"UNRATE": "Unemployment", "PAYEMS": "Payrolls", "HOUST": "Housing starts", "W875RX1": "Real income",
         "DPCERA3M086SBEA": "Real spending", "WPSFD49207": "PPI inflation", "GS10": "10-year yield",
         "M2SL": "M2 growth", "INDPRO": "Industrial production", "CPIAUCSL": "CPI inflation"}


def load() -> dict[str, pd.Series]:
    df = pd.read_csv(HERE / "data" / "fred_monthly.csv", parse_dates=["date"]).set_index("date")
    out = {}
    for sid, (_, tr) in {**TEST, **IN_SAMPLE}.items():
        s = df[sid].dropna()
        if tr == "diff":
            x = s.diff()
        elif tr == "dlog100":
            x = 100 * np.log(s).diff()
        else:
            x = 1200 * np.log(s).diff()
        out[sid] = x.loc[START:].dropna().rename(sid)
    return out
