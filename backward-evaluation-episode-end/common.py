"""Series definitions and settings shared by the analysis scripts (PREREGISTRATION.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("backward-evaluation", "backward-evaluation-pooling"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.append(p)

OUT = HERE / "results"
START = "1960-01-01"
FIRST_ORIGIN_TEST = "1990-01-01"
FIRST_ORIGIN_DEV = "1985-01-01"
COVID = ("2020-03-01", "2020-12-01")

AR_ORDER, BLOCK, GAP, MAX_BACK, RESELECT, FOLDS = 3, 60, 12, 8, 12, 5
Z_CRIT = float(norm.ppf(1 - 0.05 / MAX_BACK))
R_PRIMARY, C_PRIMARY = 12, 1.645
R_ALT, C_ALT = 24, (0.0, Z_CRIT)
T_DF = 5.0

# key: (label, short label, transformation, group)
TEST = {
    "JP_UNEMP": ("Japan unemployment rate, monthly change (pp)", "Japan unemployment", "diff", "A"),
    "CA_UNEMP": ("Canada unemployment rate, monthly change (pp)", "Canada unemployment", "diff", "A"),
    "UK_IP": ("UK industrial production, monthly growth (%)", "UK production", "dlog100", "A"),
    "DE_IP": ("Germany industrial production, monthly growth (%)", "Germany production", "dlog100", "A"),
    "JP_IP": ("Japan industrial production, monthly growth (%)", "Japan production", "dlog100", "A"),
    "UK_10Y": ("UK 10-year yield, monthly change (pp)", "UK 10-year yield", "diff", "B"),
    "DE_10Y": ("Germany 10-year yield, monthly change (pp)", "Germany 10-year yield", "diff", "B"),
    "CA_10Y": ("Canada 10-year yield, monthly change (pp)", "Canada 10-year yield", "diff", "B"),
    "JPY_USD": ("Yen per US dollar, monthly change (%)", "Yen per dollar", "dlog100", "B"),
    "CHF_USD": ("Swiss francs per US dollar, monthly change (%)", "Swiss franc per dollar", "dlog100", "B"),
}
# the development set: the ten US series of ../backward-evaluation-pooling (in-sample)
DEV = {
    "UNRATE": ("US unemployment rate, monthly change (pp)", "US unemployment", "diff", "A"),
    "PAYEMS": ("US payrolls, monthly growth (%)", "US payrolls", "dlog100", "A"),
    "HOUST": ("US housing starts, monthly growth (%)", "US housing starts", "dlog100", "A"),
    "W875RX1": ("US real income ex transfers, monthly growth (%)", "US real income", "dlog100", "A"),
    "DPCERA3M086SBEA": ("US real spending, monthly growth (%)", "US real spending", "dlog100", "A"),
    "INDPRO": ("US industrial production, monthly growth (%)", "US industrial production", "dlog100", "A"),
    "WPSFD49207": ("US PPI inflation (% a year)", "US PPI inflation", "dlog1200", "B"),
    "GS10": ("US 10-year yield, monthly change (pp)", "US 10-year yield", "diff", "B"),
    "M2SL": ("US M2 growth (%)", "US M2 growth", "dlog100", "B"),
    "CPIAUCSL": ("US CPI inflation (% a year)", "US CPI inflation", "dlog1200", "B"),
}
SHORT = {k: v[1] for k, v in {**TEST, **DEV}.items()}
GROUP = {k: v[3] for k, v in {**TEST, **DEV}.items()}


def _transform(s: pd.Series, tr: str) -> pd.Series:
    if tr == "diff":
        return s.diff()
    if tr == "dlog100":
        return 100 * np.log(s).diff()
    return 1200 * np.log(s).diff()


def load(which: str = "test") -> dict[str, pd.Series]:
    """Transformed series from START; ``which`` is 'test' or 'dev'."""
    if which == "test":
        df = pd.read_csv(HERE / "data" / "international_monthly.csv", parse_dates=["date"]).set_index("date")
        spec = TEST
    else:
        df = pd.read_csv(ROOT / "backward-evaluation-pooling" / "data" / "fred_monthly.csv",
                         parse_dates=["date"]).set_index("date")
        spec = DEV
    out = {}
    for key, (_, _, tr, _) in spec.items():
        if key not in df.columns:
            continue
        s = df[key].dropna()
        out[key] = _transform(s, tr).loc[START:].dropna().rename(key)
    return out
