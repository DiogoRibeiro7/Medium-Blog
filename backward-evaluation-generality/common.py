"""Settings and data loading shared by the analysis scripts (PREREGISTRATION.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("backward-evaluation", "backward-evaluation-pooling", "backward-evaluation-episode-end"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.append(p)

OUT = HERE / "results"
BLOCK, GAP, MAX_BACK, RESELECT, FOLDS = 364, 28, 8, 91, 5
Z_CRIT = float(norm.ppf(1 - 0.05 / MAX_BACK))
R_CHECK = 91
C_PRIMARY, C_SECONDARY = 0.0, 1.645
DOMAINS = ("elec", "web")
COVID_SCORING = ("2020-03-01", "2020-06-30")


def load(domain: str) -> dict[str, pd.Series]:
    """Transformed daily series: log load (elec) or log(1 + views) (web), keyed by series name."""
    if domain == "elec":
        df = pd.read_csv(HERE / "data" / "electricity_daily.csv", parse_dates=["date"]).set_index("date")
        return {c: np.log(df[c].astype(float)).rename(c) for c in df.columns}
    df = pd.read_csv(HERE / "data" / "wikipedia_daily.csv.gz", parse_dates=["date"]).set_index("date")
    return {c: np.log1p(df[c].astype(float)).rename(c) for c in df.columns}


def group_of(domain: str, key: str) -> str:
    return "country" if domain == "elec" else key.split(":", 1)[0]
