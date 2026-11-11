"""Settings and data loading (PREREGISTRATION.md).

The model families come from ``../backward-evaluation-generality/models.py`` (imported, not copied);
that module's ``import common`` resolves to this file, which puts the companion folders on the path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("backward-evaluation", "backward-evaluation-pooling", "backward-evaluation-episode-end",
            "backward-evaluation-generality"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.append(p)

OUT = HERE / "results"
BLOCK, GAP, REFIT = 364, 28, 91
DOMAINS = ("dev_elec", "dev_web", "test_web")
GEN = ROOT / "backward-evaluation-generality" / "data"


def load(domain: str) -> dict[str, pd.Series]:
    if domain == "dev_elec":
        df = pd.read_csv(GEN / "electricity_daily.csv", parse_dates=["date"]).set_index("date")
        return {c: np.log(df[c].astype(float)).rename(c) for c in df.columns}
    path = GEN / "wikipedia_daily.csv.gz" if domain == "dev_web" else HERE / "data" / "wikipedia_test_daily.csv.gz"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return {c: np.log1p(df[c].astype(float)).rename(c) for c in df.columns}


def group_of(domain: str, key: str) -> str:
    return "country" if domain == "dev_elec" else key.split(":", 1)[0]
