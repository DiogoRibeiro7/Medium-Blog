"""What Erlang A expects for each row of an interval table.

For a day whose offered calls and staffing per half hour are known, Erlang A
is replayed ``reps`` times and the per-interval outcomes are pooled. The
result, written eta, is the mechanistic model's expected service level,
abandonment share and mean wait for each row; the discrepancy models learn
what reality adds on top of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gentradeoffs.mechanistic import QueueModel
from gentradeoffs.queue import ABANDONED, OFFERED, TOTAL_WAIT, WITHIN

TARGETS = ("sl", "abandon_rate", "awt")
REPS = 6


def eta_day(model: QueueModel, rng: np.random.Generator, counts: np.ndarray,
            staff: np.ndarray, reps: int = REPS) -> np.ndarray:
    """Expected (sl, abandon_rate, awt) per interval, shape (24, 3)."""
    tot = np.zeros((counts.size, 6))
    for _ in range(reps):
        tot += model.run_given(rng, counts, staff)
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where(tot[:, OFFERED] > 0, tot[:, OFFERED], np.nan)
        out = np.column_stack([tot[:, WITHIN] / off, tot[:, ABANDONED] / off,
                               tot[:, TOTAL_WAIT] / off])
    # An interval with no calls in any replay: nothing waits, nothing is lost.
    out[np.isnan(out[:, 0]), 0] = 1.0
    out[np.isnan(out[:, 1]), 1] = 0.0
    out[np.isnan(out[:, 2]), 2] = 0.0
    return out


def eta_table(model: QueueModel, rng: np.random.Generator,
              table: pd.DataFrame, reps: int = REPS) -> pd.DataFrame:
    """eta for every row of a table with a ``day`` column, in row order."""
    pieces = []
    for _, d in table.groupby("day", sort=False):
        e = eta_day(model, rng, d["offered"].to_numpy().astype(np.int64),
                    d["agents"].to_numpy().astype(np.int64), reps)
        pieces.append(pd.DataFrame(e, index=d.index,
                                   columns=[f"eta_{t}" for t in TARGETS]))
    return pd.concat(pieces).loc[table.index]
