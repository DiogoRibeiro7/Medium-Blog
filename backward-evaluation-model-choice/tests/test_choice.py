import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402,F401
from choice import CANDS, PROCS, SETS, Designs, ar_design, fit_score, run_series  # noqa: E402
from models import MAX_LAG  # noqa: E402


def daily(n, seed, shift_at=None):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-07-01", periods=n, freq="D")
    week = np.array([0.0, 0.1, 0.12, 0.1, 0.08, -0.2, -0.3])[np.asarray(dates.dayofweek)]
    x, lev = 0.0, np.zeros(n)
    for t in range(n):
        x = 0.8 * x + rng.normal(scale=0.1)
        lev[t] = x
    y = 8 + week + lev
    if shift_at:
        y[shift_at:] += 0.7
    return pd.Series(y, index=dates)


def test_ar_design_alignment():
    s = daily(200, 0)
    y = s.to_numpy()
    X = ar_design(y, 14)
    assert X.shape == (200 - MAX_LAG, 15)
    assert X[0, 1] == y[MAX_LAG - 1] and X[0, 14] == y[MAX_LAG - 14]


def test_backward_scores_for_ets_cover_rows_before_the_fit_window():
    Ds = Designs(daily(900, 1))
    past = np.arange(MAX_LAG, 400)
    out = fit_score("ETS", Ds, 500, 864, past, from_start=0)
    assert np.isfinite(out["g"]).all() and np.isfinite(out["t"]).all() and len(out["t"]) == len(past)


def test_a_model_of_a_shifted_present_scores_worse_on_the_past_than_on_the_present():
    Ds = Designs(daily(1200, 2, shift_at=800))
    past = np.arange(MAX_LAG, 700)
    present = np.arange(850, 1100)
    a = fit_score("AR1", Ds, 836, 1172, past)["t"].mean()
    b = fit_score("AR1", Ds, 836, 1172, present)["t"].mean()
    assert a > b


def test_run_series_end_to_end():
    per, dec = run_series(("test_web", "us_state:X", daily(1600, 3, shift_at=1200)))
    assert set(per.set) == set(SETS)
    for st, cands in SETS.items():
        p = per[per.set == st].set_index("proc").sum_ll
        assert np.isfinite(p).all()
        cand = p[[f"CAND:{c}" for c in cands]]
        # the oracle is never worse than any candidate, and every procedure is between oracle and the worst candidate
        assert p["ORACLE"] <= cand.min() + 1e-9
        for q in PROCS:
            assert p["ORACLE"] - 1e-9 <= p[q] <= cand.max() + 1e-9
    assert set(dec.proc) == set(PROCS) and set(CANDS) >= set(dec.choice)
