"""Unit tests for the model, projection and channel helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bill import project  # noqa: E402
from channels import duration, ols, spillover_betas  # noqa: E402
from warranted import fit, in_window  # noqa: E402


def test_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=500), rng.normal(size=500), np.ones(500)])
    y = X @ np.array([0.02, -0.1, 0.3])
    assert np.allclose(fit(X, y), [0.02, -0.1, 0.3])


def test_window_excludes_the_hole():
    q = pd.Series(pd.period_range("2009Q1", "2013Q4", freq="Q"))
    keep = in_window(q, "2009Q1", "2013Q4", ("2010Q2", "2012Q4"))
    assert keep.sum() == len(q) - 11
    assert not keep[q == pd.Period("2011Q1", "Q")].any()


def test_par_bond_duration():
    # numeric modified duration of a 10-year annual-coupon par bond at 3.5%
    r, t = 0.035, 10
    cf = np.r_[np.full(t - 1, r), 1 + r]
    times = np.arange(1, t + 1)
    price = (cf / (1 + r) ** times).sum()
    mac = (times * cf / (1 + r) ** times).sum() / price
    assert abs(duration(10, 3.5) - mac / (1 + r)) < 1e-9
    assert abs(duration(0.5) - 0.5 / 1.035) < 1e-12


def test_spillover_recovers_a_known_response():
    rng = np.random.default_rng(1)
    n = 400
    common = rng.normal(size=n)
    own_fr = rng.normal(size=n)
    cols = {"FR": common + own_fr}
    true = {"IT": 0.4, "ES": 0.0, "BE": 0.2, "AT": 0.0, "NL": 0.0, "FI": 0.0, "PT": 0.0, "IE": 0.0}
    for c, b in true.items():
        cols[c] = common + b * own_fr + 0.3 * rng.normal(size=n)
    est = spillover_betas(pd.DataFrame(cols))
    assert abs(est["IT"] - 0.4) < 0.08 and abs(est["BE"] - 0.2) < 0.08 and abs(est["ES"]) < 0.15


def test_ols_intercept_and_slope():
    x = np.arange(10.0)
    assert np.allclose(ols(2 + 3 * x, x[:, None]), [2, 3])


def test_projection_without_interest_keeps_debt_identity():
    years = np.arange(2026, 2031)
    f = project(100.0, 0.0, np.zeros(len(years)), np.zeros(len(years)), 0.0, years)
    # 2026 deficit 5.4 adds to debt; 2027 deficit 5.0; then primary balance rises 0.5 a year from -5.0
    assert abs(f.loc[0, "debt_gdp"] - 105.4) < 1e-9
    assert abs(f.loc[1, "debt_gdp"] - 110.4) < 1e-9
    assert np.allclose(f["int_gdp"], 0)
    assert abs(f.loc[2, "primary"] + 4.5) < 1e-9
