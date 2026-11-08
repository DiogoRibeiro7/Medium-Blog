"""Checks of the synthetic control, the pass-through estimate and the repricing arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import interest  # noqa: E402
import ledger  # noqa: E402
import yields  # noqa: E402


def months(n: int, start: str = "1999-01") -> list[str]:
    return [str(p) for p in pd.period_range(start, periods=n, freq="M")]


def test_synthetic_recovers_a_known_combination():
    rng = np.random.default_rng(0)
    idx = months(240)
    donors = pd.DataFrame(rng.normal(size=(240, 4)).cumsum(axis=0), index=idx, columns=list("ABCD"))
    target = 1.5 + 0.6 * donors["A"] + 0.4 * donors["C"]
    path, w, rmse = yields.synthetic(target, donors, (idx[0], idx[119]))
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    assert abs(w["A"] - 0.6) < 1e-3 and abs(w["C"] - 0.4) < 1e-3
    assert rmse < 1e-3
    assert np.allclose(path.loc[idx[120]:], target.loc[idx[120]:], atol=1e-2)


def test_repricing_acts_only_through_the_window_and_then_decays():
    gap = pd.Series({y: 2.0 for y in range(2008, 2020)})
    d = interest.repricing(gap, 0.25, range(2010, 2018))
    assert np.isclose(d[2010], 0.5)
    assert np.isclose(d[2011], 0.75 * 0.5 + 0.5)
    after = d[2015] / d[2014]
    assert np.isclose(after, 0.75)  # gap ends with the window; the rate effect decays at 1 - rho
    one = interest.repricing(gap, 1.0, range(2010, 2016))
    assert np.allclose(one.loc[2010:2014], 2.0) and np.isclose(one[2015], 0.0)


def test_simulate_converges_to_a_constant_yield():
    x = pd.Series({y: 1.0 for y in range(2010, 2060)})
    s = interest.simulate(5.0, x, 0.2, range(2010, 2060))
    assert abs(s[2059] - 1.0) < 1e-3
    assert s[2010] == 5.0 + 0.2 * (1.0 - 5.0)


def test_pass_through_estimate_recovers_rho():
    rng = np.random.default_rng(1)
    years = range(1995, 2026)
    rates, xs = {}, {}
    for c in ("DE", "FR", "NL", "FI", "AT"):
        x = pd.Series(4 + rng.normal(scale=1.0, size=len(years)).cumsum() * 0.3, index=years)
        i = [4.0]
        for t in range(1, len(years)):
            i.append(i[-1] + 0.18 * (x.iloc[t] - i[-1]) + rng.normal(scale=0.02))
        rates[c] = pd.Series(i, index=years)
        xs[c] = x
    rho, se, each = interest.estimate_rho(pd.DataFrame(rates), pd.DataFrame(xs), ("DE", "FR", "NL", "FI", "AT"), (1996, 2025))
    assert abs(rho - 0.18) < 3 * se + 0.01


def test_exposures_use_keys_between_zero_and_one():
    d = ledger.documented()
    ex = ledger.exposures(d)
    assert ((ex["key_low"] > 0) & (ex["key_high"] < 1) & (ex["key_low"] <= ex["key_high"])).all()
    de = ex.set_index("country").loc["DE"]
    assert de["efsf_esm_low"] <= de["efsf_esm_high"]
    assert ex.attrs["efsf_esm_outstanding"] > 200
