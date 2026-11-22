"""Checks of the methods on cases with a known answer, and of the assembled bid table."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import entry as E  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def test_fe_regression_recovers_a_known_coefficient():
    rng = np.random.default_rng(0)
    n = 4000
    block = rng.integers(0, 300, n)
    cell = rng.integers(0, 20, n)
    a, c = rng.normal(0, 1, 300), rng.normal(0, 1, 20)
    x = (rng.random(n) < 0.2 + 0.1 * (a[block] > 0)).astype(float)    # x correlated with the block effect
    y = 0.7 * x + a[block] + c[cell] + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x, "block_id": block, "cell": cell})
    r = E.fe_regression(df, "y", ["x"], ["block_id", "cell"]).loc["x"]
    assert abs(r.coef - 0.7) < 4 * r.se and r.se < 0.03


def tracts(ratios, n=1, cell="c", minimum=100.0):
    return pd.DataFrame({"ratio": ratios, "n": n, "cell": cell, "minimum_bid": minimum})


def test_second_draw_adds_nothing_when_all_single_bids_are_equal():
    assert abs(E.second_draw_uplift(tracts([1.5] * 20)) - 1) < 1e-12


def test_second_draw_matches_the_expected_maximum_of_two_draws():
    r = np.arange(1, 21, dtype=float)
    u = E.second_draw_uplift(tracts(r))
    emax = np.mean([np.maximum(x, r).mean() for x in r])
    assert abs(u - emax / r.mean()) < 1e-12


def test_reserve_change_bounds():
    t = pd.DataFrame({"n": 1, "shallow": False, "minimum_bid": 100.0, "high": [100.0, 120.0, 200.0]})
    r = E.reserve_change(t, shallow=False, new_per_acre=150)
    assert r["below"] == 2
    assert abs(r["all_pay"] - (150 + 150 + 200) / 420 + 1) < 1e-12
    assert abs(r["none_pay"] - 200 / 420 + 1) < 1e-12


@pytest.mark.skipif(not (HERE / "data" / "bids.csv").exists(), reason="run fetch.py and build_dataset.py first")
def test_bid_table_is_consistent():
    b = pd.read_csv(HERE / "data" / "bids.csv", dtype={"sale": str})
    g = b.groupby(["sale", "lease"])
    assert (g.size() == g.n_bids.first()).all()
    assert (g["rank"].min() == 1).all()
    top = b[b["rank"] == 1].set_index(["sale", "lease"]).amount.sort_index()
    assert (g.amount.max().sort_index() == top).all()
    assert b.sale.nunique() == 21
