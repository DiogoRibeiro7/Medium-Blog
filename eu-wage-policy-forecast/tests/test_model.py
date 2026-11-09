"""Checks on the model engine and on the forecast algebra (small economies, fast)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from empirics import median_growth  # noqa: E402
from model import Params, _link, _valuation, simulate, simulate_path, simulate_reform, with_  # noqa: E402

SMALL = Params(n_workers=3000, n_firms=60, burn_in=180, n_measure=3)


def test_valuation_solves_the_equal_pay_condition():
    # one cell with two peers paid 0.5 and 0.7; a worker of output 1 is worth w with
    # 1 - w - pi * [max(0, w - 0.5) + max(0, w - 0.7)] = 0
    n = 3
    head = -np.ones(1, dtype=np.int64)
    nxt = -np.ones(n, dtype=np.int64)
    prv = -np.ones(n, dtype=np.int64)
    wage = np.array([0.5, 0.7, 0.0])
    for i in (0, 1):
        _link(i, 0, head, nxt, prv)
    buf = np.zeros(n)
    for pi in (1.0, 0.3):
        w = _valuation(1.0, 0, -1, head, nxt, wage, buf, pi)
        assert abs(1 - w - pi * (max(0, w - 0.5) + max(0, w - 0.7))) < 1e-12
    # without peers the valuation is the output itself
    assert _valuation(1.0, 0, -1, -np.ones(1, dtype=np.int64), nxt, wage, buf, 1.0) == 1.0


def test_floor_binds_and_costs_matches():
    lo = simulate(with_(SMALL, m=0.45))
    hi = simulate(with_(SMALL, m=0.80))
    assert lo["p10"] >= 0.45 * 0.999 and hi["p10"] >= 0.80 * 0.999
    assert hi["share_at_floor"] > lo["share_at_floor"]
    assert hi["employment_rate"] < lo["employment_rate"]


def test_no_coverage_means_no_reform_effect_until_the_switch():
    d = simulate_reform(with_(SMALL, m=0.65, cover_share=0.0), years=2)
    r, b = d[d.run == "reform"].reset_index(), d[d.run == "baseline"].reset_index()
    assert np.allclose(r.mean_wage, b.mean_wage) and np.allclose(r.p90, b.p90)


def test_equal_pay_rule_curbs_counter_offers():
    d = simulate_reform(with_(SMALL, m=0.65, cover_share=1.0, visibility=1.0), years=3)
    r, b = d[d.run == "reform"], d[d.run == "baseline"]
    assert r.counter_raises_per_year.mean() < b.counter_raises_per_year.mean()
    assert r.counters_refused_per_year.mean() > 0
    assert (r.covered_emp_share > 0.99).all()


def test_a_higher_floor_lifts_every_wage_below_it():
    d = simulate_path(with_(SMALL, m=0.60), [0.70, 0.75])
    assert (d.p10 >= d.m * 0.999).all()
    assert d.share_at_floor.iloc[-1] > 0.2


def test_indexed_minimum_wage_keeps_the_bite():
    # g_mw = g_avg + drift / (1 - theta) makes the median grow at the same rate
    g_avg, theta, drift = 0.04, 0.3, -0.006
    g_mw = g_avg + drift / (1 - theta)
    assert abs(median_growth(g_avg, g_mw, theta, drift) - g_mw) < 1e-15
