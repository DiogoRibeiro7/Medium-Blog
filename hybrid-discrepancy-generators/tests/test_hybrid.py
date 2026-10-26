"""Checks of the hybrid components on cases with a known answer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gentradeoffs.mechanistic import QueueModel
from gentradeoffs.world import World, simulate_table
from hybridgen.discrepancy import Discrepancy, apply, link, unlink
from hybridgen.eta import TARGETS, eta_day
from hybridgen.hybrid import (EmpiricalDayDemand, auxiliary_coefficient,
                              day_factors)
from gentradeoffs.mechanistic import DemandModel


def test_link_round_trip():
    v = np.array([0.01, 0.3, 0.9, 0.99])
    for t in ("sl", "abandon_rate"):
        assert np.allclose(unlink(t, link(t, v)), v)
    w = np.array([0.0, 3.0, 40.0])
    assert np.allclose(unlink("awt", link("awt", w)), w)


def test_apply_counts_and_repairs_impossible_values():
    eta = pd.DataFrame({"eta_sl": [0.99, 0.5], "eta_abandon_rate": [0.0, 0.1],
                        "eta_awt": [0.5, 10.0]})
    delta = pd.DataFrame({"sl": [0.05, 0.0], "abandon_rate": [-0.01, 0.0],
                          "awt": [-1.0, 0.0]})
    out, info = apply(eta, delta)
    assert info["share_impossible"] == 0.5
    assert out["sl"].between(0, 1).all()
    assert (out["abandon_rate"] >= 0).all()
    assert (out["awt"] >= 0).all()
    assert (out["sl"] <= 1 - out["abandon_rate"] + 1e-12).all()
    out, _ = apply(eta, delta, scale="link")
    assert out["sl"].between(0, 1).all()


def test_keep_boundaries_leaves_exact_zeros_alone():
    draw = pd.DataFrame({"eta_sl": [1.0], "eta_abandon_rate": [0.0],
                         "eta_awt": [0.0]})
    delta = pd.DataFrame({"sl": [-0.05], "abandon_rate": [0.02],
                          "awt": [3.0]})
    out, _ = apply(draw, delta, keep_boundaries=True)
    assert out.iloc[0].tolist() == [1.0, 0.0, 0.0]


def test_eta_of_an_empty_day():
    q = QueueModel("A", 300.0, patience="exp", theta=1 / 200)
    e = eta_day(q, np.random.default_rng(0), np.zeros(24, np.int64),
                np.full(24, 5, np.int64))
    assert np.allclose(e[:, 0], 1.0) and np.allclose(e[:, 1:], 0.0)


def test_discrepancy_learns_a_constant_offset():
    rng = np.random.default_rng(1)
    n = 600
    table = pd.DataFrame({"day": np.repeat(np.arange(25), 24),
                          "dow": np.repeat(np.arange(25) % 5, 24),
                          "interval": np.tile(np.arange(24), 25),
                          "campaign": 0,
                          "offered": rng.integers(80, 200, n),
                          "agents": rng.integers(15, 40, n)})
    eta = pd.DataFrame({"eta_sl": rng.uniform(0.6, 0.9, n),
                        "eta_abandon_rate": rng.uniform(0.0, 0.1, n),
                        "eta_awt": rng.uniform(0, 20, n)})
    for t in TARGETS:
        table[t] = eta[f"eta_{t}"] + {"sl": -0.05, "abandon_rate": 0.02,
                                      "awt": 3.0}[t]
    for learner in ("boosting", "linear"):
        d = Discrepancy(learner, 300.0).fit(table, eta, rng).predict(table,
                                                                     eta)
        assert abs(d["sl"].mean() + 0.05) < 0.005
        assert abs(d["awt"].mean() - 3.0) < 0.3


def test_day_factors_track_the_day_shock():
    world = World()
    flat = replace(world, day_cv=0.002, interval_sigma=0.001)
    t_flat = simulate_table(np.random.default_rng(2), 60, world=flat)
    t_shock = simulate_table(np.random.default_rng(2), 60, world=world)
    f_flat = day_factors(t_flat, DemandModel.fit(t_flat))
    f_shock = day_factors(t_shock, DemandModel.fit(t_shock))
    assert f_flat.std() < 0.04
    assert 0.06 < f_shock.std() < 0.15
    d = EmpiricalDayDemand.from_table(t_shock)
    assert d.factors.size == 60


def test_auxiliary_coefficient_sees_redials():
    world = World()
    none = replace(world, retry_prob=0.0)
    lots = replace(world, retry_prob=0.9)
    b0 = auxiliary_coefficient(simulate_table(np.random.default_rng(3), 60,
                                              world=none))
    b1 = auxiliary_coefficient(simulate_table(np.random.default_rng(3), 60,
                                              world=lots))
    assert b1 > b0 + 0.15
