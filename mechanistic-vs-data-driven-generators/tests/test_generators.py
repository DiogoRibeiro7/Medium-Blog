"""Checks of the generators' defining properties."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gentradeoffs.data_driven import (CartSynthesizer, GaussianCopula,
                                      GaussianMixture, Smote)
from gentradeoffs.mechanistic import (QueueModel, days_inputs, theta_ratio,
                                      theta_smm)
from gentradeoffs.world import COLUMNS, World, simulate_table


def small_table(seed: int = 0, days: int = 15) -> pd.DataFrame:
    return simulate_table(np.random.default_rng(seed), days)


def test_table_shape_and_ranges():
    t = small_table()
    assert len(t) == 15 * 24
    assert list(t.columns) == ["day"] + COLUMNS
    assert t["sl"].between(0, 1).all()
    assert t["abandon_rate"].between(0, 1).all()
    assert (t["agents"] >= 1).all()


def test_copula_cannot_leave_observed_range():
    t = small_table()
    cop = GaussianCopula().fit(t)
    given = t[["dow", "interval", "campaign", "offered"]].iloc[:50].copy()
    given["offered"] = 10 * t["offered"].max()
    out = cop.sample_given(np.random.default_rng(1), given)
    for c in ("agents", "sl", "awt", "aht"):
        assert out[c].max() <= t[c].max() + 1e-9
        assert out[c].min() >= t[c].min() - 1e-9


def test_cart_only_returns_observed_values():
    t = small_table()
    cart = CartSynthesizer().fit(t)
    out = cart.sample(np.random.default_rng(2), 300)
    for c in ("sl", "awt", "aht", "abandon_rate"):
        assert np.isin(out[c].to_numpy(), t[c].to_numpy()).all()


def test_smote_rows_lie_between_neighbours():
    t = small_table()
    sm = Smote().fit(t)
    out = sm.sample(np.random.default_rng(3), 300)
    for c in ("sl", "awt", "aht"):
        assert out[c].max() <= t[c].max() + 1e-9
        assert out[c].min() >= t[c].min() - 1e-9


def test_mixture_conditional_matches_gaussian_formula():
    rng = np.random.default_rng(4)
    n = 4000
    base = pd.DataFrame(rng.multivariate_normal(
        np.zeros(len(COLUMNS)), np.eye(len(COLUMNS)) * 0.5 + 0.5, size=n),
        columns=COLUMNS) * 10 + 100
    gm = GaussianMixture(n_components=1).fit(base)
    given = pd.DataFrame({"offered": np.full(2000, 110.0)})
    out = gm.sample_given(np.random.default_rng(5), given)
    # equicorrelated 0.5: E[x | x_offered = mean + 1 sd] = mean + 0.5 sd
    assert abs(out["agents"].mean() - 105.0) < 0.6


def test_ratio_estimator_recovers_exponential_patience():
    world = World()
    t = small_table(days=40)
    theta = 1 / 150.0
    model = QueueModel("A", 300.0, patience="exp", theta=theta)
    rng = np.random.default_rng(6)
    rows = []
    for counts, staff in days_inputs(t):
        out = model.run_given(rng, counts, staff)
        rows.append(pd.DataFrame({
            "offered": out[:, 0], "abandon_rate": out[:, 2] / out[:, 0],
            "awt": out[:, 4] / out[:, 0]}))
    sim = pd.concat(rows)
    assert abs(theta_ratio(sim) / theta - 1) < 0.05
    assert world.mean_patience() > 0


def test_smm_recovers_exponential_patience():
    t = small_table(days=20)
    theta = 1 / 150.0
    model = QueueModel("A", 300.0, patience="exp", theta=theta)
    rng = np.random.default_rng(7)
    frames = []
    for d, (counts, staff) in enumerate(days_inputs(t)):
        out = model.run_given(rng, counts, staff)
        frames.append(pd.DataFrame({
            "day": d, "offered": out[:, 0], "agents": staff,
            "abandon_rate": out[:, 2] / out[:, 0],
            "sl": out[:, 3] / out[:, 0], "awt": out[:, 4] / out[:, 0]}))
    sim = pd.concat(frames)
    # Matching the abandonment share is a far less efficient estimator than
    # the ratio, so only gross errors are caught here.
    est = theta_smm(sim, np.random.default_rng(8), "abandon", 300.0, sims=3)
    assert abs(est / theta - 1) < 0.2
    est = theta_smm(sim, np.random.default_rng(9), "wait", 300.0, sims=3)
    assert abs(est / theta - 1) < 0.1
