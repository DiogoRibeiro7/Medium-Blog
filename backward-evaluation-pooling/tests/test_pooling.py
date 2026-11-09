import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backward-evaluation"))

from backward_eval import embed_ar  # noqa: E402
from pooling import criteria, evidence_weight, pool, run_fast  # noqa: E402
from sims import paths, simulate  # noqa: E402


def test_evidence_weight_is_half_at_threshold_and_rises():
    assert evidence_weight(2.5, 2.5) == pytest.approx(0.5)
    assert evidence_weight(None, 2.5) == 0.0
    assert evidence_weight(3.5, 2.5) == pytest.approx(norm.cdf(1.0))
    assert evidence_weight(9.1, 2.5) > 0.999


def test_pool_endpoints_are_the_components():
    rng = np.random.default_rng(0)
    zv, mB, mE = rng.normal(size=20), rng.normal(size=20), rng.normal(size=20)
    sB, sE = np.full(20, 0.7), np.full(20, 1.3)
    p0 = pool(zv, mB, sB, mE, sE, np.zeros(20))
    p1 = pool(zv, mB, sB, mE, sE, np.ones(20))
    assert np.allclose(p0.nlpd, -norm.logpdf(zv, mE, sE))
    assert np.allclose(p1.nlpd, -norm.logpdf(zv, mB, sB))
    half = pool(zv, mB, sB, mE, sE, np.full(20, 0.5))
    assert np.all(half.nlpd <= np.maximum(p0.nlpd, p1.nlpd) + 1e-12)  # a mixture is never worse than its worse part


def test_criteria_logic():
    g = {"a": {"B": 0.10, "EWP": 0.06, "P0.25": 0.02, "P0.5": 0.05, "P0.75": 0.08},
         "b": {"B": -0.04, "EWP": -0.03, "P0.25": -0.01, "P0.5": -0.02, "P0.75": -0.03}}
    cr = criteria(g, ["a", "b"])
    assert cr["C1_gains_kept"] is True  # 0.06 >= 0.05
    assert cr["C2_losses_removed"] is False  # -0.03 < -0.02
    assert cr["best_fixed"] == "P0.75" and cr["C3_weighting_matters"] is False  # 0.015 < 0.025


def test_crash_shocks_are_injected():
    mu, phi, sd, shocks = paths("crash", 600, 400)
    y = simulate(mu, phi, sd, shocks, np.random.default_rng(1))
    d = y[400] - 0.5 * y[399]
    assert d == pytest.approx(-12.0)


def test_pools_coincide_with_expanding_when_nothing_changes():
    mu, phi, sd, shocks = paths("stationary", 721, 481)
    y = simulate(mu, phi, sd, shocks, np.random.default_rng(3))
    X, z, _ = embed_ar(y, 1)
    outs, sel = run_fast(X, z, np.arange(480, 720), 60, 8, 2.498, 6, seed=0)
    if not any(s.stopped for s in sel):
        for k in ("B", "EWP", "P0.5"):
            assert np.allclose(outs[k].nlpd, outs["E"].nlpd)
