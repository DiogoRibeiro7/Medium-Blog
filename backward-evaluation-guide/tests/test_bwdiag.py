import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bwdiag import (LinearModel, RegressorModel, blocks_from_end, lag_rows, matrix_from_forecasts,  # noqa: E402
                    monitor, nlpd, repair, transfer_matrix)
from sims import factory, simulate  # noqa: E402


def test_nlpd_matches_scipy():
    rng = np.random.default_rng(0)
    z, m, s = rng.normal(size=50), rng.normal(size=50), rng.uniform(0.5, 2, 50)
    assert np.allclose(nlpd(z, m, s), -norm.logpdf(z, m, s))
    assert np.allclose(nlpd(z, m, s, 5.0), -student_t.logpdf(z, 5.0, m, s))


def test_blocks_are_aligned_to_the_end():
    b = blocks_from_end(1003, 100)
    assert b[-1] == (903, 1003) and b[0] == (3, 103) and len(b) == 10


@pytest.mark.parametrize("df", [np.inf, 5.0])
def test_repair_reads_bias_as_level_and_width_as_spread(df):
    rng = np.random.default_rng(1)
    z = rng.standard_t(5, 2000) if np.isfinite(df) else rng.normal(size=2000)
    good = repair(z, np.zeros_like(z), 1.0, df)
    assert good.level < 0.005 and good.spread < 0.005
    biased = repair(z, np.full_like(z, -1.5), 1.0, df)
    assert biased.level > 0.3 and biased.spread < 0.1 * biased.level and abs(biased.shift - 1.5) < 0.1
    narrow = repair(z, np.zeros_like(z), 0.3, df)
    assert narrow.spread > 0.3 and narrow.level < 0.01 and abs(narrow.factor - 1 / 0.3) < 0.4
    for r in (good, biased, narrow):
        assert r.level >= 0 and r.spread >= 0
        assert np.isclose(r.raw - r.level - r.spread, r.repaired)


def test_linear_student_t_recovers_coefficients():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 2))
    z = 1.0 + X @ np.array([0.5, -0.3]) + 0.7 * rng.standard_t(5, 3000)
    m = LinearModel(df=5).fit(X, z)
    assert np.allclose(m.beta, [1.0, 0.5, -0.3], atol=0.05) and abs(m.scale - 0.7) < 0.05


def test_regressor_model_scale_is_out_of_fold():
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 1))
    z = X[:, 0] + rng.normal(size=600)
    m = RegressorModel(lambda: DecisionTreeRegressor(min_samples_leaf=1), folds=3).fit(X, z)
    assert m.scale > 0.9  # an in-sample scale of a fully grown tree would be near zero


@pytest.mark.parametrize("scenario,part", [("level", "level"), ("spread", "spread"), ("dynamics", "structure")])
def test_matrix_reads_the_kind_of_change(scenario, part):
    shares = []
    for seed in range(3):
        X, z, _ = lag_rows(simulate(scenario, np.random.default_rng(seed)), 1)
        M = transfer_matrix(X, z, factory, 120, gap=1, seed=seed)
        across = np.zeros((M.k, M.k), bool)
        across[:6, 6:] = across[6:, :6] = True
        assert np.allclose(M.excess, M.level + M.spread + M.structure, equal_nan=True)
        e = np.nanmean(M.excess[across])
        assert e > 0.2 and np.nanmean(np.abs(M.excess[~across & ~np.eye(M.k, dtype=bool)])) < 0.06
        shares.append(np.nanmean(M.part(part)[across]) / e)
    assert np.mean(shares) > 0.5


def test_regimes_find_the_break_and_the_episode():
    X, z, _ = lag_rows(simulate("level", np.random.default_rng(4)), 1)
    M = transfer_matrix(X, z, factory, 120, gap=1)
    assert M.regimes(2)[0] == [(0, 5), (6, 9)]
    c = M.regime_costs(3)
    assert c[0] > 5 * max(abs(c[1]), 0.5)
    X, z, _ = lag_rows(simulate("episode", np.random.default_rng(5)), 1)
    M = transfer_matrix(X, z, factory, 120, gap=1)
    assert M.regimes(3)[0] == [(0, 3), (4, 4), (5, 9)]


def test_matrix_from_forecasts_matches_transfer_matrix():
    X, z, _ = lag_rows(simulate("spread", np.random.default_rng(6)), 1)
    blocks = blocks_from_end(len(z), 120)
    M1 = transfer_matrix(X, z, factory, 120, gap=1, seed=0)
    rng = np.random.default_rng(0)
    from bwdiag import kfold_losses
    own = {j: kfold_losses(X[a:b], z[a:b], factory, 5, rng) for j, (a, b) in enumerate(blocks)}
    M2 = matrix_from_forecasts(z, blocks, lambda i: factory().fit(X[blocks[i][0]:blocks[i][1]],
                                                                  z[blocks[i][0]:blocks[i][1]]).predict(X),
                               lambda j: own[j], df=5.0, gap=1)
    assert np.allclose(M1.excess, M2.excess, equal_nan=True)


def test_monitor_rises_after_the_change():
    X, z, _ = lag_rows(simulate("level", np.random.default_rng(7)), 1)
    out = monitor(X, z, factory, window=120, step=60, gap=1)
    before = [r["excess"] for r in out if r["origin"] < 6 * 120]
    peak = max(out, key=lambda r: r["excess"])
    assert max(before) < 0.1 and peak["excess"] > 0.3 and 6 * 120 < peak["origin"] <= 8 * 120
    assert peak["level"] > 5 * peak["spread"]
