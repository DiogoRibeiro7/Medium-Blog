import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backward_eval import (  # noqa: E402
    ARGaussian,
    ARStudentT,
    ar1_moments,
    backward_window,
    embed_ar,
    make_blocks,
    population_excess_nlpd,
    reverse_cusumsq_window,
    row_losses,
    transfer_matrix,
    variance_ratio_excess,
)
from dgp import scenarios, simulate  # noqa: E402
from schemes import CumOLS, run_window_scheme  # noqa: E402


def test_embed_ar_layout():
    y = np.arange(10.0)
    X, z, t = embed_ar(y, 2)
    assert X.shape == (8, 3) and z.shape == (8,)
    assert np.allclose(X[0], [1, 1, 0]) and z[0] == 2 and t[0] == 2
    assert np.allclose(X[-1], [1, 8, 7]) and z[-1] == 9


def test_blocks_aligned_to_end():
    b = make_blocks(125, 60)
    assert b == [(5, 65), (65, 125)]


def test_gaussian_and_t_log_loss_match_scipy():
    rng = np.random.default_rng(0)
    z, m, s = rng.normal(size=50), rng.normal(size=50), np.exp(rng.normal(size=50))
    g = row_losses(z, m, s)
    assert np.allclose(g["nlpd"], -norm.logpdf(z, m, s))
    tt = row_losses(z, m, s, df=5.0)
    assert np.allclose(tt["nlpd"], -student_t.logpdf(z, 5.0, m, s))
    assert np.isnan(tt["crps"]).all()


def test_gaussian_crps_matches_integral():
    z, m, s = 0.7, -0.2, 1.3
    xs = np.linspace(-15, 15, 300_001)
    num = np.trapezoid((norm.cdf(xs, m, s) - (xs >= z)) ** 2, xs)
    assert row_losses(np.array([z]), np.array([m]), np.array([s]))["crps"][0] == pytest.approx(num, abs=1e-4)


def test_variance_formula_is_special_case_of_population_formula():
    b = np.array([0.0, 0.5])
    for r in (0.25, 0.5, 2.0, 4.0):
        sj = 1.0
        si = np.sqrt(sj**2 / r)
        pop = population_excess_nlpd(b, si, b, sj, ar1_moments(0.0, 0.5, sj))
        assert pop == pytest.approx(variance_ratio_excess(r))
    assert variance_ratio_excess(4.0) > variance_ratio_excess(0.25) > 0


def test_cumols_matches_lstsq():
    rng = np.random.default_rng(1)
    X = np.c_[np.ones(200), rng.normal(size=(200, 2))]
    z = X @ np.array([0.3, -1.0, 2.0]) + rng.normal(size=200)
    cum = CumOLS(X, z)
    for a, b in [(0, 50), (37, 180), (100, 200)]:
        beta, s = cum.fit(np.array([a]), np.array([b]))
        ref = ARGaussian().fit(X[a:b], z[a:b])
        assert np.allclose(beta[0], ref.beta) and s[0] == pytest.approx(ref.sd)


def _series(name, seed, rows=600, at=300):
    mu, phi, sig = scenarios(rows + 1, at + 1)[name]()
    y = simulate(mu, phi, sig, np.random.default_rng(seed))
    return embed_ar(y, 1)


def test_matrix_is_flat_under_stationarity():
    vals = []
    for seed in range(5):
        X, z, _ = _series("stationary", seed)
        E = transfer_matrix(X, z, ARGaussian, 60, gap=6, seed=seed).excess("nlpd")
        vals.append(E[~np.eye(len(E), dtype=bool)].mean())
    assert abs(np.mean(vals)) < 0.05


def test_volatility_fall_is_asymmetric_in_log_loss_not_in_squared_error():
    X, z, _ = _series("variance_down", 3)
    res = transfer_matrix(X, z, ARGaussian, 60, gap=6, seed=3)
    E, S = res.excess("nlpd"), res.excess("sqerr")
    backward = E[5:, :5].mean()  # calm (late) models on turbulent (early) blocks
    forward = E[:5, 5:].mean()
    assert backward > forward + 0.2
    assert abs(S[5:, :5].mean() - S[:5, 5:].mean()) < 0.3
    C = res.mean_matrix("cover90")
    assert C[5:, :5].mean() < 0.75 < 0.95 < C[:5, 5:].mean()


def test_backward_window_keeps_everything_without_a_break():
    X, z, _ = _series("stationary", 11, rows=720, at=480)
    ch = backward_window(X, z, len(z), ARGaussian, 60, 8, 2.5, gap=6, rng=np.random.default_rng(0))
    assert ch.stopped_at is None and ch.rows == len(z)


def test_backward_window_stops_at_a_large_volatility_break():
    X, z, _ = _series("variance_down", 12, rows=720, at=480)
    ch = backward_window(X, z, len(z), ARGaussian, 60, 8, 2.5, gap=6, rng=np.random.default_rng(0))
    # 240 post-break rows: blocks 1-3 back are post-break, block 4 back is the first pre-break one
    assert ch.stopped_at == 4 and ch.rows == 240


def test_reverse_cusumsq_finds_a_large_volatility_break():
    X, z, _ = _series("variance_up", 5, rows=720, at=480)
    rows = reverse_cusumsq_window(X, z, len(z), 30, len(z))
    assert 180 <= rows <= 300


def test_student_t_fit_resists_outliers():
    rng = np.random.default_rng(2)
    X = np.c_[np.ones(300), rng.normal(size=300)]
    z = X @ np.array([0.0, 0.8]) + 0.5 * rng.normal(size=300)
    z[:6] += np.array([25, -30, 20, -25, 30, -20])
    t_fit = ARStudentT().fit(X, z)
    g_fit = ARGaussian().fit(X, z)
    assert abs(t_fit.beta[1] - 0.8) < abs(g_fit.beta[1] - 0.8)
    assert t_fit.sd < g_fit.sd


def test_window_scheme_reports_window_lengths():
    X, z, _ = _series("stationary", 4, rows=300, at=150)
    cum = CumOLS(X, z)
    o = np.arange(200, 210)
    out = run_window_scheme(cum, o, o - 60)
    assert np.all(out.window == 60) and np.all((out.pit >= 0) & (out.pit <= 1))
