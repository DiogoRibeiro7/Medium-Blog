import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402,F401
from models import MAX_LAG, ETSModel, GBMModel, design, fit_predict, t_scale  # noqa: E402
from run import SCHEMES, run_series  # noqa: E402


def daily(n, rng, shift_at=None, shift=0.0, burst=None):
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    week = np.array([0.0, 0.1, 0.12, 0.1, 0.08, -0.2, -0.3])[np.asarray(dates.dayofweek)]
    year = 0.2 * np.sin(2 * np.pi * np.asarray(dates.dayofyear) / 365.25)
    e = rng.normal(scale=0.05, size=n)
    if burst is not None:
        a, b = burst
        e[a:b] *= 6
    level = np.zeros(n)
    x = 0.0
    for t in range(n):
        x = 0.7 * x + e[t]
        level[t] = x
    y = 10 + week + year + level
    if shift_at is not None:
        y[shift_at:] += shift
    return pd.Series(y, index=dates)


def test_design_alignment():
    rng = np.random.default_rng(0)
    s = daily(400, rng)
    D = design(s.to_numpy(), s.index)
    y = s.to_numpy()
    assert len(D.z) == 400 - MAX_LAG and D.z[0] == y[MAX_LAG]
    assert D.X_ar[0, 1] == y[MAX_LAG - 1] and D.X_ar[0, 7] == y[MAX_LAG - 7]
    assert D.X_slm[5, 10] == y[MAX_LAG + 5 - 28]
    assert D.dates[0] == s.index[MAX_LAG]
    assert D.X_slm.shape[1] == 1 + 10 + 6 + 4


def test_ets_tracks_a_pure_weekly_pattern():
    z = np.tile(np.array([1.0, 2.0, 3.0, 2.5, 2.0, 0.5, 0.0]), 80) + 5
    m = ETSModel().fit(z, 0, 400)
    pred = m.predict(z, np.arange(400, 420))
    np.testing.assert_allclose(pred, z[400:420], atol=1e-6)


def test_t_scale_recovers_the_scale():
    rng = np.random.default_rng(1)
    r = 0.3 * rng.standard_t(5, size=20000)
    assert abs(t_scale(r) / 0.3 - 1) < 0.03


def test_gbm_out_of_fold_residuals_cover_the_window():
    rng = np.random.default_rng(2)
    s = daily(700, rng)
    D = design(s.to_numpy(), s.index)
    m = GBMModel().fit(D.X_gbm, D.z, 100, 500)
    assert len(m.resid) == 400
    assert m.predict(D.X_gbm, np.arange(500, 510)).shape == (10,)


def test_student_t_fit_differs_only_for_linear_families():
    rng = np.random.default_rng(3)
    s = daily(800, rng, burst=(300, 330))
    D = design(s.to_numpy(), s.index)
    rows = np.arange(700, 710)
    for fam in ("ETS", "GBM"):
        f = fit_predict(fam, D, 0, 700, rows)
        np.testing.assert_array_equal(f.mean, f.mean_t)
        assert f.sd_t < f.sd_g  # heavy-tailed residuals: the t scale is below the RMS
    f = fit_predict("SLM", D, 0, 700, rows)
    assert not np.allclose(f.mean, f.mean_t)


def test_run_series_end_to_end():
    rng = np.random.default_rng(4)
    s = daily(1300, rng, shift_at=900, shift=0.5)
    per, dec = run_series(("elec", "XX", s))
    assert set(per.scheme) == set(SCHEMES)
    tot = per.groupby(["family", "dist", "scheme"]).sum_ll.sum().unstack("scheme")
    assert np.isfinite(tot.to_numpy()).all()
    n = per.groupby(["family", "dist", "scheme"]).n.sum().unique()
    assert len(n) == 1 and n[0] == 1300 - MAX_LAG - (2 * 364 + 28)
    # where the walk never stopped, B equals E
    if (dec.stopped_at == 0).all():
        np.testing.assert_allclose(tot["B"], tot["E"])
    assert {"switch_c0", "switch_c1645", "z_end"} <= set(dec.columns)
