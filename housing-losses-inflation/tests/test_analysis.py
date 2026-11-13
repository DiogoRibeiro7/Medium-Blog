import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as A  # noqa: E402


def test_drawdowns_peak_trough_recovery_and_open_episode():
    s = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3, 1.1], index=range(2000, 2006))
    eps = A.drawdowns(s, 0.10)
    assert len(eps) == 2
    first, second = eps
    assert (first["peak"], first["trough"], first["recovery"]) == (2001, 2002, 2004)
    assert first["depth"] == pytest.approx(0.25)
    assert (second["peak"], second["trough"], second["recovery"]) == (2004, 2005, None)
    assert A.drawdowns(s, 0.30) == []


def test_decompose_caps_visible_part_at_the_real_loss():
    assert A.decompose(-0.2, 0.1) == pytest.approx((0.0, 0.2))        # nominal up: all hidden
    assert A.decompose(-0.2, -0.05) == pytest.approx((0.05, 0.15))    # partly visible
    assert A.decompose(-0.2, -0.3) == pytest.approx((0.2, 0.0))       # deflation: all visible


def test_forward_sum_and_segments_respect_gaps():
    s = pd.Series([0.1, 0.2, np.nan, 0.3, 0.4], index=range(5))
    f = A.forward_sum(s, 2)
    assert np.isnan(f.loc[0]) and np.isnan(f.loc[1]) and f.loc[2] == pytest.approx(0.7)
    segs = A.segments(pd.Series([1.0, 2.0, np.nan, 3.0], index=[10, 11, 12, 13]))
    assert [list(x.index) for x in segs] == [[10, 11], [13]]


def test_price_episode_identity():
    years = range(1990, 2001)
    nom = [0.0, 0.05, 0.02, -0.03, 0.01, 0.00, -0.02, 0.06, 0.08, 0.04, 0.03]
    inf = [0.0, 0.02, 0.10, 0.12, 0.08, 0.06, 0.03, 0.02, 0.02, 0.02, 0.02]
    panel = pd.DataFrame({"iso": "X", "year": list(years), "dlp": nom, "dlc": inf})
    panel.loc[0, ["dlp", "dlc"]] = np.nan
    E = A.price_episodes(panel, "year", "dlp", "dlc", 0.10, 1)
    assert len(E) == 1
    r = E.iloc[0]
    assert r.real_log == pytest.approx(r.nominal_log - r.inflation_log)
    assert r.visible + r.hidden == pytest.approx(-r.real_log)
    assert 0 <= r.hidden_share <= 1


def test_cluster_bootstrap_interval_covers_point_estimate():
    rng = np.random.default_rng(0)
    counts = np.column_stack([np.full(30, 10.0), rng.integers(2, 6, 30).astype(float)])
    est = counts[:, 1].sum() / counts[:, 0].sum()
    lo, hi = A.cluster_bootstrap(counts, 1, 0, rng, n_boot=500)
    assert lo <= est <= hi


def test_ols_cluster_coefficients_match_lstsq():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
    y = X @ np.array([1.0, -0.5, 2.0]) + rng.standard_normal(200)
    b, se = A.ols_cluster(y, X, rng.integers(0, 20, 200))
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert np.all(se > 0)


def test_rent_return_definition_in_extension():
    p = pd.DataFrame({"iso": ["X"], "year": [2020], "housing_rent_yd": [0.04]})
    oa = pd.DataFrame({"iso": "X", "year": range(2019, 2026),
                       "HPI": [100, 100, 110, 121, 121, 121, 121],
                       "defl": [1, 1, 1.05, 1.10, 1.10, 1.10, 1.10],
                       "HPI_RPI": [100, 100, 110, 121, 121, 121, 121]})
    X = A.extend_returns_after_2020(p, oa)
    r = X[X.year == 2021].iloc[0]
    assert r.net_yield == pytest.approx(0.04 * 100 / 110)
    assert r.total_return == pytest.approx(0.10 + r.net_yield * 1.10)
    assert r.real_total_return == pytest.approx((1 + r.total_return) / 1.05 - 1)
