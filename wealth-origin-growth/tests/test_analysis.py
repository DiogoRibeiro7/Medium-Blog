import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as A  # noqa: E402


def test_ols_cluster_equals_hc1_with_singleton_clusters():
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([0.5, -1.0]) + rng.standard_normal(n) * (1 + np.abs(X[:, 1]))
    b, se_hc1 = A.ols(y, X)
    b2, se_cr1 = A.ols(y, X, np.arange(n))
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert np.allclose(se_cr1, se_hc1)


def test_fast_fit_matches_reference():
    rng = np.random.default_rng(1)
    n = 400
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    g = rng.integers(0, 40, n)
    y = X @ np.array([1.0, 0.3, -0.2]) + rng.standard_normal(n)
    b_ref, se_ref = A.ols(y, X, g)
    codes = pd.factorize(g)[0]
    b, se = A._fit_fast(y, X, 1, codes)
    assert b == pytest.approx(b_ref[1]) and se == pytest.approx(se_ref[1])


def test_difference_parametrisation_recovers_gap():
    """Regressing on (x_inh, x_inh + x_self) returns b_inh - b_self on x_inh."""
    rng = np.random.default_rng(2)
    n = 2_000
    d = pd.DataFrame({"iso3": rng.integers(0, 60, n).astype(str), "year": rng.integers(2011, 2024, n),
                      "x_inh": rng.exponential(1.0, n), "x_self": rng.exponential(1.0, n)})
    d["x_sum"] = d["x_inh"] + d["x_self"]
    d["y"] = -0.7 * d["x_inh"] + 0.2 * d["x_self"] + rng.standard_normal(n)
    for est in ("fe", "pooled"):
        sep = A.design(d, "y", ["x_inh", "x_self"], [], est)
        b, _ = A.ols(sep["y"], sep["X"], sep["groups"])
        dif = A.design(d, "y", ["x_inh", "x_sum"], [], est)
        bd, _ = A.ols(dif["y"], dif["X"], dif["groups"])
        assert bd[dif["j0"]] == pytest.approx(b[sep["j0"]] - b[sep["j0"] + 1], abs=1e-9)
        assert bd[dif["j0"]] == pytest.approx(-0.9, abs=0.1)


def test_forbes_labels_map_to_iso3():
    countries = A.load_countries()
    f, unmapped = A.forbes_iso3(countries)
    assert unmapped == []
    assert f["iso3"].str.len().eq(3).all()
    assert not f.duplicated(["iso3", "year"]).any()


def test_panel_shares_add_up():
    p, _ = A.build_panel(A.load_countries())
    assert np.allclose(p["total_pct"], p["inherited_pct"] + p["self_made_pct"])
    assert (p.loc[p["n"] == 0, "total_pct"] == 0).all()
    assert p["year"].between(2011, 2023).all()
