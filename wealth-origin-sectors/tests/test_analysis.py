import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as A  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return A.build_panel(A.load_countries())


def test_ols_cluster_equals_hc1_with_singleton_clusters():
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([0.5, -1.0]) + rng.standard_normal(n) * (1 + np.abs(X[:, 1]))
    b, se_hc1 = A.ols(y, X)
    _, se_cr1 = A.ols(y, X, np.arange(n))
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert np.allclose(se_cr1, se_hc1)


def test_fast_fit_matches_reference():
    rng = np.random.default_rng(1)
    n = 400
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    g = rng.integers(0, 40, n)
    y = X @ np.array([1.0, 0.3, -0.2]) + rng.standard_normal(n)
    b_ref, se_ref = A.ols(y, X, g)
    b, se = A._fit_fast(y, X, 1, pd.factorize(g)[0])
    assert b == pytest.approx(b_ref[1]) and se == pytest.approx(se_ref[1])


def test_every_label_maps(built):
    _, meta = built
    assert meta["unmapped_citizenship"] == []
    # only the four non-industry labels and empty entries fall outside the 18 Forbes categories
    assert set(meta["unclassified_industry"]) <= {"(none)", "Billionaire", "Business", "Celebrity", "Politics"}
    assert meta["forbes"]["sector"].nunique() == 19


def test_shift_share_identity(built):
    """Inherited share = sector-predicted part + within-sector remainder, with each part in range."""
    p, _ = built
    w = p[p["n"] > 0]
    assert np.allclose(w["inh_share"], w["pred_share"] + w["resid_share"])
    assert w["pred_share"].between(0, 1).all() and w["inh_share"].between(0, 1).all()
    groups = w[[f"s_{k}" for k in A.GROUPS]].sum(axis=1)
    assert (groups <= 1 + 1e-9).all()


def test_leave_country_out_rho_excludes_own_fortunes(built):
    """A country that is the only holder of a sector's fortunes gets no prediction from itself."""
    _, meta = built
    f = meta["forbes"]
    known = f[f["origin"].isin(["inherited", "self_made"])]
    rho = meta["rho"]
    r = rho[(rho["iso3"] == "USA") & (rho["sector"] == "Technology")].iloc[0]
    other = known[(known["iso3"] != "USA") & (known["sector"] == "Technology")]
    expected = other.loc[other["origin"] == "inherited", "worth_m"].sum() / other["worth_m"].sum()
    assert r["rho_loo"] == pytest.approx(expected)


def test_panel_matches_previous_definitions(built):
    p, _ = built
    assert np.allclose(p["total_pct"], p["inherited_pct"] + p["self_made_pct"])
    assert (p.loc[p["n"] == 0, "total_pct"] == 0).all()
    assert p["year"].between(2011, 2023).all()


def test_reproduces_previous_headline(built):
    """Without sector controls, the TFP estimate is the one reported by wealth-origin-growth."""
    p, _ = built
    d = A.prep(A.sample(p, "billionaire_countries"))
    r = A.fit(d, "g_tfp_3", ["x_inh", "x_sum"], ["ln_y"], "fe")
    assert r["b_x_inh"] == pytest.approx(-1.167, abs=5e-4)
