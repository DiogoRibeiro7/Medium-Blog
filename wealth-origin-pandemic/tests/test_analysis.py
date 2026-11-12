import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analysis as A  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    p, meta = A.build_panel(A.load_countries())
    p = A.add_counts(p, meta["forbes"])
    return A.add_exposure(p, A.load_exposure())


def test_ols_cluster_equals_hc1_with_singleton_clusters():
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([0.5, -1.0]) + rng.standard_normal(n) * (1 + np.abs(X[:, 1]))
    b, se_hc1 = A.ols(y, X)
    _, se_cr1 = A.ols(y, X, np.arange(n))
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert np.allclose(se_cr1, se_hc1)


def test_copied_snapshot_matches_provenance():
    text = (HERE / "data" / "PROVENANCE.txt").read_text()
    for name, digest in re.findall(r"^\s+(\S+\.csv)\s+sha256 ([0-9a-f]{64})$", text, flags=re.M):
        assert hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest() == digest, name


def test_exposure_by_year_absorbs_a_differential_shock():
    """A shock proportional to exposure in some years biases the plain fit and not the interacted one."""
    rng = np.random.default_rng(3)
    rows = []
    for c in range(60):
        e = rng.standard_normal()
        for t in range(2011, 2024):
            x = rng.standard_normal() + 0.8 * e * (t >= 2019)  # composition moves with exposure late on
            shock = -2.0 * e * (t in (2019, 2020))
            rows.append(dict(iso3=str(c), year=t, z=e, x=x, s=x + rng.standard_normal(),
                             y=-0.5 * x + shock + 0.3 * rng.standard_normal()))
    d = pd.DataFrame(rows)
    plain = A.fit(d, "y", ["x", "s"], [], "fe")
    inter = A.fit_expo(d, "y", ["x", "s"], [], ["z"], "fe", common=["z"])
    assert abs(plain["b_x"] + 0.5) > 0.1
    assert inter["b_x"] == pytest.approx(-0.5, abs=0.05)


def test_every_billionaire_country_has_all_four_measures(panel):
    bc = panel[panel["billionaire_country"]].drop_duplicates("iso3")
    assert bc[list(A.EXPOSURES)].notna().all().all()


def test_two_list_average(panel):
    p = A.prep(panel).sort_values(["iso3", "year"])
    us = p[p["iso3"] == "USA"].set_index("year")
    assert np.isnan(us.loc[2011, "x_inh_avg2"])
    assert us.loc[2020, "x_inh_avg2"] == pytest.approx((us.loc[2019, "x_inh"] + us.loc[2020, "x_inh"]) / 2)


def test_reproduces_previous_sector_controlled_estimate(panel):
    d = A.prep(A.sample(panel, "billionaire_countries"))
    r = A.fit(d, "g_tfp_3", ["x_inh", "x_sum"], A.BASE_CTRL, "fe")
    assert r["b_x_inh"] == pytest.approx(-1.048, abs=5e-4)
