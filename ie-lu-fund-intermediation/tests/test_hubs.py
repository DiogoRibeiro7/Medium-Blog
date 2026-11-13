"""Identities and sanity checks for the mirror-statistics test (run from the companion folder)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import fetch  # noqa: E402
import hubs as H  # noqa: E402

W = list(range(H.FIRST, H.LAST + 1))


@pytest.fixture(scope="module")
def d() -> H.Data:
    return H.load(HERE / "data")


def test_jsonstat_flattening():
    doc = {"id": ["geo", "time"], "size": [2, 3],
           "dimension": {"geo": {"category": {"index": {"IE": 0, "LU": 1}}},
                         "time": {"category": {"index": {"2023": 0, "2024": 1, "2025": 2}}}},
           "value": {"0": 1.0, "4": 5.0, "5": 6.0}}
    df = fetch.jsonstat_to_long(doc)
    assert df[["geo", "time", "value"]].values.tolist() == [["IE", "2023", 1.0], ["LU", "2024", 5.0],
                                                             ["LU", "2025", 6.0]]


def test_annual_requires_four_quarters():
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0],
                  index=["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", "2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"])
    a = H.annual(q)
    assert a.loc[2020] == 10.0 and np.isnan(a.loc[2021])


def test_luxembourg_residual_closes_the_eu_total(d):
    for kind, eu_get, side in (("pos", H.eu_position, "A_LE"), ("flow", H.eu_flow, "ASS")):
        ext, _ = H._extra_eu(d, kind, H.F5)
        intra = H._intra_eu(d, kind, H.F5)
        get = H.position if kind == "pos" else H.flow
        lu_ext = get(d, "LU", H.WRL, H.F5, side) - intra["LU"]
        total = ext.sum(axis=1) + lu_ext
        eu = eu_get(d, H.EXT, H.F5, side)
        ok = total.notna() & eu.notna()
        assert ok.sum() >= 10
        assert np.allclose(total[ok], eu[ok])


def test_mirror_split_adds_up_and_bounds_are_ordered(d):
    for h in H.HUBS:
        p = H.hub_positions(d, h).loc[H.years()]
        assert np.allclose(p.nonresident, p.eu_reported + p.eu_imputed + p.non_eu)
        assert (p.non_eu_low <= p.non_eu + 1e-6).all() and (p.non_eu <= p.non_eu_high + 1e-6).all()
        f = H.hub_flows(d, h)
        assert np.allclose(f.nonresident, f.eu + f.non_eu)
        assert (f.non_eu_low <= f.non_eu + 1e-6).all() and (f.non_eu <= f.non_eu_high + 1e-6).all()
    c = H.combined_positions(d)
    assert (c.non_eu_low <= c.non_eu + 1e-6).all() and (c.non_eu <= c.non_eu_high + 1e-6).all()


def test_members_add_up_to_the_eu_for_us_securities(d):
    members = H.us_purchases(d).apply(H.annual).loc[W].sum(axis=1).sum()
    eu = H.annual(H.eu_flow(d, H.US, H.F, "ASS")).loc[W].sum()
    assert abs(members / eu - 1) < 0.01


def test_ownership_shares_and_attribution_are_bounded(d):
    usa = H.us_purchases(d).apply(H.annual)
    om = pd.DataFrame({h: H.non_eu_ownership(d, h) for h in H.HUBS})
    hi = pd.DataFrame({h: H.non_eu_ownership(d, h, "high") for h in H.HUBS})
    lo = H.ownership_low_joint(d, usa.reindex(H.years()))
    for frame in (om, hi, lo):
        v = frame.loc[H.years()].values
        assert np.nanmin(v) >= 0 and np.nanmax(v) <= 1
    a = H.attribution(usa, H.mid_year(om)).loc[W].sum()
    a_lo = H.attribution(usa, H.mid_year(lo)).loc[W].sum()
    a_hi = H.attribution(usa, H.mid_year(hi)).loc[W].sum()
    hub = usa.loc[W, list(H.HUBS)].sum(axis=1).sum()
    assert a_lo <= a <= a_hi <= hub


def test_ols_hac_recovers_coefficients():
    rng = np.random.default_rng(20261113)
    x = rng.normal(size=(400, 2))
    y = 1.0 + 0.5 * x[:, 0] - 0.25 * x[:, 1] + rng.normal(scale=0.1, size=400)
    b, se, r2 = H.ols_hac(y, x)
    ref = np.linalg.lstsq(np.column_stack([np.ones(400), x]), y, rcond=None)[0]
    assert np.allclose(b, ref)
    assert np.allclose(b, [1.0, 0.5, -0.25], atol=0.03)
    assert (se > 0).all() and r2 > 0.9
