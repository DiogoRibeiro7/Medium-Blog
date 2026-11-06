"""Tests of the identities, the fixed-effects estimator, the concentration measures and the JSON-stat reader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompose import decompose, finish  # noqa: E402
from dependencies import concentration  # noqa: E402
from fetch import jsonstat_to_long  # noqa: E402
from regions import fe_ols  # noqa: E402
from turn import required_growth  # noqa: E402


def toy_shares() -> pd.DataFrame:
    years = np.arange(2000, 2025)
    t = years - 2000
    d = pd.DataFrame(
        {
            "emp_m": 100 * np.exp(-0.01 * t),
            "emp_t": 500 * np.exp(0.005 * t),
            "nom_m": 40 * np.exp(0.01 * t),
            "nom_t": 200 * np.exp(0.03 * t),
            "real_m": 35 * np.exp(0.02 * t),
            "real_t": 210 * np.exp(0.015 * t),
        },
        index=pd.Index(years, name="year"),
    )
    return finish(d, "XX")


def test_finish_rescales_real_share_to_nominal_in_2020() -> None:
    s = toy_shares().set_index("year")
    assert s.loc[2020, "real_share"] == pytest.approx(s.loc[2020, "nom_share"])


def test_decomposition_identities_hold_exactly() -> None:
    r = decompose(toy_shares(), 2000, 2024).iloc[0]
    assert r.dlog_nom_share == pytest.approx(r.dlog_real_share + r.dlog_rel_price)
    assert r.dlog_emp_share == pytest.approx(r.dlog_real_share - r.dlog_rel_prod)
    # the toy series have known growth rates: real share +0.005 a year, employment share -0.015 a year
    assert r.dlog_real_share == pytest.approx(0.005 * 24)
    assert r.dlog_emp_share == pytest.approx(-0.015 * 24)


def test_fe_ols_recovers_slope_and_ignores_group_levels() -> None:
    rng = np.random.default_rng(1)
    n_groups, per = 30, 12
    g = np.repeat(np.arange(n_groups), per)
    level = rng.normal(0, 5, n_groups)[g]
    x = rng.normal(0, 1, g.size) + 0.5 * level  # correlated with the group effect, so pooled OLS is biased
    y = level + 0.7 * x + rng.normal(0, 0.5, g.size)
    d = pd.DataFrame({"country": g, "x": x, "y": y})
    r = fe_ols(d, "y", ["x"])
    assert r["n"] == g.size and r["groups"] == n_groups
    assert r["b_x"] == pytest.approx(0.7, abs=0.05)
    assert 0 < r["se_x"] < 0.1
    pooled = np.polyfit(x, y, 1)[0]
    assert abs(pooled - 0.7) > 0.2


def test_concentration_measures() -> None:
    c = pd.DataFrame(
        {
            "product": ["8507"] * 3 + ["2805"] * 2,
            "year": [2024] * 5,
            "partner": ["CN", "US", "KR", "CN", "JP"],
            "value": [60.0, 30.0, 10.0, 50.0, 50.0],
        }
    )
    r = concentration(c).set_index("product")
    assert r.loc["8507", "hhi"] == pytest.approx(0.6**2 + 0.3**2 + 0.1**2)
    assert r.loc["8507", "china_share"] == pytest.approx(0.6)
    assert r.loc["8507", "top_partner"] == "CN"
    assert r.loc["2805", "hhi"] == pytest.approx(0.5)
    assert r.loc["2805", "suppliers"] == 2


def test_jsonstat_reader_maps_flat_index_to_categories() -> None:
    doc = {
        "id": ["geo", "time"],
        "size": [2, 3],
        "dimension": {
            "geo": {"category": {"index": {"PT": 0, "ES": 1}}},
            "time": {"category": {"index": {"2000": 0, "2001": 1, "2002": 2}}},
        },
        "value": {"0": 1.0, "2": 3.0, "4": 5.0},
        "status": {"4": "p"},
    }
    d = jsonstat_to_long(doc)
    got = {(r.geo, r.time): (r.value, r.flag) for r in d.itertuples()}
    assert got == {("PT", "2000"): (1.0, ""), ("PT", "2002"): (3.0, ""), ("ES", "2001"): (5.0, "p")}


def test_required_growth_reaches_the_target() -> None:
    g = required_growth(0.143, 0.20, 11)
    assert 0.143 * np.exp(11 * g) == pytest.approx(0.20)
