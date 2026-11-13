"""Identities and sanity checks for the category-level drift (run from the companion folder)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import catdrift as C  # noqa: E402
import fetch  # noqa: E402


@pytest.fixture(scope="module")
def d() -> C.Data:
    return C.load(HERE / "data")


def test_jsonstat_flattening():
    doc = {"id": ["geo", "time"], "size": [2, 2],
           "dimension": {"geo": {"category": {"index": {"PT": 0, "ES": 1}}},
                         "time": {"category": {"index": {"2005": 0, "2019": 1}}}},
           "value": {"0": 1.0, "3": 4.0}}
    df = fetch.jsonstat_to_long(doc)
    assert df[["geo", "time", "value"]].values.tolist() == [["PT", "2005", 1.0], ["ES", "2019", 4.0]]


def test_drift_is_deflator_minus_ppp(d):
    p = C.drift_parts(d, "A0111", "CP11")
    assert np.allclose(p.drift, p.deflator - p.ppp, equal_nan=True)
    assert p.drift.notna().sum() == len(C.EUROPE)


def test_slope_does_not_depend_on_the_reference(d):
    x = C.price_level(d)
    p = C.drift_parts(d, "E011", "TOTAL")
    shifted = p.drift + 7.3          # another reference adds a constant to every economy
    assert np.isclose(C.ols(p.drift, x)["slope"], C.ols(shifted, x)["slope"])


def test_decomposition_is_exact_and_slopes_add_up(d):
    x = C.price_level(d)
    T = C.decompose(d)
    T = T[[c for c in T.columns if not c.startswith("share_")]].astype(float)
    assert np.allclose(T.drift, T.drop(columns="drift").sum(axis=1))
    parts = sum(C.ols(T[c], x)["slope"] for c in T.columns.drop("drift"))
    assert np.isclose(parts, C.ols(T.drift, x)["slope"])


def test_household_drift_matches_direct_calculation(d):
    T = C.decompose(d)
    p = C.drift_parts(d, "E011", "TOTAL")
    assert np.allclose(T.drift.astype(float), p.drift.loc[T.index])


def test_tornqvist_of_one_item_is_its_price():
    c = pd.Series([100.0, 110.0, 130.0], index=[2005, 2006, 2007])
    p = pd.Series([1.0, 1.05, 1.2], index=[2005, 2006, 2007])
    assert np.allclose(C.tornqvist([(c, p)]).values, (p / p.iloc[0]).values)


def test_ols_matches_least_squares():
    rng = np.random.default_rng(20261113)
    x = pd.Series(rng.normal(size=60))
    y = 2.0 - 3.0 * x + pd.Series(rng.normal(scale=0.2, size=60))
    o = C.ols(y, x)
    ref = np.polyfit(x, y, 1)
    assert np.isclose(o["slope"], ref[0]) and np.isclose(o["intercept"], ref[1])
    assert abs(o["slope"] + 3.0) < 0.1 and o["se"] > 0


def test_shares_sum_to_one(d):
    T = C.decompose(d)
    s = T[[c for c in T.columns if c.startswith("share_")]].astype(float).sum(axis=1)
    assert np.allclose(s, 1.0)
