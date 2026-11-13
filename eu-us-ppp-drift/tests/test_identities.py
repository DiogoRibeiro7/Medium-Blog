"""Identities and approximations the analysis relies on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drift as dr  # noqa: E402

A, B = 2005, 2019


@pytest.fixture(scope="module")
def d():
    return dr.load()


def test_drift_is_one_in_the_reference_year(d):
    for agg in dr.AGGREGATES:
        assert abs(dr.group_drift(d, agg, dr.EU_GROUP).loc[dr.BASE]) < 1e-12


def test_split_into_deflator_and_ppp_is_exact(d):
    for agg in dr.AGGREGATES:
        direct = dr.group_drift(d, agg, dr.EU_GROUP)
        parts = dr.group_parts(d, agg, dr.EU_GROUP)
        both = pd.concat([direct, parts.drift], axis=1).dropna()
        assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-10)


def test_zero_adjustment_leaves_the_drift_unchanged(d):
    zero = pd.DataFrame(0.0, index=d["V"]["GDP"].index, columns=["USA", *dr.EU_GROUP])
    a = dr.group_drift(d, "GDP", dr.EU_GROUP)
    b = dr.group_drift(d, "GDP", dr.EU_GROUP, adj=zero)
    assert np.allclose(a.dropna(), b.dropna())


def test_a_uniform_us_adjustment_moves_the_drift_one_for_one(d):
    # raising the US deflator by x in every year after the base lowers the drift by x
    idx = d["V"]["GDP"].index
    step = pd.DataFrame(0.0, index=idx, columns=["USA", *dr.EU_GROUP])
    step.loc[idx > dr.BASE, "USA"] = 0.01
    moved = dr.group_drift(d, "GDP", dr.EU_GROUP, adj=dr.cumulate(step))
    base = dr.group_drift(d, "GDP", dr.EU_GROUP)
    years = 2024 - dr.BASE
    assert abs((base - moved).loc[2024] - 0.01 * years) < 1e-10


def test_country_drift_matches_group_drift_for_a_single_country(d):
    for agg in dr.AGGREGATES:
        cd = dr.country_drift(d, agg)["DEU"]
        gd = dr.group_drift(d, agg, ["DEU"])
        both = pd.concat([cd, gd], axis=1).dropna()
        assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-10)


def test_sections_add_up_to_total_value_added(d):
    tot = d["va_V"]["_T"]
    parts = sum(d["va_V"][a].reindex(columns=tot.columns).fillna(0.0) for a in dr.SECTIONS)
    ratio = (parts / tot).loc[A:B, ["USA", *dr.EU_GROUP]]
    assert ratio.min().min() > 0.995 and ratio.max().max() < 1.005


def test_tornqvist_of_sections_reproduces_the_gva_deflator(d):
    vT, lT = d["va_V"]["_T"], d["va_LR"]["_T"]
    acc = 0.0
    for a in dr.SECTIONS:
        v, lr = d["va_V"][a], d["va_LR"][a]
        acc = acc + (dr.torn_share(v, vT) * dr.dlog(v / lr)).fillna(0.0)
    gap = (acc - dr.dlog(vT / lT)).loc[A + 1:B].sum() * 100
    assert abs(gap["USA"]) < 0.5
    assert gap.reindex(dr.EU_GROUP).abs().max() < 3.0


def test_tornqvist_of_coicop_divisions_reproduces_the_consumption_deflator(d):
    t5 = pd.read_csv(dr.DATA / "oecd_t5.csv", low_memory=False)
    divisions = [f"CP{i:02d}" for i in range(1, 13)]
    v = {k: dr._pivot(t5, EXPENDITURE=k, PRICE_BASE="V") for k in ["_T", *divisions]}
    lr = {k: dr._pivot(t5, EXPENDITURE=k, PRICE_BASE="LR") for k in ["_T", *divisions]}
    acc = sum((dr.torn_share(v[k], v["_T"]) * dr.dlog(v[k] / lr[k])).fillna(0.0) for k in divisions)
    gap = (acc - dr.dlog(v["_T"] / lr["_T"])).loc[A + 1:B].sum() * 100
    assert abs(gap["USA"]) < 0.5
    assert gap.reindex(["DEU", "FRA", "ITA", "ESP", "POL"]).abs().max() < 1.5


def test_eurostat_and_oecd_agree_on_the_us_household_consumption_ppp(d):
    es = pd.read_csv(dr.DATA / "eurostat_ppp.csv")
    e = es[(es.na_item == "PPP_EU27_2020") & (es.geo == "US") & (es.ppp_cat == "E011")].set_index("time").value
    o = 1 / d["PPP"]["HFCE"]["EU"]
    assert abs(np.log(e.loc[B] / e.loc[A]) - np.log(o.loc[B] / o.loc[A])) < 0.005
