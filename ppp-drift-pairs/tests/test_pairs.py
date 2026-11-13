"""Identities the analysis relies on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pairs as pr  # noqa: E402


@pytest.fixture(scope="module")
def d():
    return pr.load()


def test_drift_is_zero_in_the_reference_year(d):
    for agg in pr.AGGREGATES:
        row = pr.drift_vs(d, agg).loc[pr.BASE].dropna()
        assert np.allclose(row, 0.0)


def test_pairs_are_differences_of_country_scores(d):
    cd = pr.drift_vs(d, "HFCE")
    for i, j in (("DEU", "JPN"), ("FRA", "KOR"), ("POL", "CAN"), ("SWE", "NOR"), ("MEX", "AUS")):
        both = pd.concat([pr.pair_drift(d, "HFCE", i, j), cd[i] - cd[j]], axis=1).dropna()
        assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9)


def test_pair_table_has_every_pair_once(d):
    s = pr.programme_scores(pr.drift_vs(d, "HFCE"), 2005, 2019)
    pt = pr.pairs_table(s)
    n = len(s)
    assert len(pt) == n * (n - 1) // 2
    e, o = (s.group == "Eurostat").sum(), (s.group == "OECD").sum()
    counts = pt.kind.value_counts()
    assert counts["within Eurostat"] == e * (e - 1) // 2
    assert counts["within OECD"] == o * (o - 1) // 2
    assert counts["across"] == e * o


def test_block_model_gap_is_the_difference_of_group_means(d):
    s = pr.programme_scores(pr.drift_vs(d, "GDP"), 2005, 2019)
    m = pr.block_model(s)
    means = s.groupby("group").drift.mean()
    assert abs(m["beta"][1] - (means["Eurostat"] - means["OECD"])) < 1e-9
    between = sum(len(g) * (g.mean() - s.drift.mean()) ** 2 for _, g in s.groupby("group").drift)
    assert abs(m["r2"] - between / ((s.drift - s.drift.mean()) ** 2).sum()) < 1e-9


def test_permutation_p_value_is_reproducible(d):
    s = pr.programme_scores(pr.drift_vs(d, "HFCE"), 2005, 2019)
    a, b = pr.permutation_p(s, reps=500), pr.permutation_p(s, reps=500)
    assert a == b and 0 < a <= 1


def test_eurostat_and_oecd_ppps_agree_for_european_pairs(d):
    es = pd.read_csv(pr.DATA / "eurostat_ppp.csv")
    es = es[(es.na_item == "PPP_EU27_2020") & (es.ppp_cat == "E011")]
    es["time"] = es.time.astype(int)
    geo = {"DE": "DEU", "FR": "FRA", "PL": "POL", "SE": "SWE", "US": "USA", "JP": "JPN"}
    p = es[es.geo.isin(geo)].pivot_table(index="time", columns="geo", values="value").rename(columns=geo)
    o = d["PPP"]["HFCE"]
    for c in ("FRA", "POL", "SWE", "USA", "JPN"):
        a = np.log((p[c] / p["DEU"]).loc[2019] / (p[c] / p["DEU"]).loc[2005])
        b = np.log((o[c] / o["DEU"]).loc[2019] / (o[c] / o["DEU"]).loc[2005])
        assert abs(a - b) < 1e-4
