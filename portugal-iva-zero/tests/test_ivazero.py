"""Checks on the snapshot and the estimators."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ivazero as iz  # noqa: E402


def test_fully_covered_subclasses_have_the_six_percent_wedge() -> None:
    t = iz.classify(iz.tax_wedge())
    tr = t[t["group"] == "treated"]
    assert len(tr) >= 10
    assert np.allclose(tr["entry"], -iz.VAT_6, atol=0.003)
    assert np.allclose(tr["exit"], iz.VAT_6, atol=0.003)
    assert (t.loc[t["group"] == "control", "entry"].abs() < 1e-3).all()


def test_group_gap_equals_two_way_fixed_effects() -> None:
    # Balanced panel: the event coefficient with subclass and month fixed effects is the
    # difference between the group means of the change since the base month.
    rng = np.random.default_rng(1)
    units = [f"s{i}" for i in range(8)]
    months = ["m0", "m1", "m2"]
    y = pd.DataFrame(rng.normal(size=(8, 3)), index=units, columns=months)
    treated = units[:3]
    r = y.sub(y["m0"], axis=0)
    gap = iz.group_gap(r, treated, units[3:])["gap"]
    long = y.stack().rename("y").reset_index()
    long.columns = ["unit", "month", "y"]
    X = pd.get_dummies(long["unit"]).astype(float)
    X = X.join(pd.get_dummies(long["month"], drop_first=True).astype(float))
    for m in months[1:]:
        X[f"ev_{m}"] = ((long["month"] == m) & long["unit"].isin(treated)).astype(float)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), long["y"].to_numpy(), rcond=None)
    coef = dict(zip(X.columns, beta))
    for m in months[1:]:
        assert coef[f"ev_{m}"] == pytest.approx(gap[m])


def test_across_countries_removes_shocks_common_to_a_subclass() -> None:
    periods = ["2023-01", "2023-02", "2023-03"]
    shock = {"a": [0.0, 0.0, 0.05], "b": [0.0, 0.0, -0.02], "c": [0.0, 0.0, 0.01]}
    rows = []
    for geo in ("PT", "X1", "X2"):
        for s, v in shock.items():
            extra = -0.06 if (geo == "PT" and s == "a") else 0.0
            rows.append({"geo": geo, "coicop": s,
                         **{p: v[i] + (extra if i == 2 else 0.0) for i, p in enumerate(periods)}})
    lp = pd.DataFrame(rows).set_index(["geo", "coicop"])
    est = iz.across_countries(lp, "PT", ["X1", "X2"], ["a"], ["b", "c"], "2023-02")
    assert est.loc["2023-03", "gap"] == pytest.approx(-0.06)


def test_synthetic_control_weights_are_convex_and_recover_a_mixture() -> None:
    rng = np.random.default_rng(2)
    y = pd.DataFrame(rng.normal(size=(4, 30)).cumsum(axis=1), index=["T", "A", "B", "C"],
                     columns=[f"p{i:02d}" for i in range(30)])
    y.loc["T"] = 0.7 * y.loc["A"] + 0.3 * y.loc["B"]
    v, gap = iz.synthetic_control(y, "T", list(y.columns[:20]))
    assert v.sum() == pytest.approx(1.0)
    assert (v >= 0).all()
    assert v["A"] == pytest.approx(0.7, abs=1e-3)
    assert gap.abs().max() < 1e-3


def test_ols_matches_polyfit() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    y = 0.8 * x + rng.normal(size=40)
    b, _a, se = iz.ols(x, y)
    assert b == pytest.approx(np.polyfit(x, y, 1)[0])
    assert se > 0


def test_comparison_countries_have_no_tax_moves() -> None:
    wedge = iz.tax_wedge()
    t = iz.classify(wedge)
    codes = list(t.index[t["group"].isin(["treated", "control"])])
    comp, screen = iz.donors(wedge, codes)
    assert "PT" not in comp and "ES" not in comp and "HU" not in comp
    assert (screen.loc[comp, "largest_move"] < iz.SCREEN_TOL).all()


def test_incidence_shares_sum_to_one_and_fall_with_income() -> None:
    inc = pd.read_csv(iz.RESULTS / "incidence_by_quintile.csv", index_col="quintile")
    assert inc["share_of_total"].sum() == pytest.approx(1.0)
    assert (inc["saving_pct_consumption"].diff().dropna() <= 1e-12).all()
