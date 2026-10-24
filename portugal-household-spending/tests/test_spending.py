"""Checks on the snapshot and the spending measures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spending as sp  # noqa: E402


def test_coicop_divisions_add_up_to_total_consumption() -> None:
    w = sp.national_accounts()
    gap = (w[sp.DIVISIONS].sum(axis=1) - w["TOTAL"]).abs() / w["TOTAL"]
    assert gap.max() < 0.01


def test_budget_survey_shares_add_up_to_one_thousand() -> None:
    q = sp.budget_survey_quintiles()
    assert (q[sp.DIVISIONS].sum(axis=1) - 1000).abs().max() <= 3


def test_core_share_excluding_imputed_rent_by_hand() -> None:
    w = sp.national_accounts().loc[2015]
    expected = w[sp.CORE].sum() / (w["TOTAL"] - w["CP042"])
    assert sp.na_shares().loc[2015, "core_ex_imputed"] == pytest.approx(expected)


def test_quintile_weights_sum_to_one_and_inflation_is_a_weighted_mean() -> None:
    qi = sp.quintile_inflation(2015, 2025, 2015)
    p = sp.hicp()
    rel = p[sp.DIVISIONS].loc[2025] / p[sp.DIVISIONS].loc[2015] - 1
    for q in sp.QUINTILES:
        assert rel.min() - 1e-12 <= qi.loc[q, "inflation"] <= rel.max() + 1e-12


def test_basic_needs_price_index_starts_at_one() -> None:
    bp = sp.basic_needs_price(2015, 2025, 2015)
    assert bp.loc[2015, "basic_needs"] == pytest.approx(1.0)
    assert bp.loc[2015, "all_items"] == pytest.approx(1.0)


def test_2020_wave_has_no_imputed_rent_split() -> None:
    hb = sp.hbs_shares()
    assert hb.xs(2020, level="year")["core_ex_imputed"].isna().all()
    assert hb.xs(2015, level="year")["core_ex_imputed"].notna().all()
