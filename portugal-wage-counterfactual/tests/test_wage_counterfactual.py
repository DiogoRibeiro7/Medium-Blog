"""Checks on the linked inputs and the counterfactual arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wage_counterfactual as wc  # noqa: E402


@pytest.fixture(scope="module")
def inputs() -> pd.DataFrame:
    return wc.build_inputs()


def test_inputs_cover_the_whole_period(inputs: pd.DataFrame) -> None:
    assert inputs.index.min() == wc.START and inputs.index.max() == wc.END
    assert inputs.loc[wc.START + 1:, ["infl", "g_person", "g_hour", "actual"]].notna().all().all()


def test_linked_series_equals_eurostat_after_the_link(inputs: pd.DataFrame) -> None:
    euro = wc.load_eurostat()
    after = inputs.loc[wc.LINK:, "actual"] / 1000
    assert np.allclose(after, euro.loc[wc.LINK:wc.END, "comp_per_emp"])


def test_ameco_and_eurostat_agree_at_the_link() -> None:
    ameco, euro = wc.load_ameco(), wc.load_eurostat()
    gap = euro.loc[wc.LINK, "comp_per_emp"] / ameco.loc[wc.LINK, "comp_per_emp"] - 1
    assert abs(gap) < 0.05
    aws_gap = euro.loc[wc.LINK, "adj_wage_share"] - ameco.loc[wc.LINK, "adj_wage_share"]
    assert abs(aws_gap) < 0.5


def test_additive_recursion_by_hand(inputs: pd.DataFrame) -> None:
    level = inputs.loc[2000, "actual"]
    for year in range(2001, wc.END + 1):
        level *= 1 + inputs.loc[year, "infl"] + inputs.loc[year, "g_person"]
    assert wc.counterfactual(inputs, 2000).loc[wc.END] == pytest.approx(level)


def test_base_year_is_unchanged_and_end_year_is_reached(inputs: pd.DataFrame) -> None:
    path = wc.counterfactual(inputs, 1990, start_level=100.0)
    assert path.loc[1990] == 100.0
    assert path.index.max() == wc.END


def test_multiplicative_exceeds_additive_when_both_rates_positive(inputs: pd.DataFrame) -> None:
    assert wc.ratio(inputs, wc.START, rule="multiplicative") > wc.ratio(inputs, wc.START)


@pytest.mark.parametrize("base", [1975, 1985, 2000, 2010])
def test_decomposition_identity(inputs: pd.DataFrame, base: int) -> None:
    # Exact from 1995 on; before 1995 up to the rounding of the AMECO extract
    # and the separate 1995 links of the three series.
    d = wc.decomposition(inputs, base)
    tol = 1e-9 if base >= wc.LINK else 2e-3
    assert d["cf_over_actual"] == pytest.approx(
        d["cpi_over_deflator"] * d["wage_share_ratio"], rel=tol)


def test_unknown_rule_is_rejected(inputs: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        wc.counterfactual(inputs, 2000, rule="geometric")


@pytest.mark.parametrize("floor", ["nominal", "components"])
def test_floor_never_lowers_the_counterfactual(inputs: pd.DataFrame, floor: str) -> None:
    for base in range(wc.START, wc.END):
        assert wc.ratio(inputs, base, floor=floor) >= wc.ratio(inputs, base) - 1e-12


def test_nominal_floor_path_never_falls(inputs: pd.DataFrame) -> None:
    path = wc.counterfactual(inputs, wc.START, floor="nominal")
    assert (path.diff().dropna() >= 0).all()


def test_nominal_floor_is_the_rule_times_the_ratchet(inputs: pd.DataFrame) -> None:
    step = (1 + inputs["infl"] + inputs["g_person"]).loc[1990 + 1:wc.END]
    ratchet = (1 / step[step < 1]).prod()
    assert wc.ratio(inputs, 1990, floor="nominal") == pytest.approx(
        wc.ratio(inputs, 1990) * ratchet, rel=1e-12)


def test_floor_changes_nothing_without_a_falling_year(inputs: pd.DataFrame) -> None:
    assert wc.ratio(inputs, 2021, floor="nominal") == pytest.approx(wc.ratio(inputs, 2021))


def test_unknown_floor_is_rejected(inputs: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        wc.counterfactual(inputs, 2000, floor="real")
