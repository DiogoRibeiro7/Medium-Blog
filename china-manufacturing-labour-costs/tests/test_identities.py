"""Checks that the decompositions in results/ add up and agree with the published OECD series."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analysis  # noqa: E402

RES = HERE / "results"


def setup_module(module) -> None:  # noqa: ARG001
    analysis.main()


def test_industries_sum_to_manufacturing() -> None:
    total = sum(analysis.EMP[a] for a in analysis.INDUSTRIES)
    ratio = (total / analysis.EMP["C"]).loc[:, ["CHN", "MEX", "POL", "PRT", "DEU", "USA", "VNM"]]
    assert (ratio - 1).abs().max().max() < 1e-3


def test_value_added_per_person_matches_tiva() -> None:
    va_tiva = analysis.VA_C.loc[1995:2022, "CHN"] * 1e6
    va_tim = analysis.VAPP["C"].loc[1995:2022, "CHN"] * analysis.EMP["C"].loc[1995:2022, "CHN"] * 1e3
    assert ((va_tim / va_tiva) - 1).abs().max() < 2e-3


def test_labour_share_matches_published_series() -> None:
    d = pd.read_csv(RES / "ladder_2022.csv")
    implied = d.compensation / d.value_added
    assert np.allclose(implied, d.labour_share, atol=1e-3)


def test_yuan_decomposition_adds_up() -> None:
    d = pd.read_csv(RES / "china_yuan_decomposition.csv").set_index("TIME_PERIOD")
    assert np.allclose(d.comp_usd * d.yuan_per_usd, d.comp_cny, rtol=1e-4)
    total = np.log(d.comp_usd.loc[2022] / d.comp_usd.loc[1995])
    yuan = np.log(d.comp_cny.loc[2022] / d.comp_cny.loc[1995])
    currency = -np.log(d.yuan_per_usd.loc[2022] / d.yuan_per_usd.loc[1995])
    assert abs(total - (yuan + currency)) < 1e-4  # the CSV is written to four decimals


def test_shift_share_terms_add_up() -> None:
    d = pd.read_csv(RES / "shift_share.csv")
    assert np.allclose(d.within + d.between + d.interaction, d.total, atol=0.01)
    chn = d[(d.country == "CHN") & (d.period == "1995-2022")].iloc[0]
    published = analysis.VAPP["C"].loc[2022, "CHN"] - analysis.VAPP["C"].loc[1995, "CHN"]
    assert abs(chn.total - published) / published < 5e-3


def test_world_shares_are_shares() -> None:
    d = pd.read_csv(RES / "world_shares.csv")
    cols = ["china", "low_wage_asia", "eu27", "usa"]
    assert ((d[cols] >= 0) & (d[cols] <= 1)).all().all()
    assert (d[cols].sum(axis=1) <= 1.0001).all()


def test_relative_compensation_is_a_ratio_of_the_ladder() -> None:
    rel = pd.read_csv(RES / "china_relative_compensation.csv").set_index("year")
    lad = pd.read_csv(RES / "ladder_2022.csv").set_index("country")
    for c in ["MEX", "VNM", "POL", "PRT", "DEU"]:
        assert abs(rel.loc[2022, c] - lad.loc["CHN", "compensation"] / lad.loc[c, "compensation"]) < 1e-3
