"""Checks that the decompositions in results/ add up and match the published series."""

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


def test_gdp_per_hour_matches_published_series() -> None:
    y = analysis.YEAR
    ratio = (analysis.GDP_CUR.loc[y] / analysis.HOURS.loc[y] / analysis.GDPHRS_CUR.loc[y]).dropna()
    assert (ratio - 1).abs().max() < 1e-6


def test_per_capita_decomposition_adds_up() -> None:
    for name in ("country_levels_2024.csv", "group_levels_2024.csv"):
        d = pd.read_csv(RES / name)
        err = (d.lp_productivity + d.lp_hours + d.lp_employment - d.lp_total).abs().max()
        assert err < 1e-3, name  # CSVs are written to four decimals


def test_income_split_adds_up() -> None:
    d = pd.read_csv(RES / "income_split_2024.csv")
    assert np.allclose(d.lc_hour + d.tax_hour + d.rest_hour, d.gdp_hour, atol=1e-3)
    assert np.allclose(d.cfc_hour + d.net_rest_hour, d.rest_hour, atol=1e-3)
    assert np.allclose(d.gap_lc_hour + d.gap_tax_hour + d.gap_rest_hour, d.gap_gdp_hour, atol=1e-3)


def test_group_contributions_sum_to_one() -> None:
    d = pd.read_csv(RES / "gap_composition.csv")
    sums = d.groupby("year").contribution.sum()
    assert np.allclose(sums, 1.0, atol=1e-3)
    assert np.allclose(d.groupby("year").hours_share.sum(), 1.0, atol=1e-3)


def test_price_bases_agree_in_reference_year() -> None:
    p = pd.read_csv(RES / "time_paths.csv").set_index("year")
    for g in ("EU25", "North-West", "South", "Centre-East"):
        assert abs(p.loc[2020, f"{g}|const"] - p.loc[2020, f"{g}|cur"]) < 1e-6, g


def test_growth_accounting_is_additive() -> None:
    d = pd.read_csv(RES / "growth_accounting.csv").dropna(subset=["lp_growth", "mfp", "capital_deepening"])
    d = d[d.country.isin(["USA", "DEU", "FRA", "NLD", "ITA", "ESP"])]
    assert d.residual.abs().max() < 0.1  # log-versus-percent differences; Austria and Estonia are documented exceptions
    c = pd.read_csv(RES / "growth_gap_cumulative.csv")
    assert np.allclose(c.capital_deepening + c.mfp, c.lp, atol=1e-3)
