"""Checks on the industry snapshot, the decomposition and the imputation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch  # noqa: E402
import industries as ind  # noqa: E402


@pytest.fixture(scope="module")
def portugal() -> pd.DataFrame:
    return ind.load_industries("PT")


@pytest.mark.parametrize("geo", ind.GEOS)
def test_industries_add_up_to_the_total(geo: str) -> None:
    checks = ind.data_checks(geo)
    assert checks["max_gap_B1G_pct"] < 0.01
    assert checks["max_gap_D1_pct"] < 0.01
    assert checks["non_positive_value_added_cells"] == 0
    assert checks["industries_after_merges"] == 61


def test_factor_cost_matches_the_national_accounts(portugal: pd.DataFrame) -> None:
    gdp = pd.read_csv(ind.FIRST_ARTICLE_DATA / "nama_10_gdp.csv")
    gdp = gdp[(gdp["geo"] == "PT") & (gdp["year"] <= ind.LAST_YEAR)].set_index("year")
    national = 100 * gdp["D1"] / (gdp["D1"] + gdp["B2A3G"])
    t = portugal.groupby("year")[["D1", "Y"]].sum()
    by_industry = 100 * t["D1"] / t["Y"]
    assert (by_industry - national).abs().max() < 0.05


def test_economy_wide_imputation_matches_the_first_article(portugal: pd.DataFrame) -> None:
    emp = pd.read_csv(ind.FIRST_ARTICLE_DATA / "nama_10_a10_e_PT.csv").set_index("year").loc[: ind.LAST_YEAR]
    gdp = pd.read_csv(ind.FIRST_ARTICLE_DATA / "nama_10_gdp.csv")
    gdp = gdp[(gdp["geo"] == "PT") & (gdp["year"] <= ind.LAST_YEAR)].set_index("year")
    first = gdp["D1"] / emp["SAL_DC_THS_PER"] * (emp["EMP_DC_THS_PER"] - emp["SAL_DC_THS_PER"])
    here = ind.imputed_labour(portugal, "total", cap=False)
    assert ((here - first) / first).abs().max() < 0.001


def test_capped_imputation_stays_between_compensation_and_value_added(portugal: pd.DataFrame) -> None:
    labour = ind.labour_by_industry(portugal, "adjusted")
    y = portugal.set_index(["code", "year"])["Y"]
    d1 = portugal.set_index(["code", "year"])["D1"]
    assert (labour <= np.maximum(y, d1) + 1e-9).all()
    assert (labour >= d1 - 1e-9).all()


@pytest.mark.parametrize("level", ind.LEVELS)
@pytest.mark.parametrize("scope", ind.SCOPES)
def test_shift_share_adds_up_and_chains(portugal: pd.DataFrame, level: str, scope: str) -> None:
    frame = ind.grouped(portugal, "adjusted", level, scope)
    one = ind.shift_share(frame, 2016, 2017)
    within, between = ind.chained(frame, 2016, 2017)
    assert within == pytest.approx(one["within"].sum(), abs=1e-12)
    assert between == pytest.approx(one["between"].sum(), abs=1e-12)
    long = ind.shift_share(frame, 1995, 2024)
    cw, cb = ind.chained(frame, 1995, 2024)
    assert cw + cb == pytest.approx(long["within"].sum() + long["between"].sum(), abs=1e-9)


def test_grouping_is_a_partition() -> None:
    codes = [ind.MERGES.get(c, c) for c in ind.A64 if c not in ind.DROPPED]
    assert {ind.group_of(c, "a10") for c in codes} == {"A", "B-E", "F", "G-I", "J", "K", "L", "M_N", "O-Q", "R-U"}
    assert len({ind.group_of(c, "a21") for c in codes}) == 20


def test_weighted_correlation_and_permutation() -> None:
    x = np.arange(10.0)
    assert ind.weighted_corr(x, 2 * x + 1, np.ones(10)) == pytest.approx(1.0)
    assert ind.weighted_corr(x, -x, np.arange(1.0, 11.0)) == pytest.approx(-1.0)
    w = np.ones(10)
    first = ind.permutation_p(x, x[::-1].copy(), w, "check")
    assert first == ind.permutation_p(x, x[::-1].copy(), w, "check")
    assert first < 0.01


def test_parse_tsv_keeps_flags() -> None:
    text = (
        "freq,unit,nace_r2,na_item,geo\\TIME_PERIOD\t1995 \t1996 \t1997 \t1998 \n"
        "A,CP_MEUR,A01,B1G,PT\t3000.1 \t3100.0 p\t: c\t: \n"
        "A,CP_MEUR,C19,D1,PT\t100 \t-2.5 e\t7 \t8 bp\n"
    )
    frame = fetch.parse_tsv(text)
    b1g = frame[frame["na_item"] == "B1G"].set_index("year")
    assert b1g.loc[1996, "value"] == 3100.0 and b1g.loc[1996, "flag"] == "p"
    assert np.isnan(b1g.loc[1997, "value"]) and b1g.loc[1997, "flag"] == "c"
    d1 = frame[frame["na_item"] == "D1"].set_index("year")
    assert d1.loc[1996, "value"] == -2.5 and d1.loc[1998, "flag"] == "bp"
