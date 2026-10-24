"""Checks on the further analyses in ``deeper.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deeper as dp  # noqa: E402
import spending as sp  # noqa: E402


def test_working_leser_fit_recovers_an_exact_engel_curve() -> None:
    # Shares generated exactly as w = a + b log c: the fit returns b, and the elasticity of
    # spending w*c at the mean share equals the numerical derivative of log(w c) there.
    c = np.array([3000.0, 4500.0, 6000.0, 8000.0, 14000.0])
    a, b = 1.2, -0.11
    rows = []
    for year in dp.CASH_WAVES:
        for q, cq in zip(sp.QUINTILES, c):
            rec = {"quintile": q, "year": year, "cash_consumption_pps_ae": cq}
            rec.update({item: a + b * np.log(cq) for item in dp.ITEMS})
            rows.append(rec)
    cs = pd.DataFrame(rows).set_index(["quintile", "year"])
    wl = dp.working_leser(cs).set_index(["year", "item"])
    fit = wl.loc[(2015, "food")]
    assert fit["slope"] == pytest.approx(b)
    assert fit["r2"] == pytest.approx(1.0)
    c_star = np.exp((fit["mean_share"] - a) / b)
    h = 1e-6
    numeric = (np.log((a + b * np.log(c_star * np.exp(h))) * c_star * np.exp(h))
               - np.log((a + b * np.log(c_star)) * c_star)) / h
    assert fit["elasticity"] == pytest.approx(numeric, rel=1e-4)


def test_cash_shares_of_core_items_match_the_quintile_core_share() -> None:
    cs = dp.cash_shares()
    hb = sp.hbs_shares()
    for q in sp.QUINTILES:
        core = cs.loc[(q, 2015), ["food", "housing_cash"]].sum()
        assert core == pytest.approx(hb.loc[(q, 2015), "core_ex_imputed"], abs=1e-9)


def test_chained_index_equals_fixed_index_inside_the_first_segment() -> None:
    chained = dp.chained_index("QU1")
    fixed = dp.fixed_index("QU1", 1999, end=2005)
    assert chained.loc[1996] == pytest.approx(100.0)
    assert np.allclose(chained.loc[1996:2005].to_numpy(), fixed.loc[1996:2005].to_numpy())


def test_chained_index_is_continuous_at_the_links() -> None:
    chained = dp.chained_index("QU5")
    for start, _end, wave in dp.SEGMENTS[1:]:
        w = dp.wave_weights(wave, "QU5")
        p = sp.hicp()[sp.DIVISIONS]
        nxt = chained.loc[start] * float((w * p.loc[start + 1] / p.loc[start]).sum())
        assert chained.loc[start + 1] == pytest.approx(nxt)


def test_wave_weights_sum_to_one_without_imputed_rent() -> None:
    for wave in (1999, 2005, 2010, 2015):
        for q in sp.QUINTILES + ["TOTAL"]:
            w = dp.wave_weights(wave, q)
            assert w.sum() == pytest.approx(1.0)
            assert (w >= 0).all()


def test_price_volume_split_is_exact() -> None:
    pv = pd.read_csv(dp.RESULTS / "core_share_price_volume_hicp.csv", index_col="year")
    rel = pv["share"] / pv["share"].iloc[0]
    assert np.allclose(rel, pv["rel_price_hicp"] * pv["rel_volume_residual"])


def test_hicp_weight_contributions_add_up_to_the_basket_difference() -> None:
    hw = pd.read_csv(dp.RESULTS / "hicp_weights_vs_survey.csv", index_col="coicop")
    r = 1 + hw["price_change_2021_2023"]
    diff = (hw["hicp_weight_2023"] * r).sum() - (hw["survey_cash_all_2015"] * r).sum()
    assert hw["contribution_to_gap"].sum() == pytest.approx(diff, abs=1e-5)


def test_resident_bound_is_above_the_domestic_share() -> None:
    na = sp.na_shares()
    assert (na["domestic_over_national"] > 1).all()
    assert (na["core_ex_imputed_resident"] > na["core_ex_imputed"]).all()


def test_minimum_wage_daily_average() -> None:
    mw = dp.minimum_wage()
    assert mw.loc[2015] == pytest.approx(505.0)
    assert mw.loc[2025] == pytest.approx(870.0)
    # 1989: 30,000 PTE from January, 31,500 PTE from July.
    expected = (181 * 30000 + 184 * 31500) / 365 / 200.482
    assert mw.loc[1989] == pytest.approx(expected)


def test_euro_conversion_reproduces_the_ine_total() -> None:
    ine = pd.read_csv(dp.DATA / "ine_idef_totals.csv").set_index("wave")["mean_expenditure_eur_per_household"]
    e = pd.read_csv(dp.DATA / "hbs_exp_t133.csv")
    pps = e[(e["unit"] == "PPS_HH") & (e["quant_inc"] == "TOTAL")].set_index("year")["value"]
    eu = pd.read_csv(dp.RESULTS / "euro_levels.csv", index_col=["wave", "quintile"])
    for wave, total in ine.items():
        assert eu.loc[(wave, "QU1"), "factor"] * pps.loc[wave] == pytest.approx(total, rel=1e-3)
