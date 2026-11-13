"""Checks the data and the accounting behind the test of capital deepening in Chinese manufacturing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analysis  # noqa: E402
import cip  # noqa: E402

RES = HERE / "results"


def setup_module(module) -> None:  # noqa: ARG001
    analysis.main()


def test_cip_investment_matches_national_accounts() -> None:
    d = cip.load()
    tot = cip.aggregate(d, cip.ALL)
    t = pd.read_csv(HERE / "data" / "oecd_t1.csv")
    gfcf = t[(t.REF_AREA == "CHN") & (t.PRICE_BASE == "V") & (t.TRANSACTION == "P51G")].set_index("TIME_PERIOD").OBS_VALUE
    for y in (2007, 2013, 2017):
        assert abs(tot.I[y] / gfcf[y] - 1) < 0.02


def test_manufacturing_aggregate() -> None:
    d = cip.load()
    m = cip.aggregate(d, cip.MANUF)
    assert np.allclose(m.K, d["K"].loc[cip.MANUF].sum())
    assert abs(m.Y[2000] / m.VA[2000] - 1) < 1e-12          # real value added is in 2000 yuan
    one = cip.real_va(d, [21])                               # a single industry reproduces its own growth rates
    assert np.allclose(np.log(one).diff().loc[1990:], d["gVA"].loc[21].loc[1990:], atol=1e-12)


def test_period_growth_identity() -> None:
    p = pd.read_csv(RES / "cip_manufacturing_periods.csv")
    assert np.allclose(p.y_per_worker - p.k_per_worker, p.y_per_k, atol=2e-4)  # 4-decimal CSV
    o = pd.read_csv(RES / "oecd_manufacturing.csv", index_col=0)
    for a, b in [(2007, 2013), (2013, 2017)]:
        s = o[[f"yl_{a}_{b}", f"kl_{a}_{b}", f"yk_{a}_{b}"]].dropna()
        assert np.allclose(s.iloc[:, 0] - s.iloc[:, 1], s.iloc[:, 2], atol=2e-4)


def test_published_allocation_reproduces_the_capital_stock() -> None:
    d = cip.load()
    m, tot = cip.aggregate(d, cip.MANUF), cip.aggregate(d, cip.ALL)
    shares = {y: float(m.I[y] / tot.I[y]) for y in range(2014, 2018)}
    for delta in cip.DELTAS:
        k = cip.capital_path(m, tot.I, shares, delta)
        assert np.allclose(k.values, m.K.loc[2013:2017].values, rtol=1e-10)


def test_break_even_is_monotone() -> None:
    g = pd.read_csv(RES / "cip_share_grid.csv")
    for _, s in g.groupby("delta"):
        assert (np.diff(s.sort_values("share").y_per_k_growth.values) < 0).all()


def test_nbs_chain() -> None:
    t = pd.read_csv(RES / "nbs_chained.csv", index_col=0)
    assert t.loc[2017].round(9).eq(100).all() or abs(t.loc[2017, "assets_per_employee"] - 100) < 1e-9
    assert np.allclose(t.assets_per_employee, 100 * t.revenue_per_capita / t.revenue_per_100_yuan_assets)


def test_sector_shares_add_up() -> None:
    s = pd.read_csv(RES / "cip_investment_shares.csv", index_col=0)
    assert np.allclose(s.sum(axis=1), 1, atol=1e-4)  # 5-decimal CSV
