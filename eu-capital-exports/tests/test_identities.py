"""Checks the accounting identities behind the tables in results/."""

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


def test_balance_of_payments_identity() -> None:
    for y in analysis.YEARS:
        lhs = analysis.bop("CA", "BAL", y) + analysis.bop("KA", "BAL", y) + analysis.bop("EO", "NET", y)
        assert abs(lhs - analysis.bop("FA", "NET", y)) < 2


def test_portfolio_net_is_assets_minus_liabilities() -> None:
    for y in analysis.YEARS:
        net = analysis.bop("FA__P__F", "NET", y)
        assert abs(net - (analysis.bop("FA__P__F", "ASS", y) - analysis.bop("FA__P__F", "LIAB", y))) < 2


def test_sector_net_lending_adds_up() -> None:
    d = pd.read_csv(RES / "sector_net_lending.csv", index_col=0).loc[2002:]
    parts = d[["S14_S15", "S11", "S12", "S13"]].sum(axis=1)
    assert (parts - d["S1"]).abs().max() < 0.05


def test_sector_accounts_and_bop_agree_on_net_lending() -> None:
    sa = analysis.sa("EU27_2020", "S1", "B9")
    for y in analysis.YEARS:
        bop = analysis.bop("CKA", "BAL", y)
        assert abs(sa[y] / bop - 1) < 0.10, y


def test_stock_flow_reconciliation() -> None:
    d = pd.read_csv(RES / "stock_flow_2015_2025.csv")
    assert np.allclose(d.change_in_position, d.net_flows + d.other_changes, atol=0.2)


def test_household_shares() -> None:
    d = pd.read_csv(RES / "household_assets.csv")
    cols = ["deposits", "debt securities", "listed and other equity", "investment fund shares", "insurance and pensions"]
    s = d[cols].sum(axis=1)
    assert (s <= 1.0001).all() and (s >= 0.85).all()
