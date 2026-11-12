"""Checks the accounting identities behind the industry decomposition."""

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


def test_sections_add_up_to_the_economy() -> None:
    for y in analysis.YEARS:
        nw, us = analysis.group(y, analysis.SECTIONS + ["_T"])
        for fr in (nw, us):
            for k in ("D1", "GOS"):
                assert abs(fr.loc[analysis.SECTIONS, k].sum() / fr.loc["_T", k] - 1) < 0.002


def test_value_added_identity() -> None:
    for y in analysis.YEARS:
        nw, us = analysis.group(y, analysis.SECTIONS)
        for fr in (nw, us):
            ok = fr.dropna()
            assert np.allclose(ok.FCVA, ok.LC + ok.CFC + ok.NR, rtol=1e-3, atol=1.0)


def test_contributions_add_up() -> None:
    for y in analysis.YEARS:
        t = pd.read_csv(RES / f"contributions_{y}.csv", index_col=0)
        assert np.allclose(t.structure + t.margin, t.contribution, atol=1e-3)
        assert abs(t.share_of_gap.sum() - 1) < 1e-3


def test_subindustries_add_up_for_services() -> None:
    s = pd.read_csv(RES / "subindustries_2021.csv")
    for parent in ("J", "K", "M"):
        kids = s[(s.parent == parent) & (s.industry != parent)].gap.sum()
        whole = s[(s.parent == parent) & (s.industry == parent)].gap.iloc[0]
        assert abs(kids - whole) < 0.02


def test_country_contributions_add_up() -> None:
    c = pd.read_csv(RES / "countries_2021.csv", index_col=0)
    assert np.allclose(c[["L", "K", "J_M_N", "C", "other"]].sum(axis=1), c.gap, atol=1e-4)


def test_four_accountings_reconcile() -> None:
    for y in analysis.YEARS:
        v = pd.read_csv(RES / f"variants_{y}.csv", index_col=0)
        hn = sum(analysis.total_hours(c, y) for c in analysis.NW)
        hu = analysis.total_hours("USA", y)
        nw, us = analysis.group(y, analysis.SECTIONS)
        imp_gap = nw["IMP"] / hn - us["IMP"] / hu
        cfc_gap = nw["CFC"].fillna(0) / hn - us["CFC"].fillna(0) / hu
        assert np.allclose(v.net_unimputed - v.net_imputed, imp_gap.reindex(v.index), atol=1e-3)
        assert np.allclose(v.gross_imputed - v.net_imputed, cfc_gap.reindex(v.index), atol=0.02)
        assert np.allclose(v.gross_unimputed - v.gross_imputed, imp_gap.reindex(v.index), atol=1e-3)
        t = pd.read_csv(RES / f"contributions_{y}.csv", index_col=0)
        assert np.allclose(v.net_imputed, t.contribution, atol=1e-3)
