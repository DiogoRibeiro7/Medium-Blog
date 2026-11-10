"""Consistency checks on the pinned data and the helpers in ``analysis.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analysis  # noqa: E402


def test_recovered_share_basic():
    s = pd.Series({2008: 100.0, 2010: 60.0, 2012: 40.0, 2024: 70.0})
    d = analysis.recovered_share(s, 2008)
    assert d["trough_year"] == 2012
    assert d["R"] == pytest.approx(0.5)
    assert d["last_over_ref"] == pytest.approx(0.7)


def test_recovered_share_undefined_without_a_fall():
    s = pd.Series({2008: 100.0, 2010: 98.0, 2024: 300.0})
    assert np.isnan(analysis.recovered_share(s, 2008)["R"])


@pytest.mark.parametrize("kind", ["firms", "persons", "production"])
def test_scie_subclasses_add_up_to_division_41(kind):
    df = analysis.scie(kind)
    total = df["41100"] + df["41200"]
    assert np.allclose(total, df["41"], rtol=0, atol=1.0)


def test_scie_indicators_agree_where_they_overlap():
    ov = analysis.scie_overlap("firms")
    assert len(ov) > 0
    assert (ov.loc["new"] - ov.loc["old"]).abs().max().max() == 0


def test_licensed_dwellings_by_promoter_add_up():
    lic = analysis.ine("ine_licensed_by_promoter").pivot_table(index="period", columns="dim_3", values="value")
    parts = lic[["1", "2", "3", "4"]].sum(axis=1)
    assert np.allclose(parts, lic["T"], atol=1.0)


def test_lfs_splice_year_is_close_in_both_classifications():
    c1 = analysis.eurostat("lfs_age_nace1").pivot_table(index="year", columns="age", values="value")
    n2 = analysis.eurostat("lfs_age_nace2")
    c2 = n2[n2.nace_r2 == "F"].pivot_table(index="year", columns="age", values="value")
    assert abs(c1.loc[2008, "Y_GE15"] / c2.loc[2008, "Y_GE15"] - 1) < 0.01


def test_summary_is_reproduced(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "RESULTS", tmp_path)
    analysis.main()
    produced = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    pinned = (HERE / "results" / "summary.txt").read_text(encoding="utf-8")
    assert produced == pinned
