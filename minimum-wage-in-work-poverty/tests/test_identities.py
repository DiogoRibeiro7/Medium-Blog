"""Checks that the derived tables in results/ are consistent with the downloaded series."""

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


def test_equivalence_scale_of_the_thresholds() -> None:
    t = analysis.THR
    assert np.allclose((t["A2_CH2_LT14"] / t["A1"]).dropna(), 2.1, atol=0.002)
    assert np.allclose((t["A2"] / t["A1"]).dropna(), 1.5, atol=0.002)


def test_taxben_gross_minimum_matches_eurostat() -> None:
    g, m = analysis.GFE, analysis.MW_YEAR
    ratios = [g.loc[y, c] / (12 * m.loc[y, c]) for c in ("PT", "ES", "FR") for y in range(2015, 2025)]
    assert abs(np.median(ratios) - 1) < 0.02


def test_poverty_ratios_are_net_over_line() -> None:
    d = pd.read_csv(RES / "minimum_wage_vs_poverty_line.csv")
    for h in analysis.HH_LABEL:
        ok = d[[h, f"{h}_net", f"{h}_line"]].dropna()
        assert np.allclose(ok[h], ok[f"{h}_net"] / ok[f"{h}_line"], rtol=1e-3)


def test_real_growth_recomputes_for_portugal() -> None:
    d = pd.read_csv(RES / "real_growth_ses_2006_2022.csv").set_index("geo")
    m, h, s = analysis.MW_JUL, analysis.HICP, analysis.SES
    prices = np.log(h.loc[2022, "PT"] / h.loc[2006, "PT"])
    assert abs(d.loc["PT", "floor"] - (np.log(m.loc[2022, "PT"] / m.loc[2006, "PT"]) - prices)) < 1e-4
    assert abs(d.loc["PT", "median"] - (np.log(s.loc[("PT", 2022), "MED"] / s.loc[("PT", 2006), "MED"]) - prices)) < 1e-4


def test_real_indices_start_at_100() -> None:
    d = pd.read_csv(RES / "portugal_real_indices.csv").set_index("year")
    assert np.allclose(d.loc[2006, ["minimum_real", "fte_salary_real", "poverty_line_real"]], 100)


def test_group_rates_match_source_tables() -> None:
    d = pd.read_csv(RES / "in_work_poverty_groups_2025.csv").set_index("group")
    assert d.loc["self-employed", "PT"] == analysis.IW_STATUS.loc[("PT", "NSAL"), 2025]
    assert d.loc["single adult with children", "EU27_2020"] == analysis.IW_HH.loc[("EU27_2020", "A1_DCH"), 2025]
    assert ((d[["PT", "EU27_2020"]] >= 0) & (d[["PT", "EU27_2020"]] <= 100)).all().all()
