"""Unit tests for the counterfactual debt path and the shared helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import peer_mean, share  # noqa: E402
from space import counterfactual  # noqa: E402

YEARS = range(2010, 2021)


def flat(v: float) -> pd.Series:
    return pd.Series(v, index=list(YEARS), dtype=float)


def test_no_extra_investment_no_extra_debt():
    gap = counterfactual(flat(0.0), flat(60.0), flat(2.0), flat(3.0), 2011, 2019)
    assert (gap == 0).all()


def test_extra_debt_accumulates_when_rate_equals_growth():
    gap = counterfactual(flat(0.5), flat(60.0), flat(3.0), flat(3.0), 2011, 2015)
    assert abs(gap[2015] - 2.5) < 1e-12
    assert abs(gap[2020] - 2.5) < 1e-12  # nothing added after the end year, no compounding when i = g


def test_extra_debt_shrinks_when_growth_exceeds_rate():
    gap = counterfactual(flat(1.0), flat(60.0), flat(1.0), flat(4.0), 2011, 2011)
    assert gap[2011] == 1.0
    assert gap[2020] < gap[2011] and abs(gap[2012] - 1.0 * 1.01 / 1.04) < 1e-12


def test_negative_gaps_are_not_subtracted():
    gap = counterfactual(flat(-1.0), flat(60.0), flat(2.0), flat(2.0), 2011, 2019)
    assert (gap == 0).all()


def test_share_and_peer_mean():
    x = pd.DataFrame({"DE": [1.0, 2.0], "NL": [3.0, 4.0]}, index=[2020, 2021])
    y = pd.DataFrame({"DE": [100.0, 100.0], "NL": [50.0, 50.0]}, index=[2020, 2021])
    s = share(x, y)
    assert s.loc[2020, "DE"] == 1.0 and s.loc[2021, "NL"] == 8.0
    pm = peer_mean(pd.DataFrame({"A": [1.0], "B": [3.0]}), ("A", "B"))
    assert pm.iloc[0] == 2.0
