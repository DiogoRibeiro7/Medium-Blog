"""Checks of the event definition, the out-of-time split and the lead-time count."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import panel  # noqa: E402
import warning  # noqa: E402


def frame(flags: list[int], start: int = 2002, product: str = "000001") -> pd.DataFrame:
    years = list(range(start, start + len(flags)))
    return pd.DataFrame({"product": product, "year": years, "flagged": [bool(f) for f in flags], "top_partner": "CN", "top_share": 0.5, "top_value": 1.0, "imports_extra": 2.0, "extra_share": 0.6, "log_ratio": 0.1})


def test_durable_entry_needs_three_clean_years_and_a_second_flagged_year():
    d = frame([0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
    e = panel.events(d)
    assert e["entry"].tolist() == [2005, 2012]  # 2008 is a one-year flicker after an exit two years earlier


def test_entry_at_the_start_of_a_code_is_not_counted():
    d = frame([1, 1, 1, 0, 0, 0, 1, 1])
    assert panel.events(d)["entry"].tolist() == [2008]
    d = frame([0, 0, 1, 1])  # only two years observed before the flag
    assert panel.events(d).empty


def test_target_is_known_only_when_the_horizon_is_observed():
    flags = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    d = frame(flags).assign(c_top=True, c_extra=True, c_ratio=False, suppliers=5, imports_intra=1.0, exports_extra=1.0, exports_intra=1.0, cn_share=0.5, rel_unit_value=0.0, hhi=0.3, second_share=0.2)
    chall = pd.DataFrame(columns=["product", "year", "challenger", "challenger_gain", "challenger_share", "challenger_is_top"]).astype({"product": str, "year": int})
    f = warning.features(d, chall).set_index("year")
    # origin t: a first pair of flagged years (E, E+1) with t < E <= t+3, known when t+4 is observed
    assert f.loc[2005, "y3"] == 1.0 and f.loc[2005, "entry3"] == 2008
    assert f.loc[2004, "y3"] == 0.0  # 2008 is four years after 2004
    assert f.loc[2006, "y3"] == 1.0 and np.isnan(f.loc[2007, "y3"]) and np.isnan(f.loc[2010, "y3"])
    # eligible only with three unflagged years ending at t and a three-year history
    assert f["eligible"].tolist() == [False, False, False, True, True, True, False, False, False]


def test_training_origins_never_see_the_test_period():
    h = warning.H
    for T in range(warning.FIRST_TEST, 2026):
        last_train = T - h - 1
        # the outcome of the last training origin is decided by year last_train + h + 1 = T
        assert last_train + h + 1 <= T


def test_near_miss_puts_two_criteria_first():
    x = pd.DataFrame({"top_share": [0.39, 0.10, 0.6], "extra_share": [0.9, 0.9, 0.2], "log_ratio_c": [1.0, 1.0, -3.0], "n_criteria": [2, 2, 1]})
    s = warning.near_miss(x)
    assert s[0] > s[1] > s[2]
