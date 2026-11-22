from datetime import date, timedelta

import numpy as np
import pandas as pd

from swaptails.daytype import code, truth_labels


def _series():
    start = date(2025, 3, 3)  # a Monday
    days = [start + timedelta(days=i) for i in range(70)]
    level = {5: 80.0, 6: 70.0}
    values = [level.get(d.weekday(), 100.0) for d in days]
    daily = pd.Series(values, index=days)
    ordinary = pd.Series(True, index=days)
    return days, daily, ordinary


def test_labels_follow_the_nearest_level():
    days, daily, ordinary = _series()
    holiday = date(2025, 4, 2)  # a Wednesday at Sunday level
    bridge = date(2025, 4, 4)  # a Friday at Saturday level
    regional = date(2025, 4, 9)  # a Wednesday at 95, closest to a workday
    daily[holiday], daily[bridge], daily[regional] = 71.0, 81.0, 95.0
    for d in (holiday, bridge, regional):
        ordinary[d] = False
    out = truth_labels(daily, ordinary, window=28)
    assert out.loc[holiday, "truth"] == "sunday_like"
    assert out.loc[bridge, "truth"] == "saturday_like"
    assert out.loc[regional, "truth"] == "workday"
    assert np.isclose(out.loc[regional, "relative_level"], 25 / 30)
    assert out.loc[date(2025, 4, 5), "truth"] == "saturday_like"
    assert out.loc[date(2025, 4, 8), "truth"] == "workday"


def test_special_days_are_not_references():
    days, daily, ordinary = _series()
    # Make every Wednesday in the window a holiday at Sunday level: a Wednesday has no
    # ordinary same-weekday reference, so it gets no label rather than a wrong one.
    for d in days:
        if d.weekday() == 2:
            daily[d], ordinary[d] = 70.0, False
    out = truth_labels(daily, ordinary, window=28)
    assert out.loc[date(2025, 4, 2), "truth"] is None


def test_code_falls_back():
    assert code("sunday_like", "workday") == 2
    assert code(None, "saturday_like") == 1
