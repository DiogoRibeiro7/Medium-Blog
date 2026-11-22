"""The ex-post day type: what a day's demand actually looked like.

For each local day, the day's mean load is compared with three reference levels taken from
ordinary days (no holiday of any kind, not next to a national holiday, outside the year-end
window) within `truth_window_days` on either side:

- W: the median of ordinary days with the same weekday (for a Saturday or Sunday, the
  median of ordinary Tuesdays to Thursdays);
- S: the median of ordinary Saturdays;
- U: the median of ordinary Sundays.

The label is the nearest level (workday, saturday_like, sunday_like). The relative level
r = (x - U) / (W - U) is kept too: 1 is a normal working day, 0 a Sunday.

This uses load on both sides of the day, so it is a scoring device and the 'oracle'
covariate, never an input to a forecast that claims to be real.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import DAY_TYPES

MIN_REFERENCES = 2


def truth_labels(daily: pd.Series, ordinary: pd.Series, window: int) -> pd.DataFrame:
    """daily: mean load per local date; ordinary: whether each date may serve as a reference.

    Returns, per date: label (or None), relative level r, and the three reference levels.
    """
    dates = list(daily.index)
    values = daily.to_numpy(dtype=float)
    position = {d: i for i, d in enumerate(dates)}
    is_ordinary = ordinary.reindex(daily.index).fillna(False).to_numpy(dtype=bool)
    weekday = np.array([d.weekday() for d in dates])
    rows = []
    for i, d in enumerate(dates):
        x = values[i]
        lo, hi = d - timedelta(days=window), d + timedelta(days=window)
        idx = [position[o] for o in _span(lo, hi) if o in position and o != d]
        idx = np.array([j for j in idx if is_ordinary[j] and not np.isnan(values[j])], dtype=int)
        if np.isnan(x) or idx.size == 0:
            rows.append((d, None, np.nan, np.nan, np.nan, np.nan))
            continue
        wd = weekday[idx]
        same = idx[wd == weekday[i]] if weekday[i] < 5 else idx[(wd >= 1) & (wd <= 3)]
        sat, sun = idx[wd == 5], idx[wd == 6]
        if min(same.size, sat.size, sun.size) < MIN_REFERENCES:
            rows.append((d, None, np.nan, np.nan, np.nan, np.nan))
            continue
        w, s, u = (float(np.median(values[k])) for k in (same, sat, sun))
        levels = np.array([w, s, u])
        label = DAY_TYPES[int(np.argmin(np.abs(levels - x)))]
        r = (x - u) / (w - u) if w != u else np.nan
        rows.append((d, label, r, w, s, u))
    out = pd.DataFrame(rows, columns=["date", "truth", "relative_level", "ref_workday", "ref_saturday", "ref_sunday"])
    return out.set_index("date")


def _span(lo: date, hi: date):
    d = lo
    while d <= hi:
        yield d
        d += timedelta(days=1)


def code(label: str | None, fallback: str) -> int:
    """Integer code of a day type, with a fallback for missing labels."""
    return DAY_TYPES.index(label if label in DAY_TYPES else fallback)
