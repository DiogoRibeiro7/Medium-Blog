"""Hourly calls on a complete local-clock grid, weeks, and the structural checks (PLAN.md, "Data" and "Windows")."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2016-01-01 00:00")
END = pd.Timestamp("2019-12-31 23:00")
FIRST_TEST_MONDAY = pd.Timestamp("2018-12-31")
TEST_WEEKS = 52
WEEK = 168
LOW_DAY = 0.2


@dataclass
class Calls:
    hours: pd.DatetimeIndex  # every local clock hour, START..END
    calls: np.ndarray  # (len(hours),) float64

    def index(self, t: pd.Timestamp) -> int:
        return int((t - START) / pd.Timedelta(hours=1))


def grid() -> pd.DatetimeIndex:
    return pd.date_range(START, END, freq="h")


def load(path: Path) -> Calls:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    s = pd.Series({pd.Timestamp(r["hour"] + ":00"): float(r["calls"]) for r in rows})
    g = grid()
    s = s.groupby(level=0).sum().reindex(g, fill_value=0.0)
    return Calls(g, s.to_numpy(dtype=np.float64))


def evaluation_weeks() -> list[pd.Timestamp]:
    return [FIRST_TEST_MONDAY + pd.Timedelta(days=7 * i) for i in range(TEST_WEEKS)]


def daily(c: Calls) -> pd.Series:
    return pd.Series(c.calls, index=c.hours).resample("D").sum()


def low_days(c: Calls) -> pd.Series:
    """Days with fewer than LOW_DAY of their year's median daily calls."""
    d = daily(c)
    med = d.groupby(d.index.year).transform("median")
    return d[d < LOW_DAY * med]


def excluded_weeks(c: Calls) -> list[pd.Timestamp]:
    low = set(low_days(c).index.normalize())
    out = []
    for w in evaluation_weeks():
        if any(w + pd.Timedelta(days=i) in low for i in range(7)):
            out.append(w)
    return out


def structure(c: Calls) -> list[str]:
    s = pd.Series(c.calls, index=c.hours)
    lines = [f"hours on the grid {len(c.hours)} ({c.hours[0]} to {c.hours[-1]}); total calls {int(s.sum())}"]
    for y, v in s.groupby(s.index.year).sum().items():
        lines.append(f"  {y}: {int(v)} calls, {int((s[s.index.year == y] == 0).sum())} hours with no calls")
    by_hour = s.groupby(s.index.hour).sum() / s.sum()
    lines.append("share of calls by hour of day: " + ", ".join(f"{h}:{100 * v:.1f}%" for h, v in by_hour.items()))
    low = low_days(c)
    lines.append(
        f"days below {LOW_DAY:.0%} of their year's median: {len(low)}"
        + (": " + ", ".join(f"{d.date()} ({int(v)})" for d, v in low.items()) if len(low) else "")
    )
    ex = excluded_weeks(c)
    lines.append(f"test weeks excluded: {len(ex)}" + (": " + ", ".join(str(w.date()) for w in ex) if ex else ""))
    return lines
