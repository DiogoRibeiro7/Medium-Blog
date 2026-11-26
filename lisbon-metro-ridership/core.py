"""Small, tested tools: the Portuguese holiday calendar, day-type counts, day weights and changes.

Nothing here reads files; panel.py and run_analysis.py pass the data in.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# Holidays abolished for 2013-2015 and restored from 2016 (Lei 23/2012, Lei 8/2016).
SUSPENDED_2013_2015 = {"corpus_christi", "oct5", "nov1", "dec1"}

# Municipal holidays of the municipalities each system mostly serves.
MUNICIPAL = {"Lisboa": (6, 13), "Porto": (6, 24), "Almada": (6, 24)}


def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm, Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def holidays(year: int, municipality: str | None = None) -> set[dt.date]:
    """National public holidays of mainland Portugal in a year, plus one municipal holiday if asked."""
    e = easter(year)
    named = {
        "jan1": dt.date(year, 1, 1),
        "good_friday": e - dt.timedelta(days=2),
        "easter": e,
        "apr25": dt.date(year, 4, 25),
        "may1": dt.date(year, 5, 1),
        "corpus_christi": e + dt.timedelta(days=60),
        "jun10": dt.date(year, 6, 10),
        "aug15": dt.date(year, 8, 15),
        "oct5": dt.date(year, 10, 5),
        "nov1": dt.date(year, 11, 1),
        "dec1": dt.date(year, 12, 1),
        "dec8": dt.date(year, 12, 8),
        "dec25": dt.date(year, 12, 25),
    }
    if 2013 <= year <= 2015:
        named = {k: v for k, v in named.items() if k not in SUSPENDED_2013_2015}
    out = set(named.values())
    if municipality is not None:
        mo, da = MUNICIPAL[municipality]
        out.add(dt.date(year, mo, da))
    return out


def day_counts(first: str, last: str, municipality: str | None = None) -> pd.DataFrame:
    """Per month from `first` to `last` (YYYY-MM): weekdays, Saturdays and Sundays-or-holidays.

    A holiday that falls on a weekday or a Saturday counts as a Sunday.
    """
    start = pd.Timestamp(first + "-01")
    end = pd.Timestamp(last + "-01") + pd.offsets.MonthEnd(0)
    days = pd.date_range(start, end, freq="D")
    hol = set()
    for y in range(start.year, end.year + 1):
        hol |= holidays(y, municipality)
    is_hol = np.array([d.date() in hol for d in days])
    wd = days.weekday.to_numpy()
    frame = pd.DataFrame(
        {
            "weekday": (wd < 5) & ~is_hol,
            "saturday": (wd == 5) & ~is_hol,
            "sunday_holiday": (wd == 6) | is_hol,
        },
        index=days,
    )
    out = frame.groupby(days.to_period("M")).sum()
    out.index = out.index.to_timestamp()
    return out.astype(float)


def weekday_equivalents(counts: pd.DataFrame, sat: float, sun: float) -> pd.Series:
    """Days in a month expressed as weekdays: weekdays + sat x Saturdays + sun x Sundays-and-holidays."""
    return counts["weekday"] + sat * counts["saturday"] + sun * counts["sunday_holiday"]


def remove_closures(equiv: pd.Series, closures: dict[str, float]) -> pd.Series:
    """Subtract lost weekday-equivalents (date -> share of that day's service lost) from each month."""
    out = equiv.copy()
    for day, lost in closures.items():
        month = pd.Timestamp(day).to_period("M").to_timestamp()
        out.loc[month] -= lost
    return out


def two_way_residual(y: pd.Series) -> np.ndarray:
    """Residual of y after year and calendar-month fixed effects (least squares)."""
    idx = y.index
    years = pd.get_dummies(idx.year.astype(str), drop_first=True).to_numpy(float)
    months = pd.get_dummies(idx.month.astype(str), drop_first=True).to_numpy(float)
    x = np.column_stack([np.ones(len(y)), years, months])
    beta, *_ = np.linalg.lstsq(x, y.to_numpy(float), rcond=None)
    return y.to_numpy(float) - x @ beta


def fit_day_weights(passengers: pd.Series, counts: pd.DataFrame, grid: np.ndarray) -> tuple[float, float, float]:
    """Saturday and Sunday weights that best explain monthly passengers.

    Model: log passengers = year effect + calendar-month effect + log(weekday-equivalents(sat, sun)).
    The month effects take out seasonality; what identifies the weights is that the same calendar month
    has a different mix of weekdays, Saturdays and holidays from one year to the next. Grid search over
    sat in `grid` and sun in `grid` with sun <= sat. Returns (sat, sun, sum of squared residuals).
    """
    y = np.log(passengers)
    c = counts.loc[passengers.index]
    best = (np.nan, np.nan, np.inf)
    for s in grid:
        for u in grid:
            if u > s:
                continue
            r = y - np.log(weekday_equivalents(c, s, u))
            sse = float((two_way_residual(r) ** 2).sum())
            if sse < best[2]:
                best = (float(s), float(u), sse)
    return best


def change(num: float, den: float) -> float:
    """Percentage change from den to num."""
    return 100.0 * (num / den - 1.0)


def period_total(series: pd.Series, year: int, months: tuple[int, ...]) -> float:
    s = series[(series.index.year == year) & series.index.month.isin(months)]
    if len(s) != len(months):
        raise ValueError(f"{series.name}: {year} has {len(s)} of {len(months)} months")
    return float(s.sum())


def period_change(series: pd.Series, year: int, months: tuple[int, ...], per: pd.Series | None = None) -> float:
    """Change of a monthly series between the same months of year-1 and year, optionally per unit of `per`."""
    a, b = period_total(series, year, months), period_total(series, year - 1, months)
    if per is not None:
        a /= period_total(per, year, months)
        b /= period_total(per, year - 1, months)
    return change(a, b)


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])
