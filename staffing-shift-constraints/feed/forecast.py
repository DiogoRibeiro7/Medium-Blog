"""Forecasts of the 168 hours of a week, from data up to the end of the week two weeks earlier (PLAN.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from feed.calls import WEEK, Calls

FORECASTS = ("two_weeks", "mean4", "gbm", "oracle")
LAG_WEEKS = (2, 3, 4, 5)
FIRST_TRAIN_MONDAY = pd.Timestamp("2016-02-08")  # sixth week of 2016: all lags exist
GBM = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbosity": -1,
}
ROUNDS = 400


def holidays() -> set[pd.Timestamp]:
    return set(USFederalHolidayCalendar().holidays(start="2015-12-01", end="2020-01-31"))


def week_slice(c: Calls, monday: pd.Timestamp) -> slice:
    i = c.index(monday)
    return slice(i, i + WEEK)


def lagged(c: Calls, monday: pd.Timestamp, weeks: int) -> np.ndarray:
    return c.calls[week_slice(c, monday - pd.Timedelta(days=7 * weeks))]


def simple(c: Calls, monday: pd.Timestamp, name: str) -> np.ndarray:
    if name == "two_weeks":
        return lagged(c, monday, 2).copy()
    if name == "mean4":
        return np.mean([lagged(c, monday, k) for k in LAG_WEEKS], axis=0)
    if name == "oracle":
        return c.calls[week_slice(c, monday)].copy()
    raise ValueError(name)


def features(c: Calls, monday: pd.Timestamp, hol: set[pd.Timestamp]) -> pd.DataFrame:
    """Features of the 168 hours of the week starting `monday`, using data up to the end of week w - 2 only."""
    hours = pd.date_range(monday, periods=WEEK, freq="h")
    days = hours.normalize()
    f = {
        "hour": hours.hour,
        "weekday": hours.dayofweek,
        "month": hours.month,
        "doy": hours.dayofyear,
        "holiday": days.isin(hol),
        "holiday_before": (days - pd.Timedelta(days=1)).isin(hol),
        "holiday_after": (days + pd.Timedelta(days=1)).isin(hol),
    }
    for k in LAG_WEEKS:
        f[f"lag_w{k}"] = lagged(c, monday, k)
    f["week_total_w2"] = np.full(WEEK, lagged(c, monday, 2).sum())
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


def train_mondays(last_monday: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(FIRST_TRAIN_MONDAY, last_monday, freq="7D"))


def fit_gbm(c: Calls, last_train_monday: pd.Timestamp, settings: dict | None = None):
    import lightgbm as lgb

    hol = holidays()
    weeks = train_mondays(last_train_monday)
    x = pd.concat([features(c, m, hol) for m in weeks], ignore_index=True)
    y = np.concatenate([c.calls[week_slice(c, m)] for m in weeks])
    params = {**GBM, **(settings or {})}
    rounds = params.pop("rounds", ROUNDS)
    model = lgb.train(params, lgb.Dataset(x, y), num_boost_round=rounds)
    return model, hol


def gbm_forecast(model, hol, c: Calls, monday: pd.Timestamp) -> np.ndarray:
    return np.maximum(model.predict(features(c, monday, hol)), 0.0)
