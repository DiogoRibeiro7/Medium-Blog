"""A synthetic stand-in for the load series, for tests and for CI runs without network.

It has the shape of national load (daily and weekly cycles, a winter and a summer peak,
holiday dips, persistent noise) and reproduces the blackout masked in the config, so the
masking path is exercised. It is never used for any number in the article.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_pt import LOCAL_TZ, holiday_flags
from .config import Config
from .data import mask_hours, write_snapshot


def synthetic_series(cfg: Config, end: str = "2026-10-08", seed: int = 7) -> pd.Series:
    index = pd.date_range(cfg.first_day, pd.Timestamp(end, tz="UTC"), freq="h", inclusive="left")
    local = index.tz_convert(LOCAL_TZ)
    rng = np.random.default_rng(seed)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()
    years = (index - index[0]).days.to_numpy() / 365.25

    daily = 0.82 + 0.10 * np.sin((hour - 6) / 24 * 2 * np.pi) + 0.08 * np.exp(-((hour - 20) ** 2) / 6)
    weekly = np.where(dow == 6, 0.84, np.where(dow == 5, 0.91, 1.0))
    annual = 1.0 + 0.09 * np.cos((doy - 15) / 365.25 * 2 * np.pi) + 0.04 * np.exp(-((doy - 205) ** 2) / 400)
    holiday = np.where(holiday_flags(index) > 0, 0.86, 1.0)
    noise = np.empty(index.size)
    e = rng.normal(0, 0.012, index.size)
    noise[0] = e[0]
    for t in range(1, index.size):
        noise[t] = 0.95 * noise[t - 1] + e[t]
    y = 6200 * (1 + 0.01 * years) * daily * weekly * annual * holiday * (1 + noise)
    masked = mask_hours(index, cfg)
    y[masked] = rng.uniform(80, 1200, masked.sum())
    return pd.Series(y, index=index, name="load_mw")


def write_synthetic(cfg: Config, end: str = "2026-10-08") -> dict:
    series = synthetic_series(cfg, end)
    record = {"synthetic": True, "requests": [], "fetched_at": None, "cutoff": series.index.max().isoformat()}
    return write_snapshot(series, cfg, record)
