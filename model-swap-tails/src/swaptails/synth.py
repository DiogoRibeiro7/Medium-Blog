"""Synthetic data with a known structure, for the tests and for dry runs of the pipeline.

Load = level x daily profile x day-type factor x (1 + temperature effect) + noise, where the
day type follows the real holiday calendar of each country (national holidays and the
year end at Sunday level, regional holidays and bridge days in between). It is written to
the same cache files that `swaptails fetch` writes, so every later step runs unchanged.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .calendar import build_calendar, in_year_end
from .config import Config
from .data import cache_path, grid_bounds, write_manifest, write_series


def synthesise(cfg: Config, countries: list[str], seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    first, last = grid_bounds(cfg)
    index = pd.date_range(first, last, freq="h", tz="UTC")
    manifest = {"synthetic": True, "countries": {}}
    for n, key in enumerate(countries):
        country = cfg.countries[key]
        cal = build_calendar(country, cfg.first_day - timedelta(days=7), cfg.last_day + timedelta(days=7))
        local = index.tz_convert(country.tz)
        hour = local.hour.to_numpy()
        doy = local.dayofyear.to_numpy()
        dates = local.date
        factor = np.empty(index.size)
        for i, d in enumerate(dates):
            if cal.is_national(d) or (
                in_year_end(d, (12, 24), (1, 1)) and d.weekday() < 5 and d.month == 12 and d.day >= 24
            ):
                factor[i] = 0.72
            elif d.weekday() == 6:
                factor[i] = 0.72
            elif d.weekday() == 5:
                factor[i] = 0.82
            elif cal.is_regional(d):
                factor[i] = 0.92
            elif cal.is_national(d - timedelta(days=1)) and d.weekday() == 4:
                factor[i] = 0.84
            else:
                factor[i] = 1.0
        temp = 14 - 9 * np.cos(2 * np.pi * (doy - 20) / 365.25) + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
        temp = temp + rng.normal(0, 1.5, index.size)
        heating = 0.012 * np.clip(15 - temp, 0, None)
        cooling = 0.008 * np.clip(temp - 24, 0, None)
        profile = 0.85 + 0.15 * np.sin(np.pi * np.clip(hour - 6, 0, 16) / 16) ** 0.8
        level = (6000 + 2500 * n) * profile * factor * (1 + heating + cooling)
        load = level * (1 + rng.normal(0, 0.01, index.size))
        load = pd.Series(load, index=index)
        tso = load * (1 + rng.normal(0, 0.025, index.size))
        tso[(local.month == 12) & (local.day == 25)] *= 1.15  # the reference has its own bad days
        temp_series = pd.Series(temp + rng.normal(0, 0.8, index.size), index=index)
        entry = {
            "load": {"sha256": write_series(cache_path(cfg, "load", key), load)},
            "tso": {"sha256": write_series(cache_path(cfg, "tso", key), tso)},
            "weather": {"sha256": write_series(cache_path(cfg, "temp", key), temp_series)},
        }
        manifest["countries"][key] = entry
    write_manifest(cfg, manifest)
