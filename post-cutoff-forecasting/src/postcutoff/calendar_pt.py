"""Portuguese national public holidays, on the hourly UTC grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Lisbon"


def holiday_dates(years) -> dict:
    import holidays

    return dict(holidays.country_holidays("PT", years=list(years), categories=("public",)))


def holiday_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """1.0 for hours that fall on a national public holiday in Lisbon local time."""
    local = index.tz_convert(LOCAL_TZ)
    dates = holiday_dates(range(local.year.min(), local.year.max() + 1))
    days = pd.Index(local.date)
    return days.isin(list(dates)).astype(float)
