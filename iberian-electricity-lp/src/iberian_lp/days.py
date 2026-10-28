"""Section 3: four days per calendar month, drawn without replacement within each month by
one seeded generator, months in calendar order."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from iberian_lp.contract import DAYS_PER_MONTH, DAYS_SEED, FIRST_DAY, LAST_DAY


def months() -> list[list[date]]:
    """The study period's days, grouped by calendar month in order."""
    out: list[list[date]] = []
    day = FIRST_DAY
    while day <= LAST_DAY:
        if not out or (out[-1][0].year, out[-1][0].month) != (day.year, day.month):
            out.append([])
        out[-1].append(day)
        day += timedelta(days=1)
    return out


def draw_days() -> list[date]:
    rng = np.random.default_rng(DAYS_SEED)
    drawn: list[date] = []
    for month in months():
        picks = rng.choice(len(month), size=DAYS_PER_MONTH, replace=False)
        drawn += sorted(month[int(i)] for i in picks)
    return drawn
