"""The tools in core.py against answers known by construction."""

from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core  # noqa: E402


class TestCalendar(unittest.TestCase):
    def test_easter(self):
        known = {2019: (4, 21), 2024: (3, 31), 2025: (4, 20), 2026: (4, 5), 2038: (4, 25)}
        for y, (m, d) in known.items():
            self.assertEqual(core.easter(y), dt.date(y, m, d))

    def test_holidays_2026(self):
        h = core.holidays(2026)
        self.assertEqual(len(h), 13)
        self.assertIn(dt.date(2026, 4, 3), h)  # Good Friday
        self.assertIn(dt.date(2026, 6, 4), h)  # Corpus Christi
        self.assertNotIn(dt.date(2026, 6, 13), h)
        self.assertIn(dt.date(2026, 6, 13), core.holidays(2026, "Lisboa"))
        self.assertIn(dt.date(2026, 6, 24), core.holidays(2026, "Porto"))

    def test_suspended_holidays(self):
        h = core.holidays(2014)
        self.assertEqual(len(h), 9)
        self.assertNotIn(dt.date(2014, 10, 5), h)
        self.assertIn(dt.date(2016, 10, 5), core.holidays(2016))

    def test_day_counts_add_up(self):
        c = core.day_counts("2024-01", "2026-08", "Lisboa")
        days = c.sum(axis=1)
        expected = pd.Series([pd.Period(m, "M").days_in_month for m in c.index], index=c.index, dtype=float)
        pd.testing.assert_series_equal(days, expected, check_names=False)

    def test_august_2026(self):
        # August 2026: 21 weekdays, 5 Saturdays (1, 8, 15, 22, 29), 5 Sundays; 15 August is a Saturday.
        c = core.day_counts("2026-08", "2026-08", "Lisboa").iloc[0]
        self.assertEqual((c.weekday, c.saturday, c.sunday_holiday), (21.0, 4.0, 6.0))


class TestWeights(unittest.TestCase):
    def test_recovers_known_weights(self):
        counts = core.day_counts("2015-01", "2019-12", "Lisboa")
        rng = np.random.default_rng(7)
        level = pd.Series(1.0 + 0.04 * (counts.index.year - 2015), index=counts.index)
        season = pd.Series(1.0 + 0.1 * np.sin(counts.index.month.to_numpy() / 12 * 2 * np.pi), index=counts.index)
        pax = 1000 * level * season * core.weekday_equivalents(counts, 0.6, 0.4) * np.exp(rng.normal(0, 0.002, len(counts)))
        grid = np.round(np.arange(0.2, 0.9001, 0.02), 2)
        sat, sun, _ = core.fit_day_weights(pax, counts, grid)
        self.assertAlmostEqual(sat, 0.6, delta=0.06)
        self.assertAlmostEqual(sun, 0.4, delta=0.06)


class TestChanges(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
        self.s = pd.Series([100.0] * 12 + [110.0] * 12, index=idx)
        self.per = pd.Series([20.0] * 12 + [22.0] * 12, index=idx)

    def test_period_change(self):
        self.assertAlmostEqual(core.period_change(self.s, 2025, (1, 2, 3)), 10.0)
        self.assertAlmostEqual(core.period_change(self.s, 2025, (1, 2, 3), self.per), 0.0)

    def test_missing_month_raises(self):
        with self.assertRaises(ValueError):
            core.period_change(self.s.drop(self.s.index[0]), 2024, (1, 2))

    def test_remove_closures(self):
        e = pd.Series([20.0, 21.0], index=pd.to_datetime(["2025-11-01", "2025-12-01"]))
        out = core.remove_closures(e, {"2025-12-11": 1.0, "2025-12-03": 0.35})
        self.assertAlmostEqual(out.iloc[1], 19.65)
        self.assertAlmostEqual(out.iloc[0], 20.0)
        self.assertAlmostEqual(e.iloc[1], 21.0)


if __name__ == "__main__":
    unittest.main()
