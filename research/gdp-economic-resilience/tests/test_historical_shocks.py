"""Synthetic tests of shock accounting, trend estimates, and censoring."""
from __future__ import annotations

import math
import unittest

import historical_shocks as s
import pipeline as p


class TestShockBenchmarks(unittest.TestCase):
    def setUp(self) -> None:
        self.shock = s.Shock("synthetic", 2020, 2019)
        self.series = {y: 100 * 1.02 ** (y - 2019) for y in range(2010, 2023)}

    def test_ols_known_growth(self) -> None:
        self.assertAlmostEqual(s.trend_slope(self.series, 2019, 5), math.log(1.02))
        self.assertAlmostEqual(s.trend_slope(self.series, 2019, 10), math.log(1.02))

    def test_trend_following_has_zero_shortfall(self) -> None:
        _, result = s.measure(self.series, self.shock, "trend_5y", 2022)
        self.assertTrue(result["complete_common_window"])
        self.assertAlmostEqual(result["cumulative_shortfall_pct_benchmark"], 0)
        self.assertEqual(result["level_recovery_status"], "no_drop")

    def test_signed_shortfall(self) -> None:
        data = {2019: 100, 2020: 80, 2021: 90, 2022: 110}
        _, result = s.measure(data, self.shock, "flat", 2022)
        self.assertAlmostEqual(result["cumulative_shortfall_pct_benchmark"], 100 * 20 / 300)
        self.assertAlmostEqual(result["max_drop_from_anchor_pct"], 20)
        self.assertEqual(result["first_drop_year"], 2020)
        self.assertEqual(result["level_recovery_year"], 2022)
        self.assertEqual(result["years_from_onset_to_recovery"], 2)
        _, above = s.measure({2019: 100, 2020: 110, 2021: 120, 2022: 130},
                             self.shock, "flat", 2022)
        self.assertAlmostEqual(above["cumulative_shortfall_pct_benchmark"], -20)

    def test_missing_pretrend_never_imputed(self) -> None:
        data = self.series.copy()
        data.pop(2017)
        _, result = s.measure(data, self.shock, "trend_5y", 2022)
        self.assertFalse(result["complete_common_window"])
        self.assertIsNone(result["cumulative_shortfall_pct_benchmark"])
        self.assertEqual(result["observed_common_years"], 3)
        self.assertEqual(result["comparable_common_years"], 0)

    def test_missing_followup_never_shortens_window(self) -> None:
        _, result = s.measure({2019: 100, 2020: 80, 2022: 110},
                              self.shock, "flat", 2022)
        self.assertFalse(result["complete_common_window"])
        self.assertEqual(result["comparable_common_years"], 2)
        self.assertIsNone(result["cumulative_shortfall_pct_benchmark"])
        self.assertEqual(result["level_recovery_status"], "missing_followup")

    def test_no_return_is_right_censored(self) -> None:
        _, result = s.measure({2019: 100, 2020: 80, 2021: 90, 2022: 95},
                              self.shock, "flat", 2022)
        self.assertEqual(result["level_recovery_status"], "right_censored")
        self.assertIsNone(result["level_recovery_year"])

    def test_recovery_not_unconditionally_first_anchor_crossing(self) -> None:
        _, result = s.measure({2019: 100, 2020: 110, 2021: 80, 2022: 100},
                              self.shock, "flat", 2022)
        self.assertEqual(result["first_drop_year"], 2021)
        self.assertEqual(result["level_recovery_year"], 2022)

    def test_later_crisis_does_not_create_earlier_recovery_event(self) -> None:
        data = {2019: 100, 2020: 101, 2021: 102, 2022: 103, 2023: 80, 2024: 110}
        _, result = s.measure(data, self.shock, "flat", 2024)
        self.assertEqual(result["level_recovery_status"], "no_drop")
        self.assertIsNone(result["first_drop_year"])

    def test_energy_trend_excludes_pandemic_years(self) -> None:
        shock = s.SHOCKS[-1]
        data = {y: 100 * 1.03 ** (y - 2019) for y in range(2010, 2020)}
        data.update({2020: 60, 2021: 80})
        anchor, slope = s.baseline(data, shock, "trend_5y")
        self.assertEqual(anchor, 80)
        self.assertAlmostEqual(slope, math.log(1.03))
        self.assertAlmostEqual(s.expected_value(anchor, slope, 1), 82.4)

    def test_all_shocks_have_same_primary_horizon(self) -> None:
        self.assertTrue(all(x.end - x.onset + 1 == 3 for x in s.SHOCKS))

    def test_missing_anchor(self) -> None:
        _, result = s.measure({2020: 80, 2021: 90, 2022: 100},
                              self.shock, "flat", 2022)
        self.assertEqual(result["level_recovery_status"], "missing_anchor")
        self.assertIsNone(result["max_drop_from_anchor_pct"])

    def test_reject_invalid_followup(self) -> None:
        with self.assertRaises(ValueError):
            s.measure(self.series, self.shock, "flat", 2021)


class TestAssociations(unittest.TestCase):
    def test_pearson(self) -> None:
        self.assertAlmostEqual(s.pearson([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(s.pearson([1, 1, 1], [3, 2, 1]))
        self.assertIsNone(s.pearson([1, 2], [3, 4]))

    def test_pre_shock_covariates_and_missingness(self) -> None:
        rows = [{"country": "PRT", "metric": "real_gdp_per_capita",
                 "shock": "pandemic", "benchmark": "trend_5y",
                 "cumulative_shortfall_pct_benchmark": 8, "complete_common_window": True}]
        data = {(p.AGR, "PRT", 2019): 10, (p.IND, "PRT", 2019): 30,
                (p.SRV, "PRT", 2019): 50, (p.MAN, "PRT", 2020): 99}
        result = s.association_rows(p.Snapshot(data), rows)[0]
        self.assertEqual(result["structure_year"], 2019)
        self.assertIsNone(result["manufacturing_pct_gdp"])
        self.assertAlmostEqual(result["three_sector_hhi"], (100 + 900 + 2500) / 8100)


if __name__ == "__main__":
    unittest.main()
