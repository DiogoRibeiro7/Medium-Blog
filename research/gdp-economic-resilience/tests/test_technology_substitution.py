"""Synthetic accounting and sensitivity tests for substitution scenarios."""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import eurostat_sectors as euro
import pipeline as p
import technology_substitution as s


class TestScenarioAccounting(unittest.TestCase):
    def test_unchanged(self) -> None:
        result = s.effects(0.2, s.NAMED[0])
        self.assertEqual(result["nominal_change_pct_baseline_aggregate"], 0)
        self.assertEqual(result["base_price_change_pct_baseline_aggregate"], 0)

    def test_price_compression_preserves_base_price_volume(self) -> None:
        result = s.effects(0.2, s.NAMED[1])
        self.assertAlmostEqual(result["nominal_change_pct_baseline_aggregate"], -5)
        self.assertEqual(result["base_price_change_pct_baseline_aggregate"], 0)
        self.assertAlmostEqual(result["price_effect_pct_baseline_aggregate"], -5)

    def test_unit_elastic_rebound_preserves_nominal_value(self) -> None:
        result = s.effects(0.2, s.NAMED[2])
        self.assertAlmostEqual(result["nominal_change_pct_baseline_aggregate"], 0)
        self.assertAlmostEqual(result["base_price_change_pct_baseline_aggregate"], 20 / 3)

    def test_displacement_without_replacement(self) -> None:
        result = s.effects(0.2, s.NAMED[3])
        self.assertAlmostEqual(result["nominal_change_pct_baseline_aggregate"], -5)
        self.assertAlmostEqual(result["base_price_change_pct_baseline_aggregate"], -5)

    def test_combined_changes_are_multiplicative(self) -> None:
        result = s.effects(0.2, s.NAMED[4])
        self.assertAlmostEqual(result["nominal_change_pct_baseline_aggregate"], -8.75)
        self.assertAlmostEqual(result["base_price_change_pct_baseline_aggregate"], -5)
        self.assertAlmostEqual(result["price_effect_pct_baseline_aggregate"], -3.75)

    def test_domestic_replacement(self) -> None:
        result = s.effects(0.2, s.NAMED[-1])
        self.assertAlmostEqual(result["replacement_pct_baseline_aggregate"], 5)
        self.assertAlmostEqual(result["nominal_change_pct_baseline_aggregate"], 0)
        self.assertAlmostEqual(result["base_price_change_pct_baseline_aggregate"], 0)

    def test_grid_identity_and_unique_parameters(self) -> None:
        scenarios = s.grid()
        self.assertEqual(len(scenarios), 81)
        self.assertEqual(len({(x.displaced, x.price_cut, x.elasticity,
                              x.domestic_replacement) for x in scenarios}), 81)
        for scenario in scenarios:
            with self.subTest(scenario=scenario.name):
                result = s.effects(0.35, scenario)
                self.assertAlmostEqual(result["accounting_residual_pp"], 0)
                self.assertGreaterEqual(result["portfolio_plus_replacement_nominal_factor"], 0)

    def test_invalid_parameters(self) -> None:
        for params in ((-0.1, 0, 0, 0, 0), (0, 1, 0, 0, 0),
                       (0, 0, math.inf, 0, 0), (0, 0, 0, 1.1, 0)):
            with self.assertRaises(ValueError):
                s.Scenario("invalid", *params)

    def test_missing_and_invalid_exposures(self) -> None:
        for share in (None, -0.1, 1.1, math.nan):
            self.assertIsNone(s.effects(share, s.NAMED[1])[
                "nominal_change_pct_baseline_aggregate"])
        self.assertEqual(s.effects(0, s.NAMED[1])[
            "nominal_change_pct_baseline_aggregate"], 0)

    def test_reverse_stress(self) -> None:
        exposure = s.Exposure("synthetic", "PT", 2024, "target", "J", "gva", 0.2)
        result = s.reverse_stress(exposure, 5)
        self.assertAlmostEqual(result["required_isolated_cut_fraction"], 0.25)
        self.assertTrue(result["reachable_by_volume_cut"])
        self.assertFalse(s.reverse_stress(exposure, 25)["reachable_by_volume_cut"])
        limit = s.reverse_stress(exposure, 20)
        self.assertTrue(limit["reachable_by_volume_cut"])
        self.assertFalse(limit["reachable_by_strictly_positive_price"])


class TestExposureAccounting(unittest.TestCase):
    def test_nominal_group_and_missing_components(self) -> None:
        gva = {
            ("PT", 2024, code, "CP_MEUR"): euro.Cell(value, flag, "observed")
            for code, value, flag in (("TOTAL", 100, ""), ("J", 10, "p"),
                                      ("K", 5, ""), ("M", 15, ""), ("N", 5, ""))
        }
        with patch.object(euro, "GEO", ("PT",)), patch.object(s, "EURO_YEARS", (2024,)):
            records = s.euro_exposures(gva)
            group = next(x for x in records if x.portfolio == "selected_business_services")
            self.assertAlmostEqual(group.share, 0.35)
            self.assertEqual(group.source_flags, "p")
            del gva["PT", 2024, "N", "CP_MEUR"]
            group = next(x for x in s.euro_exposures(gva)
                         if x.portfolio == "selected_business_services")
            self.assertIsNone(group.share)

    def test_negative_sector_cannot_be_hidden_by_group_sum(self) -> None:
        gva = {
            ("PT", 2024, code, "CP_MEUR"): euro.Cell(value, "", "observed")
            for code, value in (("TOTAL", 100), ("J", -1), ("K", 5), ("M", 15), ("N", 5))
        }
        with patch.object(euro, "GEO", ("PT",)), patch.object(s, "EURO_YEARS", (2024,)):
            group = next(x for x in s.euro_exposures(gva)
                         if x.portfolio == "selected_business_services")
            self.assertIsNone(group.share)

    def test_fixed_eu_weights_and_missing_members(self) -> None:
        data = {(p.GDP, "PRT", 2021): 100, (p.GDP, "DEU", 2021): 300,
                (p.SRV, "PRT", 2021): 80, (p.SRV, "DEU", 2021): 40}
        with patch.object(p, "COUNTRIES", ("DEU", "PRT")), patch.object(
            p, "EU27", ("DEU", "PRT")
        ), patch.object(s, "WDI_YEARS", (2021,)):
            result = next(x for x in s.wdi_exposures(p.Snapshot(data))
                          if x.entity == "EU27" and x.portfolio == "all_services_stress")
            self.assertAlmostEqual(result.share, 0.5)
            del data[p.SRV, "DEU", 2021]
            result = next(x for x in s.wdi_exposures(p.Snapshot(data))
                          if x.entity == "EU27" and x.portfolio == "all_services_stress")
            self.assertIsNone(result.share)


if __name__ == "__main__":
    unittest.main()
