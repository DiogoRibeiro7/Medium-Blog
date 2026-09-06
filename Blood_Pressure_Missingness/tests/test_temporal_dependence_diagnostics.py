"""Regression tests for calendar-gap-aware temporal dependence diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import analysis as primary
import temporal_dependence_diagnostics as temporal


class TemporalDependenceDiagnosticsTests(unittest.TestCase):
    """Protect calendar-spacing semantics and the established residual estimand."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = primary.load_snapshot(
            PROJECT_DIR / "data" / "analysis_snapshot.csv"
        )
        cls.results = temporal.diagnose_temporal_dependence(cls.records)

    def test_observed_order_is_not_treated_as_daily_spacing(self) -> None:
        """Observed-row adjacency must preserve the actual calendar gaps."""

        spacing = self.results["observed_order_spacing"]
        self.assertIsInstance(spacing, dict)
        self.assertFalse(spacing["observed_order_is_equally_spaced_daily"])
        self.assertEqual(
            int(spacing["n_consecutive_observed_pairs"]),
            len(primary.observed_records(self.records)) - 1,
        )
        counts = spacing["calendar_gap_days_counts"]
        self.assertIsInstance(counts, dict)
        self.assertIn("33", counts)
        self.assertGreater(int(spacing["n_not_one_day_apart"]), 0)

    def test_exact_one_day_pairs_are_restricted_to_calendar_neighbors(self) -> None:
        """Lag-one diagnostics must use actual one-day separation, not row order."""

        lags = self.results["exact_calendar_lag_residual_correlations"]
        self.assertIsInstance(lags, list)
        lag_one = next(item for item in lags if int(item["lag_days"]) == 1)
        spacing = self.results["observed_order_spacing"]
        self.assertEqual(
            int(lag_one["n_pairs"]), int(spacing["n_exactly_one_day_apart"])
        )
        self.assertLess(
            int(lag_one["n_pairs"]), int(spacing["n_consecutive_observed_pairs"])
        )

    def test_residual_model_reuses_gap_aware_common_linear_estimand(self) -> None:
        """The diagnostic must not silently introduce a new trend estimand."""

        model = self.results["residual_model"]
        self.assertIsInstance(model, dict)
        self.assertEqual(model["name"], "gap_aware_common_linear")
        self.assertLess(float(model["post_minus_pre_mean_difference_mmHg"]), 0.0)
        self.assertLess(float(model["common_within_episode_slope_per_30_days"]), 0.0)

    def test_interpretation_does_not_claim_formal_independence(self) -> None:
        """Small irregular residual diagnostics must remain descriptive."""

        interpretation = self.results["interpretation"]
        self.assertIsInstance(interpretation, dict)
        self.assertTrue(interpretation["calendar_distance_preserved"])
        self.assertTrue(
            interpretation["exact_lag_pairs_restricted_to_same_gap_defined_episode"]
        )
        self.assertFalse(interpretation["row_order_lag1_is_valid_hac_time_lag"])
        self.assertFalse(interpretation["formal_serial_independence_claimed"])
        self.assertFalse(interpretation["formal_autocorrelation_test_reported"])

    def test_invalid_max_lag_is_rejected(self) -> None:
        """A non-positive calendar lag horizon is structurally invalid."""

        with self.assertRaises(ValueError):
            temporal.diagnose_temporal_dependence(self.records, max_lag_days=0)


if __name__ == "__main__":
    unittest.main()
