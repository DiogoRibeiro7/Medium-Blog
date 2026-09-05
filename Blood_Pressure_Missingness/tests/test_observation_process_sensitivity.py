"""Regression tests for observation-process sensitivity analysis."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import analysis as primary_analysis
import observation_process_sensitivity as sensitivity


class ObservationProcessSensitivityTests(unittest.TestCase):
    """Protect observation-process sensitivity invariants across data refreshes."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the committed privacy-safe snapshot once for all tests."""

        cls.snapshot_path = PROJECT_DIR / "data" / "analysis_snapshot.csv"
        cls.days = sensitivity.load_observed_days(cls.snapshot_path)
        cls.results = sensitivity.fit_sensitivity_models(cls.days)
        cls.primary_records = primary_analysis.load_snapshot(cls.snapshot_path)
        cls.primary_trend = primary_analysis.linear_trend(
            cls.primary_records,
            "mean_systolic_mmHg",
        )

    def _estimate(self, name: str) -> dict[str, float | str]:
        """Return one named sensitivity estimate from the machine results."""

        estimates = self.results["trend_estimates"]
        self.assertIsInstance(estimates, list)
        for item in estimates:
            self.assertIsInstance(item, dict)
            if item.get("name") == name:
                return item
        self.fail(f"Sensitivity estimate {name!r} was not found.")

    def test_sensitivity_loader_matches_primary_observed_day_selection(self) -> None:
        """Sensitivity and primary analysis must use the same observed days."""

        primary_days = [
            record.day_index
            for record in primary_analysis.observed_records(self.primary_records)
        ]
        sensitivity_days = [day.day_index for day in self.days]
        self.assertEqual(sensitivity_days, primary_days)
        self.assertEqual(self.results["n_observed_days"], len(primary_days))

    def test_equal_day_trend_matches_primary_analysis(self) -> None:
        """Equal-day sensitivity fit must reproduce the primary HC3 trend."""

        estimate = self._estimate("equal_day")
        self.assertAlmostEqual(
            float(estimate["slope_per_30_days"]),
            float(self.primary_trend["slope_per_30_days"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(estimate["ci95_low_per_30_days"]),
            float(self.primary_trend["ci95_low_per_30_days"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(estimate["ci95_high_per_30_days"]),
            float(self.primary_trend["ci95_high_per_30_days"]),
            places=10,
        )

    def test_sampling_adjustment_weakens_but_preserves_negative_trend(self) -> None:
        """Adjusting for observation intensity should not be presented as neutral."""

        baseline = self._estimate("equal_day")
        adjusted = self._estimate("sampling_adjusted")
        self.assertLess(float(adjusted["slope_per_30_days"]), 0.0)
        self.assertGreater(
            float(adjusted["slope_per_30_days"]),
            float(baseline["slope_per_30_days"]),
        )
        self.assertLess(float(adjusted["ci95_high_per_30_days"]), 0.0)

    def test_inverse_intensity_stress_test_crosses_zero(self) -> None:
        """The deliberate sparse-day stress test should expose trend fragility."""

        estimate = self._estimate("inverse_intensity_stress")
        self.assertLess(float(estimate["ci95_low_per_30_days"]), 0.0)
        self.assertGreater(float(estimate["ci95_high_per_30_days"]), 0.0)

    def test_inverse_intensity_is_not_labeled_as_ipw(self) -> None:
        """No inverse-probability interpretation is allowed without probabilities."""

        interpretation = self.results["interpretation"]
        self.assertIsInstance(interpretation, dict)
        self.assertFalse(interpretation["identified_missingness_mechanism"])
        self.assertTrue(interpretation["inverse_intensity_is_stress_test_not_ipw"])

    def test_loader_rejects_invalid_observed_flag(self) -> None:
        """Only observation flags 0 and 1 are valid snapshot values."""

        rows = [
            {
                "day_index": "0",
                "observed": "2",
                "n_readings": "1",
                "mean_systolic_mmHg": "120",
            },
            {
                "day_index": "1",
                "observed": "1",
                "n_readings": "1",
                "mean_systolic_mmHg": "119",
            },
            {
                "day_index": "2",
                "observed": "1",
                "n_readings": "1",
                "mean_systolic_mmHg": "118",
            },
            {
                "day_index": "3",
                "observed": "1",
                "n_readings": "1",
                "mean_systolic_mmHg": "117",
            },
        ]
        with self._temporary_snapshot(rows) as path, self.assertRaisesRegex(
            ValueError,
            "observed must be 0 or 1",
        ):
            sensitivity.load_observed_days(path)

    def test_loader_rejects_duplicate_day_indices(self) -> None:
        """Duplicate calendar-day rows must not silently alter the estimand."""

        rows = [
            {
                "day_index": "0",
                "observed": "1",
                "n_readings": "1",
                "mean_systolic_mmHg": "120",
            },
            {
                "day_index": "1",
                "observed": "1",
                "n_readings": "1",
                "mean_systolic_mmHg": "119",
            },
            {
                "day_index": "1",
                "observed": "1",
                "n_readings": "2",
                "mean_systolic_mmHg": "118",
            },
            {
                "day_index": "2",
                "observed": "1",
                "n_readings": "1",
                "mean_systolic_mmHg": "117",
            },
        ]
        with self._temporary_snapshot(rows) as path, self.assertRaisesRegex(
            ValueError,
            "day_index values must be unique",
        ):
            sensitivity.load_observed_days(path)

    @staticmethod
    def _temporary_snapshot(rows: list[dict[str, str]]) -> tempfile.TemporaryDirectory:
        """Create a temporary CSV snapshot context for malformed-input tests."""

        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "snapshot.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "day_index",
                    "observed",
                    "n_readings",
                    "mean_systolic_mmHg",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        class SnapshotContext:
            def __enter__(self) -> Path:
                return path

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                directory.cleanup()

        return SnapshotContext()  # type: ignore[return-value]


if __name__ == "__main__":
    unittest.main()
