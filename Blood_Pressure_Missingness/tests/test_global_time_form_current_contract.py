"""Current-result contract for global systolic time-form sensitivity."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from blood_pressure_missingness import public_analysis as primary
from blood_pressure_missingness.analyses import global_time_form as analysis

PROJECT_DIR = Path(__file__).resolve().parent.parent


class GlobalTimeFormCurrentContractTests(unittest.TestCase):
    """Keep the committed sensitivity artifact tied to the current snapshot."""

    def test_committed_artifact_matches_recomputed_current_results(self) -> None:
        records = primary.load_snapshot(PROJECT_DIR / "data" / "analysis_snapshot.csv")
        recomputed = analysis.analyze_global_time_form_sensitivity(records)
        path = PROJECT_DIR / "figures" / "global_time_form_sensitivity.json"
        committed = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(committed["n_observed_days"], recomputed["n_observed_days"])
        self.assertEqual(
            committed["first_observed_day_index"],
            recomputed["first_observed_day_index"],
        )
        self.assertEqual(
            committed["last_observed_day_index"],
            recomputed["last_observed_day_index"],
        )
        self.assertEqual(
            committed["all_95_percent_intervals_below_zero"],
            recomputed["all_95_percent_intervals_below_zero"],
        )

        for name, expected in recomputed["estimates"].items():
            actual = committed["estimates"][name]
            for field in (
                "estimate_per_30_days",
                "ci95_low_per_30_days",
                "ci95_high_per_30_days",
            ):
                self.assertAlmostEqual(actual[field], expected[field], places=10)

        primary_linear = primary.linear_trend(records, "mean_systolic_mmHg")
        linear = committed["estimates"]["linear_equal_day_hc3"]
        self.assertAlmostEqual(
            linear["estimate_per_30_days"],
            primary_linear["slope_per_30_days"],
            places=10,
        )
        self.assertAlmostEqual(
            linear["ci95_low_per_30_days"],
            primary_linear["ci95_low_per_30_days"],
            places=10,
        )
        self.assertAlmostEqual(
            linear["ci95_high_per_30_days"],
            primary_linear["ci95_high_per_30_days"],
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
