"""Narrative contract for global systolic time-form sensitivity."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULT_PATH = PROJECT_DIR / "figures" / "global_time_form_sensitivity.json"
NOTE_PATH = PROJECT_DIR / "GLOBAL_TIME_FORM_SENSITIVITY.md"


class GlobalTimeFormNarrativeContractTests(unittest.TestCase):
    """Keep the public methods note synchronized with the generated result."""

    def test_methods_note_reports_current_time_form_results(self) -> None:
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        note = NOTE_PATH.read_text(encoding="utf-8")

        estimates = payload["estimates"]
        expected = [
            (
                "linear_equal_day_hc3",
                "Equal-day linear OLS, HC3",
            ),
            (
                "quadratic_end_to_end_average_hc3",
                "Quadratic endpoint-average, HC3",
            ),
            (
                "theil_sen_median_pairwise_slope",
                "Theil–Sen median pairwise slope",
            ),
        ]

        for key, label in expected:
            item = estimates[key]
            estimate = float(item["estimate_per_30_days"])
            low = float(item["ci95_low_per_30_days"])
            high = float(item["ci95_high_per_30_days"])
            row = f"| {label} | {estimate:.2f} | [{low:.2f}, {high:.2f}] |"
            self.assertIn(row, note)

        self.assertEqual(
            payload["all_95_percent_intervals_below_zero"],
            all(
                float(item["ci95_high_per_30_days"]) < 0.0
                for item in estimates.values()
            ),
        )
        if payload["all_95_percent_intervals_below_zero"]:
            self.assertIn("All three current intervals are below zero.", note)

        self.assertIn(
            "Numerical closeness does not make the estimands interchangeable.",
            note,
        )
        self.assertIn("not a search for a preferred trajectory", note)
        self.assertIn("32-day interval contains no measurements", note)


if __name__ == "__main__":
    unittest.main()
