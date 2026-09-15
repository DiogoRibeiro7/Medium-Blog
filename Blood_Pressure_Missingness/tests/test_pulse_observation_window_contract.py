"""Public contract for pulse observation-window coverage."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


class PulseObservationWindowContractTests(unittest.TestCase):
    """Freeze the privacy-safe observation-window diagnostic surface."""

    def test_current_observation_window_contract(self) -> None:
        path = DATA_DIR / "pulse_oximeter_observation_window.json"
        if not path.exists():
            return

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"window", "pulse_capture", "interpretation"})

        window = payload["window"]
        capture = payload["pulse_capture"]
        interpretation = payload["interpretation"]

        self.assertEqual(
            window["definition"],
            "first_observed_pulse_day_through_bp_calendar_end",
        )
        self.assertTrue(interpretation["window_start_is_first_observed_pulse_day"])
        self.assertFalse(interpretation["device_introduction_date_identified"])
        self.assertFalse(interpretation["pre_window_absence_treated_as_missing"])
        self.assertFalse(interpretation["missingness_mechanism_identified"])
        self.assertFalse(interpretation["clinical_thresholds_applied"])
        self.assertFalse(interpretation["causal_interpretation"])

        if window["defined"]:
            self.assertIsNone(window["reason"])
            self.assertLessEqual(window["start_day_index"], window["end_day_index"])
            self.assertGreater(window["calendar_days"], 0)
            self.assertGreater(window["blood_pressure_observed_days"], 0)
            self.assertGreater(window["blood_pressure_readings"], 0)
            self.assertLessEqual(
                capture["pulse_observed_days"],
                window["blood_pressure_observed_days"],
            )
            for key in (
                "pulse_day_capture_of_bp_observed_days",
                "spo2_reading_capture_of_bp_readings",
                "bpm_spo2_reading_capture_of_bp_readings",
                "paired_bpm_reading_capture_of_bp_readings",
            ):
                self.assertGreaterEqual(capture[key], 0.0)
                self.assertLessEqual(capture[key], 1.0)


if __name__ == "__main__":
    unittest.main()
