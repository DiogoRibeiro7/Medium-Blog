"""Narrative contract for full-history versus observed-window pulse coverage."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


class PulseObservationWindowNarrativeContractTests(unittest.TestCase):
    """Keep the two pulse coverage denominators explicit in public prose."""

    def test_readme_reports_both_pulse_coverage_denominators(self) -> None:
        readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
        diagnostics = json.loads(
            (PROJECT_DIR / "data" / "pulse_oximeter_diagnostics.json").read_text(
                encoding="utf-8"
            )
        )
        window = json.loads(
            (PROJECT_DIR / "data" / "pulse_oximeter_observation_window.json").read_text(
                encoding="utf-8"
            )
        )

        total_measurements = int(diagnostics["n_measurements"])
        spo2_observed = int(diagnostics["coverage"]["spo2_observed"])
        full_history_percent = 100.0 * spo2_observed / total_measurements

        bp_window_readings = int(window["window"]["blood_pressure_readings"])
        window_spo2 = int(window["pulse_capture"]["spo2_readings"])
        observed_window_percent = 100.0 * window_spo2 / bp_window_readings

        self.assertIn(
            f"pulse-oximeter row coverage is therefore **{full_history_percent:.1f}%**",
            readme,
        )
        self.assertIn(
            f"observed-window reading-level capture of **{observed_window_percent:.1f}%**",
            readme,
        )
        self.assertIn(
            "That is a full-history denominator and should not be read as the capture rate once pulse observations begin.",
            readme,
        )
        self.assertIn(
            "This does **not** identify the true date when the pulse oximeter was introduced.",
            readme,
        )


if __name__ == "__main__":
    unittest.main()
