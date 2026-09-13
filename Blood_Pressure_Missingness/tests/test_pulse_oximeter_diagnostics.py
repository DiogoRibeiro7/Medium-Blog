"""Regression tests for aggregate pulse-oximeter diagnostics."""

from __future__ import annotations

import unittest
from datetime import date, datetime

from blood_pressure_missingness.analyses.pulse_oximeter import (
    build_pulse_oximeter_diagnostics,
)
from blood_pressure_missingness.data_sources.google_sheets import Measurement


def _measurement(
    *,
    bpm: float,
    spo2: float | None,
    bpm_spo2: float | None,
) -> Measurement:
    """Build a minimal validated measurement for diagnostic tests."""

    day = date(2026, 9, 1)
    return Measurement(
        day=day,
        timestamp=datetime(2026, 9, 1, 12, 0),
        systolic=120.0,
        diastolic=80.0,
        pulse_pressure=40.0,
        bpm=bpm,
        pill=None,
        home=None,
        sleep=None,
        meal=None,
        symptoms=None,
        spo2=spo2,
        bpm_spo2=bpm_spo2,
    )


class PulseOximeterDiagnosticsTests(unittest.TestCase):
    """Protect coverage, completeness, and device-agreement summaries."""

    def test_joint_completeness_is_explicit(self) -> None:
        diagnostics = build_pulse_oximeter_diagnostics(
            [
                _measurement(bpm=70.0, spo2=98.0, bpm_spo2=68.0),
                _measurement(bpm=71.0, spo2=97.0, bpm_spo2=None),
                _measurement(bpm=72.0, spo2=None, bpm_spo2=73.0),
                _measurement(bpm=73.0, spo2=None, bpm_spo2=None),
            ]
        )

        completeness = diagnostics["joint_field_completeness"]
        self.assertEqual(completeness["both_observed"], 1)
        self.assertEqual(completeness["spo2_without_bpm_spo2"], 1)
        self.assertEqual(completeness["bpm_spo2_without_spo2"], 1)
        self.assertEqual(completeness["both_missing"], 1)

    def test_spo2_summary_includes_dispersion(self) -> None:
        diagnostics = build_pulse_oximeter_diagnostics(
            [
                _measurement(bpm=70.0, spo2=96.0, bpm_spo2=70.0),
                _measurement(bpm=72.0, spo2=98.0, bpm_spo2=72.0),
                _measurement(bpm=74.0, spo2=100.0, bpm_spo2=74.0),
            ]
        )

        summary = diagnostics["spo2_percent"]
        self.assertEqual(summary["median"], 98.0)
        self.assertEqual(summary["sample_sd"], 2.0)

    def test_bpm_agreement_reports_bias_and_spread(self) -> None:
        diagnostics = build_pulse_oximeter_diagnostics(
            [
                _measurement(bpm=70.0, spo2=98.0, bpm_spo2=68.0),
                _measurement(bpm=72.0, spo2=98.0, bpm_spo2=72.0),
                _measurement(bpm=74.0, spo2=98.0, bpm_spo2=76.0),
            ]
        )

        agreement = diagnostics["paired_bpm_device_agreement"]
        self.assertEqual(agreement["mean_difference"], 0.0)
        self.assertEqual(agreement["median_difference"], 0.0)
        self.assertEqual(agreement["sample_sd_difference"], 2.0)
        self.assertEqual(agreement["minimum_difference"], -2.0)
        self.assertEqual(agreement["maximum_difference"], 2.0)
        self.assertEqual(agreement["maximum_absolute_difference"], 2.0)

    def test_empty_measurement_sequence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one measurement"):
            build_pulse_oximeter_diagnostics([])


if __name__ == "__main__":
    unittest.main()
