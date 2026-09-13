"""Regression tests for pulse-oximeter source-audit diagnostics."""

from __future__ import annotations

import unittest

from blood_pressure_missingness.data_sources import google_sheets as source


class PulseOximeterAuditTests(unittest.TestCase):
    """Protect aggregate diagnostics for the optional pulse-oximeter fields."""

    @staticmethod
    def _row(
        *,
        day: float,
        bpm: float,
        spo2: float | None,
        bpm_spo2: float | None,
    ) -> list[object | None]:
        """Build one valid source row with optional pulse-oximeter values."""

        return [
            day,
            0.5,
            120.0,
            80.0,
            40.0,
            bpm,
            "Yes",
            "Yes",
            "Yes",
            None,
            None,
            spo2,
            bpm_spo2,
        ]

    def test_audit_tracks_joint_field_completeness(self) -> None:
        """Same-device fields should expose incomplete-pair patterns explicitly."""

        values = [
            list(source.EXPECTED_COLUMNS),
            self._row(day=46206.0, bpm=70.0, spo2=98.0, bpm_spo2=68.0),
            self._row(day=46207.0, bpm=71.0, spo2=97.0, bpm_spo2=None),
            self._row(day=46208.0, bpm=72.0, spo2=None, bpm_spo2=73.0),
            self._row(day=46209.0, bpm=73.0, spo2=None, bpm_spo2=None),
        ]

        _, audit = source.parse_measurements(values)
        pulse_oximeter = audit["pulse_oximeter"]
        completeness = pulse_oximeter["joint_field_completeness"]

        self.assertEqual(completeness["both_observed"], 1)
        self.assertEqual(completeness["spo2_without_bpm_spo2"], 1)
        self.assertEqual(completeness["bpm_spo2_without_spo2"], 1)
        self.assertEqual(completeness["both_missing"], 1)

    def test_spo2_summary_includes_median_and_sample_sd(self) -> None:
        """SpO2 auditing should report central tendency and observed dispersion."""

        values = [
            list(source.EXPECTED_COLUMNS),
            self._row(day=46206.0, bpm=70.0, spo2=96.0, bpm_spo2=70.0),
            self._row(day=46207.0, bpm=72.0, spo2=98.0, bpm_spo2=72.0),
            self._row(day=46208.0, bpm=74.0, spo2=100.0, bpm_spo2=74.0),
        ]

        _, audit = source.parse_measurements(values)
        summary = audit["pulse_oximeter"]["spo2_percent"]

        self.assertEqual(summary["observed"], 3)
        self.assertEqual(summary["median"], 98.0)
        self.assertEqual(summary["sample_sd"], 2.0)

    def test_bpm_agreement_reports_difference_spread(self) -> None:
        """Paired BPM audit should summarize bias and descriptive difference spread."""

        values = [
            list(source.EXPECTED_COLUMNS),
            self._row(day=46206.0, bpm=70.0, spo2=98.0, bpm_spo2=68.0),
            self._row(day=46207.0, bpm=72.0, spo2=98.0, bpm_spo2=72.0),
            self._row(day=46208.0, bpm=74.0, spo2=98.0, bpm_spo2=76.0),
        ]

        _, audit = source.parse_measurements(values)
        agreement = audit["pulse_oximeter"]["paired_bpm_device_agreement"]

        self.assertEqual(agreement["observed_pairs"], 3)
        self.assertEqual(agreement["mean_bpm_bp_device"], 72.0)
        self.assertEqual(agreement["mean_bpm_spo2_device"], 72.0)
        self.assertEqual(agreement["mean_difference"], 0.0)
        self.assertEqual(agreement["median_difference"], 0.0)
        self.assertEqual(agreement["sample_sd_difference"], 2.0)
        self.assertEqual(agreement["minimum_difference"], -2.0)
        self.assertEqual(agreement["maximum_difference"], 2.0)
        self.assertEqual(agreement["maximum_absolute_difference"], 2.0)

    def test_single_pair_has_no_sample_sd(self) -> None:
        """Sample dispersion must remain undefined when only one pair is available."""

        values = [
            list(source.EXPECTED_COLUMNS),
            self._row(day=46206.0, bpm=70.0, spo2=98.0, bpm_spo2=69.0),
        ]

        _, audit = source.parse_measurements(values)
        agreement = audit["pulse_oximeter"]["paired_bpm_device_agreement"]
        summary = audit["pulse_oximeter"]["spo2_percent"]

        self.assertIsNone(agreement["sample_sd_difference"])
        self.assertIsNone(summary["sample_sd"])


if __name__ == "__main__":
    unittest.main()
