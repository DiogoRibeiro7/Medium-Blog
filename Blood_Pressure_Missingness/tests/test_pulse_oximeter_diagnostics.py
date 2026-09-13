"""Regression tests for aggregate pulse-oximeter diagnostics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from blood_pressure_missingness.analyses.pulse_oximeter import (
    DAILY_SNAPSHOT_COLUMNS,
    build_daily_snapshot,
    build_pulse_oximeter_diagnostics,
    write_daily_snapshot,
    write_diagnostics,
)
from blood_pressure_missingness.data_sources.google_sheets import Measurement


def _measurement(
    *,
    bpm: float,
    spo2: float | None,
    bpm_spo2: float | None,
    day: date = date(2026, 9, 1),
) -> Measurement:
    """Build a minimal validated measurement for diagnostic tests."""

    return Measurement(
        day=day,
        timestamp=datetime.combine(day, datetime.min.time()).replace(hour=12),
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

    def test_daily_snapshot_uses_same_relative_day_origin(self) -> None:
        measurements = [
            _measurement(
                bpm=70.0,
                spo2=None,
                bpm_spo2=None,
                day=date(2026, 9, 1),
            ),
            _measurement(
                bpm=72.0,
                spo2=98.0,
                bpm_spo2=71.0,
                day=date(2026, 9, 3),
            ),
            _measurement(
                bpm=74.0,
                spo2=96.0,
                bpm_spo2=76.0,
                day=date(2026, 9, 3),
            ),
        ]

        snapshot = build_daily_snapshot(measurements)

        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]["day_index"], 2)
        self.assertEqual(snapshot[0]["n_spo2_readings"], 2)
        self.assertEqual(snapshot[0]["mean_spo2_percent"], 97.0)
        self.assertEqual(snapshot[0]["n_paired_bpm_readings"], 2)
        self.assertEqual(snapshot[0]["mean_bpm_difference"], -0.5)

    def test_daily_snapshot_writer_has_fixed_date_free_schema(self) -> None:
        snapshot = build_daily_snapshot(
            [_measurement(bpm=70.0, spo2=98.0, bpm_spo2=69.0)]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pulse_oximeter_daily_snapshot.csv"
            write_daily_snapshot(path, snapshot)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = tuple(rows[0])

        self.assertEqual(fieldnames, DAILY_SNAPSHOT_COLUMNS)
        self.assertEqual(rows[0]["day_index"], "0")
        self.assertNotIn("date", fieldnames)
        self.assertNotIn("timestamp", fieldnames)

    def test_writer_persists_only_aggregate_diagnostics(self) -> None:
        diagnostics = build_pulse_oximeter_diagnostics(
            [_measurement(bpm=70.0, spo2=98.0, bpm_spo2=69.0)]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pulse_oximeter_diagnostics.json"
            write_diagnostics(path, diagnostics)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload, diagnostics)
        self.assertFalse(payload["interpretation"]["row_level_values_persisted"])
        self.assertFalse(payload["interpretation"]["clinical_thresholds_applied"])
        self.assertNotIn("timestamp", payload)
        self.assertNotIn("date", payload)

    def test_empty_measurement_sequence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one measurement"):
            build_pulse_oximeter_diagnostics([])
        with self.assertRaisesRegex(ValueError, "At least one measurement"):
            build_daily_snapshot([])


if __name__ == "__main__":
    unittest.main()
