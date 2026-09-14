"""Deterministic rehearsal of the pulse-oximeter refresh data path."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from blood_pressure_missingness.analyses import pulse_oximeter as pulse
from blood_pressure_missingness.analyses import pulse_oximeter_longitudinal as longitudinal
from blood_pressure_missingness.data_sources import google_sheets as source


class PulseOximeterRefreshRehearsalTests(unittest.TestCase):
    """Exercise the generated artifact family without private credentials."""

    @staticmethod
    def _synthetic_source() -> list[list[object]]:
        """Return a five-day private-source analogue with pulse-oximeter coverage."""

        rows: list[list[object]] = [list(source.EXPECTED_COLUMNS)]
        source_values = [
            ("2026-09-01", 120.0, 80.0, 70.0, 97.0, 69.0),
            ("2026-09-02", 121.0, 81.0, 72.0, 98.0, 73.0),
            ("2026-09-03", 119.0, 79.0, 71.0, 96.0, 69.0),
            ("2026-09-04", 118.0, 78.0, 73.0, 99.0, 73.0),
            ("2026-09-05", 117.0, 77.0, 72.0, 97.0, 74.0),
        ]
        for day, systolic, diastolic, bpm, spo2, bpm_spo2 in source_values:
            rows.append(
                [
                    day,
                    "08:00",
                    systolic,
                    diastolic,
                    systolic - diastolic,
                    bpm,
                    "",
                    "",
                    "",
                    "",
                    "",
                    spo2,
                    bpm_spo2,
                ]
            )
        return rows

    def test_full_generated_artifact_path_is_coherent(self) -> None:
        """All public pulse artifacts should be generatable from one source parse."""

        values = self._synthetic_source()

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"

            source.refresh(values, data_dir)
            measurements, _ = source.parse_measurements(values)

            diagnostics = pulse.build_pulse_oximeter_diagnostics(measurements)
            daily_snapshot = pulse.build_daily_snapshot(measurements)

            diagnostics_path = data_dir / "pulse_oximeter_diagnostics.json"
            daily_path = data_dir / "pulse_oximeter_daily_snapshot.csv"
            longitudinal_path = data_dir / "pulse_oximeter_longitudinal_diagnostics.json"

            pulse.write_diagnostics(diagnostics_path, diagnostics)
            pulse.write_daily_snapshot(daily_path, daily_snapshot)

            pulse_days = longitudinal.load_pulse_oximeter_days(daily_path)
            bp_calendar = longitudinal.load_blood_pressure_calendar(
                data_dir / "analysis_snapshot.csv"
            )
            longitudinal_results = longitudinal.build_longitudinal_diagnostics(
                pulse_days,
                bp_calendar,
            )
            longitudinal.write_results(longitudinal_path, longitudinal_results)

            self.assertTrue(diagnostics_path.exists())
            self.assertTrue(daily_path.exists())
            self.assertTrue(longitudinal_path.exists())

            diagnostics_payload = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
            longitudinal_payload = json.loads(
                longitudinal_path.read_text(encoding="utf-8")
            )
            with daily_path.open(newline="", encoding="utf-8") as handle:
                daily_rows = list(csv.DictReader(handle))

        self.assertEqual(diagnostics_payload["n_measurements"], 5)
        self.assertEqual(diagnostics_payload["coverage"]["spo2_observed"], 5)
        self.assertEqual(
            diagnostics_payload["paired_bpm_device_agreement"]["observed_pairs"],
            5,
        )

        self.assertEqual([int(row["day_index"]) for row in daily_rows], [0, 1, 2, 3, 4])
        self.assertEqual(
            tuple(daily_rows[0]),
            pulse.DAILY_SNAPSHOT_COLUMNS,
        )
        self.assertNotIn("date", daily_rows[0])
        self.assertNotIn("timestamp", daily_rows[0])

        coverage = longitudinal_payload["coverage"]
        self.assertEqual(coverage["blood_pressure_calendar_days"], 5)
        self.assertEqual(coverage["pulse_oximeter_observed_days"], 5)
        self.assertEqual(coverage["total_spo2_readings"], 5)
        self.assertEqual(coverage["total_paired_bpm_readings"], 5)
        self.assertEqual(coverage["pulse_oximeter_coverage_of_full_calendar"], 1.0)
        self.assertEqual(
            coverage["pulse_oximeter_coverage_within_its_observed_span"],
            1.0,
        )

        spo2_trend = longitudinal_payload["spo2_time_association"]
        bpm_difference_trend = longitudinal_payload[
            "bpm_device_difference_time_association"
        ]
        self.assertTrue(spo2_trend["estimable"])
        self.assertTrue(bpm_difference_trend["estimable"])
        self.assertIsNotNone(spo2_trend["equal_day"])
        self.assertIsNotNone(spo2_trend["reading_count_weighted"])
        self.assertIsNotNone(bpm_difference_trend["equal_day"])
        self.assertIsNotNone(bpm_difference_trend["reading_count_weighted"])

        interpretation = longitudinal_payload["interpretation"]
        self.assertFalse(interpretation["clinical_thresholds_applied"])
        self.assertFalse(interpretation["causal_interpretation"])
        self.assertFalse(interpretation["missingness_mechanism_identified"])
        self.assertTrue(
            interpretation["coverage_must_be_considered_with_time_associations"]
        )


if __name__ == "__main__":
    unittest.main()
