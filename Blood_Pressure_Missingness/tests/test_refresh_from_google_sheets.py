"""Regression tests for the secret-backed blood-pressure source refresh."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import refresh_from_google_sheets as refresh


class SourceRefreshTests(unittest.TestCase):
    """Protect source parsing and privacy-safe aggregation invariants."""

    def test_numeric_time_rejects_values_outside_one_day(self) -> None:
        """Invalid spreadsheet serial times must fail instead of wrapping."""

        for value in (-0.1, 1.0, 1.25, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                refresh._parse_source_time(value)

    def test_numeric_time_parses_valid_fraction(self) -> None:
        """A valid half-day serial should map to noon exactly."""

        parsed = refresh._parse_source_time(0.5)
        self.assertEqual((parsed.hour, parsed.minute, parsed.second), (12, 0, 0))

    def test_mismatched_diff_is_audited_but_not_published(self) -> None:
        """Published pulse pressure must come from systolic minus diastolic."""

        values = [
            list(refresh.EXPECTED_COLUMNS),
            [
                46206.0,
                0.5,
                120.0,
                80.0,
                999.0,
                70.0,
                "Yes",
                "Yes",
                "Yes",
                None,
                None,
            ],
        ]

        measurements, audit = refresh.parse_measurements(values)
        self.assertEqual(measurements[0].pulse_pressure, 40.0)
        derived = audit["derived_field_check"]
        self.assertIsInstance(derived, dict)
        self.assertEqual(derived["pulse_pressure_mismatches_on_valid_rows"], 1)
        self.assertEqual(
            derived["pulse_pressure_output_policy"],
            "derived_from_systolic_and_diastolic",
        )

        with tempfile.TemporaryDirectory() as directory:
            refresh.refresh(values, Path(directory))
            snapshot = (Path(directory) / "analysis_snapshot.csv").read_text()
        self.assertIn(",40.0,", snapshot)
        self.assertNotIn("999", snapshot)

    def test_non_positive_derived_pulse_pressure_is_rejected(self) -> None:
        """Rows with systolic not greater than diastolic are invalid source data."""

        for systolic, diastolic in ((80.0, 80.0), (79.0, 80.0)):
            values = [
                list(refresh.EXPECTED_COLUMNS),
                [
                    46206.0,
                    0.5,
                    systolic,
                    diastolic,
                    systolic - diastolic,
                    70.0,
                    "Yes",
                    "Yes",
                    "Yes",
                    None,
                    None,
                ],
            ]
            with (
                self.subTest(systolic=systolic, diastolic=diastolic),
                self.assertRaisesRegex(ValueError, "systolic must exceed diastolic"),
            ):
                refresh.parse_measurements(values)


if __name__ == "__main__":
    unittest.main()
