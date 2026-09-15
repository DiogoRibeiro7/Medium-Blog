"""Public contract for optional pulse-oximeter aggregate artifacts."""

from __future__ import annotations

import csv
import json
import re
import unittest
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

PULSE_ARTIFACTS = {
    "pulse_oximeter_diagnostics.json",
    "pulse_oximeter_daily_snapshot.csv",
    "pulse_oximeter_longitudinal_diagnostics.json",
    "pulse_oximeter_observation_window.json",
}

DAILY_COLUMNS = [
    "day_index",
    "n_spo2_readings",
    "mean_spo2_percent",
    "n_bpm_spo2_readings",
    "mean_bpm_spo2",
    "n_paired_bpm_readings",
    "mean_bpm_difference",
]

SUMMARY_FIELDS = {
    "n",
    "mean",
    "median",
    "sample_sd",
    "minimum",
    "maximum",
}

AGREEMENT_FIELDS = {
    "observed_pairs",
    "difference_definition",
    "mean_bpm_bp_device",
    "mean_bpm_spo2_device",
    "mean_difference",
    "median_difference",
    "sample_sd_difference",
    "mean_absolute_difference",
    "rmse",
    "minimum_difference",
    "maximum_difference",
    "maximum_absolute_difference",
}

TREND_FIELDS = {
    "slope_per_30_days",
    "ci95_low_per_30_days",
    "ci95_high_per_30_days",
    "p_value",
}

TIME_ASSOCIATION_FIELDS = {
    "estimable",
    "n_days",
    "minimum_days_required",
    "reason",
    "equal_day",
    "reading_count_weighted",
}

ISO_CALENDAR_DATE = re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{2}-\d{2}(?!\d)")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path.name} must contain one JSON object")
    return payload


def _assert_keys(
    test: unittest.TestCase,
    value: Any,
    expected: set[str],
) -> None:
    test.assertIsInstance(value, dict)
    test.assertEqual(set(value), expected)


class PulseOximeterPublicContractTests(unittest.TestCase):
    """Freeze the optional pulse-oximeter publication boundary."""

    def _present_artifacts(self) -> set[str]:
        return {name for name in PULSE_ARTIFACTS if (DATA_DIR / name).exists()}

    def test_artifact_family_is_atomic_when_refresh_outputs_exist(self) -> None:
        present = self._present_artifacts()
        self.assertIn(
            len(present),
            (0, len(PULSE_ARTIFACTS)),
            "Pulse-oximeter refresh artifacts must be absent together or committed together.",
        )

    def test_daily_snapshot_schema_and_privacy_when_committed(self) -> None:
        path = DATA_DIR / "pulse_oximeter_daily_snapshot.csv"
        if not path.exists():
            return

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, DAILY_COLUMNS)
            previous_day_index = -1
            for row_number, row in enumerate(reader, start=2):
                day_index = int(row["day_index"])
                self.assertGreaterEqual(day_index, 0)
                self.assertGreater(
                    day_index,
                    previous_day_index,
                    "Pulse-oximeter day_index must be strictly increasing.",
                )
                previous_day_index = day_index

                for field, value in row.items():
                    if value:
                        self.assertIsNone(
                            ISO_CALENDAR_DATE.search(value),
                            f"calendar date leaked in {path.name}:{row_number}:{field}",
                        )

                n_spo2 = int(row["n_spo2_readings"])
                n_bpm_spo2 = int(row["n_bpm_spo2_readings"])
                n_paired = int(row["n_paired_bpm_readings"])
                self.assertGreaterEqual(n_spo2, 0)
                self.assertGreaterEqual(n_bpm_spo2, 0)
                self.assertGreaterEqual(n_paired, 0)
                self.assertLessEqual(n_paired, n_bpm_spo2)
                self.assertGreater(n_spo2 + n_bpm_spo2, 0)

                self.assertEqual(bool(row["mean_spo2_percent"]), n_spo2 > 0)
                self.assertEqual(bool(row["mean_bpm_spo2"]), n_bpm_spo2 > 0)
                self.assertEqual(bool(row["mean_bpm_difference"]), n_paired > 0)

    def test_diagnostics_schema_when_committed(self) -> None:
        path = DATA_DIR / "pulse_oximeter_diagnostics.json"
        if not path.exists():
            return

        payload = _load_json(path)
        _assert_keys(
            self,
            payload,
            {
                "n_measurements",
                "coverage",
                "joint_field_completeness",
                "spo2_percent",
                "bpm_spo2",
                "paired_bpm_device_agreement",
                "interpretation",
            },
        )
        _assert_keys(
            self,
            payload["coverage"],
            {
                "spo2_observed",
                "spo2_coverage_rate",
                "bpm_spo2_observed",
                "bpm_spo2_coverage_rate",
            },
        )
        _assert_keys(
            self,
            payload["joint_field_completeness"],
            {
                "both_observed",
                "spo2_without_bpm_spo2",
                "bpm_spo2_without_spo2",
                "both_missing",
            },
        )
        _assert_keys(self, payload["spo2_percent"], SUMMARY_FIELDS)
        _assert_keys(self, payload["bpm_spo2"], SUMMARY_FIELDS)
        _assert_keys(
            self,
            payload["paired_bpm_device_agreement"],
            AGREEMENT_FIELDS,
        )
        _assert_keys(
            self,
            payload["interpretation"],
            {
                "scope",
                "clinical_thresholds_applied",
                "row_level_values_persisted",
            },
        )
        self.assertFalse(payload["interpretation"]["clinical_thresholds_applied"])
        self.assertFalse(payload["interpretation"]["row_level_values_persisted"])

    def test_longitudinal_schema_when_committed(self) -> None:
        path = DATA_DIR / "pulse_oximeter_longitudinal_diagnostics.json"
        if not path.exists():
            return

        payload = _load_json(path)
        _assert_keys(
            self,
            payload,
            {
                "coverage",
                "spo2_time_association",
                "bpm_device_difference_time_association",
                "interpretation",
            },
        )
        _assert_keys(
            self,
            payload["coverage"],
            {
                "blood_pressure_calendar_days",
                "blood_pressure_observed_days",
                "pulse_oximeter_observed_days",
                "first_pulse_oximeter_day_index",
                "last_pulse_oximeter_day_index",
                "pulse_oximeter_span_days",
                "pulse_oximeter_coverage_of_full_calendar",
                "pulse_oximeter_coverage_of_bp_observed_days",
                "pulse_oximeter_coverage_within_its_observed_span",
                "total_spo2_readings",
                "total_bpm_spo2_readings",
                "total_paired_bpm_readings",
            },
        )

        for name in (
            "spo2_time_association",
            "bpm_device_difference_time_association",
        ):
            association = payload[name]
            _assert_keys(self, association, TIME_ASSOCIATION_FIELDS)
            if association["estimable"]:
                self.assertIsNone(association["reason"])
                _assert_keys(self, association["equal_day"], TREND_FIELDS)
                _assert_keys(
                    self,
                    association["reading_count_weighted"],
                    TREND_FIELDS,
                )
            else:
                self.assertIsNotNone(association["reason"])
                self.assertIsNone(association["equal_day"])
                self.assertIsNone(association["reading_count_weighted"])

        _assert_keys(
            self,
            payload["interpretation"],
            {
                "scope",
                "coverage_must_be_considered_with_time_associations",
                "missingness_mechanism_identified",
                "clinical_thresholds_applied",
                "causal_interpretation",
            },
        )
        interpretation = payload["interpretation"]
        self.assertTrue(
            interpretation["coverage_must_be_considered_with_time_associations"]
        )
        self.assertFalse(interpretation["missingness_mechanism_identified"])
        self.assertFalse(interpretation["clinical_thresholds_applied"])
        self.assertFalse(interpretation["causal_interpretation"])


if __name__ == "__main__":
    unittest.main()
