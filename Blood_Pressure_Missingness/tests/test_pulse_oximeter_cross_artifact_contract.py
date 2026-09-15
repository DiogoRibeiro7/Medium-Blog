"""Cross-artifact consistency contract for public pulse-oximeter outputs."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

PULSE_PATHS = (
    DATA_DIR / "pulse_oximeter_diagnostics.json",
    DATA_DIR / "pulse_oximeter_daily_snapshot.csv",
    DATA_DIR / "pulse_oximeter_longitudinal_diagnostics.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path.name} must contain one JSON object")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _weighted_daily_mean(
    rows: list[dict[str, str]],
    count_field: str,
    mean_field: str,
) -> float:
    weighted_sum = 0.0
    total_count = 0
    for row in rows:
        count = int(row[count_field])
        if count == 0:
            continue
        weighted_sum += count * float(row[mean_field])
        total_count += count
    if total_count == 0:
        raise AssertionError(f"{count_field} has no observed values")
    return weighted_sum / total_count


class PulseOximeterCrossArtifactContractTests(unittest.TestCase):
    """Require all committed pulse aggregates to reconcile numerically."""

    def _load_committed_artifacts(
        self,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, str]],
        dict[str, Any],
        dict[str, Any],
        list[dict[str, str]],
    ] | None:
        present = [path.exists() for path in PULSE_PATHS]
        if not any(present):
            return None
        self.assertTrue(all(present), "Pulse artifacts must be committed atomically.")
        return (
            _load_json(PULSE_PATHS[0]),
            _load_csv_rows(PULSE_PATHS[1]),
            _load_json(PULSE_PATHS[2]),
            _load_json(DATA_DIR / "source_audit.json"),
            _load_csv_rows(DATA_DIR / "analysis_snapshot.csv"),
        )

    def test_committed_pulse_artifacts_reconcile_counts(self) -> None:
        artifacts = self._load_committed_artifacts()
        if artifacts is None:
            return
        diagnostics, daily_rows, longitudinal, source_audit, bp_rows = artifacts

        diagnostic_coverage = diagnostics["coverage"]
        joint = diagnostics["joint_field_completeness"]
        agreement = diagnostics["paired_bpm_device_agreement"]
        audit_pulse = source_audit["pulse_oximeter"]
        longitudinal_coverage = longitudinal["coverage"]

        n_measurements = int(diagnostics["n_measurements"])
        spo2_total = sum(int(row["n_spo2_readings"]) for row in daily_rows)
        bpm_spo2_total = sum(
            int(row["n_bpm_spo2_readings"]) for row in daily_rows
        )
        paired_total = sum(int(row["n_paired_bpm_readings"]) for row in daily_rows)

        self.assertEqual(n_measurements, int(source_audit["valid_measurements"]))
        self.assertEqual(sum(int(joint[name]) for name in joint), n_measurements)
        self.assertEqual(
            int(joint["both_observed"]) + int(joint["spo2_without_bpm_spo2"]),
            int(diagnostic_coverage["spo2_observed"]),
        )
        self.assertEqual(
            int(joint["both_observed"]) + int(joint["bpm_spo2_without_spo2"]),
            int(diagnostic_coverage["bpm_spo2_observed"]),
        )

        self.assertEqual(spo2_total, int(diagnostic_coverage["spo2_observed"]))
        self.assertEqual(spo2_total, int(diagnostics["spo2_percent"]["n"]))
        self.assertEqual(spo2_total, int(audit_pulse["spo2_percent"]["observed"]))
        self.assertEqual(spo2_total, int(longitudinal_coverage["total_spo2_readings"]))

        self.assertEqual(bpm_spo2_total, int(diagnostic_coverage["bpm_spo2_observed"]))
        self.assertEqual(bpm_spo2_total, int(diagnostics["bpm_spo2"]["n"]))
        self.assertEqual(bpm_spo2_total, int(audit_pulse["bpm_spo2"]["observed"]))
        self.assertEqual(
            bpm_spo2_total,
            int(longitudinal_coverage["total_bpm_spo2_readings"]),
        )

        self.assertEqual(paired_total, int(agreement["observed_pairs"]))
        self.assertEqual(
            paired_total,
            int(audit_pulse["paired_bpm_device_agreement"]["observed_pairs"]),
        )
        self.assertEqual(
            paired_total,
            int(longitudinal_coverage["total_paired_bpm_readings"]),
        )

        self.assertEqual(
            len(daily_rows),
            int(longitudinal_coverage["pulse_oximeter_observed_days"]),
        )
        if daily_rows:
            self.assertEqual(
                int(daily_rows[0]["day_index"]),
                int(longitudinal_coverage["first_pulse_oximeter_day_index"]),
            )
            self.assertEqual(
                int(daily_rows[-1]["day_index"]),
                int(longitudinal_coverage["last_pulse_oximeter_day_index"]),
            )

        n_bp_observed = sum(int(row["observed"]) for row in bp_rows)
        self.assertEqual(
            len(bp_rows),
            int(longitudinal_coverage["blood_pressure_calendar_days"]),
        )
        self.assertEqual(
            n_bp_observed,
            int(longitudinal_coverage["blood_pressure_observed_days"]),
        )

        self.assertAlmostEqual(
            float(diagnostic_coverage["spo2_coverage_rate"]),
            spo2_total / n_measurements,
            places=8,
        )
        self.assertAlmostEqual(
            float(diagnostic_coverage["bpm_spo2_coverage_rate"]),
            bpm_spo2_total / n_measurements,
            places=8,
        )
        self.assertAlmostEqual(
            float(longitudinal_coverage["pulse_oximeter_coverage_of_full_calendar"]),
            len(daily_rows) / len(bp_rows),
            places=8,
        )
        self.assertAlmostEqual(
            float(longitudinal_coverage["pulse_oximeter_coverage_of_bp_observed_days"]),
            len(daily_rows) / n_bp_observed,
            places=8,
        )

    def test_committed_pulse_artifacts_reconcile_weighted_aggregates(self) -> None:
        artifacts = self._load_committed_artifacts()
        if artifacts is None:
            return
        diagnostics, daily_rows, longitudinal, source_audit, _ = artifacts

        spo2_mean = _weighted_daily_mean(
            daily_rows,
            "n_spo2_readings",
            "mean_spo2_percent",
        )
        bpm_spo2_mean = _weighted_daily_mean(
            daily_rows,
            "n_bpm_spo2_readings",
            "mean_bpm_spo2",
        )
        paired_difference_mean = _weighted_daily_mean(
            daily_rows,
            "n_paired_bpm_readings",
            "mean_bpm_difference",
        )

        self.assertAlmostEqual(
            spo2_mean,
            float(diagnostics["spo2_percent"]["mean"]),
            places=7,
        )
        self.assertAlmostEqual(
            spo2_mean,
            float(source_audit["pulse_oximeter"]["spo2_percent"]["mean"]),
            places=7,
        )
        self.assertAlmostEqual(
            bpm_spo2_mean,
            float(diagnostics["bpm_spo2"]["mean"]),
            places=7,
        )
        self.assertAlmostEqual(
            bpm_spo2_mean,
            float(source_audit["pulse_oximeter"]["bpm_spo2"]["mean"]),
            places=7,
        )
        self.assertAlmostEqual(
            paired_difference_mean,
            float(diagnostics["paired_bpm_device_agreement"]["mean_difference"]),
            places=7,
        )
        self.assertAlmostEqual(
            paired_difference_mean,
            float(
                source_audit["pulse_oximeter"]["paired_bpm_device_agreement"]
                ["mean_difference"]
            ),
            places=7,
        )

        coverage = longitudinal["coverage"]
        if daily_rows:
            first_day = int(daily_rows[0]["day_index"])
            last_day = int(daily_rows[-1]["day_index"])
            span_days = last_day - first_day + 1
            self.assertEqual(span_days, int(coverage["pulse_oximeter_span_days"]))
            self.assertAlmostEqual(
                float(coverage["pulse_oximeter_coverage_within_its_observed_span"]),
                len(daily_rows) / span_days,
                places=8,
            )


if __name__ == "__main__":
    unittest.main()
