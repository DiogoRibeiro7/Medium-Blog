"""Contracts for pulse-oximeter BPM pairing semantics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from blood_pressure_missingness.analyses import pulse_oximeter_longitudinal as analysis

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


class PulseOximeterPairingContractTests(unittest.TestCase):
    """Every pulse-oximeter BPM must have the BP-monitor BPM needed for pairing."""

    def test_loader_rejects_unpaired_bpm_spo2_count(self) -> None:
        """A valid BP row always supplies the comparison BPM for bpm_spo2."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pulse.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "day_index",
                        "n_spo2_readings",
                        "mean_spo2_percent",
                        "n_bpm_spo2_readings",
                        "mean_bpm_spo2",
                        "n_paired_bpm_readings",
                        "mean_bpm_difference",
                    ]
                )
                writer.writerow([0, 2, 98.0, 2, 71.0, 1, 1.0])

            with self.assertRaisesRegex(
                ValueError,
                "n_paired_bpm_readings must equal n_bpm_spo2_readings",
            ):
                analysis.load_pulse_oximeter_days(path)

    def test_committed_artifacts_preserve_pairing_identity(self) -> None:
        """Published daily and overall counts must encode the same pairing rule."""

        daily_path = DATA_DIR / "pulse_oximeter_daily_snapshot.csv"
        diagnostics_path = DATA_DIR / "pulse_oximeter_diagnostics.json"
        if not daily_path.exists() and not diagnostics_path.exists():
            return
        self.assertTrue(daily_path.exists())
        self.assertTrue(diagnostics_path.exists())

        with daily_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertEqual(
                int(row["n_paired_bpm_readings"]),
                int(row["n_bpm_spo2_readings"]),
            )

        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        self.assertEqual(
            int(diagnostics["paired_bpm_device_agreement"]["observed_pairs"]),
            int(diagnostics["coverage"]["bpm_spo2_observed"]),
        )


if __name__ == "__main__":
    unittest.main()
