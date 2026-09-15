"""Day-level containment contract for public pulse-oximeter aggregates."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


class PulseOximeterDailyBloodPressureContainmentTests(unittest.TestCase):
    """Pulse aggregates must remain subsets of BP measurement rows each day."""

    def test_pulse_daily_counts_do_not_exceed_bp_readings(self) -> None:
        pulse_path = DATA_DIR / "pulse_oximeter_daily_snapshot.csv"
        bp_path = DATA_DIR / "analysis_snapshot.csv"
        if not pulse_path.exists():
            return

        with bp_path.open("r", encoding="utf-8", newline="") as handle:
            bp_rows = {
                int(row["day_index"]): row
                for row in csv.DictReader(handle)
            }
        with pulse_path.open("r", encoding="utf-8", newline="") as handle:
            pulse_rows = list(csv.DictReader(handle))

        for pulse_row in pulse_rows:
            day_index = int(pulse_row["day_index"])
            self.assertIn(day_index, bp_rows)

            bp_row = bp_rows[day_index]
            self.assertEqual(
                int(bp_row["observed"]),
                1,
                f"pulse day {day_index} must be an observed BP day",
            )
            n_bp = int(bp_row["n_readings"])
            self.assertGreater(n_bp, 0)

            self.assertLessEqual(
                int(pulse_row["n_spo2_readings"]),
                n_bp,
                f"SpO2 count exceeds BP readings on day {day_index}",
            )
            self.assertLessEqual(
                int(pulse_row["n_bpm_spo2_readings"]),
                n_bp,
                f"bpm_spo2 count exceeds BP readings on day {day_index}",
            )
            self.assertLessEqual(
                int(pulse_row["n_paired_bpm_readings"]),
                n_bp,
                f"paired BPM count exceeds BP readings on day {day_index}",
            )


if __name__ == "__main__":
    unittest.main()
