"""Tests that an expanding AP archive never becomes a post-hoc official score."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

STUDY: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))

from analysis import load_inputs  # noqa: E402
from snapshot_history import build_history, dated_snapshots  # noqa: E402


class SnapshotHistoryTests(unittest.TestCase):
    """Verify the dated comparison without altering the registered model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = dated_snapshots()
        _, cls.lock = load_inputs()
        cls.report = build_history(cls.records, cls.lock)

    def test_original_two_distinct_live_reports_are_retained(self) -> None:
        self.assertGreaterEqual(len(self.records), 2)
        self.assertEqual(
            [path.name for path, _ in self.records[:2]],
            ["ap_house_live_2026-11-04.json", "ap_house_live_2026-11-05.json"],
        )
        self.assertTrue(all(snapshot.uncalled > 0 for _, snapshot in self.records))

    def test_november_fifth_source_accounting(self) -> None:
        _, november_fifth = self.records[1]
        self.assertEqual((november_fifth.democratic_won, november_fifth.republican_won), (240, 147))
        self.assertEqual((november_fifth.democratic_leading, november_fifth.republican_leading), (21, 27))
        self.assertEqual(november_fifth.uncalled, 48)
        self.assertEqual(november_fifth.republican_trend, 174)

    def test_scenario_difference_is_signed_as_registered(self) -> None:
        entries = self.report["snapshots"]
        self.assertEqual([s["conditional_republican_seat_change"] for s in entries[:2]], [-48, -46])
        self.assertAlmostEqual(entries[0]["conditional_ols_forecast_minus_scenario"], 15.67711197369437)
        self.assertAlmostEqual(entries[1]["conditional_ols_forecast_minus_scenario"], 13.67711197369437)
        self.assertTrue(all(s["conditional_ols_interval_inclusion"] for s in entries[:2]))
        self.assertFalse(self.report["official_initial_election_outcome_scored"])

    def test_20_more_calls_and_two_seat_scenario_movement(self) -> None:
        self.assertEqual(self.report["movements"][0],
            {
                "from_snapshot": "ap_house_live_2026-11-04.json",
                "to_snapshot": "ap_house_live_2026-11-05.json",
                "additional_called_races": 20,
                "change_in_uncalled_races": -20,
                "change_in_conditional_republican_seats": 2,
                "change_in_conditional_ols_signed_difference": -2.0,
            })

    def test_report_equals_committed_evidence(self) -> None:
        committed = json.loads((STUDY / "results" / "snapshot_history.json").read_text(encoding="utf-8"))
        self.assertEqual(self.report, committed)
        self.assertEqual(self.report["registered_house_seats_defended"], 220)

    def test_misnamed_archive_record_is_rejected(self) -> None:
        original = self.records[1][0].read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "ap_house_live_2026-11-04.json").write_text(
                self.records[0][0].read_text(encoding="utf-8"), encoding="utf-8"
            )
            (folder / "ap_house_live_2026-11-06.json").write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "File name disagrees"):
                dated_snapshots(folder)

    def test_forged_final_result_in_second_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "ap_house_live_2026-11-04.json").write_text(
                self.records[0][0].read_text(encoding="utf-8"), encoding="utf-8"
            )
            second = json.loads(self.records[1][0].read_text(encoding="utf-8"))
            second["observed_republican_seat_change"] = -46
            (folder / "ap_house_live_2026-11-05.json").write_text(
                json.dumps(second), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "No observed seat-change"):
                dated_snapshots(folder)


if __name__ == "__main__":
    unittest.main()
