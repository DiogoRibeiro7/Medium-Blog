"""Regression and guard tests for the frozen 2026 AP snapshot audit."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

STUDY: Path = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("post_election_audit", STUDY / "analysis.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load post-election analysis module")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class SnapshotTests(unittest.TestCase):
    """Keep unofficial AP counts separate from initial-election final outcomes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot, cls.manifest = audit.load_inputs()

    def test_house_accounting_reconciles(self) -> None:
        self.assertEqual(self.snapshot.seats, 435)
        self.assertEqual(self.snapshot.uncalled, 68)
        self.assertEqual(self.snapshot.republican_trend, 172)
        self.assertEqual(self.snapshot.democratic_won, 228)

    def test_live_leads_are_not_called_winners(self) -> None:
        self.assertEqual(self.snapshot.republican_won, 139)
        self.assertEqual(self.snapshot.republican_leading, 33)
        self.assertNotEqual(self.snapshot.republican_won, self.snapshot.republican_trend)

    def test_pre_election_lock_preserved(self) -> None:
        self.assertEqual(self.manifest["registration_date"], "2026-11-01")
        self.assertEqual(self.manifest["forecast_information_as_of"], "2026-10-31")
        self.assertAlmostEqual(self.manifest["primary_forecast"]["predicted_seat_change"], -32.32288802630563)
        self.assertIsNone(self.manifest["target"]["observed_2026_outcome"])

    def test_signed_difference_uses_registered_prediction_minus_observation(self) -> None:
        report = audit.diagnostics(self.snapshot, self.manifest)
        self.assertEqual(report["illustrative_change_if_all_leads_hold"], -48)
        self.assertAlmostEqual(
            report["candidate_scenario_differences"]["ols"]["if_leads_hold_predicted_minus_conditional_change"],
            15.67711197369437,
        )

    def test_integer_boundary_is_172_to_203(self) -> None:
        report = audit.diagnostics(self.snapshot, self.manifest)
        lo, hi = report["integer_republican_seats_in_locked_ols_interval"]
        self.assertEqual((lo, hi), (172, 203))
        bounds = report["locked_ols_95_interval"]
        self.assertTrue(bounds[0] <= 172 - 220 <= bounds[1])
        self.assertFalse(bounds[0] <= 171 - 220 <= bounds[1])

    def test_bootstrap_band_is_not_interchangeable_with_ols_interval(self) -> None:
        report = audit.diagnostics(self.snapshot, self.manifest)
        self.assertTrue(report["if_leads_hold_interval_inclusion"])
        self.assertFalse(report["if_leads_hold_bootstrap_sensitivity_band_inclusion"])

    def test_rejects_snapshot_falsely_declared_final(self) -> None:
        value = json.loads(audit.SNAPSHOT.read_text(encoding="utf-8"))
        value["official_initial_election_party_division_available"] = True
        with self.assertRaises(ValueError):
            audit.HouseSnapshot.from_dict(value)

    def test_rejects_observed_target_inside_live_snapshot(self) -> None:
        value = json.loads(audit.SNAPSHOT.read_text(encoding="utf-8"))
        value["observed_republican_seat_change"] = -48
        with self.assertRaises(ValueError):
            audit.HouseSnapshot.from_dict(value)

    def test_rejects_false_live_accounting(self) -> None:
        value = json.loads(audit.SNAPSHOT.read_text(encoding="utf-8"))
        value["house"]["republican_leading_uncalled"] += 1
        with self.assertRaises(ValueError):
            audit.HouseSnapshot.from_dict(value)

    def test_rejects_boolean_as_seat_count(self) -> None:
        value = json.loads(audit.SNAPSHOT.read_text(encoding="utf-8"))
        value["house"]["republican_won"] = True
        with self.assertRaises(TypeError):
            audit.HouseSnapshot.from_dict(value)

    def test_generated_report_matches_committed_snapshot(self) -> None:
        expected = json.loads((STUDY / "results" / "provisional_diagnostics.json").read_text(encoding="utf-8"))
        actual = audit.diagnostics(self.snapshot, self.manifest)
        self.assertEqual(actual, expected)
        self.assertFalse(actual["final_outcome_recorded"])

    def test_eight_original_png_figures_can_be_reproduced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "figures"
            audit.make_figures(self.snapshot, self.manifest, audit.diagnostics(self.snapshot, self.manifest), target)
            figures = sorted(target.glob("fig*.png"))
            self.assertEqual(len(figures), 8)
            self.assertTrue(all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in figures))


if __name__ == "__main__":
    unittest.main()
