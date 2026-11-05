"""Synthetic-only tests for the future initial-election source handoff.

The 175-seat example is invented *solely* to exercise consistency checks.
It is not a reported result of the 2026 U.S. House election.
"""

from __future__ import annotations

import copy
import sys
import unittest
from datetime import date
from pathlib import Path
from typing import Any

ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import load_inputs  # noqa: E402
from final_outcome_handoff import (  # noqa: E402
    HISTORIAN_URL,
    PRESIDENCY_URL,
    prepare_handoff,
    validate_source_receipt,
)


class OfficialHandoffTests(unittest.TestCase):
    """Guard the original scorer from unsupported source transcriptions."""

    @classmethod
    def setUpClass(cls) -> None:
        _, cls.manifest = load_inputs()
        cls.reviewed_on = date(2027, 1, 5)

    @staticmethod
    def example() -> dict[str, Any]:
        """Build an unambiguously synthetic 435-seat 120th House example."""

        return {
            "record_kind": "final_2026_initial_house_party_divisions",
            "election_date": "2026-11-03",
            "analyst_confirmed_direct_source_inspection": True,
            "house_historian": {
                "source_url": HISTORIAN_URL,
                "retrieved_on": "2027-01-03",
                "congress": 120,
                "initial_election_party_division": True,
                "published_final": True,
                "democratic_initial_seats": 260,
                "republican_initial_seats": 175,
                "other_initial_seats": 0,
            },
            "american_presidency_project": {
                "source_url": PRESIDENCY_URL,
                "retrieved_on": "2027-01-04",
                "election_year": 2026,
                "house_seats_to_defend": 220,
                "republican_house_seat_change": -45,
                "published_final": True,
            },
        }

    def validate(self, payload: dict[str, Any]) -> Any:
        return validate_source_receipt(payload, self.manifest, reviewed_on=self.reviewed_on)

    def test_synthetic_matching_sources_prepare_an_unscored_command(self) -> None:
        receipt = self.validate(self.example())
        self.assertEqual(receipt.republican_seat_change, -45)
        report = prepare_handoff(receipt, reviewed_on=self.reviewed_on)
        self.assertFalse(report["official_forecast_score_computed"])
        self.assertIn("score_after_election.py", report["registered_scorer_command_not_executed"])
        self.assertIn("--actual-change -45", report["registered_scorer_command_not_executed"])
        self.assertIn("--source-retrieved-on 2027-01-03", report["registered_scorer_command_not_executed"])
        self.assertNotIn("ols_absolute_error_house_seats", report)

    def test_rejects_ap_or_other_source_instead_of_historian(self) -> None:
        payload = self.example()
        payload["house_historian"]["source_url"] = "https://apnews.com/hub/elections"
        with self.assertRaisesRegex(ValueError, "House Historian"):
            self.validate(payload)

    def test_rejects_nonfinal_or_membership_based_historian(self) -> None:
        for field in ("published_final", "initial_election_party_division"):
            with self.subTest(field=field):
                payload = self.example()
                payload["house_historian"][field] = False
                with self.assertRaises(ValueError):
                    self.validate(payload)

    def test_rejects_nonfinal_presidency_row(self) -> None:
        payload = self.example()
        payload["american_presidency_project"]["published_final"] = False
        with self.assertRaisesRegex(ValueError, "final outcome"):
            self.validate(payload)

    def test_rejects_source_disagreement_and_mismatched_defended_seats(self) -> None:
        for key, value in (("republican_house_seat_change", -44), ("house_seats_to_defend", 219)):
            with self.subTest(key=key):
                payload = self.example()
                payload["american_presidency_project"][key] = value
                with self.assertRaises(ValueError):
                    self.validate(payload)

    def test_rejects_invalid_totals_and_other_party_alignment(self) -> None:
        payload = self.example()
        payload["house_historian"]["democratic_initial_seats"] = 259
        with self.assertRaisesRegex(ValueError, "total 435"):
            self.validate(payload)
        payload = self.example()
        payload["house_historian"]["democratic_initial_seats"] = 259
        payload["house_historian"]["other_initial_seats"] = 1
        with self.assertRaisesRegex(ValueError, "caucus alignment"):
            self.validate(payload)

    def test_rejects_boolean_and_string_seat_counts(self) -> None:
        for invalid in (True, "175"):
            with self.subTest(invalid=invalid):
                payload = self.example()
                payload["house_historian"]["republican_initial_seats"] = invalid
                with self.assertRaises(TypeError):
                    self.validate(payload)

    def test_rejects_unverified_or_wrong_election(self) -> None:
        payload = self.example()
        payload["analyst_confirmed_direct_source_inspection"] = False
        with self.assertRaisesRegex(ValueError, "manual verification"):
            self.validate(payload)
        payload = self.example()
        payload["election_date"] = "2024-11-05"
        with self.assertRaisesRegex(ValueError, "election date"):
            self.validate(payload)

    def test_rejects_future_dated_or_pre_election_retrieval(self) -> None:
        payload = self.example()
        payload["house_historian"]["retrieved_on"] = "2026-11-02"
        with self.assertRaisesRegex(ValueError, "post-election"):
            self.validate(payload)
        payload = self.example()
        payload["american_presidency_project"]["retrieved_on"] = "2027-01-06"
        with self.assertRaisesRegex(ValueError, "postdate"):
            self.validate(payload)

    def test_registered_lock_remains_unobserved(self) -> None:
        modified = copy.deepcopy(self.manifest)
        modified["target"]["observed_2026_outcome"] = -45
        with self.assertRaisesRegex(ValueError, "must not contain"):
            validate_source_receipt(self.example(), modified, reviewed_on=self.reviewed_on)
        self.assertIsNone(self.manifest["target"]["observed_2026_outcome"])


if __name__ == "__main__":
    unittest.main()
