"""Hypothetical-only tests of the future scoring contract.

These tests DO NOT contain, fetch, or register a 2026 election outcome.
All synthetic examples use an explicitly injected post-election clock.
"""

from __future__ import annotations

import json
import unittest
from datetime import date

from forecast_lock import FROZEN_OLS_FORECAST, build_forecast_lock
from score_after_election import score_verified_outcome

OFFICIAL_URL = "https://history.house.gov/Institution/Party-Divisions/Party-Divisions/"


class ScoringGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = build_forecast_lock()

    def _score(self, **changed: object) -> dict[str, object]:
        options: dict[str, object] = {
            "observed_change": -32,  # Arbitrary hypothetical example, not an observation.
            "final_source_url": OFFICIAL_URL,
            "source_retrieved_on": date(2026, 11, 5),
            "confirmed_final": True,
            "evaluated_on": date(2026, 11, 6),
            "manifest": self.lock,
        }
        options.update(changed)
        return score_verified_outcome(**options)

    def test_scoring_is_impossible_before_election_day(self) -> None:
        with self.assertRaisesRegex(ValueError, "before 2026-11-03"):
            self._score(evaluated_on=date(2026, 11, 1))

    def test_final_result_must_be_verifiably_after_election(self) -> None:
        with self.assertRaisesRegex(ValueError, "retrieved after election"):
            self._score(source_retrieved_on=date(2026, 11, 2))
        with self.assertRaisesRegex(ValueError, "retrieved after election"):
            self._score(source_retrieved_on=date(2026, 11, 7))
        with self.assertRaisesRegex(ValueError, "final-result verification"):
            self._score(confirmed_final=False)

    def test_only_defined_official_https_publishers_are_allowed(self) -> None:
        for value in (
            "http://history.house.gov/Institution/Party-Divisions/",
            "https://example.com/results",
            "https://www.fec.gov",
            "https://user@history.house.gov/Institution/Party-Divisions/",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._score(final_source_url=value)

    def test_hypothetical_change_must_be_integer_and_feasible(self) -> None:
        for value in (-221, 216):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._score(observed_change=value)
        for value in (True, -31.25, "-31"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self._score(observed_change=value)

    def test_hypothetical_in_range_result_scores_correctly(self) -> None:
        # The synthetic result is deliberately not stored in the repository.
        report = self._score()
        self.assertEqual(report["observed_president_party_seat_change"], -32)
        self.assertAlmostEqual(report["ols_registered_forecast"], FROZEN_OLS_FORECAST)
        self.assertAlmostEqual(report["ols_signed_error_predicted_minus_observed"],
                               FROZEN_OLS_FORECAST + 32)
        self.assertAlmostEqual(report["ols_absolute_error_house_seats"],
                               abs(FROZEN_OLS_FORECAST + 32))
        self.assertTrue(report["ols_observation_inside_conditional_95_interval"])
        self.assertEqual(len(report["registered_manifest_canonical_sha256"]), 64)
        self.assertEqual(report["official_source_url"], OFFICIAL_URL)

    def test_hypothetical_outside_interval_does_not_rewrite_lock(self) -> None:
        before = json.dumps(self.lock, sort_keys=True)
        report = self._score(observed_change=-80)
        self.assertFalse(report["ols_observation_inside_conditional_95_interval"])
        self.assertFalse(report["observation_inside_pairs_residual_sensitivity_band"])
        self.assertAlmostEqual(report["ols_signed_error_predicted_minus_observed"],
                               FROZEN_OLS_FORECAST + 80)
        self.assertEqual(before, json.dumps(self.lock, sort_keys=True))
        self.assertIsNone(self.lock["target"]["observed_2026_outcome"])


if __name__ == "__main__":
    unittest.main()
