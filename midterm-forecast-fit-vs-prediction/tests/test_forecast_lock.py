"""Tests for the content-addressed pre-election forecast registry."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from analysis import ROOT
from forecast_lock import (
    ELECTION_DATE, FINGERPRINT_PATHS, FORECAST_AS_OF, FROZEN_CANDIDATES,
    FROZEN_OLS_95, FROZEN_OLS_FORECAST, REGISTERED_ON, SOURCE_COMMIT,
    build_forecast_lock, normalized_sha256, verify_lock,
)


class ForecastLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_forecast_lock()

    def test_committed_manifest_matches_ex_ante_evidence_exactly(self) -> None:
        committed = json.loads((ROOT / "results" / "forecast_lock.json").read_text())
        self.assertEqual(committed, self.manifest)
        verify_lock(committed, current_root=ROOT)

    def test_registration_and_information_cutoff_are_distinct(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["forecast_information_as_of"], FORECAST_AS_OF)
        self.assertEqual(manifest["registration_date"], REGISTERED_ON)
        self.assertEqual(manifest["general_election_date"], ELECTION_DATE)
        self.assertEqual(manifest["source_commit"], SOURCE_COMMIT)
        self.assertLess(FORECAST_AS_OF, REGISTERED_ON)
        self.assertLess(REGISTERED_ON, ELECTION_DATE)
        self.assertIsNone(manifest["target"]["observed_2026_outcome"])

    def test_frozen_inputs_and_models_match_original_numbers(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["historical_sample"]["observed_elections"], 18)
        self.assertEqual(manifest["historical_sample"]["unsupported_missing_ballot_year"], 1986)
        self.assertEqual(manifest["scenario"]["president_party_seats_defended"], 220)
        self.assertEqual(manifest["scenario"]["president_party_signed_generic_margin_points"], -8)
        self.assertEqual(manifest["primary_forecast"]["predicted_seat_change"],
                         FROZEN_OLS_FORECAST)
        self.assertEqual(manifest["primary_forecast"]["model_based_95_prediction_interval"],
                         list(FROZEN_OLS_95))
        self.assertEqual(manifest["alternative_conditional_point_forecasts"],
                         FROZEN_CANDIDATES)

    def test_sha256_newline_normalization_does_not_depend_on_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example.txt"
            path.write_bytes(b"sample,year\r\nA,2022\r\n")
            first = normalized_sha256(path)
            path.write_bytes(b"sample,year\nA,2022\n")
            self.assertEqual(first, normalized_sha256(path))

    def test_tampering_with_frozen_fields_is_rejected(self) -> None:
        for path, changed in (
            (("source_commit",), "wrong"),
            (("registration_date",), "2026-10-31"),
            (("target", "observed_2026_outcome"), -32),
            (("scenario", "president_party_seats_defended"), 218),
            (("scenario", "president_party_signed_generic_margin_points"), -6.0),
            (("historical_sample", "observed_elections"), 19),
            (("primary_forecast", "predicted_seat_change"), -20.0),
            (("primary_forecast", "in_sample_r_squared"), 0.97),
            (("bootstrap_95_sensitivity_interval",), [-1.0, 1.0]),
            (("alternative_conditional_point_forecasts", "student_t"), -20.0),
        ):
            with self.subTest(path=path):
                corrupted = deepcopy(self.manifest)
                location = corrupted
                for key in path[:-1]:
                    location = location[key]
                location[path[-1]] = changed
                with self.assertRaises(ValueError):
                    verify_lock(corrupted)

    def test_source_byte_change_is_detected_without_editing_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            replica = Path(temp)
            for name in FINGERPRINT_PATHS:
                target = replica / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            verify_lock(self.manifest, current_root=replica)
            source = replica / "data" / "2026_scenario.csv"
            source.write_bytes(source.read_bytes() + b"# post-hoc edit\n")
            with self.assertRaisesRegex(ValueError, "Source differs"):
                verify_lock(self.manifest, current_root=replica)


if __name__ == "__main__":
    unittest.main()
