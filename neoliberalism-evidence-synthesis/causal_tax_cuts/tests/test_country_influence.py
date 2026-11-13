"""Synthetic tests of leave-one-country-out influence, not statistical inference."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from causal_tax_cuts.country_influence import (
    estimate_country_influence,
    save_country_influence,
)
from causal_tax_cuts.tests.test_event_study import panel_with_known_effects


class CountryInfluenceTests(unittest.TestCase):
    """Verify refitting, failures, sign changes and honest output metadata."""

    def test_recompute_after_dropping_each_treated_country(self) -> None:
        deleted, summaries, audit = estimate_country_influence(
            panel_with_known_effects(), lags=5, max_horizon=3, min_events=1
        )
        self.assertEqual(audit.n_countries, 5)
        self.assertEqual(audit.n_full_matched_events, 2)
        self.assertEqual(audit.n_failed_country_deletions, 0)
        self.assertEqual(len(deleted), 5 * 8)

        after_three = [row for row in deleted if row.event_time == 3]
        by_country = {row.excluded_country: row for row in after_three}
        self.assertAlmostEqual(by_country["AAA"].mean_contrast or 0, 3.0)
        self.assertAlmostEqual(by_country["BBB"].mean_contrast or 0, 2.0)
        full_three = next(row for row in summaries if row.event_time == 3)
        self.assertAlmostEqual(full_three.full_mean_contrast, 2.5)
        self.assertAlmostEqual(full_three.minimum_deletion_mean or 0, 2.0)
        self.assertAlmostEqual(full_three.maximum_deletion_mean or 0, 3.0)
        self.assertEqual(full_three.n_sign_reversals, 0)

    def test_dropping_a_donor_changes_matching_contrast(self) -> None:
        panel = panel_with_known_effects()
        for year in range(2000, 2020):
            original = panel[("CCC", year)]
            panel[("CCC", year)] = replace(
                original,
                outcome=original.outcome + 0.25 * (year - 2000),
            )
        deleted, summaries, _ = estimate_country_influence(
            panel, lags=5, max_horizon=3, min_events=1
        )
        donor_deleted = next(
            row
            for row in deleted
            if row.excluded_country == "CCC" and row.event_time == 3
        )
        full = next(row for row in summaries if row.event_time == 3)
        self.assertGreater(donor_deleted.mean_contrast or 0, full.full_mean_contrast)

    def test_report_not_estimable_country_deletions_without_fake_zeros(self) -> None:
        deleted, summaries, audit = estimate_country_influence(
            panel_with_known_effects(), lags=5, max_horizon=2, min_events=2
        )
        self.assertEqual(audit.failed_countries, ("AAA", "BBB"))
        self.assertEqual(audit.n_failed_country_deletions, 2)
        failures = [row for row in deleted if row.status != "success"]
        self.assertEqual(len(failures), 14)
        self.assertTrue(all(row.mean_contrast is None for row in failures))
        self.assertTrue(all(row.shift_from_full is None for row in failures))
        self.assertTrue(all(row.n_failed_deletions == 2 for row in summaries))
        self.assertTrue(all(row.n_successful_deletions == 3 for row in summaries))

    def test_detect_sign_reversal_when_one_treated_country_dominates(self) -> None:
        panel = panel_with_known_effects()
        for year in range(2006, 2020):
            original = panel[("AAA", year)]
            panel[("AAA", year)] = replace(original, outcome=original.outcome - 12)
        deleted, summaries, audit = estimate_country_influence(
            panel, lags=5, max_horizon=3, min_events=1
        )
        self.assertEqual(audit.n_failed_country_deletions, 0)
        full = next(row for row in summaries if row.event_time == 3)
        self.assertLess(full.full_mean_contrast, 0)
        self.assertEqual(full.n_sign_reversals, 1)
        first_removed = next(
            row
            for row in deleted
            if row.excluded_country == "AAA" and row.event_time == 3
        )
        self.assertGreater(first_removed.mean_contrast or 0, 0)

    def test_results_are_deterministic_independent_of_input_order(self) -> None:
        panel = panel_with_known_effects()
        forward = estimate_country_influence(panel, lags=5, max_horizon=2, min_events=1)
        reversed_panel = dict(reversed(list(panel.items())))
        backward = estimate_country_influence(
            reversed_panel, lags=5, max_horizon=2, min_events=1
        )
        self.assertEqual(forward, backward)

    def test_reject_underidentified_two_country_sample(self) -> None:
        two = {
            key: row
            for key, row in panel_with_known_effects().items()
            if key[0] in {"AAA", "BBB"}
        }
        with self.assertRaisesRegex(ValueError, "at least three countries"):
            estimate_country_influence(two, min_events=1)

    def test_output_names_and_metadata_do_not_claim_confidence_intervals(self) -> None:
        deleted, summaries, audit = estimate_country_influence(
            panel_with_known_effects(), lags=5, max_horizon=2, min_events=2
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_country_influence(
                output,
                deleted,
                summaries,
                audit,
                source_panel=output / "synthetic_panel.csv",
                outcome="top1_share_pct",
                covariates=(),
                lags=5,
                max_horizon=2,
            )
            with (output / "country_deletions.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5 * 7)
            self.assertEqual(
                len([row for row in rows if row["mean_contrast"] == ""]), 14
            )
            metadata = json.loads((output / "country_influence_audit.json").read_text())
            self.assertEqual(metadata["n_failed_country_deletions"], 2)
            self.assertIn("no bootstrap", metadata["inference"])
            self.assertIn("not_cluster_confidence_interval", metadata["status"])


if __name__ == "__main__":
    unittest.main()
