"""Network-free validation of balanced pre-reform placebo calculations."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from causal_tax_cuts.analysis import Key, Observation
from causal_tax_cuts.event_study import estimate_event_study, save_event_study


def panel_with_known_effects(*, covariates: bool = False) -> dict[Key, Observation]:
    """Two known post-event level shifts with identical pre-event slopes."""
    panel: dict[Key, Observation] = {}
    for index, country in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE")):
        for year in range(2000, 2020):
            treated = int((country, year) in {("AAA", 2006), ("BBB", 2011)})
            shock = 2.0 if country == "AAA" and year >= 2006 else 0.0
            shock += 3.0 if country == "BBB" and year >= 2011 else 0.0
            panel[(country, year)] = Observation(
                country=country,
                year=year,
                treated=treated,
                outcome=20.0 + index + 0.4 * (year - 2000) + shock,
                covariates=(100.0 + index, 2.0 * index) if covariates else (),
            )
    return panel


class EventStudyTests(unittest.TestCase):
    """Check data-leakage controls, balanced cohorts and placebo meanings."""

    def test_parallel_pretrends_and_known_post_event_effects(self) -> None:
        contrasts, summaries, audit = estimate_event_study(
            panel_with_known_effects(), lags=5, max_horizon=3, min_events=2
        )
        self.assertEqual(
            [row.event_time for row in summaries], [-5, -4, -3, -2, 0, 1, 2, 3]
        )
        self.assertEqual(len(contrasts), 16)
        self.assertEqual(audit.n_matched_events, 2)
        self.assertEqual(audit.n_treated_countries, 2)
        self.assertAlmostEqual(audit.mean_pretrend_slope_gap, 0)
        for summary in summaries:
            self.assertEqual(summary.n_events, 2)
            expected = 0.0 if summary.event_time < 0 else 2.5
            self.assertAlmostEqual(summary.mean_contrast, expected)

    def test_nonparallel_pretrend_is_visible(self) -> None:
        panel = panel_with_known_effects()
        for year in range(2000, 2006):
            original = panel[("AAA", year)]
            panel[("AAA", year)] = replace(
                original,
                outcome=original.outcome + 0.2 * (year - 2000),
            )
        _, summaries, audit = estimate_event_study(
            panel, lags=5, max_horizon=3, min_events=2
        )
        self.assertAlmostEqual(summaries[0].mean_contrast, -0.4)
        self.assertAlmostEqual(audit.mean_pretrend_slope_gap, 0.1)

    def test_same_donors_across_leads_and_followup(self) -> None:
        panel = panel_with_known_effects()
        original = panel[("CCC", 2009)]
        panel[("CCC", 2009)] = replace(original, treated=1)
        contrasts, _, _ = estimate_event_study(
            panel, lags=5, max_horizon=5, min_events=2
        )
        focal = [
            row for row in contrasts if (row.country, row.reform_year) == ("AAA", 2006)
        ]
        self.assertEqual({row.donors for row in focal}, {focal[0].donors})
        self.assertNotIn("CCC", focal[0].donors)
        self.assertEqual(len(focal), 10)

    def test_missing_followup_excludes_event_at_every_horizon(self) -> None:
        panel = panel_with_known_effects()
        panel.pop(("AAA", 2010))
        contrasts, summaries, audit = estimate_event_study(
            panel, lags=5, max_horizon=5, min_events=1
        )
        self.assertEqual(audit.n_original_event_rows, 2)
        self.assertEqual(audit.n_eligible_events, 1)
        self.assertEqual(audit.n_matched_events, 1)
        self.assertTrue(all(row.country == "BBB" for row in contrasts))
        self.assertTrue(all(row.n_events == 1 for row in summaries))

    def test_covariate_refinement_respects_donor_limit(self) -> None:
        contrasts, _, audit = estimate_event_study(
            panel_with_known_effects(covariates=True),
            lags=5,
            max_horizon=2,
            max_donors=2,
            min_events=2,
        )
        self.assertEqual(audit.max_donors, 2)
        self.assertTrue(all(len(item.donors) <= 2 for item in contrasts))

    def test_fail_closed_on_too_few_matched_events(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum 3"):
            estimate_event_study(
                panel_with_known_effects(), lags=5, max_horizon=0, min_events=3
            )

    def test_reject_invalid_windows_and_empty_panel(self) -> None:
        with self.assertRaisesRegex(ValueError, "lags >= 3"):
            estimate_event_study(panel_with_known_effects(), lags=2)
        with self.assertRaisesRegex(ValueError, "no observations"):
            estimate_event_study({}, min_events=1)

    def test_outputs_have_declared_units_and_no_fake_pvalues(self) -> None:
        contrasts, summaries, audit = estimate_event_study(
            panel_with_known_effects(), lags=5, max_horizon=2, min_events=2
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_event_study(
                output,
                contrasts,
                summaries,
                audit,
                source_panel=output / "panel.csv",
                outcome="top1_share_pct",
                covariates=(),
                lags=5,
                max_horizon=2,
            )
            with (output / "event_time_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)
            self.assertEqual(
                [int(row["event_time"]) for row in rows[:4]], [-5, -4, -3, -2]
            )
            audit_record = json.loads((output / "event_study_audit.json").read_text())
            self.assertEqual(audit_record["baseline"], "year t-1")
            self.assertIn("no p-values", audit_record["inference"])
            self.assertEqual(audit_record["n_matched_events"], 2)


if __name__ == "__main__":
    unittest.main()
