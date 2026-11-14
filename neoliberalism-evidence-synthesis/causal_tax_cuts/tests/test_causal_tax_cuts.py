"""Synthetic unit tests only; these are never presented as empirical findings."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.analysis import Observation, estimate, load_panel
from causal_tax_cuts.prepare import PAPER_COUNTRIES, prepare_panel


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Create controlled test inputs without relying on any external network."""
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_panel(*, covariates: bool = False) -> dict[tuple[str, int], Observation]:
    """Known +2/+3 event-onset effects with unchanged pre-trends."""
    observations: dict[tuple[str, int], Observation] = {}
    for index, country in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE")):
        for year in range(2000, 2020):
            treated = int((country, year) in {("AAA", 2006), ("BBB", 2011)})
            effect = 2.0 if country == "AAA" and year >= 2006 else 0.0
            effect += 3.0 if country == "BBB" and year >= 2011 else 0.0
            features = (100.0 + index + year / 10, 0.25 * index) if covariates else ()
            observations[(country, year)] = Observation(
                country,
                year,
                treated,
                20 + index + 0.4 * (year - 2000) + effect,
                features,
            )
    return observations


class PanelValidationTests(unittest.TestCase):
    """Reject suspicious country-year data before estimation."""

    def test_reject_duplicate_country_year(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path = Path(directory) / "panel.csv"
            row = {"country": "USA", "year": 1982, "treated": 1, "top1_share_pct": 12.5}
            write_csv(panel_path, list(row), [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_panel(panel_path, "top1_share_pct", ())

    def test_reject_nonfinite_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path = Path(directory) / "panel.csv"
            row = {
                "country": "USA",
                "year": 1982,
                "treated": 1,
                "top1_share_pct": "NaN",
            }
            write_csv(panel_path, list(row), [row])
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                load_panel(panel_path, "top1_share_pct", ())

    def test_reject_nonbinary_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path = Path(directory) / "panel.csv"
            row = {"country": "USA", "year": 1982, "treated": 2, "top1_share_pct": 12.5}
            write_csv(panel_path, list(row), [row])
            with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                load_panel(panel_path, "top1_share_pct", ())


class CausalDesignTests(unittest.TestCase):
    """Exercise strict history matching, horizon censoring and effect units."""

    def test_known_synthetic_effect_without_covariates(self) -> None:
        effects, summaries = estimate(
            synthetic_panel(), lags=5, max_horizon=3, min_events=2
        )
        self.assertEqual(len(effects), 8)
        for summary in summaries:
            self.assertEqual(summary.n_events, 2)
            self.assertEqual(summary.n_treated_countries, 2)
            self.assertAlmostEqual(summary.att, 2.5)
            self.assertAlmostEqual(summary.mean_pretrend_gap, 0)
            self.assertEqual(summary.n_unmatched, 0)

    def test_collinear_covariates_have_finite_ridge_distance(self) -> None:
        effects, summaries = estimate(
            synthetic_panel(covariates=True),
            lags=5,
            max_horizon=2,
            max_donors=2,
            min_events=2,
        )
        self.assertEqual(len(summaries), 3)
        self.assertEqual(len(effects), 6)
        self.assertTrue(all(item.mean_distance is not None for item in effects))
        self.assertTrue(all(math.isfinite(item.mean_distance) for item in effects))
        self.assertTrue(all(abs(summary.att - 2.5) < 1e-8 for summary in summaries))

    def test_censor_control_with_future_reform(self) -> None:
        effects, _ = estimate(synthetic_panel(), lags=5, max_horizon=5, min_events=2)
        first_reform = next(
            effect
            for effect in effects
            if effect.country == "AAA" and effect.year == 2006 and effect.horizon == 5
        )
        self.assertNotIn("BBB", first_reform.donors)

    def test_requires_multiple_matched_events(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum 3"):
            estimate(synthetic_panel(), lags=5, max_horizon=0, min_events=3)

    def test_history_must_be_complete(self) -> None:
        panel = synthetic_panel()
        for year in range(2001, 2006):
            panel.pop(("AAA", year))
        with self.assertRaisesRegex(ValueError, "only 1 matched"):
            estimate(panel, lags=5, max_horizon=0, min_events=2)


class SourcePreparationTests(unittest.TestCase):
    """Test 18-country source auditing and no invented missing measurements."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.events = self.directory / "author_reforms.csv"
        self.outcomes = self.directory / "outcomes.csv"
        self.country_codes = sorted(PAPER_COUNTRIES)
        self.outcome_rows = [
            {"country": country, "year": year, "top1_share_pct": 10.0 + index}
            for index, country in enumerate(self.country_codes)
            for year in range(2000, 2006)
        ]
        write_csv(
            self.outcomes, ["country", "year", "top1_share_pct"], self.outcome_rows
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(
        self, *, mode: str, expected: int = 18
    ) -> tuple[list[dict[str, str]], dict[str, object]]:
        """Exercise source preparation with a compact synthetic date window."""
        return prepare_panel(
            events_path=self.events,
            outcomes_path=self.outcomes,
            event_mode=mode,
            event_country="Nation",
            event_year="Date",
            treatment_field="Shock",
            outcome_country="country",
            outcome_year="year",
            outcome_fields=("top1_share_pct",),
            start_year=2000,
            end_year=2005,
            expected_events=expected,
        )

    def test_positive_event_list_expands_only_after_event_count_check(self) -> None:
        rows = [{"Nation": country, "Date": 2003} for country in self.country_codes]
        rows.append({"Nation": "Portugal", "Date": 2003})
        write_csv(self.events, ["Nation", "Date"], rows)
        joined, audit = self.prepare(mode="positive-events")
        self.assertEqual(len(joined), 18 * 6)
        self.assertEqual(sum(int(row["treated"]) for row in joined), 18)
        self.assertEqual(audit["skipped_event_country_labels"], {"Portugal": 1})

    def test_not_every_author_release_has_the_paper_event_count(self) -> None:
        rows = [{"Nation": country, "Date": 2003} for country in self.country_codes]
        write_csv(self.events, ["Nation", "Date"], rows)
        with self.assertRaisesRegex(ValueError, "expected 30"):
            self.prepare(mode="positive-events", expected=30)

    def test_binary_panel_rejects_missing_event_year(self) -> None:
        rows = [
            {"Nation": country, "Date": year, "Shock": int(year == 2003)}
            for country in self.country_codes
            for year in range(2000, 2006)
            if (country, year) != (self.country_codes[0], 2004)
        ]
        write_csv(self.events, ["Nation", "Date", "Shock"], rows)
        with self.assertRaisesRegex(ValueError, "missing 1 country-years"):
            self.prepare(mode="binary-panel")

    def test_missing_outcome_does_not_become_zero(self) -> None:
        rows = [{"Nation": country, "Date": 2003} for country in self.country_codes]
        write_csv(self.events, ["Nation", "Date"], rows)
        missing = {"country": self.country_codes[0], "year": 2003}
        for row in self.outcome_rows:
            if row["country"] == missing["country"] and row["year"] == missing["year"]:
                row["top1_share_pct"] = ""
        write_csv(
            self.outcomes, ["country", "year", "top1_share_pct"], self.outcome_rows
        )
        joined, audit = self.prepare(mode="positive-events")
        self.assertEqual(len(joined), 18 * 6 - 1)
        self.assertEqual(audit["n_lost_event_years_due_to_outcome_gaps"], 1)
        self.assertFalse(
            any(
                row["country"] == missing["country"]
                and int(row["year"]) == missing["year"]
                for row in joined
            )
        )


if __name__ == "__main__":
    unittest.main()
