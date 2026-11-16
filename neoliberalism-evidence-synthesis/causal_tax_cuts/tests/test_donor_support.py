"""Synthetic tests for donor exposure and fixed-scale marginal overlap."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.analysis import Key, Observation
from causal_tax_cuts.donor_support import audit_future_donor_exposure, save_support
from causal_tax_cuts.macro_matching import MacroValues


def fixture() -> tuple[dict[Key, Observation], dict[Key, MacroValues]]:
    """2010 tax cut with two donors reforming during five-year follow-up."""
    levels = {
        "AAA": (1.0, 2.0, 3.0),
        "BBB": (1.0, 2.0, 3.0),
        "CCC": (3.0, 3.0, 6.0),
        "DDD": (4.0, 5.0, 8.0),
        "EEE": (0.0, 1.0, 2.0),
        "FFF": (7.0, 8.0, 9.0),
    }
    panel: dict[Key, Observation] = {}
    macro: dict[Key, MacroValues] = {}
    for index, (country, values) in enumerate(levels.items()):
        for year in range(2000, 2019):
            treated = int(
                (country, year) in {("AAA", 2010), ("EEE", 2011), ("BBB", 2013)}
            )
            panel[country, year] = Observation(
                country, year, treated, 12.0 + index + (year - 2000) * 0.2, ()
            )
            macro[country, year] = values
    return panel, macro


class DonorSupportTests(unittest.TestCase):
    """Future reforms change donor availability, not lagged WDI features."""

    def test_horizon_risk_sets_and_range_loss(self) -> None:
        panel, controls = fixture()
        events, donors, horizons, report = audit_future_donor_exposure(panel, controls)
        original = next(
            row for row in events if (row.country, row.reform_year) == ("AAA", 2010)
        )
        self.assertEqual(original.status, "baseline_donors_available")
        self.assertEqual(original.n_pre_censor_donors, 5)
        self.assertEqual(original.n_horizon_zero_donors, 5)
        self.assertEqual(original.n_horizon_five_donors, 3)
        self.assertEqual(original.n_future_exposed_donors_by_five, 2)

        aaa = {row.horizon: row for row in horizons if row.country == "AAA"}
        self.assertEqual(aaa[-1].n_surviving_donors, 5)
        self.assertEqual(aaa[0].n_surviving_donors, 5)
        self.assertEqual(aaa[1].n_surviving_donors, 4)
        self.assertEqual(aaa[2].n_surviving_donors, 4)
        self.assertEqual(aaa[3].n_surviving_donors, 3)
        self.assertFalse(aaa[5].within_all_three_marginal_ranges)
        self.assertTrue(aaa[-1].within_all_three_marginal_ranges)
        self.assertAlmostEqual(aaa[-1].nearest_fixed_standardized_distance, 0.0)
        self.assertGreater(aaa[5].nearest_fixed_standardized_distance, 0)
        self.assertEqual(len(report["by_horizon"]), 7)

        reforming = {row.donor: row for row in donors if row.country == "AAA"}
        self.assertEqual(reforming["EEE"].first_censored_horizon, 1)
        self.assertEqual(reforming["BBB"].first_censored_horizon, 3)
        self.assertIsNone(reforming["CCC"].first_censored_horizon)

    def test_same_year_reform_is_censored_at_horizon_zero(self) -> None:
        panel, controls = fixture()
        prior = panel["EEE", 2011]
        panel["EEE", 2011] = Observation(
            prior.country, prior.year, 0, prior.outcome, ()
        )
        now = panel["EEE", 2010]
        panel["EEE", 2010] = Observation(now.country, now.year, 1, now.outcome, ())
        events, _, horizons, _ = audit_future_donor_exposure(panel, controls)
        original = next(row for row in events if row.country == "AAA")
        self.assertEqual(original.n_pre_censor_donors, 5)
        self.assertEqual(original.n_horizon_zero_donors, 4)
        aaa = {row.horizon: row for row in horizons if row.country == "AAA"}
        self.assertEqual(aaa[0].n_censored_donors, 1)

    def test_no_post_treatment_macro_leakage(self) -> None:
        panel, controls = fixture()
        before = audit_future_donor_exposure(panel, controls)
        # Isolate the 2010 cohort: later events legitimately need their own
        # post-2010 pre-treatment WDI histories, so their eligibility may vary.
        for key in list(controls):
            if key[1] >= 2010:
                controls[key] = (None, None, None)
        after = audit_future_donor_exposure(panel, controls)
        for index in range(3):
            self.assertEqual(
                [row for row in before[index] if row.country == "AAA"],
                [row for row in after[index] if row.country == "AAA"],
            )

    def test_incomplete_future_wid_window_is_constant_across_horizons(self) -> None:
        panel, controls = fixture()
        panel.pop(("CCC", 2015))
        controls.pop(("CCC", 2015))
        _, _, horizons, _ = audit_future_donor_exposure(panel, controls)
        aaa = {row.horizon: row for row in horizons if row.country == "AAA"}
        self.assertEqual(aaa[-1].n_surviving_donors, 4)
        self.assertEqual(aaa[5].n_surviving_donors, 2)

    def test_missing_pre_reform_macro_excludes_donor_only(self) -> None:
        panel, controls = fixture()
        controls["CCC", 2008] = (None, 3.0, 6.0)
        _, donor_rows, horizons, _ = audit_future_donor_exposure(panel, controls)
        self.assertNotIn(
            "CCC", {row.donor for row in donor_rows if row.country == "AAA"}
        )
        aaa = next(
            row for row in horizons if row.country == "AAA" and row.horizon == -1
        )
        self.assertEqual(aaa.n_surviving_donors, 4)

    def test_zero_variance_does_not_invent_distance_or_support(self) -> None:
        panel, controls = fixture()
        for key in list(controls):
            controls[key] = (1.0, 2.0, 3.0)
        _, donors, horizons, _ = audit_future_donor_exposure(panel, controls)
        aaa = next(row for row in horizons if row.country == "AAA" and row.horizon == 5)
        self.assertTrue(aaa.within_all_three_marginal_ranges)
        self.assertAlmostEqual(aaa.nearest_fixed_standardized_distance, 0)
        self.assertTrue(
            all(row.fixed_pre5_standardized_distance == 0 for row in donors)
        )

    def test_reject_nonidentical_country_year_joins(self) -> None:
        panel, controls = fixture()
        controls.pop(("AAA", 2008))
        with self.assertRaisesRegex(ValueError, "identical keys"):
            audit_future_donor_exposure(panel, controls)

    def test_saved_audit_reconciles_previous_five_year_matching(self) -> None:
        panel, controls = fixture()
        events, donor_rows, horizons, audit = audit_future_donor_exposure(
            panel, controls
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "joined.csv"
            source.write_text("fixture\n", encoding="utf-8")
            save_support(
                directory,
                events,
                donor_rows,
                horizons,
                audit,
                source_panel=source,
                panel=panel,
                controls=controls,
            )
            result = json.loads((directory / "donor_support_audit.json").read_text())
            self.assertTrue(result["reference_matched_event_ids_consistent"])
            self.assertEqual(len(result["joined_panel_sha256"]), 64)
            with (directory / "future_donor_exposure.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)
            self.assertEqual(
                len((directory / "support_by_horizon.csv").read_text().splitlines())
                - 1,
                len(horizons),
            )


if __name__ == "__main__":
    unittest.main()
