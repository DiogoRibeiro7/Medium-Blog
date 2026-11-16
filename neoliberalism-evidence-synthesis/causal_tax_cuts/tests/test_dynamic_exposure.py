"""Synthetic and adversarial tests for time-varying policy-path diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.analysis import Key, Observation
from causal_tax_cuts.dynamic_exposure import (
    _path,
    audit_dynamic_exposure,
    save_dynamic_exposure,
)
from causal_tax_cuts.macro_matching import MacroValues


def fixture() -> tuple[dict[Key, Observation], dict[Key, MacroValues]]:
    """One initiating reform, five donors, and two future donor reforms."""
    levels = {
        "AAA": (1.0, 2.0, 3.0),
        "BBB": (1.0, 2.0, 3.0),
        "CCC": (3.0, 3.0, 6.0),
        "DDD": (4.0, 5.0, 8.0),
        "EEE": (0.0, 1.0, 2.0),
        "FFF": (7.0, 8.0, 9.0),
    }
    panel: dict[Key, Observation] = {}
    controls: dict[Key, MacroValues] = {}
    for index, (country, values) in enumerate(levels.items()):
        for year in range(2000, 2019):
            treated = int(
                (country, year) in {("AAA", 2010), ("EEE", 2011), ("BBB", 2013)}
            )
            panel[country, year] = Observation(
                country, year, treated, 10.0 + index + 0.1 * (year - 2000), ()
            )
            controls[country, year] = values
    return panel, controls


class DynamicExposureTests(unittest.TestCase):
    """Separate repeat policy cuts from cohort selection and WDI availability."""

    def test_baseline_and_future_risk_sets_match_prior_audit(self) -> None:
        panel, controls = fixture()
        paths, recurrences, rows, report = audit_dynamic_exposure(panel, controls)
        aaa = {row.horizon: row for row in rows if row.country == "AAA"}
        self.assertEqual(aaa[0].n_baseline_comparators, 5)
        self.assertEqual(aaa[0].n_no_cut_comparators_through_h, 5)
        self.assertEqual(aaa[1].n_no_cut_comparators_through_h, 4)
        self.assertEqual(aaa[3].n_no_cut_comparators_through_h, 3)
        self.assertEqual(aaa[5].n_both_strategy_adherent_pairs, 3)
        self.assertTrue(aaa[5].treated_single_cut_strategy_adherent)
        self.assertTrue(report["reconciled_to_prior_donor_risk_sets"])
        self.assertEqual(
            len(paths), 6 * (len(recurrences) + report["n_pre_censor_comparator_pairs"])
        )

    def test_repeat_cut_censors_treated_sustained_strategy(self) -> None:
        panel, controls = fixture()
        original = panel["AAA", 2012]
        panel["AAA", 2012] = Observation(
            original.country, original.year, 1, original.outcome, ()
        )
        paths, recurrences, rows, _ = audit_dynamic_exposure(panel, controls)
        aaa = {
            row.horizon: row
            for row in rows
            if row.country == "AAA" and row.reform_year == 2010
        }
        self.assertTrue(aaa[1].treated_single_cut_strategy_adherent)
        self.assertFalse(aaa[2].treated_single_cut_strategy_adherent)
        self.assertEqual(aaa[2].treated_additional_cuts_through_h, 1)
        self.assertEqual(aaa[2].n_no_cut_comparators_through_h, 4)
        self.assertEqual(aaa[2].n_both_strategy_adherent_pairs, 0)
        reform = next(
            row
            for row in recurrences
            if (row.country, row.reform_year) == ("AAA", 2010)
        )
        self.assertEqual(reform.first_additional_major_cut_year, 2012)
        self.assertEqual(reform.n_additional_major_cuts_through_h5, 1)
        treated_rows = [
            row
            for row in paths
            if row.focal_country == "AAA"
            and row.reform_year == 2010
            and row.role == "initiator"
        ]
        self.assertEqual(
            [row.cumulative_major_cuts_since_t for row in treated_rows],
            [1, 1, 2, 2, 2, 2],
        )

    def test_simultaneous_comparator_cut_is_baseline_nonadherence(self) -> None:
        panel, controls = fixture()
        old = panel["EEE", 2011]
        panel["EEE", 2011] = Observation(old.country, old.year, 0, old.outcome, ())
        at_t = panel["EEE", 2010]
        panel["EEE", 2010] = Observation(at_t.country, at_t.year, 1, at_t.outcome, ())
        paths, _, rows, _ = audit_dynamic_exposure(panel, controls)
        aaa = next(
            row
            for row in rows
            if row.country == "AAA" and row.reform_year == 2010 and row.horizon == 0
        )
        self.assertEqual(aaa.n_baseline_comparators, 5)
        self.assertEqual(aaa.n_no_cut_comparators_through_h, 4)
        eee = next(
            row
            for row in paths
            if (row.focal_country, row.reform_year, row.role, row.country, row.horizon)
            == ("AAA", 2010, "comparator", "EEE", 0)
        )
        self.assertEqual(eee.deviations_from_declared_strategy, 1)
        self.assertFalse(eee.adherent_to_declared_strategy)

    def test_postbaseline_macro_missingness_is_coverage_only(self) -> None:
        panel, controls = fixture()
        _, _, before, _ = audit_dynamic_exposure(panel, controls)
        controls["CCC", 2011] = (None, 3.0, 6.0)
        _, _, after, _ = audit_dynamic_exposure(panel, controls)
        original = {
            row.horizon: row
            for row in before
            if row.country == "AAA" and row.reform_year == 2010
        }
        modified = {
            row.horizon: row
            for row in after
            if row.country == "AAA" and row.reform_year == 2010
        }
        for h in range(6):
            self.assertEqual(
                original[h].n_no_cut_comparators_through_h,
                modified[h].n_no_cut_comparators_through_h,
            )
        self.assertEqual(
            original[2].n_no_cut_comparators_with_sequential_wdi,
            modified[2].n_no_cut_comparators_with_sequential_wdi + 1,
        )
        self.assertEqual(
            original[1].n_no_cut_comparators_with_sequential_wdi,
            modified[1].n_no_cut_comparators_with_sequential_wdi,
        )

    def test_missing_treated_postbaseline_wdi_preserves_exposure_counts(self) -> None:
        panel, controls = fixture()
        controls["AAA", 2012] = (None, None, None)
        _, _, rows, _ = audit_dynamic_exposure(panel, controls)
        aaa = {
            row.horizon: row
            for row in rows
            if row.country == "AAA" and row.reform_year == 2010
        }
        self.assertTrue(aaa[2].treated_all_predecision_wdi_complete)
        self.assertFalse(aaa[3].treated_all_predecision_wdi_complete)
        self.assertEqual(aaa[3].n_no_cut_comparators_through_h, 3)
        self.assertEqual(aaa[3].n_both_strategy_adherent_pairs, 3)
        self.assertEqual(aaa[3].n_both_strategy_adherent_pairs_with_sequential_wdi, 0)

    def test_invalid_role_and_nonidentical_joins(self) -> None:
        panel, controls = fixture()
        with self.assertRaisesRegex(ValueError, "initiating country"):
            _path(
                panel, controls, focal="AAA", year=2010, role="initiator", country="BBB"
            )
        controls.pop(("AAA", 2009))
        with self.assertRaisesRegex(ValueError, "keys must match"):
            audit_dynamic_exposure(panel, controls)

    def test_empty_reform_cohort_is_explicit_failure(self) -> None:
        panel, controls = fixture()
        for key, row in list(panel.items()):
            if row.treated:
                panel[key] = Observation(row.country, row.year, 0, row.outcome, ())
        with self.assertRaisesRegex(ValueError, "No eligible"):
            audit_dynamic_exposure(panel, controls)

    def test_save_reconciles_frozen_source_and_rejects_wrong_counts(self) -> None:
        panel, controls = fixture()
        paths, recurrences, horizons, report = audit_dynamic_exposure(panel, controls)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            joined = output / "joined.csv"
            joined.write_text("fixture\n", encoding="utf-8")
            frozen = output / "prior.json"
            original = {
                "joined_panel_sha256": hashlib.sha256(joined.read_bytes()).hexdigest(),
                "by_horizon": [
                    {
                        "horizon": h,
                        "n_donor_reform_pairs_at_risk": next(
                            row["n_no_cut_comparator_pairs"]
                            for row in report["by_horizon"]
                            if row["horizon"] == h
                        ),
                    }
                    for h in range(6)
                ],
            }
            frozen.write_text(json.dumps(original), encoding="utf-8")
            save_dynamic_exposure(
                output,
                paths,
                recurrences,
                horizons,
                report,
                source_panel=joined,
                frozen_reference=frozen,
            )
            saved = json.loads((output / "dynamic_exposure_audit.json").read_text())
            self.assertEqual(saved["frozen_reference_status"], "same_source_validated")
            with (output / "dynamic_policy_paths.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(paths))
            original["by_horizon"][2]["n_donor_reform_pairs_at_risk"] += 1
            frozen.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen donor exposure mismatch"):
                save_dynamic_exposure(
                    output,
                    paths,
                    recurrences,
                    horizons,
                    report,
                    source_panel=joined,
                    frozen_reference=frozen,
                )

    def test_stale_reference_marks_vintage_instead_of_falsely_matching(self) -> None:
        panel, controls = fixture()
        paths, recurrences, horizons, report = audit_dynamic_exposure(panel, controls)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            joined = output / "joined.csv"
            joined.write_text("updated\n", encoding="utf-8")
            frozen = output / "prior.json"
            frozen.write_text(
                json.dumps({"joined_panel_sha256": "0" * 64}), encoding="utf-8"
            )
            save_dynamic_exposure(
                output,
                paths,
                recurrences,
                horizons,
                report,
                source_panel=joined,
                frozen_reference=frozen,
            )
            saved = json.loads((output / "dynamic_exposure_audit.json").read_text())
            self.assertEqual(saved["frozen_reference_status"], "source_vintage_changed")


if __name__ == "__main__":
    unittest.main()
