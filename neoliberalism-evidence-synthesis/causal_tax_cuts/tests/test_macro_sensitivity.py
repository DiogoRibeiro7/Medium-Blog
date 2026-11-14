"""Offline tests for the pre-reform matching sensitivity grid."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.analysis import Key, Observation
from causal_tax_cuts.macro_matching import MacroValues, audit_macro_matching
from causal_tax_cuts.macro_sensitivity import (
    _fixed_pre3_diagnostics,
    audit_sensitivity_grid,
    save_sensitivity,
)


def fixture() -> tuple[dict[Key, Observation], dict[Key, MacroValues]]:
    """One event with seven donors, complete windows and varied WDI means."""
    profiles = {
        "AAA": (1.0, 2.0, 3.0),
        "BBB": (1.0, 2.0, 3.0),
        "CCC": (5.0, 3.0, 8.0),
        "DDD": (6.0, 3.0, 4.0),
        "EEE": (3.0, 9.0, 5.0),
        "FFF": (4.0, 2.0, 12.0),
        "GGG": (2.0, 6.0, 5.0),
        "HHH": (3.0, 3.0, 6.0),
    }
    panel: dict[Key, Observation] = {}
    controls: dict[Key, MacroValues] = {}
    for index, (country, features) in enumerate(profiles.items()):
        for year in range(2000, 2020):
            panel[country, year] = Observation(
                country,
                year,
                int((country, year) == ("AAA", 2010)),
                12.0 + index + (0.2 + 0.01 * index) * (year - 2000),
                (),
            )
            controls[country, year] = features
    return panel, controls


class SensitivityTests(unittest.TestCase):
    """Cohort intersection, source stability, balance windows and validation."""

    def test_nine_rules_and_reference_reuse_existing_audit(self) -> None:
        panel, controls = fixture()
        grid, details, report = audit_sensitivity_grid(panel, controls)
        self.assertEqual(len(grid), 9)
        self.assertEqual(len(details), 9)
        self.assertEqual(report["n_common_matched_events"], 1)
        reference = next(row for row in grid if (row.lags, row.max_donors) == (5, 5))
        _, _, old_audit = audit_macro_matching(panel, controls)
        self.assertEqual(reference.n_matched_events, old_audit["n_matched_events"])
        self.assertEqual(reference.n_selected_donor_event_pairs, 5)
        self.assertEqual(reference.mean_reference_donor_jaccard_common, 1.0)
        self.assertEqual(report["reference_rule"], {"lags": 5, "max_donors": 5})

    def test_donor_caps_change_size_and_comparability_is_fixed(self) -> None:
        panel, controls = fixture()
        grid, details, _ = audit_sensitivity_grid(panel, controls)
        for cap in (3, 5, 7):
            point = next(row for row in grid if (row.lags, row.max_donors) == (5, cap))
            event = next(
                row for row in details if (row.lags, row.max_donors) == (5, cap)
            )
            self.assertEqual(len(event.selected_donors), cap)
            self.assertEqual(point.n_common_cohort_events, 1)
        # The common-cohort diagnostics do not vary with the number of
        # calendar years chosen for the *selection* profile.
        self.assertEqual(
            {(row.lags, row.max_donors) for row in grid},
            {(lag, cap) for lag in (3, 5, 7) for cap in (3, 5, 7)},
        )

    def test_fixed_three_year_outcome_slope_has_known_units(self) -> None:
        panel, controls = fixture()
        slope, *_ = _fixed_pre3_diagnostics(panel, controls, "AAA", 2010, ("CCC",))
        self.assertAlmostEqual(slope, -0.02)

    def test_no_macro_at_or_after_event_affects_grid(self) -> None:
        panel, controls = fixture()
        original = audit_sensitivity_grid(panel, controls)
        for key in list(controls):
            if key[1] >= 2010:
                controls[key] = (None, None, None)
        changed = audit_sensitivity_grid(panel, controls)
        self.assertEqual(original, changed)

    def test_common_cohort_can_be_empty_without_zero_imputation(self) -> None:
        panel, controls = fixture()
        controls["AAA", 2003] = (None, 2.0, 3.0)
        grid, details, report = audit_sensitivity_grid(panel, controls)
        self.assertEqual(report["n_common_matched_events"], 0)
        self.assertEqual(next(row for row in grid if row.lags == 7).n_matched_events, 0)
        self.assertEqual(next(row for row in grid if row.lags == 5).n_matched_events, 1)
        self.assertIsNone(grid[0].mean_abs_fixed_pre3_outcome_slope_gap_common)
        self.assertEqual(len(details), 9)

    def test_donor_variance_zero_is_missing_smd_not_perfect_balance(self) -> None:
        panel, controls = fixture()
        for key, values in list(controls.items()):
            controls[key] = (values[0], values[1], 3.0)
        grid, details, _ = audit_sensitivity_grid(panel, controls)
        self.assertIsNone(grid[0].mean_abs_fixed_pre3_trade_smd_common)
        self.assertIsNone(details[0].fixed_pre3_trade_smd)

    def test_reject_invalid_or_duplicated_grid_without_reference(self) -> None:
        panel, controls = fixture()
        cases = [
            ((2, 5), (3, 5)),
            ((3, 7), (3, 5)),
            ((3, 5, 5), (3, 5)),
            ((3, 5), (3, 0, 5)),
            ((3, 5), (3, 5, 5)),
            ((), (3, 5)),
        ]
        for lags, donors in cases:
            with (
                self.subTest(lags=lags, donors=donors),
                self.assertRaisesRegex(ValueError, "Grid"),
            ):
                audit_sensitivity_grid(
                    panel, controls, lag_values=lags, donor_caps=donors
                )

    def test_reference_is_checked_only_on_identical_joined_source(self) -> None:
        panel, controls = fixture()
        grid, details, audit = audit_sensitivity_grid(panel, controls)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            panel_file = temporary / "panel.csv"
            panel_file.write_text("fixture\n", encoding="utf-8")
            panel_sha256 = hashlib.sha256(panel_file.read_bytes()).hexdigest()
            _, _, old = audit_macro_matching(panel, controls)
            old["joined_panel_sha256"] = panel_sha256
            reference_file = temporary / "old.json"
            reference_file.write_text(json.dumps(old), encoding="utf-8")
            save_sensitivity(
                temporary,
                grid,
                details,
                audit,
                source_panel=panel_file,
                frozen_reference=reference_file,
                panel=panel,
                controls=controls,
            )
            saved = json.loads((temporary / "macro_sensitivity_audit.json").read_text())
            self.assertEqual(
                saved["frozen_reference_check"],
                "same_source_and_reference_estimates_reproduced",
            )
            self.assertTrue(saved["reference_matches_prior_audit_by_construction"])
            event_csv = temporary / "macro_sensitivity_events.csv"
            with event_csv.open("r", encoding="utf-8") as saved_events:
                self.assertEqual(sum(1 for _ in saved_events) - 1, 9)

            old["n_matched_events"] = 999
            reference_file.write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reference rule changed"):
                save_sensitivity(
                    temporary,
                    grid,
                    details,
                    audit,
                    source_panel=panel_file,
                    frozen_reference=reference_file,
                    panel=panel,
                    controls=controls,
                )
            old["joined_panel_sha256"] = "a" * 64
            reference_file.write_text(json.dumps(old), encoding="utf-8")
            save_sensitivity(
                temporary,
                grid,
                details,
                audit,
                source_panel=panel_file,
                frozen_reference=reference_file,
                panel=panel,
                controls=controls,
            )
            saved = json.loads((temporary / "macro_sensitivity_audit.json").read_text())
            self.assertEqual(
                saved["frozen_reference_check"],
                "source_vintage_changed_not_comparable",
            )


if __name__ == "__main__":
    unittest.main()
