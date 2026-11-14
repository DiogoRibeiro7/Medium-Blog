"""Offline fixtures for the pre-treatment WDI eligibility and balance audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.analysis import Key, Observation
from causal_tax_cuts.macro_matching import (
    MacroValues,
    audit_macro_matching,
    read_macro_controls,
    save_audit,
)
from causal_tax_cuts.world_bank import CORE_COLUMNS


def fake_joined_panel() -> tuple[dict[Key, Observation], dict[Key, MacroValues]]:
    """One 2010 event with six donors and twelve-year observed windows."""
    panel: dict[Key, Observation] = {}
    controls: dict[Key, MacroValues] = {}
    profiles = {
        "AAA": (1.0, 2.0, 3.0),
        "BBB": (1.0, 2.0, 3.0),
        "CCC": (9.0, 2.0, 3.0),
        "DDD": (4.0, 2.0, 3.0),
        "EEE": (2.0, 5.0, 3.0),
        "FFF": (2.0, 2.0, 12.0),
        "GGG": (1.0, 3.0, 3.0),
    }
    for rank, (country, macro) in enumerate(profiles.items()):
        for year in range(2000, 2018):
            panel[country, year] = Observation(
                country,
                year,
                int((country, year) == ("AAA", 2010)),
                8.0 + rank + 0.2 * (year - 2000),
                (),
            )
            controls[country, year] = macro
    return panel, controls


class MacroMatchingTests(unittest.TestCase):
    """Check missingness, time ordering, exposure and balance semantics."""

    def test_balanced_macro_matching_prefers_exact_lagged_profile(self) -> None:
        panel, controls = fake_joined_panel()
        events, balance, report = audit_macro_matching(panel, controls, max_donors=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].status, "matched")
        self.assertEqual(events[0].selected_donors, ("BBB",))
        self.assertEqual(events[0].history_only_donors, 6)
        self.assertEqual(events[0].macro_complete_donors, 6)
        self.assertEqual(report["n_matched_events"], 1)
        self.assertEqual(len(balance), 3)
        gdp = next(row for row in balance if row.control == CORE_COLUMNS[0])
        self.assertGreater(abs(gdp.standardized_gap_pool), 0.0)
        self.assertAlmostEqual(gdp.standardized_gap_selected, 0.0)

    def test_post_treatment_macro_changes_do_not_change_selection(self) -> None:
        panel, controls = fake_joined_panel()
        original, _, _ = audit_macro_matching(panel, controls, max_donors=2)
        for country in {key[0] for key in controls}:
            for year in range(2010, 2018):
                # A complete-case estimator would incorrectly discard these
                # rows. This audit never looks at contemporaneous or future WDI.
                controls[country, year] = (None, None, None)
        changed, _, _ = audit_macro_matching(panel, controls, max_donors=2)
        self.assertEqual(original, changed)

    def test_missing_treated_lag_is_not_filled_or_post_selected(self) -> None:
        panel, controls = fake_joined_panel()
        controls["AAA", 2008] = (None, 2.0, 3.0)
        events, balance, report = audit_macro_matching(panel, controls)
        self.assertEqual(events[0].status, "incomplete_treated_pre_macro")
        self.assertEqual(events[0].history_only_donors, 6)
        self.assertEqual(events[0].macro_complete_donors, 0)
        self.assertEqual(report["n_matched_events"], 0)
        self.assertFalse(balance)

    def test_missing_donor_lags_reduce_pool_without_deleting_outcomes(self) -> None:
        panel, controls = fake_joined_panel()
        controls["CCC", 2006] = (9.0, None, 3.0)
        events, _, report = audit_macro_matching(panel, controls)
        self.assertEqual(events[0].history_only_donors, 6)
        self.assertEqual(events[0].macro_complete_donors, 5)
        self.assertEqual(report["n_original_history_donors_across_matched_events"], 6)
        self.assertEqual(report["n_macro_complete_donors_across_matched_events"], 5)

    def test_future_exposed_donor_is_censored_with_explicit_caveat(self) -> None:
        panel, controls = fake_joined_panel()
        row = panel["BBB", 2012]
        panel["BBB", 2012] = Observation(row.country, row.year, 1, row.outcome, ())
        events, _, audit = audit_macro_matching(panel, controls, max_donors=1)
        self.assertEqual(events[0].history_only_donors, 5)
        self.assertNotIn("BBB", events[0].selected_donors)
        self.assertIn("future reform", " ".join(audit["caveats"]))

    def test_incomplete_treatment_history_and_outcomes_audited(self) -> None:
        panel, controls = fake_joined_panel()
        panel.pop(("AAA", 2005))
        controls.pop(("AAA", 2005))
        events, _, _ = audit_macro_matching(panel, controls)
        self.assertEqual(events[0].status, "invalid_treatment_history")

        panel, controls = fake_joined_panel()
        panel.pop(("AAA", 2013))
        controls.pop(("AAA", 2013))
        events, _, _ = audit_macro_matching(panel, controls)
        self.assertEqual(events[0].status, "incomplete_treated_outcome_window")

    def test_zero_donor_variance_is_undefined_not_zero_balance(self) -> None:
        panel, controls = fake_joined_panel()
        for key, values in list(controls.items()):
            controls[key] = (values[0], values[1], 3.0)
        _, balance, audit = audit_macro_matching(panel, controls)
        trade = next(row for row in balance if row.control == CORE_COLUMNS[2])
        self.assertIsNone(trade.standardized_gap_pool)
        self.assertIsNone(trade.standardized_gap_selected)
        self.assertEqual(
            audit["variable_balance"][CORE_COLUMNS[2]]["n_evaluable_events"], 0
        )

    def test_mismatch_rejected_without_guessing_missing_rows(self) -> None:
        panel, controls = fake_joined_panel()
        controls.pop(("AAA", 2001))
        with self.assertRaisesRegex(ValueError, "keys differ"):
            audit_macro_matching(panel, controls)

    def test_no_history_eligible_donors_reported(self) -> None:
        panel, controls = fake_joined_panel()
        for country in ("BBB", "CCC", "DDD", "EEE", "FFF", "GGG"):
            row = panel[country, 2009]
            panel[country, 2009] = Observation(
                row.country, row.year, 1, row.outcome, ()
            )
        events, _, audit = audit_macro_matching(panel, controls)
        target = next(event for event in events if event.country == "AAA")
        self.assertEqual(target.status, "no_history_eligible_donors")
        self.assertEqual(audit["n_matched_events"], 0)

    def test_csv_reader_preserves_missing_and_rejects_nonfinite(self) -> None:
        panel, controls = fake_joined_panel()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "joined.csv"

            def write(negative: bool = False) -> None:
                with path.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(("country", "year", *CORE_COLUMNS))
                    for key, values in controls.items():
                        writer.writerow(
                            (
                                key[0],
                                key[1],
                                *(
                                    values
                                    if key != ("AAA", 2005)
                                    else (("nan" if negative else ""), 2.0, 3.0)
                                ),
                            )
                        )

            write()
            loaded = read_macro_controls(path, set(panel))
            self.assertIsNone(loaded["AAA", 2005][0])
            write(negative=True)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                read_macro_controls(path, set(panel))

    def test_writes_audit_with_event_and_balance_evidence(self) -> None:
        panel, controls = fake_joined_panel()
        events, balance, audit = audit_macro_matching(panel, controls)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            joined = directory / "joined.csv"
            joined.write_text("test fixture\n", encoding="utf-8")
            save_audit(directory, events, balance, audit, source_panel=joined)
            saved = json.loads((directory / "macro_balance_audit.json").read_text())
            self.assertEqual(saved["n_original_treated_events"], 1)
            self.assertEqual(len(saved["joined_panel_sha256"]), 64)
            self.assertIn(
                "BBB", (directory / "macro_event_eligibility.csv").read_text()
            )
            self.assertIn(
                CORE_COLUMNS[0], (directory / "macro_pretrend_balance.csv").read_text()
            )


if __name__ == "__main__":
    unittest.main()
