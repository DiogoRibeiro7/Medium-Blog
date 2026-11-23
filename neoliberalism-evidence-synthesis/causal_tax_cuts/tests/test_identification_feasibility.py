"""Synthetic invariants for the policy-specific identification feasibility gate."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from causal_tax_cuts.identification_feasibility import (
    HORIZONS,
    NOT_IDENTIFIED,
    evaluate_identification,
    normalized_scientific_csv_digest,
    run,
    verify_frozen_reference,
)


def synthetic() -> dict[str, list[dict[str, str]]]:
    """Create two completely fictional reform events and one continuation.

    AAA has legal context and early geometric support that disappears by h=5.
    BBB has no legal context and maintains geometric support through h=5.
    Neither has identified policy timing or a causal counterfactual.
    """
    legal = [
        {
            "country": "AAA",
            "author_year": "2001",
            "is_onset": "True",
            "baseline_donor_eligible_onset": "True",
            "legal_context_status": "provision_period_context_only",
            "author_composite_effective_date_verified": "False",
            "causal_estimand_identified": "False",
        },
        {
            "country": "AAA",
            "author_year": "2002",
            "is_onset": "False",
            "baseline_donor_eligible_onset": "False",
            "legal_context_status": "unreviewed",
            "author_composite_effective_date_verified": "False",
            "causal_estimand_identified": "False",
        },
        {
            "country": "BBB",
            "author_year": "2003",
            "is_onset": "True",
            "baseline_donor_eligible_onset": "True",
            "legal_context_status": "unreviewed",
            "author_composite_effective_date_verified": "False",
            "causal_estimand_identified": "False",
        },
    ]
    hull: list[dict[str, str]] = []
    dynamic: list[dict[str, str]] = []
    for country, year in (("AAA", 2001), ("BBB", 2003)):
        for h in HORIZONS:
            n = 3 if h < 4 else 2
            strict = country == "BBB" or h < 4
            hull.append(
                {
                    "country": country,
                    "reform_year": str(year),
                    "horizon": str(h),
                    "n_surviving_donors": str(n),
                    "strict_joint_hull_support": str(strict),
                    "within_all_three_marginal_ranges": str(True),
                }
            )
            if h >= 0:
                dynamic.append(
                    {
                        "country": country,
                        "reform_year": str(year),
                        "horizon": str(h),
                        "n_no_cut_comparators_through_h": str(n),
                        "treated_additional_cuts_through_h": str(
                            int(country == "AAA" and h >= 1)
                        ),
                    }
                )
    political = [
        {
            "country": "AAA",
            "reform_year": "2001",
            "treated_all_three_pre5_complete": "False",
            "n_horizon_five_donors_all_three_complete": "0",
        },
        {
            "country": "BBB",
            "reform_year": "2003",
            "treated_all_three_pre5_complete": "True",
            "n_horizon_five_donors_all_three_complete": "1",
        },
    ]
    macro = [
        {
            "country": "AAA",
            "reform_year": "2001",
            "status": "matched",
            "pretrend_slope_gap_selected": "-0.12",
        },
        {
            "country": "BBB",
            "reform_year": "2003",
            "status": "matched",
            "pretrend_slope_gap_selected": "0.24",
        },
    ]
    return {
        "legal": legal,
        "hull": hull,
        "dynamic": dynamic,
        "political": political,
        "macro": macro,
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Persist synthetic source rows to a consistent UTF-8 CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest(path: Path, status: str, **other: object) -> None:
    """Create an explicitly artificial provenance audit."""
    data = {"status": status, "joined_panel_sha256": "a" * 64, **other}
    path.write_text(json.dumps(data), encoding="utf-8")


class IdentificationFeasibilityTests(unittest.TestCase):
    """No effect estimates may be inferred from these descriptive gates."""

    def test_early_support_does_not_imply_five_year_support(self) -> None:
        rows = synthetic()
        events, horizons, summary = evaluate_identification(
            rows["legal"],
            rows["hull"],
            rows["dynamic"],
            rows["political"],
            rows["macro"],
            expected_reforms=2,
        )
        documented = next(row for row in events if row.country == "AAA")
        undocumented = next(row for row in events if row.country == "BBB")
        self.assertTrue(documented.provision_context_reviewed)
        self.assertTrue(documented.strict_joint_wdi_support_at_h3)
        self.assertFalse(documented.strict_joint_wdi_support_at_h5)
        self.assertTrue(documented.additional_major_cut_through_h5)
        self.assertFalse(documented.political_pre5_history_complete)
        self.assertTrue(undocumented.strict_joint_wdi_support_at_h5)
        self.assertFalse(undocumented.provision_context_reviewed)
        self.assertEqual(summary["h5_support_without_legal_context"], ["BBB-2003"])
        self.assertEqual(
            summary["n_documented_eligible_with_h5_strict_joint_wdi_support"], 0
        )
        self.assertEqual(summary["n_eligible_with_repeat_cut_by_h5"], 1)
        self.assertEqual(summary["n_policy_specific_causal_designs_supported"], 0)
        self.assertTrue(all(row.feasibility_status == NOT_IDENTIFIED for row in events))
        self.assertEqual(len(horizons), 2 * len(HORIZONS))
        self.assertIsNone(horizons[0].repeated_initiator_cut_by_horizon)

    def test_refuses_documentary_flag_attribution_not_in_original_source(self) -> None:
        rows = synthetic()
        rows["legal"][0]["author_composite_effective_date_verified"] = "True"
        with self.assertRaisesRegex(ValueError, "does not identify"):
            evaluate_identification(
                rows["legal"],
                rows["hull"],
                rows["dynamic"],
                rows["political"],
                rows["macro"],
                expected_reforms=2,
            )

    def test_donor_counts_must_agree_across_horizon_sources(self) -> None:
        rows = synthetic()
        next(row for row in rows["dynamic"] if row["horizon"] == "5")[
            "n_no_cut_comparators_through_h"
        ] = "999"
        with self.assertRaisesRegex(ValueError, "Contradictory dynamic"):
            evaluate_identification(
                rows["legal"],
                rows["hull"],
                rows["dynamic"],
                rows["political"],
                rows["macro"],
                expected_reforms=2,
            )

    def test_no_missing_horizon_or_duplicate_source_key(self) -> None:
        rows = synthetic()
        rows["hull"].pop()
        with self.assertRaisesRegex(ValueError, "Missing joint-support horizon"):
            evaluate_identification(
                rows["legal"],
                rows["hull"],
                rows["dynamic"],
                rows["political"],
                rows["macro"],
                expected_reforms=2,
            )
        rows = synthetic()
        rows["hull"].append(deepcopy(rows["hull"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            evaluate_identification(
                rows["legal"],
                rows["hull"],
                rows["dynamic"],
                rows["political"],
                rows["macro"],
                expected_reforms=2,
            )

    def test_political_coverage_is_not_equivalent_to_causal_balance(self) -> None:
        rows = synthetic()
        rows["political"][0]["treated_all_three_pre5_complete"] = "True"
        rows["political"][0]["n_horizon_five_donors_all_three_complete"] = "2"
        events, _, summary = evaluate_identification(
            rows["legal"],
            rows["hull"],
            rows["dynamic"],
            rows["political"],
            rows["macro"],
            expected_reforms=2,
        )
        self.assertTrue(events[0].political_pre5_history_complete)
        self.assertFalse(events[0].policy_specific_causal_design_supported)
        self.assertEqual(
            summary["n_policy_specific_interventions_attributed_to_source_law"], 0
        )

    def test_macro_slope_is_diagnostic_not_a_screening_threshold(self) -> None:
        rows = synthetic()
        rows["macro"][0]["pretrend_slope_gap_selected"] = "20.5"
        result, _, _ = evaluate_identification(
            rows["legal"],
            rows["hull"],
            rows["dynamic"],
            rows["political"],
            rows["macro"],
            expected_reforms=2,
        )
        self.assertEqual(result[0].pretrend_slope_gap_selected_pp_per_year, 20.5)
        self.assertEqual(result[0].feasibility_status, NOT_IDENTIFIED)
        rows["macro"][0]["pretrend_slope_gap_selected"] = "nan"
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            evaluate_identification(
                rows["legal"],
                rows["hull"],
                rows["dynamic"],
                rows["political"],
                rows["macro"],
                expected_reforms=2,
            )

    def test_source_vintage_mismatch_fails_closed(self) -> None:
        rows = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_paths = {}
            for key, values in rows.items():
                path = root / f"{key}.csv"
                write_rows(path, values)
                data_paths[key] = path
            audits = {}
            status_map = {
                "legal": "legislation_context_coverage_and_nonidentification_only",
                "hull": "three_wdi_dimension_joint_convex_hull_diagnostic_only",
                "dynamic": "dynamic_policy_pathway_and_covariate_availability_audit_only",
                "political": "pre_tax_cut_political_institutional_coverage_only",
                "macro": "pre_reform_support_and_balance_only_no_adjusted_effects",
            }
            for key, status in status_map.items():
                path = root / f"{key}.json"
                manifest(path, status)
                audits[key] = path
            audits["political"].write_text(
                json.dumps(
                    {
                        "status": status_map["political"],
                        "joined_panel_sha256": "b" * 64,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Incompatible WID/WDI"):
                run(
                    legal_file=data_paths["legal"],
                    hull_file=data_paths["hull"],
                    dynamic_file=data_paths["dynamic"],
                    political_file=data_paths["political"],
                    macro_file=data_paths["macro"],
                    legal_audit=audits["legal"],
                    hull_audit=audits["hull"],
                    dynamic_audit=audits["dynamic"],
                    political_audit=audits["political"],
                    macro_audit=audits["macro"],
                    output_dir=root / "out",
                )

    def test_requires_unaltered_frozen_eligible_event_count(self) -> None:
        rows = synthetic()
        with self.assertRaisesRegex(ValueError, "Eligible cohort changed"):
            evaluate_identification(
                rows["legal"],
                rows["hull"],
                rows["dynamic"],
                rows["political"],
                rows["macro"],
                expected_reforms=20,
            )

    def test_bool_parser_rejects_implicit_truthiness(self) -> None:
        rows = synthetic()
        rows["legal"][0]["is_onset"] = "false"
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            evaluate_identification(
                rows["legal"],
                rows["hull"],
                rows["dynamic"],
                rows["political"],
                rows["macro"],
                expected_reforms=2,
            )


    def test_normalized_csv_hash_only_ignores_machine_roundoff(self) -> None:
        """Distinguish 1e-16 binary-float noise from meaningful changes."""
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "derived.csv"
            sample.write_text(
                "country,mean_selected_distance\nAAA,2.1461024177280827\n",
                encoding="utf-8",
            )
            expected = normalized_scientific_csv_digest(
                sample, "mean_selected_distance"
            )
            sample.write_text(
                "country,mean_selected_distance\nAAA,2.146102417728083\n",
                encoding="utf-8",
            )
            self.assertEqual(
                normalized_scientific_csv_digest(sample, "mean_selected_distance"),
                expected,
            )
            sample.write_text(
                "country,mean_selected_distance\nAAA,2.146103417728083\n",
                encoding="utf-8",
            )
            self.assertNotEqual(
                normalized_scientific_csv_digest(sample, "mean_selected_distance"),
                expected,
            )

    def test_frozen_verification_requires_same_scientific_data(self) -> None:
        """Keep changed original sources or substantive results distinguishable."""
        baseline = {
            "joined_panel_sha256": "a" * 64,
            "author_source_sha256": "b" * 64,
            "status": "policy_specific_identification_feasibility_only",
            "n_original_eligible_reform_onsets": 2,
            "input_sha256": {
                "joint_wdi_hull": "1" * 64,
                "macro_matching_pretrends": "2" * 64,
                "legal_register": "3" * 64,
            },
            "normalized_input_sha256": {
                "joint_wdi_hull": "4" * 64,
                "macro_matching_pretrends": "5" * 64,
            },
        }
        self.assertEqual(
            verify_frozen_reference(baseline, deepcopy(baseline)),
            "exact_raw_input_reproduction",
        )
        drift = deepcopy(baseline)
        drift["input_sha256"]["joint_wdi_hull"] = "0" * 64
        self.assertEqual(
            verify_frozen_reference(drift, baseline),
            "same_original_data_and_scientific_inputs_derived_roundoff_only",
        )
        bad_science = deepcopy(drift)
        bad_science["normalized_input_sha256"]["joint_wdi_hull"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "normalized scientific inputs"):
            verify_frozen_reference(bad_science, baseline)
        bad_summary = deepcopy(drift)
        bad_summary["n_original_eligible_reform_onsets"] = 3
        with self.assertRaisesRegex(ValueError, "results or normalized"):
            verify_frozen_reference(bad_summary, baseline)
        unexpected = deepcopy(drift)
        unexpected["input_sha256"]["legal_register"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "Unexpected raw input"):
            verify_frozen_reference(unexpected, baseline)
        new_vintage = deepcopy(drift)
        new_vintage["joined_panel_sha256"] = "9" * 64
        self.assertEqual(
            verify_frozen_reference(new_vintage, baseline),
            "original_source_vintage_changed",
        )


if __name__ == "__main__":
    unittest.main()
