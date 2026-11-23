"""Synthetic tests for Finland/Germany source-first policy-component gates."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.policy_component_candidates import (
    assess_candidates,
    load_components,
    save_audit,
)

OFFICIAL = "https://www.finlex.fi/fi/lainsaadanto/1985/997"
GERMAN = "https://dserver.bundestag.de/gm/16/160221.pdf"
BANK = "https://publications.bof.fi/items/005ef29d-d4bc-405a-879d-b3611dde7332"
MINISTRY = (
    "https://www.bundesfinanzministerium.de/Monatsberichte/"
    "2001-2016/Inhalte/Monatsbericht-Archiv-Downloads/2008/"
    "Monatsbericht_Mai_2008.pdf?__blob=publicationFile&v=1"
)


def component(
    country: str, year: int, name: str, kind: str,
    before: str, after: str, start: str,
) -> dict[str, str]:
    """Build one fictional, fully labelled source record."""
    return {
        "country": country,
        "flag_year": str(year),
        "component_id": name,
        "instrument": "Synthetic measured tax instrument",
        "change_kind": kind,
        "value_before": before,
        "value_after": after,
        "unit": "percent",
        "law_title": "Fictional tax statute",
        "law_identifier": "Section 1",
        "law_enacted_on": "" if country == "FIN" else "2007-08-14",
        "period_start": start,
        "period_scope": "Fictional taxpayer-year applicability",
        "cash_flow_caveat": "Advance payments retained the old rate",
        "evidence_class": (
            "statute_index_and_contemporaneous_ministry_report"
            if country == "FIN"
            else "parliamentary_law_dossier_and_federal_finance_ministry"
        ),
        "official_source_url": OFFICIAL if country == "FIN" else GERMAN,
        "corroborating_source_url": BANK if country == "FIN" else MINISTRY,
        "composite_flag_attribution": "unverified",
    }


def input_rows() -> list[dict[str, str]]:
    """Two cases, a transition and a later effective-year step."""
    return [
        component("FIN", 1986, "rate", "rate_reduction", "43", "33", "1986-01-01"),
        component(
            "FIN", 1986, "advance", "advance_payment_transition",
            "43", "43", "1986-01-01",
        ),
        component(
            "DEU", 2008, "rate", "rate_reduction",
            "25", "15", "2008-01-01",
        ),
        component(
            "DEU", 2008, "later", "later_capital_income_withholding",
            "", "25", "2009-01-01",
        ),
    ]


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    """Write source-contract CSV rows, including explicit blank values."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def historical() -> tuple[list[dict[str, str]], dict[str, object]]:
    """Fabricate exactly 20 prior eligibility cases for contract tests."""
    base: list[dict[str, str]] = []
    for index in range(18):
        base.append(
            {
                "country": f"X{index:02}",
                "reform_year": str(1980 + index),
                "provision_context_reviewed": "True" if index < 5 else "False",
                "strict_joint_wdi_support_at_h5": "False",
                "n_surviving_donors_at_h5": "2",
                "n_h5_donors_with_complete_political_history": "1",
                "pretrend_slope_gap_selected_pp_per_year": "0.15",
                "additional_major_cut_through_h5": "False",
                "policy_specific_causal_design_supported": "False",
                "feasibility_status": "not_identified_with_current_evidence",
            }
        )
    for country, year, donors, political, gap in (
        ("FIN", 1986, 8, 7, -0.135),
        ("DEU", 2008, 16, 15, 0.496),
    ):
        base.append(
            {
                "country": country,
                "reform_year": str(year),
                "provision_context_reviewed": "False",
                "strict_joint_wdi_support_at_h5": "True",
                "n_surviving_donors_at_h5": str(donors),
                "n_h5_donors_with_complete_political_history": str(political),
                "pretrend_slope_gap_selected_pp_per_year": str(gap),
                "additional_major_cut_through_h5": "False",
                "policy_specific_causal_design_supported": "False",
                "feasibility_status": "not_identified_with_current_evidence",
            }
        )
    prior: dict[str, object] = {
        "status": "policy_specific_identification_feasibility_only",
        "n_original_author_flag_years": 30,
        "n_original_eligible_reform_onsets": 20,
        "n_eligible_onsets_with_statute_provision_context": 5,
        "n_eligible_with_h5_strict_joint_wdi_support": 2,
        "h5_support_without_legal_context": ["FIN-1986", "DEU-2008"],
        "joined_panel_sha256": "0" * 64,
    }
    return base, prior


class PolicyComponentTests(unittest.TestCase):
    """Ensure a documented tax rule never becomes a causal effect by fiat."""

    def _load(self, root: Path, rows: list[dict[str, str]] | None = None):
        path = root / "components.csv"
        write_csv(path, input_rows() if rows is None else rows)
        return load_components(path), path

    def test_component_source_time_and_missing_date_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records, _ = self._load(Path(temporary))
            self.assertEqual(len(records), 4)
            fin = [r for r in records if r.country == "FIN"]
            deu = [r for r in records if r.country == "DEU"]
            self.assertTrue(all(r.law_enacted_on is None for r in fin))
            self.assertEqual(
                next(r.period_start for r in deu if r.component_id == "later"),
                "2009-01-01",
            )
            self.assertTrue(all(r.composite_flag_attribution == "unverified" for r in records))

    def test_positive_and_negative_rate_changes_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = input_rows()
            rows[0]["value_after"] = "53"
            write_csv(root / "components.csv", rows)
            with self.assertRaisesRegex(ValueError, "before > after"):
                load_components(root / "components.csv")
            rows = input_rows()
            rows[1]["value_after"] = "33"
            write_csv(root / "components.csv", rows)
            with self.assertRaisesRegex(ValueError, "preserve the old rate"):
                load_components(root / "components.csv")

    def test_unofficial_url_or_verified_mapping_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = input_rows()
            rows[0]["official_source_url"] = "https://not-government.example/act"
            with self.assertRaisesRegex(ValueError, "Unapproved source"):
                self._load(root, rows)
            rows = input_rows()
            rows[0]["composite_flag_attribution"] = "verified"
            with self.assertRaisesRegex(ValueError, "attribution is verified"):
                self._load(root, rows)

    def test_delayed_applicability_is_not_reassigned_to_author_year(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = input_rows()
            rows[-1]["period_start"] = "2010-01-01"
            with self.assertRaisesRegex(ValueError, "outside the declared flag"):
                self._load(root, rows)
            rows = input_rows()
            rows[-1]["law_enacted_on"] = "2009-02-01"
            with self.assertRaisesRegex(ValueError, "Enactment date follows"):
                self._load(root, rows)

    def test_duplicate_component_or_missing_country_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = input_rows()
            with self.assertRaisesRegex(ValueError, "duplicate policy component"):
                self._load(root, [*rows, rows[0]])
            with self.assertRaisesRegex(ValueError, "Both original"):
                self._load(root, rows[:2])

    def test_frozen_20_event_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records, _ = self._load(root)
            old, manifest = historical()
            candidates, summary = assess_candidates(records, old, manifest)
            self.assertEqual([(x.country, x.reform_year) for x in candidates],
                             [("FIN", 1986), ("DEU", 2008)])
            self.assertEqual(
                [x.original_h5_donors for x in candidates], [8, 16]
            )
            self.assertEqual(
                [x.original_pretrend_slope_gap_pp_per_year for x in candidates],
                [-0.135, 0.496],
            )
            self.assertTrue(candidates[0].has_advance_payment_transition)
            self.assertEqual(candidates[1].n_components_with_later_tax_year, 1)
            self.assertEqual(summary["n_eligible_onsets_with_statutory_or_contemporaneous_context"], 7)
            self.assertEqual(summary["n_policy_specific_causal_estimands_identified"], 0)
            self.assertFalse(any(row.policy_specific_effect_identified for row in candidates))

    def test_changed_original_cohort_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records, _ = self._load(Path(temporary))
            rows, manifest = historical()
            rows.pop()
            with self.assertRaisesRegex(ValueError, "Original event sample changed"):
                assess_candidates(records, rows, manifest)
            rows, manifest = historical()
            manifest["n_eligible_with_h5_strict_joint_wdi_support"] = 3
            with self.assertRaisesRegex(ValueError, "not the frozen"):
                assess_candidates(records, rows, manifest)

    def test_case_without_joint_hull_support_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records, _ = self._load(Path(temporary))
            rows, manifest = historical()
            next(r for r in rows if r["country"] == "FIN")[
                "strict_joint_wdi_support_at_h5"
            ] = "False"
            with self.assertRaisesRegex(ValueError, "no longer has original"):
                assess_candidates(records, rows, manifest)

    def test_deterministic_json_and_full_precision_prior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records, components_file = self._load(root)
            rows, prior = historical()
            candidates, summary = assess_candidates(records, rows, prior)
            old_csv = root / "prior.csv"
            old_json = root / "prior.json"
            write_csv(old_csv, rows)
            old_json.write_text(json.dumps(prior), encoding="utf-8")
            output = root / "results"
            a = save_audit(
                output, records, candidates, summary,
                components_file=components_file,
                previous_csv=old_csv, previous_json=old_json,
            )
            previous = (output / "policy_component_audit.json").read_bytes()
            b = save_audit(
                output, records, candidates, summary,
                components_file=components_file,
                previous_csv=old_csv, previous_json=old_json,
            )
            self.assertEqual(a, b)
            self.assertEqual(
                (output / "policy_component_audit.json").read_bytes(), previous
            )
            self.assertEqual(
                len(list(csv.DictReader(
                    (output / "policy_component_evidence.csv").open(
                        encoding="utf-8", newline=""
                    )
                ))),
                4,
            )
            self.assertIn("source_sha256", a)


if __name__ == "__main__":
    unittest.main()
