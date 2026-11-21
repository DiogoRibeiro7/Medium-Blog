"""Synthetic tests for the DPI2023 donor comparability audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.political_comparability import (
    DonorExposure,
    PoliticalYear,
    audit_comparability,
    load_donor_exposures,
    load_political_years,
    save_comparability,
)


def fixture() -> tuple[
    dict[tuple[str, int], PoliticalYear], list[DonorExposure]
]:
    """Build synthetic countries with known histories and donor censoring."""
    data: dict[tuple[str, int], PoliticalYear] = {}
    for country, orientation, checks, election in (
        ("AAA", "right", 2.0, 7.0),
        ("BBB", "right", 3.0, 7.0),
        ("CCC", "left", 5.0, 6.0),
        ("DDD", "right", 4.0, 7.0),
        ("EEE", "right", 2.0, 7.0),
    ):
        for year in range(2004, 2015):
            actual = (
                "left"
                if country == "DDD" and year == 2009
                else orientation
            )
            if country == "EEE" and year == 2007:
                actual = None
            data[country, year] = PoliticalYear(
                country, year, actual, checks, election
            )
    donors = [
        DonorExposure("AAA", 2010, "BBB", 3),
        DonorExposure("AAA", 2010, "CCC", None),
        DonorExposure("AAA", 2010, "DDD", 0),
        DonorExposure("AAA", 2010, "EEE", None),
    ]
    return data, donors


class ComparabilityTests(unittest.TestCase):
    """Check data leakage, categorical handling, and sample reconciliation."""

    def test_pre_reform_categories_and_censoring(self) -> None:
        years, donors = fixture()
        rows, horizons, report = audit_comparability(years, donors)
        detail = {row.donor: row for row in rows}
        self.assertTrue(detail["BBB"].same_orientation_t_minus_1)
        self.assertTrue(detail["BBB"].same_orientation_all_five_years)
        self.assertFalse(detail["CCC"].same_eiec_t_minus_1)
        self.assertFalse(detail["DDD"].same_orientation_all_five_years)
        self.assertEqual(detail["BBB"].abs_checks_five_year_mean_gap, 1.0)
        self.assertEqual(detail["CCC"].abs_checks_five_year_mean_gap, 3.0)
        self.assertFalse(detail["EEE"].both_full_pre5)
        self.assertIsNone(detail["EEE"].same_orientation_t_minus_1)
        self.assertEqual(
            [row.donors_at_risk for row in horizons],
            [4, 3, 3, 3, 2, 2, 2],
        )
        self.assertEqual(report["by_horizon"][0]["complete_treated_donor_pairs"], 3)
        self.assertEqual(report["by_horizon"][-1]["complete_treated_donor_pairs"], 1)
        self.assertEqual(report["by_horizon"][-1]["same_orientation_at_t_minus_1"], 0)

    def test_post_reform_data_cannot_affect_political_comparisons(self) -> None:
        years, donors = fixture()
        before = audit_comparability(years, donors)
        for country, year in list(years):
            if year >= 2010:
                years[country, year] = PoliticalYear(
                    country, year, None, None, None
                )
        self.assertEqual(before, audit_comparability(years, donors))

    def test_empty_donor_pool_is_rejected(self) -> None:
        years, _ = fixture()
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            audit_comparability(years, [])

    def test_orientation_missingness_and_duplicate_year(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "controls.csv"
            header = [
                "country", "year", "executive_orientation",
                "orientation_status", "checks", "electoral_competitiveness",
            ]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerow({
                    "country": "AAA", "year": "2007",
                    "executive_orientation": "", "orientation_status": "unknown",
                    "checks": "2.5", "electoral_competitiveness": "6.5",
                })
            self.assertIsNone(
                load_political_years(source)["AAA", 2007].executive_orientation
            )
            with source.open("a", encoding="utf-8") as handle:
                handle.write("AAA,2007,right,known,2,7\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_political_years(source)

    def test_duplicate_donors_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "donors.csv"
            source.write_text(
                "country,reform_year,donor,first_censored_horizon\n"
                "AAA,2010,BBB,3\n", encoding="utf-8"
            )
            self.assertEqual(
                load_donor_exposures(source)[0].first_censored_horizon, 3
            )
            with source.open("a", encoding="utf-8") as handle:
                handle.write("AAA,2010,BBB,5\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_donor_exposures(source)

    def test_source_manifest_preserves_original_risk_sets(self) -> None:
        years, donors = fixture()
        pairs, horizons, audit = audit_comparability(years, donors)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            controls = root / "controls.csv"
            donors_csv = root / "donors.csv"
            controls.write_text("synthetic\n", encoding="utf-8")
            donors_csv.write_text("synthetic\n", encoding="utf-8")
            source_audit = root / "audit.json"
            manifest = {
                "status": "pre_tax_cut_political_institutional_coverage_only",
                "n_reforms_baseline_eligible": 1,
                "n_pre_censor_donor_pairs": 4,
                "n_horizon_five_donor_pairs": 2,
                "source_csv_sha256": "a" * 64,
                "joined_panel_sha256": "b" * 64,
            }
            source_audit.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            output = root / "output"
            save_comparability(
                output, pairs, horizons, audit,
                political_controls=controls, donor_exposures=donors_csv,
                source_audit=source_audit,
            )
            actual = json.loads(
                (output / "political_comparability_audit.json").read_text()
            )
            self.assertEqual(actual["n_initial_pairs"], 4)
            self.assertEqual(actual["source_dpi_csv_sha256"], "a" * 64)
            self.assertTrue(
                (output / "political_comparability_pairs.csv").exists()
            )
            manifest["n_pre_censor_donor_pairs"] = 5
            source_audit.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                save_comparability(
                    output, pairs, horizons, audit,
                    political_controls=controls, donor_exposures=donors_csv,
                    source_audit=source_audit,
                )


if __name__ == "__main__":
    unittest.main()
