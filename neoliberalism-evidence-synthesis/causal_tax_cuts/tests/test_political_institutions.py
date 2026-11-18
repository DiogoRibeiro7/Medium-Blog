"""Synthetic source-contract and time-order tests for DPI2023 coverage."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from causal_tax_cuts.donor_support import audit_future_donor_exposure, save_support
from causal_tax_cuts.political_institutions import (
    DPI_TO_ISO3,
    PoliticalYear,
    audit_political_coverage,
    parse_dpi,
    save_political_audit,
)
from causal_tax_cuts.tests.test_donor_support import fixture


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal CSV with the actual DPI2023 field names/date format."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["countryname", "year", "execrlc", "checks", "eiec"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(
    label: str,
    year: str = "2005-01-01",
    orientation: str = "Right",
    checks: str = "3",
    electoral: str = "6.5",
) -> dict[str, str]:
    """Build one deliberately artificial DPI-style country-year."""
    return {
        "countryname": label,
        "year": year,
        "execrlc": orientation,
        "checks": checks,
        "eiec": electoral,
    }


def _synthetic_political() -> dict[tuple[str, int], PoliticalYear]:
    """Provide observed pre-reform controls for synthetic donor-support cases."""
    countries = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
    political: dict[tuple[str, int], PoliticalYear] = {}
    for country in countries:
        for year in range(2000, 2019):
            political[country, year] = PoliticalYear(
                country, year, "right", "known", 3.0, 6.5
            )
    # One donor's unknown class must not become a numeric political score.
    political["BBB", 2008] = PoliticalYear("BBB", 2008, None, "unknown", 3.0, 6.5)
    return political


class DpiSourceTests(unittest.TestCase):
    """Reject coding ambiguity rather than silently filling political data."""

    def test_exact_labels_and_three_missingness_classes(self) -> None:
        rows = [
            _row("FRG/Germany", orientation="Left"),
            _row("GDR"),
            _row("UK", orientation="Center"),
            _row("USA", orientation="0"),
            _row("Switzerland", orientation="-999"),
            _row("Canada", orientation="", checks="", electoral=""),
            _row("France", orientation="3.0"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            _write_source(source, rows)
            parsed, meta = parse_dpi(source, require_all_countries=False)
            self.assertEqual(meta["source_rows_read"], 7)
            self.assertEqual(meta["selected_country_years_1975_to_2015"], 6)
            self.assertEqual(
                meta["source_csv_sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        self.assertEqual(parsed["DEU", 2005].executive_orientation, "left")
        self.assertEqual(parsed["GBR", 2005].executive_orientation, "center")
        self.assertEqual(parsed["DEU", 2005].electoral_competitiveness, 6.5)
        self.assertNotIn(("GDR", 2005), parsed)
        self.assertEqual(parsed["USA", 2005].orientation_status, "unknown")
        self.assertEqual(parsed["CHE", 2005].orientation_status, "not_applicable")
        self.assertEqual(parsed["CAN", 2005].orientation_status, "missing")
        self.assertIsNone(parsed["CAN", 2005].checks)
        self.assertEqual(parsed["FRA", 2005].executive_orientation, "left")

    def test_country_mapping_is_bijective_for_paper_sample(self) -> None:
        self.assertEqual(len(DPI_TO_ISO3), 18)
        self.assertEqual(len(set(DPI_TO_ISO3.values())), 18)
        self.assertEqual(DPI_TO_ISO3["FRG/Germany"], "DEU")
        self.assertNotIn("GDR", DPI_TO_ISO3)

    def test_full_sample_is_required_for_live_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            _write_source(source, [_row("Australia")])
            with self.assertRaisesRegex(ValueError, "Missing country mapping"):
                parse_dpi(source)

    def test_duplicate_country_year_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            _write_source(source, [_row("Australia"), _row("Australia")])
            with self.assertRaisesRegex(ValueError, "Duplicate DPI country-year"):
                parse_dpi(source, require_all_countries=False)

    def test_non_january_snapshots_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            _write_source(source, [_row("Australia", year="2005-03-01")])
            with self.assertRaisesRegex(ValueError, "January-1 snapshot"):
                parse_dpi(source, require_all_countries=False)

    def test_unrecognized_orientation_and_invalid_numbers_rejected(self) -> None:
        invalid = [
            (_row("Australia", orientation="Conservative"), "orientation"),
            (_row("Australia", checks="nan"), "Non-finite"),
            (_row("Australia", checks="0"), "Impossible DPI checks"),
            (_row("Australia", electoral="8"), "Impossible DPI eiec"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            for rows, error in invalid:
                with self.subTest(error=error):
                    _write_source(source, [rows])
                    with self.assertRaisesRegex(ValueError, error):
                        parse_dpi(source, require_all_countries=False)


class PoliticalCoverageTests(unittest.TestCase):
    """Assert that adding controls cannot change the prior reform-risk sets."""

    def test_fixed_reform_cohort_and_horizon_five_donors(self) -> None:
        panel, macro = fixture()
        prior_events, prior_donors, _, prior = audit_future_donor_exposure(panel, macro)
        events, donors, result = audit_political_coverage(
            panel, macro, _synthetic_political()
        )
        self.assertEqual(
            result["n_reforms_baseline_eligible"],
            sum(row.status == "baseline_donors_available" for row in prior_events),
        )
        self.assertEqual(result["n_pre_censor_donor_pairs"], len(prior_donors))
        self.assertEqual(
            result["n_horizon_five_donor_pairs"],
            prior["by_horizon"][-1]["n_donor_reform_pairs_at_risk"],
        )
        self.assertTrue(result["reconciled_to_original_donor_risk_sets"])
        original = next(row for row in events if row.country == "AAA")
        self.assertTrue(original.treated_all_three_pre5_complete)
        self.assertEqual(original.n_pre_censor_donors, 5)
        self.assertEqual(original.n_horizon_five_donors, 3)
        unknown = next(
            row for row in donors if row.country == "AAA" and row.donor == "BBB"
        )
        self.assertFalse(unknown.donor_all_three_pre5_complete)
        self.assertEqual(unknown.n_donor_pre5_orientation, 4)
        self.assertEqual(unknown.n_donor_pre5_checks, 5)
        self.assertEqual(unknown.first_censored_horizon, 3)

    def test_no_treatment_or_post_treatment_covariate_leakage(self) -> None:
        panel, macro = fixture()
        political = _synthetic_political()
        before = audit_political_coverage(panel, macro, political)
        for key, row in tuple(political.items()):
            if key[1] >= 2010:
                political[key] = replace(
                    row, executive_orientation=None, orientation_status="unknown"
                )
        after = audit_political_coverage(panel, macro, political)
        self.assertEqual(
            [row for row in before[0] if row.country == "AAA"],
            [row for row in after[0] if row.country == "AAA"],
        )
        self.assertEqual(
            [row for row in before[1] if row.country == "AAA"],
            [row for row in after[1] if row.country == "AAA"],
        )

    def test_reject_wid_wdi_key_mismatch(self) -> None:
        panel, macro = fixture()
        macro.pop(("AAA", 2008))
        with self.assertRaisesRegex(ValueError, "identical"):
            audit_political_coverage(panel, macro, _synthetic_political())

    def test_manifest_and_early_years_are_auditable(self) -> None:
        panel, macro = fixture()
        political = _synthetic_political()
        events, donors, result = audit_political_coverage(panel, macro, political)
        prior_events, prior_donors, horizons, prior = audit_future_donor_exposure(
            panel, macro
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "joined.csv"
            source.write_text("synthetic joined fixture\n", encoding="utf-8")
            baseline_dir = directory / "baseline"
            save_support(
                baseline_dir,
                prior_events,
                prior_donors,
                horizons,
                prior,
                source_panel=source,
                panel=panel,
                controls=macro,
            )
            frozen = baseline_dir / "donor_support_audit.json"
            output = directory / "political"
            source_meta = {"source_csv_sha256": "0" * 64}
            save_political_audit(
                output,
                political,
                events,
                donors,
                result,
                source_meta,
                source_panel=source,
                frozen_reference=frozen,
            )
            report = json.loads((output / "political_coverage_audit.json").read_text())
            self.assertEqual(report["frozen_reference_status"], "same_source_validated")
            self.assertEqual(report["source_csv_sha256"], "0" * 64)
            self.assertNotIn("measured_at_utc", report)
            with (output / "dpi_political_controls.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                controls = list(csv.DictReader(handle))
            self.assertEqual(len(controls), 918)
            early = next(
                row
                for row in controls
                if row["country"] == "AUS" and row["year"] == "1965"
            )
            self.assertEqual(early["orientation_status"], "out_of_source_period")
            self.assertEqual(early["executive_orientation"], "")
            with (output / "political_donor_coverage.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(donors))

            # Same source SHA must never silently accept altered frozen cohorts.
            inconsistent = json.loads(frozen.read_text(encoding="utf-8"))
            inconsistent["n_pre_censor_donor_event_pairs"] += 1
            frozen.write_text(json.dumps(inconsistent), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen matching risk sets differ"):
                save_political_audit(
                    output,
                    political,
                    events,
                    donors,
                    result,
                    source_meta,
                    source_panel=source,
                    frozen_reference=frozen,
                )


if __name__ == "__main__":
    unittest.main()
