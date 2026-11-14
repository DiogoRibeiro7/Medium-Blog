"""Offline validation and regression tests for the empirical evidence synthesis."""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from analysis import (
    COUNTRIES,
    DATA_DIR,
    DataIntegrityError,
    build_analysis,
    read_housing,
    read_labour_shares,
    read_salaries,
    read_us_history,
    serialize_panel,
    verify_snapshots,
    write_or_check_results,
)
from figures import render_all


class EvidenceSnapshotTests(unittest.TestCase):
    """Test numerical contracts, provenance and explicitly noncausal scope."""

    def test_manifest_snapshots_are_frozen(self) -> None:
        """Each local source has a separately recorded cryptographic fingerprint."""

        manifest = verify_snapshots()
        self.assertEqual(len(manifest["snapshots"]), 6)
        self.assertEqual(manifest["origin_repository"], "DiogoRibeiro7/Medium-Blog")

    def test_all_five_geographies_have_exactly_eleven_years(self) -> None:
        """The reported change must use the same reference year in every country."""

        self.assertEqual(len(read_housing()), 55)
        self.assertEqual(len(read_salaries()), 55)
        for geo in COUNTRIES:
            years = {year for (country, year) in read_housing() if country == geo}
            self.assertEqual(years, set(range(2015, 2026)))

    def test_fte_salary_has_the_declared_unit_and_country_set(self) -> None:
        """No median or other salary unit may silently enter the derived series."""

        with (DATA_DIR / "fte_salary_2015_2025.csv").open(
            newline="", encoding="utf-8"
        ) as h:
            observations = list(csv.DictReader(h))
        self.assertTrue(
            all(
                (r["dataset"], r["freq"], r["unit"]) == ("nama_10_fte", "A", "EUR")
                for r in observations
            )
        )
        self.assertEqual({r["geo"] for r in observations}, set(COUNTRIES))

    def test_2015_is_exactly_100_on_all_derived_indices(self) -> None:
        """Compare proportional changes, not unscaled national levels."""

        panel, _ = build_analysis()
        for row in (row for row in panel if row["year"] == "2015"):
            for key in (
                "house_price_index_2015_100",
                "nominal_fte_salary_index_2015_100",
                "house_price_to_fte_salary_index_2015_100",
                "official_price_income_index_2015_100",
            ):
                self.assertAlmostEqual(float(row[key]), 100.0, places=5)

    def test_price_to_salary_is_the_ratio_of_two_indices(self) -> None:
        """Check the exact accounting identity before six-decimal CSV rounding."""

        panel, summary = build_analysis()
        for row in panel:
            hpi = float(row["house_price_index_2015_100"])
            salary = float(row["nominal_fte_salary_index_2015_100"])
            ratio = float(row["house_price_to_fte_salary_index_2015_100"])
            self.assertAlmostEqual(ratio, 100.0 * hpi / salary, places=5)
        self.assertEqual(summary["status"], "descriptive_only_not_causal")

    def test_portugal_2025_anchors_follow_original_data(self) -> None:
        """Guard against a fabricated or unit-shifted Portugal headline number."""

        _, summary = build_analysis()
        pt = summary["housing_wage"]["endpoints"]["PT"]
        self.assertAlmostEqual(pt["nominal_house_price_growth_pct"], 163.847, places=2)
        self.assertAlmostEqual(pt["nominal_fte_salary_growth_pct"], 58.473, places=2)
        self.assertAlmostEqual(
            pt["house_price_to_fte_salary_growth_pct"], 66.493, places=2
        )
        self.assertAlmostEqual(pt["official_price_income_growth_pct"], 60.857, places=2)

    def test_official_income_ratio_is_not_salary_ratio(self) -> None:
        """Do not accidentally relabel gross average salaries as disposable income."""

        _, summary = build_analysis()
        for item in summary["housing_wage"]["endpoints"].values():
            self.assertNotEqual(
                item["official_price_income_growth_pct"],
                item["house_price_to_fte_salary_growth_pct"],
            )

    def test_labour_share_definitions_disagree_about_endpoint_direction(self) -> None:
        """Raw and self-employment-adjusted shares need not share a sign."""

        labour = read_labour_shares()
        self.assertEqual(len(labour), 18 * 30)
        raw = (
            labour[(2024, "raw", "market_prices")]
            - labour[(1995, "raw", "market_prices")]
        )
        adjusted = (
            labour[(2024, "employment", "market_prices")]
            - labour[(1995, "employment", "market_prices")]
        )
        self.assertGreater(raw, 0)
        self.assertLess(adjusted, 0)

    def test_us_series_have_distinct_coverage_and_are_not_merged_by_position(
        self,
    ) -> None:
        """Census Gini starts in 1967, while tax-rate and GDP snapshots start in 1964."""

        us = read_us_history()
        self.assertEqual((min(us["tax"]), max(us["tax"])), (1964, 2018))
        self.assertEqual((min(us["gini"]), max(us["gini"])), (1967, 2018))
        self.assertEqual(us["tax"][1980][0], 70)
        self.assertEqual(us["tax"][1988][0], 28)
        self.assertNotEqual(us["gini"][1992][0], us["gini"][1993][0])

    def test_committed_results_regenerate_byte_for_byte(self) -> None:
        """Analytical outputs must be deterministic on any supported machine."""

        report = write_or_check_results(check=True)
        self.assertIn("PT", report["housing_wage"]["endpoints"])
        self.assertEqual(len(serialize_panel(build_analysis()[0]).splitlines()), 56)

    def test_source_hash_tampering_is_rejected(self) -> None:
        """An accidental snapshot edit must not silently change published graphs."""

        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)
            for source in DATA_DIR.iterdir():
                shutil.copy2(source, local / source.name)
            path = local / "us_top_income_tax_rate_1964_2018.csv"
            path.write_text(
                path.read_text(encoding="utf-8").replace("1988,28", "1988,29"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataIntegrityError, "hash changed"):
                verify_snapshots(local)

    def test_missing_salary_year_is_rejected_even_without_manifest(self) -> None:
        """Reject unbalanced panels rather than interpolate or forward-fill."""

        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)
            src = DATA_DIR / "fte_salary_2015_2025.csv"
            lines = src.read_text(encoding="utf-8").splitlines()
            keep = [line for line in lines if ",PT,2020," not in line]
            (local / src.name).write_text("\n".join(keep) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DataIntegrityError, "Incomplete salary panel"):
                read_salaries(local)

    def test_nonfinite_house_price_is_rejected(self) -> None:
        """NaN cannot enter an index and quietly contaminate calculated growth."""

        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)
            src = DATA_DIR / "housing_indices_2015_2025.csv"
            rows = src.read_text(encoding="utf-8").splitlines()
            parts = rows[1].split(",")
            parts[2] = "NaN"
            rows[1] = ",".join(parts)
            (local / src.name).write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DataIntegrityError, "Invalid HPI"):
                read_housing(local)

    def test_figures_are_renderable_publication_pngs(self) -> None:
        """Generate real, correctly encoded figures from data, not placeholder images."""

        with tempfile.TemporaryDirectory() as folder:
            files = render_all(output_dir=Path(folder))
            self.assertEqual(len(files), 5)
            for png in files:
                with self.subTest(figure=png.name):
                    self.assertGreater(png.stat().st_size, 30_000)
                    self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
