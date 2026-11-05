"""Reproducibility checks for the observed, non-causal 54-country study."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from observed_figures import (
    BILLIONAIRE, FROZEN_SHA256, N_COUNTRIES, analyse, fit, load_panel, render,
)

DATA = Path(__file__).resolve().parents[1] / "data/observed_citizenship_54_2026.csv"


class ExploratoryFigureTests(unittest.TestCase):
    """Guard against data edits, invalid attribution and model drift."""

    @classmethod
    def setUpClass(cls) -> None:
        """Read the pinned panel only once."""
        cls.frame = load_panel(DATA)

    def test_source_contract(self) -> None:
        """Use the exact data archived by the original CI run."""
        self.assertEqual(len(self.frame), N_COUNTRIES)
        self.assertEqual(self.frame.iso3.nunique(), N_COUNTRIES)
        self.assertTrue(self.frame.billionaire_method.eq("citizenship").all())
        self.assertEqual(len(FROZEN_SHA256), 64)

    def test_regression_reproduces_previous_ci(self) -> None:
        """Match the previously verified exploratory model within rounding."""
        models = analyse(self.frame)["models"]
        self.assertAlmostEqual(models["raw"]["r_squared"], 0.5471, places=4)
        self.assertAlmostEqual(models["gdp"]["r_squared"], 0.8049, places=4)
        self.assertAlmostEqual(models["joint"]["r_squared"], 0.8113, places=4)
        self.assertAlmostEqual(
            models["joint"]["coef"][BILLIONAIRE], -7.9118834291161315, places=6
        )

    def test_robustness_reports_different_samples(self) -> None:
        """A later observation cutoff is not the original 54-country model."""
        models = analyse(self.frame)["models"]
        self.assertEqual(models["poverty_2022_plus"]["n"], 43)
        self.assertEqual(models["poverty_2023_plus"]["n"], 34)

    def test_leave_one_out_complete(self) -> None:
        """All 54 country omissions must be recorded exactly once."""
        rows = analyse(self.frame)["leave_one_out"]
        self.assertEqual(len(rows), 54)
        self.assertEqual(len({r["excluded_iso3"] for r in rows}), 54)

    def test_population_weighting_differs(self) -> None:
        """Population weighting describes a different estimand."""
        models = analyse(self.frame)["models"]
        self.assertNotAlmostEqual(
            models["population_weighted"]["coef"][BILLIONAIRE],
            models["joint"]["coef"][BILLIONAIRE],
            places=3,
        )

    def test_bad_weights_rejected(self) -> None:
        """Do not silently align or impute invalid weights."""
        with self.assertRaises(ValueError):
            fit(self.frame, [BILLIONAIRE], weights=[1.0, 2.0])

    def test_sha_is_enforced(self) -> None:
        """A mutated source file must not be modelled."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mutated.csv"
            p.write_bytes(DATA.read_bytes().replace(b"Albania", b"Albanaa"))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_panel(p)

    def test_eight_publication_width_figures(self) -> None:
        """Ensure the full figure set regenerates and hashes are recorded."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "figures"
            records = render(self.frame, analyse(self.frame), path)
            self.assertEqual(len(records), 8)
            for figure in records.values():
                self.assertGreaterEqual(figure["width_px"], 2200)
                self.assertTrue((path / figure["png"]).is_file())
                self.assertTrue((path / figure["svg"]).is_file())

    def test_refuse_overwrite(self) -> None:
        """Preserve previous source-linked publication figures."""
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            (output / "existing.txt").write_text("reviewed")
            with self.assertRaises(FileExistsError):
                render(self.frame, analyse(self.frame), output)


if __name__ == "__main__":
    unittest.main()
