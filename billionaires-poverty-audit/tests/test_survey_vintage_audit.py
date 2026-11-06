"""Regression and input-contract tests for poverty survey-vintage sensitivity."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("survey_vintage_audit", ROOT / "survey_vintage_audit.py")
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SurveyVintageAuditTests(unittest.TestCase):
    """Verify sample accounting without suggesting causal identification."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = module.load_panel(module.PANEL)

    def test_frozen_panel_and_year_distribution(self) -> None:
        result = module.audit(self.frame)
        self.assertEqual(result["n"], 54)
        self.assertEqual(sum(x["countries"] for x in result["poverty_year_distribution"]), 54)
        self.assertLess(result["min_poverty_year"], result["max_poverty_year"])
        self.assertEqual(result["classification"], "exploratory_citizenship_not_causal")

    def test_threshold_samples_are_monotone_and_same_final_baseline(self) -> None:
        result = module.audit(self.frame)
        records = result["maximum_lag_restrictions"]
        self.assertEqual(records[-1]["n"], 54)
        self.assertEqual(records[-1]["excluded_iso3"], [])
        self.assertEqual(sorted(record["n"] for record in records),
                         [record["n"] for record in records])
        self.assertAlmostEqual(
            records[-1]["billionaire_coef_pp_per_log10"],
            result["baseline"]["coef"][module.BILLIONAIRE],
            places=9,
        )

    def test_rejects_future_poverty_year(self) -> None:
        corrupted = self.frame.copy()
        corrupted.loc[corrupted.index[0], "poverty_year"] = 2030
        with self.assertRaisesRegex(ValueError, "newer"):
            module.audit(corrupted)

    def test_rejects_invalid_minimum_and_missing_data(self) -> None:
        with self.assertRaises(ValueError):
            module.audit(self.frame, min_countries=True)
        corrupted = self.frame.copy()
        corrupted.loc[corrupted.index[0], "population"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            module.audit(corrupted)

    def test_output_refuses_to_replace_existing_evidence(self) -> None:
        result = module.audit(self.frame)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "vintage"
            module.write_results(result, output)
            self.assertTrue((output / "audit.json").is_file())
            self.assertTrue((output / "lag_sensitivity.csv").is_file())
            with self.assertRaises(FileExistsError):
                module.write_results(result, output)


if __name__ == "__main__":
    unittest.main()
