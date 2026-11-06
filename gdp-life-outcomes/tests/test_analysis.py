"""Unit and contract tests for the GDP–welfare descriptive study."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import GDP, GINI, LIFE, hc3_ols, last_reported, load_panel, wide_year


class RegressionTests(unittest.TestCase):
    """Check OLS numerics and the safeguards on inference."""

    def test_correct_linear_slope(self) -> None:
        x = np.linspace(-3, 3, 101)
        # Even residual pattern is orthogonal to the symmetric x grid.
        outcome = 12.0 + 3.75 * x + np.cos(x) / 30
        fit = hc3_ols(np.column_stack([np.ones(len(x)), x]), outcome)
        self.assertAlmostEqual(fit["parameters"][1], 3.75, delta=0.01)
        self.assertLess(fit["lower"][1], 3.75)
        self.assertGreater(fit["upper"][1], 3.75)

    def test_rejects_rank_deficiency(self) -> None:
        x = np.ones((10, 2))
        with self.assertRaises(ValueError):
            hc3_ols(x, np.arange(10, dtype=float))

    def test_rejects_nonfinite_input(self) -> None:
        x = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
        y = np.arange(10, dtype=float)
        y[4] = np.nan
        with self.assertRaises(ValueError):
            hc3_ols(x, y)


class DataContractTests(unittest.TestCase):
    """Protect against aggregate duplication and undisclosed year substitution."""

    def setUp(self) -> None:
        self.data = pd.DataFrame([
            {"country_code": "AAA", "country_name": "Alpha", "region": "R1",
             "income_group": "Upper", "year": 2018, "indicator_id": GINI, "value": 40.0},
            {"country_code": "AAA", "country_name": "Alpha", "region": "R1",
             "income_group": "Upper", "year": 2022, "indicator_id": GINI, "value": 39.0},
            {"country_code": "AAA", "country_name": "Alpha", "region": "R1",
             "income_group": "Upper", "year": 2023, "indicator_id": LIFE, "value": 80.0},
            {"country_code": "AAA", "country_name": "Alpha", "region": "R1",
             "income_group": "Upper", "year": 2023, "indicator_id": GDP, "value": 34000.0},
        ])

    def test_latest_vintage_is_explicit(self) -> None:
        latest = last_reported(self.data, GINI, 2018, 2023)
        self.assertEqual(int(latest.iloc[0][f"{GINI}_vintage"]), 2022)
        self.assertAlmostEqual(float(latest.iloc[0][GINI]), 39)

    def test_no_out_of_window_imputation(self) -> None:
        latest = last_reported(self.data, GINI, 2023, 2023)
        self.assertTrue(latest.empty)

    def test_one_country_per_pivot_row(self) -> None:
        wide = wide_year(self.data, 2023)
        self.assertEqual(len(wide), 1)
        self.assertAlmostEqual(float(wide.iloc[0][LIFE]), 80.0)

    def test_duplicate_observation_is_rejected(self) -> None:
        duplicated = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicates.csv"
            duplicated.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_panel(path)


if __name__ == "__main__":
    unittest.main()
