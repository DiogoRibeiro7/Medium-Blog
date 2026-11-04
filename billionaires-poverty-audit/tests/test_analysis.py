"""Deterministic unit tests for input validation and statistical invariants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import analyse, validate_frame  # noqa: E402


class ModelTests(unittest.TestCase):
    """Test mechanics only, never pass synthetic records off as empirical data."""

    @staticmethod
    def illustration() -> pd.DataFrame:
        """Create synthetic observations solely for unit testing."""
        rows: list[dict[str, object]] = []
        for i in range(12):
            rows.append({
                "country": f"Synthetic-{i}",
                "iso3": "X" + chr(65 + i // 26) + chr(65 + i % 26),
                "billionaires": i + 1,
                "population": 1_000_000 + i * 800_000,
                "poverty_pct": 78.0 - 5.0 * i + (i % 3) * 0.5,
                "gdp_ppp_per_capita": 1_700.0 + (i + 1) ** 2 * 1000,
                "poverty_year": 2023,
                "gdp_year": 2024,
                "billionaire_method": "residence",
            })
        return pd.DataFrame(rows)

    def test_valid_data(self) -> None:
        data, method = validate_frame(self.illustration())
        self.assertEqual(method, "residence")
        self.assertEqual(len(data), 12)

    def test_fit_has_same_sample_in_all_models(self) -> None:
        result = analyse(self.illustration())
        self.assertEqual(result["n_model_countries"], 12)
        self.assertEqual(
            {v["n"] for v in result["models"].values()}, {12}
        )

    def test_nested_model_r2_cannot_decline(self) -> None:
        result = analyse(self.illustration())
        self.assertGreaterEqual(
            result["incremental_r2_billionaires_conditional_on_gdp"], -1e-12
        )

    def test_zero_billionaires_excluded_and_counted(self) -> None:
        data = self.illustration()
        data.loc[0, "billionaires"] = 0
        result = analyse(data)
        self.assertEqual(result["n_zero_billionaire_countries_excluded_from_log_fit"], 1)
        self.assertEqual(result["n_model_countries"], 11)

    def test_reject_duplicate_country_codes(self) -> None:
        data = self.illustration()
        data.loc[1, "iso3"] = data.loc[0, "iso3"]
        with self.assertRaisesRegex(ValueError, "Each country ISO3"):
            validate_frame(data)

    def test_reject_mixed_country_attribution(self) -> None:
        data = self.illustration()
        data.loc[1, "billionaire_method"] = "citizenship"
        with self.assertRaisesRegex(ValueError, "one and only one"):
            validate_frame(data)

    def test_reject_negative_count(self) -> None:
        data = self.illustration()
        data.loc[0, "billionaires"] = -2
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_frame(data)

    def test_reject_zero_population(self) -> None:
        data = self.illustration()
        data.loc[0, "population"] = 0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            validate_frame(data)

    def test_reject_poverty_over_100(self) -> None:
        data = self.illustration()
        data.loc[0, "poverty_pct"] = 101.0
        with self.assertRaisesRegex(ValueError, "between zero and 100"):
            validate_frame(data)

    def test_reject_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Required columns"):
            validate_frame(self.illustration().drop(columns=["gdp_year"]))

    def test_reject_nonfinite_values(self) -> None:
        data = self.illustration()
        data.loc[0, "gdp_ppp_per_capita"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_frame(data)

    def test_reject_negative_gdp(self) -> None:
        data = self.illustration()
        data.loc[0, "gdp_ppp_per_capita"] = -1
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            validate_frame(data)

    def test_reject_no_density_variation(self) -> None:
        data = self.illustration()
        data["population"] = data["billionaires"] * 100_000
        with self.assertRaisesRegex(ValueError, "nonzero variation"):
            analyse(data)

    def test_raw_slope_is_not_a_causal_claim(self) -> None:
        result = analyse(self.illustration())
        self.assertIn("not causal", result["evidence"])


if __name__ == "__main__":
    unittest.main()
