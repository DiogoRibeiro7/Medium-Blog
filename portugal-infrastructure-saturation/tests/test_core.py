"""Checks of the numerical tools against answers known by construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    composite,
    dimension_scores,
    gabaix_ibragimov,
    leave_group_out_r2,
    ols_hc1,
    rank_under_random_weights,
    spearman,
    weighted_median,
    winsor_z,
)
from fetch import content_hash  # noqa: E402
from panel import municipal  # noqa: E402


class TestCore(unittest.TestCase):
    def test_zipf_exponent_of_exact_pareto_quantiles(self):
        n = 2000
        rank = np.arange(1, n + 1)
        sizes = 1e6 / (rank - 0.5)  # rank - 1/2 proportional to size^-1, so zeta = 1
        zeta, se = gabaix_ibragimov(sizes, n)
        self.assertAlmostEqual(zeta, 1.0, places=6)
        self.assertAlmostEqual(se, np.sqrt(2 / n), places=6)

    def test_ols_recovers_noiseless_coefficients(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 3))
        y = 0.5 + X @ np.array([1.0, -2.0, 0.25])
        b, se, r2 = ols_hc1(y, X)
        np.testing.assert_allclose(b, [0.5, 1.0, -2.0, 0.25], atol=1e-10)
        self.assertAlmostEqual(r2, 1.0, places=10)
        self.assertTrue(np.all(se < 1e-8))

    def test_ols_hc1_matches_textbook_formula(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=500)
        y = 1 + 2 * x + rng.normal(size=500) * (1 + np.abs(x))
        b, se, _ = ols_hc1(y, x[:, None])
        Z = np.column_stack([np.ones(500), x])
        e = y - Z @ b
        inv = np.linalg.inv(Z.T @ Z)
        V = inv @ (Z.T @ np.diag(e**2) @ Z) @ inv * 500 / 498
        np.testing.assert_allclose(se, np.sqrt(np.diag(V)), rtol=1e-10)

    def test_winsor_z_is_standardised(self):
        x = pd.Series(np.r_[np.arange(100.0), 1e6])
        z = winsor_z(x)
        self.assertAlmostEqual(z.mean(), 0.0, places=10)
        self.assertAlmostEqual(z.std(ddof=0), 1.0, places=10)
        self.assertLess(z.max(), 3)  # the outlier was clipped before scaling

    def test_dimension_scores_ignore_missing_indicators(self):
        ind = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 2.0, 3.0]})
        s = dimension_scores(ind, {"d": ["a", "b"]})
        z = ind.apply(winsor_z)
        self.assertAlmostEqual(s.loc[0, "d"], z.loc[0, "a"])
        self.assertAlmostEqual(s.loc[2, "d"], (z.loc[2, "a"] + z.loc[2, "b"]) / 2)

    def test_dominant_unit_is_first_under_any_weights(self):
        rng = np.random.default_rng(3)
        scores = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
        scores.iloc[0] = 10.0
        r = rank_under_random_weights(scores, draws=500, seed=0)
        self.assertEqual(r.loc[0, "rank_p95"], 1)
        self.assertEqual(r.loc[0, "p_top30"], 1.0)
        self.assertAlmostEqual(composite(scores).loc[0], 10.0)

    def test_leave_group_out_r2_is_one_for_an_exact_linear_law(self):
        x = np.linspace(0, 1, 60)
        groups = np.repeat(np.arange(6), 10)
        self.assertAlmostEqual(leave_group_out_r2(3 * x + 1, x[:, None], groups), 1.0, places=10)

    def test_weighted_median_and_spearman(self):
        self.assertEqual(weighted_median(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 1.0, 10.0])), 3.0)
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(spearman(a, a**3), 1.0)


class TestData(unittest.TestCase):
    def test_content_hash_ignores_extraction_time(self):
        one = [{"IndicadorCod": "1", "DataUltimoAtualizacao": "2026-01-01", "DataExtracao": "a", "Dados": {"2021": [{"v": 1}]}}]
        two = [{"IndicadorCod": "1", "DataUltimoAtualizacao": "2026-01-01", "DataExtracao": "b", "Dados": {"2021": [{"v": 1}]}}]
        three = [{"IndicadorCod": "1", "DataUltimoAtualizacao": "2026-01-01", "DataExtracao": "a", "Dados": {"2021": [{"v": 2}]}}]
        self.assertEqual(content_hash(one), content_hash(two))
        self.assertNotEqual(content_hash(one), content_hash(three))

    def test_municipal_codes_are_reduced_to_dico(self):
        df = pd.DataFrame({"geocod": ["PT", "1A0", "1A01106", "1106", "110601", "1A0110601"], "valor": ["1"] * 6})
        out = municipal(df)
        self.assertEqual(list(out["dico"]), ["1106", "1106"])


if __name__ == "__main__":
    unittest.main()
