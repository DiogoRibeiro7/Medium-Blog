"""Checks of the numerical tools against answers known by construction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import panel  # noqa: E402
from core import demean, fixed_effects, log_growth, ols_hc1, percentile_of, ring, spearman  # noqa: E402
from fetch import ine_content_hash  # noqa: E402


class TestCore(unittest.TestCase):
    def test_log_growth_of_a_constant_rate(self):
        g = log_growth(pd.Series([100.0]), pd.Series([100.0 * np.exp(0.06)]), pd.Series([6]))
        self.assertAlmostEqual(g.iloc[0], 1.0)

    def test_ring_is_the_fua_outside_the_city(self):
        self.assertEqual(ring(pd.Series([1000.0]), pd.Series([400.0])).iloc[0], 600.0)

    def test_demean_removes_group_means(self):
        s = pd.Series([1.0, 3.0, 10.0, 14.0])
        out = demean(s, pd.Series(["a", "a", "b", "b"]))
        np.testing.assert_allclose(out.to_numpy(), [-1.0, 1.0, -2.0, 2.0])

    def test_ols_recovers_an_exact_line(self):
        x = np.arange(10.0)
        beta, se, r2 = ols_hc1(2.0 + 0.5 * x, x)
        np.testing.assert_allclose(beta, [2.0, 0.5], atol=1e-12)
        self.assertAlmostEqual(r2, 1.0)
        np.testing.assert_allclose(se, [0.0, 0.0], atol=1e-10)

    def test_hc1_matches_the_sandwich_by_hand(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        y = 1.0 + 2.0 * x + rng.normal(size=200) * (1 + np.abs(x))
        beta, se, _ = ols_hc1(y, x)
        Z = np.column_stack([np.ones(200), x])
        u = y - Z @ beta
        bread = np.linalg.inv(Z.T @ Z)
        meat = sum(np.outer(z, z) * e**2 for z, e in zip(Z, u, strict=True))
        np.testing.assert_allclose(se, np.sqrt(np.diag(bread @ meat @ bread * 200 / 198)))

    def test_fixed_effects_ignore_group_intercepts(self):
        rng = np.random.default_rng(7)
        groups = pd.Series(np.repeat(list("abcde"), 40))
        x = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
        level = groups.map({"a": -5.0, "b": 0.0, "c": 3.0, "d": 8.0, "e": 20.0})
        y = level + 0.3 * x.x1 - 1.2 * x.x2
        beta, _, r2, g = fixed_effects(y, x, groups)
        np.testing.assert_allclose(beta, [0.3, -1.2], atol=1e-10)
        self.assertAlmostEqual(r2, 1.0)
        self.assertEqual(g, 5)

    def test_spearman_is_one_for_monotone_data(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
        self.assertAlmostEqual(spearman(a, a**3), 1.0)
        self.assertAlmostEqual(spearman(a, -a), -1.0)

    def test_percentile_counts_values_strictly_below(self):
        v = pd.Series([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(percentile_of(v, 3.0), 50.0)
        self.assertEqual(percentile_of(v, 9.0), 100.0)

    def test_ine_hash_ignores_the_extraction_time(self):
        a = [{"IndicadorCod": "0011654", "DataUltimoAtualizacao": "2023-06-01", "DataExtracao": "2026-11-23T10:00", "Dados": {"2021": [1]}}]
        b = [dict(a[0], DataExtracao="2026-11-24T09:00")]
        c = [dict(a[0], Dados={"2021": [2]})]
        self.assertEqual(ine_content_hash(a), ine_content_hash(b))
        self.assertNotEqual(ine_content_hash(a), ine_content_hash(c))


class TestPanel(unittest.TestCase):
    def test_jsonstat_unpacks_row_major_positions_and_keeps_only_city_codes(self):
        doc = {
            "id": ["freq", "indic_ur", "cities", "time"],
            "size": [1, 1, 3, 2],
            "dimension": {
                "freq": {"category": {"index": {"A": 0}}},
                "indic_ur": {"category": {"index": {"DE1001V": 0}}},
                "cities": {
                    "category": {"index": {"PT": 0, "PT001C": 1, "PT001T": 2}, "label": {"PT": "Portugal", "PT001C": "Lisboa", "PT001T": "x"}}
                },
                "time": {"category": {"index": {"2015": 0, "2021": 1}}},
            },
            # flat position = city * 2 + year; position 0 and 1 are the country total
            "value": {"0": 1.0, "2": 10.0, "3": 11.0, "4": 99.0, "5": 98.0},
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(panel, "RAW", Path(tmp)):
            (Path(tmp) / "eu_pop_city.json").write_text(json.dumps(doc))
            df = panel.jsonstat("eu_pop_city")
            self.assertEqual(sorted(df.code.unique()), ["PT001C", "PT001T"])
            w, names = panel.wide("eu_pop_city")
            self.assertEqual(list(w.index), ["PT001"])
            self.assertEqual(w.loc["PT001", 2015], 10.0)
            self.assertEqual(w.loc["PT001", 2021], 11.0)
            self.assertEqual(names["PT001"], "Lisboa")

    def test_common_year_prefers_the_target_then_the_later_year(self):
        a = pd.DataFrame({2014: [1.0], 2015: [np.nan], 2016: [1.0]}, index=["X"])
        self.assertEqual(panel.common_year("X", [a], (2014, 2016, 2015)), 2016)
        self.assertIsNone(panel.common_year("Y", [a], (2014, 2016, 2015)))


if __name__ == "__main__":
    unittest.main()
