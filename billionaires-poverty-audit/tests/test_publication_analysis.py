"""Contracts for publication_analysis.py (revised article numbers and figures)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import publication_analysis as pa  # noqa: E402
from observed_figures import BILLIONAIRE, GDP, load_panel  # noqa: E402


class PublicationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_panel(pa.PANEL)

    def test_baseline_models_match_published_values(self) -> None:
        joint = pa.ols_hc3(self.frame, [BILLIONAIRE, GDP])
        gdp = pa.ols_hc3(self.frame, [GDP])
        raw = pa.ols_hc3(self.frame, [BILLIONAIRE])
        self.assertEqual(joint["n"], 54)
        self.assertAlmostEqual(raw["r_squared"], 0.5471, places=4)
        self.assertAlmostEqual(gdp["r_squared"], 0.8049, places=4)
        self.assertAlmostEqual(joint["r_squared"], 0.8113, places=4)
        self.assertAlmostEqual(joint["coef"][BILLIONAIRE], -7.912, places=3)
        self.assertAlmostEqual(joint["hc3_p"][BILLIONAIRE], 0.119, places=3)

    def test_loo_cv_matches_hat_matrix_identity(self) -> None:
        cols = [BILLIONAIRE, GDP]
        X = sm.add_constant(self.frame[cols]).to_numpy(dtype=float)
        y = self.frame.poverty_pct.to_numpy(dtype=float)
        fit = sm.OLS(y, X).fit()
        h = fit.get_influence().hat_matrix_diag
        press = np.sqrt(np.mean((fit.resid / (1.0 - h)) ** 2))
        self.assertAlmostEqual(pa.loo_cv(self.frame, cols)["rmse"], float(press), places=9)

    def test_fractional_logit_ame_matches_finite_difference(self) -> None:
        cols = [BILLIONAIRE, GDP]
        result = pa.fractional_logit(self.frame, cols)
        beta = np.array([result["coef"][k] for k in ["const", *cols]])
        X = sm.add_constant(self.frame[cols]).to_numpy(dtype=float)
        eps = 1e-6
        shifted = X.copy()
        shifted[:, 1] += eps
        expit = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        numeric = 100.0 * np.mean(expit(shifted @ beta) - expit(X @ beta)) / eps
        self.assertAlmostEqual(result["ame_billionaire_pp_per_tenfold"], numeric, places=4)
        lo, hi = result["ame_ci95"]
        self.assertLess(lo, result["ame_billionaire_pp_per_tenfold"])
        self.assertGreater(hi, result["ame_billionaire_pp_per_tenfold"])
        self.assertGreaterEqual(result["fitted_min_pct"], 0.0)
        self.assertLessEqual(result["fitted_max_pct"], 100.0)

    def test_placebo_proxies_preserve_correlation_and_are_deterministic(self) -> None:
        a = pa.placebo_proxies(self.frame, 200, 7)
        b = pa.placebo_proxies(self.frame, 200, 7)
        np.testing.assert_allclose(a["_r2_draws"], b["_r2_draws"])
        self.assertAlmostEqual(a["corr_billionaire_gdp"], 0.7676, places=4)
        # Every proxy correlates with GDP exactly as billionaire density does,
        # so its R2 cannot exceed what GDP explains plus its own residual share.
        self.assertTrue(np.all(a["_r2_draws"] > 0.0))
        self.assertTrue(np.all(a["_r2_draws"] < 1.0))

    def test_gdp_bands_partition_the_sample(self) -> None:
        bands = pa.gdp_bands(self.frame)
        self.assertEqual([b["n"] for b in bands], [18, 18, 18])
        members = sorted(sum((b["iso3"] for b in bands), []))
        self.assertEqual(members, sorted(self.frame.iso3.tolist()))

    def test_committed_numbers_reproduce(self) -> None:
        numbers, _ = pa.compute(self.frame)
        numbers = json.loads(json.dumps(numbers))
        committed = json.loads((pa.RESULTS / "numbers.json").read_text(encoding="utf-8"))
        problems: list[str] = []
        pa._compare(numbers, committed, "numbers", problems)
        self.assertEqual(problems, [])

    def test_figures_render(self) -> None:
        numbers, draws = pa.compute(self.frame)
        with tempfile.TemporaryDirectory() as tmp:
            pa.render(self.frame, json.loads(json.dumps(numbers)), draws, Path(tmp))
            for name in pa.FIGURE_FILES:
                data = (Path(tmp) / name).read_bytes()
                self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
                width = int.from_bytes(data[16:20], "big")
                self.assertGreaterEqual(width, 2200)

    def test_fmt_uses_typographic_minus(self) -> None:
        self.assertEqual(pa.fmt(-7.912), "−7.9")
        self.assertEqual(pa.fmt(2.04, sign=True), "+2.0")
        self.assertTrue(math.isclose(float(pa.fmt(3.14159, 3)), 3.142))


if __name__ == "__main__":
    unittest.main()
