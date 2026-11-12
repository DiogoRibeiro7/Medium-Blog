"""Contracts for part6_past_volatility.py: the logit, the ranking measure and the tests on simulated lists."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_forbes_persistence as ffp  # noqa: E402
import part6_past_volatility as p6  # noqa: E402

LEVELS = [int(round(e * 1000)) for e in ffp.VOL_EDGES]


def simulated_rows(seed: int, trait_sd: float) -> pd.DataFrame:
    rows = pd.DataFrame(ffp.aggregate(p6.simulate_lists(seed, trait_sd, n=6000)))
    for col in ("key", "prev", "tenure", "age", "trend"):
        rows[col] = rows[col].astype(str)
    return rows


class LogitTests(unittest.TestCase):
    def test_grouped_logit_matches_statsmodels(self) -> None:
        rng = np.random.default_rng(3)
        X = np.column_stack([np.ones(60), rng.normal(size=60), rng.integers(0, 2, 60)])
        n = rng.integers(5, 50, 60).astype(float)
        k = rng.binomial(n.astype(int), 1 / (1 + np.exp(-(X @ [-1.5, 0.8, -0.4])))).astype(float)
        ours = p6.fit_logit(X, k, n)
        ref = sm.GLM(np.column_stack([k, n - k]), X, family=sm.families.Binomial()).fit()
        np.testing.assert_allclose(ours["beta"], ref.params, atol=1e-5)
        self.assertAlmostEqual(ours["ll"], ref.llf - float(np.sum(np.log(
            [__import__("math").comb(int(a), int(b)) for a, b in zip(n, k)]))), places=5)

    def test_step_halving_survives_an_empty_category(self) -> None:
        X = np.column_stack([np.ones(4), [0, 0, 1, 1], [0, 1, 0, 0]])
        n = np.array([100.0, 100, 100, 100])
        k = np.array([10.0, 0, 30, 30])
        f = p6.fit_logit(X, k, n, ridge=1.0)
        self.assertTrue(np.all(np.isfinite(f["beta"])))
        self.assertLess(f["beta"][2], -1)

    def test_auc_within_year_ignores_year_levels(self) -> None:
        # one year with many departures and one with few: a model that only knows the year ranks nobody
        p = np.array([0.3, 0.3, 0.1, 0.1])
        k = np.array([30.0, 30, 2, 2])
        n = np.array([100.0, 100, 100, 100])
        year = np.array([1, 1, 2, 2])
        self.assertGreater(p6.auc(p, k, n), 0.5)
        self.assertAlmostEqual(p6.auc(p, k, n, year), 0.5)
        self.assertAlmostEqual(p6.auc(np.array([0.9, 0.1]), np.array([5.0, 0]), np.array([5.0, 5]), np.array([1, 1])), 1.0)


class SimulatedListTests(unittest.TestCase):
    """The tests must find a volatility trait when one is built in, and not when none is."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.none = simulated_rows(21, 0.0)
        cls.strong = simulated_rows(21, 0.8)
        cls.extra = {sd: simulated_rows(22, sd) for sd in (0.0, 0.8)}

    def test_persistence_appears_only_with_a_trait(self) -> None:
        r0, r1 = p6.rises(self.none, LEVELS)["far"], p6.rises(self.strong, LEVELS)["far"]
        self.assertLess(abs(r0["ratio_top"] - r0["ratio_bottom"]), 0.35)
        self.assertGreater(r1["ratio_top"] - r1["ratio_bottom"], 0.8)

    def test_distance_beats_volatility_either_way(self) -> None:
        for rows in (self.none, self.strong):
            c = p6.hazard_cells(rows, p6._years(p6.VOL_YEARS), lambda h: h >= 2)
            m = p6.compare_models(c, [x for x in p6.MODELS if x[0] in ("constant", "distance", "volatility")])["models"]
            self.assertGreater(m["distance"]["logloss_gain_pct"], 5)
            self.assertLess(m["volatility"]["logloss_gain_pct"], 2)

    def test_within_band_ratio_tracks_the_trait(self) -> None:
        ratio = lambda rows: p6.standardised_high_low(  # noqa: E731
            p6.hazard_cells(rows, p6._years(p6.VOL_YEARS), lambda h: h >= 2))["high_over_low"]
        lo = (ratio(self.none) + ratio(self.extra[0.0])) / 2
        hi = (ratio(self.strong) + ratio(self.extra[0.8])) / 2
        self.assertLess(lo, 1.2)
        self.assertGreater(hi, lo + 0.2)


if __name__ == "__main__":
    unittest.main()
