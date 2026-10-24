"""Fast checks of the combining rules, the closed forms and the generators.

Run from the project folder: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import theory  # noqa: E402
from generators import (  # noqa: E402
    Encoder,
    GaussianCopula,
    IndependentMarginals,
    SequentialCART,
    SMOTEGenerator,
    synthesize,
)
from inference import combine, covers, predicted_coverage  # noqa: E402


def toy_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    level = rng.choice(["low", "mid", "high"], size=n, p=[0.5, 0.3, 0.2])
    code = pd.Series(level).map({"low": 1, "mid": 2, "high": 3}).to_numpy()
    age = rng.integers(18, 80, size=n)
    income = np.where(rng.random(n) < 0.2 + 0.2 * (code == 3), "rich", "poor")
    return pd.DataFrame({"level": level, "code": code, "age": age, "income": income})


class CombiningRules(unittest.TestCase):
    def setUp(self):
        self.q = np.array([[1.0, 2.0], [1.2, 1.8], [0.8, 2.2], [1.1, 2.1]])
        self.u = np.array([[0.04, 0.09], [0.05, 0.08], [0.03, 0.10], [0.04, 0.09]])

    def test_variances(self):
        ubar = self.u.mean(axis=0)
        m, k = 4, 1.0
        self.assertTrue(np.allclose(combine(self.q, self.u, k, "naive").variance, ubar / m))
        self.assertTrue(np.allclose(combine(self.q, self.u, k, "simple").variance, ubar * (k + 1 / m)))
        self.assertTrue(np.allclose(combine(self.q, self.u, k, "proper").variance, ubar * (k + (1 + k) / m)))

    def test_raghunathan_and_fallback(self):
        b = self.q.var(axis=0, ddof=1)
        ubar = self.u.mean(axis=0)
        iv = combine(self.q, self.u, 1.0, "raghunathan")
        T = (1 + 1 / 4) * b - ubar
        expected = np.where(T > 0, T, ubar)
        self.assertTrue(np.allclose(iv.variance, expected))
        self.assertTrue(np.array_equal(iv.fallback, T <= 0))

    def test_estimate_is_mean(self):
        iv = combine(self.q, self.u, 1.0, "simple")
        self.assertTrue(np.allclose(iv.estimate, self.q.mean(axis=0)))
        self.assertTrue(np.all(covers(iv, iv.estimate)))


class ClosedForms(unittest.TestCase):
    def test_predicted_coverage(self):
        self.assertAlmostEqual(predicted_coverage(1.0), 0.95, places=10)
        self.assertAlmostEqual(predicted_coverage(2.0), 0.8342, places=4)

    def test_matched_rules_are_exact(self):
        for m in (1, 2, 5, 20):
            self.assertAlmostEqual(theory.coverage("simple", 1.0, m, proper=False), 0.95, places=10)
            self.assertAlmostEqual(theory.coverage("proper", 1.0, m, proper=True), 0.95, places=10)

    def test_size_study(self):
        # one synthetic set of k n rows: variance understated by k + 1 (simple), 2k + 1 (proper)
        self.assertAlmostEqual(theory.size_coverage(5, proper=False), predicted_coverage(6.0), places=12)
        self.assertAlmostEqual(theory.size_coverage(5, proper=True), predicted_coverage(11.0), places=12)


class Generators(unittest.TestCase):
    def setUp(self):
        self.df = toy_frame(600, seed=1)
        self.enc = Encoder.from_frame(self.df)
        self.M = self.enc.to_codes(self.df)

    def test_round_trip(self):
        back = self.enc.from_codes(self.M)
        pd.testing.assert_frame_equal(back, self.df, check_dtype=False)

    def test_shapes_and_levels(self):
        rng = np.random.default_rng(0)
        for gen in (IndependentMarginals, GaussianCopula, SequentialCART):
            S = gen(self.enc).fit(self.M, rng).sample(250, rng)
            self.assertEqual(S.shape, (250, 4))
            out = self.enc.from_codes(S)
            self.assertTrue(set(out["level"]) <= {"low", "mid", "high"})
            self.assertTrue(set(out["age"]) <= set(self.df["age"]))

    def test_cart_keeps_a_deterministic_recoding(self):
        rng = np.random.default_rng(0)
        S = self.enc.from_codes(SequentialCART(self.enc).fit(self.M, rng).sample(2000, rng))
        mapping = {"low": 1, "mid": 2, "high": 3}
        self.assertTrue(np.all(S["level"].map(mapping) == S["code"]))

    def test_smote_stays_inside_the_range(self):
        rng = np.random.default_rng(0)
        target = self.enc.columns.index("income")
        S = self.enc.from_codes(SMOTEGenerator(self.enc, class_col=target).fit(self.M, rng).sample(1000, rng))
        self.assertGreaterEqual(S["age"].min(), self.df["age"].min())
        self.assertLessEqual(S["age"].max(), self.df["age"].max())

    def test_proper_synthesis_refits(self):
        rng = np.random.default_rng(0)
        sets = synthesize(lambda: GaussianCopula(self.enc), self.M, 300, 3, rng, proper=True)
        self.assertEqual(len(sets), 3)
        self.assertFalse(np.array_equal(sets[0], sets[1]))


if __name__ == "__main__":
    unittest.main()
