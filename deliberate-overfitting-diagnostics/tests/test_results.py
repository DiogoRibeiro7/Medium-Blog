"""Checks on the helper functions and on the committed results.

Run from the project folder: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common import clean_rule, simulate  # noqa: E402
from exp2_noise_floor import cover_hart_lower, loo_1nn_error, nn_residual_variance, plug_in_floor  # noqa: E402

RES = HERE / "results"


class Helpers(unittest.TestCase):
    def test_simulated_flip_rate_is_the_bayes_error(self):
        rng = np.random.default_rng(1)
        x, y = simulate(200_000, 0.2, rng)
        self.assertAlmostEqual(np.mean(y != clean_rule(x)), 0.2, delta=0.004)

    def test_plug_in_floor_counts_minorities(self):
        keys = pd.Series(["a", "a", "a", "b", "b", "c"])
        y = np.array([1, 1, 0, 0, 1, 1])
        floor, singleton, cells = plug_in_floor(keys, y)
        self.assertAlmostEqual(floor, 2 / 6)
        self.assertAlmostEqual(singleton, 1 / 6)
        self.assertEqual(cells, 3)

    def test_cover_hart_inverts_the_asymptotic_nn_risk(self):
        for p in (0.0, 0.1, 0.25, 0.4):
            self.assertAlmostEqual(cover_hart_lower(2 * p * (1 - p)), p, places=12)

    def test_loo_1nn_never_uses_the_point_itself(self):
        x = np.arange(10, dtype=float)[:, None]
        y = np.array([0, 1] * 5)
        self.assertEqual(loo_1nn_error(x, y), 1.0)

    def test_nn_residual_variance_on_pure_noise(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((20_000, 3))
        y = 0.7 * rng.standard_normal(20_000)
        prod, diff = nn_residual_variance(x, y)
        self.assertAlmostEqual(prod, 0.49, delta=0.03)
        self.assertAlmostEqual(diff, 0.49, delta=0.03)


class CommittedResults(unittest.TestCase):
    def test_every_model_memorises_every_version(self):
        a = pd.read_csv(RES / "exp1a_memorisation_grid.csv")
        acc = a[a["train_acc"].notna()]
        self.assertGreaterEqual(acc["train_acc"].min(), 0.999)
        self.assertEqual(set(acc["labels"]), {"real", "shuffled"})

    def test_memorisation_rule_has_no_size_control(self):
        b = pd.read_csv(RES / "exp1b_detection_power.csv")
        null = b[b["eta"] == 0.5]
        self.assertEqual(null["memorised"].mean(), 1.0)
        self.assertLessEqual((null["p_leaves"] <= 0.05).mean(), 0.12)
        self.assertLessEqual((null["p_holdout"] <= 0.05).mean(), 0.12)

    def test_random_id_fits_perfectly_and_predicts_nothing(self):
        c = pd.read_csv(RES / "exp1c_feature_set_ranking.csv").set_index("feature_set")
        r = c.loc["Random row ID only"]
        self.assertEqual(r["tree_train_acc"], 1.0)
        self.assertLessEqual(r["reg_xgb_test_acc"], r["majority_baseline"] + 0.005)

    def test_tree_training_error_equals_duplicate_floor(self):
        d = pd.read_csv(RES / "exp2a_adult_duplicate_floor.csv")
        np.testing.assert_allclose(d["overfit_tree_train_error"], d["plug_in_floor"], atol=1e-9)
        self.assertTrue((np.diff(d["plug_in_floor"]) <= 1e-12).all())
        # held-out error is above the floor in expectation; allow fold-size rounding
        self.assertTrue((d["xgb_cv_error"] >= d["plug_in_floor"] - 1e-4).all())

    def test_cover_hart_bound_is_below_the_nn_error(self):
        ch = pd.read_csv(RES / "exp2b_cover_hart.csv")
        self.assertTrue((ch["cover_hart_lower"] <= ch["r_1nn"] + 1e-12).all())

    def test_bug_matrix(self):
        m = pd.read_csv(RES / "exp3_detection_matrix.csv")
        flagged = m[m["flagged"]].groupby("check")["bug"].apply(set).to_dict()
        self.assertFalse(m[(m["bug"] == "No bug")]["flagged"].any())
        self.assertEqual(flagged.get("C2 overfit full training set"),
                         {"Every feature made constant", "Learning rate a million times too small"})
        self.assertEqual(flagged.get("C5 single-column audit"), {"Target leak in one column"})
        self.assertEqual(flagged.get("C6 train-test duplicate audit"),
                         {"Minority class oversampled before the split"})
        self.assertIn("Labels misaligned after a sort", flagged.get("C3 shuffle-calibrated memorisation", set()))


if __name__ == "__main__":
    unittest.main()
