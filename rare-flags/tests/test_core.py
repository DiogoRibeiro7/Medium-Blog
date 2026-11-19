"""Tests of the methods on cases where the answer is known.

python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flags_core as fc  # noqa: E402


class TestArithmetic(unittest.TestCase):
    def test_mammography(self):
        """1% base rate, 90% sensitivity, 9% false positives: about one positive in eleven is right."""
        self.assertAlmostEqual(float(fc.ppv(0.01, 0.9, 0.09)), 0.009 / (0.009 + 0.0891))

    def test_frequencies_add_up(self):
        f = fc.frequencies(1000, 0.03, 0.5, 0.1)
        self.assertAlmostEqual(f["positive"] + f["negative"], 1000)
        self.assertAlmostEqual(f["true_flags"], 15)
        self.assertAlmostEqual(f["false_flags"], 97)

    def test_required_fpr_inverts_ppv(self):
        for b in (0.0001, 0.03, 0.2):
            for t in (0.5, 0.9, 0.99):
                self.assertAlmostEqual(float(fc.ppv(b, 0.8, fc.required_fpr(b, 0.8, t))), t)

    def test_likelihood_ratio_moves_odds(self):
        b, t = 0.03, 0.9
        lr = fc.likelihood_ratio_needed(b, t)
        posterior_odds = b / (1 - b) * lr
        self.assertAlmostEqual(posterior_odds / (1 + posterior_odds), t)

    def test_false_per_true(self):
        self.assertAlmostEqual(float(fc.false_per_true(0.5, 1.0, 0.25)), 0.25)


class TestUncertainty(unittest.TestCase):
    def test_jeffreys_zero_of_n(self):
        lo, hi = fc.jeffreys(0, 88)
        self.assertEqual(lo, 0.0)
        self.assertTrue(0.02 < hi < 0.04)

    def test_jeffreys_symmetric(self):
        lo, hi = fc.jeffreys(50, 100)
        self.assertAlmostEqual(lo, 1 - hi)

    def test_ppv_interval_covers_point(self):
        lo, hi = fc.ppv_interval(0.05, 40, 100, 10, 100)
        point = float(fc.ppv(0.05, 0.4, 0.1))
        self.assertTrue(lo < point < hi)


class TestFraudTools(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = np.r_[np.ones(50, int), np.zeros(5000, int)]
        self.p = np.r_[rng.uniform(0.6, 1.0, 50), rng.uniform(0.0, 0.7, 5000)]

    def test_perfect_separation(self):
        y = np.r_[np.ones(10, int), np.zeros(100, int)]
        p = np.r_[np.full(10, 0.9), np.full(100, 0.1)]
        a = fc.at_recall(y, p, 1.0)
        self.assertEqual((a["alerts"], a["true"], a["false"]), (10, 10, 0))

    def test_rebase_to_same_rate_changes_nothing(self):
        prec, fpt = fc.rebase(30, 10, 0.01, 0.01)
        self.assertAlmostEqual(prec, 0.75)
        self.assertAlmostEqual(fpt, 1 / 3)

    def test_rebase_to_rarer_lowers_precision(self):
        prec, _ = fc.rebase(30, 10, 0.01, 0.001)
        self.assertLess(prec, 0.75)

    def test_curve_at_own_rate(self):
        r1, p1 = fc.precision_recall_curve(self.y, self.p)
        r2, p2 = fc.precision_recall_curve(self.y, self.p, self.y.mean())
        np.testing.assert_allclose(r1, r2)
        np.testing.assert_allclose(p1, p2)


class TestDetectorCounts(unittest.TestCase):
    def test_threshold_and_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folders in fc.GROUPS.values():
                for folder in folders:
                    d = root / "Data_and_Results" / folder
                    d.mkdir(parents=True)
                    for det in fc.DETECTORS:
                        recs = [{"score": s} for s in (0.2, 0.5, 0.51, 0.9)]
                        (d / f"{det}.json").write_text(json.dumps(recs))
            c = fc.detector_counts(root)
            ai = c[(c.group == "ai") & (c.detector == "GPTZero")].iloc[0]
            self.assertEqual((ai.flagged, ai.essays), (4, 8))  # two folders pooled; 0.5 is not a flag
            native = c[(c.group == "native") & (c.detector == "Sapling")].iloc[0]
            self.assertEqual((native.flagged, native.essays), (2, 4))


if __name__ == "__main__":
    unittest.main()
