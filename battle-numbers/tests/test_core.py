"""Tests on synthetic battles, where the truth is known.

python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import battle_core as bc  # noqa: E402


def battles(n, seed):
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(10, 1, n))
    d = np.exp(rng.normal(9.6, 1, n))
    return rng, a, d


class TestLanchester(unittest.TestCase):
    def test_square_law_losses_give_slope_minus_one(self):
        """Losses proportional to the enemy's size: the loss ratio falls one-for-one with the force ratio."""
        rng, a, d = battles(4000, 1)
        noise = np.exp(rng.normal(0, 0.3, (2, a.size)))
        c = pd.DataFrame({"lr": np.log(a / d), "llr": np.log((0.05 * d * noise[0]) / (0.05 * a * noise[1]))})
        self.assertAlmostEqual(bc.casualty_slope(c), -1.0, delta=0.05)

    def test_linear_law_losses_give_slope_zero(self):
        """Losses proportional to the product of the sizes: the loss ratio does not depend on the force ratio."""
        rng, a, d = battles(4000, 2)
        noise = np.exp(rng.normal(0, 0.3, (2, a.size)))
        c = pd.DataFrame({"lr": np.log(a / d), "llr": np.log((1e-6 * a * d * noise[0]) / (2e-6 * a * d * noise[1]))})
        self.assertAlmostEqual(bc.casualty_slope(c), 0.0, delta=0.05)

    def test_implied_spread_recovers_the_quality_spread(self):
        """Square-law winners with logistic quality differences of standard deviation 1.5: the spread comes back."""
        rng = np.random.default_rng(3)
        lr = rng.normal(0.4, 0.7, 40000)
        scale = 1.5 * np.sqrt(3) / np.pi
        quality = rng.logistic(0, scale, lr.size)
        win = (2 * lr + quality > 0).astype(int)
        slope = bc.logit(win, lr).params[1]
        self.assertAlmostEqual(bc.implied_spread(slope, 2), 1.5, delta=0.08)


class TestRecords(unittest.TestCase):
    def test_prior_record_counts_only_earlier_battles(self):
        b = pd.DataFrame(
            {
                "isqno": [1, 2, 3, 4],
                "date": pd.to_datetime(["1800-01-01", "1800-06-01", "1800-06-01", "1801-01-01"]),
                "wina": [1, -1, 1, 1],
            }
        )
        # commander X attacks in 1, 2, 3 and defends in 4; Y defends in 1 and 4
        links = pd.DataFrame(
            {
                "isqno": [1, 1, 2, 3, 4, 4],
                "attacker": [1, 0, 1, 1, 1, 0],
                "wikidata_id": ["X", "Y", "X", "X", "Z", "X"],
            }
        )
        r = bc.prior_records(b, links).set_index("isqno")
        self.assertEqual(r.loc[1, "a_prior_n"], 0)
        self.assertEqual(r.loc[2, "a_prior_n"], 1)  # battle 3 is on the same day and does not count
        self.assertEqual(r.loc[3, "a_prior_n"], 1)
        self.assertEqual(r.loc[4, "d_prior_n"], 3)
        self.assertEqual(r.loc[4, "d_prior_w"], 2)  # won 1 and 3 as attacker, lost 2
        self.assertEqual(r.loc[4, "d_prior_l"], 1)

    def test_record_edge(self):
        rec = pd.DataFrame(
            {"a_prior_w": [2, 0, 1], "a_prior_l": [0, 0, 1], "d_prior_w": [0, 0, 3], "d_prior_l": [1, 0, 0]}
        )
        self.assertEqual(list(bc.record_edge(rec)), [1, 0, -1])


class TestHindsight(unittest.TestCase):
    def pairs(self, seed, follows_result):
        rng = np.random.default_rng(seed)
        rows = []
        for p in range(30):
            res = rng.permutation([1, -1, 1, -1])
            for r in res:
                lead = r if follows_result else rng.integers(-1, 2)
                rows.append(("a", f"b{p}", r, lead))
        return pd.DataFrame(rows, columns=["c1", "c2", "res", "lead"])

    def test_a_code_read_off_the_result_is_caught(self):
        r, p = bc.within_pair(self.pairs(4, True), "lead", permutations=999)
        self.assertAlmostEqual(r, 1.0)
        self.assertLess(p, 0.01)

    def test_a_code_unrelated_to_the_result_is_not(self):
        r, p = bc.within_pair(self.pairs(5, False), "lead", permutations=999)
        self.assertLess(abs(r), 0.3)
        self.assertGreater(p, 0.05)


class TestHelpers(unittest.TestCase):
    def test_auc(self):
        self.assertAlmostEqual(bc.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)

    def test_larger_side_share(self):
        d = pd.DataFrame({"lr": [0.5, -0.5, 0.2, 0.0], "win": [1, 1, 0, 1]})
        share, n = bc.larger_side_share(d)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(share, 1 / 3)

    def test_attenuation_without_error_is_one(self):
        d = pd.DataFrame({"lr": [0.1, 0.5, -0.3, 1.2], "att_err": 0.0, "def_err": 0.0})
        self.assertAlmostEqual(bc.attenuation(d)[0], 1.0)


if __name__ == "__main__":
    unittest.main()
