"""Checks of the estimators and the policy rule on cases where the answer is known.

python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import debt_model as M  # noqa: E402
import frf  # noqa: E402


def simulated_country(seed=0, n=400, a=-3.0, rho=0.04, gamma=0.5, phi=0.5):
    rng = np.random.default_rng(seed)
    debt, pb, gap = [80.0], [0.0], [0.0]
    for _ in range(n):
        g = 0.6 * gap[-1] + rng.normal(0, 1.5)
        p = a + rho * debt[-1] + gamma * g + phi * pb[-1] + rng.normal(0, 0.8)
        debt.append(debt[-1] + 1.0 - p + rng.normal(0, 1.0))
        pb.append(p)
        gap.append(g)
    years = np.arange(1700, 1700 + n + 1)
    return pd.DataFrame({"country": "SIM", "year": years, "debt": debt, "pb": pb, "gap": gap})


class TestEstimators(unittest.TestCase):
    def test_ols_recovers_a_known_rule(self):
        df = simulated_country()
        y, X = frf.design(df, years=(1701, 2100))
        r = frf.ols(y, X)
        b = dict(zip(r["cols"], r["b"], strict=True))
        self.assertAlmostEqual(b["debt_l"], 0.04, delta=0.02)
        self.assertAlmostEqual(b["pb_l"], 0.5, delta=0.1)
        self.assertAlmostEqual(b["gap"], 0.5, delta=0.1)

    def test_huber_matches_ols_without_outliers(self):
        df = simulated_country(seed=1)
        y, X = frf.design(df, years=(1701, 2100))
        b_ols = frf.ols(y, X)["b"]
        b_hub = frf.huber(y, X)["b"]
        np.testing.assert_allclose(b_hub[1:4], b_ols[1:4], atol=0.05)

    def test_random_effects_without_heterogeneity(self):
        re = frf.random_effects([0.04, 0.04, 0.04], [0.01, 0.02, 0.03])
        self.assertEqual(re["tau2"], 0.0)
        self.assertAlmostEqual(re["mu"], 0.04)
        np.testing.assert_allclose(re["eb"], 0.04)

    def test_random_effects_shrinks_noisy_estimates_more(self):
        re = frf.random_effects([0.0, 0.1, 0.05, 0.2], [0.01, 0.01, 0.01, 0.2])
        self.assertLess(re["shrink"][3], re["shrink"][0])


class TestRule(unittest.TestCase):
    def test_anchored_intercept(self):
        a = M.anchored_intercept(0.04, 0.5, 84.5, 2.5)
        p = M.Policy("t", "t", kind="rule", a=a, rho=0.04, phi=0.5)
        self.assertAlmostEqual(p.long_run_pb(84.5), 2.5)

    def test_steady_state_debt_balances(self):
        p = M.Policy("t", "t", kind="rule", a=-3.0, rho=0.04, phi=0.5)
        d = M.steady_state_debt(p, 3.6, 3.1)
        self.assertAlmostEqual(p.long_run_pb(d), M.SFA_LATER + d * (3.6 - 3.1) / 103.1, places=10)

    def test_fixed_policy_reproduces_the_2035_article(self):
        r = M.run(M.Policy("fixed", "fixed", kind="fixed", pbs=2.5))
        self.assertAlmostEqual(r.loc[2027, "debt"], 84.5, delta=0.05)
        self.assertAlmostEqual(r.loc[2035, "debt"], 63.83, delta=0.01)

    def test_rule_keeps_2027_budget(self):
        p = M.Policy("t", "t", kind="rule", a=-3.0, rho=0.04, gamma=0.5, phi=0.5)
        r = M.run(p)
        self.assertAlmostEqual(r.loc[2027, "primary_balance"], 2.5)
        expected_2028 = -3.0 + 0.04 * r.loc[2027, "debt"] + 0.5 * 2.5
        self.assertAlmostEqual(r.loc[2028, "primary_balance"], expected_2028, places=10)


@unittest.skipUnless((HERE / "data" / "panel.csv").exists(), "run fetch_data.py and build_data.py first")
class TestData(unittest.TestCase):
    def test_panel(self):
        p = pd.read_csv(HERE / "data" / "panel.csv")
        self.assertEqual(p.country.nunique(), 27)
        pt = p[p.country == "PRT"].set_index("year")
        self.assertAlmostEqual(pt.loc[2025, "debt"], 89.15, delta=0.01)
        self.assertAlmostEqual(pt.loc[2025, "pb"], 2.62, delta=0.01)
        self.assertEqual(int(p.year.max()), 2025)


if __name__ == "__main__":
    unittest.main()
