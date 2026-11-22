"""Tests for the rule simulator, the calibration and the comparison helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rules as R  # noqa: E402
import run_analysis as A  # noqa: E402

J = {y: t for t, y in enumerate(R.YEARS)}


def one_shock(year: int, col: int, value: float) -> np.ndarray:
    d = R.zero_draws()
    d[0, J[year], col] = value
    return d


class TestMacroBlock(unittest.TestCase):
    def test_fixed_rule_reproduces_the_earlier_fixed_surplus_path(self):
        # portugal-debt-2035-probability / portugal-fiscal-reaction: 75.3 (2030), 63.8 (2035), 44.3 (2045)
        s = R.simulate(R.Rule("f", "f", "fixed", level=2.5), R.Macro(rho_gap=0.467), R.zero_draws())
        self.assertAlmostEqual(s["debt"][0, J[2030]], 75.288, places=2)
        self.assertAlmostEqual(s["debt"][0, J[2035]], 63.828, places=2)
        self.assertAlmostEqual(s["debt"][0, J[2045]], 44.284, places=2)

    def test_debt_identity_closes(self):
        rng = np.random.default_rng(1)
        draws = rng.normal(0, 1, size=(50, R.H, 5))
        s = R.simulate(R.Rule("d", "d", "debt", level=3.0, kappa=0.1), R.Macro(), draws)
        d_prev = np.column_stack([np.full(50, R.D0), s["debt"][:, :-1]])
        g = s["growth"] / 100
        sfa = np.full((50, R.H), R.SFA_LATER)
        sfa[:, 0] = R.SFA_2027
        sfa = sfa + draws[:, :, 3]
        rebuilt = d_prev / (1 + g) + s["interest"] - s["pb"] + sfa
        np.testing.assert_allclose(rebuilt, s["debt"], atol=1e-10)

    def test_real_time_gap_misses_the_innovation(self):
        macro = R.Macro(gap_miss=1.0, mu=0.0)
        d = one_shock(2028, 0, -4.0)
        s = R.simulate(R.Rule("f", "f", "fixed", level=2.5), macro, d)
        # the rule sees no 2028 innovation, so the 2028 primary balance is the level and the structural
        # balance rises by the cyclical part the automatic stabilisers take
        self.assertAlmostEqual(s["pb"][0, J[2028]], 2.5, places=10)
        self.assertGreater(s["sp"][0, J[2028]] - s["sp"][0, J[2027]], 1.0)


class TestRules(unittest.TestCase):
    def test_debt_rule_formula(self):
        rule = R.Rule("d", "d", "debt", level=3.0, kappa=0.1)
        s = R.simulate(rule, R.Macro(), R.zero_draws())
        expected = 3.0 + 0.1 * (s["debt"][0, J[2028] - 1] - R.D_REF)
        self.assertAlmostEqual(s["pb"][0, J[2028]], expected, places=10)

    def test_structural_rule_holds_the_headline_balance(self):
        s = R.simulate(R.Rule("s", "s", "structural", level=0.2), R.Macro(mu=0.0), R.zero_draws())
        np.testing.assert_allclose(s["bal"][0, 1:], 0.2, atol=1e-10)

    def test_anchored_spending_path_reaches_its_target_at_the_end_of_the_plan(self):
        s = R.simulate(R.Rule("x", "x", "spending", level=3.5), R.Macro(mu=0.0), R.zero_draws())
        self.assertAlmostEqual(s["pb"][0, J[2031]], 3.5, places=8)
        self.assertAlmostEqual(s["pb"][0, J[2035]], 3.5, places=8)
        self.assertLess(s["pb"][0, J[2028]], s["pb"][0, J[2030]])

    def test_spending_path_lets_an_inflation_surprise_raise_the_balance(self):
        rule = R.Rule("x", "x", "spending", level=2.5)
        base = R.simulate(rule, R.Macro(), R.zero_draws())
        hot = R.simulate(rule, R.Macro(), one_shock(2029, 1, 3.0))
        self.assertGreater(hot["pb"][0, J[2029]] - base["pb"][0, J[2029]], 1.0)
        fixed = R.Rule("f", "f", "fixed", level=2.5)
        self.assertAlmostEqual(
            R.simulate(fixed, R.Macro(), one_shock(2029, 1, 3.0))["pb"][0, J[2029]],
            R.simulate(fixed, R.Macro(), R.zero_draws())["pb"][0, J[2029]],
        )

    def test_amortisation_repays_a_shortfall_in_six_equal_instalments(self):
        rule = R.Rule("f", "f", "fixed", level=2.5, amortise=6)
        s = R.simulate(rule, R.Macro(mu=0.0, edp=False), one_shock(2028, 4, -6.0))
        extra = s["pb"][0] - 2.5
        self.assertAlmostEqual(extra[J[2028]], -6.0, places=10)
        np.testing.assert_allclose(extra[J[2029] : J[2034] + 1], 1.0, atol=1e-10)
        self.assertAlmostEqual(extra[J[2035]], 0.0, places=10)

    def test_a_windfall_is_credited_against_later_repayments(self):
        rule = R.Rule("f", "f", "fixed", level=2.5, amortise=6)
        d = one_shock(2028, 4, 3.0)
        d[0, J[2029], 4] = -6.0
        s = R.simulate(rule, R.Macro(mu=0.0, edp=False), d)
        extra = s["pb"][0] - 2.5
        # instalments of 1.0 from 2030; the first three are paid out of the 3.0 credit
        np.testing.assert_allclose(extra[J[2030] : J[2032] + 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(extra[J[2033] : J[2035] + 1], 1.0, atol=1e-10)

    def test_corrective_arm_adds_half_a_point_after_a_breach(self):
        rule = R.Rule("f", "f", "fixed", level=2.5)
        s = R.simulate(rule, R.Macro(mu=0.0), one_shock(2028, 4, -6.0))
        self.assertLess(s["bal"][0, J[2028]], -3.0)
        self.assertAlmostEqual(s["pb"][0, J[2029]], 3.0, places=10)


class TestCalibrationAndComparison(unittest.TestCase):
    def test_calibration_hits_the_target_for_every_kind(self):
        rng = np.random.default_rng(7)
        draws = rng.normal(0, 0.8, size=(300, R.H, 5))
        for rule in A.MAIN + A.VARIANTS:
            c = R.calibrate(rule, R.Macro(), draws, 2.5)
            self.assertAlmostEqual(R.effort(R.simulate(c, R.Macro(), draws)), 2.5, places=4, msg=rule.key)

    def test_block_draws_keep_years_together(self):
        shocks = pd.DataFrame(np.arange(28 * 5, dtype=float).reshape(28, 5), index=range(1998, 2026),
                              columns=["u", "v", "dm", "sfa", "e"])  # fmt: skip
        draws, years = A.block_draws(shocks, 200, 3)
        self.assertEqual(draws.shape, (200, R.H, 5))
        for t in range(0, R.H - 2, 3):
            np.testing.assert_array_equal(years[:, t + 1] - years[:, t], 1)
            np.testing.assert_array_equal(years[:, t + 2] - years[:, t + 1], 1)

    def test_dominance_and_weights(self):
        res = {
            "a": {"p90_2035": 80, "p_breach": 0.3, "p_forced_tightening": 0.5, "p_at_or_below_60_2035": 0.3},
            "b": {"p90_2035": 90, "p_breach": 0.3, "p_forced_tightening": 0.1, "p_at_or_below_60_2035": 0.4},
            "c": {"p90_2035": 95, "p_breach": 0.4, "p_forced_tightening": 0.6, "p_at_or_below_60_2035": 0.2},
        }
        dom = A.dominated(res, ["a", "b", "c"])
        self.assertEqual(dom["a"], [])
        self.assertEqual(dom["b"], [])
        self.assertEqual(sorted(dom["c"]), ["a", "b"])
        wm = A.weight_map(res, ["a", "b", "c"], step=0.1)
        self.assertAlmostEqual(sum(wm["share"].values()), 1.0)
        self.assertEqual(wm["share"]["c"], 0.0)
        corner = {tuple(g["w"]): g["best"] for g in wm["grid"]}
        self.assertEqual(corner[(1.0, 0.0, 0.0)], "a")
        self.assertEqual(corner[(0.0, 0.0, 1.0)], "b")


if __name__ == "__main__":
    unittest.main()
