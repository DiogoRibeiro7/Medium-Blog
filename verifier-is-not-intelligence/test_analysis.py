"""Check the exact examples and edge cases."""
import math
import unittest

from analysis import JointMass, break_even_probability, expected_minutes, selected_outcome, serial_success


class ScenarioTests(unittest.TestCase):
    def test_perfect_verifier(self) -> None:
        m = JointMass(0.55, 0, 0, 0.45)
        for n in (1, 2, 20):
            accepted, quality = selected_outcome(m, n)
            self.assertAlmostEqual(accepted, quality)
            self.assertAlmostEqual(accepted, 1 - 0.45 ** n)

    def test_imperfect_verifier(self) -> None:
        m = JointMass(0.25, 0.35, 0.30, 0.10)
        a1, q1 = selected_outcome(m, 1)
        a20, q20 = selected_outcome(m, 20)
        self.assertAlmostEqual(a1, 0.6)
        self.assertAlmostEqual(q1, 0.55)
        self.assertGreater(a20, a1)
        self.assertLess(q20, q1)
        self.assertAlmostEqual(q20, 5 / 12, places=7)
        self.assertEqual(selected_outcome(JointMass(0, 0, 0.4, 0.6), 5), (0, 0.4))

    def test_reliability(self) -> None:
        self.assertAlmostEqual(serial_success(0.99, 100), 0.3660323412732292)
        self.assertAlmostEqual(serial_success(0.95, 20), 0.3584859224085419)
        self.assertEqual(serial_success(1, 100), 1)

    def test_review_cost(self) -> None:
        threshold = break_even_probability(60, 15, 20, 90)
        self.assertAlmostEqual(threshold, 25 / 90)
        self.assertAlmostEqual(expected_minutes(15, 20, 90, threshold), 60)
        self.assertAlmostEqual(expected_minutes(15, 20, 90, 0.4), 71)

    def test_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            JointMass(0.6, 0.6, 0, 0)
        with self.assertRaises(ValueError):
            serial_success(math.nan, 2)
        with self.assertRaises(ValueError):
            selected_outcome(JointMass(1, 0, 0, 0), 0)
        with self.assertRaises(ValueError):
            expected_minutes(15, 20, 90, 1.5)


if __name__ == "__main__":
    unittest.main()
