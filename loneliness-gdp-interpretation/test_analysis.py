"""Regression tests for the exact synthetic examples."""
import unittest

from analysis import cells, daily_membership, pooled, reported_prevalence


class CounterexampleTests(unittest.TestCase):
    def test_pooled_gap(self) -> None:
        rows = cells()
        self.assertAlmostEqual(rows[0].prevalence, rows[1].prevalence)
        self.assertAlmostEqual(rows[2].prevalence, rows[3].prevalence)
        self.assertAlmostEqual(pooled(rows, "Struggling"), 0.32)
        self.assertAlmostEqual(pooled(rows, "Comfortable"), 0.15)

    def test_measurement(self) -> None:
        self.assertAlmostEqual(reported_prevalence(0.20, 0.90, 0.125), 0.28)
        self.assertAlmostEqual(reported_prevalence(0.20, 0.70, 0.025), 0.16)
        with self.assertRaises(ValueError):
            reported_prevalence(1.1, 0.9, 0.1)

    def test_daily(self) -> None:
        for rotating, expected in ((False, 23), (True, 100)):
            groups = daily_membership(rotating)
            self.assertTrue(all(len(group) == 23 for group in groups))
            self.assertEqual(len(set().union(*groups)), expected)


if __name__ == "__main__":
    unittest.main()
