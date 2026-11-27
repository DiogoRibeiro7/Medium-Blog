"""Check training-only fitting and held-out prediction."""

import unittest

import numpy as np

from causal_tax_cuts.heldout_prefit import predict


class HeldoutTests(unittest.TestCase):
    """Test known weights and invalid inputs."""

    def test_interpolation(self) -> None:
        target = np.array([2.0, 3.0])
        donors = np.array([[1.0, 2.0], [3.0, 4.0]])
        value, effective = predict(target, donors, np.array([4.0, 6.0]))
        self.assertAlmostEqual(value, 5.0)
        self.assertAlmostEqual(effective, 2.0)

    def test_holdout_does_not_change_weights(self) -> None:
        target = np.array([2.0, 3.0])
        donors = np.array([[1.0, 2.0], [3.0, 4.0]])
        _, first = predict(target, donors, np.array([4.0, 6.0]))
        _, second = predict(target, donors, np.array([40.0, 60.0]))
        self.assertAlmostEqual(first, second)

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            predict(np.array([1.0]), np.array([[1.0]]), np.array([2.0]))
