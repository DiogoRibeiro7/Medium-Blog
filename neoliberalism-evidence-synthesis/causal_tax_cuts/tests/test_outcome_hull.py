"""Synthetic geometry tests for the expanded support audit."""

import unittest

import numpy as np

from causal_tax_cuts.outcome_hull import in_hull


class OutcomeHullTests(unittest.TestCase):
    """Check exact interpolation and failure modes."""

    def test_inside(self) -> None:
        self.assertTrue(in_hull(np.array([[-1.0, 0.0], [1.0, 0.0]])))

    def test_outside_joint(self) -> None:
        self.assertFalse(in_hull(np.array([[-1.0, 2.0], [2.0, -1.0]])))

    def test_extra_dimension_can_remove_support(self) -> None:
        self.assertTrue(in_hull(np.array([[-1.0], [1.0]])))
        self.assertFalse(in_hull(np.array([[-1.0, 1.0], [1.0, 1.0]])))

    def test_empty(self) -> None:
        self.assertFalse(in_hull(np.empty((0, 3))))

    def test_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            in_hull(np.array([[np.nan]]))

    def test_deletion(self) -> None:
        matrix = np.array([[-1.0], [1.0]])
        self.assertTrue(in_hull(matrix))
        self.assertFalse(in_hull(np.delete(matrix, 0, axis=0)))
