"""Validate convex projection and its boundary behavior."""

import unittest

import numpy as np

from causal_tax_cuts.convex_approximation import project


class ProjectionTests(unittest.TestCase):
    """Exercise known geometric solutions."""

    def test_inside(self) -> None:
        matrix = np.array([[-1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(project(matrix), [0.5, 0.5])

    def test_outside(self) -> None:
        matrix = np.array([[1.0, 1.0], [1.0, -1.0]])
        np.testing.assert_allclose(project(matrix) @ matrix, [1.0, 0.0])

    def test_vertex(self) -> None:
        matrix = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(project(matrix), [1.0, 0.0])

    def test_duplicate(self) -> None:
        matrix = np.array([[1.0], [1.0]])
        weights = project(matrix)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(weights @ matrix[:, 0]), 1.0)

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            project(np.array([[np.nan]]))
