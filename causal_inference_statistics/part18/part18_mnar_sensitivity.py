"""Population selection-bias sensitivity for Part 18.

Uses deterministic Gauss-Hermite integration over the independent standard
normal baseline covariate and outcome noise. No Monte Carlo sampling is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit


@dataclass(frozen=True)
class SelectionBenchmark:
    """Observed-data population quantities for a missingness scenario."""

    outcome_dependence: float
    observed_fraction: float
    complete_case_effect: float
    bias: float


def population_complete_case_effect(
    outcome_dependence: float, *, order: int = 40
) -> SelectionBenchmark:
    """Compute population OLS under Pr(R=1|A,X,Y)=expit(.8-X+.6A-gamma*Y).

    A is Bernoulli(.5), X and outcome noise are independent N(0,1), and
    Y=A+1.5X+noise. The selected population OLS includes intercept, A and X.

    Args:
        outcome_dependence: Nonnegative gamma multiplying Y in logit selection.
        order: Positive Gaussian quadrature node count per normal dimension.

    Returns:
        Selected fraction and the selected-population OLS treatment coefficient.

    Raises:
        ValueError: For invalid sensitivity or quadrature parameters.
    """
    if not np.isfinite(outcome_dependence) or outcome_dependence < 0:
        raise ValueError("outcome_dependence must be finite and nonnegative")
    if type(order) is not int or order < 3:
        raise ValueError("order must be an integer of at least 3")

    nodes, weights = np.polynomial.hermite.hermgauss(order)
    normal_nodes: NDArray[np.float64] = np.sqrt(2.0) * nodes
    normal_weights: NDArray[np.float64] = weights / np.sqrt(np.pi)
    x, error = np.meshgrid(normal_nodes, normal_nodes, indexing="ij")
    joint_weights = np.outer(normal_weights, normal_weights)
    matrix = np.zeros((3, 3), dtype=float)
    vector = np.zeros(3, dtype=float)
    observed_fraction = 0.0

    for treatment in (0.0, 1.0):
        y = treatment + 1.5 * x + error
        response_prob = expit(0.8 - x + 0.6 * treatment - outcome_dependence * y)
        weighted_prob = 0.5 * joint_weights * response_prob
        design = np.stack((np.ones_like(x), np.full_like(x, treatment), x), axis=-1)
        matrix += np.einsum("ij,ijk,ijl->kl", weighted_prob, design, design)
        vector += np.einsum("ij,ijk,ij->k", weighted_prob, design, y)
        observed_fraction += float(weighted_prob.sum())

    coefficient = float(np.linalg.solve(matrix, vector)[1])
    return SelectionBenchmark(
        outcome_dependence=outcome_dependence,
        observed_fraction=observed_fraction,
        complete_case_effect=coefficient,
        bias=coefficient - 1.0,
    )


if __name__ == "__main__":
    for gamma in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5):
        print(population_complete_case_effect(gamma))
