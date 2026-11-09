"""How to point the matrix and the backward rule at your own model and series.

    python example_your_model.py

The model here is scikit-learn gradient boosting on twelve lags, wrapped so that it
issues a Gaussian forecast (its spread comes from out-of-fold residuals). The series is
simulated: an AR(1) whose volatility halves two thirds of the way through. Replace
``make_rows`` and ``factory`` with your own feature builder and model.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from backward_eval import ResidualGaussian, backward_window, transfer_matrix
from dgp import simulate, step

LAGS, BLOCK, GAP = 12, 120, 12


def make_rows(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows in time order: the previous ``lags`` values as features, the next value as target."""
    X = np.column_stack([y[lags - k : len(y) - k] for k in range(1, lags + 1)])
    return X, y[lags:]


def factory() -> ResidualGaussian:
    return ResidualGaussian(lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05,
                                                                  min_samples_leaf=20, random_state=0))


def main() -> None:
    n = 1200 + LAGS
    mu, phi, sigma = np.zeros(n), np.full(n, 0.6), step(n, 800 + LAGS, 2.0, 1.0)
    y = simulate(mu, phi, sigma, np.random.default_rng(3))
    X, z = make_rows(y, LAGS)

    res = transfer_matrix(X, z, factory, BLOCK, gap=GAP, folds=5, seed=0)
    E = res.excess("nlpd")
    C = res.mean_matrix("cover90")
    k = res.k
    print("Excess log loss of each block's model (rows) on each block (columns):")
    print(np.array2string(E, precision=2, suppress_small=True, max_line_width=140))
    print("\nToday's model on past blocks, oldest first (excess log loss / 90% coverage):")
    for j in range(k - 1):
        print(f"  block {j + 1}: {E[k - 1, j]:+.2f} / {C[k - 1, j]:.2f}")
    ch = backward_window(X, z, len(z), factory, BLOCK, max_back=k - 1, z_crit=2.5, gap=GAP,
                         rng=np.random.default_rng(0))
    print(f"\nBackward rule: train on the last {ch.rows} rows (stopped at block {ch.stopped_at} back); "
          f"the volatility change is {len(z) - 800} rows before the end.")


if __name__ == "__main__":
    main()
