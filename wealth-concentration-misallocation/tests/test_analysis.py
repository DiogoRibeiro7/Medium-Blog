import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import ols  # noqa: E402


def test_ols_coefficients_and_hc1():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = 1.0 + 2.0 * x + rng.standard_normal(n) * (1 + np.abs(x))
    X = np.column_stack([np.ones(n), x])
    b, se = ols(y, X)
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    u = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * u**2) @ X @ XtXi * n / (n - 2)
    assert np.allclose(se, np.sqrt(np.diag(V)))


def test_cluster_se_with_one_obs_per_cluster_equals_hc1():
    rng = np.random.default_rng(1)
    n = 300
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([0.5, -1.0]) + rng.standard_normal(n)
    _, se_hc1 = ols(y, X)
    _, se_cr1 = ols(y, X, np.arange(n))
    # with G = n clusters, CR1's factor G/(G-1)*(n-1)/(n-k) equals HC1's n/(n-k)
    assert np.allclose(se_cr1, se_hc1, rtol=1e-12)
