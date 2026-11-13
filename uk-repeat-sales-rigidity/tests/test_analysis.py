import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as A  # noqa: E402


def _hist(x):
    k = np.floor(x / A.BIN).astype(int)
    return pd.Series(k).value_counts().sort_index()


def test_density_jump_is_zero_without_anchoring():
    rng = np.random.default_rng(0)
    x = rng.normal(0.02, 0.08, 400_000)
    theta, se, _, _ = A.density_jump(_hist(x), 0.0)
    assert abs(theta) < 4 * se + 0.02


@pytest.mark.parametrize("q,mu,sd", [(0.2, 0.02, 0.08), (0.5, 0.02, 0.08), (0.3, -0.05, 0.08),
                                     (0.3, 0.10, 0.10), (0.3, 0.02, 0.15)])
def test_missing_share_recovers_simulated_anchoring(q, mu, sd):
    """Sellers facing a nominal loss sell at exactly the old price with probability q, in weak,
    normal and strong simulated markets."""
    rng = np.random.default_rng(1)
    x = rng.normal(mu, sd, 600_000)
    anchored = (x < 0) & (rng.random(x.size) < q)
    x = np.where(anchored, 0.0, x)
    counts = _hist(x)
    theta_ex, se, _, _ = A.density_jump(counts, 0.0, drop_exact=float(anchored.sum()))
    assert A.missing_share(theta_ex) == pytest.approx(q, abs=0.05)
    theta_all, _, _, _ = A.density_jump(counts, 0.0)
    assert theta_all > theta_ex                     # the spike at zero adds to the jump


def test_placebo_cutoff_has_no_jump():
    rng = np.random.default_rng(2)
    x = rng.normal(0.02, 0.08, 400_000)
    x = np.where((x < 0) & (rng.random(x.size) < 0.4), 0.0, x)
    theta, se, _, _ = A.density_jump(_hist(x), 0.10)
    assert abs(theta) < 4 * se + 0.03
