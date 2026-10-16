import numpy as np
import pandas as pd

from hansard_drift.estimate import _q, draw_theta


def test_stratified_posterior_covers_the_truth():
    """Draw validation samples from a population with known stratum shares many times and
    check that the 95% interval for the corrected total covers the truth about 95% of the time."""
    rng = np.random.default_rng(3)
    N = np.array([4000, 600, 30000, 2500])
    p = np.array([0.92, 0.6, 0.04, 0.15])
    n = np.array([30, 25, 25, 10])
    truth = float((N * p).sum())
    covered = 0
    reps = 300
    for r in range(reps):
        k = rng.binomial(n, p)
        t = pd.DataFrame({"N": N, "n": n, "k": k})
        theta = draw_theta(t, 3000, 0.5, seed=r)
        q = _q((theta * N).sum(axis=1))
        covered += q["lo"] <= truth <= q["hi"]
    assert 0.90 <= covered / reps <= 0.995


def test_zero_positive_stratum_keeps_uncertainty():
    t = pd.DataFrame({"N": [10000], "n": [25], "k": [0]})
    theta = draw_theta(t, 5000, 0.5, seed=1)
    q = _q(theta[:, 0])
    assert q["lo"] >= 0 and q["hi"] > 0.03
