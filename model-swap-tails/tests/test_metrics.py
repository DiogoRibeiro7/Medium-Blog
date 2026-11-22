import numpy as np

from swaptails.metrics import block_indices, bootstrap, cvar, kendall_tau, rank, wape


def test_wape_and_cvar():
    assert wape([1, 2], [10, 10]) == 0.15
    v = np.arange(1, 101, dtype=float)
    assert cvar(v, 0.95) == np.mean(v[-5:])
    assert cvar(np.array([np.nan, 1.0]), 0.5) == 1.0


def test_block_indices_are_circular_blocks():
    idx = block_indices(10, 3, 5, seed=0)
    assert idx.shape == (5, 10)
    for row in idx:
        assert ((np.diff(row[:3]) % 10) == 1).all()


def test_paired_bootstrap_of_identical_runs():
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 2, size=(1, 2, 50))
    y = rng.uniform(10, 20, size=(1, 2, 50))
    a2, y2 = np.concatenate([a, a]), np.concatenate([y, y])
    valid = np.ones((2, 50), dtype=bool)
    reps = bootstrap(a2, y2, valid, 0.9, 7, 100, 0)
    assert np.allclose(reps["wape"][:, 0], reps["wape"][:, 1])
    assert np.allclose(reps["cvar"][:, 0], reps["cvar"][:, 1])
    assert (reps["cvar"][:, 0] >= reps["wape"][:, 0] * 0.5).all()


def test_rank_and_tau():
    r1 = rank({"a": 0.1, "b": 0.2, "c": 0.3})
    r2 = rank({"a": 0.3, "b": 0.2, "c": 0.1})
    assert r1 == {"a": 1, "b": 2, "c": 3}
    assert kendall_tau(r1, r1) == 1.0 and kendall_tau(r1, r2) == -1.0
