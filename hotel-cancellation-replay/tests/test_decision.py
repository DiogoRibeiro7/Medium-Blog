import numpy as np
import pytest
from scipy.stats import binom

from cancelreplay import decision as D


def test_poisson_binomial_is_a_distribution_with_the_right_mean():
    p = np.random.default_rng(1).random(60)
    pmf = D.poisson_binomial_pmf(p)
    assert pmf.sum() == pytest.approx(1.0)
    assert (np.arange(len(pmf)) * pmf).sum() == pytest.approx(p.sum())


def test_equal_probabilities_give_the_binomial():
    pmf = D.poisson_binomial_pmf(np.full(25, 0.2))
    assert np.allclose(pmf, binom.pmf(np.arange(26), 25, 0.2))


def test_blocks_of_one_are_independent_bookings():
    p = np.random.default_rng(2).random(30)
    assert np.allclose(D.block_pmf(p, np.ones(30, dtype=int)), D.poisson_binomial_pmf(p))


def test_blocks_widen_the_distribution_with_the_same_mean():
    p = np.full(20, 0.15)
    one_block = D.block_pmf(np.array([0.15]), np.array([20]))
    k = np.arange(21)
    assert (k * one_block).sum() == pytest.approx((k * D.poisson_binomial_pmf(p)).sum())
    var = lambda pmf: (k ** 2 * pmf).sum() - (k * pmf).sum() ** 2  # noqa: E731
    assert var(one_block) > 10 * var(D.poisson_binomial_pmf(p))


def test_release_rule_is_the_quantile_and_moves_with_the_cost_ratio():
    pmf = D.poisson_binomial_pmf(np.full(40, 0.25))
    cdf = np.cumsum(pmf)
    q = D.optimal_release(pmf, 1 / 3)
    assert q == 0 or cdf[q - 1] <= 1 / 3
    assert cdf[q] > 1 / 3
    assert D.optimal_release(pmf, 1 / 5) <= q <= D.optimal_release(pmf, 1 / 2)


def test_rule_maximises_expected_profit():
    pmf = D.poisson_binomial_pmf(np.random.default_rng(3).random(30) * 0.4)
    k = np.arange(len(pmf))
    expected = [(pmf * D.profit(np.full_like(k, q, dtype=float), k, np.ones(len(k)), 3.0)).sum()
                for q in range(len(pmf))]
    assert D.optimal_release(pmf, 1 / 3) == int(np.argmax(expected))


def test_profit_counts_walks_at_the_walk_multiple():
    assert D.profit(np.array([5]), np.array([3]), np.array([100.0]), 3.0)[0] == 500 - 600
    assert D.profit(np.array([2]), np.array([3]), np.array([100.0]), 3.0)[0] == 200


def test_labels_interface_ignores_everything_below_the_threshold():
    assert D.release_from_labels(np.full(100, 0.3)) == 0
    assert D.release_from_mean(np.full(100, 0.3)) == 30
