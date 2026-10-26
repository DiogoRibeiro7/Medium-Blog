"""The decision: how many rooms to put back on sale for one arrival night.

At decision time the hotel holds n visible bookings for arrival date d. It may
resell q of their rooms. Each resold room earns r. If fewer than q of the n
bookings release their room (N < q), q - N guests arrive to no room and are
walked at a cost w each. Profit relative to reselling nothing:

    profit(q) = r q - w max(q - N, 0).

The expected-profit maximiser is the largest q with P(N <= q - 1) <= r / w,
which needs the distribution of N, not only its mean.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def poisson_binomial_pmf(p: np.ndarray) -> np.ndarray:
    """Exact distribution of a sum of independent Bernoulli(p_i)."""
    pmf = np.ones(1)
    for pi in np.asarray(p, dtype=float):
        nxt = np.zeros(len(pmf) + 1)
        nxt[:-1] += pmf * (1.0 - pi)
        nxt[1:] += pmf * pi
        pmf = nxt
    return pmf


def block_pmf(block_prob: np.ndarray, block_size: np.ndarray) -> np.ndarray:
    """Distribution of N when each block of bookings releases all its rooms or none.

    Bookings entered together by the same agent for the same arrival are
    cancelled together, so the unit of risk is the block. Block b releases
    ``block_size[b]`` rooms with probability ``block_prob[b]``.
    """
    pmf = np.ones(1)
    for pb, sb in zip(np.asarray(block_prob, dtype=float), np.asarray(block_size, dtype=int)):
        nxt = np.zeros(len(pmf) + sb)
        nxt[: len(pmf)] += pmf * (1.0 - pb)
        nxt[sb:] += pmf * pb
        pmf = nxt
    return pmf


def inflated_pmf(p: np.ndarray, dispersion: float) -> np.ndarray:
    """Discretised normal with the Poisson-binomial mean and inflated variance.

    Cancellations of bookings made together (groups, tour-operator blocks) are
    correlated, so the variance of N exceeds sum p(1-p). ``dispersion`` is the
    ratio of the observed to the independent variance.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    mean = p.sum()
    sd = np.sqrt(max(dispersion, 1.0) * max((p * (1 - p)).sum(), 1e-12))
    edges = np.arange(n + 2) - 0.5
    cdf = norm.cdf(edges, loc=mean, scale=sd)
    pmf = np.diff(cdf)
    pmf[0] += cdf[0]
    pmf[-1] += 1.0 - cdf[-1]
    return pmf


def optimal_release(pmf: np.ndarray, ratio: float) -> int:
    """Largest q with P(N <= q - 1) <= ratio, where ratio = r / w."""
    cdf = np.cumsum(pmf)
    # q = 0 is always allowed; q >= 1 is allowed when cdf[q - 1] <= ratio.
    allowed = np.flatnonzero(cdf[:-1] <= ratio + 1e-12)
    return int(allowed[-1] + 1) if len(allowed) else 0


def profit(q: np.ndarray, released: np.ndarray, price: np.ndarray, walk_multiple: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    released = np.asarray(released, dtype=float)
    price = np.asarray(price, dtype=float)
    return price * q - walk_multiple * price * np.maximum(q - released, 0.0)


# Interfaces between the model and the decision ---------------------------------


def release_from_labels(p: np.ndarray, threshold: float = 0.5) -> int:
    """The model sends a yes/no per booking; the decision counts the yeses."""
    return int((np.asarray(p) >= threshold).sum())


def release_from_mean(p: np.ndarray) -> int:
    """The model sends probabilities; the decision uses only their sum."""
    return int(np.round(np.sum(p)))


def release_from_distribution(p: np.ndarray, ratio: float, dispersion: float = 1.0) -> int:
    """The model sends probabilities; the decision uses the whole distribution of N."""
    pmf = poisson_binomial_pmf(p) if dispersion <= 1.0 else inflated_pmf(p, dispersion)
    return optimal_release(pmf, ratio)
