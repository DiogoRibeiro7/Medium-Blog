"""Summary statistics over replications.

Replications use common random numbers: replication ``r`` draws the same primary delays
and the same scenario events for every plan. Differences between plans are therefore
paired, and their confidence intervals are computed on the per-replication differences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Z_95 = 1.959963984540054


def mean_ci(values: Sequence[float]) -> dict[str, float]:
    """Mean, standard deviation and a normal-approximation 95% interval for the mean."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"mean": math.nan, "sd": math.nan, "lo": math.nan, "hi": math.nan, "n": 0.0}
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    half = Z_95 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": m, "sd": sd, "lo": m - half, "hi": m + half, "n": float(n)}


def paired_difference(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Mean of ``a - b`` over paired replications, with its 95% interval.

    Raises:
        ValueError: When the two samples are not the same length.

    """
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return mean_ci([x - y for x, y in zip(a, b, strict=True)])


def quantiles(values: Sequence[float], probs: Sequence[float]) -> list[float]:
    """Empirical quantiles, ignoring non-finite values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [math.nan for _ in probs]
    return [float(v) for v in np.quantile(x, probs)]


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated when both coordinates are minimised."""
    front: list[int] = []
    for i, (x, y) in enumerate(points):
        dominated = any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for j, (x2, y2) in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda k: points[k][0])
