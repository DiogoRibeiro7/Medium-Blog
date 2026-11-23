"""Small numerical tools used by run_analysis.py. Pure functions, tested in tests/test_core.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gabaix_ibragimov(sizes: np.ndarray, n: int) -> tuple[float, float]:
    """Pareto tail exponent of the n largest units: OLS of log(rank - 1/2) on log size.

    Gabaix and Ibragimov (2011) show that the shift of 1/2 removes most of the small-sample bias
    of the rank-size regression; the standard error is zeta * sqrt(2 / n).
    """
    x = np.sort(np.asarray(sizes, dtype=float))[::-1][:n]
    rank = np.arange(1, n + 1)
    slope = np.polyfit(np.log(x), np.log(rank - 0.5), 1)[0]
    zeta = -slope
    return float(zeta), float(zeta * np.sqrt(2.0 / n))


def ols_hc1(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with an intercept added; returns coefficients, HC1 standard errors and R squared."""
    y = np.asarray(y, dtype=float)
    Z = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    n, k = Z.shape
    bread = np.linalg.inv(Z.T @ Z)
    meat = (Z * resid[:, None] ** 2).T @ Z
    cov = bread @ meat @ bread * n / (n - k)
    r2 = 1.0 - resid.var() / y.var()
    return beta, np.sqrt(np.diag(cov)), float(r2)


def winsor_z(x: pd.Series, lo: float = 0.025, hi: float = 0.975) -> pd.Series:
    """Standardise after clipping to the given quantiles, so a few extreme units do not set the scale."""
    clipped = x.clip(x.quantile(lo), x.quantile(hi))
    return (clipped - clipped.mean()) / clipped.std(ddof=0)


def dimension_scores(indicators: pd.DataFrame, dims: dict[str, list[str]]) -> pd.DataFrame:
    """Mean of the standardised indicators of each dimension, using the indicators that are present."""
    z = indicators.apply(winsor_z)
    return pd.DataFrame({d: z[cols].mean(axis=1, skipna=True) for d, cols in dims.items()})


def composite(scores: pd.DataFrame, weights: np.ndarray | None = None) -> pd.Series:
    w = np.full(scores.shape[1], 1.0 / scores.shape[1]) if weights is None else np.asarray(weights)
    return pd.Series(scores.values @ w, index=scores.index)


def rank_under_random_weights(scores: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    """Rank of each unit (1 = most pressure) under Dirichlet(1, ..., 1) weights on the dimensions.

    Returns the median rank, the 5th and 95th percentile ranks and the share of draws in which
    the unit is among the 30 highest.
    """
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(scores.shape[1]), size=draws)
    S = scores.values @ W.T  # units x draws
    ranks = (-S).argsort(axis=0).argsort(axis=0) + 1
    return pd.DataFrame(
        {
            "rank_median": np.median(ranks, axis=1),
            "rank_p05": np.percentile(ranks, 5, axis=1),
            "rank_p95": np.percentile(ranks, 95, axis=1),
            "p_top30": (ranks <= 30).mean(axis=1),
        },
        index=scores.index,
    )


def leave_group_out_r2(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> float:
    """Out-of-sample R squared when each group (a NUTS III region) is predicted from the others."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    pred = np.empty_like(y)
    for g in np.unique(groups):
        test = groups == g
        Z = np.column_stack([np.ones((~test).sum()), X[~test]])
        beta, *_ = np.linalg.lstsq(Z, y[~test], rcond=None)
        pred[test] = np.column_stack([np.ones(test.sum()), X[test]]) @ beta
    return float(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1])


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna()
    v, w = values[ok].to_numpy(), weights[ok].to_numpy()
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cw, 0.5 * cw[-1])])
