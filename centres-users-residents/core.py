"""Numerical tools used by run_analysis.py. Pure functions, tested in tests/test_core.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_growth(x0: pd.Series, x1: pd.Series, years: pd.Series) -> pd.Series:
    """Annual growth in per cent, as 100 times the log change divided by the number of years."""
    return 100.0 * np.log(x1 / x0) / years


def ring(fua: pd.Series, city: pd.Series) -> pd.Series:
    """The part of the functional urban area outside the city."""
    return fua - city


def demean(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract each group's mean: what is left varies only within groups (here, within countries)."""
    return df - df.groupby(groups).transform("mean")


def ols_hc1(y: np.ndarray, X: np.ndarray, intercept: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with HC1 standard errors and R squared (computed on y as given)."""
    y = np.asarray(y, dtype=float)
    Z = np.asarray(X, dtype=float)
    if intercept:
        Z = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    n, k = Z.shape
    bread = np.linalg.inv(Z.T @ Z)
    cov = bread @ ((Z * resid[:, None] ** 2).T @ Z) @ bread * n / (n - k)
    tss = ((y - y.mean()) ** 2).sum() if intercept else (y**2).sum()
    return beta, np.sqrt(np.diag(cov)), float(1.0 - (resid**2).sum() / tss)


def fixed_effects(y: pd.Series, X: pd.DataFrame, groups: pd.Series) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Within (fixed-effects) estimator: OLS on group-demeaned data, HC1 errors with the degrees of
    freedom corrected for the group means. Returns coefficients, standard errors, within R squared and
    the number of groups."""
    yd = demean(y.to_frame(), groups).iloc[:, 0].to_numpy()
    Xd = demean(X, groups).to_numpy()
    beta, se, r2 = ols_hc1(yd, Xd, intercept=False)
    n, k, g = len(yd), X.shape[1], groups.nunique()
    se = se * np.sqrt((n - k) / (n - k - g))
    return beta, se, r2, g


def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1])


def percentile_of(values: pd.Series, x: float) -> float:
    """Share of values strictly below x, in per cent (0 = lowest, 100 = above every value)."""
    v = values.dropna()
    return float(100.0 * (v < x).mean())
