"""Small, tested functions behind every number in the article.

Aggregation: a Tornqvist volume index for a group of industries, because chain-linked volumes of
different industries cannot be added.
Growth accounting: compound rates, and the exact split of output growth into input growth and
output per unit of input, in log points.
Comparisons: decoupling classes, OLS slopes with HC1 errors, and the arithmetic of required rates.
"""

import math

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------ aggregation
def tornqvist_index(volumes, nominal):
    """Volume index (first year = 1) for a group of industries.

    volumes, nominal: DataFrames indexed by year, one column per industry, with chain-linked volumes
    and current-price values. Each year's log growth is the sum over industries of the log growth of
    the volume, weighted by the mean of the industry's nominal share in the two years."""
    volumes = volumes.sort_index()
    nominal = nominal.loc[volumes.index, volumes.columns]
    shares = nominal.div(nominal.sum(axis=1), axis=0)
    weights = 0.5 * (shares + shares.shift(1))
    dlog = np.log(volumes).diff()
    growth = (weights * dlog).sum(axis=1, min_count=1)
    growth.iloc[0] = 0.0
    return np.exp(growth.cumsum())


# ------------------------------------------------------------------------------------ growth accounting
def cagr(v0, v1, years):
    """Compound annual growth rate from v0 to v1 over `years` years."""
    return (v1 / v0) ** (1 / years) - 1


def log_rate(v0, v1, years):
    """Average log growth, in percent per year."""
    return 100 * math.log(v1 / v0) / years


def decompose(output0, output1, input0, input1, years):
    """Split the log growth of output into the log growth of an input and of output per unit of it.

    Returns (output, input, output per input) in percent per year; output = input + output per input."""
    o = log_rate(output0, output1, years)
    i = log_rate(input0, input1, years)
    return o, i, o - i


def required_productivity_growth(g_output, g_input):
    """Growth of output per unit of input needed for output to grow at g_output while the input grows at g_input."""
    return (1 + g_output) / (1 + g_input) - 1


def years_to_factor(factor, rate):
    """Years for a quantity changing at `rate` per year to change by `factor`."""
    return math.log(factor) / math.log(1 + rate)


def classify_decoupling(output_ratio, input_ratio):
    """'absolute' if output rose and the input fell, 'relative' if both rose but output rose faster,
    'none' otherwise. Ratios are end / start."""
    if output_ratio > 1 and input_ratio < 1:
        return "absolute"
    if output_ratio > 1 and output_ratio > input_ratio >= 1:
        return "relative"
    return "none"


# ---------------------------------------------------------------------------------------- comparisons
def ols_slope(x, y):
    """Least-squares slope of y on x with an intercept, its HC1 standard error, and R squared."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    meat = X.T @ (X * resid[:, None] ** 2)
    cov = bread @ meat @ bread * n / (n - k)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return beta[1], math.sqrt(cov[1, 1]), r2


def ratio_range(values):
    """Largest value divided by the smallest."""
    v = pd.Series(values, dtype=float).dropna()
    return v.max() / v.min()


def coefficient_of_variation(values):
    """Standard deviation over mean, population form."""
    v = pd.Series(values, dtype=float).dropna()
    return v.std(ddof=0) / v.mean()


HOURS_PER_YEAR = 24 * 365.25
