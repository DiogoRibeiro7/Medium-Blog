"""Small, tested functions behind every number in the article.

Decoupling: log changes, the split of a change in material use into a change in GDP and a change in
material intensity, and each material group's contribution to a total change.
Drift and growth: a regression of the yearly log change in material use on the yearly log change in
GDP, with country effects, whose intercept is the drift at zero growth and whose slope is the
response to growth; the growth rate at which the two cancel.
Circularity: Eurostat's circular material use rate, the recycled tonnage it implies, and the
recycling or material use needed for a target rate.
"""

import math

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------- units
def log_change(v0, v1):
    """Log change from v0 to v1 (natural log units)."""
    return math.log(v1 / v0)


def per_year(log_points, years):
    """A log change spread evenly over `years`, in percent per year."""
    return 100 * log_points / years


def pct(log_points):
    """A log change as a percentage change."""
    return 100 * (math.exp(log_points) - 1)


# --------------------------------------------------------------------------------------- decoupling
def intensity_split(m0, m1, y0, y1):
    """Split the log change of material use into the log change of GDP and of material use per unit
    of GDP. Returns (material, gdp, intensity), with material = gdp + intensity exactly."""
    m = log_change(m0, m1)
    y = log_change(y0, y1)
    return m, y, m - y


def contributions(parts0, parts1):
    """Each part's share of the change in a total, in levels: (part1 - part0) / (total1 - total0).

    parts0, parts1: mappings from part name to value. The shares add up to one."""
    total = sum(parts1.values()) - sum(parts0.values())
    return {k: (parts1[k] - parts0[k]) / total for k in parts0}


# ---------------------------------------------------------------------------------- drift and growth
def ols_slope(x, y):
    """Least-squares slope of y on x with an intercept: slope, HC1 standard error, R squared, intercept."""
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
    return float(beta[1]), math.sqrt(cov[1, 1]), float(r2), float(beta[0])


def within_slope(y, x, groups):
    """Slope of y on x with a separate intercept per group (fixed effects), its standard error
    clustered by group, and the group intercepts (a Series indexed by group)."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "x": np.asarray(x, dtype=float), "g": np.asarray(groups)})
    yd = df.y - df.groupby("g").y.transform("mean")
    xd = df.x - df.groupby("g").x.transform("mean")
    b = float((xd * yd).sum() / (xd * xd).sum())
    resid = yd - b * xd
    score = (xd * resid).groupby(df.g).sum()
    se = math.sqrt(float((score**2).sum())) / float((xd * xd).sum())
    intercepts = df.groupby("g").y.mean() - b * df.groupby("g").x.mean()
    return b, se, intercepts


def breakeven_growth(drift, response):
    """Growth rate at which material use stays flat: drift + response * growth = 0."""
    return -drift / response


# ------------------------------------------------------------------------------------- circularity
def recycled_from_rate(rate, dmc):
    """Recycled tonnage implied by a circular material use rate (a fraction) and DMC:
    rate = U / (DMC + U), so U = rate * DMC / (1 - rate)."""
    return rate * dmc / (1 - rate)


def circular_rate(recycled, dmc):
    """Circular material use rate (a fraction): recycled over DMC plus recycled."""
    return recycled / (dmc + recycled)


def recycled_needed(target, material_use):
    """Recycled tonnage needed for a target rate when total material use (DMC + recycled) is fixed."""
    return target * material_use


def material_use_allowed(target, recycled):
    """Total material use (DMC + recycled) compatible with a target rate at a given recycled tonnage."""
    return recycled / target


def max_window_increase(series, window):
    """Largest increase of a yearly series over `window` years, and the starting year."""
    s = pd.Series(series).dropna()
    best = None
    for y in s.index:
        if y + window in s.index:
            d = s[y + window] - s[y]
            if best is None or d > best[0]:
                best = (float(d), int(y))
    return best
