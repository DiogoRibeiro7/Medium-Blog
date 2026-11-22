"""Small, tested functions behind every number in the article.

The identity. For an industry, or a group of industries, the implicit price of value added is

    P = w / (a * s),   so   ln P = ln w - ln a - ln s

where w is compensation per hour worked by employees, a = Y / H is real value added per hour worked
by everyone, and s = w * H / (P * Y) is the labour share adjusted for the self-employed (their hours
are valued at the employees' hourly compensation). The three terms add up to the price exactly, in
log points, for any two years. Comparing two groups, the price gap is the wage gap, minus the
productivity gap, minus the labour-share gap.

Aggregation: a Tornqvist volume index, because chain-linked volumes of different industries cannot
be summed. Comparisons: covariance shares of a variance, OLS slopes with HC1 errors, elasticities
across countries and forecast errors.
"""

import math

import numpy as np
import pandas as pd

TABLES = ("cp", "clv", "d1", "hours", "emp_hours")


# ------------------------------------------------------------------------------------------ aggregation
def tornqvist_log_volume(volumes, nominal):
    """Cumulative log change of a Tornqvist volume index (first year = 0).

    volumes, nominal: DataFrames indexed by year, one column per industry. Each year's log growth is
    the sum over industries of the log growth of the volume, weighted by the mean of the industry's
    nominal share in the two years."""
    volumes = volumes.sort_index()
    nominal = nominal.loc[volumes.index, volumes.columns]
    shares = nominal.div(nominal.sum(axis=1), axis=0)
    weights = 0.5 * (shares + shares.shift(1))
    dlog = np.log(volumes).diff()
    growth = (weights * dlog).sum(axis=1, min_count=1)
    growth.iloc[0] = 0.0
    return growth.cumsum()


def block_identity(t, cols, y0, y1):
    """Cumulative log changes since y0 of the price, wage, productivity and labour-share terms.

    t: dict of DataFrames (index year, one column per industry) with keys
       cp (value added, current prices), clv (chain-linked volumes), d1 (compensation of employees),
       hours (hours worked by everyone) and emp_hours (hours worked by employees).
    Returns a DataFrame indexed by year with columns price, wage, prod and lshare, where
    price = wage - prod - lshare holds exactly in every year."""
    cols = list(cols)
    sub = {k: t[k].loc[y0:y1, cols] for k in TABLES}
    for k, df in sub.items():
        if df.isna().any().any() or len(df) != y1 - y0 + 1:
            raise ValueError(f"missing {k} for {cols} in {y0}-{y1}")

    def total(k):
        s = sub[k].sum(axis=1)
        return np.log(s / s.iloc[0])

    volume = tornqvist_log_volume(sub["clv"], sub["cp"]) if len(cols) > 1 else np.log(sub["clv"][cols[0]] / sub["clv"][cols[0]].iloc[0])
    price = total("cp") - volume
    wage = total("d1") - total("emp_hours")
    prod = volume - total("hours")
    lshare = wage - prod - price
    return pd.DataFrame({"price": price, "wage": wage, "prod": prod, "lshare": lshare})


def gap(a, b):
    """Gap between two groups, as contributions to the price gap: the wage gap, minus the
    productivity gap ("productivity"), minus the labour-share gap ("margins"). Rows are years."""
    d = a - b
    return pd.DataFrame({"price": d["price"], "wage": d["wage"], "productivity": -d["prod"], "margins": -d["lshare"]})


def substitute_productivity(t, cols, path):
    """Copy of t in which the volume of each industry in cols grows with its own hours times a given
    productivity path (a Series of cumulative log changes indexed by year, 0 in the first year).
    Current-price value added, compensation and hours are unchanged, so only the price moves."""
    out = {k: t[k].copy() for k in TABLES}
    years = path.index
    for c in cols:
        h = t["hours"].loc[years, c]
        out["clv"].loc[years, c] = t["clv"].loc[years[0], c] * (h / h.iloc[0]) * np.exp(path)
    return out


# ----------------------------------------------------------------------------------------- statistics
def covariance_shares(price, parts, weights=None):
    """Share of the cross-sectional variance of `price` carried by each part: cov(part, price) / var(price).

    When the parts add up to the price, the shares add up to one. Optional weights (normalised here)."""
    p = np.asarray(price, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    pc = p - w @ p
    var = w @ (pc * pc)
    out = {}
    for name, x in parts.items():
        x = np.asarray(x, dtype=float)
        out[name] = float(w @ ((x - w @ x) * pc) / var)
    return out


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
    return float(beta[1]), math.sqrt(cov[1, 1]), float(r2)


def within(values, groups):
    """Values minus their group mean (country fixed effects by demeaning)."""
    s = pd.Series(np.asarray(values, dtype=float))
    g = pd.Series(np.asarray(groups))
    return (s - s.groupby(g).transform("mean")).to_numpy()


def rmse(errors):
    e = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(e * e)))


# ------------------------------------------------------------------------------------------- units
def pct(log_points):
    """A log change (in log units) as a percentage change."""
    return 100 * (math.exp(log_points) - 1)


def per_year(log_points, years):
    """A log change spread evenly over `years`, in percent per year (log points)."""
    return 100 * log_points / years


def ratio_range(values):
    """Largest value divided by the smallest."""
    v = pd.Series(values, dtype=float).dropna()
    return float(v.max() / v.min())
