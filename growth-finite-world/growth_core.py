"""Small, tested functions behind every number in the article.

Valuation: a perpetuity, the growth a price implies, and how much of a value sits beyond a horizon.
Saturation: how long a firm can outgrow its market, and the growth path of a logistic market share.
Growth accounting: compound rates, the split of GDP growth into energy growth and energy productivity,
and the classification of a country as decoupling or not.
Returns: the real total return of an index from monthly prices, dividends and CPI.
"""

import math

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------- valuation
def perpetuity_value(cash_flow, r, g=0.0):
    """Present value of a cash flow received every year for ever, first payment in one year,
    growing at g per year, discounted at r. Requires r > g."""
    if r <= g:
        raise ValueError("a growing perpetuity has a finite value only when r > g")
    return cash_flow / (r - g)


def implied_growth(price, cash_flow, r):
    """The perpetual growth rate g that makes price = cash_flow / (r - g)."""
    return r - cash_flow / price


def share_beyond(years, r, g=0.0):
    """Fraction of a growing perpetuity's value that arrives after `years` years.

    The payments from year years+1 onward are worth ((1+g)/(1+r))**years of the whole."""
    if r <= g:
        raise ValueError("requires r > g")
    return ((1 + g) / (1 + r)) ** years


def years_to_share(fraction, r, g=0.0):
    """Number of years after which only `fraction` of the value remains to be received."""
    return math.log(fraction) / math.log((1 + g) / (1 + r))


def halving_time(rate):
    """Years for a quantity shrinking by `rate` per year (e.g. real value under inflation) to halve."""
    return math.log(2) / math.log(1 + rate)


def doubling_time(rate):
    """Years for a quantity growing at `rate` per year to double."""
    return math.log(2) / math.log(1 + rate)


# -------------------------------------------------------------------------------------------- saturation
def years_to_saturate(share0, g_firm, g_market):
    """Years until a firm growing at g_firm in a market growing at g_market goes from share0 to 100%."""
    if g_firm <= g_market:
        return math.inf
    return math.log(1 / share0) / math.log((1 + g_firm) / (1 + g_market))


def logistic_share(t, share0, k):
    """Market share following a logistic curve with initial share share0 and rate k, at times t."""
    t = np.asarray(t, dtype=float)
    return 1.0 / (1.0 + (1.0 / share0 - 1.0) * np.exp(-k * t))


def firm_growth_path(t, share0, k, g_market):
    """Annual revenue growth of a firm whose share is logistic, in a market growing at g_market.

    Revenue(t) = share(t) * (1 + g_market)**t; returns growth from t-1 to t for t >= 1."""
    t = np.asarray(t, dtype=float)
    rev = logistic_share(t, share0, k) * (1 + g_market) ** t
    return rev[1:] / rev[:-1] - 1


# ----------------------------------------------------------------------------------- growth accounting
def cagr(v0, v1, years):
    """Compound annual growth rate from v0 to v1 over `years` years."""
    return (v1 / v0) ** (1 / years) - 1


def required_productivity_growth(g_output, g_input):
    """Annual growth of output per unit of input needed for output to grow at g_output
    while the input grows at g_input."""
    return (1 + g_output) / (1 + g_input) - 1


def years_to_factor(factor, rate):
    """Years for a quantity changing at `rate` per year to change by `factor`."""
    return math.log(factor) / math.log(1 + rate)


def decompose_log_growth(gdp0, gdp1, e0, e1, years):
    """Split average log growth of GDP into log growth of energy and of GDP per unit of energy.

    Returns (gdp, energy, productivity) in percent per year; gdp = energy + productivity exactly."""
    g = math.log(gdp1 / gdp0) / years
    e = math.log(e1 / e0) / years
    return 100 * g, 100 * e, 100 * (g - e)


def classify_decoupling(gdp_change, input_change):
    """'absolute' if GDP rose and the input fell, 'relative' if both rose but GDP rose faster,
    'none' otherwise. Changes are ratios end/start."""
    if gdp_change > 1 and input_change < 1:
        return "absolute"
    if gdp_change > 1 and input_change >= 1 and gdp_change > input_change:
        return "relative"
    return "none"


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


# --------------------------------------------------------------------------------------------- returns
def real_total_return_index(price, dividend, cpi):
    """Monthly real total-return index from monthly price, annualised dividend and CPI.

    Each month the investor receives one twelfth of the annual dividend rate and reinvests it."""
    price = pd.Series(price, dtype=float).reset_index(drop=True)
    dividend = pd.Series(dividend, dtype=float).reset_index(drop=True)
    cpi = pd.Series(cpi, dtype=float).reset_index(drop=True)
    gross = (price + dividend / 12).iloc[1:].to_numpy() / price.iloc[:-1].to_numpy()
    inflation = cpi.iloc[1:].to_numpy() / cpi.iloc[:-1].to_numpy()
    return pd.Series(np.concatenate([[1.0], np.cumprod(gross / inflation)]))
