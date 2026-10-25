"""Model fits and the rules for combining estimates from synthetic data.

Notation, for one coefficient and m synthetic data sets of n_syn rows each,
synthesised from n_obs real rows (k = n_syn / n_obs):

    q_l      estimate from synthetic data set l
    u_l      its squared standard error, computed as if the data were real
    qbar     mean of the q_l
    ubar     mean of the u_l
    b        between-data-set variance of the q_l (divisor m - 1)

Rules for the variance of qbar:

    naive            ubar / m           treat the stacked synthetic data as real
    simple           ubar (k + 1/m)     synthesis from the fitted generator
    proper           ubar (k + (1+k)/m) synthesis after drawing the parameters
    raghunathan      (1 + 1/m) b - ubar fully synthetic data, proper synthesis

The "simple" and "proper" rules are the T_s estimators of Raab, Nowok and
Dibben (2016) without and with the posterior predictive draw; they only need
ubar. The Raghunathan-Reiter-Rubin (2003) rule estimates the extra variance
from b, which makes it noisy and sometimes negative; when it is not positive
we use Reiter's (2002) adjustment, k * ubar, and count how often that happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with an intercept already in X; classical variances."""
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    var = sigma2 * np.diag(np.linalg.inv(XtX))
    return beta, var


def logit(X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton-Raphson; model-based variances.

    A tiny ridge (1e-8) keeps the Hessian invertible when a synthetic data
    set happens to separate a category perfectly.
    """
    p = X.shape[1]
    beta = np.zeros(p)
    ridge = 1e-8 * np.eye(p)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu)
        H = (X * w[:, None]).T @ X + ridge
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    H = (X * (mu * (1 - mu))[:, None]).T @ X + ridge
    return beta, np.diag(np.linalg.inv(H))


@dataclass
class Interval:
    estimate: np.ndarray
    variance: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fallback: np.ndarray  # True where the rule needed the fallback variance


def _interval(est, var, df=None, level=0.95, fallback=None):
    if df is None:
        crit = stats.norm.ppf(0.5 + level / 2)
    else:
        crit = stats.t.ppf(0.5 + level / 2, df)
    half = crit * np.sqrt(var)
    if fallback is None:
        fallback = np.zeros_like(est, dtype=bool)
    return Interval(est, var, est - half, est + half, fallback)


def combine(q: np.ndarray, u: np.ndarray, k: float, rule: str) -> Interval:
    """Combine m estimates (rows of q, u) under one of the rules above."""
    q = np.atleast_2d(q)
    u = np.atleast_2d(u)
    m = q.shape[0]
    qbar = q.mean(axis=0)
    ubar = u.mean(axis=0)
    if rule == "naive":
        return _interval(qbar, ubar / m)
    if rule == "simple":
        return _interval(qbar, ubar * (k + 1.0 / m))
    if rule == "proper":
        return _interval(qbar, ubar * (k + (1.0 + k) / m))
    if rule == "raghunathan":
        if m < 2:
            raise ValueError("the Raghunathan-Reiter-Rubin rule needs m >= 2")
        b = q.var(axis=0, ddof=1)
        T = (1.0 + 1.0 / m) * b - ubar
        bad = T <= 0
        T = np.where(bad, k * ubar, T)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(bad, 0.0, ubar / ((1.0 + 1.0 / m) * b))
        df = np.where(bad, np.inf, (m - 1) * (1.0 - r) ** 2)
        df = np.maximum(df, 1.0)
        return _interval(qbar, T, df=df, fallback=bad)
    raise ValueError(f"unknown rule {rule!r}")


def covers(iv: Interval, truth: np.ndarray) -> np.ndarray:
    return (iv.lower <= truth) & (truth <= iv.upper)


def predicted_coverage(var_ratio: float, level: float = 0.95) -> float:
    """Coverage of a normal interval whose variance is too small by var_ratio.

    If the true sampling variance is r times the variance the interval uses,
    the nominal z-interval covers with probability 2 Phi(z / sqrt(r)) - 1.
    """
    z = stats.norm.ppf(0.5 + level / 2)
    return float(2 * stats.norm.cdf(z / np.sqrt(var_ratio)) - 1)
