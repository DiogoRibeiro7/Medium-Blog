"""Fiscal reaction function estimators, written out in numpy.

pb_t = a + rho * d_{t-1} + gamma * gap_t + phi * pb_{t-1} + delta20 * 1[2020] + delta21 * 1[2021] + e_t

- ols: least squares with Newey-West (Bartlett) standard errors.
- huber: Huber M-estimation by iteratively reweighted least squares.
- pooled_fe: common slopes with country fixed effects, standard errors clustered by country.
- random_effects: DerSimonian-Laird pooling of country slopes and empirical Bayes shrinkage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["const", "debt_l", "gap", "pb_l", "y2020", "y2021"]


def design(df: pd.DataFrame, years=(1996, 2025), y_col: str = "pb", extra: dict | None = None):
    """Regressand and regressors for one country, years inclusive."""
    d = df.sort_values("year").copy()
    d["debt_l"] = d["debt"].shift(1)
    d["pb_l"] = d[y_col].shift(1)
    d = d[(d.year >= years[0]) & (d.year <= years[1])].dropna(subset=[y_col, "debt_l", "pb_l", "gap"])
    yr = d.year.values
    X = pd.DataFrame(
        {
            "const": 1.0,
            "debt_l": d.debt_l.values,
            "gap": d.gap.values,
            "pb_l": d.pb_l.values,
            "y2020": (yr == 2020).astype(float),
            "y2021": (yr == 2021).astype(float),
        },
        index=yr,
    )
    for name, start in (extra or {}).items():
        X[name] = (yr >= start).astype(float)
    X = X.loc[:, (X != 0).any(axis=0)]
    return d[y_col].values, X


def ols(y, X, hac_lags: int = 2) -> dict:
    y, Xv = np.asarray(y, float), np.asarray(X, float)
    n, k = Xv.shape
    xtx_inv = np.linalg.inv(Xv.T @ Xv)
    b = xtx_inv @ Xv.T @ y
    e = y - Xv @ b
    xe = Xv * e[:, None]
    meat = xe.T @ xe
    for lag in range(1, hac_lags + 1):
        w = 1 - lag / (hac_lags + 1)
        gmat = xe[lag:].T @ xe[:-lag]
        meat += w * (gmat + gmat.T)
    cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    tss = (y - y.mean()) @ (y - y.mean())
    cols = list(X.columns) if hasattr(X, "columns") else list(range(k))
    return dict(b=b, e=e, cov=cov, se=np.sqrt(np.diag(cov)), n=n, k=k, r2=1 - (e @ e) / tss, cols=cols)


def huber(y, X, c: float = 1.345, iters: int = 100) -> dict:
    y, Xv = np.asarray(y, float), np.asarray(X, float)
    b = np.linalg.lstsq(Xv, y, rcond=None)[0]
    for _ in range(iters):
        e = y - Xv @ b
        scale = np.median(np.abs(e - np.median(e))) / 0.6745
        r = np.abs(e) / max(scale, 1e-12)
        w = np.where(r <= c, 1.0, c / r)
        b_new = np.linalg.lstsq(Xv * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new
            break
        b = b_new
    return dict(b=b, w=w, cols=list(X.columns))


def pooled_fe(panel: pd.DataFrame) -> dict:
    ys, xs, ids = [], [], []
    slopes = ["debt_l", "gap", "pb_l"]
    for c, g in panel.groupby("country"):
        y, X = design(g)
        X = X.reindex(columns=COLUMNS, fill_value=0.0).drop(columns="const")
        Xd = X.copy()
        Xd[slopes] = X[slopes] - X[slopes].mean()
        ys.append(y - y.mean())
        xs.append(Xd.values)
        ids += [c] * len(y)
    Y, XX, ids = np.concatenate(ys), np.vstack(xs), np.array(ids)
    xtx_inv = np.linalg.inv(XX.T @ XX)
    b = xtx_inv @ XX.T @ Y
    e = Y - XX @ b
    groups = np.unique(ids)
    meat = sum(np.outer(XX[ids == c].T @ e[ids == c], XX[ids == c].T @ e[ids == c]) for c in groups)
    n, k, G = len(Y), XX.shape[1], len(groups)
    cov = xtx_inv @ meat @ xtx_inv * G / (G - 1) * (n - 1) / (n - k)
    return dict(b=b, se=np.sqrt(np.diag(cov)), cols=COLUMNS[1:], n=n, countries=G)


def random_effects(est, se) -> dict:
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se**2
    mu_fe = np.sum(w * est) / np.sum(w)
    q = np.sum(w * (est - mu_fe) ** 2)
    k = len(est)
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))
    wr = 1 / (se**2 + tau2)
    mu = np.sum(wr * est) / np.sum(wr)
    se_mu = np.sqrt(1 / np.sum(wr))
    shrink = tau2 / (tau2 + se**2)
    eb = shrink * est + (1 - shrink) * mu
    eb_se = np.sqrt(shrink * se**2)
    return dict(mu=mu, se_mu=se_mu, tau2=tau2, q=q, eb=eb, eb_se=eb_se, shrink=shrink)
