"""Model families for daily series (PREREGISTRATION.md).

All families work on an aligned design: row ``i`` of every array is day ``i + MAX_LAG`` of the
transformed series ``y`` (log load, or log(1 + page views)), so that every lag exists.

* ``AR``   autoregression of order 7 with intercept;
* ``SLM``  seasonal linear model: lags 1-7, 14, 21, 28, day-of-week dummies, annual Fourier terms
           (two harmonics), intercept;
* ``ETS``  additive Holt-Winters without trend, weekly season (ETS(A,N,A)), smoothing parameters
           by least squares, states filtered through the data;
* ``GBM``  LightGBM regression on lags, rolling means and calendar features; its predictive scale
           comes from three contiguous out-of-fold residual blocks inside the training window.

Every family gives a Gaussian and a Student-t(5) predictive distribution. For AR and SLM the
Student-t version is a Student-t regression (maximum likelihood, IRLS), so extreme days barely move
the coefficients; for ETS and GBM the point forecast is the least-squares one and only the scale is
estimated by Student-t maximum likelihood on the residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

import common  # noqa: F401  (companion folders on sys.path)
from backward_eval import ARGaussian, ARStudentT

MAX_LAG = 28
T_DF = 5.0
SLM_LAGS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)
GBM_LAGS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)
GBM_PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=15, min_data_in_leaf=20,
                  feature_fraction=0.9, max_bin=63, verbose=-1, num_threads=1, deterministic=True,
                  force_row_wise=True, seed=1)
GBM_ROUNDS = 100
FAMILIES = ("AR", "SLM", "ETS", "GBM")


# ------------------------------------------------------------------ designs


def _lags(y: np.ndarray, lags) -> np.ndarray:
    n = len(y) - MAX_LAG
    return np.column_stack([y[MAX_LAG - k : MAX_LAG - k + n] for k in lags])


def _calendar(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    d = dates[MAX_LAG:]
    dow = np.asarray(d.dayofweek)
    doy = np.asarray(d.dayofyear, dtype=float)
    return dow, doy


@dataclass
class Design:
    z: np.ndarray            # target, aligned
    dates: pd.DatetimeIndex  # dates of the aligned rows
    X_ar: np.ndarray
    X_slm: np.ndarray
    X_gbm: np.ndarray


def design(y: np.ndarray, dates: pd.DatetimeIndex) -> Design:
    y = np.asarray(y, dtype=float)
    z = y[MAX_LAG:]
    one = np.ones((len(z), 1))
    dow, doy = _calendar(dates)
    dummies = np.column_stack([(dow == k).astype(float) for k in range(1, 7)])
    w = 2 * np.pi * doy / 365.25
    fourier = np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    X_ar = np.hstack([one, _lags(y, range(1, 8))])
    X_slm = np.hstack([one, _lags(y, SLM_LAGS), dummies, fourier])
    L = _lags(y, GBM_LAGS)
    r7 = L[:, :7].mean(axis=1, keepdims=True)
    r28 = np.column_stack([y[MAX_LAG - k : MAX_LAG - k + len(z)] for k in range(1, 29)]).mean(axis=1, keepdims=True)
    X_gbm = np.hstack([L, r7, r28, dow[:, None].astype(float), np.sin(w)[:, None], np.cos(w)[:, None]])
    return Design(z=z, dates=dates[MAX_LAG:], X_ar=X_ar, X_slm=X_slm, X_gbm=X_gbm)


# ------------------------------------------------------------------ scales


def gauss_scale(r: np.ndarray) -> float:
    return float(np.sqrt(np.mean(r**2))) or 1e-6


def t_scale(r: np.ndarray, df: float = T_DF, iters: int = 200, tol: float = 1e-10) -> float:
    """Maximum-likelihood scale of a Student-t with location 0 (EM iterations)."""
    r = np.asarray(r, dtype=float)
    s = float(np.median(np.abs(r)) / 0.7267) or 1e-6  # t5 quantile 0.75 = 0.7267
    for _ in range(iters):
        w = (df + 1.0) / (df + (r / s) ** 2)
        s_new = float(np.sqrt(np.sum(w * r**2) / len(r))) or 1e-6
        if abs(s_new - s) < tol * s:
            s = s_new
            break
        s = s_new
    return s


# ------------------------------------------------------------------ ETS(A,N,A)


@njit(cache=False)
def _hw_filter(y, a, g, m, start, stop):
    """One-step forecasts f[t] for t in [start + m, stop), states initialised on y[start:start+m]."""
    f = np.full(stop, np.nan)
    lvl = 0.0
    for k in range(m):
        lvl += y[start + k]
    lvl /= m
    s = np.empty(m)
    for k in range(m):
        s[k] = y[start + k] - lvl
    for t in range(start + m, stop):
        idx = (t - start) % m
        f[t] = lvl + s[idx]
        new = a * (y[t] - s[idx]) + (1.0 - a) * lvl
        s[idx] = g * (y[t] - new) + (1.0 - g) * s[idx]
        lvl = new
    return f


@njit(cache=False)
def _hw_sse(y, a, g, m, start, stop, burn):
    f = _hw_filter(y, a, g, m, start, stop)
    sse = 0.0
    for t in range(start + burn, stop):
        e = y[t] - f[t]
        sse += e * e
    return sse


def _logit_inv(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-v))


class ETSModel:
    m, burn = 7, MAX_LAG

    def fit(self, z: np.ndarray, a: int, b: int) -> "ETSModel":
        from scipy.optimize import minimize

        def obj(v):
            return _hw_sse(z, _logit_inv(v[0]), _logit_inv(v[1]), self.m, a, b, self.burn)

        best = None
        for x0 in ([0.0, -3.0], [-2.0, -4.0]):
            r = minimize(obj, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-8, maxiter=400))
            if best is None or r.fun < best.fun:
                best = r
        self.alpha, self.gamma = _logit_inv(best.x[0]), _logit_inv(best.x[1])
        self.start = a
        f = _hw_filter(z, self.alpha, self.gamma, self.m, a, b)
        self.resid = z[a + self.burn : b] - f[a + self.burn : b]
        return self

    def predict(self, z: np.ndarray, rows: np.ndarray) -> np.ndarray:
        f = _hw_filter(z, self.alpha, self.gamma, self.m, self.start, int(rows[-1]) + 1)
        return f[rows]


# ------------------------------------------------------------------ GBM


class GBMModel:
    folds = 3

    def _train(self, X, z):
        import lightgbm as lgb
        return lgb.train(GBM_PARAMS, lgb.Dataset(X, z, free_raw_data=True), num_boost_round=GBM_ROUNDS)

    def fit(self, X: np.ndarray, z: np.ndarray, a: int, b: int) -> "GBMModel":
        idx = np.arange(a, b)
        parts = np.array_split(idx, self.folds)
        res = []
        for k in range(self.folds):
            tr = np.concatenate([parts[i] for i in range(self.folds) if i != k])
            m = self._train(X[tr], z[tr])
            res.append(z[parts[k]] - m.predict(X[parts[k]]))
        self.resid = np.concatenate(res)
        self.model = self._train(X[a:b], z[a:b])
        return self

    def predict(self, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return self.model.predict(X[rows])


# ------------------------------------------------------------------ one fitted window, both distributions


@dataclass
class Fitted:
    mean: np.ndarray        # point forecasts for the requested rows (Gaussian / least squares)
    sd_g: float             # Gaussian scale
    mean_t: np.ndarray      # point forecasts under the Student-t fit (differs for AR and SLM only)
    sd_t: float             # Student-t scale


def fit_predict(family: str, D: Design, a: int, b: int, rows: np.ndarray) -> Fitted:
    """Fit ``family`` on aligned rows [a, b) and forecast ``rows`` one step ahead."""
    if family in ("AR", "SLM"):
        X = D.X_ar if family == "AR" else D.X_slm
        g = ARGaussian().fit(X[a:b], D.z[a:b])
        tm = ARStudentT(df=T_DF).fit(X[a:b], D.z[a:b])
        mg, sg = g.predict(X[rows])
        mt, st = tm.predict(X[rows])
        return Fitted(mg, float(sg[0]), mt, float(st[0]))
    if family == "ETS":
        m = ETSModel().fit(D.z, a, b)
        mu = m.predict(D.z, rows)
        return Fitted(mu, gauss_scale(m.resid), mu, t_scale(m.resid))
    if family == "GBM":
        m = GBMModel().fit(D.X_gbm, D.z, a, b)
        mu = m.predict(D.X_gbm, rows)
        return Fitted(mu, gauss_scale(m.resid), mu, t_scale(m.resid))
    raise ValueError(family)
