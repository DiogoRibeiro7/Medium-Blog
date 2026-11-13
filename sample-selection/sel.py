"""Sample-selection estimators written from scratch (NumPy/SciPy).

Model (Heckman's type-2 Tobit): a selection equation d* = z'gamma + v, d = 1[d* > 0],
and an outcome equation y = x'beta + u observed only when d = 1, with (u, v)
bivariate normal, var(v) = 1, var(u) = sigma^2, corr(u, v) = rho.

  probit           maximum likelihood probit by Newton-Raphson
  heckit           Heckman's two-step estimator with the corrected covariance
                   matrix (Heckman 1979; Greene 2003, section 22.4), as in the R
                   package sampleSelection (heckit / selection(method = "2step"))
  heckman_ml       full-information maximum likelihood with analytic gradient,
                   sigma and rho parameterised as log(sigma) and atanh(rho)
  series_cf        a control-function estimator that replaces the inverse Mills
                   ratio by a polynomial in it (in the spirit of Newey 2009); it
                   identifies the slopes, not the intercept, and needs an
                   exclusion restriction
  imr_diagnostics  how close the inverse Mills ratio is to a linear function of
                   the outcome regressors among the selected
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import log_ndtr

LOG2PI = np.log(2 * np.pi)


def mills(t):
    """phi(t) / Phi(t), computed in logs so that it is stable for very negative t."""
    return np.exp(-0.5 * t * t - 0.5 * LOG2PI - log_ndtr(t))


# ------------------------------------------------------------------ probit
@dataclass
class Probit:
    coef: np.ndarray
    cov: np.ndarray
    loglik: float
    index: np.ndarray


def probit(Z, d, tol=1e-10, maxit=100):
    Z = np.asarray(Z, float)
    d = np.asarray(d, float)
    q = 2 * d - 1
    g = np.linalg.lstsq(Z, (d - d.mean()) / max(d.std(), 1e-8) * 0.5, rcond=None)[0]
    for _ in range(maxit):
        t = q * (Z @ g)
        lam = mills(t)
        grad = Z.T @ (q * lam)
        w = lam * (lam + t)
        H = (Z.T * w) @ Z
        step = np.linalg.solve(H, grad)
        g = g + step
        if np.max(np.abs(step)) < tol:
            break
    t = q * (Z @ g)
    lam = mills(t)
    H = (Z.T * (lam * (lam + t))) @ Z
    return Probit(g, np.linalg.inv(H), float(np.sum(log_ndtr(t))), Z @ g)


# ------------------------------------------------------------------ two-step
@dataclass
class Heckit:
    beta: np.ndarray          # outcome coefficients
    beta_lambda: float        # coefficient of the inverse Mills ratio
    sigma: float
    rho: float
    cov: np.ndarray           # covariance of (beta, beta_lambda), corrected
    probit: Probit
    lam: np.ndarray           # inverse Mills ratio among the selected
    resid: np.ndarray

    def se(self):
        return np.sqrt(np.diag(self.cov))


def heckit(Z, d, X, y):
    """Two-step estimator. Z: selection regressors (all n rows); d: 0/1;
    X, y: outcome regressors and outcome for all rows (only rows with d = 1 used)."""
    Z, X = np.asarray(Z, float), np.asarray(X, float)
    d = np.asarray(d).astype(bool)
    pr = probit(Z, d)
    zg = pr.index[d]
    lam = mills(zg)
    delta = lam * (lam + zg)
    W = np.column_stack([X[d], lam])
    ys = np.asarray(y, float)[d]
    WtW_inv = np.linalg.inv(W.T @ W)
    b = WtW_inv @ W.T @ ys
    e = ys - W @ b
    n1 = d.sum()
    bl = b[-1]
    sigma2 = e @ e / n1 + bl ** 2 * delta.mean()
    sigma = np.sqrt(sigma2)
    rho = bl / sigma
    Zs = Z[d]
    A = W.T @ ((1 - rho ** 2 * delta)[:, None] * W)
    F = (W.T * delta) @ Zs
    Q = rho ** 2 * F @ pr.cov @ F.T
    cov = sigma2 * WtW_inv @ (A + Q) @ WtW_inv
    return Heckit(b[:-1], bl, sigma, rho, cov, pr, lam, e)


# ------------------------------------------------------------------ maximum likelihood
@dataclass
class HeckmanML:
    gamma: np.ndarray
    beta: np.ndarray
    sigma: float
    rho: float
    loglik: float
    cov: np.ndarray           # covariance of (gamma, beta, sigma, rho)
    converged: bool
    theta: np.ndarray

    def se(self):
        return np.sqrt(np.diag(self.cov))


def _negll(theta, Z, d, X, y, kz, kx, rho_fixed=None):
    g = theta[:kz]
    b = theta[kz:kz + kx]
    ls = theta[kz + kx]
    if rho_fixed is None:
        tau = theta[kz + kx + 1]
        rho = np.tanh(tau)
    else:
        rho = rho_fixed
    s = np.exp(ls)
    zg = Z @ g
    r = np.sqrt(1 - rho ** 2)
    n0 = ~d
    ll0 = log_ndtr(-zg[n0])
    e = (y[d] - X[d] @ b) / s
    a = (zg[d] + rho * e) / r
    ll1 = log_ndtr(a) - 0.5 * e * e - 0.5 * LOG2PI - ls
    nll = -(ll0.sum() + ll1.sum())
    # gradient
    m0 = mills(-zg[n0])
    m1 = mills(a)
    gg = -(Z[n0].T @ m0) + Z[d].T @ (m1 / r)
    gb = X[d].T @ (-m1 * rho / (s * r) + e / s)
    gls = np.sum(-m1 * rho * e / r + e * e - 1)
    grad = [gg, gb, [gls]]
    if rho_fixed is None:
        gtau = np.sum(m1 * (e + rho * zg[d]) / r)
        grad.append([gtau])
    return nll, -np.concatenate(grad)


def heckman_ml_multistart(Z, d, X, y, rho_starts=(-0.8, 0.0, 0.8)):
    """Maximum likelihood from the two-step start and from starts with rho set to
    each value in rho_starts (other parameters from the two-step estimates);
    returns the fit with the highest log-likelihood and the number of distinct
    maxima found (log-likelihoods more than 1e-3 apart)."""
    Z, X = np.asarray(Z, float), np.asarray(X, float)
    d = np.asarray(d).astype(bool)
    fits = [heckman_ml(Z, d, X, y)]
    base = fits[0].theta.copy()
    for r0 in rho_starts:
        st = base.copy()
        st[-1] = np.arctanh(r0)
        fits.append(heckman_ml(Z, d, X, y, start=st))
    ok = [f for f in fits if np.isfinite(f.loglik)]
    best = max(ok, key=lambda f: f.loglik)
    lls = sorted({round(f.loglik, 3) for f in ok})
    distinct = 1 + sum(1 for a, b in zip(lls, lls[1:]) if b - a > 1e-3)
    return best, distinct


def heckman_ml(Z, d, X, y, start=None, rho_fixed=None):
    """Maximum likelihood. Starts from the two-step estimates unless `start` is
    given. With rho_fixed, rho is held at that value (a profile in rho)."""
    Z, X = np.asarray(Z, float), np.asarray(X, float)
    d = np.asarray(d).astype(bool)
    y = np.where(d, np.asarray(y, float), 0.0)
    kz, kx = Z.shape[1], X.shape[1]
    if start is None:
        h = heckit(Z, d, X, y)
        r0 = float(np.clip(h.rho, -0.9, 0.9))
        start = np.concatenate([h.probit.coef, h.beta, [np.log(h.sigma)], [np.arctanh(r0)]])
    if rho_fixed is not None:
        start = start[:kz + kx + 1]
    args = (Z, d, X, y, kz, kx, rho_fixed)
    res = optimize.minimize(_negll, start, args=args, jac=True, method="BFGS", options={"gtol": 1e-7, "maxiter": 2000})
    theta = res.x
    # Newton polish with a Hessian from finite differences of the analytic gradient
    decrement = np.inf
    for _ in range(50):
        f0, g0 = _negll(theta, *args)
        H = _hessian(theta, args)
        try:
            step = np.linalg.solve(H, g0)
        except np.linalg.LinAlgError:
            break
        decrement = float(g0 @ step)
        if not np.isfinite(decrement) or abs(decrement) < 1e-12:
            break
        t = 1.0
        while t > 1e-6:
            f1 = _negll(theta - t * step, *args)[0]
            if np.isfinite(f1) and f1 <= f0 + 1e-12:
                break
            t /= 2
        if t <= 1e-6:
            break
        theta = theta - t * step
    f_final, g_final = _negll(theta, *args)
    H = _hessian(theta, args)
    k = len(theta)
    try:
        C = np.linalg.inv(H)
        converged = bool(abs(g_final @ C @ g_final) < 1e-8 and np.all(np.linalg.eigvalsh(H) > 0))
    except np.linalg.LinAlgError:
        C = np.full((k, k), np.nan)
        converged = False
    sigma = np.exp(theta[kz + kx])
    rho = np.tanh(theta[kz + kx + 1]) if rho_fixed is None else rho_fixed
    # delta method from (log sigma, atanh rho) to (sigma, rho)
    J = np.eye(k)
    J[kz + kx, kz + kx] = sigma
    if rho_fixed is None:
        J[kz + kx + 1, kz + kx + 1] = 1 - rho ** 2
    cov = J @ C @ J.T
    return HeckmanML(theta[:kz], theta[kz:kz + kx], float(sigma), float(rho), float(-f_final), cov, converged, theta)


def _hessian(theta, args):
    k = len(theta)
    H = np.zeros((k, k))
    for j in range(k):
        h = 1e-5 * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        H[:, j] = (_negll(tp, *args)[1] - _negll(tm, *args)[1]) / (2 * h)
    return 0.5 * (H + H.T)


# ------------------------------------------------------------------ control function with a series
def series_cf(Z, d, X, y, degree=3):
    """OLS of y on X and a polynomial in the estimated inverse Mills ratio
    (degree 1 is the two-step estimator). Returns the outcome slopes; the
    intercept is not identified without further assumptions."""
    Z, X = np.asarray(Z, float), np.asarray(X, float)
    d = np.asarray(d).astype(bool)
    pr = probit(Z, d)
    lam = mills(pr.index[d])
    P = np.column_stack([lam ** p for p in range(1, degree + 1)])
    W = np.column_stack([X[d], P])
    b = np.linalg.lstsq(W, np.asarray(y, float)[d], rcond=None)[0]
    return b[:X.shape[1]]


# ------------------------------------------------------------------ diagnostics
def imr_diagnostics(Z, d, X):
    """R^2 of the inverse Mills ratio on the outcome regressors among the selected,
    and the condition number of the standardised outcome design with the ratio
    added (the collinearity that decides how well the correction is identified)."""
    Z, X = np.asarray(Z, float), np.asarray(X, float)
    d = np.asarray(d).astype(bool)
    pr = probit(Z, d)
    lam = mills(pr.index[d])
    Xs = X[d]
    fit = Xs @ np.linalg.lstsq(Xs, lam, rcond=None)[0]
    r2 = 1 - np.sum((lam - fit) ** 2) / np.sum((lam - lam.mean()) ** 2)
    W = np.column_stack([Xs, lam])
    Wn = W / np.sqrt(np.sum(W ** 2, axis=0))
    sv = np.linalg.svd(Wn, compute_uv=False)
    return float(r2), float(sv[0] / sv[-1]), pr


def manski_quantile_bounds(y_obs, p_obs, q=0.5):
    """Worst-case bounds on the q-quantile of the outcome distribution of the
    whole population when only a share p_obs is observed (Manski 1994):
    lower = quantile (q - (1 - p)) / p of the observed outcomes (or -inf),
    upper = quantile q / p of the observed outcomes (or +inf)."""
    y_obs = np.sort(np.asarray(y_obs, float))
    lo_p = (q - (1 - p_obs)) / p_obs
    hi_p = q / p_obs
    lo = np.quantile(y_obs, lo_p) if lo_p > 0 else -np.inf
    hi = np.quantile(y_obs, hi_p) if hi_p < 1 else np.inf
    return lo, hi, lo_p, hi_p
