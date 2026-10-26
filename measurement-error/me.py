"""Measurement-error corrections for regression, written from scratch in NumPy.

Notation: y outcome, w (n,) or (n, k) error-prone measurement(s) of a scalar
covariate x, Z (n, q) covariates measured without error (no intercept column;
one is added). Every estimator returns the coefficient vector for
[intercept, x, Z...] so results line up.

* ols / logit                 naive fits (and the building blocks)
* moment_correct              linear regression with a known error variance
                              (Fuller 1987, sec. 2.2)
* rc_replicates               regression calibration with k replicates
                              (Carroll et al. 2006, ch. 4)
* rc_validation               regression calibration with an internal
                              validation subsample
* simex                       simulation-extrapolation (Cook & Stefanski 1994)
* tsls                        two-stage least squares, a replicate or second
                              indicator as the instrument
* mime                        multiple imputation of x from a validation
                              subsample, outcome included (Cole et al. 2006)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def design(x, Z):
    x = np.asarray(x, float).reshape(len(x), -1)
    Z = np.empty((len(x), 0)) if Z is None else np.asarray(Z, float).reshape(len(x), -1)
    return np.column_stack([np.ones(len(x)), x, Z])


@dataclass
class Fit:
    coef: np.ndarray
    se: np.ndarray = field(default=None)


# ----------------------------------------------------------------- naive fits
def ols(X, y, se=True):
    """OLS on a full design matrix; HC1 standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    if not se:
        return Fit(b)
    e = y - X @ b
    n, p = X.shape
    meat = (X * (e ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv * n / (n - p)
    return Fit(b, np.sqrt(np.diag(V)))


def logit(X, y, se=True, beta0=None, iters=50, tol=1e-10):
    """Logistic regression by Newton-Raphson; model-based standard errors."""
    b = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    if not se:
        return Fit(b)
    eta = np.clip(X @ b, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    H = (X * (p * (1 - p))[:, None]).T @ X
    return Fit(b, np.sqrt(np.diag(np.linalg.inv(H))))


def fit_model(X, y, family, se=True):
    return ols(X, y, se) if family == "gaussian" else logit(X, y, se)


# --------------------------------------------------- linear moment correction
def moment_correct(w, Z, y, sigma_u2):
    """Fuller's corrected least squares: (D'D/n - diag(0, S_uu, 0..))^-1 D'y/n.

    w may hold several error-prone columns (n, k) with uncorrelated errors of
    variances sigma_u2 (length k); they come first after the intercept."""
    D = design(w, Z)
    n = len(y)
    s = np.atleast_1d(np.asarray(sigma_u2, float))
    k = len(s)
    M = D.T @ D / n
    M[1:k + 1, 1:k + 1] -= np.diag(s)
    return Fit(np.linalg.solve(M, D.T @ y / n))


# ------------------------------------------------------ regression calibration
def replicate_error_variance(W):
    """Within-person variance of k replicates (n, k), pooled over persons."""
    W = np.asarray(W, float)
    k = W.shape[1]
    return np.mean(np.var(W, axis=1, ddof=1)), k


def calibrate_replicates(W, Z):
    """E[x | mean of replicates, Z] by the best linear predictor (Carroll et al. 2006, 4.4)."""
    W = np.asarray(W, float)
    s_u2, k = replicate_error_variance(W)
    wbar = W.mean(axis=1)
    Zc = np.empty((len(wbar), 0)) if Z is None else np.asarray(Z, float).reshape(len(wbar), -1)
    V = np.column_stack([wbar, Zc])
    mu = V.mean(axis=0)
    S = np.cov(V, rowvar=False).reshape(V.shape[1], V.shape[1])
    Sxv = S[0].copy()
    Sxv[0] -= s_u2 / k                # cov(x, wbar) = var(wbar) - s_u^2 / k
    xhat = mu[0] + (V - mu) @ np.linalg.solve(S, Sxv)
    return xhat, s_u2


def rc_replicates(W, Z, y, family="gaussian"):
    xhat, _ = calibrate_replicates(W, Z)
    return fit_model(design(xhat, Z), y, family, se=False)


def _distinct(W, y):
    """Drop replicate columns identical to an earlier one (Berkson-type readings)."""
    W = np.asarray(W, float).reshape(len(y), -1)
    keep = [0] + [j for j in range(1, W.shape[1])
                  if not any(np.allclose(W[:, j], W[:, i]) for i in range(j))]
    return W[:, keep]


def rc_validation(W, Z, y, x, val, family="gaussian"):
    """Calibrate x on (W, Z) in the validation rows; use x itself where observed."""
    W = _distinct(W, y)
    C = design(W, Z)
    g = np.linalg.lstsq(C[val], x[val], rcond=None)[0]
    xhat = np.where(val, x, C @ g)
    return fit_model(design(xhat, Z), y, family, se=False)


# -------------------------------------------------------------------- SIMEX
LAMBDAS = (0.5, 1.0, 1.5, 2.0)


def simex(w, Z, y, sigma_u2, family="gaussian", B=50, lambdas=LAMBDAS, rng=None,
          extrapolant="quadratic", return_path=False):
    """Simulation-extrapolation.

    extrapolant "quadratic" is the default of R's simex package; "rational"
    fits g(l) = a + b / (c + l), which is exact for linear regression with
    classical error (Cook & Stefanski 1994)."""
    rng = np.random.default_rng(rng)
    lam = np.r_[0.0, lambdas]
    D0 = design(w, Z)
    path = [fit_model(D0, y, family, se=False).coef]
    for l in lambdas:
        noise = np.sqrt(l * sigma_u2) * rng.standard_normal((B, len(w)))
        if family == "gaussian":                       # all B least-squares fits at once
            Dn = np.repeat(D0[None], B, axis=0)
            Dn[:, :, 1] += noise
            XtX = np.einsum("bnp,bnq->bpq", Dn, Dn)
            Xty = np.einsum("bnp,n->bp", Dn, y)
            path.append(np.linalg.solve(XtX, Xty[..., None])[..., 0].mean(axis=0))
        else:
            bs = []
            for b in range(B):
                Dn = D0.copy()
                Dn[:, 1] += noise[b]
                bs.append(fit_model(Dn, y, family, se=False).coef)
            path.append(np.mean(bs, axis=0))
    path = np.array(path)
    est = extrapolate(lam, path, extrapolant)
    return (Fit(est), lam, path) if return_path else Fit(est)


_C_GRID = 1.0 + np.geomspace(1e-3, 1e3, 400)


def extrapolate(lam, path, extrapolant="quadratic"):
    """Extrapolate each column of path(lambda) to lambda = -1.

    The rational form a + b / (c + l) is fitted by profiling: for each c on a
    fine grid (c > 1, so the extrapolant is finite at -1) a and b are linear
    least squares, and the c with the smallest residual sum of squares wins."""
    A = np.column_stack([np.ones_like(lam), lam, lam ** 2])
    quad = np.array([1.0, -1.0, 1.0]) @ np.linalg.lstsq(A, path, rcond=None)[0]
    if extrapolant == "quadratic":
        return quad
    T = 1 / (_C_GRID[:, None] + lam[None, :])                 # (grid, L)
    Tc = T - T.mean(axis=1, keepdims=True)
    Gc = path - path.mean(axis=0)                             # (L, p)
    b = (Tc @ Gc) / np.sum(Tc ** 2, axis=1)[:, None]          # (grid, p)
    a = path.mean(axis=0)[None, :] - b * T.mean(axis=1)[:, None]
    fitted = a[:, None, :] + b[:, None, :] * T[:, :, None]   # (grid, L, p)
    sse = np.sum((fitted - path[None]) ** 2, axis=1)          # (grid, p)
    k = np.argmin(sse, axis=0)
    cols = np.arange(path.shape[1])
    return a[k, cols] + b[k, cols] / (_C_GRID[k] - 1)


# ---------------------------------------------------------------------- 2SLS
def tsls(y, X, Zi):
    """2SLS with all exogenous columns of X included in the instrument matrix Zi."""
    Xh = Zi @ np.linalg.lstsq(Zi, X, rcond=None)[0]
    b = np.linalg.solve(Xh.T @ X, Xh.T @ y)
    e = y - X @ b
    n, p = X.shape
    A = np.linalg.inv(Xh.T @ Xh)
    meat = (Xh * (e ** 2)[:, None]).T @ Xh
    V = A @ meat @ A * n / (n - p)
    return Fit(b, np.sqrt(np.diag(V)))


def iv_replicate(W, Z, y):
    """x measured by W[:,0]; the second replicate W[:,1] is the instrument."""
    W = np.asarray(W, float)
    return tsls(y, design(W[:, 0], Z), design(W[:, 1], Z))


# ------------------------------------------------- multiple imputation of x
def mime(W, Z, y, x, val, family="gaussian", m=20, rng=None):
    """Impute x where unobserved from a Bayesian linear regression of x on
    (W, Z, y) fitted in the validation rows; analyse each completed data set;
    combine with Rubin's rules."""
    rng = np.random.default_rng(rng)
    W = _distinct(W, y)
    C = np.column_stack([design(W, Z), y])
    Cv, xv = C[val], x[val]
    nv, q = Cv.shape
    V = np.linalg.inv(Cv.T @ Cv)
    g = V @ Cv.T @ xv
    rss = np.sum((xv - Cv @ g) ** 2)
    L = np.linalg.cholesky(V)
    ests, vars_ = [], []
    for _ in range(m):
        s2 = rss / rng.chisquare(nv - q)
        gd = g + np.sqrt(s2) * L @ rng.standard_normal(q)
        xi = np.where(val, x, C @ gd + np.sqrt(s2) * rng.standard_normal(len(y)))
        f = fit_model(design(xi, Z), y, family)
        ests.append(f.coef)
        vars_.append(f.se ** 2)
    ests, vars_ = np.array(ests), np.array(vars_)
    qbar = ests.mean(axis=0)
    T = vars_.mean(axis=0) + (1 + 1 / m) * ests.var(axis=0, ddof=1)
    return Fit(qbar, np.sqrt(T))


# ------------------------------------------------------------- bootstrap SE
def bootstrap_se(fn, n, B=100, rng=None):
    """Nonparametric bootstrap of fn(idx) -> coef vector, resampling rows."""
    rng = np.random.default_rng(rng)
    reps = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return np.std(reps, axis=0, ddof=1)


def ci_covers(est, se, truth, level=0.95):
    q = stats.norm.ppf(0.5 + level / 2)
    return abs(est - truth) <= q * se
