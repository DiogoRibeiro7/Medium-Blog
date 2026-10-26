"""Robust regression estimators written from scratch.

* M-estimation by iteratively reweighted least squares (Huber, bisquare)
* LTS by FAST-LTS (Rousseeuw & Van Driessen 2006, Data Min. Knowl. Disc. 12)
* S-estimation by FAST-S (Salibian-Barrera & Yohai 2006, JCGS 15)
* MM-estimation (Yohai 1987, Ann. Stat. 15): S start, then a bisquare
  M-step with the S-scale held fixed

Tuning follows R's robustbase::lmrob defaults: bisquare rho with c = 1.54764
and b = 0.5 for the S-estimator (50% breakdown), c = 4.685061 for the MM step
(95% efficiency at the normal), 500 random subsamples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

C_S, B_S = 1.54764, 0.5          # S: 50% breakdown point
C_MM = 4.685061                  # MM: 95% efficiency
C_HUBER = 1.345                  # Huber: 95% efficiency


# --------------------------------------------------------------- functions
def rho_bisq(u, c):
    """Bisquare rho, normalised so that rho(inf) = 1."""
    z = np.clip(np.abs(u) / c, 0, 1)
    return 1 - (1 - z ** 2) ** 3


def psi_bisq(u, c):
    z = u / c
    return np.where(np.abs(z) < 1, u * (1 - z ** 2) ** 2, 0.0)


def dpsi_bisq(u, c):
    z2 = (u / c) ** 2
    return np.where(z2 < 1, (1 - z2) * (1 - 5 * z2), 0.0)


def w_bisq(u, c):
    z2 = (u / c) ** 2
    return np.where(z2 < 1, (1 - z2) ** 2, 0.0)


def psi_huber(u, c):
    return np.clip(u, -c, c)


def dpsi_huber(u, c):
    return (np.abs(u) <= c).astype(float)


def w_huber(u, c):
    a = np.abs(u)
    return np.where(a <= c, 1.0, c / np.maximum(a, 1e-300))


def mad(r):
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def mad0(r):
    """MAD about zero, the usual scale start for regression residuals."""
    return 1.4826 * np.median(np.abs(r))


# ------------------------------------------------------------------ helpers
def wls(X, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def s_scale(r, s0=None, iters=200, tol=1e-10, p=0):
    """M-scale: solve sum(rho(r/s)) / (n - p) = b by fixed-point iteration
    (the n - p denominator is robustbase's small-sample convention)."""
    n = len(r)
    b = B_S * (n - p) / n
    s = mad0(r) if s0 is None else s0
    if s <= 0:
        s = np.mean(np.abs(r)) + 1e-12
    for _ in range(iters):
        s_new = s * np.sqrt(np.mean(rho_bisq(r / s, C_S)) / b)
        if abs(s_new / s - 1) < tol:
            return s_new
        s = s_new
    return s


def _s_isteps(X, y, beta, s, k):
    """k IRWLS steps for the S objective (weights from the bisquare with c = C_S)."""
    p = X.shape[1]
    for _ in range(k):
        r = y - X @ beta
        s = s_scale(r, s, iters=1, p=p)
        w = w_bisq(r / s, C_S)
        if w.sum() <= 0:
            break
        beta = wls(X, y, w)
    r = y - X @ beta
    return beta, s_scale(r, s, p=p)


@dataclass
class Fit:
    coef: np.ndarray
    scale: float
    resid: np.ndarray
    weights: np.ndarray | None = None
    cov: np.ndarray | None = None
    init: "Fit | None" = None

    @property
    def se(self):
        return np.sqrt(np.diag(self.cov))


# --------------------------------------------------------------- estimators
def ols(X, y, robust_se=True):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    n, p = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    if robust_se:
        cov = n / (n - p) * XtXi @ ((X * r[:, None] ** 2).T @ X) @ XtXi
    else:
        cov = r @ r / (n - p) * XtXi
    return Fit(b, np.sqrt(r @ r / (n - p)), r, np.ones(n), cov)


def m_estimate(X, y, kind="huber", c=None, beta0=None, scale=None, iters=200, tol=1e-9,
               cov="H1"):
    """M-estimator by IRWLS. Scale: fixed if given, otherwise re-estimated by
    the MAD of the residuals about zero at every step, as statsmodels RLM does)."""
    c = c or (C_HUBER if kind == "huber" else C_MM)
    w_fun = w_huber if kind == "huber" else w_bisq
    beta = np.linalg.lstsq(X, y, rcond=None)[0] if beta0 is None else beta0.copy()
    fixed = scale is not None
    s = scale
    for _ in range(iters):
        r = y - X @ beta
        if not fixed:
            s = mad0(r)
        w = w_fun(r / s, c)
        beta_new = wls(X, y, w)
        if np.max(np.abs(beta_new - beta)) < tol * (1 + np.max(np.abs(beta))):
            beta = beta_new
            break
        beta = beta_new
    r = y - X @ beta
    if not fixed:
        s = mad0(r)
    V = m_cov(X, r, s, kind, c) if cov == "H1" else m_sandwich(X, r, s, kind, c)
    return Fit(beta, s, r, w_fun(r / s, c), V)


def m_cov(X, r, s, kind, c):
    """Huber (1981) H1 sandwich, as in statsmodels RLM."""
    n, p = X.shape
    u = r / s
    if kind == "huber":
        ps, dps = psi_huber(u, c), dpsi_huber(u, c)
    else:
        ps, dps = psi_bisq(u, c), dpsi_bisq(u, c)
    m = dps.mean()
    K = 1 + p / n * dps.var() / m ** 2
    return K ** 2 * (np.sum(ps ** 2) / (n - p)) / m ** 2 * s ** 2 * np.linalg.inv(X.T @ X)


def m_sandwich(X, r, s, kind, c):
    """Sandwich s^2 A^-1 B A^-1 with A = mean(psi' x x'), B = mean(psi^2 x x'), times n/(n-p).

    Points with zero weight drop out of both A and B, so bad leverage points do
    not shrink the variance as they do in the H1 formula."""
    n, p = X.shape
    u = r / s
    if kind == "huber":
        ps, dps = psi_huber(u, c), dpsi_huber(u, c)
    else:
        ps, dps = psi_bisq(u, c), dpsi_bisq(u, c)
    A = (X * dps[:, None]).T @ X / n
    B = (X * (ps ** 2)[:, None]).T @ X / n
    Ai = np.linalg.inv(A)
    return s ** 2 * Ai @ B @ Ai / (n - p)


def _subsample_fits(X, y, n_sub, rng):
    n, p = X.shape
    for _ in range(n_sub):
        for _try in range(20):
            idx = rng.choice(n, p, replace=False)
            A = X[idx]
            if np.linalg.matrix_rank(A) == p:
                yield np.linalg.solve(A, y[idx])
                break


def lts(X, y, h=None, n_sub=500, n_best=10, seed=0):
    """FAST-LTS: least trimmed squares with h = floor((n + p + 1) / 2)."""
    n, p = X.shape
    h = h or (n + p + 1) // 2
    rng = np.random.default_rng(seed)

    def csteps(beta, k):
        for _ in range(k):
            r2 = (y - X @ beta) ** 2
            keep = np.argpartition(r2, h - 1)[:h]
            new = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            if np.allclose(new, beta):
                beta = new
                break
            beta = new
        r2 = np.sort((y - X @ beta) ** 2)[:h]
        return beta, r2.sum()

    cands = [csteps(b, 2) for b in _subsample_fits(X, y, n_sub, rng)]
    cands.sort(key=lambda t: t[1])
    best = min((csteps(b, 100) for b, _ in cands[:n_best]), key=lambda t: t[1])
    beta = best[0]
    r = y - X @ beta
    # consistency factor for the trimmed scale at the normal
    alpha = h / n
    q = stats.norm.ppf((1 + alpha) / 2)
    cons = 1 / np.sqrt(1 - 2 * q * stats.norm.pdf(q) / alpha) if h < n else 1.0
    s = cons * np.sqrt(best[1] / h)
    return Fit(beta, s, r)


def s_estimate(X, y, n_sub=500, k=2, n_best=5, seed=0):
    """FAST-S with the bisquare rho (c = 1.54764, b = 0.5)."""
    rng = np.random.default_rng(seed)
    cands = []
    for b in _subsample_fits(X, y, n_sub, rng):
        b, s = _s_isteps(X, y, b, mad0(y - X @ b), k)
        cands.append((s, b))
    cands.sort(key=lambda t: t[0])
    best = None
    for s, b in cands[:n_best]:
        for _ in range(500):
            b_new, s_new = _s_isteps(X, y, b, s, 1)
            if abs(s_new - s) < 1e-10 * s:
                b, s = b_new, s_new
                break
            b, s = b_new, s_new
        if best is None or s < best[0]:
            best = (s, b)
    s, b = best
    r = y - X @ b
    return Fit(b, s, r, w_bisq(r / s, C_S))


def mm_estimate(X, y, n_sub=500, seed=0, c=C_MM):
    """MM: S-estimate start, then bisquare M-step with the S-scale fixed."""
    S = s_estimate(X, y, n_sub=n_sub, seed=seed)
    fit = m_estimate(X, y, kind="bisquare", c=c, beta0=S.coef, scale=S.scale, cov="sandwich")
    fit.init = S
    return fit


def flag_outliers(fit, cutoff=2.5):
    """Observations whose standardised robust residual exceeds the cutoff."""
    return np.abs(fit.resid / fit.scale) > cutoff
