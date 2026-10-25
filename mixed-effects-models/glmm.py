"""Random-intercept logistic regression by adaptive Gauss-Hermite quadrature.

    logit P(y_ij = 1 | b_j) = x_ij' beta + b_j,     b_j ~ N(0, s^2)

For each group the integrand is centred at its mode and scaled by its
curvature (Pinheiro & Bates 1995, "Approximations to the log-likelihood
function in the nonlinear mixed-effects model", JCGS 4(1)). With n_agq = 1
this is the Laplace approximation (lme4's default); more nodes converge to
the exact marginal likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.special import expit, log_expit


@dataclass
class GLMMResult:
    beta: np.ndarray
    se: np.ndarray
    sigma: float
    modes: np.ndarray        # conditional modes of b_j
    labels: np.ndarray
    loglik: float


def _group_modes(eta, y, g, J, s2, iters=30):
    b = np.zeros(J)
    for _ in range(iters):
        p = expit(eta + b[g])
        grad = np.bincount(g, y - p, J) - b / s2
        hess = -np.bincount(g, p * (1 - p), J) - 1 / s2
        step = grad / hess
        b = b - step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = expit(eta + b[g])
    curv = np.bincount(g, p * (1 - p), J) + 1 / s2
    return b, curv


def loglik(params, X, y, g, J, n_agq):
    beta, log_s = params[:-1], params[-1]
    s2 = np.exp(2 * log_s)
    eta = X @ beta
    b_hat, curv = _group_modes(eta, y, g, J, s2)
    scale = 1 / np.sqrt(curv)
    z, w = np.polynomial.hermite.hermgauss(n_agq)
    total = np.zeros(J)
    terms = []
    for zk, wk in zip(z, w):
        b = b_hat + np.sqrt(2) * scale * zk
        ll_y = np.bincount(g, y * log_expit(eta + b[g]) +
                           (1 - y) * log_expit(-(eta + b[g])), J)
        log_prior = -0.5 * np.log(2 * np.pi * s2) - b ** 2 / (2 * s2)
        terms.append(np.log(wk) + zk ** 2 + ll_y + log_prior)
    terms = np.array(terms)
    mx = terms.max(0)
    total = mx + np.log(np.exp(terms - mx).sum(0)) + np.log(np.sqrt(2) * scale)
    return total.sum()


def fit(X, y, groups, n_agq=25):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    labels, g = np.unique(np.asarray(groups), return_inverse=True)
    J = len(labels)
    b0 = np.linalg.lstsq(X, np.log((y + .5) / (1.5 - y)), rcond=None)[0] * 0.5
    x0 = np.r_[b0, np.log(0.5)]
    f = lambda p: -loglik(p, X, y, g, J, n_agq)
    res = optimize.minimize(f, x0, method="BFGS", options={"gtol": 1e-6})
    # observed-information SEs by finite differences
    k = len(res.x)
    H = np.zeros((k, k))
    eps = 1e-4
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * eps, np.eye(k)[j] * eps
            H[i, j] = H[j, i] = (f(res.x + ei + ej) - f(res.x + ei - ej)
                                 - f(res.x - ei + ej) + f(res.x - ei - ej)) / (4 * eps ** 2)
    cov = np.linalg.inv(H)
    beta = res.x[:-1]
    s = float(np.exp(res.x[-1]))
    modes, _ = _group_modes(X @ beta, y, g, J, s ** 2)
    return GLMMResult(beta=beta, se=np.sqrt(np.diag(cov))[:-1], sigma=s,
                      modes=modes, labels=labels, loglik=-res.fun)
