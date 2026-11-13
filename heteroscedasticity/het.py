"""Heteroscedasticity: inference, weighting and prediction intervals, from scratch in NumPy.

Inference for OLS coefficients
* hc_cov            classical and HC0, HC1, HC2, HC3, HC4, HC4m, HC5 covariance
                    matrices, with the weights of R's sandwich::vcovHC
                    (MacKinnon & White 1985; Cribari-Neto 2004; Cribari-Neto &
                    da Silva 2011; Cribari-Neto, Souza & Vasconcellos 2007)
* wild_test         wild-bootstrap test of one coefficient: restricted
                    (null imposed) or unrestricted residuals; Rademacher,
                    Mammen or Webb six-point weights (Davidson & Flachaire
                    2008; Webb 2023)
* bp_test, white_test   Koenker's studentized Breusch-Pagan test and White's test

Weighting
* wls               weighted least squares with any of the covariance types
* fgls              feasible GLS with an exponential (Harvey 1976) or power
                    (Park 1966) variance function fitted to log squared residuals

Prediction intervals
* quantreg          linear quantile regression by linear programming (HiGHS)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

HC_TYPES = ("const", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5")


@dataclass
class Fit:
    coef: np.ndarray
    resid: np.ndarray
    hat: np.ndarray
    XtX_inv: np.ndarray
    X: np.ndarray
    df: int

    def cov(self, type="HC3"):
        return hc_cov(self.X, self.resid, self.hat, self.XtX_inv, type)

    def se(self, type="HC3"):
        return np.sqrt(np.diag(self.cov(type)))


def ols(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    return Fit(b, e, h, XtX_inv, X, X.shape[0] - X.shape[1])


def omega(e, h, type, n_minus_p=None):
    """Diagonal 'meat' weights of each covariance type, as in sandwich::vcovHC."""
    n = len(e)
    p = int(round(h.sum()))
    df = n - p if n_minus_p is None else n_minus_p
    if type == "const":
        return np.full(n, np.sum(e ** 2) / df)
    if type == "HC0":
        return e ** 2
    if type == "HC1":
        return e ** 2 * n / df
    if type == "HC2":
        return e ** 2 / (1 - h)
    if type == "HC3":
        return e ** 2 / (1 - h) ** 2
    if type == "HC4":
        delta = np.minimum(4, n * h / p)
        return e ** 2 / (1 - h) ** delta
    if type == "HC4m":
        delta = np.minimum(1.0, n * h / p) + np.minimum(1.5, n * h / p)
        return e ** 2 / (1 - h) ** delta
    if type == "HC5":
        delta = np.minimum(n * h / p, np.maximum(4, n * 0.7 * h.max() / p))
        return e ** 2 / np.sqrt((1 - h) ** delta)
    raise ValueError(type)


def hc_cov(X, e, h, XtX_inv, type="HC3"):
    w = omega(e, h, type)
    return XtX_inv @ (X.T * w) @ X @ XtX_inv


def t_pvalue(est, se, b0=0.0, df=None):
    t = (est - b0) / se
    return 2 * (stats.norm.sf(abs(t)) if df is None else stats.t.sf(abs(t), df))


# ------------------------------------------------------------ wild bootstrap
def _weights(kind, shape, rng):
    if kind == "rademacher":
        return rng.choice([-1.0, 1.0], size=shape)
    if kind == "mammen":
        s5 = np.sqrt(5)
        p = (s5 + 1) / (2 * s5)
        return np.where(rng.random(shape) < p, -(s5 - 1) / 2, (s5 + 1) / 2)
    if kind == "webb":
        vals = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
        return vals[rng.integers(0, 6, size=shape)]
    raise ValueError(kind)


def wild_test(X, y, j, b0=0.0, B=999, kind="rademacher", restricted=True, hc="HC1", rng=None, V=None,
              return_draws=False):
    """Wild-bootstrap p-value for H0: beta_j = b0 against a two-sided alternative.

    The statistic is the t ratio with the `hc` covariance. With restricted=True
    the bootstrap data are generated from the fit with beta_j = b0 imposed
    (WR in Davidson & Flachaire 2008); otherwise from the unrestricted fit,
    centred at the estimate (WU). V, an (n, B) array of weights, can be given
    to reproduce a draw exactly."""
    rng = np.random.default_rng(rng)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    f = ols(X, y)
    t_obs = (f.coef[j] - b0) / np.sqrt(np.diag(f.cov(hc)))[j]
    if restricted:
        keep = [i for i in range(k) if i != j]
        yr = y - b0 * X[:, j]
        Xr = X[:, keep]
        br = np.linalg.lstsq(Xr, yr, rcond=None)[0]
        base_resid = yr - Xr @ br
        centre = b0
    else:
        base_resid = f.resid
        centre = f.coef[j]
    if V is None:
        V = _weights(kind, (n, B), rng)
    U = base_resid[:, None] * V                      # bootstrap errors
    A = f.XtX_inv @ X.T                              # (k, n)
    # beta* - (fit the bootstrap data are built around) = A U; for the restricted
    # design that fit has beta_j = b0 exactly, for the unrestricted it is the OLS fit
    dbeta_j = A[j] @ U
    Estar = U - X @ (A @ U)                          # residuals of each bootstrap fit
    W = omega_matrix(Estar, f.hat, hc)
    se_star = np.sqrt((A[j] ** 2) @ W)
    t_star = dbeta_j / se_star
    p = np.mean(np.abs(t_star) >= abs(t_obs))
    return (p, t_obs, t_star) if return_draws else p


def omega_matrix(E, h, type):
    """omega() applied column by column to an (n, B) array of residual vectors."""
    n = E.shape[0]
    p = int(round(h.sum()))
    h = h[:, None]
    if type == "HC0":
        return E ** 2
    if type == "HC1":
        return E ** 2 * n / (n - p)
    if type == "HC2":
        return E ** 2 / (1 - h)
    if type == "HC3":
        return E ** 2 / (1 - h) ** 2
    return np.column_stack([omega(E[:, b], h[:, 0], type) for b in range(E.shape[1])])


# ----------------------------------------------------------- tests of variance
def bp_test(resid, Z):
    """Koenker's (1981) studentized Breusch-Pagan LM test: n R^2 of e^2 on Z (with constant)."""
    e2 = np.asarray(resid) ** 2
    Z = np.column_stack([np.ones(len(e2)), np.asarray(Z, float).reshape(len(e2), -1)])
    g = np.linalg.lstsq(Z, e2, rcond=None)[0]
    r2 = 1 - np.sum((e2 - Z @ g) ** 2) / np.sum((e2 - e2.mean()) ** 2)
    lm = len(e2) * r2
    df = Z.shape[1] - 1
    return lm, stats.chi2.sf(lm, df)


def white_test(resid, X):
    """White (1980): n R^2 of e^2 on the regressors, their squares and cross products.

    X excludes the constant."""
    X = np.asarray(X, float).reshape(len(resid), -1)
    cols = [X[:, i] for i in range(X.shape[1])]
    cols += [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
    Z = np.column_stack(cols)
    # drop exact duplicates (e.g. the square of a dummy)
    keep = []
    for i in range(Z.shape[1]):
        if not any(np.allclose(Z[:, i], Z[:, j]) for j in keep):
            keep.append(i)
    return bp_test(resid, Z[:, keep])


# --------------------------------------------------------------- weighting
def wls(X, y, w):
    """Weighted least squares; the returned Fit is on the transformed scale, so
    fit.cov(type) gives classical or HC covariances of the WLS coefficients."""
    s = np.sqrt(np.asarray(w, float))
    return ols(np.asarray(X, float) * s[:, None], np.asarray(y, float) * s)


def fgls(X, y, Z, form="exponential", floor=1e-12):
    """Feasible GLS: fit log e^2 on Z (exponential) or on log Z (power) from the
    OLS residuals, weight by the inverse of the fitted variance."""
    f = ols(X, y)
    e2 = np.maximum(f.resid ** 2, floor)
    Z = np.asarray(Z, float).reshape(len(y), -1)
    if form == "power":
        Z = np.log(Z)
    D = np.column_stack([np.ones(len(y)), Z])
    g = np.linalg.lstsq(D, np.log(e2), rcond=None)[0]
    var = np.exp(D @ g)
    return wls(X, y, 1 / var), var, g


# ------------------------------------------------------------ quantile regression
def quantreg(X, y, tau):
    """Linear quantile regression: minimise the check loss by linear programming."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    # variables: beta+ (k), beta- (k), u+ (n), u- (n); X(b+ - b-) + u+ - u- = y
    from scipy import sparse
    c = np.r_[np.zeros(2 * k), tau * np.ones(n), (1 - tau) * np.ones(n)]
    I = sparse.identity(n, format="csr")
    A = sparse.hstack([sparse.csr_matrix(X), sparse.csr_matrix(-X), I, -I], format="csr")
    res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    return res.x[:k] - res.x[k:2 * k]
