"""Multiple imputation by chained equations, and Rubin's rules.

Imputation methods follow the R package mice (van Buuren & Groothuis-Oudshoorn
2011, J. Stat. Softw. 45(3)):

* "norm"   Bayesian linear regression: draw sigma^2 and beta from their
           posterior given the observed rows, then add residual noise.
* "pmm"    predictive mean matching, type-1 matching as in mice: predictions
           for observed rows use beta-hat, predictions for missing rows use the
           posterior draw beta*; each missing value is replaced by the observed
           value of one of the `donors` nearest observed rows.
* "logreg" logistic regression for 0/1 variables: draw beta from the normal
           approximation to its posterior, then draw Bernoulli values.
* "rf"     random-forest imputation in the spirit of mice's rf method (Doove,
           van Buuren & Dusseldorp 2014): fit a small forest on the observed
           rows and, for each missing row, draw one observed value from the
           pooled donors that share its leaf in any tree.

Each of the m imputations runs its own chain for `maxit` iterations, starting
from random draws of the observed values.

`pool` implements Rubin (1987) with the Barnard & Rubin (1999) small-sample
degrees of freedom and the fraction of missing information as mice reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import expit

RIDGE = 1e-5  # same order as mice's default ridge


# --------------------------------------------------------------- draws
def _norm_draw(X, y, rng):
    """Posterior draw of (beta, sigma) for y ~ N(X beta, sigma^2), flat prior."""
    n, p = X.shape
    XtX = X.T @ X
    XtX += RIDGE * np.diag(np.diag(XtX) + 1e-12)
    L = np.linalg.cholesky(XtX)
    beta_hat = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta_hat
    df = max(n - p, 1)
    sigma = np.sqrt(resid @ resid / rng.chisquare(df))
    z = rng.standard_normal(p)
    beta_star = beta_hat + sigma * np.linalg.solve(L.T, z)   # cov = sigma^2 (X'X)^-1
    return beta_hat, beta_star, sigma


def _logreg_draw(X, y, rng, iters=25):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        mu = expit(X @ beta)
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-4 * np.eye(p)
        step = np.linalg.solve(H, X.T @ (y - mu) - 1e-4 * beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = expit(X @ beta)
    H = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-4 * np.eye(p)
    L = np.linalg.cholesky(np.linalg.inv(H))
    return beta + L @ rng.standard_normal(p)


def _pmm_match(yhat_obs, yhat_mis, y_obs, donors, rng):
    """For each missing row pick one of the `donors` closest observed rows."""
    order = np.argsort(yhat_obs)
    ys, vs = yhat_obs[order], y_obs[order]
    n_obs = len(ys)
    k = min(donors, n_obs)
    w = min(2 * k, n_obs)
    pos = np.searchsorted(ys, yhat_mis)
    # a window of 2k sorted neighbours around the insertion point holds the k nearest
    start = np.clip(pos - k, 0, n_obs - w)
    idx = start[:, None] + np.arange(w)[None, :]
    dist = np.abs(ys[idx] - yhat_mis[:, None])
    dist += rng.random(dist.shape) * 1e-12          # random tie-breaking
    nearest = np.take_along_axis(idx, np.argsort(dist, axis=1)[:, :k], axis=1)
    pick = nearest[np.arange(len(yhat_mis)), rng.integers(0, k, len(yhat_mis))]
    return vs[pick]


def _rf_draw(X_obs, y_obs, X_mis, rng, ntree=10, min_leaf=5):
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=ntree, min_samples_leaf=min_leaf,
                               max_features=1 / 3, random_state=int(rng.integers(2**31)))
    rf.fit(X_obs, y_obs)
    Lo, Lm = rf.apply(X_obs), rf.apply(X_mis)
    n_mis = len(X_mis)
    # leaf sizes for every (missing row, tree); a donor drawn uniformly from the pooled
    # leaves picks tree t with probability proportional to its leaf size
    sizes = np.empty((n_mis, ntree))
    sorted_leaf, sorted_val = [], []
    for t in range(ntree):
        o = np.argsort(Lo[:, t], kind="stable")
        sl = Lo[o, t]
        sorted_leaf.append(sl); sorted_val.append(y_obs[o])
        sizes[:, t] = (np.searchsorted(sl, Lm[:, t], "right")
                       - np.searchsorted(sl, Lm[:, t], "left"))
    cum = np.cumsum(sizes, axis=1)
    u = rng.random(n_mis) * cum[:, -1]
    tree = (u[:, None] >= cum).sum(axis=1)
    out = np.empty(n_mis)
    for t in range(ntree):
        r = tree == t
        if r.any():
            lo = np.searchsorted(sorted_leaf[t], Lm[r, t], "left")
            k = rng.integers(0, sizes[r, t].astype(int))
            out[r] = sorted_val[t][lo + k]
    return out


# ----------------------------------------------------------------- MICE
def impute_one(X_obs, y_obs, X_mis, method, rng, donors=5):
    if method == "rf":
        return _rf_draw(X_obs[:, 1:], y_obs, X_mis[:, 1:], rng)
    if method == "logreg":
        beta = _logreg_draw(X_obs, y_obs, rng)
        return (rng.random(len(X_mis)) < expit(X_mis @ beta)).astype(float)
    beta_hat, beta_star, sigma = _norm_draw(X_obs, y_obs, rng)
    if method == "norm":
        return X_mis @ beta_star + sigma * rng.standard_normal(len(X_mis))
    if method == "pmm":
        return _pmm_match(X_obs @ beta_hat, X_mis @ beta_star, y_obs, donors, rng)
    raise ValueError(method)


def mice(data, methods, predictors=None, m=20, maxit=10, seed=0, donors=5,
         post=None, delta=None, bounds=None):
    """Return a list of m completed copies of `data` (2-D float array, NaN = missing).

    methods    : {column index: "norm" | "pmm" | "logreg" | "rf"} for incomplete columns
    predictors : {column index: list of predictor column indices}; default all others
    post       : optional {column index: callable(completed array) -> column}
                 recomputes a column after each update (passive imputation)
    delta      : optional {column index: shift} added to every freshly drawn
                 imputation of that column (delta-adjustment sensitivity analysis)
    bounds     : optional {column index: (low, high)} clip after the shift
    """
    data = np.asarray(data, float)
    n, k = data.shape
    miss = np.isnan(data)
    targets = [j for j in methods if miss[:, j].any()]
    preds = predictors or {}
    out = []
    for i in range(m):
        rng = np.random.default_rng(list(np.atleast_1d(seed)) + [i])
        Z = data.copy()
        for j in range(k):
            if miss[:, j].any():
                obs = Z[~miss[:, j], j]
                Z[miss[:, j], j] = rng.choice(obs, miss[:, j].sum())
        if post:
            for j, f in post.items():
                Z[:, j] = np.where(miss[:, j], f(Z), Z[:, j])
        iters = maxit if len(targets) > 1 or post else 1
        for _ in range(iters):
            for j in targets:
                cols = preds.get(j, [c for c in range(k) if c != j])
                X = np.column_stack([np.ones(n), Z[:, cols]])
                o, mm = ~miss[:, j], miss[:, j]
                Z[mm, j] = impute_one(X[o], Z[o, j], X[mm], methods[j], rng, donors)
                if delta and j in delta:
                    Z[mm, j] = Z[mm, j] + delta[j]
                    if bounds and j in bounds:
                        Z[mm, j] = np.clip(Z[mm, j], *bounds[j])
                if post:
                    for jj, f in post.items():
                        Z[:, jj] = np.where(miss[:, jj], f(Z), Z[:, jj])
        out.append(Z)
    return out


# ---------------------------------------------------------- Rubin's rules
@dataclass
class Pooled:
    est: np.ndarray
    se: np.ndarray
    df: np.ndarray
    ubar: np.ndarray
    b: np.ndarray
    riv: np.ndarray
    fmi: np.ndarray
    m: int

    def ci(self, level=0.95):
        q = stats.t.ppf(0.5 + level / 2, self.df)
        return self.est - q * self.se, self.est + q * self.se

    def p(self):
        return 2 * stats.t.sf(np.abs(self.est / self.se), self.df)


def pool(estimates, variances, df_complete):
    """Rubin's rules for scalar parameters.

    estimates, variances : arrays of shape (m, n_par)
    df_complete          : complete-data residual degrees of freedom (n - p)
    """
    Q = np.atleast_2d(np.asarray(estimates, float))
    U = np.atleast_2d(np.asarray(variances, float))
    m = Q.shape[0]
    qbar = Q.mean(0)
    ubar = U.mean(0)
    b = Q.var(0, ddof=1) if m > 1 else np.zeros_like(qbar)
    t = ubar + (1 + 1 / m) * b
    riv = (1 + 1 / m) * b / ubar
    lam = (1 + 1 / m) * b / t
    with np.errstate(divide="ignore", invalid="ignore"):
        df_old = np.where(lam > 0, (m - 1) / lam ** 2, np.inf)
    df_obs = (df_complete + 1) / (df_complete + 3) * df_complete * (1 - lam)
    df = np.where(np.isinf(df_old), df_obs, df_old * df_obs / (df_old + df_obs))
    fmi = (riv + 2 / (df + 3)) / (1 + riv)
    return Pooled(est=qbar, se=np.sqrt(t), df=df, ubar=ubar, b=b, riv=riv,
                  fmi=fmi, m=m)


def ols(X, y, robust=False):
    """OLS estimates and their variances (classical, or HC1 if robust); X includes the intercept."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    n, p = X.shape
    if robust:
        meat = (X * e[:, None] ** 2).T @ X
        return beta, n / (n - p) * np.diag(XtX_inv @ meat @ XtX_inv)
    return beta, e @ e / (n - p) * np.diag(XtX_inv)
