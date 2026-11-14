"""Extreme value tools for storm-wind-extremes-portugal.

Everything here is written out rather than taken from a package so that each step the
article describes can be read in one place: the generalized extreme value (GEV)
likelihood with analytic gradients, maximum likelihood fits with linear predictors for
the location and the log scale, profile likelihood intervals for return levels and for
the return period of a given value, a joint fit with one shape parameter shared by many
series, and the madogram estimator of the extremal coefficient.

Parametrisation: G(x) = exp{-[1 + xi (x - mu) / sigma]^(-1/xi)}, so xi > 0 is a heavy
(Frechet-type) tail, xi = 0 the Gumbel limit and xi < 0 a tail with an upper end point.
scipy.stats.genextreme uses c = -xi.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

XI_EPS = 1e-6
BIG = 1e10
# Tight convergence, then rounding, so that runs on different machines agree to the printed digits.
OPT = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000, "maxfun": 20000}
DECIMALS = 7


# -----------------------------------------------------------------------------
# GEV distribution
# -----------------------------------------------------------------------------

def gev_cdf(x, mu, sigma, xi):
    x, mu, sigma, xi = np.broadcast_arrays(*(np.asarray(a, float) for a in (x, mu, sigma, xi)))
    z = (x - mu) / sigma
    out = np.empty_like(z)
    small = np.abs(xi) < XI_EPS
    out[small] = np.exp(-np.exp(-z[small]))
    t = 1 + xi[~small] * z[~small]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        val = np.exp(-np.where(t > 0, t, np.nan) ** (-1 / xi[~small]))
    # outside the support: below the lower end point (xi > 0) the cdf is 0, above the
    # upper end point (xi < 0) it is 1
    val = np.where(t > 0, val, np.where(xi[~small] > 0, 0.0, 1.0))
    out[~small] = val
    return out


def gev_quantile(p, mu, sigma, xi):
    """Value exceeded with probability 1 - p in one block (p is the non-exceedance probability)."""
    y = -np.log(np.asarray(p, float))
    xi = np.asarray(xi, float)
    safe = np.where(np.abs(xi) < XI_EPS, 1.0, xi)
    with np.errstate(divide="ignore", invalid="ignore"):
        gumbel = mu - sigma * np.log(y)
        general = mu + sigma * (y ** (-xi) - 1) / safe
    return np.where(np.abs(xi) < XI_EPS, gumbel, general)


def return_level(T, mu, sigma, xi):
    """Level exceeded on average once every T blocks."""
    return gev_quantile(1 - 1 / np.asarray(T, float), mu, sigma, xi)


def return_period(x, mu, sigma, xi):
    """Mean number of blocks between maxima at or above x (inf above the upper end point)."""
    p = 1 - gev_cdf(x, mu, sigma, xi)
    with np.errstate(divide="ignore"):
        return np.where(p > 0, 1 / np.maximum(p, 1e-300), np.inf)


def _terms(x, mu, s, xi):
    """Log-likelihood terms and their derivatives with respect to mu, log sigma and xi.

    Returns (ll, d_mu, d_s, d_xi, ok) elementwise; ok is False outside the support.
    """
    sigma = np.exp(s)
    z = (x - mu) / sigma
    xi = np.where(np.abs(xi) < XI_EPS, np.where(xi >= 0, XI_EPS, -XI_EPS), xi)
    t = 1 + xi * z
    ok = t > 0
    tt = np.where(ok, t, 1.0)
    lt = np.log(tt)
    u = np.exp(-lt / xi)
    ll = -s - (1 + 1 / xi) * lt - u
    d_mu = (1 + xi - u) / (sigma * tt)
    d_s = -1 + z * (1 + xi - u) / tt
    d_xi = (1 - u) * lt / xi ** 2 - (z / tt) * (xi + 1 - u) / xi
    return ll, d_mu, d_s, d_xi, ok


# -----------------------------------------------------------------------------
# Maximum likelihood with linear predictors
# -----------------------------------------------------------------------------

@dataclass
class GevFit:
    beta_mu: np.ndarray
    beta_s: np.ndarray
    xi: float
    nll: float
    n: int
    converged: bool
    cov: np.ndarray | None = None
    names: list = field(default_factory=list)

    @property
    def mu(self) -> float:
        return float(self.beta_mu[0])

    @property
    def sigma(self) -> float:
        return float(np.exp(self.beta_s[0]))

    @property
    def theta(self) -> np.ndarray:
        return np.r_[self.beta_mu, self.beta_s, self.xi]

    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov)) if self.cov is not None else np.full(len(self.theta), np.nan)


def _design(n, X):
    return np.ones((n, 1)) if X is None else np.column_stack([np.ones(n), np.asarray(X, float)])


def gev_nll(theta, x, Xm, Xs, xi_bounds=(-0.6, 0.6)):
    km = Xm.shape[1]
    bm, bs, xi = theta[:km], theta[km:-1], theta[-1]
    mu, s = Xm @ bm, Xs @ bs
    ll, d_mu, d_s, d_xi, ok = _terms(x, mu, s, xi)
    if not ok.all() or not (xi_bounds[0] <= xi <= xi_bounds[1]):
        return BIG, np.zeros_like(theta)
    grad = np.r_[-(Xm.T @ d_mu), -(Xs.T @ d_s), -d_xi.sum()]
    return -ll.sum(), grad


def _start(x):
    m, sd = np.mean(x), np.std(x, ddof=1)
    sig = max(sd * np.sqrt(6) / np.pi, 1e-3)
    return m - 0.5772 * sig, np.log(sig)


def fit_gev(x, Xmu=None, Xsig=None, start=None, fixed_xi=None, hessian=False, xi_bounds=(-0.6, 0.6),
            multi_start=True) -> GevFit:
    """GEV by maximum likelihood; mu = [1, Xmu] beta_mu and log sigma = [1, Xsig] beta_s.

    Several starting values are tried (the stationary moment fit with shape -0.1, 0 and 0.1,
    plus `start` if given) and the best optimum is kept. With multi_start=False and a
    feasible `start`, only that start is used (for the many refits of a permutation test).
    """
    x = np.asarray(x, float)
    n = len(x)
    Xm, Xs = _design(n, Xmu), _design(n, Xsig)
    km, ks = Xm.shape[1], Xs.shape[1]
    m0, s0 = _start(x)
    starts = []
    if start is not None:
        starts.append(np.asarray(start, float))
    if multi_start or start is None or gev_nll(starts[0], x, Xm, Xs, xi_bounds)[0] >= BIG:
        for xi0 in (0.0, -0.1, 0.1):
            th = np.zeros(km + ks + 1)
            th[0], th[km], th[-1] = m0, s0, xi0
            starts.append(th)
    best = None
    for th0 in starts:
        th0 = _feasible(th0.copy() if fixed_xi is None else np.r_[th0[:-1], fixed_xi], x, km)
        if fixed_xi is not None:

            def f(t):
                v, g = gev_nll(np.r_[t, fixed_xi], x, Xm, Xs, xi_bounds)
                return v, g[:-1]
            if f(th0[:-1])[0] >= BIG:
                continue
            res = optimize.minimize(f, th0[:-1], jac=True, method="L-BFGS-B", options=OPT)
            theta = np.r_[np.round(res.x, DECIMALS), fixed_xi]
        else:
            if gev_nll(th0, x, Xm, Xs, xi_bounds)[0] >= BIG:
                continue
            res = optimize.minimize(gev_nll, th0, args=(x, Xm, Xs, xi_bounds), jac=True, method="L-BFGS-B", options=OPT)
            theta = np.round(res.x, DECIMALS)
        val = gev_nll(theta, x, Xm, Xs, xi_bounds)[0]
        if best is None or val < best[1] - 1e-9:
            best = (theta, val, bool(res.success))
    if best is None:
        raise RuntimeError("no feasible starting value for the GEV fit")
    theta, val, conv = best
    cov = None
    if hessian and fixed_xi is None:
        H = numerical_hessian(lambda t: gev_nll(t, x, Xm, Xs, xi_bounds)[1], theta)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = None
    return GevFit(theta[:km], theta[km:km + ks], float(theta[-1]), float(val), n, conv, cov)


def _feasible(th, x, km):
    """Widen the starting scale until every observation is inside the GEV support."""
    xi, mu, s = th[-1], th[0], th[km]
    if abs(xi) > XI_EPS:
        edge = (x.max() - mu) if xi < 0 else (mu - x.min())
        need = abs(xi) * edge * 1.25
        if need > np.exp(s):
            th[km] = np.log(need)
    return th


def numerical_hessian(grad, theta, h=1e-5):
    k = len(theta)
    H = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = h * max(1.0, abs(theta[i]))
        H[i] = (grad(theta + e) - grad(theta - e)) / (2 * e[i])
    return (H + H.T) / 2


# -----------------------------------------------------------------------------
# Profile likelihood
# -----------------------------------------------------------------------------

def _profile_level_nll(x, zp, y, xi_bounds=(-0.6, 0.6)):
    """Minus the profile log-likelihood at a fixed quantile zp = quantile with -log(p) = y.

    mu is eliminated through mu = zp - sigma (y^-xi - 1)/xi; the remaining (log sigma, xi)
    are optimised from a grid of shape starting values.
    """
    def nll(t):
        s, xi = t
        sigma = np.exp(s)
        a = -np.log(y) if abs(xi) < XI_EPS else (y ** (-xi) - 1) / xi
        mu = zp - sigma * a
        ll, *_ , ok = _terms(x, mu, s, xi)
        if not ok.all() or not (xi_bounds[0] <= xi <= xi_bounds[1]):
            return BIG
        return -ll.sum()
    _, s0 = _start(x)
    best = BIG
    for xi0 in (-0.3, -0.15, 0.0, 0.15, 0.3):
        for ds in (-0.3, 0.0, 0.3):
            t0 = np.array([s0 + ds, xi0])
            if nll(t0) >= BIG:
                continue
            res = optimize.minimize(nll, t0, method="Nelder-Mead",
                                    options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
            best = min(best, res.fun)
    return best


def profile_return_period(x, xk, log10_T=np.linspace(0.3, 6, 115), level=0.95):
    """Profile-likelihood interval for the return period of the value xk.

    Returns (T_hat, lo, hi, curve) where curve holds (log10 T, deviance). An upper limit
    beyond the grid is reported as inf.
    """
    from scipy.stats import chi2
    x = np.asarray(x, float)
    fit = fit_gev(x)
    T_hat = float(return_period(xk, fit.mu, fit.sigma, fit.xi))
    crit = chi2.ppf(level, 1)
    dev = []
    for lt in log10_T:
        p = 10.0 ** (-lt)
        y = -np.log1p(-p)
        dev.append(2 * (_profile_level_nll(x, xk, y) - fit.nll))
    dev = np.maximum(np.array(dev), 0.0)
    inside = dev <= crit
    if not inside.any():
        lo = hi = np.nan
    else:
        idx = np.where(inside)[0]
        lo = _crossing(log10_T, dev, crit, idx[0], -1)
        hi = _crossing(log10_T, dev, crit, idx[-1], +1)
    return T_hat, lo, hi, np.column_stack([log10_T, dev])


def _crossing(grid, dev, crit, i, step):
    j = i + step
    if j < 0:
        return 10.0 ** grid[0]
    if j >= len(grid):
        return np.inf
    # linear interpolation of the deviance between the inside and outside grid points
    g = grid[i] + (crit - dev[i]) * (grid[j] - grid[i]) / (dev[j] - dev[i])
    return 10.0 ** g


def profile_return_level(x, T, level=0.95, n_grid=81):
    from scipy.stats import chi2
    x = np.asarray(x, float)
    fit = fit_gev(x)
    zhat = float(return_level(T, fit.mu, fit.sigma, fit.xi))
    y = -np.log1p(-1 / T)
    crit = chi2.ppf(level, 1)
    sd = np.std(x)
    grid = np.linspace(zhat - 2.5 * sd, zhat + 6 * sd, n_grid)
    dev = np.maximum(np.array([2 * (_profile_level_nll(x, z, y) - fit.nll) for z in grid]), 0.0)
    inside = np.where(dev <= crit)[0]
    lo = _lin(grid, dev, crit, inside[0], -1)
    hi = _lin(grid, dev, crit, inside[-1], +1)
    return zhat, lo, hi


def _lin(grid, dev, crit, i, step):
    j = i + step
    if j < 0 or j >= len(grid):
        return -np.inf if step < 0 else np.inf
    return grid[i] + (crit - dev[i]) * (grid[j] - grid[i]) / (dev[j] - dev[i])


# -----------------------------------------------------------------------------
# One shape parameter shared by many series
# -----------------------------------------------------------------------------

def pooled_nll(theta, M):
    """Independence (composite) negative log-likelihood: series j has its own mu_j and
    log sigma_j, all share xi. M is (n_blocks, n_series); NaN entries are skipped."""
    J = M.shape[1]
    mu, s, xi = theta[:J], theta[J:2 * J], theta[-1]
    mask = ~np.isnan(M)
    X = np.where(mask, M, 0.0)
    ll, d_mu, d_s, d_xi, ok = _terms(X, mu[None, :], s[None, :], xi)
    if not ok[mask].all() or not (-0.6 <= xi <= 0.6):
        return BIG, np.zeros_like(theta)
    ll, d_mu, d_s, d_xi = (np.where(mask, a, 0.0) for a in (ll, d_mu, d_s, d_xi))
    return -ll.sum(), -np.r_[d_mu.sum(0), d_s.sum(0), d_xi.sum()]


def fit_pooled_shape(M, start=None):
    M = np.asarray(M, float)
    J = M.shape[1]
    if start is None:
        st = [_start(M[~np.isnan(M[:, j]), j]) for j in range(J)]
        starts = [np.r_[[a for a, _ in st], [b for _, b in st], xi0] for xi0 in (0.0, -0.1, 0.1)]
    else:
        starts = [np.asarray(start, float)]
    best = None
    for th0 in starts:
        if pooled_nll(th0, M)[0] >= BIG:
            continue
        res = optimize.minimize(pooled_nll, th0, args=(M,), jac=True, method="L-BFGS-B", options=OPT)
        if best is None or res.fun < best.fun:
            best = res
    th = np.round(best.x, DECIMALS)
    return {"mu": th[:J], "sigma": np.exp(th[J:2 * J]), "xi": float(th[-1]), "nll": float(pooled_nll(th, M)[0]), "theta": th}


# -----------------------------------------------------------------------------
# Peaks over a threshold
# -----------------------------------------------------------------------------

def runs_decluster(x, u, r=1):
    """Cluster maxima of exceedances of u in the sequence x (NaN breaks nothing; it is
    treated as below the threshold). A cluster ends after r consecutive values at or below u.
    Returns (indices of cluster maxima, number of exceedances, number of clusters)."""
    x = np.asarray(x, float)
    above = np.where(np.isnan(x), False, x > u)
    peaks, n_exc = [], int(above.sum())
    i, n = 0, len(x)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j, last, best = i, i, i
        while j < n:
            if above[j]:
                last = j
                if x[j] > x[best]:
                    best = j
            elif j - last >= r:
                break
            j += 1
        peaks.append(best)
        i = last + 1 + r
    return np.array(peaks, int), n_exc, len(peaks)


def intervals_extremal_index(x, u):
    """Ferro and Segers (2003) intervals estimator of the extremal index."""
    x = np.asarray(x, float)
    s = np.where(~np.isnan(x) & (x > u))[0]
    if len(s) < 3:
        return np.nan
    T = np.diff(s)
    if T.max() <= 2:
        th = 2 * T.sum() ** 2 / (len(T) * (T ** 2).sum())
    else:
        th = 2 * (T - 1).sum() ** 2 / (len(T) * ((T - 1) * (T - 2)).sum())
    return float(min(1.0, th))


def gpd_pooled_nll(theta, Y, J):
    """Y is a list of excess arrays, one per series; series j has log scale theta[j], all
    share the shape theta[-1]."""
    xi = theta[-1]
    if not (-0.6 <= xi <= 0.6):
        return BIG, np.zeros_like(theta)
    xi_ = xi if abs(xi) >= XI_EPS else (XI_EPS if xi >= 0 else -XI_EPS)
    nll, grad = 0.0, np.zeros_like(theta)
    for j, y in enumerate(Y):
        s = theta[j]
        sig = np.exp(s)
        t = 1 + xi_ * y / sig
        if (t <= 0).any():
            return BIG, np.zeros_like(theta)
        lt = np.log(t)
        nll += len(y) * s + (1 + 1 / xi_) * lt.sum()
        # d/ds and d/dxi of the negative log-likelihood
        grad[j] = len(y) - (1 + 1 / xi_) * np.sum(xi_ * y / sig / t)
        grad[-1] += -lt.sum() / xi_ ** 2 + (1 + 1 / xi_) * np.sum(y / sig / t)
    return nll, grad


def fit_gpd_pooled(Y):
    J = len(Y)
    th0 = np.r_[[np.log(np.mean(y)) for y in Y], 0.0]
    best = None
    for xi0 in (0.0, -0.1, 0.1):
        th = th0.copy()
        th[-1] = xi0
        if gpd_pooled_nll(th, Y, J)[0] >= BIG:
            continue
        res = optimize.minimize(gpd_pooled_nll, th, args=(Y, J), jac=True, method="L-BFGS-B", options=OPT)
        if best is None or res.fun < best.fun:
            best = res
    th = np.round(best.x, DECIMALS)
    return {"sigma": np.exp(th[:J]), "xi": float(th[-1]), "nll": float(gpd_pooled_nll(th, Y, J)[0])}


def gpd_sf(y, sigma, xi):
    """Survival function of the GPD for excesses y >= 0 (0 beyond the upper end point)."""
    y = np.asarray(y, float)
    if abs(xi) < XI_EPS:
        return np.exp(-y / sigma)
    t = 1 + xi * y / sigma
    return np.where(t > 0, np.maximum(t, 0) ** (-1 / xi), 0.0)


# -----------------------------------------------------------------------------
# Spatial dependence of maxima
# -----------------------------------------------------------------------------

def ranks_to_uniform(M):
    """Column-wise ranks scaled to (0, 1): rank / (n + 1)."""
    from scipy.stats import rankdata
    M = np.asarray(M, float)
    return np.column_stack([rankdata(M[:, j]) / (M.shape[0] + 1) for j in range(M.shape[1])])


def madogram_theta(u1, u2):
    """Pairwise extremal coefficient from the F-madogram (Cooley, Naveau and Poncet 2006):
    nu = E|F(X1) - F(X2)| / 2, theta = (1 + 2 nu) / (1 - 2 nu); 1 = complete dependence,
    2 = independence."""
    nu = 0.5 * np.mean(np.abs(np.asarray(u1) - np.asarray(u2)))
    return (1 + 2 * nu) / (1 - 2 * nu)


def field_theta(U):
    """Extremal coefficient of a whole field: with unit Frechet margins Z = -1/log U, the
    field maximum of a max-stable field is Frechet with scale theta, so 1/max_j Z is
    exponential with rate theta and theta_hat = n / sum_s (1 / max_j Z_sj)."""
    Z = -1.0 / np.log(np.asarray(U, float))
    m = Z.max(axis=1)
    return len(m) / np.sum(1.0 / m)


def mann_kendall_slope(y):
    """Sen's slope (median of pairwise slopes) of an equally spaced series."""
    y = np.asarray(y, float)
    i, j = np.triu_indices(len(y), 1)
    return float(np.median((y[j] - y[i]) / (j - i)))
