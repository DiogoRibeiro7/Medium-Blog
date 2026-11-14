"""Count-data regression models written from scratch (NumPy/SciPy).

All models are fitted by maximum likelihood with analytic gradients: BFGS
followed by Newton steps that use a Hessian from finite differences of the
analytic gradient. Standard errors come from the inverse of that Hessian.

  poisson          Poisson regression (log link), with model-based and sandwich
                   standard errors
  negbin           negative binomial type 2 (NB2), Var(y) = mu + mu^2 / theta,
                   with log(theta) constant or linear in regressors W
                   (heterogeneous dispersion)
  zeroinfl         zero-inflated Poisson or NB2 (Lambert 1992): with probability
                   pi = 1 / (1 + exp(-z'g)) an observation is a structural zero,
                   otherwise a draw from the count distribution
  hurdle           hurdle model (Mullahy 1986): a logit for y > 0 and a
                   zero-truncated Poisson or NB2 for the positive counts
  vuong            Vuong's (1989) statistic for two models fitted to the same
                   data, with the optional AIC and BIC corrections (as printed by
                   pscl::vuong)

Every fit returns a Fit object with the parameters, covariance matrix,
log-likelihood, per-observation log-likelihoods, and functions that give the
fitted mean and the probability of any count for every observation, or for new
design matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import optimize, stats
from scipy.special import digamma, expit, gammaln, log_expit


# ------------------------------------------------------------------ distributions
def _logf_pois(y, eta):
    return y * eta - np.exp(eta) - gammaln(y + 1)


def _logf_nb(y, eta, tau):
    """log NB2 pmf with mean exp(eta) and size theta = exp(tau)."""
    th = np.exp(tau)
    lmt = np.logaddexp(tau, eta)          # log(theta + mu)
    return gammaln(y + th) - gammaln(th) - gammaln(y + 1) + th * (tau - lmt) + y * (eta - lmt)


def _dlogf_nb(y, eta, tau):
    """Derivatives of the NB2 log pmf with respect to eta = log(mu) and tau = log(theta)."""
    th, mu = np.exp(tau), np.exp(eta)
    d_eta = th * (y - mu) / (th + mu)
    d_th = digamma(y + th) - digamma(th) + (tau - np.logaddexp(tau, eta)) + (mu - y) / (th + mu)
    return d_eta, th * d_th


def _logf0_nb(eta, tau):
    return np.exp(tau) * (tau - np.logaddexp(tau, eta))


def _dlogf0_nb(eta, tau):
    th, mu = np.exp(tau), np.exp(eta)
    return -th * mu / (th + mu), th * ((tau - np.logaddexp(tau, eta)) + mu / (th + mu))


def _log1mexp(a):
    """log(1 - exp(a)) for a < 0, stable."""
    a = np.atleast_1d(np.asarray(a, float))
    out = np.empty_like(a)
    m = a > -0.6931
    out[m] = np.log(-np.expm1(a[m]))
    out[~m] = np.log1p(-np.exp(a[~m]))
    return out


def _col(A, n):
    return np.ones((n, 1)) if A is None else np.asarray(A, float).reshape(n, -1)


# ------------------------------------------------------------------ negative log-likelihoods
def negll_negbin(X, y, W):
    kx = X.shape[1]

    def f(p):
        eta, tau = X @ p[:kx], W @ p[kx:]
        ll = _logf_nb(y, eta, tau)
        de, dt = _dlogf_nb(y, eta, tau)
        return -ll.sum(), -np.concatenate([X.T @ de, W.T @ dt]), ll
    return f


def negll_zeroinfl(X, y, Z, nb):
    kx, kz = X.shape[1], Z.shape[1]
    zero = y == 0

    def f(p):
        eta, zeta = X @ p[:kx], Z @ p[kx:kx + kz]
        lpi, l1mpi = log_expit(zeta), log_expit(-zeta)
        if nb:
            tau = p[-1]
            lf, lf0 = _logf_nb(y, eta, tau), _logf0_nb(eta, tau)
        else:
            lf, lf0 = _logf_pois(y, eta), -np.exp(eta)
        ll = np.where(zero, np.logaddexp(lpi, l1mpi + lf0), l1mpi + lf)
        pi = np.exp(lpi)
        w0 = np.exp(l1mpi + lf0 - ll)                       # share of P(0) from the count component
        dz = np.where(zero, (1 - w0) * (1 - pi) - w0 * pi, -pi)
        if nb:
            de_y, dt_y = _dlogf_nb(y, eta, tau)
            de_0, dt_0 = _dlogf0_nb(eta, tau)
            de = np.where(zero, w0 * de_0, de_y)
            dt = np.where(zero, w0 * dt_0, dt_y)
            g = np.concatenate([X.T @ de, Z.T @ dz, [dt.sum()]])
        else:
            mu = np.exp(eta)
            g = np.concatenate([X.T @ np.where(zero, -w0 * mu, y - mu), Z.T @ dz])
        return -ll.sum(), -g, ll
    return f


def negll_truncated(X, y, nb):
    """Zero-truncated Poisson or NB2 for positive counts y."""
    kx = X.shape[1]

    def f(p):
        eta = X @ p[:kx]
        if nb:
            tau = p[-1]
            lf, lf0 = _logf_nb(y, eta, tau), _logf0_nb(eta, tau)
        else:
            lf, lf0 = _logf_pois(y, eta), -np.exp(eta)
        l1m = _log1mexp(lf0)
        ll = lf - l1m
        r = np.exp(lf0 - l1m)                               # f(0) / (1 - f(0))
        if nb:
            de_y, dt_y = _dlogf_nb(y, eta, tau)
            de_0, dt_0 = _dlogf0_nb(eta, tau)
            g = np.concatenate([X.T @ (de_y + r * de_0), [(dt_y + r * dt_0).sum()]])
        else:
            mu = np.exp(eta)
            g = X.T @ (y - mu - r * mu)
        return -ll.sum(), -g, ll
    return f


# ------------------------------------------------------------------ the fit object
@dataclass
class Fit:
    model: str
    params: np.ndarray
    cov: np.ndarray
    loglik: float
    ll_i: np.ndarray
    converged: bool
    blocks: dict                       # part name -> slice of params
    _pmf: Callable = field(repr=False)
    _mean: Callable = field(repr=False)
    extra: dict = field(default_factory=dict, repr=False)

    def se(self):
        return np.sqrt(np.diag(self.cov))

    def block(self, name):
        return self.params[self.blocks[name]]

    def pmf(self, k, **new):
        """P(Y = k) for every observation (new design matrices as keywords X, Z or W)."""
        return self._pmf(np.asarray(k, float), **new)

    def mean(self, **new):
        return self._mean(**new)

    @property
    def nparams(self):
        return len(self.params)

    @property
    def aic(self):
        return -2 * self.loglik + 2 * self.nparams

    def bic(self, n):
        return -2 * self.loglik + np.log(n) * self.nparams


def _guard(negll):
    """+inf and a zero gradient where the log-likelihood overflows, so that line
    searches back off instead of stopping on NaN."""
    def f(p):
        with np.errstate(all="ignore"):
            v, g = negll(p)[:2]
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            return np.inf, np.zeros_like(p)
        return v, g
    return f


def _hessian(f, theta, eps=1e-5):
    k = len(theta)
    H = np.zeros((k, k))
    for j in range(k):
        h = eps * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        H[:, j] = (f(tp)[1] - f(tm)[1]) / (2 * h)
    return 0.5 * (H + H.T)


def _maximise(negll, start):
    """BFGS, then Newton steps with backtracking. Returns (theta, cov, converged, value)."""
    f = _guard(negll)
    theta = optimize.minimize(f, np.asarray(start, float), jac=True, method="BFGS",
                              options={"gtol": 1e-8, "maxiter": 5000}).x
    for _ in range(50):
        f0, g0 = f(theta)
        H = _hessian(f, theta)
        try:
            step = np.linalg.solve(H, g0)
        except np.linalg.LinAlgError:
            break
        dec = float(g0 @ step)
        if not np.isfinite(dec) or abs(dec) < 1e-12:
            break
        t = 1.0
        while t > 1e-6:
            f1 = f(theta - t * step)[0]
            if np.isfinite(f1) and f1 <= f0 + 1e-12:
                break
            t /= 2
        if t <= 1e-6:
            break
        theta = theta - t * step
    v, g = f(theta)
    H = _hessian(f, theta)
    try:
        cov = np.linalg.inv(H)
        converged = bool(np.isfinite(v) and abs(g @ cov @ g) < 1e-8 and np.all(np.linalg.eigvalsh(H) > 0))
    except np.linalg.LinAlgError:
        cov = np.full((len(theta), len(theta)), np.nan)
        converged = False
    return theta, cov, converged, v


def _best(negll, starts):
    best = None
    for st in starts:
        r = _maximise(negll, st)
        if best is None or r[3] < best[3] - 1e-9:
            best = r
    return best


# ------------------------------------------------------------------ Poisson
def poisson(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    b = np.linalg.lstsq(X, np.log(y + 0.5), rcond=None)[0]
    for _ in range(100):
        mu = np.exp(X @ b)
        step = np.linalg.solve((X.T * mu) @ X, X.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    mu = np.exp(X @ b)
    cov = np.linalg.inv((X.T * mu) @ X)
    ll_i = _logf_pois(y, X @ b)
    sc = X * (y - mu)[:, None]

    def pmf(k, X=X):
        return np.exp(_logf_pois(k, X @ b))

    def mean(X=X):
        return np.exp(X @ b)

    return Fit("Poisson", b, cov, float(ll_i.sum()), ll_i, True, {"count": slice(0, len(b))}, pmf, mean,
               {"cov_robust": cov @ (sc.T @ sc) @ cov})


# ------------------------------------------------------------------ negative binomial (NB2)
def negbin(X, y, W=None, start=None):
    """NB2 with log(theta) = W d; W = None gives a constant theta."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, kx = X.shape
    W = _col(W, n)
    kw = W.shape[1]
    f = negll_negbin(X, y, W)
    if start is None:
        p0 = poisson(X, y)
        mu = p0.mean()
        a = max(np.sum(((y - mu) ** 2 - y) / mu ** 2) / (n - kx), 0.05)    # moment estimate of 1 / theta
        start = np.concatenate([p0.params, [-np.log(a)], np.zeros(kw - 1)])
    theta, cov, conv, _ = _maximise(f, start)
    b, d = theta[:kx], theta[kx:]
    ll_i = f(theta)[2]

    def pmf(k, X=X, W=W):
        return np.exp(_logf_nb(k, X @ b, _col(W, len(X)) @ d))

    def mean(X=X, W=W):
        return np.exp(X @ b)

    # expected information for the mean coefficients with theta held at its estimate
    # (the standard errors printed by MASS::glm.nb)
    mu, th = np.exp(X @ b), np.exp(W @ d)
    cov_exp = np.linalg.inv((X.T * (mu / (1 + mu / th))) @ X)
    return Fit("NB" if kw == 1 else "NB, dispersion varies", theta, cov, float(ll_i.sum()), ll_i, conv,
               {"count": slice(0, kx), "dispersion": slice(kx, kx + kw)}, pmf, mean, {"cov_expected": cov_exp})


# ------------------------------------------------------------------ zero-inflated models
def zeroinfl(X, y, Z=None, dist="negbin", start=None):
    """Zero-inflated Poisson (dist='poisson') or NB2 (dist='negbin'); the
    inflation logit is linear in Z (Z = None: a constant)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, kx = X.shape
    Z = _col(Z, n)
    kz = Z.shape[1]
    nb = dist == "negbin"
    f = negll_zeroinfl(X, y, Z, nb)
    if start is None:
        base = negbin(X, y) if nb else poisson(X, y)
        starts = []
        for a in (-2.0, 0.0):
            g0 = np.zeros(kz)
            g0[0] = a
            starts.append(np.concatenate([base.block("count"), g0] + ([base.block("dispersion")] if nb else [])))
    else:
        starts = [start]
    theta, cov, conv, _ = _best(f, starts)
    ll_i = f(theta)[2]
    b, g = theta[:kx], theta[kx:kx + kz]
    tau = theta[-1] if nb else None

    def count_pmf(k, e):
        return np.exp(_logf_nb(k, e, tau)) if nb else np.exp(_logf_pois(k, e))

    def pmf(k, X=X, Z=Z):
        pi = expit(_col(Z, len(X)) @ g)
        return (1 - pi) * count_pmf(k, X @ b) + pi * (k == 0)

    def mean(X=X, Z=Z):
        return (1 - expit(_col(Z, len(X)) @ g)) * np.exp(X @ b)

    def inflation(Z=Z):
        return expit(_col(Z, len(Z)) @ g)

    blocks = {"count": slice(0, kx), "zero": slice(kx, kx + kz)}
    if nb:
        blocks["dispersion"] = slice(kx + kz, kx + kz + 1)
    return Fit("ZINB" if nb else "ZIP", theta, cov, float(ll_i.sum()), ll_i, conv, blocks, pmf, mean,
               {"inflation": inflation})


# ------------------------------------------------------------------ hurdle models
def _logit(Z, d):
    g = np.zeros(Z.shape[1])
    for _ in range(100):
        p = expit(Z @ g)
        step = np.linalg.solve((Z.T * (p * (1 - p))) @ Z, Z.T @ (d - p))
        g = g + step
        if np.max(np.abs(step)) < 1e-12:
            break
    p = expit(Z @ g)
    return g, np.linalg.inv((Z.T * (p * (1 - p))) @ Z)


def hurdle(X, y, Z=None, dist="negbin"):
    """Hurdle model: logit for y > 0 on Z (Z = None: the count regressors X) and a
    zero-truncated Poisson or NB2 on X for the positive counts."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, kx = X.shape
    Z = X if Z is None else _col(Z, n)
    kz = Z.shape[1]
    nb = dist == "negbin"
    pos = y > 0
    g, cov_g = _logit(Z, pos.astype(float))
    f = negll_truncated(X[pos], y[pos], nb)
    base = negbin(X[pos], y[pos]) if nb else poisson(X[pos], y[pos])
    starts = [base.params.copy()]
    if nb:
        # the zero-truncated NB likelihood can have a second stationary point at a
        # large theta; start also from theta = 1
        s1 = base.params.copy()
        s1[-1] = 0.0
        starts.append(s1)
    theta_c, cov_c, conv, _ = _best(f, starts)
    ll_i = np.where(pos, log_expit(Z @ g), log_expit(-(Z @ g)))
    ll_i[pos] += f(theta_c)[2]
    params = np.concatenate([theta_c[:kx], g] + ([theta_c[kx:]] if nb else []))
    k = len(params)
    cov = np.zeros((k, k))
    ic = list(range(kx)) + ([k - 1] if nb else [])
    cov[np.ix_(ic, ic)] = cov_c
    cov[kx:kx + kz, kx:kx + kz] = cov_g
    b, tau = theta_c[:kx], (theta_c[-1] if nb else None)

    def lf0(e):
        return _logf0_nb(e, tau) if nb else -np.exp(e)

    def pmf(k, X=X, Z=Z):
        e = X @ b
        lf = _logf_nb(k, e, tau) if nb else _logf_pois(k, e)
        ppos = expit(_col(Z, len(X)) @ g)
        return np.where(k == 0, 1 - ppos, ppos * np.exp(lf - _log1mexp(lf0(e))))

    def mean(X=X, Z=Z):
        e = X @ b
        return expit(_col(Z, len(X)) @ g) * np.exp(e - _log1mexp(lf0(e)))

    blocks = {"count": slice(0, kx), "zero": slice(kx, kx + kz)}
    if nb:
        blocks["dispersion"] = slice(k - 1, k)
    return Fit("Hurdle-NB" if nb else "Hurdle-P", params, cov, float(ll_i.sum()), ll_i, conv, blocks, pmf, mean)


# ------------------------------------------------------------------ comparisons
def vuong(f1, f2, correction=None):
    """Vuong's statistic for model 1 against model 2 (positive favours model 1)
    and its one-sided p-value; correction None, 'aic' or 'bic'."""
    m = f1.ll_i - f2.ll_i
    n = len(m)
    adj = {None: 0.0, "aic": f1.nparams - f2.nparams, "bic": (f1.nparams - f2.nparams) * np.log(n) / 2}[correction]
    v = (m.sum() - adj) / (np.sqrt(n) * m.std(ddof=1))
    return float(v), float(stats.norm.sf(v))


def expected_counts(fit, kmax, **new):
    """Expected number of observations with each count 0..kmax."""
    return np.array([fit.pmf(k, **new).sum() for k in range(kmax + 1)])
