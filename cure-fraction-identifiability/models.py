"""Univariate mixture cure models and their no-cure counterparts, fitted by maximum likelihood.

Mixture cure model: a unit never has the event with probability pi; otherwise its event time
has survival function S0(t). The population survival function is

    S(t) = pi + (1 - pi) S0(t),

which levels off at pi. Two latency families are implemented, lognormal and Weibull, each with a
cure version (pi estimated) and a no-cure version (pi = 0). Parameters are unconstrained:

    cure lognormal:   (logit pi, mu, log sigma)           log T0 ~ Normal(mu, sigma^2)
    cure Weibull:     (logit pi, log scale, log shape)    S0(t) = exp(-(t / scale)^shape)
    no-cure versions: the same without logit pi.

Also: the Kaplan-Meier estimator, the Kaplan-Meier plateau estimate of pi (the value of the
Kaplan-Meier curve at the largest uncensored time) and the Maller-Zhou alpha_n statistic for
sufficient follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr

FAMILIES = ("lognormal", "weibull")


def _log_latency(family: str, params: np.ndarray, t: np.ndarray, logt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log f0(t) and log S0(t) for the latency family."""
    if family == "lognormal":
        mu, log_s = params
        s = np.exp(log_s)
        u = (logt - mu) / s
        log_f = -0.5 * u * u - log_s - 0.5 * np.log(2 * np.pi) - logt
        log_S = log_ndtr(-u)
    elif family == "weibull":
        log_scale, log_shape = params
        k = np.exp(log_shape)
        z = k * (logt - log_scale)
        H = np.exp(np.minimum(z, 700.0))  # cumulative hazard (t / scale)^k, clipped against overflow
        log_f = log_shape - log_scale + (k - 1) * (logt - log_scale) - H
        log_S = -H
    else:
        raise ValueError(family)
    return log_f, log_S


@dataclass
class Fit:
    family: str
    cure: bool
    params: np.ndarray
    loglik: float
    converged: bool
    pi: float  # 0 for no-cure fits
    pi_lo: float  # 90% Wald interval on the logit scale, mapped back
    pi_hi: float

    @property
    def aic(self) -> float:
        return 2 * len(self.params) - 2 * self.loglik

    def survival(self, t: np.ndarray | float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        lat = self.params[1:] if self.cure else self.params
        _, log_S = _log_latency(self.family, lat, t, np.log(t))
        S0 = np.exp(log_S)
        return self.pi + (1 - self.pi) * S0


def negloglik(theta: np.ndarray, family: str, cure: bool, t: np.ndarray, e: np.ndarray, logt: np.ndarray) -> float:
    if cure:
        a, lat = theta[0], theta[1:]
        log_pi = -np.logaddexp(0.0, -a)
        log_1mpi = -np.logaddexp(0.0, a)
    else:
        lat = theta
        log_pi, log_1mpi = -np.inf, 0.0
    log_f, log_S = _log_latency(family, lat, t, logt)
    ll_event = log_1mpi + log_f
    ll_cens = np.logaddexp(log_pi, log_1mpi + log_S) if cure else log_S
    ll = np.where(e == 1, ll_event, ll_cens)
    v = -float(np.sum(ll))
    return v if np.isfinite(v) else 1e300


def _start(family: str, cure: bool, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    te = t[e == 1]
    logte = np.log(te) if len(te) > 1 else np.log(t)
    mu = float(np.median(logte))
    sd = float(max(np.std(logte), 0.2))
    lat = np.array([mu, np.log(sd)]) if family == "lognormal" else np.array([mu + 0.4, np.log(1.2 / sd)])
    if not cure:
        return lat
    frac_cens = 1 - e.mean()
    p0 = min(max(0.5 * frac_cens, 0.02), 0.9)
    return np.concatenate([[np.log(p0 / (1 - p0))], lat])


def _hessian(f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    k = len(x)
    H = np.zeros((k, k))
    fx = f(x)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h
        H[i, i] = (f(x + ei) - 2 * fx + f(x - ei)) / h**2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def fit(t: np.ndarray, e: np.ndarray, family: str = "lognormal", cure: bool = True, z90: float = 1.6448536) -> Fit:
    t = np.maximum(np.asarray(t, float), 1e-6)
    e = np.asarray(e, int)
    logt = np.log(t)
    obj = lambda th: negloglik(th, family, cure, t, e, logt)  # noqa: E731
    best = None
    starts = [_start(family, cure, t, e)]
    if cure:
        s2 = starts[0].copy()
        s2[0] = -2.0
        starts.append(s2)
    for x0 in starts:
        r = minimize(obj, x0, method="BFGS", options={"gtol": 1e-5, "maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    # BFGS often stops with "precision loss" at an optimum whose gradient is already ~1e-5;
    # count those as converged
    ok = bool(best.success) or bool(np.max(np.abs(best.jac)) < 1e-3)
    if not cure:
        return Fit(family, False, th, -best.fun, ok, 0.0, 0.0, 0.0)
    a = th[0]
    pi = float(expit(a))
    lo, hi = 0.0, 1.0
    try:
        H = _hessian(obj, th)
        cov = np.linalg.inv(H)
        se = float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.inf
        if np.isfinite(se):
            lo, hi = float(expit(a - z90 * se)), float(expit(a + z90 * se))
    except np.linalg.LinAlgError:
        pass
    return Fit(family, True, th, -best.fun, ok, pi, lo, hi)


def kaplan_meier(t: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Event times and the Kaplan-Meier survival just after each of them."""
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    times = np.unique(t[e == 1])
    S, out = 1.0, []
    for u in times:
        n_at_risk = np.sum(t >= u)
        d = np.sum((t == u) & (e == 1))
        S *= 1 - d / n_at_risk
        out.append(S)
    return times, np.array(out)


def km_at(t: np.ndarray, e: np.ndarray, grid: np.ndarray) -> np.ndarray:
    times, surv = kaplan_meier(t, e)
    idx = np.searchsorted(times, np.asarray(grid, float), side="right") - 1
    return np.where(idx >= 0, surv[np.maximum(idx, 0)] if len(surv) else 1.0, 1.0)


def km_plateau(t: np.ndarray, e: np.ndarray) -> float:
    """Kaplan-Meier at the largest uncensored time: the nonparametric estimate of pi, and an upper bound for it."""
    _, surv = kaplan_meier(t, e)
    return float(surv[-1]) if len(surv) else 1.0


def maller_zhou_alpha(t: np.ndarray, e: np.ndarray) -> float:
    """alpha_n = (1 - N_n / n)^n, N_n = uncensored times in (2 Y*_(n) - Y_(n), Y*_(n)].

    Small values (below 0.05) are read as evidence of sufficient follow-up (Maller and Zhou 1996).
    """
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    n = len(t)
    if e.sum() == 0:
        return 1.0
    y_star = t[e == 1].max()
    y_max = t.max()
    lo = 2 * y_star - y_max
    N = int(np.sum((e == 1) & (t > lo) & (t <= y_star)))
    return float((1 - N / n) ** n)


def _check() -> None:
    """Recover a known cure fraction with long follow-up."""
    rng = np.random.default_rng(1)
    n = 5000
    cured = rng.random(n) < 0.3
    T = np.exp(1.0 + 0.5 * rng.standard_normal(n))
    C = rng.uniform(0, 40, n)
    t = np.where(cured, C, np.minimum(T, C))
    e = (~cured) & (T <= C)
    f = fit(t, e, "lognormal", True)
    assert abs(f.pi - 0.3) < 0.03, f.pi
    assert abs(f.params[1] - 1.0) < 0.05 and abs(np.exp(f.params[2]) - 0.5) < 0.05
    assert abs(km_plateau(t, e) - 0.3) < 0.03
    assert maller_zhou_alpha(t, e) < 0.05


if __name__ == "__main__":
    _check()
    print("model check passed")
