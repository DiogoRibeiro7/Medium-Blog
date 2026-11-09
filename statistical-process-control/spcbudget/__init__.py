"""spcbudget: control-chart limits designed around a false-alarm budget.

A small companion module for Part 4 of the statistical-process-control series. It collects the tools the
article uses to turn a stated false-alarm budget into limits or thresholds with a stated guarantee:

Model-based guarantees (normal data, limits estimated from a Phase I sample)
    conditional_fap           conditional false-alarm probability of limits mu_hat +/- k sigma_hat
    guarantee_probability     P(conditional false-alarm probability <= alpha), exact under normality
    guaranteed_k              the k that makes that probability 1 - eps (a two-sided tolerance factor)
    bootstrap_guaranteed_k    the same target by parametric or nonparametric bootstrap (Gandy and Kvaloy 2013)
    line_guaranteed_k         a common k that guarantees the false-alarm probability of a whole line of charts

Budget allocation
    sidak_alpha, sidak_k      per-chart rate and known-parameter limit for a line-level budget

Distribution-free guarantees (exchangeable in-control scores)
    min_calibration_size      in-control points needed to guarantee a rate (Wilks 1941)
    calibrated_threshold      order-statistic threshold with a calibration-conditional guarantee
    conformal_pvalues         conformal p-values of new scores against a calibration set
    benjamini_hochberg        BH step-up rejections

No distributional assumption, long-run frequency only
    AdaptiveThreshold         adaptive conformal threshold (Gibbs and Candes 2021)

Everything is deterministic given its inputs (random draws use the generator passed in).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

__all__ = [
    "conditional_fap", "guarantee_probability", "guaranteed_k", "bootstrap_guaranteed_k",
    "line_guaranteed_k", "sidak_alpha", "sidak_k", "min_calibration_size", "calibrated_threshold",
    "conformal_pvalues", "benjamini_hochberg", "AdaptiveThreshold", "CalibratedThreshold",
]

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(96)   # probabilists' Gauss-Hermite nodes
_GH_W = _GH_W / _GH_W.sum()


# ---------------------------------------------------------------------------------------------------
# Model-based guarantees
# ---------------------------------------------------------------------------------------------------


def conditional_fap(z, s, k):
    """False-alarm probability of limits mu_hat +/- k sigma_hat for one N(mu, sigma^2) observation.

    z = (mu_hat - mu) / sigma_obs and s = sigma_hat / sigma, where sigma_obs is the standard deviation of
    the charted statistic (sigma for individuals, sigma / sqrt(n) for subgroup means).
    """
    z = np.asarray(z, dtype=float)
    s = np.asarray(s, dtype=float)
    return special.ndtr(-(z + k * s)) + special.ndtr(z - k * s)


def _s_threshold(z: float, k: float, alpha: float) -> float:
    """The s at which conditional_fap(z, s, k) = alpha; the probability falls as s grows."""
    f = lambda u: special.ndtr(-(z + u)) + special.ndtr(z - u) - alpha  # u = k s
    if f(0.0) <= 0:
        return 0.0
    hi = 1.0
    while f(hi) > 0:
        hi *= 2
    return optimize.brentq(f, 0.0, hi, xtol=1e-13) / k


def guarantee_probability(k: float, m: int, alpha: float, nu: int | None = None) -> float:
    """P(conditional false-alarm probability <= alpha) for limits from m Phase I points.

    Individuals chart: the mean of m observations and their sample standard deviation (nu = m - 1).
    X-bar chart with m subgroups of size n: pass nu = m (n - 1) for the pooled standard deviation; the
    charted statistic is then the subgroup mean and z has variance 1/m in its units. Exact under normality,
    up to a 96-point Gauss-Hermite rule over the centre-line error.
    """
    nu = m - 1 if nu is None else nu
    zs = _GH_X / math.sqrt(m)
    probs = [stats.chi2.sf(nu * _s_threshold(abs(z), k, alpha) ** 2, nu) for z in zs]
    return float(np.dot(_GH_W, probs))


def guaranteed_k(m: int, alpha: float, eps: float, nu: int | None = None) -> float:
    """Smallest k with P(conditional false-alarm probability <= alpha) >= 1 - eps.

    For the individuals chart this is the exact two-sided normal tolerance factor with content 1 - alpha and
    confidence 1 - eps (Goedhart, Schoonhoven and Does 2017, 2018).
    """
    f = lambda k: guarantee_probability(k, m, alpha, nu) - (1 - eps)
    lo, hi = stats.norm.isf(alpha / 2) * 0.5, stats.norm.isf(alpha / 2) * 1.5
    while f(hi) < 0:
        hi *= 1.5
    return optimize.brentq(f, lo, hi, xtol=1e-10)


def bootstrap_guaranteed_k(x, alpha: float, eps: float, rng: np.random.Generator, B: int = 2000,
                           kind: str = "parametric") -> float:
    """Bootstrap version of guaranteed_k for an individuals chart (Gandy and Kvaloy 2013).

    Resamples the Phase I data x (from N(mean, sd^2) if kind == 'parametric', with replacement otherwise),
    re-estimates the limits, evaluates their false-alarm probability under the fitted (or empirical)
    distribution, and returns the smallest k whose bootstrap (1 - eps) quantile of that probability is at
    most alpha. With the empirical distribution the probability can only take values in steps of 1/m, so the
    nonparametric version cannot resolve rates much below 1/m.
    """
    x = np.asarray(x, dtype=float)
    m = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    if kind == "parametric":
        xb = rng.normal(mu, sd, size=(B, m))
    else:
        xb = x[rng.integers(0, m, size=(B, m))]
    mub, sdb = xb.mean(axis=1), xb.std(axis=1, ddof=1)

    def q(k):
        if kind == "parametric":
            p = conditional_fap((mub - mu) / sd, sdb / sd, k)
        else:
            lo, hi = mub - k * sdb, mub + k * sdb
            xs = np.sort(x)
            p = (np.searchsorted(xs, lo, side="left") + (m - np.searchsorted(xs, hi, side="right"))) / m
        return np.quantile(p, 1 - eps, method="higher")

    lo, hi = 0.5, 30.0
    if q(hi) > alpha:
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if q(mid) <= alpha:
            hi = mid
        else:
            lo = mid
    return hi


def line_guaranteed_k(m: int, K: int, line_alpha: float, eps: float, rng: np.random.Generator,
                      reps: int = 4000, nu: int | None = None) -> float:
    """Common k for K independent charts (each with its own m-point Phase I) such that the line's
    false-alarm probability 1 - prod(1 - p_j) is at most line_alpha with probability 1 - eps.

    Monte Carlo over `reps` simulated lines with common random numbers, so the result is a deterministic
    function of the generator state.
    """
    nu = m - 1 if nu is None else nu
    z = rng.normal(0.0, 1.0 / math.sqrt(m), size=(reps, K))
    s = np.sqrt(rng.chisquare(nu, size=(reps, K)) / nu)

    def share(k):
        logq = np.sum(np.log1p(-conditional_fap(z, s, k)), axis=1)
        return np.mean(-np.expm1(logq) <= line_alpha)

    lo, hi = sidak_k(line_alpha, K) * 0.8, sidak_k(line_alpha, K) * 3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if share(mid) >= 1 - eps:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------------------------------


def sidak_alpha(line_alpha: float, K: int) -> float:
    """Per-chart false-alarm probability that gives K independent charts a line probability line_alpha."""
    return -math.expm1(math.log1p(-line_alpha) / K)


def sidak_k(line_alpha: float, K: int) -> float:
    """Two-sided normal limit width for the per-chart Sidak rate, with known parameters."""
    return float(stats.norm.isf(sidak_alpha(line_alpha, K) / 2))


# ---------------------------------------------------------------------------------------------------
# Distribution-free guarantees
# ---------------------------------------------------------------------------------------------------


def _guarantee_rank(n: int, alpha: float, eps: float) -> int:
    """Largest r such that the r-th largest of n calibration scores, used as a threshold, has
    exceedance probability <= alpha with probability >= 1 - eps; 0 if even the maximum does not."""
    # exceedance of the r-th largest ~ Beta(r, n - r + 1); P(Beta <= alpha) = P(Binomial(n, alpha) >= r)
    r = 0
    while r < n and stats.binom.sf(r, n, alpha) >= 1 - eps:   # sf(r) = P(Bin >= r + 1)
        r += 1
    return r


def min_calibration_size(alpha: float, eps: float, rank: int = 1) -> int:
    """Smallest n for which the rank-th largest of n exchangeable in-control scores is a threshold whose
    exceedance probability is at most alpha with probability at least 1 - eps (Wilks 1941). For rank 1 this
    is ceil(log(eps) / log(1 - alpha))."""
    if rank == 1:
        return int(math.ceil(math.log(eps) / math.log1p(-alpha)))
    n = rank
    while stats.binom.sf(rank - 1, n, alpha) < 1 - eps:
        n = int(n * 1.05) + 1
    lo = max(rank, int(n / 1.05) - 2)
    while stats.binom.sf(rank - 1, lo, alpha) < 1 - eps:
        lo += 1
    return lo


@dataclass
class CalibratedThreshold:
    threshold: float     # alarm when a new score is strictly above this value
    rank: int            # the threshold is the rank-th largest calibration score
    n: int
    guarantee: float     # P(exceedance probability <= alpha) for this rank, under exchangeability


def calibrated_threshold(scores, alpha: float, eps: float) -> CalibratedThreshold:
    """Order-statistic threshold with a calibration-conditional guarantee: with probability at least
    1 - eps over the calibration set, a new exchangeable in-control score exceeds it with probability at
    most alpha (Wilks 1941; Vovk 2012; Bates et al. 2023). Raises ValueError if the calibration set is too
    small for any rank to give the guarantee."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    r = _guarantee_rank(n, alpha, eps)
    if r == 0:
        raise ValueError(f"{n} calibration scores cannot guarantee alpha = {alpha} at eps = {eps}; "
                         f"need at least {min_calibration_size(alpha, eps)}")
    return CalibratedThreshold(float(s[n - r]), r, n, float(stats.binom.sf(r - 1, n, alpha)))


def conformal_pvalues(calibration, test):
    """Conformal p-values (1 + #{calibration >= score}) / (n + 1); large scores are unusual."""
    cal = np.sort(np.asarray(calibration, dtype=float))
    test = np.asarray(test, dtype=float)
    n = len(cal)
    ge = n - np.searchsorted(cal, test, side="left")
    return (1 + ge) / (n + 1)


def benjamini_hochberg(pvalues, q: float):
    """Boolean mask of BH step-up rejections at level q (Benjamini and Hochberg 1995)."""
    p = np.asarray(pvalues, dtype=float)
    K = p.size
    order = np.argsort(p, kind="stable")
    passed = p[order] <= q * np.arange(1, K + 1) / K
    reject = np.zeros(K, dtype=bool)
    if passed.any():
        reject[order[: np.flatnonzero(passed).max() + 1]] = True
    return reject


# ---------------------------------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------------------------------


@dataclass
class AdaptiveThreshold:
    """Adaptive conformal threshold (Gibbs and Candes 2021) for a stream of monitoring scores.

    At each step the threshold is the empirical (1 - alpha_t) quantile of the last `window` scores; an alarm
    is raised when the new score exceeds it, and alpha_t is updated by alpha_t + gamma (alpha - alarm). Over
    any T steps the alarm frequency is within (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T) of alpha,
    whatever the data. When alpha_t <= 0 no alarm is possible; when alpha_t >= 1 every score alarms.
    """
    alpha: float
    gamma: float
    window: int
    history: list = field(default_factory=list)
    alpha_t: float | None = None

    def __post_init__(self):
        if self.alpha_t is None:
            self.alpha_t = self.alpha

    def threshold(self) -> float:
        if not self.history or self.alpha_t >= 1:
            return -math.inf
        if self.alpha_t <= 0:
            return math.inf
        h = np.asarray(self.history[-self.window:])
        return float(np.quantile(h, 1 - self.alpha_t, method="higher"))

    def update(self, score: float) -> bool:
        alarm = bool(score > self.threshold())
        self.alpha_t += self.gamma * (self.alpha - alarm)
        self.history.append(float(score))
        return alarm
