"""Helpers for the drift-detection companion: replicas of default drift scores and thresholds, simulated
null streams, a CUSUM average-run-length calculator and CUSUM limits with a guaranteed in-control ARL.

Nothing here downloads data. The replicas follow the source code of the libraries named in each docstring,
read at the commits listed in the README; they reproduce the score and the default rule, not the libraries'
reporting layers.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, signal, stats

# ---------------------------------------------------------------------------------------------------
# Streams under the null
# ---------------------------------------------------------------------------------------------------


def ar1(phi: float, n: int, rng: np.random.Generator, burn: int = 500) -> np.ndarray:
    """Gaussian AR(1) with unit marginal variance (started at zero, first `burn` values dropped)."""
    e = rng.standard_normal(n + burn) * math.sqrt(1.0 - phi * phi)
    return signal.lfilter([1.0], [1.0, -phi], e)[burn:]


def markov_bernoulli(p: float, rho: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary two-state Markov chain with P(1) = p and lag-1 autocorrelation rho (runs of errors).
    P(1 | 1) = p + rho (1 - p) and P(1 | 0) = p (1 - rho)."""
    p11 = p + rho * (1.0 - p)
    p01 = p * (1.0 - rho)
    u = rng.random(n)
    x = np.empty(n, dtype=np.int8)
    x[0] = u[0] < p
    for t in range(1, n):
        x[t] = u[t] < (p11 if x[t - 1] else p01)
    return x


def run_detector(detector, x) -> list[int]:
    """Feed a river detector one value at a time; return the indices at which it signals. River detectors
    reset themselves on the update after a signal."""
    alarms = []
    for i, v in enumerate(x):
        detector.update(float(v))
        if detector.drift_detected:
            alarms.append(i)
    return alarms


def page_hinkley_statistic(x, delta: float = 0.005, alpha: float = 1 - 0.0001):
    """Upper Page-Hinkley statistic written as a CUSUM: S_t = alpha S_{t-1} + (x_t - xbar_t - delta), where
    xbar_t is the running mean including x_t, and the statistic is S_t - min_{s<=t} S_s. With alpha = 1
    this is Page's CUSUM of the deviations from a self-starting mean, with reference value delta. Returns
    the statistic at each step (no reset)."""
    s, smin, mean = 0.0, math.inf, 0.0
    out = np.empty(len(x))
    for t, v in enumerate(x):
        mean += (v - mean) / (t + 1)
        s = alpha * s + (v - mean) - delta
        smin = min(smin, s)
        out[t] = s - smin
    return out


# ---------------------------------------------------------------------------------------------------
# Batch drift scores (reference window against current window)
# ---------------------------------------------------------------------------------------------------


def psi_quantile_bins(reference, current, bins: int = 10, floor: float = 1e-4) -> float:
    """Population stability index with bins at reference quantiles (the textbook construction)."""
    r = np.asarray(reference, float)
    c = np.asarray(current, float)
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(r, edges)[0] / len(r)
    q = np.histogram(c, edges)[0] / len(c)
    p = np.where(p == 0, floor, p)
    q = np.where(q == 0, floor, q)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_evidently(reference, current) -> float:
    """PSI as computed by Evidently's legacy stat test for a numerical column with more than 20 distinct
    reference values: equal-width bins from numpy's 'sturges' rule on the pooled data, empty bins filled
    with 1e-4 (or a millionth of the smallest non-empty share when that is below 1e-4)."""
    r = np.asarray(reference, float)
    c = np.asarray(current, float)
    edges = np.histogram_bin_edges(np.concatenate([r, c]), bins="sturges")
    p = np.histogram(r, edges)[0] / len(r)
    q = np.histogram(c, edges)[0] / len(c)
    for a in (p, q):
        nz = a[a != 0].min()
        a[a == 0] = nz / 1e6 if nz <= 1e-4 else 1e-4
    return float(np.sum((q - p) * np.log(q / p)))


def psi_null_mean(bins: int, n_ref: int, n_cur: int) -> float:
    """Large-sample mean of the PSI between two samples from one distribution: the PSI is then approximately
    (1/n_ref + 1/n_cur) times a chi-square with bins - 1 degrees of freedom (Yurdakul and Naranjo 2020)."""
    return (bins - 1) * (1.0 / n_ref + 1.0 / n_cur)


def wasserstein_normed(reference, current) -> float:
    """Evidently's 'wasserstein' score: first Wasserstein distance divided by max(std(reference), 0.001)
    (numpy std, ddof = 0). The default rule calls drift at a score >= 0.1."""
    r = np.asarray(reference, float)
    return float(stats.wasserstein_distance(r, current) / max(float(np.std(r)), 0.001))


def ks_statistic(a, b) -> float:
    return float(stats.ks_2samp(a, b).statistic)


def nannyml_upper_threshold(reference_chunk_scores) -> float:
    """NannyML's default univariate-drift alert threshold: mean + 3 x np.nanstd (ddof = 0) of the scores of
    the reference chunks, each computed against the full reference (no lower threshold)."""
    s = np.asarray(reference_chunk_scores, float)
    return float(np.nanmean(s) + 3.0 * np.nanstd(s))


# ---------------------------------------------------------------------------------------------------
# CUSUM average run length (Brook and Evans 1972) and a guaranteed decision interval
# ---------------------------------------------------------------------------------------------------


def _cusum_arl_one_sided(k: float, h: float, cdf, states: int = 120) -> float:
    """ARL from zero of the upper CUSUM S_t = max(0, S_{t-1} + z_t - k), signal when S_t > h, for i.i.d.
    z with distribution function cdf. Markov chain on `states` cells of width 2h / (2 states - 1)."""
    w = 2.0 * h / (2 * states - 1)
    i = np.arange(states)[:, None]
    j = np.arange(states)[None, :]
    upper = cdf((j + 0.5) * w - i * w + k)
    lower = cdf((j - 0.5) * w - i * w + k)
    P = upper - lower
    P[:, 0] = cdf(0.5 * w - np.arange(states) * w + k)
    L = np.linalg.solve(np.eye(states) - P, np.ones(states))
    return float(L[0])


def cusum_arl(k: float, h: float, shift: float = 0.0, scale: float = 1.0, two_sided: bool = True,
              sample=None) -> float:
    """ARL of a CUSUM with reference value k and decision interval h for standardized observations
    z ~ N(shift, scale^2), or, if `sample` is given, z drawn from its empirical distribution (then shift and
    scale are ignored). Two-sided: 1/ARL = 1/ARL_upper + 1/ARL_lower (exact enough in control for k > 0)."""
    if sample is None:
        def up(v):
            return stats.norm.cdf((v - shift) / scale)

        def down(v):
            return stats.norm.cdf((v + shift) / scale)
    else:
        z = np.sort(np.asarray(sample, float))
        n = len(z)

        def up(v):
            return np.searchsorted(z, v, side="right") / n

        def down(v):   # distribution function of -z
            return 1.0 - np.searchsorted(z, -v, side="left") / n
    a = _cusum_arl_one_sided(k, h, up)
    if not two_sided:
        return a
    b = _cusum_arl_one_sided(k, h, down)
    return 1.0 / (1.0 / a + 1.0 / b)


def cusum_h_for_arl(k: float, arl: float, shift: float = 0.0, scale: float = 1.0, sample=None,
                    hi: float = 15.0) -> float:
    """Decision interval giving a two-sided ARL equal to `arl` for the given input distribution (capped at
    `hi`)."""
    def f(h):
        return math.log(cusum_arl(k, h, shift, scale, sample=sample)) - math.log(arl)
    if f(hi) < 0:
        return hi
    return float(optimize.brentq(f, 0.05, hi, xtol=1e-4))


def guaranteed_cusum_h(x, k: float, arl: float, eps: float, rng: np.random.Generator, B: int = 400,
                       kind: str = "parametric") -> float:
    """Decision interval for a two-sided CUSUM on (x_t - mean_hat) / sd_hat, with mean and sd estimated from
    the Phase I sample x, such that the conditional in-control ARL is at least `arl` with probability about
    1 - eps (Gandy and Kvaloy 2013). Parametric: bootstrap samples from N(mean_hat, sd_hat^2), exact ARL by
    Markov chain. Nonparametric: bootstrap samples drawn from x, ARL under the empirical distribution of x."""
    x = np.asarray(x, float)
    m = len(x)
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    need = np.empty(B)
    for b in range(B):
        if kind == "parametric":
            xs = rng.normal(mu, sd, m)
            mb, sb = xs.mean(), xs.std(ddof=1)
            need[b] = cusum_h_for_arl(k, arl, shift=(mu - mb) / sb, scale=sd / sb)
        else:
            xs = x[rng.integers(0, m, m)]
            mb, sb = xs.mean(), xs.std(ddof=1)
            need[b] = cusum_h_for_arl(k, arl, sample=(x - mb) / sb)
    return float(np.quantile(need, 1.0 - eps))


def cusum_alarms(z, k: float, h: float, sides: bool = False) -> list:
    """Indices at which a two-sided CUSUM on standardized z signals; both sums restart at zero after a
    signal. With sides=True, (index, +1 or -1) pairs, -1 when only the lower sum crossed."""
    up = dn = 0.0
    out = []
    for i, v in enumerate(z):
        up = max(0.0, up + v - k)
        dn = max(0.0, dn - v - k)
        if up > h or dn > h:
            out.append((i, 1 if up > h else -1) if sides else i)
            up = dn = 0.0
    return out
