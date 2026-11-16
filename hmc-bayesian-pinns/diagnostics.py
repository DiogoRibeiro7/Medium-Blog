"""Rank-normalised MCMC diagnostics (Vehtari, Gelman, Simpson, Carpenter and Bürkner, 2021).

The functions take draws of one scalar quantity with shape ``(chains, draws)``.
They follow the definitions in "Rank-normalization, folding, and localization:
an improved R-hat for assessing convergence of MCMC", Bayesian Analysis 16(2),
667-718: split chains, rank-normalise, use Geyer's initial monotone sequence
for the autocorrelation sum, and summarise the tails through indicator
functions. NumPy and SciPy only; ArviZ is used in the tests as a cross-check
when it is installed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _as_chains(draws: FloatArray) -> FloatArray:
    arr = np.asarray(draws, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 8:
        raise ValueError("draws must have shape (chains, at least 8 draws)")
    return arr


def split_chains(draws: FloatArray) -> FloatArray:
    """Split each chain into its first and second halves (odd middle draw dropped)."""
    arr = _as_chains(draws)
    half = arr.shape[1] // 2
    return np.concatenate((arr[:, :half], arr[:, -half:]), axis=0)


def rank_normalise(draws: FloatArray) -> FloatArray:
    """Replace draws by normal scores of their pooled ranks (average ranks for ties)."""
    arr = np.asarray(draws, dtype=np.float64)
    ranks = stats.rankdata(arr, method="average").reshape(arr.shape)
    return stats.norm.ppf((ranks - 0.375) / (arr.size + 0.25))


def _rhat_basic(chains: FloatArray) -> float:
    n = chains.shape[1]
    within = np.mean(np.var(chains, axis=1, ddof=1))
    between = n * np.var(np.mean(chains, axis=1), ddof=1)
    if within <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def rhat_rank(draws: FloatArray) -> float:
    """Rank-normalised split-Rhat: the larger of the bulk and folded (tail) versions."""
    split = split_chains(draws)
    bulk = _rhat_basic(rank_normalise(split))
    folded = np.abs(split - np.median(split))
    tail = _rhat_basic(rank_normalise(folded))
    return float(max(bulk, tail))


def _autocovariance(x: FloatArray) -> FloatArray:
    n = x.shape[0]
    centred = x - np.mean(x)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[:n]
    return acov / n


def ess(chains: FloatArray) -> float:
    """Effective sample size of already-split chains, Geyer initial monotone sequence.

    This is the multi-chain estimator used by Stan: autocorrelations are combined
    across chains through the variance estimate, pairs of lags are summed while
    positive, and the paired sums are forced to be non-increasing.
    """
    chains = _as_chains(chains)
    m, n = chains.shape
    acov = np.stack([_autocovariance(c) for c in chains])
    chain_mean = np.mean(chains, axis=1)
    mean_var = np.mean(acov[:, 0]) * n / (n - 1)
    var_plus = mean_var * (n - 1) / n
    if m > 1:
        var_plus += np.var(chain_mean, ddof=1)
    if var_plus <= 0:
        return float("nan")
    rho_all = 1.0 - (mean_var - np.mean(acov, axis=0)) / var_plus
    rho = np.zeros(n)
    rho[0] = 1.0
    rho_even, rho_odd = 1.0, rho_all[1]
    rho[1] = rho_odd
    # Geyer's initial positive sequence: keep pairs of lags while their sum is positive.
    t = 1
    while t < n - 3 and rho_even + rho_odd > 0.0:
        rho_even, rho_odd = rho_all[t + 1], rho_all[t + 2]
        if rho_even + rho_odd >= 0:
            rho[t + 1], rho[t + 2] = rho_even, rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:
        rho[max_t + 1] = rho_even
    # Geyer's initial monotone sequence: paired sums may not increase.
    t = 1
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2.0
        t += 2
    tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + np.sum(rho[max_t + 1 : max_t + 2])
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def ess_bulk(draws: FloatArray) -> float:
    """Bulk effective sample size: ESS of rank-normalised split chains."""
    return ess(rank_normalise(split_chains(draws)))


def ess_quantile(draws: FloatArray, probability: float) -> float:
    """ESS of the indicator I(theta <= q_p), the quantity a p-quantile estimate depends on."""
    if not 0 < probability < 1:
        raise ValueError("probability must lie in (0, 1)")
    split = split_chains(draws)
    threshold = np.quantile(split, probability)
    return ess((split <= threshold).astype(np.float64))


def ess_tail(draws: FloatArray, probabilities: tuple[float, float] = (0.05, 0.95)) -> float:
    """Tail effective sample size: the smaller of the 5% and 95% quantile ESS values."""
    return float(min(ess_quantile(draws, p) for p in probabilities))


def mcse_quantile(draws: FloatArray, probability: float) -> float:
    """Monte Carlo standard error of a quantile estimate (Vehtari et al. 2021, section 4.3)."""
    arr = _as_chains(draws)
    effective = ess_quantile(arr, probability)
    if not np.isfinite(effective) or effective <= 0:
        return float("nan")  # e.g. a chain that never moved: the indicator is constant
    flat = np.sort(arr.ravel())
    size = flat.size
    a = stats.beta.ppf([0.1586553, 0.8413447], effective * probability + 1, effective * (1 - probability) + 1)
    low = flat[int(max(np.floor(a[0] * size), 0))]
    high = flat[int(min(np.ceil(a[1] * size), size - 1))]
    return float((high - low) / 2)
