"""Statistical building blocks: survey estimation, meta-analysis, disproportionality.

Everything here is written out rather than imported from a specialised package,
so that each formula can be read next to the article.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- survey
@dataclass
class SurveyEstimate:
    est: float
    se: float
    n: int            # unweighted sample size in the domain

    @property
    def ci(self) -> tuple[float, float]:
        return self.est - 1.96 * self.se, self.est + 1.96 * self.se


def survey_mean(y, w, strata, psu, domain=None) -> SurveyEstimate:
    """Weighted mean of y over a domain, with a Taylor-linearised standard error.

    The design is stratified with clusters (PSUs) sampled with replacement within
    strata, the standard NHANES approximation. Domain estimation keeps the whole
    sample in the variance calculation and zeroes the linearised values outside
    the domain, so subgroup standard errors reflect the design correctly.
    Proportions are means of 0/1 variables.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    strata = np.asarray(strata)
    psu = np.asarray(psu)
    d = np.ones(len(y), dtype=bool) if domain is None else np.asarray(domain, dtype=bool)
    d = d & np.isfinite(y) & np.isfinite(w) & (w > 0)
    wd = np.where(d, w, 0.0)
    yd = np.where(d, y, 0.0)
    total_w = wd.sum()
    if total_w <= 0:
        return SurveyEstimate(float("nan"), float("nan"), 0)
    est = (wd * yd).sum() / total_w
    z = wd * (yd - est) / total_w
    var = 0.0
    keys = np.char.add(np.char.add(strata.astype(str), "|"), psu.astype(str))
    for h in np.unique(strata):
        in_h = strata == h
        psus, inv = np.unique(keys[in_h], return_inverse=True)
        n_h = len(psus)
        if n_h < 2:
            continue  # a stratum with a single PSU contributes nothing (or is collapsed upstream)
        totals = np.bincount(inv, weights=z[in_h], minlength=n_h)
        var += n_h / (n_h - 1) * ((totals - totals.mean()) ** 2).sum()
    return SurveyEstimate(float(est), float(np.sqrt(var)), int(d.sum()))


def weighted_quantile(x, w, q):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    order = np.argsort(x, kind="stable")  # stable: tie order (and so the result) must not depend on the CPU
    x, w = x[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(q, cw, x)


def within_person_sd(readings: np.ndarray) -> tuple[float, int]:
    """Pooled within-person standard deviation of replicate readings.

    readings is an (n_people, n_replicates) array with NaN for missing readings.
    Each person with at least two readings contributes their sum of squared
    deviations from their own mean and (k - 1) degrees of freedom.
    """
    r = np.asarray(readings, dtype=float)
    k = np.isfinite(r).sum(axis=1)
    use = k >= 2
    r = r[use]
    k = k[use]
    m = np.nanmean(r, axis=1)
    ss = np.nansum((r - m[:, None]) ** 2, axis=1)
    df = (k - 1).sum()
    return float(np.sqrt(ss.sum() / df)), int(use.sum())


# --------------------------------------------------------------------------- meta-analysis
def ln_vr(sd_t, n_t, sd_c, n_c):
    """Log variability ratio with the small-sample bias correction (Nakagawa et al. 2015).

    lnVR = ln(SD_t / SD_c) + 1/(2(n_t - 1)) - 1/(2(n_c - 1)),
    var  = 1/(2(n_t - 1)) + 1/(2(n_c - 1)).
    """
    sd_t, n_t, sd_c, n_c = map(np.asarray, (sd_t, n_t, sd_c, n_c))
    y = np.log(sd_t / sd_c) + 1 / (2 * (n_t - 1)) - 1 / (2 * (n_c - 1))
    v = 1 / (2 * (n_t - 1)) + 1 / (2 * (n_c - 1))
    return y, v


@dataclass
class Pooled:
    est: float
    se: float
    tau2: float
    i2: float
    q: float
    k: int

    @property
    def ci(self) -> tuple[float, float]:
        return self.est - 1.96 * self.se, self.est + 1.96 * self.se

    @property
    def p(self) -> float:
        return float(2 * stats.norm.sf(abs(self.est / self.se))) if self.se > 0 else float("nan")


def dersimonian_laird(y, v) -> Pooled:
    """Random-effects pooled estimate (DerSimonian and Laird, 1986)."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = len(y)
    if k == 0:
        return Pooled(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0)
    w = 1 / v
    fixed = (w * y).sum() / w.sum()
    q = float((w * (y - fixed) ** 2).sum())
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1 / (v + tau2)
    est = (ws * y).sum() / ws.sum()
    se = np.sqrt(1 / ws.sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0
    return Pooled(float(est), float(se), float(tau2), float(i2), q, k)


def log_risk_ratio(a, n1, c, n0):
    """Log risk ratio and its variance, with a 0.5 correction when any cell is zero."""
    a, n1, c, n0 = (np.asarray(x, dtype=float) for x in (a, n1, c, n0))
    zero = (a == 0) | (c == 0) | (a == n1) | (c == n0)
    a2 = np.where(zero, a + 0.5, a)
    c2 = np.where(zero, c + 0.5, c)
    n12 = np.where(zero, n1 + 1, n1)
    n02 = np.where(zero, n0 + 1, n0)
    y = np.log((a2 / n12) / (c2 / n02))
    v = 1 / a2 - 1 / n12 + 1 / c2 - 1 / n02
    return y, v


# --------------------------------------------------------------------------- disproportionality
@dataclass
class ROR:
    ror: float
    lo: float
    hi: float
    a: int
    b: int
    c: int
    d: int


def reporting_odds_ratio(a: int, b: int, c: int, d: int) -> ROR:
    """Reporting odds ratio (a/b)/(c/d) with a Wald 95% interval on the log scale.

    a: reports with the drug and the event, b: the drug without the event,
    c: other drugs with the event, d: other drugs without the event.
    """
    if min(a, b, c, d) <= 0:
        return ROR(float("nan"), float("nan"), float("nan"), a, b, c, d)
    lr = np.log((a / b) / (c / d))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return ROR(float(np.exp(lr)), float(np.exp(lr - 1.96 * se)), float(np.exp(lr + 1.96 * se)), a, b, c, d)


# --------------------------------------------------------------------------- one-patient experiments
def p_wrong_direction(d, k):
    """Chance that the mean of k on-minus-off differences has the wrong sign.

    d is the true benefit in units of the SD of one on-minus-off difference.
    """
    return stats.norm.cdf(-np.asarray(d) * np.sqrt(np.asarray(k)))


def readings_needed(shift: float, sd_within: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Readings on and off (each) to detect a mean shift with a one-sided z-test.

    With m readings in each period, the difference of means has SD
    sd_within * sqrt(2/m); solve shift / that = z_alpha + z_power.
    """
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    m = 2 * (z * sd_within / shift) ** 2
    return int(np.ceil(m))
