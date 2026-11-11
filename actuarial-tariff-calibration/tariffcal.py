"""Tools shared by the parts of the actuarial-tariff-calibration series.

Everything is exposure-weighted. A *rate* is a predicted claim frequency (claims per unit of exposure); a premium
for a policy is its exposure times its rate. Only NumPy and scikit-learn's isotonic regression are used.

    gini(rate, exposure, claims)          exposure-weighted Gini of the ordered Lorenz (concentration) curve
    lorenz(rate, exposure, claims)        the curve itself; tied rates form one straight segment
    premium_gini(rate, exposure)          Gini of the premium's own Lorenz curve (its dispersion)
    poisson_deviance(claims, expected)    total Poisson deviance
    murphy(rate, exposure, claims)        MCB, DSC and UNC of the Poisson deviance by isotonic recalibration
    lower_convex_hull(x, y)               greatest convex minorant of a curve given by its vertices
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def lorenz(rate: np.ndarray, exposure: np.ndarray, claims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz curve (a concentration curve): policies sorted by rate from lowest to highest.

    Horizontal axis: cumulative share of exposure. Vertical axis: cumulative share of claims. Policies with equal
    rates form one block joined by a straight segment, so the curve does not depend on how ties are stored.
    """
    order = np.argsort(rate, kind="stable")
    r, w, c = rate[order], exposure[order], claims[order]
    last = np.r_[r[1:] != r[:-1], True]
    cw = np.r_[0.0, np.cumsum(w)[last]] / w.sum()
    cc = np.r_[0.0, np.cumsum(c)[last]] / c.sum()
    return cw, cc


def gini(rate: np.ndarray, exposure: np.ndarray, claims: np.ndarray) -> float:
    """Twice the area between the ordered Lorenz curve and the diagonal. Depends only on the order of the rates."""
    cw, cc = lorenz(rate, exposure, claims)
    return float(1.0 - 2.0 * np.trapezoid(cc, cw))


def premium_gini(rate: np.ndarray, exposure: np.ndarray) -> float:
    """Gini of the premium's own Lorenz curve: the claims are replaced by the premiums themselves.

    For an auto-calibrated premium the concentration curve of the claims and the Lorenz curve of the premium
    coincide in the population (Denuit, Sznajder and Trufin 2019), so the two Gini indices agree.
    """
    return gini(rate, exposure, exposure * rate)


def poisson_deviance(claims: np.ndarray, expected: np.ndarray) -> float:
    """Total Poisson deviance 2 sum [y log(y / m) - (y - m)], with the y log y term set to zero when y = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(claims > 0, claims * np.log(np.where(claims > 0, claims, 1.0) / expected), 0.0)
    return float(2.0 * np.sum(t - (claims - expected)))


def isotonic_recalibration(rate_fit, exposure_fit, claims_fit) -> IsotonicRegression:
    """Isotonic regression of the observed frequency on the rate, weighted by exposure (pool-adjacent-violators)."""
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        rate_fit, claims_fit / exposure_fit, sample_weight=exposure_fit)


class StepRecalibration:
    """Isotonic recalibration used as a tariff: a step function of the rate, fitted on one sample, applied to others.

    The pool-adjacent-violators fit is computed first; adjacent steps are then merged, smallest exposure first and
    into the neighbour with the closer rate, until every step carries at least `min_exposure` units of exposure.
    Merging adjacent steps of a non-decreasing sequence keeps it non-decreasing, and every step still charges its
    own claims on the fitting sample, so the result is auto-calibrated in-sample. With min_exposure = 0 this is the
    plain isotonic fit, evaluated as a step function rather than interpolated.
    """

    def __init__(self, min_exposure: float = 0.0):
        self.min_exposure = float(min_exposure)

    def fit(self, rate: np.ndarray, exposure: np.ndarray, claims: np.ndarray) -> "StepRecalibration":
        order = np.argsort(rate, kind="stable")
        r, e, c = rate[order], exposure[order], claims[order]
        fitted = isotonic_recalibration(r, e, c).predict(r)
        start = np.r_[0, np.flatnonzero(np.diff(fitted) != 0) + 1]
        bounds = list(r[start])                              # lowest rate of each step
        expo = list(np.add.reduceat(e, start))
        clm = list(np.add.reduceat(c, start))
        while len(expo) > 1 and min(expo) < self.min_exposure:
            j = int(np.argmin(expo))
            rate_j = clm[j] / expo[j]
            if j == 0:
                k = 1
            elif j == len(expo) - 1:
                k = j - 1
            else:
                left, right = clm[j - 1] / expo[j - 1], clm[j + 1] / expo[j + 1]
                k = j - 1 if abs(rate_j - left) <= abs(right - rate_j) else j + 1
            lo, hi = min(j, k), max(j, k)
            expo[lo] += expo[hi]
            clm[lo] += clm[hi]
            del expo[hi], clm[hi], bounds[hi]
        self.lower_ = np.array(bounds)
        self.exposure_ = np.array(expo)
        self.claims_ = np.array(clm)
        self.rates_ = self.claims_ / self.exposure_
        return self

    @property
    def steps(self) -> int:
        return int(self.rates_.size)

    def predict(self, rate: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.lower_, rate, side="right") - 1, 0, self.rates_.size - 1)
        return self.rates_[idx]


def murphy(rate: np.ndarray, exposure: np.ndarray, claims: np.ndarray) -> dict:
    """Decompose the Poisson deviance of a rate as D = UNC - DSC + MCB (Gneiting and Resin 2023).

    The recalibrated rate is the isotonic regression of claims / exposure on the rate in the same sample (the
    CORP approach of Dimitriadis, Gneiting and Jordan 2021). UNC is the deviance of the flat rate, DSC how much
    the recalibrated rate improves on it, MCB how much the rate loses against its own recalibration.
    """
    rc = isotonic_recalibration(rate, exposure, claims).predict(rate)
    flat = claims.sum() / exposure.sum()
    d = poisson_deviance(claims, exposure * rate)
    d_rc = poisson_deviance(claims, exposure * rc)
    unc = poisson_deviance(claims, exposure * flat)
    return {"D": d, "MCB": d - d_rc, "DSC": unc - d_rc, "UNC": unc, "rc": rc, "blocks": int(np.unique(rc).size)}


def lower_convex_hull(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices of the greatest convex minorant of the piecewise-linear curve through (x, y), x increasing."""
    hx, hy = [x[0]], [y[0]]
    for xi, yi in zip(x[1:], y[1:]):
        while len(hx) >= 2 and (hy[-1] - hy[-2]) * (xi - hx[-1]) >= (yi - hy[-1]) * (hx[-1] - hx[-2]):
            hx.pop()
            hy.pop()
        hx.append(xi)
        hy.append(yi)
    return np.array(hx), np.array(hy)
