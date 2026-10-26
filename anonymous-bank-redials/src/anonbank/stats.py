"""Interval-level statistics that might reveal redials.

All of them start from the surplus of offered calls in each half hour over
what the calendar predicts: the mean of that weekday and half hour across
the data, rescaled so that each day's expected total equals its observed
total (which removes day-level shocks).

* ``lag_slope(L)``: the within-day slope of the surplus in interval t on
  the number of calls abandoned in interval t - L, without intercept and
  with the abandonments demeaned within the day. L = 1 is the statistic
  of the earlier article; negative L looks forward in time, where redials
  cannot act.
* ``lag_minus_lead``: the slope at L = 1 minus the slope at L = -1. Demand
  that clusters in time raises both equally; redials raise only the first.
* ``controlled_slope``: the coefficient on last interval's abandonments in
  a regression of the surplus on last interval's abandonments and last
  interval's surplus. Clustered demand acts through the surplus; what is
  left on the abandonments is the response of calls to hanging up.
* ``acf1`` and ``dispersion``: the within-day lag-1 autocorrelation of the
  surplus and its variance relative to the expected count (1 for Poisson
  counts around the calendar); they describe how demand clusters, and the
  joint estimator uses them to separate clustering from redials.

Each statistic has a day-cluster bootstrap for its sampling error.
"""

from __future__ import annotations

import numpy as np


def surplus(offered: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Offered minus expected, expected rescaled to each day's total."""
    scale = offered.sum(axis=1, keepdims=True) / expected.sum(axis=1,
                                                              keepdims=True)
    return offered - scale * expected


def calendar_expectation(offered: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """Mean offered calls by weekday and interval, for every day."""
    out = np.empty_like(offered, dtype=float)
    for d in np.unique(dow):
        rows = dow == d
        out[rows] = offered[rows].mean(axis=0)
    return out


def _pairs(r: np.ndarray, a: np.ndarray, lag: int):
    if lag > 0:
        x, y = a[:, :-lag], r[:, lag:]
    else:
        x, y = a[:, -lag:], r[:, :lag]
    return x - x.mean(axis=1, keepdims=True), y


def lag_slope_parts(r: np.ndarray, a: np.ndarray, lag: int):
    x, y = _pairs(r, a, lag)
    return (x * y).sum(axis=1), (x * x).sum(axis=1)


def lag_slope(r: np.ndarray, a: np.ndarray, lag: int,
              days: np.ndarray | None = None) -> float:
    num, den = lag_slope_parts(r, a, lag)
    if days is not None:
        num, den = num[days], den[days]
    return float(num.sum() / den.sum())


def controlled_parts(r: np.ndarray, a: np.ndarray):
    """Per-day cross-products for the regression of r_t on (a_{t-1},
    r_{t-1}), all demeaned within the day."""
    def dm(v):
        return v - v.mean(axis=1, keepdims=True)
    x1, x2, y = dm(a[:, :-1]), dm(r[:, :-1]), dm(r[:, 1:])
    return np.stack([(x1 * x1).sum(1), (x1 * x2).sum(1), (x2 * x2).sum(1),
                     (x1 * y).sum(1), (x2 * y).sum(1)], axis=1)


def controlled_from_parts(p: np.ndarray) -> float:
    s = p.sum(axis=0)
    xtx = np.array([[s[0], s[1]], [s[1], s[2]]])
    xty = np.array([s[3], s[4]])
    return float(np.linalg.solve(xtx, xty)[0])


class Statistics:
    """All statistics of one table of days, with bootstrap support."""

    def __init__(self, offered: np.ndarray, abandoned: np.ndarray,
                 dow: np.ndarray, expected: np.ndarray | None = None):
        if expected is None:
            expected = calendar_expectation(offered, dow)
        self.r = surplus(offered, expected)
        self.a = abandoned.astype(float)
        self.parts = {L: lag_slope_parts(self.r, self.a, L)
                      for L in (-3, -2, -1, 1, 2, 3)}
        self.cparts = controlled_parts(self.r, self.a)
        self.n_days = offered.shape[0]
        rd = self.r - self.r.mean(axis=1, keepdims=True)
        self.acparts = np.stack([(rd[:, 1:] * rd[:, :-1]).sum(1),
                                 (rd[:, 1:] ** 2).sum(1),
                                 (rd[:, :-1] ** 2).sum(1)], axis=1)
        scale = offered.sum(axis=1, keepdims=True) / expected.sum(
            axis=1, keepdims=True)
        self.dparts = np.stack([(self.r ** 2).sum(1),
                                (scale * expected).sum(1)], axis=1)

    def lag(self, L: int, days=None) -> float:
        num, den = self.parts[L]
        if days is not None:
            num, den = num[days], den[days]
        return float(num.sum() / den.sum())

    def lag_minus_lead(self, days=None) -> float:
        return self.lag(1, days) - self.lag(-1, days)

    def controlled(self, days=None) -> float:
        p = self.cparts if days is None else self.cparts[days]
        return controlled_from_parts(p)

    def acf1(self, days=None) -> float:
        p = (self.acparts if days is None else self.acparts[days]).sum(0)
        return float(p[0] / np.sqrt(p[1] * p[2]))

    def dispersion(self, days=None) -> float:
        p = (self.dparts if days is None else self.dparts[days]).sum(0)
        return float(p[0] / p[1])

    def get(self, name: str, days=None) -> float:
        return {"lag": lambda: self.lag(1, days),
                "lead": lambda: self.lag(-1, days),
                "lag_minus_lead": lambda: self.lag_minus_lead(days),
                "controlled": lambda: self.controlled(days),
                "acf1": lambda: self.acf1(days),
                "dispersion": lambda: self.dispersion(days)}[name]()

    def vector(self, names, days=None) -> np.ndarray:
        return np.array([self.get(n, days) for n in names])

    def bootstrap(self, name: str, rng: np.random.Generator,
                  reps: int = 1000) -> np.ndarray:
        idx = rng.integers(0, self.n_days, size=(reps, self.n_days))
        return np.array([self.get(name, i) for i in idx])


STATISTICS = ("lag", "lag_minus_lead", "controlled")
