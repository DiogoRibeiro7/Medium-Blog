"""A simulator of this centre, for checking estimators and for indirect
inference.

Each simulated day takes the staffing of a real day (agents per half hour,
replayed as given) and draws fresh calls as Poisson counts per half hour:

    fresh rate = profile[dow, interval] x day factor x clustering,

where the profile is the mean offered calls by weekday and interval in the
data, deflated for the redials that the profile already contains; the day
factor is drawn from the real days' totals relative to the profile; and
the clustering, if switched on, is a log-normal AR(1) multiplier that
carries busy and quiet spells from one half hour to the next within the
day. Calls queue in an Erlang A queue (exponential handle times with the
data's mean, exponential patience) and a caller who hangs up calls again
with probability p after a delay drawn from a stated distribution.

The queue itself is the event-by-event simulator of the earlier articles
(``gentradeoffs.queue``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gentradeoffs.queue import (ABANDONED, OFFERED, Callers, run,
                                uniform_arrivals)

from .table import N_INTERVALS


@dataclass
class Centre:
    """Everything the simulator needs, estimated from the half-hour table."""

    profile: np.ndarray            # (7, N) mean offered calls by dow
    abandon_share: np.ndarray      # (7, N) abandoned / offered by dow
    day_factors: np.ndarray        # real days' offered / profile totals
    staffing: np.ndarray           # (days, N) agents of each real day
    staffing_dow: np.ndarray       # dow of each staffing day
    service_mean: float
    theta: float

    @classmethod
    def from_table(cls, offered: np.ndarray, abandoned: np.ndarray,
                   agents: np.ndarray, dow: np.ndarray, wait_total: float,
                   service_mean: float) -> "Centre":
        prof = np.zeros((7, offered.shape[1]))
        ab = np.zeros((7, offered.shape[1]))
        for d in np.unique(dow):
            rows = dow == d
            prof[d] = offered[rows].mean(axis=0)
            ab[d] = abandoned[rows].sum(axis=0) / np.maximum(
                offered[rows].sum(axis=0), 1)
        factors = offered.sum(axis=1) / prof[dow].sum(axis=1)
        theta = abandoned.sum() / wait_total
        return cls(prof, ab, factors, np.maximum(agents, 1).astype(np.int64),
                   dow.copy(), service_mean, theta)

    def calibrate_service(self, target_abandon: float, seed: int,
                          n_days: int = 247, lo: float = 1.0,
                          hi: float = 2.5, iters: int = 14) -> float:
        """Scale the handle time until a simulated year without redials or
        clustering abandons the same share of calls as the data.

        The recorded handle time leaves out after-call work and time agents
        are logged in but not taking calls, so the effective handle time is
        longer than the recorded one. Returns the multiplier and sets
        ``service_mean`` to the effective value.
        """
        base = self.service_mean
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            self.service_mean = base * mid
            off, ab, _ = Simulator(self).days(np.random.default_rng(seed),
                                              n_days)
            if ab.sum() / off.sum() > target_abandon:
                hi = mid
            else:
                lo = mid
        k = 0.5 * (lo + hi)
        self.service_mean = base * k
        return k


@dataclass
class Clustering:
    """Log-normal AR(1) multiplier on the fresh rate within a day."""

    rho: float = 0.0
    sigma: float = 0.0

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.sigma <= 0:
            return np.ones(n)
        z = np.empty(n)
        z[0] = rng.normal(0.0, self.sigma)
        innov = self.sigma * np.sqrt(1 - self.rho ** 2)
        for t in range(1, n):
            z[t] = self.rho * z[t - 1] + rng.normal(0.0, innov)
        return np.exp(z - self.sigma ** 2 / 2)


@dataclass
class Delay:
    """Delay between hanging up and calling again, in seconds."""

    mean: float = 600.0
    empirical: np.ndarray | None = field(default=None, repr=False)

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.empirical is not None:
            return self.empirical[rng.integers(0, self.empirical.size, n)]
        return rng.exponential(self.mean, n)


@dataclass
class Simulator:
    centre: Centre
    p: float = 0.0
    delay: Delay = field(default_factory=Delay)
    clustering: Clustering = field(default_factory=Clustering)

    def fresh_rate(self, dow: int) -> np.ndarray:
        """Profile deflated for the redials it already contains."""
        c = self.centre
        return c.profile[dow] * (1.0 - self.p * c.abandon_share[dow])

    def day(self, rng: np.random.Generator, k: int) -> tuple[np.ndarray,
                                                             np.ndarray]:
        """Offered and abandoned per interval on a day staffed like real
        day ``k``."""
        c = self.centre
        dow = int(c.staffing_dow[k])
        rate = (self.fresh_rate(dow)
                * c.day_factors[rng.integers(0, c.day_factors.size)]
                * self.clustering.draw(rng, N_INTERVALS))
        counts = rng.poisson(rate)
        arrivals = uniform_arrivals(rng, counts)
        n = arrivals.size
        callers = Callers(
            arrivals, rng.exponential(c.service_mean, n),
            rng.exponential(1.0 / c.theta, n), rng.random(n),
            self.delay.draw(rng, n), rng.exponential(c.service_mean, n),
            rng.exponential(1.0 / c.theta, n), rng.random(n),
            self.delay.draw(rng, n))
        out, _ = run(callers, staff=c.staffing[k], p_retry=self.p)
        return out[:, OFFERED], out[:, ABANDONED]

    def days(self, rng: np.random.Generator, n_days: int):
        """``n_days`` days, cycling through the real staffing days."""
        c = self.centre
        k = np.resize(np.arange(c.staffing.shape[0]), n_days)
        off = np.empty((n_days, N_INTERVALS))
        ab = np.empty((n_days, N_INTERVALS))
        for i, kk in enumerate(k):
            off[i], ab[i] = self.day(rng, int(kk))
        return off, ab, c.staffing_dow[k]
