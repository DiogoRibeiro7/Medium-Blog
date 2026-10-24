"""The analyst's mechanistic generators and how they are calibrated.

A ``QueueModel`` is a queue the analyst believes in: Poisson arrivals, a
service-time law, a patience law and, optionally, redials. The two textbook
members are Erlang C (exponential service, nobody hangs up) and Erlang A
(exponential service, exponential patience with rate ``theta``). The other
variants exist to take the misspecification apart one assumption at a time.

A ``MechanisticGenerator`` adds the analyst's demand model and the company's
own staffing rule, and produces whole interval tables.

Calibration uses only the interval table, never call-level records.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import PoissonRegressor

from .queue import (ABANDONED, N_INTERVALS, OFFERED, TOTAL_WAIT, WITHIN,
                    Callers, run, uniform_arrivals)
from .world import COLUMNS, StaffingPolicy, World, rows


@dataclass(frozen=True)
class QueueModel:
    """A queue the analyst can simulate.

    ``patience`` is ``"none"`` (Erlang C), ``"exp"`` (rate ``theta``),
    ``"lognormal"`` (median ``patience_median``, log-sd ``patience_sigma``)
    or ``"true"`` (the world's own mixture; used only in the ablation).
    ``service`` is ``"exp"`` or ``"lognormal"`` with ``service_cv``.
    ``retry_prob`` and ``retry_delay_mean`` add redials.
    """

    name: str
    service_mean: float
    service: str = "exp"
    service_cv: float = 1.0
    patience: str = "exp"
    theta: float = 0.0
    patience_median: float = 200.0
    patience_sigma: float = 1.0
    retry_prob: float = 0.0
    retry_delay_mean: float = 600.0
    world: World | None = None

    def draw_service(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.service == "exp":
            return rng.exponential(self.service_mean, size=n)
        s2 = np.log1p(self.service_cv ** 2)
        return rng.lognormal(np.log(self.service_mean) - s2 / 2, np.sqrt(s2),
                             size=n)

    def draw_patience(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.patience == "none" or (self.patience == "exp"
                                       and self.theta <= 0):
            return np.full(n, np.inf)
        if self.patience == "exp":
            return rng.exponential(1.0 / self.theta, size=n)
        if self.patience == "lognormal":
            return rng.lognormal(np.log(self.patience_median),
                                 self.patience_sigma, size=n)
        if self.patience == "true":
            return (self.world or World()).draw_patience(rng, n)
        raise ValueError(f"unknown patience law {self.patience!r}")

    def callers(self, rng: np.random.Generator,
                counts: np.ndarray) -> Callers:
        arrivals = uniform_arrivals(rng, counts)
        n = arrivals.size
        m = n if self.retry_prob > 0 else 0
        return Callers(arrivals, self.draw_service(rng, n),
                       self.draw_patience(rng, n), rng.random(n),
                       rng.exponential(self.retry_delay_mean, n),
                       self.draw_service(rng, m), self.draw_patience(rng, m),
                       rng.random(m), rng.exponential(self.retry_delay_mean, m))

    def run_given(self, rng: np.random.Generator, counts: np.ndarray,
                  staff: np.ndarray, extra: int = 0) -> np.ndarray:
        """One day given the offered calls per interval and the staffing."""
        out, _ = run(self.callers(rng, counts), staff=staff,
                     p_retry=self.retry_prob, extra=extra)
        return out

    def with_(self, **changes) -> "QueueModel":
        return replace(self, **changes)


# --------------------------------------------------------------------------
# Calibration from the interval table


def answered(table: pd.DataFrame) -> np.ndarray:
    return table["offered"].to_numpy() * (1 - table["abandon_rate"].to_numpy())


def service_mean(table: pd.DataFrame) -> float:
    """Handle time averaged over answered calls."""
    w = answered(table)
    return float(np.sum(table["aht"].to_numpy() * w) / np.sum(w))


def service_cv(table: pd.DataFrame) -> float:
    """Service-time CV from the spread of interval averages.

    An interval's AHT is a mean of ``m`` handle times, so its variance is
    sigma^2 / m. Weighting squared deviations by ``m`` estimates sigma^2.
    """
    w = answered(table)
    mean = service_mean(table)
    s2 = np.mean((table["aht"].to_numpy() - mean) ** 2 * w)
    return float(np.sqrt(s2) / mean)


def theta_ratio(table: pd.DataFrame) -> float:
    """Abandonments divided by total time spent waiting.

    Under exponential patience the hazard of hanging up is constant and this
    ratio is its maximum-likelihood estimate. Under any other patience law
    it estimates something else.
    """
    off = table["offered"].to_numpy()
    ab = np.sum(table["abandon_rate"].to_numpy() * off)
    wait = np.sum(table["awt"].to_numpy() * off)
    return float(ab / wait)


def table_moments(table: pd.DataFrame) -> dict[str, float]:
    off = table["offered"].to_numpy()
    tot = off.sum()
    return {
        "abandon": float(np.sum(table["abandon_rate"].to_numpy() * off) / tot),
        "wait": float(np.sum(table["awt"].to_numpy() * off) / tot),
        "sl": float(np.sum(table["sl"].to_numpy() * off) / tot),
    }


def days_inputs(table: pd.DataFrame):
    """(offered per interval, agents per interval) for each day."""
    for _, d in table.groupby("day", sort=True):
        yield (d["offered"].to_numpy().astype(np.int64),
               d["agents"].to_numpy().astype(np.int64))


def theta_smm(table: pd.DataFrame, rng: np.random.Generator, moment: str,
              service: float, sims: int = 1, lo: float = 1e-4,
              hi: float = 0.1, xtol: float = 1e-3) -> float:
    """Simulated method of moments for the exponential patience rate.

    Replays the observed days (their offered calls and staffing) ``sims``
    times through Erlang A and finds the theta whose simulated ``moment`` -- the overall
    abandonment share, or the mean time waited per offered call -- equals
    the observed one. Common random numbers keep the simulated moment a
    monotone step function of theta, and Brent's method on log theta finds
    the crossing to within ``xtol``.
    """
    target = table_moments(table)[moment]
    replay = []
    for counts, staff in days_inputs(table):
        for _ in range(sims):
            arrivals = uniform_arrivals(rng, counts)
            n = arrivals.size
            replay.append((arrivals, rng.exponential(service, n),
                           rng.exponential(1.0, n), staff))
    empty = np.zeros(0)

    def simulated(theta: float) -> float:
        num = den = 0.0
        for arrivals, serv, base, staff in replay:
            n = arrivals.size
            callers = Callers(arrivals, serv, base / theta, np.ones(n),
                              np.zeros(n), empty, empty, empty, empty)
            out, _ = run(callers, staff=staff)
            den += out[:, OFFERED].sum()
            num += (out[:, ABANDONED].sum() if moment == "abandon"
                    else out[:, TOTAL_WAIT].sum())
        return num / den

    root = brentq(lambda x: simulated(np.exp(x)) - target, np.log(lo),
                  np.log(hi), xtol=xtol)
    return float(np.exp(root))


# --------------------------------------------------------------------------
# Demand model and the joint generator


@dataclass
class DemandModel:
    """Poisson rates multiplicative in interval, day of week and campaign.

    ``day_cv`` and ``interval_sigma`` add the gamma day shock and log-normal
    interval shock of the true process; both are zero in the fitted model
    and are switched on only in the ablation.
    """

    log_rate: np.ndarray            # (5, 2, 24): dow, campaign, interval
    campaign_prob: float
    day_cv: float = 0.0
    interval_sigma: float = 0.0

    @staticmethod
    def _design(interval: np.ndarray, dow: np.ndarray,
                campaign: np.ndarray) -> np.ndarray:
        X = np.zeros((interval.size, (N_INTERVALS - 1) + 4 + 1))
        idx = np.arange(interval.size)
        m = interval > 0
        X[idx[m], interval[m] - 1] = 1.0
        m = dow > 0
        X[idx[m], N_INTERVALS - 1 + dow[m] - 1] = 1.0
        X[:, -1] = campaign
        return X

    @classmethod
    def fit(cls, table: pd.DataFrame) -> "DemandModel":
        X = cls._design(table["interval"].to_numpy(int),
                        table["dow"].to_numpy(int),
                        table["campaign"].to_numpy(float))
        reg = PoissonRegressor(alpha=0.0, max_iter=1000, tol=1e-10)
        reg.fit(X, table["offered"].to_numpy())
        log_rate = np.empty((5, 2, N_INTERVALS))
        interval = np.arange(N_INTERVALS)
        for dow in range(5):
            for camp in (0, 1):
                g = cls._design(interval, np.full(N_INTERVALS, dow),
                                np.full(N_INTERVALS, float(camp)))
                log_rate[dow, camp] = np.log(reg.predict(g))
        days = table.groupby("day")["campaign"].first()
        return cls(log_rate, float(days.mean()))

    def counts(self, rng: np.random.Generator, dow: int, campaign: bool,
               multiplier: float = 1.0) -> np.ndarray:
        rate = np.exp(self.log_rate[dow, int(campaign)]) * multiplier
        if self.day_cv > 0:
            k = 1.0 / self.day_cv ** 2
            rate = rate * rng.gamma(k, 1.0 / k)
        if self.interval_sigma > 0:
            s = self.interval_sigma
            rate = rate * rng.lognormal(-s * s / 2, s, size=rate.size)
        return rng.poisson(rate)

    def with_(self, **changes) -> "DemandModel":
        return replace(self, **changes)


@dataclass
class MechanisticGenerator:
    """Calendar + demand model + the company's staffing rule + a queue.

    ``planning_world`` supplies the inputs of the planners' forecast, which
    is the company's own and therefore known to the analyst.
    """

    queue: QueueModel
    demand: DemandModel
    policy: StaffingPolicy
    planning_world: World

    @property
    def name(self) -> str:
        return self.queue.name

    def day(self, rng: np.random.Generator, dow: int, campaign: bool,
            multiplier: float = 1.0, extra: int = 0):
        forecast = self.policy.forecast(rng, dow, campaign,
                                        self.planning_world)
        attend = self.policy.attendance_draws(rng)
        counts = self.demand.counts(rng, dow, campaign, multiplier)
        out, staff = run(self.queue.callers(rng, counts),
                         p_retry=self.queue.retry_prob,
                         policy=self.policy.vector(), forecast=forecast,
                         attend=attend, extra=extra)
        return out, staff

    def sample_days(self, rng: np.random.Generator, n_days: int,
                    multiplier: float = 1.0,
                    calendar: list[tuple[int, bool]] | None = None
                    ) -> pd.DataFrame:
        frames = []
        for i in range(n_days):
            if calendar is None:
                dow = i % 5
                campaign = bool(rng.random() < self.demand.campaign_prob)
            else:
                dow, campaign = calendar[i]
            out, staff = self.day(rng, dow, campaign, multiplier)
            frame = rows(out, staff, dow, campaign)
            frame.insert(0, "day", i)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def sample(self, rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
        n_days = int(np.ceil(n_rows / N_INTERVALS))
        return self.sample_days(rng, n_days)[COLUMNS].iloc[:n_rows]


def erlang_models(table: pd.DataFrame) -> dict[str, QueueModel]:
    """Erlang C and Erlang A calibrated on a training table."""
    mean = service_mean(table)
    return {
        "Erlang C": QueueModel("Erlang C", mean, patience="none"),
        "Erlang A": QueueModel("Erlang A", mean, patience="exp",
                               theta=theta_ratio(table)),
    }


def predict_given(model: QueueModel, rng: np.random.Generator,
                  counts: np.ndarray, staff: np.ndarray,
                  reps: int = 4) -> np.ndarray:
    """Per-interval totals summed over ``reps`` replays of one day."""
    tot = np.zeros((N_INTERVALS, 6))
    for _ in range(reps):
        tot += model.run_given(rng, counts, staff)
    return tot


def rates(tot: np.ndarray) -> dict[str, float]:
    s = tot.sum(axis=0) if tot.ndim == 2 else tot
    return {"sl": s[WITHIN] / s[OFFERED], "abandon": s[ABANDONED] / s[OFFERED]}
