"""The contact centre that plays the part of reality.

It is richer than any model the analyst writes down later:

* fresh calls are Poisson within an interval, but the rate carries a
  day-level shock and an interval-level shock on top of the intraday
  profile, the day-of-week effect and announced campaign days;
* service times are log-normal with coefficient of variation 1;
* patience is a mixture: a fifth of callers hang up quickly (exponential,
  mean 20 s) and the rest are log-normal with median 200 s;
* six in ten callers who hang up call again, on average ten minutes later,
  and the redial is counted as a new offered call;
* staffing follows a square-root rule on the planners' forecast, is revised
  during the day when offered calls run above or below forecast, and loses
  agents to absence.

The analyst sees only the interval table this produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .queue import (MAX_SLOTS, N_INTERVALS, Callers, run, summarise,
                    uniform_arrivals)

COLUMNS = ["dow", "interval", "campaign", "offered", "agents",
           "abandon_rate", "sl", "awt", "aht"]


def base_profile() -> np.ndarray:
    """Fresh calls per hour by half-hour interval, 08:00 to 20:00."""
    hours = 8.0 + 0.5 * np.arange(N_INTERVALS) + 0.25
    return (150.0
            + 270.0 * np.exp(-((hours - 10.5) / 1.6) ** 2)
            + 210.0 * np.exp(-((hours - 14.75) / 1.9) ** 2))


@dataclass(frozen=True)
class World:
    """Parameters of the true data-generating process."""

    dow_factor: tuple[float, ...] = (1.18, 1.05, 1.00, 0.97, 0.88)
    campaign_prob: float = 0.20
    campaign_factor: float = 1.30
    day_cv: float = 0.10
    interval_sigma: float = 0.05
    service_mean: float = 300.0
    service_cv: float = 1.0
    fast_share: float = 0.20
    fast_mean: float = 20.0
    patience_median: float = 200.0
    patience_sigma: float = 0.8
    retry_prob: float = 0.6
    retry_delay_mean: float = 600.0

    def mean_patience(self) -> float:
        slow = self.patience_median * np.exp(self.patience_sigma ** 2 / 2)
        return self.fast_share * self.fast_mean + (1 - self.fast_share) * slow

    def draw_service(self, rng: np.random.Generator, n: int) -> np.ndarray:
        s2 = np.log1p(self.service_cv ** 2)
        mu = np.log(self.service_mean) - s2 / 2
        return rng.lognormal(mu, np.sqrt(s2), size=n)

    def draw_patience(self, rng: np.random.Generator, n: int) -> np.ndarray:
        fast = rng.random(n) < self.fast_share
        out = rng.lognormal(np.log(self.patience_median), self.patience_sigma,
                            size=n)
        out[fast] = rng.exponential(self.fast_mean, size=int(fast.sum()))
        return out

    def callers(self, rng: np.random.Generator,
                counts: np.ndarray) -> Callers:
        """Fresh callers for the given counts, plus a pool for redials."""
        arrivals = uniform_arrivals(rng, counts)
        n = arrivals.size
        return Callers(arrivals, self.draw_service(rng, n),
                       self.draw_patience(rng, n), rng.random(n),
                       rng.exponential(self.retry_delay_mean, n),
                       self.draw_service(rng, n), self.draw_patience(rng, n),
                       rng.random(n),
                       rng.exponential(self.retry_delay_mean, n))


@dataclass(frozen=True)
class StaffingPolicy:
    """How the planners staff the centre. This is the company's own rule."""

    beta: float = 0.9               # square-root staffing safety factor
    planned_aht: float = 300.0      # handle time used in planning (s)
    forecast_sigma: float = 0.06    # log-sd of the planners' forecast error
    reactive_from: int = 4          # first interval revised intraday
    reactive_elasticity: float = 0.7
    max_add: int = 5
    max_cut: int = 3
    attendance: float = 0.95

    def vector(self) -> np.ndarray:
        return np.array([self.beta, self.planned_aht, self.reactive_from,
                         self.reactive_elasticity, self.max_add, self.max_cut,
                         self.attendance], dtype=float)

    def forecast(self, rng: np.random.Generator, dow: int, campaign: bool,
                 world: World) -> np.ndarray:
        """Planned calls per interval: knows the calendar, not the shocks."""
        lam = base_profile() / 2.0 * world.dow_factor[dow]
        if campaign:
            lam = lam * world.campaign_factor
        noise = rng.lognormal(-self.forecast_sigma ** 2 / 2,
                              self.forecast_sigma, size=N_INTERVALS)
        return lam * noise

    @staticmethod
    def attendance_draws(rng: np.random.Generator) -> np.ndarray:
        return rng.random((N_INTERVALS, MAX_SLOTS))


@dataclass
class Day:
    """One day: its calendar, callers and the planners' inputs."""

    dow: int
    campaign: bool
    counts: np.ndarray            # fresh calls per interval
    callers: Callers
    forecast: np.ndarray
    attend: np.ndarray
    extra: dict = field(default_factory=dict)


def draw_day(rng: np.random.Generator, dow: int, campaign: bool,
             world: World, policy: StaffingPolicy,
             multiplier: float = 1.0) -> Day:
    """Draw the callers of one day and the planners' inputs.

    ``multiplier`` scales fresh demand without the planners being told, as
    an unannounced surge would.
    """
    forecast = policy.forecast(rng, dow, campaign, world)
    attend = policy.attendance_draws(rng)
    k = 1.0 / world.day_cv ** 2
    shock = rng.gamma(k, 1.0 / k)
    eps = rng.lognormal(-world.interval_sigma ** 2 / 2, world.interval_sigma,
                        size=N_INTERVALS)
    rate = base_profile() / 2.0 * world.dow_factor[dow] * shock * eps
    if campaign:
        rate = rate * world.campaign_factor
    counts = rng.poisson(rate * multiplier)
    return Day(dow, campaign, counts, world.callers(rng, counts), forecast,
               attend, {"shock": shock})


def play(day: Day, world: World, policy: StaffingPolicy,
         extra: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run a day under the planners' rule, with ``extra`` agents added."""
    return run(day.callers, p_retry=world.retry_prob, policy=policy.vector(),
               forecast=day.forecast, attend=day.attend, extra=extra)


def rows(out: np.ndarray, staff: np.ndarray, dow: int,
         campaign: bool) -> pd.DataFrame:
    frame = pd.DataFrame(summarise(out, staff))
    frame.insert(0, "campaign", int(campaign))
    frame.insert(0, "interval", np.arange(N_INTERVALS))
    frame.insert(0, "dow", dow)
    return frame[COLUMNS]


def calendar(rng: np.random.Generator, n_days: int,
             world: World) -> list[tuple[int, bool]]:
    """Consecutive weekdays starting on a Monday; campaigns at random."""
    return [(d % 5, bool(rng.random() < world.campaign_prob))
            for d in range(n_days)]


def simulate_table(rng: np.random.Generator, n_days: int,
                   world: World | None = None,
                   policy: StaffingPolicy | None = None,
                   multiplier: float = 1.0, extra: int = 0,
                   keep_days: bool = False):
    """Simulate ``n_days`` of operation and return the interval table.

    One row per (day, interval) plus a ``day`` column. With ``keep_days``
    the ``Day`` objects come back too, so the same callers can be replayed
    under another staffing plan.
    """
    world = world or World()
    policy = policy or StaffingPolicy()
    days, frames = [], []
    for i, (dow, campaign) in enumerate(calendar(rng, n_days, world)):
        day = draw_day(rng, dow, campaign, world, policy, multiplier)
        out, staff = play(day, world, policy, extra)
        frame = rows(out, staff, dow, campaign)
        frame.insert(0, "day", i)
        frames.append(frame)
        if keep_days:
            days.append(day)
    table = pd.concat(frames, ignore_index=True)
    return (table, days) if keep_days else table

