"""Three places to add what Erlang A misses.

``OutputHybrid`` keeps Erlang A as it is and adds a learned discrepancy to
its outputs (Kennedy and O'Hagan's formulation). It needs no idea of what is
missing, only residuals.

``EmpiricalDayDemand`` corrects an input instead: the Poisson demand model
has no day-to-day variation beyond the calendar, and the correction is to
resample each simulated day's level from the day levels seen in the data.

``GreyBox`` adds a structural slot for the mechanism the analyst suspects is
missing -- callers who hang up and call again -- and estimates its size from
the interval table by indirect inference: an auxiliary regression is run on
the real table and on tables simulated at candidate redial probabilities,
and the probability whose simulated coefficient matches the real one is
chosen. The slot's shape (the mean delay before a redial) is an assumption
the analyst brings; the table supplies only its size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gentradeoffs.evaluation import pooled
from gentradeoffs.mechanistic import (DemandModel, MechanisticGenerator,
                                      QueueModel, service_mean, theta_ratio)
from gentradeoffs.queue import OFFERED
from gentradeoffs.world import StaffingPolicy, World, rows

from .discrepancy import Discrepancy, apply
from .eta import REPS, TARGETS, eta_day

# --------------------------------------------------------------------------
# Output discrepancy


@dataclass
class OutputHybrid:
    """Erlang A's generator with a learned correction added to its outputs."""

    base: MechanisticGenerator
    delta: Discrepancy
    reps: int = REPS

    @property
    def name(self) -> str:
        return f"Erlang A + {self.delta.label}"

    def sample_days(self, rng: np.random.Generator, n_days: int,
                    multiplier: float = 1.0,
                    calendar: list[tuple[int, bool]] | None = None,
                    extra: int = 0) -> tuple[pd.DataFrame, dict]:
        frames, impossible = [], []
        for i in range(n_days):
            if calendar is None:
                dow = i % 5
                campaign = bool(rng.random() < self.base.demand.campaign_prob)
            else:
                dow, campaign = calendar[i]
            out, staff = self.base.day(rng, dow, campaign, multiplier, extra)
            frame = rows(out, staff, dow, campaign)
            counts = out[:, OFFERED].astype(np.int64)
            eta = pd.DataFrame(eta_day(self.base.queue, rng, counts, staff,
                                       self.reps),
                               columns=[f"eta_{t}" for t in TARGETS],
                               index=frame.index)
            delta = self.delta.predict(frame, eta)
            draw = pd.DataFrame({f"eta_{t}": frame[t].to_numpy()
                                 for t in TARGETS}, index=frame.index)
            fixed, info = apply(draw, delta, self.delta.scale,
                                keep_boundaries=True)
            impossible.append(info["share_impossible"])
            for t in TARGETS:
                frame[t] = np.where(frame[t].isna(), np.nan, fixed[t])
            frame.insert(0, "day", i)
            frames.append(frame)
        return (pd.concat(frames, ignore_index=True),
                {"share_impossible": float(np.mean(impossible))})

    def predict_given(self, rng: np.random.Generator,
                      table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """eta + delta for rows whose offered calls and agents are given."""
        from .eta import eta_table
        eta = eta_table(self.base.queue, rng, table, self.reps)
        return apply(eta, self.delta.predict(table, eta), self.delta.scale)


# --------------------------------------------------------------------------
# Structural slot for redials, sized by indirect inference


def auxiliary_coefficient(table: pd.DataFrame) -> float:
    """Within-day slope of surplus offered calls on last interval's
    abandonments.

    Offered calls are compared with a Poisson demand model fitted to the
    same table and rescaled to each day's total, which removes the day-level
    level shift. The surplus in interval t is then regressed, without an
    intercept, on the number of calls abandoned in interval t-1, demeaned
    within the day. Redials make the slope positive; it is an auxiliary
    statistic, not an estimate of anything by itself.
    """
    demand = DemandModel.fit(table)
    num = den = 0.0
    for _, d in table.groupby("day", sort=True):
        dow = int(d["dow"].iloc[0])
        camp = int(d["campaign"].iloc[0])
        expected = np.exp(demand.log_rate[dow, camp])
        offered = d["offered"].to_numpy(float)
        scale = offered.sum() / expected.sum()
        surplus = offered - scale * expected
        abandoned = np.nan_to_num(d["abandon_rate"].to_numpy(float)) * offered
        x = abandoned[:-1] - abandoned[:-1].mean()
        y = surplus[1:]
        num += float(x @ y)
        den += float(x @ x)
    return num / den


@dataclass
class RedialEstimate:
    p_hat: float
    beta_observed: float
    grid: np.ndarray
    beta_simulated: np.ndarray
    delay_assumed: float


@dataclass
class EmpiricalDayDemand(DemandModel):
    """The fitted Poisson demand model, with each simulated day's level
    drawn from the day levels observed in the training table."""

    factors: np.ndarray = field(default_factory=lambda: np.ones(1))

    @classmethod
    def from_table(cls, table: pd.DataFrame,
                   base: DemandModel | None = None) -> "EmpiricalDayDemand":
        base = base or DemandModel.fit(table)
        factors = day_factors(table, base)
        return cls(base.log_rate, base.campaign_prob, factors=factors)

    def counts(self, rng: np.random.Generator, dow: int, campaign: bool,
               multiplier: float = 1.0) -> np.ndarray:
        rate = np.exp(self.log_rate[dow, int(campaign)]) * multiplier
        return rng.poisson(rate * self.factors[rng.integers(
            self.factors.size)])


def day_factors(table: pd.DataFrame, demand: DemandModel) -> np.ndarray:
    """Each day's offered calls divided by what the demand model expects."""
    out = []
    for _, d in table.groupby("day", sort=True):
        expected = np.exp(demand.log_rate[int(d["dow"].iloc[0]),
                                          int(d["campaign"].iloc[0])])
        out.append(d["offered"].sum() / expected.sum())
    return np.array(out)


def day_factor_generator(table: pd.DataFrame, policy: StaffingPolicy,
                         world: World, p: float = 0.0,
                         delay: float = 600.0) -> MechanisticGenerator:
    """Erlang A (optionally with redials) on resampled day levels."""
    gen = redial_generator(table, p, delay, policy, world)
    demand = EmpiricalDayDemand.from_table(table)
    demand = demand.with_(log_rate=gen.demand.log_rate)
    queue = gen.queue.with_(name="Erlang A + resampled day levels"
                            + (" + estimated redials" if p > 0 else ""))
    return MechanisticGenerator(queue, demand, policy, world)


def _deflated(demand: DemandModel, p: float, abandon: float) -> DemandModel:
    return demand.with_(log_rate=demand.log_rate + np.log(1 - p * abandon))


def redial_generator(table: pd.DataFrame, p: float, delay: float,
                     policy: StaffingPolicy, world: World) -> MechanisticGenerator:
    """Erlang A with redials at probability ``p`` and mean delay ``delay``,
    calibrated on ``table`` and with fresh demand deflated so that offered
    calls still match it."""
    queue = QueueModel(f"Erlang A + redials (p={p:.2f})", service_mean(table),
                       patience="exp", theta=theta_ratio(table),
                       retry_prob=p, retry_delay_mean=delay)
    demand = _deflated(DemandModel.fit(table), p, pooled(table)["abandon"])
    return MechanisticGenerator(queue, demand, policy, world)


def estimate_redials(table: pd.DataFrame, rng_seed: int, delay: float,
                     policy: StaffingPolicy, world: World,
                     grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                     sim_days: int = 600) -> RedialEstimate:
    """Indirect inference for the redial probability.

    The auxiliary coefficient is computed on ``sim_days`` simulated days at
    each grid value, with the same random numbers at every value so that the
    simulated coefficient moves smoothly with p. The binding function --
    simulated coefficient against p -- is smoothed by a quadratic fitted
    through the grid, made monotone, and inverted at the observed
    coefficient; values beyond the ends of the grid go to the nearest end.
    """
    beta_obs = auxiliary_coefficient(table)
    sims = []
    for p in grid:
        gen = redial_generator(table, p, delay, policy, world)
        sim = gen.sample_days(np.random.default_rng(rng_seed), sim_days)
        sims.append(auxiliary_coefficient(sim))
    sims = np.array(sims)
    fine = np.linspace(grid[0], grid[-1], 501)
    binding = np.maximum.accumulate(np.polyval(np.polyfit(grid, sims, 2),
                                               fine))
    p_hat = float(np.interp(beta_obs, binding, fine))
    return RedialEstimate(p_hat, beta_obs, np.array(grid), sims, delay)


@dataclass
class GreyBox:
    """Erlang A with a redial slot sized from the table."""

    estimate: RedialEstimate
    generator: MechanisticGenerator = field(repr=False)

    @property
    def name(self) -> str:
        return "Erlang A + estimated redials"

    @classmethod
    def fit(cls, table: pd.DataFrame, rng_seed: int, delay: float,
            policy: StaffingPolicy, world: World, **kw) -> "GreyBox":
        est = estimate_redials(table, rng_seed, delay, policy, world, **kw)
        return cls(est, redial_generator(table, est.p_hat, delay, policy,
                                         world))

    def sample_days(self, rng, n_days, multiplier=1.0, calendar=None,
                    extra: int = 0):
        frames = []
        for i in range(n_days):
            if calendar is None:
                dow = i % 5
                campaign = bool(rng.random()
                                < self.generator.demand.campaign_prob)
            else:
                dow, campaign = calendar[i]
            out, staff = self.generator.day(rng, dow, campaign, multiplier,
                                            extra)
            frame = rows(out, staff, dow, campaign)
            frame.insert(0, "day", i)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True), {}

