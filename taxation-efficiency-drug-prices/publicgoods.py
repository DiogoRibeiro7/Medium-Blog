"""Shared tools for the taxation-efficiency-drug-prices series: a linear public-goods economy.

Everything here is a model. The behavioural rule follows the explanation of declining contributions
given by Fischbacher and Gaechter (2010): people are conditional cooperators who, on average, give
less than they believe the others give, and their beliefs follow what the others gave last round.
The free parameters are calibrated in part1/premises_numbers.py to published summary figures.
Nothing in this module reads data.

Units. Each person has an endowment E per round. A token contributed to the public good returns
MPCR tokens to every member of a group of n people, so one token creates G = n * MPCR tokens of
total value (the marginal group return). With 1 < G, full contribution is efficient; with MPCR < 1,
keeping the token is privately better whatever the others do.

Welfare per person and round, in units of E:
    voluntary:       1 + (G - 1) s                 s = mean contribution / E
    peer sanctions:  1 + (G - 1) s - (cost of punishing + loss from being punished) / E
    compulsion:      1 - lam + G (1 - w)           the whole endowment is raised by a tax with
                                                    marginal cost of public funds lam, and a
                                                    share w of the revenue does not reach the good
Compulsion beats voluntary provision exactly when s < s_star(G, lam, w) = (G (1 - w) - lam) / (G - 1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

ENDOWMENT = 20.0


@dataclass(frozen=True)
class Behaviour:
    """Parameters of the contribution rule.

    Person i contributes beta_i times his belief about the others' average contribution, plus a
    decision error, clipped to [0, E]. After each round the belief moves towards what the others
    actually gave: b <- theta * b + (1 - theta) * observed.

    beta_mean    mean slope of the conditional contribution (1 = matches the others exactly)
    beta_conc    concentration of the Beta distribution of slopes (larger = less heterogeneity)
    belief0      mean first-round belief about the others' average, share of E
    belief0_sd   spread of first-round beliefs across people, share of E
    theta        weight on the previous belief when beliefs are updated
    noise_sd     standard deviation of the decision error, share of E
    """

    beta_mean: float = 0.9
    beta_conc: float = 4.0
    belief0: float = 0.6
    belief0_sd: float = 0.2
    theta: float = 0.0
    noise_sd: float = 0.1


@dataclass(frozen=True)
class Sanctions:
    """Peer punishment, applied after contributions are seen.

    Punishers are the people whose slope is at least `punisher_beta`. Each punisher assigns to each
    other member gamma points per unit of that member's shortfall below the mean of the rest of the
    group (shortfall measured as a share of E). A point costs the punisher `cost` and the target
    `impact`. Prosocial points received raise the target's contribution floor by eta (share of E)
    per point; the floor relaxes by `floor_decay` each round. A share `antisocial` of punishers
    also punishes members who gave more than they did, with the same intensity; antisocial points
    received lower the target's slope and floor by a factor (1 - eta * points).
    """

    gamma: float = 3.0
    eta: float = 0.1
    floor_decay: float = 0.9
    cost: float = 1.0
    impact: float = 3.0
    antisocial: float = 0.0
    cap: float = 10.0
    punisher_beta: float = 0.5


@dataclass
class Path:
    """Round-by-round averages over all simulated groups."""

    contribution: np.ndarray            # mean contribution share of E, by round
    full_share: np.ndarray              # share of people contributing at least 97.5 % of E
    sanction_cost: np.ndarray           # punishment cost per person (giver + target), share of E
    welfare: np.ndarray                 # welfare per person, units of E


@dataclass(frozen=True)
class Draws:
    """Common random numbers, so that calibration objectives are smooth in the parameters."""

    u_beta: np.ndarray        # (groups, n)
    z_belief: np.ndarray      # (groups, n)
    z_noise: np.ndarray       # (rounds, groups, n)
    u_anti: np.ndarray        # (groups, n)

    @property
    def shape(self) -> tuple[int, int, int]:
        rounds, groups, n = self.z_noise.shape
        return rounds, groups, n


def make_draws(seed: int, groups: int, n: int, rounds: int) -> Draws:
    rng = np.random.default_rng(seed)
    return Draws(
        u_beta=rng.random((groups, n)),
        z_belief=rng.standard_normal((groups, n)),
        z_noise=rng.standard_normal((rounds, groups, n)),
        u_anti=rng.random((groups, n)),
    )


def slopes(u: np.ndarray, mean: float, conc: float) -> np.ndarray:
    """Beta-distributed slopes with the given mean and concentration, by inverse CDF."""
    mean = float(np.clip(mean, 1e-4, 1 - 1e-4))
    return stats.beta.ppf(u, mean * conc, (1 - mean) * conc)


def simulate(
    behaviour: Behaviour,
    draws: Draws,
    mpcr: float,
    rounds: int,
    sanctions: Sanctions | None = None,
) -> Path:
    """Simulate many groups playing the linear public-goods game for a number of rounds."""
    max_rounds, groups, n = draws.shape
    if rounds > max_rounds:
        raise ValueError("not enough random draws for this many rounds")
    E = ENDOWMENT
    beta = slopes(draws.u_beta, behaviour.beta_mean, behaviour.beta_conc)
    belief = np.clip(behaviour.belief0 + behaviour.belief0_sd * draws.z_belief, 0.0, 1.0)  # share
    floor = np.zeros((groups, n))
    damp = np.ones((groups, n))
    idx = np.arange(n)
    if sanctions is not None:
        punisher = beta >= sanctions.punisher_beta
        antisocial = punisher & (draws.u_anti < sanctions.antisocial)

    contribution = np.empty(rounds)
    full_share = np.empty(rounds)
    sanction_cost = np.zeros(rounds)
    welfare = np.empty(rounds)
    for t in range(rounds):
        share = beta * damp * belief + behaviour.noise_sd * draws.z_noise[t]
        share = np.clip(np.maximum(share, floor), 0.0, 1.0)
        c = E * share

        total = c.sum(axis=1, keepdims=True)
        others_mean = (total - c) / (n - 1)
        payoff = E - c + mpcr * total

        if sanctions is not None:
            short = np.maximum(others_mean - c, 0.0) / E                       # by target k
            pro = sanctions.gamma * punisher[:, :, None] * short[:, None, :]    # [g, j, k]
            above = np.maximum(c[:, None, :] - c[:, :, None], 0.0) / E          # k gave more than j
            anti = sanctions.gamma * antisocial[:, :, None] * above
            pro[:, idx, idx] = 0.0
            anti[:, idx, idx] = 0.0
            total_pts = pro + anti
            scale = np.where(total_pts > sanctions.cap, sanctions.cap / np.maximum(total_pts, 1e-12), 1.0)
            pro *= scale
            anti *= scale
            given = (pro + anti).sum(axis=2)
            pro_recv = pro.sum(axis=1)
            anti_recv = anti.sum(axis=1)
            received = pro_recv + anti_recv
            payoff = payoff - sanctions.cost * given - sanctions.impact * received
            sanction_cost[t] = (sanctions.cost * given + sanctions.impact * received).mean() / E
            hurt = np.clip(1.0 - sanctions.eta * anti_recv, 0.0, 1.0)
            damp = damp * hurt
            floor = np.clip((sanctions.floor_decay * floor + sanctions.eta * pro_recv) * hurt, 0.0, 1.0)

        contribution[t] = share.mean()
        full_share[t] = (share >= 0.975).mean()
        welfare[t] = payoff.mean() / E
        belief = behaviour.theta * belief + (1 - behaviour.theta) * others_mean / E

    return Path(contribution, full_share, sanction_cost, welfare)


def welfare_voluntary(s, G):
    return 1.0 + (np.asarray(G, dtype=float) - 1.0) * np.asarray(s, dtype=float)


def welfare_compulsory(G, lam, w):
    return 1.0 - np.asarray(lam, dtype=float) + np.asarray(G, dtype=float) * (1.0 - np.asarray(w, dtype=float))


def s_star(G, lam, w):
    """Voluntary provision share below which compulsion yields more welfare."""
    G = np.asarray(G, dtype=float)
    return (G * (1.0 - np.asarray(w, dtype=float)) - np.asarray(lam, dtype=float)) / (G - 1.0)


def g_star(s, lam, w):
    """Marginal group return above which compulsion yields more welfare, given voluntary share s.

    Solves s = s_star(G, lam, w) for G. Returns inf when compulsion never wins (s >= 1 - w).
    """
    s = np.asarray(s, dtype=float)
    lam = np.asarray(lam, dtype=float)
    w = np.asarray(w, dtype=float)
    denom = (1.0 - w) - s
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (lam - s) / denom
    return np.where(denom > 0, np.maximum(g, 1.0), np.inf)


def equivalent_share(welfare, G):
    """Voluntary share that would give the same welfare per person and round."""
    return (np.asarray(welfare, dtype=float) - 1.0) / (G - 1.0)


__all__ = [
    "ENDOWMENT",
    "Behaviour",
    "Sanctions",
    "Path",
    "Draws",
    "make_draws",
    "slopes",
    "simulate",
    "welfare_voluntary",
    "welfare_compulsory",
    "s_star",
    "g_star",
    "equivalent_share",
    "replace",
]
