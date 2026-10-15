"""The synthetic depot-leasing instance and its weekly demand process.

Ten candidate depots, thirty customer zones in three regions. Weekly demand in
a zone is its mean multiplied by three independent mean-one lognormal shocks:
one common to all zones, one shared within a region and one specific to the
zone, so E[demand] equals the zone mean exactly.

`build_instance` consumes the generator in a fixed order and returns it, so
the history drawn afterwards is part of the same seeded stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEED = 20261015

N_SITES, N_ZONES, N_REGIONS = 10, 30, 3
CENTRES = np.array([[25.0, 70.0], [72.0, 75.0], [55.0, 25.0]])  # km
PENALTY = 140.0                                    # EUR per pallet outsourced
SIG_COMMON, SIG_REGION, SIG_ZONE = 0.15, 0.25, 0.20


@dataclass(frozen=True)
class Instance:
    region: np.ndarray      # (zones,) region index of each zone
    zones_xy: np.ndarray    # (zones, 2) km
    sites_xy: np.ndarray    # (sites, 2) km
    mu: np.ndarray          # (zones,) mean pallets per week
    cap: np.ndarray         # (sites,) pallets per week
    fixed: np.ndarray       # (sites,) lease, EUR per week
    cost: np.ndarray        # (sites, zones) EUR per pallet
    penalty: float = PENALTY

    @property
    def n_sites(self) -> int:
        return len(self.cap)

    @property
    def n_zones(self) -> int:
        return len(self.mu)


def build_instance(seed: int = SEED) -> tuple[Instance, np.random.Generator]:
    """Return the instance and the generator, positioned after the instance draws."""
    rng = np.random.default_rng(seed)
    region = np.repeat(np.arange(N_REGIONS), N_ZONES // N_REGIONS)
    zones_xy = np.clip(CENTRES[region] + rng.normal(0, 12, (N_ZONES, 2)), 0, 100)
    sites_xy = rng.uniform(5, 95, (N_SITES, 2))
    mu = rng.uniform(20, 55, N_ZONES)
    cap = rng.choice(np.arange(300, 501, 25), N_SITES).astype(float)
    fixed = np.round(1500 + 9.0 * cap + rng.normal(0, 300, N_SITES), -1)
    dist = np.linalg.norm(sites_xy[:, None, :] - zones_xy[None, :, :], axis=2)
    cost = 6.0 + 0.9 * dist
    inst = Instance(region=region, zones_xy=zones_xy, sites_xy=sites_xy, mu=mu,
                    cap=cap, fixed=fixed, cost=cost)
    return inst, rng


def _lognormal_mean_one(sigma: float, size, gen: np.random.Generator) -> np.ndarray:
    return np.exp(gen.normal(-0.5 * sigma**2, sigma, size))


def draw_weeks(inst: Instance, n: int, gen: np.random.Generator) -> np.ndarray:
    """Draw n independent weeks of demand, shape (n, zones)."""
    common = _lognormal_mean_one(SIG_COMMON, (n, 1), gen)
    reg = _lognormal_mean_one(SIG_REGION, (n, N_REGIONS), gen)[:, inst.region]
    zone = _lognormal_mean_one(SIG_ZONE, (n, inst.n_zones), gen)
    return inst.mu[None, :] * common * reg * zone
