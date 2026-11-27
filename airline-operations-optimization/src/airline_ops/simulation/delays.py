"""Primary delays and the expected-propagation penalty used by the robust planner.

Primary delays are the delays a flight would have if every resource it waits for were
on time: ground handling, boarding, technical and en-route effects. Reactionary
(knock-on) delays are not drawn; the simulator produces them by propagating primary
delays through aircraft, crews and slots.

The model: with probability ``p`` a flight gets a log-normal departure delay whose
parameters give a mean primary delay per flight of ``mean_primary_min``; the actual
block time is the scheduled block times ``1 + bias + sd * Z`` with standard normal
``Z``. The mean is anchored to EUROCONTROL's CODA figures (all-causes minus
reactionary minutes per flight); ``p`` and ``sigma`` are calibrated in the experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from airline_ops.config import DelayConfig


@dataclass(frozen=True, slots=True)
class DelayModel:
    """Parameters of the primary delay distribution."""

    mean_primary_min: float
    p_delayed: float
    sigma: float
    block_noise_sd: float
    block_bias: float

    @classmethod
    def from_config(cls, cfg: DelayConfig, *, mean_override: float | None = None) -> DelayModel:
        """Build from configuration, optionally replacing the mean (delay-climate sensitivity)."""
        return cls(
            mean_primary_min=cfg.mean_primary_min if mean_override is None else mean_override,
            p_delayed=cfg.p_delayed,
            sigma=cfg.sigma,
            block_noise_sd=cfg.block_noise_sd,
            block_bias=cfg.block_bias,
        )

    @property
    def mu(self) -> float:
        """Log-scale location of the positive part, chosen to hit the target mean."""
        mean_positive = self.mean_primary_min / self.p_delayed
        return math.log(mean_positive) - 0.5 * self.sigma**2

    def departure_delays(
        self, rng: np.random.Generator, n: int, scale: NDArray[np.float64] | float = 1.0
    ) -> NDArray[np.float64]:
        """Draw ``n`` primary departure delays (minutes), multiplied by ``scale``."""
        delayed = rng.random(n) < self.p_delayed
        size = rng.lognormal(self.mu, self.sigma, n)
        out: NDArray[np.float64] = np.where(delayed, size, 0.0) * scale
        return out

    def block_factors(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        """Draw ``n`` multiplicative factors applied to scheduled block times."""
        z = rng.standard_normal(n)
        out: NDArray[np.float64] = np.clip(1.0 + self.block_bias + self.block_noise_sd * z, 0.85, 1.4)
        return out


class PropagationPenalty:
    """Expected delay that a connection with slack ``s`` passes on: ``E[max(0, A - s)]``.

    ``A`` is the primary arrival delay of the incoming flight. The expectation is taken
    over a fixed Monte Carlo sample drawn with a planning seed that is never used for
    evaluation, so the robust planner does not see the scenarios it is scored on.
    """

    def __init__(self, model: DelayModel, samples: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._dep = model.departure_delays(rng, samples)
        self._factor = model.block_factors(rng, samples) - 1.0
        self._cache: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    def _sorted(self, block: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if block not in self._cache:
            a = np.sort(self._dep + block * self._factor)
            tail_sums = np.concatenate([np.cumsum(a[::-1])[::-1], [0.0]])
            self._cache[block] = (a, tail_sums)
        return self._cache[block]

    def expected_excess(self, block: int, slack: float) -> float:
        """``E[max(0, A - slack)]`` in minutes for an incoming flight of the given block time."""
        a, tail_sums = self._sorted(block)
        k = int(np.searchsorted(a, slack, side="right"))
        count = a.size - k
        return float((tail_sums[k] - slack * count) / a.size)


class EmpiricalPenalty:
    """The same expectation, from per-flight samples of arrival delay (a learned predictor).

    ``samples[f]`` holds observed arrival delays of flight ``f`` in a training history.
    With ``point=True`` the predictor keeps only the mean and the penalty becomes
    ``max(0, mean - slack)``, which is what a point forecast can say about a connection.
    """

    def __init__(self, samples: list[NDArray[np.float64]], *, point: bool = False) -> None:
        self.point = point
        self._sorted = [np.sort(s) for s in samples]
        self._tails = [np.concatenate([np.cumsum(s[::-1])[::-1], [0.0]]) for s in self._sorted]
        self._means = [float(s.mean()) if s.size else 0.0 for s in samples]

    def expected_excess_flight(self, flight: int, slack: float) -> float:
        """Penalty for a connection out of ``flight`` with the given slack."""
        if self.point:
            return max(0.0, self._means[flight] - slack)
        a, tails = self._sorted[flight], self._tails[flight]
        if a.size == 0:
            return 0.0
        k = int(np.searchsorted(a, slack, side="right"))
        return float((tails[k] - slack * (a.size - k)) / a.size)
