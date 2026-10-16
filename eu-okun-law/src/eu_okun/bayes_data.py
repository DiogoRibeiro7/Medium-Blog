"""Arrays for the hierarchical Bayesian Okun model (pure NumPy, no JAX).

The model works on a balanced country-by-quarter grid. Every quarter in the window
must be observed for every country: values are never imputed. Observations whose
change relies on a flagged Eurostat value enter the grid but are masked out of the
likelihood (see ``BAYES_PROTOCOL.md``, section 2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from eu_okun.baseline import _quarter_before
from eu_okun.contract import EU27, EurostatDataError, quarter_sequence
from eu_okun.review import Flags, Panel, flagged_change

PRIMARY_WINDOWS: tuple[tuple[str, str], ...] = (("2009-Q2", "2019-Q4"),)
PANDEMIC_EXCLUDED_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2009-Q2", "2019-Q4"),
    ("2022-Q3", "2026-Q2"),
)
# Quarters whose values may never be used, as outcomes or as lag/difference history.
EXCLUDED_INPUTS: tuple[tuple[str, str], ...] = (("2020-Q1", "2021-Q4"),)


@dataclass(frozen=True)
class BayesData:
    """Balanced arrays for N countries and T quarters (segments may be non-adjacent).

    ``y[i, t]`` is the change in the unemployment rate (pp). ``g[i, t]`` is GDP growth
    in the same quarter and ``z[i, t, k]`` is ``g[i, t-k-1] - g[i, t]`` for the k-th
    lag, so that the coefficient on ``g`` is the coefficient *sum*. ``dp`` is the change
    in participation. ``mask`` is True where the likelihood term is used, and
    ``segment_start`` marks the first quarter of each contiguous segment, where the
    AR(1) error has no predecessor.
    """

    countries: tuple[str, ...]
    quarters: tuple[str, ...]
    lags: int
    y: np.ndarray
    g: np.ndarray
    z: np.ndarray
    dp: np.ndarray
    mask: np.ndarray
    segment_start: np.ndarray
    masked_reasons: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_countries(self) -> int:
        return len(self.countries)

    @property
    def n_quarters(self) -> int:
        return len(self.quarters)

    def gdp_design(self) -> np.ndarray:
        """Classical design [1, g_t, g_{t-1}, ..., g_{t-K}] per country, shape (N, T, K+2)."""
        lagged = self.z + self.g[:, :, None]
        ones = np.ones_like(self.g)[:, :, None]
        return np.concatenate([ones, self.g[:, :, None], lagged], axis=2)


def window_quarters(windows: Sequence[tuple[str, str]]) -> tuple[list[str], np.ndarray]:
    """Quarters of the requested windows and a flag for each segment's first quarter."""
    quarters: list[str] = []
    starts: list[bool] = []
    previous_end: str | None = None
    for first, last in windows:
        segment = quarter_sequence(first, last)
        if previous_end is not None and segment[0] <= previous_end:
            raise ValueError("Windows must be ordered and non-overlapping.")
        quarters.extend(segment)
        starts.extend([True] + [False] * (len(segment) - 1))
        previous_end = last
    return quarters, np.asarray(starts, dtype=bool)


def build_bayes_data(
    panel: Panel,
    flags: Flags,
    windows: Sequence[tuple[str, str]] = PRIMARY_WINDOWS,
    lags: int = 2,
    countries: Sequence[str] = EU27,
    excluded: Sequence[tuple[str, str]] = EXCLUDED_INPUTS,
) -> BayesData:
    """Assemble model arrays; raise if any required observation is missing.

    Lag and difference history may come from before a window (it is observed data),
    but never from an ``excluded`` period: a window must start late enough that no
    outcome, lag or difference touches one (2022-Q3 for two GDP lags after 2021).
    """
    if lags < 1:
        raise ValueError("The model needs at least one GDP lag.")
    if not countries or len(set(countries)) != len(countries):
        raise ValueError("Countries must be unique and non-empty.")
    quarters, segment_start = window_quarters(windows)
    window_set = set(quarters)
    gaps = {q for first, last in excluded for q in quarter_sequence(first, last)}
    n, t = len(countries), len(quarters)
    y = np.empty((n, t))
    g = np.empty((n, t))
    z = np.empty((n, t, lags))
    dp = np.empty((n, t))
    mask = np.ones((n, t), dtype=bool)
    reasons: dict[str, list[str]] = {}

    def value(geo: str, quarter: str, name: str) -> float:
        if quarter in gaps:
            raise EurostatDataError(f"{geo}/{quarter} lies in an excluded period; start "
                                    "the window later.")
        row = panel.get((geo, quarter))
        number = None if row is None else row[name]
        if number is None:
            raise EurostatDataError(f"Missing {name} for {geo}/{quarter}; nothing is imputed.")
        return number

    for i, geo in enumerate(countries):
        for j, quarter in enumerate(quarters):
            previous = _quarter_before(quarter, 1)
            y[i, j] = value(geo, quarter, "unemployment") - value(geo, previous, "unemployment")
            dp[i, j] = value(geo, quarter, "participation") - value(geo, previous, "participation")
            g[i, j] = value(geo, quarter, "gdp")
            for k in range(lags):
                z[i, j, k] = value(geo, _quarter_before(quarter, k + 1), "gdp") - g[i, j]
            if flagged_change(flags, geo, quarter, lags):
                mask[i, j] = False
                reasons.setdefault(geo, []).append(quarter)
        # The AR(1) term for t uses the residual at t-1: mask it when t-1 is masked.
        follows = np.zeros(t, dtype=bool)
        follows[1:] = ~mask[i, :-1] & ~segment_start[1:]
        mask[i] &= ~follows
    if not window_set:
        raise ValueError("Empty estimation window.")
    return BayesData(
        countries=tuple(countries), quarters=tuple(quarters), lags=lags, y=y, g=g, z=z,
        dp=dp, mask=mask, segment_start=segment_start, masked_reasons=reasons,
    )
