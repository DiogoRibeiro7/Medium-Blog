"""Arrays for the hierarchical Bayesian Okun model (pure NumPy, no JAX).

The model works on a balanced country-by-quarter grid. Every quarter in the window
must be observed for every country: values are never imputed. Observations whose
change relies on a flagged Eurostat value enter the grid but are masked out of the
likelihood (see ``BAYES_PROTOCOL.md``, section 2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
# First quarter of the post-pandemic regime (amendment 4); 2020-2021 are never sampled.
POST_START = "2022-Q1"
PROVISIONAL_FLAGS = frozenset({"p"})


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

    def y_lag(self) -> np.ndarray:
        """y[i, t-1] inside a segment; zero at a segment's first quarter (never used there)."""
        lagged = np.concatenate([np.zeros_like(self.y[:, :1]), self.y[:, :-1]], axis=1)
        lagged[:, self.segment_start] = 0.0
        return lagged

    def post(self) -> np.ndarray:
        """True for quarters of the post-pandemic regime (from ``POST_START``)."""
        return np.asarray([q >= POST_START for q in self.quarters], dtype=bool)

    def calendar_gaps(self) -> np.ndarray:
        """Calendar quarters since the previous sample quarter (zero for the first)."""
        index = np.asarray([quarter_index(q) for q in self.quarters], dtype=float)
        return np.concatenate([[0.0], np.diff(index)])


def quarter_index(quarter: str) -> int:
    """Consecutive integer for a quarter label such as ``2009-Q2``."""
    year, q = quarter.split("-Q")
    return int(year) * 4 + int(q) - 1


def regime_terms(data: BayesData, draws: Mapping[str, np.ndarray],
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """theta, alpha and sigma by country and quarter, for draws with any leading shape.

    ``draws`` holds country parameters of shape (..., N). Without a regime shift or a
    drift the results keep shape (..., N, 1) and are exactly the stored parameters;
    otherwise the post-2021 shifts (``delta``, ``alpha_shift``, ``log_sigma_shift``)
    and the common drift ``kappa`` (..., T) are added.
    """
    theta = np.asarray(draws["theta"])[..., :, None]
    alpha = np.asarray(draws["alpha"])[..., :, None]
    sigma = np.asarray(draws["sigma"])[..., :, None]
    post = data.post().astype(float)
    if "delta" in draws:
        theta = theta + np.asarray(draws["delta"])[..., :, None] * post
    if "kappa" in draws:
        theta = theta + np.asarray(draws["kappa"])[..., None, :]
    if "alpha_shift" in draws:
        alpha = alpha + np.asarray(draws["alpha_shift"])[..., :, None] * post
    if "log_sigma_shift" in draws:
        shift = np.asarray(draws["log_sigma_shift"])[..., None, None]
        sigma = sigma * np.exp(shift * post)
    return theta, alpha, sigma


def ar1_path(e: np.ndarray, rho: np.ndarray | float, segment_start: np.ndarray) -> np.ndarray:
    """Stationary unit-variance AR(1) along the last axis, restarted at segment starts.

    ``e`` holds standard normal innovations (..., T) and ``rho`` broadcasts against
    ``e[..., 0]``: f[t] = rho*f[t-1] + sqrt(1 - rho^2)*e[t], and f = e at a start.
    """
    rho = np.asarray(rho, dtype=float)
    scale = np.sqrt(np.maximum(1.0 - rho**2, 0.0))
    f = np.empty_like(e, dtype=float)
    for j in range(e.shape[-1]):
        if segment_start[j]:
            f[..., j] = e[..., j]
        else:
            f[..., j] = rho * f[..., j - 1] + scale * e[..., j]
    return f


def ardl_path(x: np.ndarray, psi: np.ndarray, y: np.ndarray,
              segment_start: np.ndarray) -> np.ndarray:
    """Simulate y[t] = x[t] + psi*y[t-1] along the last axis, shape (..., N, T).

    ``x`` is everything except the lagged term and ``psi`` has shape (..., N). The
    ARDL likelihood conditions on each segment's first quarter, so the simulation starts
    there from the observed ``y`` (N, T).
    """
    out = np.empty(np.broadcast_shapes(x.shape, y.shape), dtype=float)
    for j in range(out.shape[-1]):
        if segment_start[j]:
            out[..., j] = y[:, j]
        else:
            out[..., j] = x[..., j] + psi * out[..., j - 1]
    return out


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
    provisional_gdp: bool = True,
) -> BayesData:
    """Assemble model arrays; raise if any required observation is missing.

    Lag and difference history may come from before a window (it is observed data),
    but never from an ``excluded`` period: a window must start late enough that no
    outcome, lag or difference touches one (2022-Q3 for two GDP lags after 2021).
    With ``provisional_gdp=False``, terms whose GDP inputs (lags 0..K) carry a
    provisional flag are masked as well (amendment 4).
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
            provisional = not provisional_gdp and any(
                flags.get((geo, _quarter_before(quarter, k)), {}).get("gdp", "")
                in PROVISIONAL_FLAGS for k in range(lags + 1))
            if flagged_change(flags, geo, quarter, lags) or provisional:
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
