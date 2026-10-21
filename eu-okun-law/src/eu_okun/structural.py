"""Local projections on externally identified shocks (``IDENTIFICATION_PROTOCOL.md``).

Responses of cumulative GDP, the unemployment rate and the participation rate to two
common shocks identified outside this study: ECB monetary policy surprises
(Jarociński-Karadi) and OPEC oil supply surprises (Känzig). Estimation is by
lag-augmented local projections pooled over countries with country fixed effects and
standard errors clustered by quarter. The shock-conditional Okun coefficient
(unemployment response / GDP response) gets a weak-instrument-robust Anderson-Rubin
(Fieller) confidence set.

These responses are causal only under the assumptions A1-A6 of the protocol. Nothing in
this module identifies productivity, labour-supply or country-specific GDP shocks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import stats

from eu_okun.contract import EU27, quarter_sequence
from eu_okun.review import Panel

PANDEMIC = ("2020-Q1", "2021-Q4")
HORIZONS: tuple[int, ...] = tuple(range(9))
LAGS = 4
OKUN_HORIZONS = (4, 8)
COUNTRY_HORIZON = 4
LEVEL = 0.90
# Euro-area entry (the ECB's policy applies from that quarter on).
EURO_ENTRY: dict[str, str] = {
    **{g: "1999-Q1" for g in ("AT", "BE", "DE", "ES", "FI", "FR", "IE", "IT", "LU", "NL",
                              "PT")},
    "EL": "2001-Q1", "SI": "2007-Q1", "CY": "2008-Q1", "MT": "2008-Q1", "SK": "2009-Q1",
    "EE": "2011-Q1", "LV": "2014-Q1", "LT": "2015-Q1", "HR": "2023-Q1",
}
OUTCOMES: tuple[str, ...] = ("gdp", "unemployment", "participation")
OUTCOME_LABELS = {"gdp": "Real GDP (cumulative, %)", "unemployment": "Unemployment rate (pp)",
                  "participation": "Participation rate (pp)"}


@dataclass(frozen=True)
class ShockSpec:
    """One shock series, its normalisation and the countries it applies to."""

    key: str
    series: str
    scale: float | None          # size of the normalised shock; None: one standard deviation
    euro_area_only: bool
    label: str
    unit: str
    identified: bool = True      # False for the raw surprise kept as a contrast


SHOCKS: dict[str, ShockSpec] = {
    "MP": ShockSpec("MP", "MP_median", 0.10, True, "ECB monetary policy shock",
                    "+10 bp (1-year OIS scale), a tightening"),
    "OIL": ShockSpec("OIL", "oil_surprise", 10.0, False, "Oil supply news shock",
                     "+10% oil supply surprise (negative supply news)"),
}
VARIANTS: dict[str, ShockSpec] = {
    "MP_pm": replace(SHOCKS["MP"], key="MP_pm", series="MP_pm",
                     label="ECB monetary policy shock (poor man's sign restrictions)"),
    "MP_raw": replace(SHOCKS["MP"], key="MP_raw", series="pc1_hf", identified=False,
                      label="Raw ECB policy surprise (includes information effects)"),
    "OIL_var": replace(SHOCKS["OIL"], key="OIL_var", series="oil_news_shock", scale=None,
                       label="Oil supply news shock (VAR-based)",
                       unit="one standard deviation of the quarterly series"),
}


@dataclass(frozen=True)
class Settings:
    """Choices that the robustness checks vary (protocol, section 6)."""

    lags: int = LAGS
    exclude_pandemic: bool = True
    driscoll_kraay: bool = False
    # Quarters added before t-4 in the pandemic window. The protocol's window (0) covers
    # the quarters of every variable; 1 also covers the level behind the lag-4 change.
    pandemic_margin: int = 0


PRIMARY = Settings()


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelArrays:
    """Countries by quarters, NaN where a value is missing."""

    countries: tuple[str, ...]
    quarters: tuple[str, ...]
    u: np.ndarray
    p: np.ndarray
    g: np.ndarray

    def index(self, quarter: str) -> int:
        return self.quarters.index(quarter)


def panel_arrays(panel: Panel, first: str, last: str,
                 countries: Sequence[str] = EU27) -> PanelArrays:
    """Unemployment, participation and GDP growth as arrays (nothing imputed)."""
    quarters = quarter_sequence(first, last)

    def grid(name: str) -> np.ndarray:
        out = np.full((len(countries), len(quarters)), np.nan)
        for i, geo in enumerate(countries):
            for j, q in enumerate(quarters):
                value = panel.get((geo, q), {}).get(name)
                if value is not None:
                    out[i, j] = float(value)
        return out

    return PanelArrays(tuple(countries), tuple(quarters), grid("unemployment"),
                       grid("participation"), grid("gdp"))


def in_euro_area(geo: str, quarter: str) -> bool:
    entry = EURO_ENTRY.get(geo)
    return entry is not None and quarter >= entry


def touches_pandemic(quarters: Sequence[str]) -> bool:
    first, last = PANDEMIC
    return any(first <= q <= last for q in quarters)


@dataclass(frozen=True)
class Design:
    """Rows of one local projection: outcomes, regressors (shock first), groups."""

    y: dict[str, np.ndarray]      # outcome -> (n,)
    x: np.ndarray                 # (n, k); column 0 is the shock at t
    country: np.ndarray           # (n,) country index
    time: np.ndarray              # (n,) quarter index of the origin t
    names: tuple[str, ...]


def lp_design(arrays: PanelArrays, shock: Mapping[str, float], h: int, spec: ShockSpec,
              settings: Settings, outcomes: Sequence[str] = OUTCOMES,
              countries: Sequence[int] | None = None) -> Design:
    """Rows (i, t) with every outcome, control and shock value observed."""
    q = arrays.quarters
    lags = settings.lags
    rows_y: dict[str, list[float]] = {o: [] for o in outcomes}
    rows_x: list[list[float]] = []
    c_idx: list[int] = []
    t_idx: list[int] = []
    selected = range(len(arrays.countries)) if countries is None else countries
    for i in selected:
        geo = arrays.countries[i]
        for t in range(lags + 1, len(q) - h):
            quarter = q[t]
            if spec.euro_area_only and not in_euro_area(geo, quarter):
                continue
            start = max(0, t - 4 - settings.pandemic_margin)
            if settings.exclude_pandemic and touches_pandemic(q[start: t + h + 1]):
                continue
            shocks = [shock[q[t - lag]] for lag in range(lags + 1) if q[t - lag] in shock]
            if len(shocks) != lags + 1:
                continue
            u, p, g = arrays.u[i], arrays.p[i], arrays.g[i]
            lhs = {
                "gdp": float(np.sum(g[t: t + h + 1])),
                "unemployment": u[t + h] - u[t - 1],
                "participation": p[t + h] - p[t - 1],
            }
            controls = ([u[t - lag] - u[t - lag - 1] for lag in range(1, lags + 1)]
                        + [g[t - lag] for lag in range(1, lags + 1)]
                        + [p[t - lag] - p[t - lag - 1] for lag in range(1, lags + 1)])
            values = [lhs[o] for o in outcomes] + controls
            if not np.all(np.isfinite(values)):
                continue
            for o in outcomes:
                rows_y[o].append(float(lhs[o]))
            rows_x.append([float(v) for v in shocks] + [float(v) for v in controls])
            c_idx.append(i)
            t_idx.append(t)
    names = (("shock",) + tuple(f"shock_l{lag}" for lag in range(1, lags + 1))
             + tuple(f"du_l{lag}" for lag in range(1, lags + 1))
             + tuple(f"g_l{lag}" for lag in range(1, lags + 1))
             + tuple(f"dp_l{lag}" for lag in range(1, lags + 1)))
    return Design({o: np.asarray(v) for o, v in rows_y.items()},
                  np.asarray(rows_x).reshape(-1, len(names)), np.asarray(c_idx, dtype=int),
                  np.asarray(t_idx, dtype=int), names)


# ----------------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------------


def demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group means (country fixed effects by the within transformation)."""
    out = np.array(values, dtype=float, copy=True)
    for g in np.unique(groups):
        rows = groups == g
        out[rows] = out[rows] - out[rows].mean(axis=0)
    return out


def cluster_scores(xd: np.ndarray, residuals: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Sum of x_it * e_it within each quarter, ordered by quarter: (T_used, k)."""
    times = np.unique(time)
    return np.asarray([(xd[time == t] * residuals[time == t, None]).sum(axis=0)
                       for t in times])


def long_run_meat(s_a: np.ndarray, s_b: np.ndarray, times: np.ndarray, lags: int) -> np.ndarray:
    """sum_t s_a s_b' plus Bartlett-weighted cross products at calendar lags 1..lags.

    ``lags = 0`` gives clustering by quarter; ``lags > 0`` the Driscoll-Kraay estimator.
    """
    meat = s_a.T @ s_b
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        for j, t in enumerate(times):
            k = np.searchsorted(times, t - lag)
            if k < len(times) and times[k] == t - lag:
                meat += weight * (np.outer(s_a[j], s_b[k]) + np.outer(s_a[k], s_b[j]))
    return np.asarray(meat)


@dataclass(frozen=True)
class Fit:
    """Pooled fixed-effects LP for several outcomes on the same rows."""

    coef: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    bread: np.ndarray
    times: np.ndarray
    n: int
    clusters: int
    k: int

    def covariance(self, a: str, b: str, lags: int = 0) -> np.ndarray:
        """Cluster-robust covariance between the coefficients of outcomes a and b."""
        meat = long_run_meat(self.scores[a], self.scores[b], self.times, lags)
        g = self.clusters
        factor = g / (g - 1) * (self.n - 1) / max(self.n - self.k, 1)
        return np.asarray(factor * self.bread @ meat @ self.bread)


def fit_lp(design: Design) -> Fit:
    """Within-country OLS of every outcome on the same regressors."""
    if design.x.shape[0] <= design.x.shape[1] + 2:
        raise ValueError("Too few observations for the local projection.")
    xd = demean(design.x, design.country)
    bread = np.linalg.inv(xd.T @ xd)
    coef, scores = {}, {}
    for outcome, y in design.y.items():
        yd = demean(y, design.country)
        b = bread @ xd.T @ yd
        coef[outcome] = b
        scores[outcome] = cluster_scores(xd, yd - xd @ b, design.time)
    k = design.x.shape[1] + len(np.unique(design.country))
    return Fit(coef, scores, bread, np.unique(design.time), int(design.x.shape[0]),
               int(len(np.unique(design.time))), k)


@dataclass(frozen=True)
class Response:
    """Normalised response of one outcome at one horizon."""

    shock: str
    outcome: str
    h: int
    estimate: float
    se: float
    low68: float
    high68: float
    low90: float
    high90: float
    n: int
    clusters: int


def _response(shock: str, outcome: str, h: int, fit: Fit, scale: float,
              lags: int) -> Response:
    estimate = float(fit.coef[outcome][0]) * scale
    se = math.sqrt(float(fit.covariance(outcome, outcome, lags)[0, 0])) * abs(scale)
    z68 = float(stats.t.ppf(0.84, fit.clusters - 1))
    z90 = float(stats.t.ppf(0.95, fit.clusters - 1))
    return Response(shock, outcome, h, estimate, se, estimate - z68 * se, estimate + z68 * se,
                    estimate - z90 * se, estimate + z90 * se, fit.n, fit.clusters)


@dataclass(frozen=True)
class OkunSet:
    """Anderson-Rubin (Fieller) confidence set for kappa = du response / GDP response."""

    shock: str
    h: int
    estimate: float
    first_stage_t: float
    kind: str                    # "bounded", "two rays" or "all"
    lower: float
    upper: float
    n: int
    clusters: int

    @property
    def informative(self) -> bool:
        return self.kind == "bounded"

    def text(self) -> str:
        if self.kind == "bounded":
            return f"[{self.lower:+.3f}, {self.upper:+.3f}]"
        if self.kind == "two rays":
            return f"(-inf, {self.lower:+.3f}] and [{self.upper:+.3f}, +inf)"
        return "(-inf, +inf)"


def anderson_rubin_set(a: float, b: float, v_aa: float, v_ab: float, v_bb: float,
                       critical: float) -> tuple[str, float, float]:
    """{kappa: (a - kappa b)^2 <= c^2 Var(a - kappa b)}, solved in closed form.

    ``a`` and ``b`` are the reduced-form responses of the unemployment change and of
    cumulative GDP to the shock; with one instrument the AR set is Fieller's set.
    """
    c2 = critical**2
    qa = b * b - c2 * v_bb
    qb = -2.0 * (a * b - c2 * v_ab)
    qc = a * a - c2 * v_aa
    disc = qb * qb - 4.0 * qa * qc
    if qa > 0:
        root = math.sqrt(max(disc, 0.0))
        return "bounded", (-qb - root) / (2 * qa), (-qb + root) / (2 * qa)
    if disc <= 0:
        return "all", -math.inf, math.inf
    root = math.sqrt(disc)
    r1, r2 = sorted(((-qb - root) / (2 * qa), (-qb + root) / (2 * qa)))
    return "two rays", r1, r2


def okun_set(shock: str, h: int, fit: Fit, lags: int, level: float = LEVEL) -> OkunSet:
    """Point estimate, first-stage t and AR set of the shock-conditional Okun coefficient."""
    a, b = float(fit.coef["unemployment"][0]), float(fit.coef["gdp"][0])
    v_aa = float(fit.covariance("unemployment", "unemployment", lags)[0, 0])
    v_bb = float(fit.covariance("gdp", "gdp", lags)[0, 0])
    v_ab = float(fit.covariance("unemployment", "gdp", lags)[0, 0])
    critical = float(stats.t.ppf(0.5 + level / 2, fit.clusters - 1))
    kind, lower, upper = anderson_rubin_set(a, b, v_aa, v_ab, v_bb, critical)
    return OkunSet(shock, h, a / b, b / math.sqrt(v_bb), kind, lower, upper, fit.n,
                   fit.clusters)


# ----------------------------------------------------------------------------------
# Diagnostics and countries
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Predictability:
    """F-test of the shock on lagged cross-country means (protocol, A3)."""

    shock: str
    f_statistic: float
    p_value: float
    n: int
    autocorrelation: float


def newey_west(x: np.ndarray, e: np.ndarray, lags: int,
               time: np.ndarray | None = None) -> np.ndarray:
    """HAC covariance of OLS coefficients (Bartlett kernel).

    With ``time`` (one integer period index per row), lags are counted in calendar periods:
    rows ``lag`` periods apart are paired, so a gap in the rows is not treated as adjacent.
    Without it, lags are counted in rows.
    """
    bread = np.linalg.inv(x.T @ x)
    scores = x * e[:, None]
    meat = scores.T @ scores
    if time is None:
        pairs = [(lag, np.arange(lag, len(e)), np.arange(len(e) - lag))
                 for lag in range(1, lags + 1)]
    else:
        index = {int(t): i for i, t in enumerate(time)}
        pairs = []
        for lag in range(1, lags + 1):
            matched = [(i, index[int(t) - lag]) for i, t in enumerate(time)
                       if int(t) - lag in index]
            pairs.append((lag, np.asarray([a for a, _ in matched], dtype=int),
                          np.asarray([b for _, b in matched], dtype=int)))
    for lag, current, previous in pairs:
        w = 1.0 - lag / (lags + 1.0)
        gamma = scores[current].T @ scores[previous]
        meat += w * (gamma + gamma.T)
    return np.asarray(bread @ meat @ bread)


def predictability(arrays: PanelArrays, shock: Mapping[str, float], spec: ShockSpec,
                   lags: int = LAGS) -> Predictability:
    """Regress the quarterly shock on lags of the cross-country mean du and g.

    Under the null the shock is independent of the lagged regressors, so the regression
    error is the shock itself and its variance does not depend on them: the classical F
    test is valid. Robust (HC1, HC3) and HAC variants over-reject with about 60 quarters
    and nine regressors (the tests check the size by simulation).
    """
    q = arrays.quarters
    du = np.diff(arrays.u, axis=1, prepend=np.nan)
    rows, ys = [], []
    for t in range(lags + 1, len(q)):
        if q[t] not in shock:
            continue
        members = [i for i, geo in enumerate(arrays.countries)
                   if not spec.euro_area_only or in_euro_area(geo, q[t])]
        lagged_du = [float(np.nanmean(du[members, t - lag])) for lag in range(1, lags + 1)]
        lagged_g = [float(np.nanmean(arrays.g[members, t - lag])) for lag in range(1, lags + 1)]
        values = lagged_du + lagged_g
        if np.all(np.isfinite(values)):
            rows.append([1.0, *values])
            ys.append(shock[q[t]])
    x, y = np.asarray(rows), np.asarray(ys)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    n_obs, k = x.shape
    residuals = y - x @ beta
    cov = float(residuals @ residuals) / (n_obs - k) * np.linalg.inv(x.T @ x)
    r = beta[1:]
    wald = float(r @ np.linalg.solve(cov[1:, 1:], r))
    q_df, n = len(r), len(y)
    f = wald / q_df
    p = float(stats.f.sf(f, q_df, n - x.shape[1]))
    autocorr = float(np.corrcoef(y[1:], y[:-1])[0, 1])
    return Predictability(spec.key, f, p, n, autocorr)


@dataclass(frozen=True)
class CountryResponse:
    """Country-by-country LP at one horizon (descriptive)."""

    shock: str
    geo: str
    outcome: str
    h: int
    estimate: float
    low90: float
    high90: float
    n: int


def country_responses(arrays: PanelArrays, shock: Mapping[str, float], spec: ShockSpec,
                      scale: float, settings: Settings,
                      h: int = COUNTRY_HORIZON) -> list[CountryResponse]:
    """Per-country time-series LPs with Newey-West standard errors (h + 1 lags, counted in
    calendar quarters)."""
    out = []
    for i, geo in enumerate(arrays.countries):
        design = lp_design(arrays, shock, h, spec, settings, countries=[i])
        n, k = design.x.shape
        if n < k + 10:
            continue
        x = np.column_stack([np.ones(n), design.x])
        for outcome in ("gdp", "unemployment"):
            y = design.y[outcome]
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            cov = newey_west(x, y - x @ beta, h + 1, design.time)
            se = math.sqrt(float(cov[1, 1])) * abs(scale)
            crit = float(stats.t.ppf(0.95, n - x.shape[1]))
            est = float(beta[1]) * scale
            out.append(CountryResponse(spec.key, geo, outcome, h, est, est - crit * se,
                                       est + crit * se, n))
    return out


# ----------------------------------------------------------------------------------
# Everything for one shock
# ----------------------------------------------------------------------------------


@dataclass
class ShockResults:
    """Responses, Okun sets and diagnostics of one shock under one setting."""

    spec: ShockSpec
    settings: Settings
    scale: float
    responses: list[Response]
    okun: list[OkunSet]


def shock_scale(spec: ShockSpec, shock: Mapping[str, float], quarters: Sequence[str]) -> float:
    """The normalisation: the protocol's fixed size, or one standard deviation."""
    if spec.scale is not None:
        return spec.scale
    values = np.asarray([shock[q] for q in quarters if q in shock])
    return float(values.std(ddof=1))


def estimate_shock(arrays: PanelArrays, shocks: Mapping[str, Mapping[str, float]],
                   spec: ShockSpec, settings: Settings = PRIMARY,
                   horizons: Sequence[int] = HORIZONS) -> ShockResults:
    """Responses of every outcome at every horizon, and the Okun sets."""
    series = shocks[spec.series]
    sample_quarters = [q for q in arrays.quarters if q >= "2009-Q2"]
    scale = shock_scale(spec, series, sample_quarters)
    hac = 0
    responses, okun = [], []
    for h in horizons:
        design = lp_design(arrays, series, h, spec, settings)
        fit = fit_lp(design)
        hac = h + 1 if settings.driscoll_kraay else 0
        for outcome in OUTCOMES:
            responses.append(_response(spec.key, outcome, h, fit, scale, hac))
        if h in OKUN_HORIZONS:
            okun.append(okun_set(spec.key, h, fit, hac))
    return ShockResults(spec, settings, scale, responses, okun)


def as_rows(items: Sequence[Any]) -> list[dict[str, Any]]:
    """Dataclass instances as dictionaries (for CSV output)."""
    from dataclasses import asdict

    return [asdict(item) for item in items]
