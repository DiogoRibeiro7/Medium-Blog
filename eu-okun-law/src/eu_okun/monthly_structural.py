"""Monthly local projections on the identified common shocks.

Implements ``MONTHLY_IDENTIFICATION_PROTOCOL.md``: responses of the unemployment rate and
of industrial production to ECB monetary policy surprises (Jarociński-Karadi) and OPEC oil
supply surprises (Känzig) at their native monthly frequency. Estimation is by
lag-augmented local projections pooled over countries with country fixed effects and
standard errors clustered by month, reusing the estimators of the quarterly study
(``structural.py``). The shock-conditional ratio ``kappa_IP`` (unemployment response /
industrial-production response) gets the same Anderson-Rubin (Fieller) set. A placebo
regression of the pre-shock change tests the outcome-timing assumption A7.

``kappa_IP`` is not the GDP-based Okun coefficient: industrial production covers NACE
B-D only. Responses are causal only under A1-A5 and A7 of the protocol.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from eu_okun.contract import EU27
from eu_okun.monthly_data import PRIMARY_GROUP, UNEMPLOYMENT_METHOD, MonthlyPanel, month_sequence
from eu_okun.structural import (
    EURO_ENTRY,
    Design,
    Fit,
    OkunSet,
    Predictability,
    Response,
    ShockSpec,
    _response,
    anderson_rubin_set,
    fit_lp,
    newey_west,
)

PANDEMIC = ("2020-01", "2021-12")
PANDEMIC_PRE = 12              # an observation's window starts at t-12 (protocol, section 3)
HORIZONS: tuple[int, ...] = tuple(range(25))
LAGS = 12
RATIO_HORIZONS = (12, 24)
COUNTRY_HORIZON = 12
LEVEL = 0.90
FIRST_ORIGIN = "2009-01"
LONG_FIRST_ORIGIN = "2001-02"
OUTCOMES: tuple[str, ...] = ("ip", "unemployment")
OUTCOME_LABELS = {"ip": "Industrial production (cumulative, %)",
                  "unemployment": "Unemployment rate (pp)"}

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
                       unit="one standard deviation of the monthly series"),
}


@dataclass(frozen=True)
class Settings:
    """Choices that the robustness checks vary (protocol, section 7)."""

    lags: int = LAGS
    exclude_pandemic: bool = True
    driscoll_kraay: bool = False
    first_origin: str = FIRST_ORIGIN
    sample: str = "primary"      # "primary": group D countries; "all": every member
    du_controls: bool = True     # False: lags of the shock and of dip only
    pandemic_margin: int = 0     # months added before t-12 in the pandemic window


PRIMARY = Settings()


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonthlyArrays:
    """Countries by months, NaN where a value is missing (nothing imputed)."""

    countries: tuple[str, ...]
    months: tuple[str, ...]
    u: np.ndarray                # unemployment rate (%)
    lip: np.ndarray              # 100 * ln(industrial production index)

    @property
    def du(self) -> np.ndarray:
        return np.diff(self.u, axis=1, prepend=np.nan)

    @property
    def dip(self) -> np.ndarray:
        return np.diff(self.lip, axis=1, prepend=np.nan)


def monthly_arrays(panel: MonthlyPanel, first: str, last: str,
                   countries: Sequence[str] = EU27) -> MonthlyArrays:
    """Unemployment rate and log production as arrays."""
    months = month_sequence(first, last)
    u = np.full((len(countries), len(months)), np.nan)
    lip = np.full_like(u, np.nan)
    for i, geo in enumerate(countries):
        for j, m in enumerate(months):
            cell = panel.get((geo, m), {})
            rate, ip = cell.get("unemployment"), cell.get("ip")
            if rate is not None:
                u[i, j] = rate
            if ip is not None and ip > 0:
                lip[i, j] = 100.0 * math.log(ip)
    return MonthlyArrays(tuple(countries), tuple(months), u, lip)


def entry_month(geo: str) -> str | None:
    """First month of euro-area membership (from the quarterly entry dates)."""
    quarter = EURO_ENTRY.get(geo)
    if quarter is None:
        return None
    return f"{quarter[:4]}-{(int(quarter[-1]) - 1) * 3 + 1:02d}"


def in_euro_area(geo: str, month: str) -> bool:
    entry = entry_month(geo)
    return entry is not None and month >= entry


def touches_pandemic(months: Sequence[str]) -> bool:
    first, last = PANDEMIC
    return any(first <= m <= last for m in months)


def sample_countries(arrays: MonthlyArrays, settings: Settings) -> list[int]:
    """Indices of the primary (group D) or of all countries."""
    if settings.sample == "all":
        return list(range(len(arrays.countries)))
    if settings.sample != "primary":
        raise ValueError(f"Unknown sample {settings.sample!r}.")
    return [i for i, geo in enumerate(arrays.countries)
            if UNEMPLOYMENT_METHOD[geo] == PRIMARY_GROUP]


def shock_array(months: Sequence[str], shock: Mapping[str, float]) -> np.ndarray:
    return np.asarray([shock.get(m, np.nan) for m in months], dtype=float)


def lp_design(arrays: MonthlyArrays, shock: Mapping[str, float], h: int, spec: ShockSpec,
              settings: Settings, outcomes: Sequence[str] = OUTCOMES,
              countries: Sequence[int] | None = None, placebo: bool = False) -> Design:
    """Rows (i, t) with every outcome, control and shock value observed.

    ``placebo``: the outcome is the change from t-2 to t-1 and every control is shifted
    one month back (lags 2..L+1), so nothing in the row is dated after t-1 except the shock.
    """
    lags, shift = settings.lags, int(placebo)
    months = np.asarray(arrays.months)
    n_months = len(months)
    pre = max(PANDEMIC_PRE, lags) + shift
    first = max(lags + shift + 1, pre)
    ts = np.arange(first, n_months - (0 if placebo else h))
    if ts.size == 0:
        raise ValueError("No origins for the local projection.")
    s = shock_array(arrays.months, shock)
    shocks = np.column_stack([s[ts - lag] for lag in range(lags + 1)])
    pandemic = np.asarray([PANDEMIC[0] <= m <= PANDEMIC[1] for m in arrays.months])
    cum = np.concatenate([[0], np.cumsum(pandemic)])
    end = ts + (0 if placebo else h) + 1
    touches = (cum[end] - cum[np.maximum(ts - pre - settings.pandemic_margin, 0)]) > 0
    base = (months[ts] >= settings.first_origin) & np.all(np.isfinite(shocks), axis=1)
    if settings.exclude_pandemic:
        base &= ~touches
    lag_idx = range(1 + shift, lags + 1 + shift)
    du, dip = arrays.du, arrays.dip
    selected = sample_countries(arrays, settings) if countries is None else list(countries)
    ys: dict[str, list[np.ndarray]] = {o: [] for o in outcomes}
    xs: list[np.ndarray] = []
    c_idx: list[np.ndarray] = []
    t_idx: list[np.ndarray] = []
    for i in selected:
        ok = base.copy()
        if spec.euro_area_only:
            entry = entry_month(arrays.countries[i])
            ok &= (months[ts] >= entry) if entry is not None else np.zeros_like(ok)
        u, lip = arrays.u[i], arrays.lip[i]
        if placebo:
            lhs = {"unemployment": u[ts - 1] - u[ts - 2], "ip": lip[ts - 1] - lip[ts - 2]}
        else:
            lhs = {"unemployment": u[ts + h] - u[ts - 1], "ip": lip[ts + h] - lip[ts - 1]}
        controls = ([du[i, ts - lag] for lag in lag_idx] if settings.du_controls else [])
        controls += [dip[i, ts - lag] for lag in lag_idx]
        x = np.column_stack([shocks, *controls])
        y = np.column_stack([lhs[o] for o in outcomes])
        ok &= np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
        for k, o in enumerate(outcomes):
            ys[o].append(y[ok, k])
        xs.append(x[ok])
        c_idx.append(np.full(int(ok.sum()), i))
        t_idx.append(ts[ok])
    names = (("shock",) + tuple(f"shock_l{lag}" for lag in range(1, lags + 1))
             + (tuple(f"du_l{lag}" for lag in lag_idx) if settings.du_controls else ())
             + tuple(f"dip_l{lag}" for lag in lag_idx))
    return Design({o: np.concatenate(v) for o, v in ys.items()},
                  np.concatenate(xs).reshape(-1, len(names)),
                  np.concatenate(c_idx).astype(int), np.concatenate(t_idx).astype(int), names)


# ----------------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------------


def ratio_set(shock: str, h: int, fit: Fit, lags: int, level: float = LEVEL) -> OkunSet:
    """Point estimate, first-stage t and AR set of kappa_IP = du response / IP response."""
    a, b = float(fit.coef["unemployment"][0]), float(fit.coef["ip"][0])
    v_aa = float(fit.covariance("unemployment", "unemployment", lags)[0, 0])
    v_bb = float(fit.covariance("ip", "ip", lags)[0, 0])
    v_ab = float(fit.covariance("unemployment", "ip", lags)[0, 0])
    critical = float(stats.t.ppf(0.5 + level / 2, fit.clusters - 1))
    kind, lower, upper = anderson_rubin_set(a, b, v_aa, v_ab, v_bb, critical)
    return OkunSet(shock, h, a / b, b / math.sqrt(v_bb), kind, lower, upper, fit.n,
                   fit.clusters)


@dataclass
class MonthlyResults:
    """Responses and ratio sets of one shock under one setting."""

    spec: ShockSpec
    settings: Settings
    scale: float
    responses: list[Response]
    ratios: list[OkunSet]

    def response(self, outcome: str, h: int) -> Response:
        return next(r for r in self.responses if r.outcome == outcome and r.h == h)


def shock_scale(spec: ShockSpec, shock: Mapping[str, float], months: Sequence[str],
                first: str = FIRST_ORIGIN) -> float:
    """The normalisation: the protocol's fixed size, or one standard deviation since 2009."""
    if spec.scale is not None:
        return spec.scale
    values = np.asarray([shock[m] for m in months if m >= first and m in shock])
    return float(values.std(ddof=1))


def estimate_shock(arrays: MonthlyArrays, shocks: Mapping[str, Mapping[str, float]],
                   spec: ShockSpec, settings: Settings = PRIMARY,
                   horizons: Sequence[int] = HORIZONS,
                   outcomes: Sequence[str] = OUTCOMES) -> MonthlyResults:
    """Responses of every outcome at every horizon and, with both outcomes, the ratio sets."""
    series = shocks[spec.series]
    scale = shock_scale(spec, series, arrays.months)
    responses, ratios = [], []
    for h in horizons:
        fit = fit_lp(lp_design(arrays, series, h, spec, settings, outcomes))
        hac = h + 1 if settings.driscoll_kraay else 0
        for outcome in outcomes:
            responses.append(_response(spec.key, outcome, h, fit, scale, hac))
        if h in RATIO_HORIZONS and set(outcomes) == {"ip", "unemployment"}:
            ratios.append(ratio_set(spec.key, h, fit, hac))
    return MonthlyResults(spec, settings, scale, responses, ratios)


# ----------------------------------------------------------------------------------
# Diagnostics and countries
# ----------------------------------------------------------------------------------


def classical_f(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """F-test that every coefficient but the intercept (column 0) is zero."""
    n_obs, k = x.shape
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    cov = float(residuals @ residuals) / (n_obs - k) * np.linalg.inv(x.T @ x)
    r = beta[1:]
    f = float(r @ np.linalg.solve(cov[1:, 1:], r)) / len(r)
    return f, float(stats.f.sf(f, len(r), n_obs - k))


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else math.nan


def predictability(arrays: MonthlyArrays, shock: Mapping[str, float], spec: ShockSpec,
                   countries: Sequence[int], lags: int = LAGS, first: str = FIRST_ORIGIN,
                   exclude_pandemic: bool = True) -> Predictability:
    """Regress the monthly shock on 12 lags of the cross-country means of du and dip.

    Under the null the shock is independent of the lagged regressors, so its variance does
    not depend on them and the classical F-test is valid (the tests check its size by
    simulation with a zero-inflated, heavy-tailed shock).
    """
    du, dip, months = arrays.du, arrays.dip, arrays.months
    rows, ys = [], []
    for t in range(max(lags + 1, PANDEMIC_PRE), len(months)):
        month = months[t]
        if month < first or month not in shock:
            continue
        if exclude_pandemic and touches_pandemic(months[t - PANDEMIC_PRE: t + 1]):
            continue
        members = [i for i in countries
                   if not spec.euro_area_only or in_euro_area(arrays.countries[i], month)]
        if not members:
            continue
        values = ([_mean(du[members, t - lag]) for lag in range(1, lags + 1)]
                  + [_mean(dip[members, t - lag]) for lag in range(1, lags + 1)])
        if np.all(np.isfinite(values)):
            rows.append([1.0, *values])
            ys.append(shock[month])
    x, y = np.asarray(rows), np.asarray(ys)
    f, p = classical_f(x, y)
    series = np.asarray([shock[m] for m in months if m >= first and m in shock])
    autocorr = float(np.corrcoef(series[1:], series[:-1])[0, 1])
    return Predictability(spec.key, f, p, len(y), autocorr)


@dataclass(frozen=True)
class Placebo:
    """Response of the pre-shock change (t-2 to t-1) to the shock at t; zero under A3, A7."""

    shock: str
    sample: str
    outcome: str
    estimate: float
    se: float
    t: float
    p_value: float
    n: int
    clusters: int


def placebo(arrays: MonthlyArrays, shock: Mapping[str, float], spec: ShockSpec,
            settings: Settings, scale: float) -> list[Placebo]:
    """Placebo regressions for both outcomes, clustered by month (protocol, section 5)."""
    fit = fit_lp(lp_design(arrays, shock, 0, spec, settings, OUTCOMES, placebo=True))
    out = []
    for outcome in OUTCOMES:
        b = float(fit.coef[outcome][0])
        se = math.sqrt(float(fit.covariance(outcome, outcome, 0)[0, 0]))
        t = b / se
        p = float(2 * stats.t.sf(abs(t), fit.clusters - 1))
        out.append(Placebo(spec.key, settings.sample, outcome, b * scale, se * abs(scale), t, p,
                           fit.n, fit.clusters))
    return out


@dataclass(frozen=True)
class CountryResponse:
    """Country-by-country LP at one horizon (descriptive)."""

    shock: str
    geo: str
    group: str
    outcome: str
    h: int
    estimate: float
    low90: float
    high90: float
    n: int


def member_countries(arrays: MonthlyArrays, spec: ShockSpec) -> list[int]:
    """Euro-area countries (at any time) for MP, the EU-27 otherwise."""
    return [i for i, geo in enumerate(arrays.countries)
            if not spec.euro_area_only or entry_month(geo) is not None]


def country_responses(arrays: MonthlyArrays, shock: Mapping[str, float], spec: ShockSpec,
                      scale: float, settings: Settings,
                      h: int = COUNTRY_HORIZON) -> list[CountryResponse]:
    """Per-country time-series LPs with Newey-West standard errors (h + 1 lags)."""
    out = []
    for i in member_countries(arrays, spec):
        design = lp_design(arrays, shock, h, spec, settings, countries=[i])
        n, k = design.x.shape
        if n < k + 10:
            continue
        x = np.column_stack([np.ones(n), design.x])
        for outcome in OUTCOMES:
            y = design.y[outcome]
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            cov = newey_west(x, y - x @ beta, h + 1)
            se = math.sqrt(float(cov[1, 1])) * abs(scale)
            crit = float(stats.t.ppf(0.95, n - x.shape[1]))
            est = float(beta[1]) * scale
            geo = arrays.countries[i]
            out.append(CountryResponse(spec.key, geo, UNEMPLOYMENT_METHOD[geo], outcome, h, est,
                                       est - crit * se, est + crit * se, n))
    return out
