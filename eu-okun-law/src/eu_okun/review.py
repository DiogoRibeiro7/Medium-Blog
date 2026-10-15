"""Critical review of the classical Okun results; no new statistical model is introduced.

Three kinds of procedure live here:

* independent re-computation of the published OLS, HAC and coefficient-sum numbers
  through a deliberately different numerical route (QR solve, kernel-matrix HAC,
  reparameterised coefficient sums);
* re-estimation of the *existing* distributed-lag specifications on pre-declared
  alternative samples (pandemic treatment, Eurostat quality flags, sub-periods)
  and HAC bandwidths;
* re-scoring of the existing expanding-window forecasts on alternative evaluation sets.

Every number is a descriptive association. Nothing here identifies a causal GDP shock.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import (
    _quarter_before,
    _read_panel,
    design_matrix,
    fit_hac_covariance,
    write_baselines,
)
from eu_okun.comparison import (
    FOCUS_COUNTRIES,
    LAG_ORDERS,
    NORMAL_975,
    common_sample_designs,
    cumulative_gdp,
    write_comparisons,
)
from eu_okun.contract import EU27, SERIES, EurostatDataError, quarter_sequence
from eu_okun.dependence import paired_block_comparisons, run_country_pairs
from eu_okun.forecasting import (
    BASELINE_MODEL,
    MODEL_FEATURES,
    ForecastRecord,
    complete_panel_examples,
    expanding_forecasts,
    summarize_forecasts,
    write_validation,
)
from eu_okun.regimes import write_regimes
from eu_okun.stability import fit_stability, holm_adjust, write_stability

Panel = dict[tuple[str, str], dict[str, float | None]]
Flags = dict[tuple[str, str], dict[str, str]]
Design = tuple[np.ndarray, np.ndarray, list[str]]

PANDEMIC_FIRST, PANDEMIC_LAST = "2020-Q1", "2021-Q4"
HAC_BANDWIDTH = 4
BANDWIDTHS: tuple[int, ...] = (0, 2, 4, 8)
CORE_LAGS: tuple[int, ...] = (1, 2, 4)
HEADLINE_LAGS = 2
# A labour-force change is unusable if it crosses a break into quarter t, or if either
# endpoint is flagged as unreliable or as following a different definition.
BREAK_FLAGS = frozenset({"b", "bu"})
ENDPOINT_FLAGS = frozenset({"u", "bu", "d"})
VERIFY_TOLERANCE = 1e-9


def in_pandemic(quarter: str) -> bool:
    """Quarters removed by every pandemic-excluded sample."""
    return PANDEMIC_FIRST <= quarter <= PANDEMIC_LAST


def quarter_index(quarter: str) -> int:
    """Consecutive integer for a YYYY-Qn period identifier."""
    quarter_sequence(quarter, quarter)
    return int(quarter[:4]) * 4 + int(quarter[-1]) - 1


# ----------------------------------------------------------------------------------
# Snapshot inputs
# ----------------------------------------------------------------------------------


def read_flags(path: Path) -> Flags:
    """Eurostat quality flags per country-quarter: GDP, and labour-force survey.

    Unemployment and participation come from the same survey and carry identical flags
    in Eurostat's releases; that identity is checked rather than assumed.
    """
    flags: Flags = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["unemployment_flag"] != row["participation_flag"]:
                raise EurostatDataError(
                    f"Unemployment and participation flags differ at {row['geo']}/{row['quarter']}."
                )
            flags[(row["geo"], row["quarter"])] = {
                "gdp": row["gdp_flag"], "lfs": row["unemployment_flag"],
            }
    return flags


def source_labels(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Units, adjustments, flag meanings and update times as Eurostat labels them."""
    labels: dict[str, dict[str, Any]] = {}
    for spec in SERIES:
        payload = json.loads((data_dir / "raw" / f"{spec.dataset}.json").read_text("utf-8"))
        dimensions = payload["dimension"]
        filters = {
            dim: dimensions[dim]["category"]["label"][code]
            for dim, code in spec.filters.items() if dim != "freq"
        }
        extension = payload.get("extension", {})
        metadata = [
            a.get("href", "") for a in extension.get("annotation", [])
            if a.get("type") == "ESMS_HTML"
        ]
        labels[spec.name] = {
            "metadata_url": metadata[0] if metadata else "",
            "dataset": spec.dataset,
            "label": payload.get("label", ""),
            "updated": payload.get("updated", ""),
            "filters": filters,
            "flag_meanings": extension.get("status", {}).get("label", {}),
        }
    return labels


# ----------------------------------------------------------------------------------
# Independent numerical route
# ----------------------------------------------------------------------------------


def independent_ols_hac(
    x: np.ndarray, y: np.ndarray, maxlags: int, periods: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """OLS and Bartlett HAC covariance computed differently from ``baseline``.

    Coefficients come from a QR decomposition rather than ``lstsq``. The HAC "meat" is
    X' (K * e e') X with an explicit n-by-n Bartlett kernel K on *calendar* distances,
    instead of a loop over lag orders, so an indexing error in either implementation
    cannot be reproduced by the other.
    """
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size or len(periods) != y.size:
        raise ValueError("Incompatible design, response and period dimensions.")
    q, r = np.linalg.qr(x)
    beta = np.linalg.solve(r, q.T @ y)
    residuals = y - x @ beta
    index = np.asarray([quarter_index(p) for p in periods], dtype=float)
    gaps = np.abs(index[:, None] - index[None, :])
    kernel = np.where(gaps <= maxlags, 1.0 - gaps / (maxlags + 1.0), 0.0)
    meat = x.T @ (kernel * np.outer(residuals, residuals)) @ x
    r_inverse = np.linalg.inv(r)
    bread = r_inverse @ r_inverse.T
    return beta, bread @ meat @ bread


def reparameterized_sum(
    x: np.ndarray, y: np.ndarray, gdp_terms: int, maxlags: int, periods: Sequence[str],
) -> tuple[float, float]:
    """Coefficient sum and HAC standard error read directly off one coefficient.

    With z_k = g_{t-k} - g_t for k >= 1, the regression on (1, g_t, z_1, ..., z_K, ...)
    has the GDP coefficient sum as the coefficient on g_t. Its HAC variance is a
    diagonal element, so this checks the contrast variance c'Vc without using it.
    """
    if not 1 <= gdp_terms < x.shape[1]:
        raise ValueError("Invalid number of GDP terms.")
    z = x.copy()
    for column in range(2, gdp_terms + 1):
        z[:, column] = x[:, column] - x[:, 1]
    beta, covariance = independent_ols_hac(z, y, maxlags, periods)
    return float(beta[1]), math.sqrt(max(0.0, float(covariance[1, 1])))


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Centred R-squared from a QR solve."""
    q, r = np.linalg.qr(x)
    residuals = y - x @ np.linalg.solve(r, q.T @ y)
    return 1.0 - float(residuals @ residuals) / float(np.sum((y - y.mean()) ** 2))


# ----------------------------------------------------------------------------------
# Pre-declared alternative samples for the existing specifications
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSpec:
    """One pre-declared estimation sample."""

    name: str
    label: str
    lag_orders: tuple[int, ...]
    common: bool
    min_quarters: int


SAMPLES: tuple[SampleSpec, ...] = (
    SampleSpec("full", "2009-2026 including 2020-2021 (as published)", LAG_ORDERS, True, 30),
    SampleSpec(
        "pandemic_excluded", "2020-2021 removed from outcomes and lag history",
        LAG_ORDERS, True, 30,
    ),
    SampleSpec("pre_2020", "2009-2019 only", LAG_ORDERS, True, 30),
    SampleSpec(
        "quality_screened",
        "pandemic excluded and flagged labour-force changes removed (per country)",
        LAG_ORDERS, False, 25,
    ),
    SampleSpec("post_2021", "2022 onwards; GDP lags 0-1 only", (0, 1), True, 12),
)
SAMPLE_NAMES: tuple[str, ...] = tuple(spec.name for spec in SAMPLES)
CORE_SAMPLES: tuple[str, ...] = ("full", "pandemic_excluded", "pre_2020", "quality_screened")


def flagged_change(flags: Flags, geo: str, quarter: str, max_lag: int) -> bool:
    """True if the outcome at ``quarter`` relies on a flagged observation.

    Labour-force flags: a break invalidates the change *into* the break quarter; a
    low-reliability or definition-differs flag invalidates every change touching that
    quarter. GDP: a break flag on any GDP regressor up to ``max_lag`` invalidates the row.
    """
    now = flags.get((geo, quarter), {}).get("lfs", "")
    before = flags.get((geo, _quarter_before(quarter, 1)), {}).get("lfs", "")
    if now in BREAK_FLAGS or now in ENDPOINT_FLAGS or before in ENDPOINT_FLAGS:
        return True
    return any(
        flags.get((geo, _quarter_before(quarter, lag)), {}).get("gdp", "") in BREAK_FLAGS
        for lag in range(max_lag + 1)
    )


def _sample_panel(panel: Panel, name: str) -> tuple[Panel, Callable[[str], bool]]:
    """Restrict inputs (including lag history) and eligible outcome quarters."""
    keep: Callable[[str], bool]
    if name == "full":
        keep = lambda q: True  # noqa: E731
    elif name in ("pandemic_excluded", "quality_screened"):
        keep = lambda q: not in_pandemic(q)  # noqa: E731
    elif name == "pre_2020":
        keep = lambda q: q <= "2019-Q4"  # noqa: E731
    elif name == "post_2021":
        keep = lambda q: q >= "2022-Q1"  # noqa: E731
    else:
        raise ValueError(f"Unknown sample {name}.")
    return {key: value for key, value in panel.items() if keep(key[1])}, keep


def sample_designs(
    panel: Panel, flags: Flags, spec: SampleSpec, end: str,
    countries: Sequence[str] = EU27,
) -> dict[tuple[str, int, bool], Design]:
    """Designs for every country, lag order and participation setting of one sample.

    Common samples share one calendar intersection across all countries and
    specifications, exactly as the published comparison does. The quality-screened
    sample is country-specific (flags differ by country) but still common across the
    lag orders and participation settings of each country.
    """
    restricted, keep = _sample_panel(panel, spec.name)
    eligible = [q for q in quarter_sequence("2009-Q1", end) if keep(q)]
    if spec.common:
        aligned, dates = common_sample_designs(
            restricted, eligible, countries, spec.lag_orders, min_quarters=spec.min_quarters
        )
        return {key: (x, y, list(dates)) for key, (x, y) in aligned.items()}
    output: dict[tuple[str, int, bool], Design] = {}
    max_lag = max(spec.lag_orders)
    for geo in countries:
        candidates: dict[tuple[str, int, bool], Design] = {}
        overlap = {q for q in eligible if not flagged_change(flags, geo, q, max_lag)}
        for lags in spec.lag_orders:
            for participation in (False, True):
                x, y, dates = design_matrix(restricted, geo, eligible, lags, participation)
                candidates[(geo, lags, participation)] = (x, y, dates)
                overlap.intersection_update(dates)
        common = [q for q in eligible if q in overlap]
        if len(common) < spec.min_quarters:
            raise EurostatDataError(f"Too few screened quarters for {geo}: {len(common)}.")
        for key, (x, y, dates) in candidates.items():
            rows = {date: i for i, date in enumerate(dates)}
            select = [rows[q] for q in common]
            output[key] = (x[select, :], y[select], common)
    return output


@dataclass(frozen=True)
class GridRow:
    """One coefficient-sum estimate: sample x country x GDP lags x participation."""

    sample: str
    geo: str
    gdp_lags: int
    participation: bool
    n: int
    first: str
    last: str
    gdp_sum: float
    se: float
    ci_low: float
    ci_high: float
    r_squared: float


def fit_grid(
    panel: Panel, flags: Flags, end: str, countries: Sequence[str] = EU27,
    bandwidth: int = HAC_BANDWIDTH,
) -> list[GridRow]:
    """Fit the existing OLS/HAC specification on every pre-declared sample."""
    rows: list[GridRow] = []
    for spec in SAMPLES:
        designs = sample_designs(panel, flags, spec, end, countries)
        for (geo, lags, participation), (x, y, dates) in sorted(designs.items()):
            beta, covariance, r2, _ = fit_hac_covariance(
                x, y, maxlags=bandwidth, periods=dates
            )
            total, se, low, high = cumulative_gdp(beta, covariance, lags + 1)
            rows.append(GridRow(
                spec.name, geo, lags, participation, len(y), dates[0], dates[-1],
                total, se, low, high, r2,
            ))
    return rows


def grid_lookup(grid: Iterable[GridRow]) -> dict[tuple[str, str, int, bool], GridRow]:
    """Index grid rows by (sample, country, lags, participation)."""
    return {(r.sample, r.geo, r.gdp_lags, r.participation): r for r in grid}


@dataclass(frozen=True)
class Robustness:
    """Pre-declared robustness verdict for one country's GDP coefficient sum."""

    geo: str
    verdict: str
    verdict_without_pandemic: str
    core_estimates: int
    negative_estimates: int
    ci_excluding_zero: int
    core_min: float
    core_median: float
    core_max: float
    headline_full: float
    headline_excluded: float
    headline_excluded_low: float
    headline_excluded_high: float
    full_outside_excluded_ci: bool
    pandemic_leverage_share: float
    lag_effect: float
    participation_effect: float
    pandemic_effect: float
    sample_effect: float
    dominant_dimension: str
    excluded_ci_excludes_zero_all_bandwidths: bool


VERDICTS: dict[str, str] = {
    "robust_negative": "Robust negative",
    "negative_precision_dependent": "Negative; significance depends on specification",
    "sign_unstable": "Sign-unstable",
}


def _range(values: Iterable[float]) -> float:
    items = list(values)
    return max(items) - min(items)


def verdict_for(rows: Sequence[GridRow]) -> str:
    """Robust negative: every point estimate < 0 and every 95% interval below zero.
    Precision-dependent: every point estimate < 0 but some interval reaches zero.
    Sign-unstable: at least one point estimate >= 0.
    """
    if any(r.gdp_sum >= 0 for r in rows):
        return "sign_unstable"
    if all(r.ci_high < 0 for r in rows):
        return "robust_negative"
    return "negative_precision_dependent"


def classify_country(
    geo: str, lookup: Mapping[tuple[str, str, int, bool], GridRow],
    leverage: float, bandwidth_ci: Mapping[int, tuple[float, float]],
) -> Robustness:
    """Apply the pre-declared verdict rules to the core grid of one country.

    Core grid: GDP lags 1, 2 and 4, with and without participation, on the full,
    pandemic-excluded, pre-2020 and quality-screened samples (24 estimates). The
    second verdict drops the full sample (18 estimates), separating instability that
    comes only from including 2020-2021 from instability that remains without it.
    """
    core = [
        lookup[(sample, geo, lags, participation)]
        for sample in CORE_SAMPLES for lags in CORE_LAGS for participation in (False, True)
    ]
    negative = sum(r.gdp_sum < 0 for r in core)
    values = sorted(r.gdp_sum for r in core)
    effects = {
        "GDP lag order": max(
            _range(lookup[(s, geo, k, p)].gdp_sum for k in CORE_LAGS)
            for s in CORE_SAMPLES for p in (False, True)
        ),
        "participation control": max(
            abs(lookup[(s, geo, k, True)].gdp_sum - lookup[(s, geo, k, False)].gdp_sum)
            for s in CORE_SAMPLES for k in CORE_LAGS
        ),
        "pandemic treatment": max(
            abs(lookup[("full", geo, k, p)].gdp_sum
                - lookup[("pandemic_excluded", geo, k, p)].gdp_sum)
            for k in CORE_LAGS for p in (False, True)
        ),
        "sample selection": max(
            _range(lookup[(s, geo, k, p)].gdp_sum
                   for s in ("pandemic_excluded", "pre_2020", "quality_screened"))
            for k in CORE_LAGS for p in (False, True)
        ),
    }
    full = lookup[("full", geo, HEADLINE_LAGS, False)]
    excluded = lookup[("pandemic_excluded", geo, HEADLINE_LAGS, False)]
    return Robustness(
        geo=geo, verdict=verdict_for(core),
        verdict_without_pandemic=verdict_for([r for r in core if r.sample != "full"]),
        core_estimates=len(core), negative_estimates=negative,
        ci_excluding_zero=sum(r.ci_high < 0 or r.ci_low > 0 for r in core),
        core_min=values[0], core_median=float(np.median(values)), core_max=values[-1],
        headline_full=full.gdp_sum, headline_excluded=excluded.gdp_sum,
        headline_excluded_low=excluded.ci_low, headline_excluded_high=excluded.ci_high,
        full_outside_excluded_ci=not excluded.ci_low <= full.gdp_sum <= excluded.ci_high,
        pandemic_leverage_share=leverage,
        lag_effect=effects["GDP lag order"],
        participation_effect=effects["participation control"],
        pandemic_effect=effects["pandemic treatment"],
        sample_effect=effects["sample selection"],
        dominant_dimension=max(effects, key=effects.__getitem__),
        excluded_ci_excludes_zero_all_bandwidths=all(
            high < 0 or low > 0 for low, high in bandwidth_ci.values()
        ),
    )


def pandemic_leverage(x: np.ndarray, dates: Sequence[str]) -> float:
    """Share of current-quarter GDP variation (sum of squared deviations) from 2020-2021."""
    g = x[:, 1]
    squares = (g - g.mean()) ** 2
    mask = np.asarray([in_pandemic(q) for q in dates])
    return float(squares[mask].sum() / squares.sum())


def bandwidth_intervals(
    panel: Panel, flags: Flags, end: str, countries: Sequence[str] = EU27,
) -> dict[str, dict[int, tuple[float, float]]]:
    """Headline (pandemic-excluded, K=2, no participation) interval per HAC bandwidth."""
    spec = next(s for s in SAMPLES if s.name == "pandemic_excluded")
    designs = sample_designs(panel, flags, spec, end, countries)
    output: dict[str, dict[int, tuple[float, float]]] = {}
    for geo in countries:
        x, y, dates = designs[(geo, HEADLINE_LAGS, False)]
        output[geo] = {}
        for bandwidth in BANDWIDTHS:
            beta, covariance, _, _ = fit_hac_covariance(x, y, bandwidth, dates)
            _, _, low, high = cumulative_gdp(beta, covariance, HEADLINE_LAGS + 1)
            output[geo][bandwidth] = (low, high)
    return output


# ----------------------------------------------------------------------------------
# Verification of the published pipeline
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One verification: how many quantities were compared and the largest discrepancy."""

    area: str
    quantity: str
    method: str
    compared: int
    max_abs_difference: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.max_abs_difference <= self.tolerance


def _check(area: str, quantity: str, method: str, pairs: Sequence[tuple[float, float]],
           tolerance: float = VERIFY_TOLERANCE) -> Check:
    if not pairs:
        raise ValueError(f"No quantities to verify for {area}/{quantity}.")
    worst = max(abs(a - b) for a, b in pairs)
    return Check(area, quantity, method, len(pairs), worst, tolerance)


def verify_comparison(
    panel: Panel, end: str, published: Mapping[str, Any],
) -> list[Check]:
    """Recompute all 216 published coefficient sums by the independent route."""
    quarters = quarter_sequence("2009-Q1", end)
    aligned, dates = common_sample_designs(panel, quarters)
    if dates != published["sample_quarters"]:
        raise EurostatDataError("Published comparison used a different common sample.")
    sums: list[tuple[float, float]] = []
    ses: list[tuple[float, float]] = []
    lows: list[tuple[float, float]] = []
    r2s: list[tuple[float, float]] = []
    covs: list[tuple[float, float]] = []
    for row in published["models"]:
        x, y = aligned[(row["geo"], row["gdp_lags"], row["controls_participation"])]
        total, se = reparameterized_sum(x, y, row["gdp_lags"] + 1, HAC_BANDWIDTH, dates)
        sums.append((total, row["gdp_sum"]))
        ses.append((se, row["gdp_sum_hac_se"]))
        lows.append((total - NORMAL_975 * se, row["ci95_low"]))
        r2s.append((r_squared(x, y), row["r_squared"]))
        _, module_cov, _, _ = fit_hac_covariance(x, y, HAC_BANDWIDTH, dates)
        _, own_cov = independent_ols_hac(x, y, HAC_BANDWIDTH, dates)
        covs.extend(zip(own_cov.ravel().tolist(), module_cov.ravel().tolist(), strict=True))
    return [
        _check("Country comparison", "GDP coefficient sum", "reparameterised regression", sums),
        _check("Country comparison", "HAC s.e. of the sum", "diagonal of reparameterised "
               "kernel-matrix HAC", ses),
        _check("Country comparison", "95% interval bound", "estimate - 1.96 s.e.", lows),
        _check("Country comparison", "R-squared", "QR residuals", r2s),
        _check("Country comparison", "HAC covariance entries", "kernel-matrix HAC", covs),
    ]


def verify_baseline(panel: Panel, end: str, published: Mapping[str, Any]) -> list[Check]:
    """Recompute per-country baseline coefficients and marginal HAC errors.

    The published baseline measured HAC lags in rows rather than calendar quarters
    (corrected in this review). The two coincide when a country's sample has no
    internal calendar gap, so the number of gapped samples is reported alongside.
    """
    quarters = quarter_sequence("2009-Q1", end)
    coefficients: list[tuple[float, float]] = []
    errors: list[tuple[float, float]] = []
    for model in published["models"]:
        x, y, dates = design_matrix(
            panel, model["geo"], quarters, model["lags"], model["participation"]
        )
        beta, covariance = independent_ols_hac(x, y, HAC_BANDWIDTH, dates)
        names = list(model["coefficients"])
        coefficients.extend(
            (float(beta[i]), model["coefficients"][name]) for i, name in enumerate(names)
        )
        errors.extend(
            (math.sqrt(float(covariance[i, i])), model["hac_se"][name])
            for i, name in enumerate(names)
        )
    return [
        _check("Country baselines", "OLS coefficients", "QR solve", coefficients),
        _check("Country baselines", "marginal HAC s.e.", "calendar kernel-matrix HAC", errors),
    ]


def gapped_baseline_samples(panel: Panel, end: str) -> int:
    """Country baseline samples whose complete cases skip a calendar quarter."""
    quarters = quarter_sequence("2009-Q1", end)
    gapped = 0
    for geo in EU27:
        for lags in LAG_ORDERS:
            for participation in (False, True):
                _, _, dates = design_matrix(panel, geo, quarters, lags, participation)
                index = [quarter_index(q) for q in dates]
                gapped += any(b - a != 1 for a, b in zip(index, index[1:], strict=False))
    return gapped


def verify_regimes(panel: Panel, end: str, published: Mapping[str, Any]) -> list[Check]:
    """Recompute every published regime coefficient sum and standard error."""
    sums, ses = [], []
    for name, window in published["windows"].items():
        if window.get("status") != "estimated":
            continue
        restricted, keep = _sample_panel(
            panel, {"pre_pandemic": "pre_2020"}.get(name, name)
        )
        eligible = [q for q in quarter_sequence("2009-Q1", end) if keep(q)]
        aligned, dates = common_sample_designs(
            restricted, eligible, EU27, tuple(window["lag_orders"]),
            min_quarters=12,
        )
        if dates[0] != window["first"] or len(dates) != window["n_quarters"]:
            raise EurostatDataError(f"Regime {name} sample differs from the publication.")
        for row in published["models"]:
            if row["regime"] != name:
                continue
            x, y = aligned[(row["geo"], row["gdp_lags"], row["controls_participation"])]
            total, se = reparameterized_sum(x, y, row["gdp_lags"] + 1, HAC_BANDWIDTH, dates)
            sums.append((total, row["gdp_sum"]))
            ses.append((se, row["gdp_sum_hac_se"]))
    return [
        _check("Regime sensitivity", "GDP coefficient sum", "reparameterised regression", sums),
        _check("Regime sensitivity", "HAC s.e. of the sum", "kernel-matrix HAC", ses),
    ]


def stability_designs(
    panel: Panel, end: str, countries: Sequence[str] = EU27,
) -> tuple[dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray]], list[str],
           dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray]], list[str]]:
    """The pre-2020 and post-2021 common samples used by the stability tests."""
    pre = {key: row for key, row in panel.items() if key[1] <= "2019-Q4"}
    post = {key: row for key, row in panel.items() if key[1] >= "2022-Q1"}
    pre_design, pre_dates = common_sample_designs(
        pre, quarter_sequence("2009-Q1", "2019-Q4"), countries, (1,), min_quarters=30
    )
    post_design, post_dates = common_sample_designs(
        post, quarter_sequence("2022-Q1", end), countries, (1,), min_quarters=12
    )
    return pre_design, pre_dates, post_design, post_dates


def verify_stability(panel: Panel, end: str, published: Mapping[str, Any]) -> list[Check]:
    """Re-derive the interacted pre/post tests from two separate regressions.

    With full interactions and no HAC cross-terms between the periods (they are more
    than four quarters apart), the pooled estimate equals separate pre and post fits:
    change = post sum - pre sum, Var(change) = Var(pre sum) + Var(post sum), and the
    joint Wald statistic uses the summed coefficient covariances.
    """
    pre, pre_dates, post, post_dates = stability_designs(panel, end)
    changes, errors, walds = [], [], []
    for row in published["results"]:
        key = (row["geo"], 1, row["controls_participation"])
        b_pre, v_pre = independent_ols_hac(*pre[key], HAC_BANDWIDTH, pre_dates)
        b_post, v_post = independent_ols_hac(*post[key], HAC_BANDWIDTH, post_dates)
        contrast = np.zeros(b_pre.size)
        contrast[1:3] = 1.0
        change = float(contrast @ (b_post - b_pre))
        variance = float(contrast @ (v_pre + v_post) @ contrast)
        delta = b_post - b_pre
        wald = float(delta @ np.linalg.solve(v_pre + v_post, delta))
        changes.append((change, row["change_post_minus_pre"]))
        errors.append((math.sqrt(variance), row["change_hac_se"]))
        walds.append((wald, row["joint_wald"]))
    return [
        _check("Pre/post stability", "change in GDP sum", "separate pre and post fits", changes),
        _check("Pre/post stability", "HAC s.e. of the change", "summed period covariances",
               errors),
        _check("Pre/post stability", "joint Wald statistic", "summed period covariances",
               walds, tolerance=1e-6),
    ]


def verify_pairs(panel: Panel, end: str, published: Mapping[str, Any]) -> list[Check]:
    """Recompute the observed country differences and re-run the seeded bootstrap."""
    designs, dates = common_sample_designs(
        panel, quarter_sequence("2009-Q1", end), FOCUS_COUNTRIES, (2,)
    )
    if dates != published["quarter_sample"]:
        raise EurostatDataError("Published country pairs used a different sample.")
    sums = {}
    for geo in FOCUS_COUNTRIES:
        x, y = designs[(geo, 2, False)]
        q, r = np.linalg.qr(x)
        sums[geo] = float(np.linalg.solve(r, q.T @ y)[1:4].sum())
    observed = [
        (sums[row["country_a"]] - sums[row["country_b"]], row["estimate_a_minus_b"])
        for row in published["pairs"]
    ]
    samples = {geo: designs[(geo, 2, False)] for geo in FOCUS_COUNTRIES}
    rerun: list[tuple[float, float]] = []
    for length in sorted({row["block_length"] for row in published["pairs"]}):
        again = {
            (r.country_a, r.country_b): r
            for r in paired_block_comparisons(
                samples, replicates=published["replicates"], block_length=length,
                seed=published["seed"],
            )
        }
        for row in published["pairs"]:
            if row["block_length"] == length:
                match = again[(row["country_a"], row["country_b"])]
                rerun.extend([(match.ci95_low, row["ci95_low"]),
                              (match.ci95_high, row["ci95_high"])])
    return [
        _check("Country pairs", "observed difference of sums", "QR solve", observed),
        _check("Country pairs", "bootstrap interval bounds", "same seed, re-run (reproducibility "
               "only; not independent)", rerun),
    ]


def _independent_forecast(
    panel: Panel, geo: str, training: Sequence[str], target: str, model: str,
) -> float:
    """Re-derive one forecast from raw panel values instead of ForecastExample rows."""

    def value(quarter: str, name: str) -> float:
        number = panel[(geo, quarter)][name]
        if number is None:
            raise EurostatDataError(f"Missing {name} for {geo}/{quarter}.")
        return number

    def row(quarter: str) -> tuple[float, list[float]]:
        q1, q2 = _quarter_before(quarter, 1), _quarter_before(quarter, 2)
        features = {
            "du_lag1": value(q1, "unemployment") - value(q2, "unemployment"),
            "dp_lag1": value(q1, "participation") - value(q2, "participation"),
            "gdp_lag1": value(q1, "gdp"),
            "gdp_lag2": value(q2, "gdp"),
        }
        target_change = value(quarter, "unemployment") - value(q1, "unemployment")
        return target_change, [1.0, *(features[name] for name in MODEL_FEATURES[model])]

    if model == "zero_change":
        return 0.0
    responses = np.asarray([row(q)[0] for q in training])
    if model == "historical_mean":
        return float(responses.mean())
    x = np.asarray([row(q)[1] for q in training])
    q, r = np.linalg.qr(x)
    beta = np.linalg.solve(r, q.T @ responses)
    return float(beta @ np.asarray(row(target)[1]))


def verify_forecasts(
    panel: Panel, end: str, records: Sequence[ForecastRecord], published: Mapping[str, Any],
) -> list[Check]:
    """Recompute every forecast from the raw panel and every published score."""
    _, common = complete_panel_examples(panel, quarter_sequence("2009-Q1", end))
    predictions = []
    for record in records:
        training = [q for q in common if q < record.target_quarter]
        predictions.append((
            _independent_forecast(panel, record.geo, training, record.target_quarter,
                                  record.model),
            record.predicted_du_pp,
        ))
    errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        errors[(record.geo, record.model)].append(record.predicted_du_pp - record.actual_du_pp)
    maes = [
        (float(np.mean(np.abs(errors[(s["geo"], s["model"])]))), s["mae_pp"])
        for s in published["scores"]
    ]
    rmses = [
        (math.sqrt(float(np.mean(np.square(errors[(s["geo"], s["model"])])))), s["rmse_pp"])
        for s in published["scores"]
    ]
    return [
        _check("Forecast evaluation", "point forecasts", "QR solve on raw panel values",
               predictions),
        _check("Forecast evaluation", "MAE", "recomputed from forecasts", maes),
        _check("Forecast evaluation", "RMSE", "recomputed from forecasts", rmses),
    ]


def _numeric_difference(left: Any, right: Any, path: str = "") -> float:
    """Largest absolute difference between two JSON documents of identical structure."""
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            raise EurostatDataError(f"Archived output differs at {path or '/'}.")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise EurostatDataError(f"Archived output keys differ at {path or '/'}.")
        return max(
            (_numeric_difference(left[k], right[k], f"{path}/{k}") for k in left), default=0.0
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise EurostatDataError(f"Archived output lengths differ at {path or '/'}.")
        return max(
            (_numeric_difference(a, b, f"{path}/{i}")
             for i, (a, b) in enumerate(zip(left, right, strict=True))),
            default=0.0,
        )
    if left != right:
        raise EurostatDataError(f"Archived output differs at {path or '/'}.")
    return 0.0


PIPELINE_FILES: tuple[str, ...] = (
    "ols_hac.json",
    "comparison/comparison.json",
    "regimes/regimes.json",
    "inference/stability.json",
    "inference/country_pairs.json",
    "forecast/scores.json",
)


@dataclass(frozen=True)
class ArchivedFile:
    """An archived pipeline output compared with its regeneration from the snapshot."""

    path: str
    sha256: str
    max_abs_difference: float


def regenerate_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Re-run every published command on the snapshot (offline)."""
    write_baselines(data_dir, output_dir / "ols_hac.json")
    write_comparisons(data_dir, output_dir / "comparison")
    write_regimes(data_dir, output_dir / "regimes")
    write_stability(data_dir, output_dir / "inference")
    run_country_pairs(data_dir, output_dir / "inference")
    write_validation(data_dir, output_dir / "forecast")


def compare_archived(regenerated: Path, archived: Path) -> list[ArchivedFile]:
    """Compare archived CI outputs with the regenerated ones, file by file."""
    results = []
    for name in PIPELINE_FILES:
        path = archived / name
        if not path.exists():
            raise EurostatDataError(f"Archived output missing: {name}.")
        difference = _numeric_difference(
            json.loads(path.read_text("utf-8")),
            json.loads((regenerated / name).read_text("utf-8")),
        )
        results.append(ArchivedFile(
            name, hashlib.sha256(path.read_bytes()).hexdigest(), difference
        ))
    return results


# ----------------------------------------------------------------------------------
# Stability, bootstrap and forecast reviews
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityReview:
    """Published pre/post statistics with leave-one-out and bandwidth sensitivity."""

    geo: str
    participation: bool
    pre_sum: float
    post_sum: float
    change: float
    holm_p: dict[int, float]
    post_loo_min: float
    post_loo_max: float
    gdp_sd_pre: float
    gdp_sd_post: float
    post_provisional_share: float
    post_definition_flagged: bool

    @property
    def post_sign_stable(self) -> bool:
        return self.post_loo_min > 0 or self.post_loo_max < 0


def review_stability(
    panel: Panel, flags: Flags, end: str, countries: Sequence[str] = EU27,
) -> list[StabilityReview]:
    """Leave-one-quarter-out post-2021 sums and Holm p-values at several bandwidths."""
    by_bandwidth = {
        bandwidth: {
            (r.geo, r.controls_participation): r
            for r in fit_stability(panel, end, countries, hac_bandwidth=bandwidth)[0]
        }
        for bandwidth in BANDWIDTHS
    }
    pre, pre_dates, post, post_dates = stability_designs(panel, end, countries)
    output = []
    for geo in countries:
        for participation in (False, True):
            key = (geo, 1, participation)
            x_post, y_post = post[key]
            loo = []
            for drop in range(y_post.size):
                keep = np.arange(y_post.size) != drop
                beta = np.linalg.lstsq(x_post[keep], y_post[keep], rcond=None)[0]
                loo.append(float(beta[1] + beta[2]))
            published = by_bandwidth[HAC_BANDWIDTH][(geo, participation)]
            output.append(StabilityReview(
                geo=geo, participation=participation,
                pre_sum=published.pre_gdp_sum, post_sum=published.post_gdp_sum,
                change=published.change_post_minus_pre,
                holm_p={b: rows[(geo, participation)].p_change_holm
                        for b, rows in by_bandwidth.items()},
                post_loo_min=min(loo), post_loo_max=max(loo),
                gdp_sd_pre=float(np.std(pre[key][0][:, 1])),
                gdp_sd_post=float(np.std(x_post[:, 1])),
                post_provisional_share=float(np.mean(
                    [flags[(geo, q)]["gdp"] == "p" for q in post_dates]
                )),
                post_definition_flagged=any(
                    flags[(geo, q)]["lfs"] in ENDPOINT_FLAGS for q in post_dates
                ),
            ))
    return output


@dataclass(frozen=True)
class PairReview:
    """A country-pair bootstrap result for one sample and block length."""

    sample: str
    country_a: str
    country_b: str
    block_length: int
    n: int
    estimate: float
    ci_low: float
    ci_high: float
    sign_p: float
    holm_p: float

    @property
    def position(self) -> float:
        """Where the observed estimate sits inside its percentile interval (0..1)."""
        width = self.ci_high - self.ci_low
        return (self.estimate - self.ci_low) / width if width > 0 else 0.5


def review_pairs(
    panel: Panel, end: str, published: Mapping[str, Any], replicates: int,
) -> list[PairReview]:
    """Published (full-sample) pairs next to the same bootstrap without 2020-2021."""
    rows = [
        PairReview("full", r["country_a"], r["country_b"], r["block_length"],
                   len(published["quarter_sample"]), r["estimate_a_minus_b"],
                   r["ci95_low"], r["ci95_high"], r["bootstrap_sign_p"], r["holm_sign_p"])
        for r in published["pairs"]
    ]
    restricted, keep = _sample_panel(panel, "pandemic_excluded")
    eligible = [q for q in quarter_sequence("2009-Q1", end) if keep(q)]
    designs, dates = common_sample_designs(restricted, eligible, FOCUS_COUNTRIES, (2,))
    samples = {geo: designs[(geo, 2, False)] for geo in FOCUS_COUNTRIES}
    for length in (4, 8, 12):
        for r in paired_block_comparisons(
            samples, replicates=replicates, block_length=length, seed=published["seed"]
        ):
            rows.append(PairReview(
                "pandemic_excluded", r.country_a, r.country_b, length, len(dates),
                r.estimate_a_minus_b, r.ci95_low, r.ci95_high, r.bootstrap_sign_p,
                r.holm_sign_p,
            ))
    return rows


FORECAST_SETS: dict[str, str] = {
    "published": "all published targets (training includes 2020-2021)",
    "non_pandemic_targets": "targets outside 2020-Q1..2022-Q2 (training unchanged)",
    "pandemic_free": "2020-2021 removed from training and from every target's inputs",
}


@dataclass(frozen=True)
class ForecastReview:
    """One model in one evaluation set: MAE skill against AR(1) across countries."""

    evaluation: str
    model: str
    origins: int
    countries_better_mae: int
    median_skill_vs_ar1: float
    significantly_better: int
    significantly_worse: int


def _loss_t(differences: np.ndarray, quarters: Sequence[str]) -> float:
    """Mean absolute-loss differential over its HAC standard error (DM-type ratio)."""
    if float(np.var(differences)) <= 0:
        return 0.0
    ones = np.ones((differences.size, 1))
    beta, covariance, _, _ = fit_hac_covariance(
        ones, differences, maxlags=HAC_BANDWIDTH, periods=list(quarters)
    )
    return float(beta[0] / math.sqrt(covariance[0, 0]))


def score_forecast_set(
    evaluation: str, records: Sequence[ForecastRecord],
) -> tuple[list[ForecastReview], dict[str, dict[str, float]]]:
    """MAE skill and DM-type loss-differential counts against the AR(1) benchmark."""
    scores = summarize_forecasts(records)
    by_key: dict[tuple[str, str], list[ForecastRecord]] = defaultdict(list)
    for record in records:
        by_key[(record.geo, record.model)].append(record)
    skill: dict[str, dict[str, float]] = defaultdict(dict)
    for score in scores:
        skill[score.model][score.geo] = score.skill_vs_ar1
    output = []
    for model in MODEL_FEATURES:
        if model == BASELINE_MODEL:
            continue
        better = worse = 0
        for geo in skill[model]:
            own, benchmark = by_key[(geo, model)], by_key[(geo, BASELINE_MODEL)]
            differences = np.asarray([
                abs(a.predicted_du_pp - a.actual_du_pp) - abs(b.predicted_du_pp - b.actual_du_pp)
                for a, b in zip(own, benchmark, strict=True)
            ])
            t = _loss_t(differences, [a.target_quarter for a in own])
            better += t < -NORMAL_975
            worse += t > NORMAL_975
        values = list(skill[model].values())
        output.append(ForecastReview(
            evaluation, model, len({r.target_quarter for r in records}),
            sum(v > 0 for v in values), float(np.median(values)), better, worse,
        ))
    return output, {model: dict(values) for model, values in skill.items()}


def review_forecasts(
    panel: Panel, end: str, records: Sequence[ForecastRecord],
) -> tuple[list[ForecastReview], dict[str, dict[str, dict[str, float]]]]:
    """Score the published forecasts on three pre-declared evaluation sets."""
    affected = set(quarter_sequence("2020-Q1", "2022-Q2"))
    sets = {
        "published": list(records),
        "non_pandemic_targets": [r for r in records if r.target_quarter not in affected],
    }
    restricted, _ = _sample_panel(panel, "pandemic_excluded")
    sets["pandemic_free"] = expanding_forecasts(restricted, end)[0]
    rows: list[ForecastReview] = []
    skills: dict[str, dict[str, dict[str, float]]] = {}
    for name, subset in sets.items():
        review, skill = score_forecast_set(name, subset)
        rows.extend(review)
        skills[name] = skill
    return rows, skills


# ----------------------------------------------------------------------------------
# Data audit summaries
# ----------------------------------------------------------------------------------


def quarter_runs(quarters: Iterable[str]) -> str:
    """Compress sorted quarters into ranges: 2021-Q1..2021-Q4, 2023-Q2."""
    ordered = sorted(set(quarters), key=quarter_index)
    parts: list[str] = []
    start = previous = None
    for quarter in ordered:
        if previous is not None and quarter_index(quarter) == quarter_index(previous) + 1:
            previous = quarter
            continue
        if start is not None and previous is not None:
            parts.append(start if start == previous else f"{start}..{previous}")
        start = previous = quarter
    if start is not None and previous is not None:
        parts.append(start if start == previous else f"{start}..{previous}")
    return ", ".join(parts)


@dataclass(frozen=True)
class FlagSummary:
    """Where one Eurostat flag occurs for one series family and country."""

    series: str
    flag: str
    geo: str
    cells: int
    quarters: str


def summarize_flags(flags: Flags) -> list[FlagSummary]:
    """Flag occurrences by series family, flag and country."""
    found: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for (geo, quarter), values in flags.items():
        for series, flag in values.items():
            if flag:
                found[(series, flag, geo)].append(quarter)
    return [
        FlagSummary(series, flag, geo, len(quarters), quarter_runs(quarters))
        for (series, flag, geo), quarters in sorted(found.items())
    ]


def summarize_missing(panel: Panel) -> list[dict[str, Any]]:
    """Missing cells grouped by series and identical quarter pattern."""
    pattern: dict[tuple[str, str], list[str]] = {}
    for series in ("gdp", "unemployment", "participation"):
        for geo in EU27:
            missing = [q for (g, q), row in panel.items() if g == geo and row[series] is None]
            if missing:
                pattern[(series, geo)] = missing
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (series, geo), quarters in pattern.items():
        groups[(series, quarter_runs(quarters))].append(geo)
    return [
        {"series": series, "quarters": runs, "countries": sorted(geos),
         "cells": len(geos) * len(pattern[(series, geos[0])])}
        for (series, runs), geos in sorted(groups.items())
    ]


# ----------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------


@dataclass
class ReviewResult:
    """Everything the report and figures are generated from."""

    manifest: dict[str, Any]
    audit: dict[str, Any]
    labels: dict[str, dict[str, Any]]
    panel: Panel
    flags: Flags
    end: str
    grid: list[GridRow]
    robustness: list[Robustness]
    bandwidth_ci: dict[str, dict[int, tuple[float, float]]]
    checks: list[Check]
    gapped_baselines: int
    archived: list[ArchivedFile] | None
    stability: list[StabilityReview]
    stability_design: dict[str, Any]
    pairs: list[PairReview]
    forecasts: list[ForecastReview]
    forecast_skill: dict[str, dict[str, dict[str, float]]]
    flag_summary: list[FlagSummary]
    missing: list[dict[str, Any]]
    pair_replicates: int


def run_review(
    data_dir: Path, published_dir: Path | None = None, pair_replicates: int = 999,
) -> ReviewResult:
    """Audit the snapshot, regenerate the pipeline and run every review procedure."""
    audit = audit_snapshot(data_dir)
    manifest = json.loads((data_dir / "snapshot_manifest.json").read_text("utf-8"))
    end = audit["analysis_period"]["end"]
    if not isinstance(end, str):
        raise EurostatDataError("Invalid analysis period in the audit.")
    panel = _read_panel(data_dir / "processed" / "eu27_quarterly.csv")
    flags = read_flags(data_dir / "processed" / "eu27_quarterly.csv")

    with tempfile.TemporaryDirectory() as scratch:
        regenerated = Path(scratch)
        regenerate_pipeline(data_dir, regenerated)
        published = {
            name: json.loads((regenerated / name).read_text("utf-8")) for name in PIPELINE_FILES
        }
        archived = (
            compare_archived(regenerated, published_dir) if published_dir is not None else None
        )

    records = expanding_forecasts(panel, end)[0]
    checks = [
        *verify_baseline(panel, end, published["ols_hac.json"]),
        *verify_comparison(panel, end, published["comparison/comparison.json"]),
        *verify_regimes(panel, end, published["regimes/regimes.json"]),
        *verify_stability(panel, end, published["inference/stability.json"]),
        *verify_pairs(panel, end, published["inference/country_pairs.json"]),
        *verify_forecasts(panel, end, records, published["forecast/scores.json"]),
    ]

    grid = fit_grid(panel, flags, end)
    lookup = grid_lookup(grid)
    full_designs = sample_designs(panel, flags, SAMPLES[0], end)
    intervals = bandwidth_intervals(panel, flags, end)
    robustness = []
    for geo in EU27:
        x_full, _, full_dates = full_designs[(geo, HEADLINE_LAGS, False)]
        robustness.append(classify_country(
            geo, lookup, pandemic_leverage(x_full, full_dates), intervals[geo]
        ))
    stability_json = published["inference/stability.json"]
    forecast_rows, skill = review_forecasts(panel, end, records)
    return ReviewResult(
        manifest=manifest, audit=audit, labels=source_labels(data_dir), panel=panel,
        flags=flags, end=end, grid=grid, robustness=robustness, bandwidth_ci=intervals,
        checks=checks, gapped_baselines=gapped_baseline_samples(panel, end),
        archived=archived,
        stability=review_stability(panel, flags, end),
        stability_design=stability_json["design"],
        pairs=review_pairs(panel, end, published["inference/country_pairs.json"],
                           pair_replicates),
        forecasts=forecast_rows, forecast_skill=skill,
        flag_summary=summarize_flags(flags), missing=summarize_missing(panel),
        pair_replicates=pair_replicates,
    )


def holm_significant(review: ReviewResult, bandwidth: int, level: float = 0.05) -> int:
    """Number of pre/post change tests below ``level`` after Holm adjustment."""
    return sum(row.holm_p[bandwidth] < level for row in review.stability)


__all__ = [
    "Check", "GridRow", "ReviewResult", "Robustness", "SAMPLES", "VERDICTS",
    "classify_country", "fit_grid", "flagged_change", "holm_adjust", "holm_significant",
    "independent_ols_hac", "quarter_runs", "reparameterized_sum", "run_review",
    "sample_designs", "summarize_flags",
]
