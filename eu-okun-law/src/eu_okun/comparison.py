"""Common-sample sensitivity of classical EU-27 Okun-law regressions.

Estimates are descriptive associations. Country-level HAC confidence intervals
must not be interpreted as structural GDP-shock effects or as pairwise tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel, design_matrix, fit_hac_covariance
from eu_okun.contract import EU27, EurostatDataError, quarter_sequence

LAG_ORDERS: tuple[int, ...] = (0, 1, 2, 4)
FOCUS_COUNTRIES: tuple[str, ...] = ("PT", "ES", "NL", "LU")
NORMAL_975 = 1.959963984540054


@dataclass(frozen=True)
class ComparisonRow:
    """One specification fitted on the identical cross-country calendar sample."""

    geo: str
    gdp_lags: int
    controls_participation: bool
    n: int
    sample_start: str
    sample_end: str
    gdp_current: float
    gdp_sum: float
    gdp_sum_hac_se: float
    ci95_low: float
    ci95_high: float
    r_squared: float
    condition_number: float


Design = tuple[np.ndarray, np.ndarray]


def common_sample_designs(
    panel: dict[tuple[str, str], dict[str, float | None]],
    quarters: list[str],
    countries: Sequence[str] = EU27,
    lag_orders: Sequence[int] = LAG_ORDERS,
    min_quarters: int = 30,
) -> tuple[dict[tuple[str, int, bool], Design], list[str]]:
    """Apply one calendar-quarter intersection to *all* model specifications.

    Construct candidate designs on the original (possibly incomplete) panel
    before intersecting periods; missing observations are never shifted to
    nearby quarters or filled with zeros.
    """
    if not countries or not lag_orders or len(set(countries)) != len(countries):
        raise ValueError("Countries and lag orders must be nonempty and unique.")
    if any(lag < 0 for lag in lag_orders) or len(set(lag_orders)) != len(lag_orders):
        raise ValueError("GDP lag orders must be unique nonnegative integers.")
    if quarters != sorted(set(quarters)):
        raise ValueError("Quarters must be unique and sorted.")
    candidates: dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray, list[str]]] = {}
    overlap = set(quarters)
    for geo in countries:
        for lags in lag_orders:
            for with_participation in (False, True):
                x, y, dates = design_matrix(panel, geo, quarters, lags, with_participation)
                candidates[(geo, lags, with_participation)] = (x, y, dates)
                overlap.intersection_update(dates)
    common_dates = [quarter for quarter in quarters if quarter in overlap]
    if min_quarters < 12:
        raise ValueError("At least 12 common quarters must be requested.")
    if len(common_dates) < min_quarters:
        raise EurostatDataError(
            f"Fewer than {min_quarters} common quarters across all countries/models."
        )
    aligned: dict[tuple[str, int, bool], Design] = {}
    for key, (x, y, dates) in candidates.items():
        indices = {date: i for i, date in enumerate(dates)}
        select = [indices[date] for date in common_dates]
        aligned[key] = (x[select, :], y[select])
    return aligned, common_dates


def cumulative_gdp(
    beta: np.ndarray, covariance: np.ndarray, number_gdp_terms: int,
) -> tuple[float, float, float, float]:
    """Return the coefficient sum and normal-approximation HAC interval.

    Variance is c' V c, including every pairwise HAC covariance. Summing
    marginal standard errors instead would be statistically incorrect.
    """
    if (
        beta.ndim != 1 or covariance.shape != (beta.size, beta.size)
        or not 1 <= number_gdp_terms < beta.size
        or not np.isfinite(beta).all() or not np.isfinite(covariance).all()
    ):
        raise ValueError("Invalid contrast dimensions or nonfinite estimates.")
    contrast = np.zeros(beta.size, dtype=float)
    contrast[1 : number_gdp_terms + 1] = 1.0
    estimate = float(contrast @ beta)
    variance = float(contrast @ covariance @ contrast)
    if variance < -1e-10:
        raise ValueError("Negative HAC contrast variance.")
    se = math.sqrt(max(0.0, variance))
    return estimate, se, estimate - NORMAL_975 * se, estimate + NORMAL_975 * se


def fit_comparisons(
    panel: dict[tuple[str, str], dict[str, float | None]],
    quarters: list[str],
    countries: Sequence[str] = EU27,
    lag_orders: Sequence[int] = LAG_ORDERS,
    hac_bandwidth: int = 4,
    min_quarters: int = 30,
) -> tuple[list[ComparisonRow], list[str]]:
    """Fit all specifications and reject, rather than hide, invalid models."""
    aligned, common_dates = common_sample_designs(
        panel, quarters, countries, lag_orders, min_quarters=min_quarters
    )
    results: list[ComparisonRow] = []
    for geo in countries:
        for lags in lag_orders:
            for with_participation in (False, True):
                x, y = aligned[(geo, lags, with_participation)]
                try:
                    beta, covariance, r2, condition = fit_hac_covariance(
                        x, y, maxlags=hac_bandwidth, periods=common_dates
                    )
                    effect, se, lower, upper = cumulative_gdp(beta, covariance, lags + 1)
                except ValueError as exc:
                    raise EurostatDataError(
                        f"Invalid comparative model for {geo}, GDP lags={lags}, "
                        f"participation={with_participation}: {exc}"
                    ) from exc
                results.append(
                    ComparisonRow(
                        geo=geo,
                        gdp_lags=lags,
                        controls_participation=with_participation,
                        n=len(common_dates),
                        sample_start=common_dates[0],
                        sample_end=common_dates[-1],
                        gdp_current=float(beta[1]),
                        gdp_sum=effect,
                        gdp_sum_hac_se=se,
                        ci95_low=lower,
                        ci95_high=upper,
                        r_squared=r2,
                        condition_number=condition,
                    )
                )
    return results, common_dates


def format_report(
    results: Sequence[ComparisonRow], sample: Sequence[str], source_hash: str,
) -> str:
    """Render pre-specified, alphabetic tables without causal rankings."""
    lookup = {(r.geo, r.gdp_lags, r.controls_participation): r for r in results}
    lines = [
        "# EU-27: descriptive Okun coefficient sensitivity",
        "",
        "Research output. These are conditional OLS associations, NOT causal",
        "GDP-shock impulse responses or cross-country significance tests.",
        "",
        f"Common panel: {len(sample)} quarters, {sample[0]} to {sample[-1]}; "
        f"{len(results)} regression specifications.",
        f"Snapshot manifest SHA-256: `{source_hash}`.",
        "",
        "The GDP coefficient sum is the sum of partial regression coefficients",
        "for GDP growth at lags 0 through K. Each coefficient measures",
        "percentage points of change in the unemployment rate per percentage",
        "point of quarterly GDP growth, conditional on other regressors.",
        "The reported interval is a two-sided 95% *normal-approximation*",
        "HAC confidence interval using the **full** covariance matrix.",
        "",
        "## All 27 countries: pre-specified K=2, no participation control",
        "",
        "| Country | GDP coefficient sum | 95% HAC interval | R² | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for geo in EU27:
        r = lookup[(geo, 2, False)]
        lines.append(
            f"| {geo} | {r.gdp_sum:+.3f} | [{r.ci95_low:+.3f}, "
            f"{r.ci95_high:+.3f}] | {r.r_squared:.3f} | {r.n} |"
        )
    lines.extend([
        "",
        "## Lag and participation sensitivity: Portugal, Spain, Netherlands, Luxembourg",
        "",
        "| Country | K | Participation change | Sum | 95% HAC interval | n |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ])
    for geo in FOCUS_COUNTRIES:
        for lags in LAG_ORDERS:
            for controlled in (False, True):
                r = lookup[(geo, lags, controlled)]
                lines.append(
                    f"| {geo} | {lags} | {'included' if controlled else 'excluded'} "
                    f"| {r.gdp_sum:+.3f} | [{r.ci95_low:+.3f}, "
                    f"{r.ci95_high:+.3f}] | {r.n} |"
                )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Every model uses the **same calendar quarters across countries and",
        "  specifications**, avoiding artificial changes caused by sample selection.",
        "- Different GDP lag orders change the coefficient-sum definition.",
        "  Sensitivity across specifications is not a formal confidence interval.",
        "- A country-by-country HAC interval does not provide inference for",
        "  pairwise country differences: cross-country common shocks may correlate",
        "  estimation errors. Ranking countries by point estimates is not justified.",
        "- Contemporaneous GDP growth and participation are potentially endogenous.",
        "  No demand/productivity shock has been identified; no causal estimate exists.",
        "- Eurostat quality flags and the 2021 Labour Force Survey changes remain",
        "  comparability issues. Source revision and institutional mechanisms have",
        "  not been fully addressed.",
        "- These regressions are fitted in-sample; predictive comparisons need",
        "  strictly time-ordered out-of-sample evaluation.",
        "",
    ])
    return "\n".join(lines)


def write_comparisons(directory: Path, output_dir: Path) -> int:
    """Validate provenance, fit the common-sample panel and write three outputs."""
    audit = audit_snapshot(directory)
    period = audit["analysis_period"]
    start, end = period["start"], period["end"]
    if not isinstance(start, str) or not isinstance(end, str):
        raise EurostatDataError("Invalid analysis period in snapshot audit.")
    quarter_grid = quarter_sequence(start, end)
    panel = _read_panel(directory / "processed" / "eu27_quarterly.csv")
    results, dates = fit_comparisons(panel, quarter_grid)
    if len(results) != len(EU27) * len(LAG_ORDERS) * 2:
        raise EurostatDataError("Unexpected comparative model coverage.")
    manifest_hash = hashlib.sha256((directory / "snapshot_manifest.json").read_bytes())
    digest = manifest_hash.hexdigest()
    data: dict[str, Any] = {
        "description": "Common-sample descriptive OLS / HAC sensitivity, NOT causal",
        "normal_ci_level": 0.95,
        "hac_bandwidth_quarters": 4,
        "sample_quarters": dates,
        "snapshot_manifest_sha256": digest,
        "processed_csv_sha256": audit["processed_csv_sha256"],
        "models": [asdict(model) for model in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(model) for model in results)
    (output_dir / "COMPARISON.md").write_text(
        format_report(results, dates, digest), encoding="utf-8"
    )
    return len(results)
