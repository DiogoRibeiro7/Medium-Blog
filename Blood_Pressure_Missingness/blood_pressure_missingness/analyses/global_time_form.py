"""Sensitivity of the global systolic time association to time-form choice.

The primary analysis reports an equal-observed-day linear OLS association with
HC3 covariance. This module asks whether the negative global association depends
materially on that particular straight-line specification.

Three deliberately distinct summaries are reported:

1. the primary linear OLS slope per 30 days;
2. a quadratic OLS fit summarized by its fitted end-to-end average change per
   30 days, with an HC3 interval for that contrast;
3. the Theil-Sen median pairwise slope and its nonparametric confidence interval.

These are sensitivity estimands, not interchangeable estimates of one common
parameter, and no preferred trajectory is selected here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

from blood_pressure_missingness import public_analysis as primary

DAYS_PER_REPORTING_PERIOD = 30.0


@dataclass(frozen=True)
class IntervalEstimate:
    """One scalar estimate with a two-sided 95% interval."""

    estimate_per_30_days: float
    ci95_low_per_30_days: float
    ci95_high_per_30_days: float


def _observed_arrays(
    records: Sequence[primary.DailyRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed day indices and finite systolic daily means."""

    observed = primary.observed_records(records)
    if len(observed) < 3:
        raise ValueError("At least three observed days are required.")

    x = np.asarray([record.day_index for record in observed], dtype=float)
    y = np.asarray(
        [float(record.mean_systolic_mmHg) for record in observed],
        dtype=float,
    )
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Observed day indices and systolic means must be finite.")
    if len(np.unique(x)) != len(x):
        raise ValueError("Observed day indices must be unique.")
    return x, y


def _linear_hc3(x: np.ndarray, y: np.ndarray) -> IntervalEstimate:
    """Fit the primary equal-day linear HC3 association."""

    fitted = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    interval = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
    return IntervalEstimate(
        estimate_per_30_days=float(fitted.params[1] * DAYS_PER_REPORTING_PERIOD),
        ci95_low_per_30_days=float(interval[1, 0] * DAYS_PER_REPORTING_PERIOD),
        ci95_high_per_30_days=float(interval[1, 1] * DAYS_PER_REPORTING_PERIOD),
    )


def _quadratic_end_to_end_hc3(
    x: np.ndarray,
    y: np.ndarray,
) -> IntervalEstimate:
    """Fit quadratic OLS and summarize its fitted endpoint contrast per 30 days."""

    design = np.column_stack([np.ones(len(x)), x, x**2])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Quadratic global design matrix is rank deficient.")

    fitted = sm.OLS(y, design).fit(cov_type="HC3")
    x0 = float(np.min(x))
    x1 = float(np.max(x))
    span = x1 - x0
    if span <= 0.0:
        raise ValueError("Observed days must span positive calendar time.")

    endpoint_contrast = np.asarray(
        [0.0, x1 - x0, x1**2 - x0**2],
        dtype=float,
    )
    scale = DAYS_PER_REPORTING_PERIOD / span
    contrast = endpoint_contrast * scale
    estimate = float(contrast @ np.asarray(fitted.params, dtype=float))
    variance = float(contrast @ np.asarray(fitted.cov_params()) @ contrast)
    if variance < 0.0 and abs(variance) < 1e-12:
        variance = 0.0
    if variance < 0.0:
        raise ValueError("Quadratic endpoint-contrast variance must be non-negative.")
    standard_error = math.sqrt(variance)
    z975 = float(stats.norm.ppf(0.975))
    return IntervalEstimate(
        estimate_per_30_days=estimate,
        ci95_low_per_30_days=estimate - z975 * standard_error,
        ci95_high_per_30_days=estimate + z975 * standard_error,
    )


def _theil_sen(x: np.ndarray, y: np.ndarray) -> IntervalEstimate:
    """Return the Theil-Sen median pairwise slope on the same calendar-day scale."""

    result = stats.theilslopes(y, x, alpha=0.95)
    return IntervalEstimate(
        estimate_per_30_days=float(result.slope * DAYS_PER_REPORTING_PERIOD),
        ci95_low_per_30_days=float(result.low_slope * DAYS_PER_REPORTING_PERIOD),
        ci95_high_per_30_days=float(result.high_slope * DAYS_PER_REPORTING_PERIOD),
    )


def analyze_global_time_form_sensitivity(
    records: Sequence[primary.DailyRecord],
) -> dict[str, object]:
    """Compare linear, quadratic-contrast, and Theil-Sen global time summaries."""

    x, y = _observed_arrays(records)
    linear = _linear_hc3(x, y)
    quadratic = _quadratic_end_to_end_hc3(x, y)
    theil_sen = _theil_sen(x, y)

    estimates = {
        "linear_equal_day_hc3": linear.__dict__,
        "quadratic_end_to_end_average_hc3": quadratic.__dict__,
        "theil_sen_median_pairwise_slope": theil_sen.__dict__,
    }
    all_negative = all(
        float(item["ci95_high_per_30_days"]) < 0.0
        for item in estimates.values()
    )

    return {
        "n_observed_days": len(x),
        "first_observed_day_index": int(np.min(x)),
        "last_observed_day_index": int(np.max(x)),
        "estimates": estimates,
        "all_95_percent_intervals_below_zero": all_negative,
        "interpretation": {
            "purpose": "test_global_time_association_sensitivity_to_time_form_and_slope_estimator",
            "preferred_model_selected_by_this_analysis": False,
            "estimands_are_interchangeable": False,
            "quadratic_summary_is_endpoint_average_change": True,
            "theil_sen_is_median_pairwise_slope": True,
            "causal_interpretation": False,
        },
    }


def write_results(path: Path, results: dict[str, object]) -> None:
    """Write the derived sensitivity results as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/analysis_snapshot.csv"),
        help="Privacy-safe day-indexed blood-pressure snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/global_time_form_sensitivity.json"),
        help="Path receiving the derived global time-form sensitivity results.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the sensitivity analysis and write the derived output."""

    args = parse_args()
    records = primary.load_snapshot(args.data)
    results = analyze_global_time_form_sensitivity(records)
    write_results(args.output, results)
    print(
        "Wrote global time-form sensitivity for "
        f"{results['n_observed_days']} observed days."
    )


if __name__ == "__main__":
    main()
