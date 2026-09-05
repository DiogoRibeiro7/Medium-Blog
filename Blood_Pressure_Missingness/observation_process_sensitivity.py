"""Sensitivity analysis for irregular blood-pressure observation intensity.

This module quantifies how the estimated systolic time trend changes under several
reasonable ways of handling unequal numbers of readings per observed day. The
specifications are sensitivity analyses, not corrections for an identified
missing-data mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

DAYS_PER_REPORTING_PERIOD: Final[float] = 30.0


@dataclass(frozen=True)
class ObservedDay:
    """One observed calendar day from the privacy-safe aggregate snapshot."""

    day_index: int
    n_readings: int
    mean_systolic_mmHg: float


@dataclass(frozen=True)
class TrendEstimate:
    """One 30-day systolic trend estimate with a robust confidence interval."""

    name: str
    slope_per_30_days: float
    ci95_low_per_30_days: float
    ci95_high_per_30_days: float
    p_value: float


def load_observed_days(path: Path) -> list[ObservedDay]:
    """Load observed days from the public snapshot and validate required fields."""

    rows: list[ObservedDay] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"day_index", "observed", "n_readings", "mean_systolic_mmHg"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Snapshot is missing sensitivity-analysis columns.")

        for row in reader:
            if int(row["observed"]) == 0:
                continue
            day_index = int(row["day_index"])
            n_readings = int(row["n_readings"])
            systolic = float(row["mean_systolic_mmHg"])
            if day_index < 0:
                raise ValueError("day_index must be non-negative.")
            if n_readings <= 0:
                raise ValueError("Observed days must have a positive reading count.")
            if not math.isfinite(systolic):
                raise ValueError("Observed systolic means must be finite.")
            rows.append(
                ObservedDay(
                    day_index=day_index,
                    n_readings=n_readings,
                    mean_systolic_mmHg=systolic,
                )
            )

    if len(rows) < 4:
        raise ValueError("At least four observed days are required for sensitivity analysis.")
    return rows


def _trend_result(name: str, fitted: object, slope_index: int = 1) -> TrendEstimate:
    """Convert a statsmodels robust fit to the common 30-day trend representation."""

    params = np.asarray(getattr(fitted, "params"), dtype=float)
    pvalues = np.asarray(getattr(fitted, "pvalues"), dtype=float)
    interval = np.asarray(getattr(fitted, "conf_int")(alpha=0.05), dtype=float)
    return TrendEstimate(
        name=name,
        slope_per_30_days=float(params[slope_index] * DAYS_PER_REPORTING_PERIOD),
        ci95_low_per_30_days=float(
            interval[slope_index, 0] * DAYS_PER_REPORTING_PERIOD
        ),
        ci95_high_per_30_days=float(
            interval[slope_index, 1] * DAYS_PER_REPORTING_PERIOD
        ),
        p_value=float(pvalues[slope_index]),
    )


def fit_sensitivity_models(days: Sequence[ObservedDay]) -> dict[str, object]:
    """Fit observation-process sensitivity specifications with HC3 covariance.

    The specifications deliberately answer different questions:

    * ``equal_day``: every observed calendar day receives equal weight.
    * ``sampling_adjusted``: adds ``log(1 + readings/day)`` as a covariate.
    * ``reading_weighted``: weights each daily mean by its number of readings.
    * ``capped_weight``: limits reading-count influence to at most three.
    * ``inverse_intensity_stress``: deliberately upweights sparsely sampled days.

    The inverse-intensity model is a stress test only; it is not an inverse
    probability weight because observation probabilities are not identified.
    """

    t = np.asarray([day.day_index for day in days], dtype=float)
    n = np.asarray([day.n_readings for day in days], dtype=float)
    y = np.asarray([day.mean_systolic_mmHg for day in days], dtype=float)

    baseline = sm.OLS(y, sm.add_constant(t)).fit(cov_type="HC3")

    adjusted_design = sm.add_constant(np.column_stack([t, np.log1p(n)]))
    adjusted = sm.OLS(y, adjusted_design).fit(cov_type="HC3")

    reading_weighted = sm.WLS(
        y,
        sm.add_constant(t),
        weights=n,
    ).fit(cov_type="HC3")

    capped_weight = sm.WLS(
        y,
        sm.add_constant(t),
        weights=np.minimum(n, 3.0),
    ).fit(cov_type="HC3")

    inverse_intensity = sm.WLS(
        y,
        sm.add_constant(t),
        weights=1.0 / n,
    ).fit(cov_type="HC3")

    estimates = [
        _trend_result("equal_day", baseline),
        _trend_result("sampling_adjusted", adjusted),
        _trend_result("reading_weighted", reading_weighted),
        _trend_result("capped_weight", capped_weight),
        _trend_result("inverse_intensity_stress", inverse_intensity),
    ]

    adjusted_params = np.asarray(adjusted.params, dtype=float)
    adjusted_ci = np.asarray(adjusted.conf_int(alpha=0.05), dtype=float)
    adjusted_p = np.asarray(adjusted.pvalues, dtype=float)

    return {
        "n_observed_days": len(days),
        "trend_estimates": [estimate.__dict__ for estimate in estimates],
        "sampling_adjustment": {
            "covariate": "log1p_n_readings",
            "coefficient_mmHg": float(adjusted_params[2]),
            "ci95_low_mmHg": float(adjusted_ci[2, 0]),
            "ci95_high_mmHg": float(adjusted_ci[2, 1]),
            "p_value": float(adjusted_p[2]),
        },
        "interpretation": {
            "identified_missingness_mechanism": False,
            "inverse_intensity_is_stress_test_not_ipw": True,
        },
    }


def plot_sensitivity(results: dict[str, object], output_path: Path) -> None:
    """Plot 30-day trend estimates and their robust 95% confidence intervals."""

    raw_estimates = results.get("trend_estimates")
    if not isinstance(raw_estimates, list) or not raw_estimates:
        raise ValueError("Sensitivity results contain no trend estimates.")

    names: list[str] = []
    estimates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for item in raw_estimates:
        if not isinstance(item, dict):
            raise TypeError("Each trend estimate must be a dictionary.")
        name = str(item["name"])
        estimate = float(item["slope_per_30_days"])
        low = float(item["ci95_low_per_30_days"])
        high = float(item["ci95_high_per_30_days"])
        names.append(name.replace("_", " "))
        estimates.append(estimate)
        lower_errors.append(estimate - low)
        upper_errors.append(high - estimate)

    positions = np.arange(len(names), dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.errorbar(
        estimates,
        positions,
        xerr=np.asarray([lower_errors, upper_errors]),
        fmt="o",
        capsize=4,
    )
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(positions, labels=names)
    ax.set_xlabel("Estimated systolic change per 30 days (mmHg)")
    ax.set_title("Systolic trend sensitivity to observation intensity")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/analysis_snapshot.csv"),
        help="Privacy-safe day-indexed snapshot.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("figures/observation_process_sensitivity.json"),
        help="Machine-readable sensitivity results.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("figures/observation_process_sensitivity.svg"),
        help="Coefficient-interval figure.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the sensitivity analysis and write public derived outputs."""

    args = parse_args()
    days = load_observed_days(args.data)
    results = fit_sensitivity_models(days)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_sensitivity(results, args.output_figure)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
