"""Sensitivity of the gap-defined episode contrast to observation intensity.

The gap-aware analysis shows that the global systolic association is dominated by
an episode-level contrast across the longest unobserved interval. This module asks
whether that contrast is itself sensitive to unequal numbers of readings per
observed day.

These specifications are sensitivity analyses. In particular, inverse intensity
is a deliberate stress test and is not inverse-probability weighting because the
probability of observing a day is not identified from this tracker.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

import analysis as primary
import gap_aware_trend_decomposition as gap_aware

DAYS_PER_REPORTING_PERIOD: Final[float] = 30.0
SPECIFICATION_NAMES: Final[tuple[str, ...]] = (
    "equal_day",
    "sampling_adjusted",
    "reading_weighted",
    "capped_weight",
    "inverse_intensity_stress",
)


@dataclass(frozen=True)
class EpisodeSensitivityEstimate:
    """One episode-contrast sensitivity estimate with HC3 intervals."""

    name: str
    episode_difference_mmHg: float
    episode_ci95_low_mmHg: float
    episode_ci95_high_mmHg: float
    episode_p_value: float
    within_episode_slope_per_30_days: float
    within_slope_ci95_low_per_30_days: float
    within_slope_ci95_high_per_30_days: float
    within_slope_p_value: float


def _observed_systolic(record: primary.DailyRecord) -> float:
    """Return one finite systolic daily mean."""

    if not record.observed or record.mean_systolic_mmHg is None:
        raise ValueError("Expected an observed record with a systolic mean.")
    value = float(record.mean_systolic_mmHg)
    if not math.isfinite(value):
        raise ValueError("Observed systolic means must be finite.")
    return value


def _episode_design_arrays(
    records: Sequence[primary.DailyRecord],
) -> tuple[
    gap_aware.InternalGap,
    list[primary.DailyRecord],
    list[primary.DailyRecord],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build a fixed gap-aware episode design from the public snapshot."""

    gap = gap_aware.dominant_internal_gap(records)
    before, after = gap_aware.split_observation_episodes(records, gap)
    combined = [*before, *after]

    y = np.asarray([_observed_systolic(record) for record in combined], dtype=float)
    n_readings = np.asarray([record.n_readings for record in combined], dtype=float)
    if np.any(~np.isfinite(n_readings)) or np.any(n_readings <= 0.0):
        raise ValueError("Observed reading counts must be finite and positive.")

    episode = np.asarray(
        [0.0] * len(before) + [1.0] * len(after),
        dtype=float,
    )
    before_mean_day = float(np.mean([record.day_index for record in before]))
    after_mean_day = float(np.mean([record.day_index for record in after]))
    within_time = np.asarray(
        [record.day_index - before_mean_day for record in before]
        + [record.day_index - after_mean_day for record in after],
        dtype=float,
    )
    return gap, before, after, y, n_readings, episode, within_time


def _fit_model(
    name: str,
    y: np.ndarray,
    episode: np.ndarray,
    within_time: np.ndarray,
    *,
    adjustment: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[EpisodeSensitivityEstimate, object]:
    """Fit one HC3 sensitivity specification and extract common estimands."""

    columns = [episode, within_time]
    if adjustment is not None:
        if adjustment.shape != y.shape or np.any(~np.isfinite(adjustment)):
            raise ValueError("Sampling adjustment must be finite and aligned.")
        columns.append(adjustment)

    design = sm.add_constant(np.column_stack(columns))
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"Design matrix for {name!r} is rank deficient.")

    if weights is None:
        fitted = sm.OLS(y, design).fit(cov_type="HC3")
    else:
        if weights.shape != y.shape or np.any(~np.isfinite(weights)):
            raise ValueError("Weights must be finite and aligned.")
        if np.any(weights <= 0.0):
            raise ValueError("Weights must be strictly positive.")
        fitted = sm.WLS(y, design, weights=weights).fit(cov_type="HC3")

    interval = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
    params = np.asarray(fitted.params, dtype=float)
    pvalues = np.asarray(fitted.pvalues, dtype=float)
    return (
        EpisodeSensitivityEstimate(
            name=name,
            episode_difference_mmHg=float(params[1]),
            episode_ci95_low_mmHg=float(interval[1, 0]),
            episode_ci95_high_mmHg=float(interval[1, 1]),
            episode_p_value=float(pvalues[1]),
            within_episode_slope_per_30_days=float(
                params[2] * DAYS_PER_REPORTING_PERIOD
            ),
            within_slope_ci95_low_per_30_days=float(
                interval[2, 0] * DAYS_PER_REPORTING_PERIOD
            ),
            within_slope_ci95_high_per_30_days=float(
                interval[2, 1] * DAYS_PER_REPORTING_PERIOD
            ),
            within_slope_p_value=float(pvalues[2]),
        ),
        fitted,
    )


def fit_episode_observation_sensitivity(
    records: Sequence[primary.DailyRecord],
) -> dict[str, object]:
    """Fit episode-contrast sensitivity specifications with HC3 covariance."""

    gap, before, after, y, n_readings, episode, within_time = _episode_design_arrays(
        records
    )
    log_readings = np.log1p(n_readings)

    equal_day, _ = _fit_model(
        "equal_day", y, episode, within_time
    )
    sampling_adjusted, adjusted_fit = _fit_model(
        "sampling_adjusted",
        y,
        episode,
        within_time,
        adjustment=log_readings,
    )
    reading_weighted, _ = _fit_model(
        "reading_weighted",
        y,
        episode,
        within_time,
        weights=n_readings,
    )
    capped_weight, _ = _fit_model(
        "capped_weight",
        y,
        episode,
        within_time,
        weights=np.minimum(n_readings, 3.0),
    )
    inverse_intensity, _ = _fit_model(
        "inverse_intensity_stress",
        y,
        episode,
        within_time,
        weights=1.0 / n_readings,
    )

    estimates = [
        equal_day,
        sampling_adjusted,
        reading_weighted,
        capped_weight,
        inverse_intensity,
    ]
    adjusted_params = np.asarray(adjusted_fit.params, dtype=float)
    adjusted_ci = np.asarray(adjusted_fit.conf_int(alpha=0.05), dtype=float)
    adjusted_pvalues = np.asarray(adjusted_fit.pvalues, dtype=float)

    return {
        "dominant_internal_gap": gap.__dict__,
        "n_observed_days": len(before) + len(after),
        "episode_sizes": {
            "pre_gap_observed_days": len(before),
            "post_gap_observed_days": len(after),
        },
        "sampling_intensity": {
            "pre_gap_mean_readings_per_observed_day": float(
                np.mean(n_readings[: len(before)])
            ),
            "post_gap_mean_readings_per_observed_day": float(
                np.mean(n_readings[len(before) :])
            ),
        },
        "estimates": [estimate.__dict__ for estimate in estimates],
        "sampling_adjustment": {
            "covariate": "log1p_n_readings",
            "coefficient_mmHg": float(adjusted_params[3]),
            "ci95_low_mmHg": float(adjusted_ci[3, 0]),
            "ci95_high_mmHg": float(adjusted_ci[3, 1]),
            "p_value": float(adjusted_pvalues[3]),
        },
        "interpretation": {
            "identified_missingness_mechanism": False,
            "inverse_intensity_is_stress_test_not_ipw": True,
            "episode_split_is_gap_defined_not_estimated_from_outcomes": True,
        },
    }


def plot_episode_sensitivity(
    results: dict[str, object], output_path: Path
) -> None:
    """Plot episode-level contrasts and robust 95% confidence intervals."""

    raw = results.get("estimates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Episode sensitivity results contain no estimates.")

    labels: list[str] = []
    estimates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("Each sensitivity estimate must be a dictionary.")
        estimate = float(item["episode_difference_mmHg"])
        low = float(item["episode_ci95_low_mmHg"])
        high = float(item["episode_ci95_high_mmHg"])
        labels.append(str(item["name"]).replace("_", " "))
        estimates.append(estimate)
        lower_errors.append(estimate - low)
        upper_errors.append(high - estimate)

    positions = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.errorbar(
        estimates,
        positions,
        xerr=np.asarray([lower_errors, upper_errors]),
        fmt="o",
        capsize=4,
    )
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(positions, labels=labels)
    ax.set_xlabel("Post-minus-pre systolic episode contrast (mmHg)")
    ax.set_title("Episode contrast sensitivity to observation intensity")
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
        default=Path("figures/episode_observation_sensitivity.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("figures/episode_observation_sensitivity.svg"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the sensitivity analysis and write public derived outputs."""

    args = parse_args()
    records = primary.load_snapshot(args.data)
    results = fit_episode_observation_sensitivity(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_episode_sensitivity(results, args.output_figure)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
