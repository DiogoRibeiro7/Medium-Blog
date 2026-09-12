"""Sensitivity of the gap-defined episode contrast to within-episode time form.

The gap-aware analysis estimates a post-minus-pre systolic level contrast while
adjusting for a common linear trend within the two observation episodes. This
module asks whether that contrast depends materially on that specific functional
form.

The richer specifications are sensitivity checks, not preferred models. With only
25 observed calendar days, and only eight in the pre-gap episode, the most flexible
model is deliberately treated as a stress specification rather than evidence that
a quadratic trajectory is scientifically correct.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

import analysis as primary
import gap_aware_trend_decomposition as gap_aware


@dataclass(frozen=True)
class TimeFormEstimate:
    """One HC3 episode-contrast estimate under a time-form specification."""

    name: str
    n_parameters: int
    episode_difference_mmHg: float
    episode_ci95_low_mmHg: float
    episode_ci95_high_mmHg: float
    episode_p_value: float
    aic: float
    bic: float


def _episode_arrays(
    records: Sequence[primary.DailyRecord],
) -> tuple[
    gap_aware.InternalGap,
    list[primary.DailyRecord],
    list[primary.DailyRecord],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return the fixed gap split and episode-centered regression arrays."""

    gap = gap_aware.dominant_internal_gap(records)
    before, after = gap_aware.split_observation_episodes(records, gap)
    combined = [*before, *after]

    y = np.asarray(
        [float(record.mean_systolic_mmHg) for record in combined], dtype=float
    )
    if np.any(~np.isfinite(y)):
        raise ValueError("Observed systolic means must be finite.")

    episode = np.asarray(
        [0.0] * len(before) + [1.0] * len(after), dtype=float
    )
    before_mean_day = float(np.mean([record.day_index for record in before]))
    after_mean_day = float(np.mean([record.day_index for record in after]))
    within_time = np.asarray(
        [record.day_index - before_mean_day for record in before]
        + [record.day_index - after_mean_day for record in after],
        dtype=float,
    )
    return gap, before, after, y, episode, within_time


def _fit(
    name: str,
    y: np.ndarray,
    columns: Sequence[np.ndarray],
) -> TimeFormEstimate:
    """Fit one HC3 OLS model and extract the episode coefficient."""

    design = sm.add_constant(np.column_stack(columns))
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"Design matrix for {name!r} is rank deficient.")

    fitted = sm.OLS(y, design).fit(cov_type="HC3")
    interval = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
    return TimeFormEstimate(
        name=name,
        n_parameters=int(design.shape[1]),
        episode_difference_mmHg=float(fitted.params[1]),
        episode_ci95_low_mmHg=float(interval[1, 0]),
        episode_ci95_high_mmHg=float(interval[1, 1]),
        episode_p_value=float(fitted.pvalues[1]),
        aic=float(fitted.aic),
        bic=float(fitted.bic),
    )


def fit_episode_time_form_sensitivity(
    records: Sequence[primary.DailyRecord],
) -> dict[str, object]:
    """Fit ordinary and stress time-form alternatives around the episode split."""

    gap, before, after, y, episode, within_time = _episode_arrays(records)
    episode_within = episode * within_time
    within_squared = within_time**2

    estimates = [
        _fit("episode_only", y, [episode]),
        _fit("common_linear", y, [episode, within_time]),
        _fit(
            "separate_linear_slopes",
            y,
            [episode, within_time, episode_within],
        ),
        _fit(
            "common_quadratic",
            y,
            [episode, within_time, within_squared],
        ),
        _fit(
            "separate_linear_plus_common_quadratic_stress",
            y,
            [episode, within_time, episode_within, within_squared],
        ),
    ]

    return {
        "dominant_internal_gap": gap.__dict__,
        "n_observed_days": len(before) + len(after),
        "episode_sizes": {
            "pre_gap_observed_days": len(before),
            "post_gap_observed_days": len(after),
        },
        "estimates": [estimate.__dict__ for estimate in estimates],
        "interpretation": {
            "preferred_model_selected_by_this_analysis": False,
            "stress_model_is_claimed_true_trajectory": False,
            "common_linear_matches_gap_aware_primary_estimand": True,
            "purpose": "test_episode_contrast_sensitivity_to_within_episode_time_form",
        },
    }


def plot_time_form_sensitivity(
    results: dict[str, object], output_path: Path
) -> None:
    """Plot episode contrasts and robust 95% intervals across specifications."""

    raw = results.get("estimates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Time-form sensitivity results contain no estimates.")

    labels: list[str] = []
    estimates: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("Each time-form estimate must be a dictionary.")
        estimate = float(item["episode_difference_mmHg"])
        low = float(item["episode_ci95_low_mmHg"])
        high = float(item["episode_ci95_high_mmHg"])
        labels.append(str(item["name"]).replace("_", " "))
        estimates.append(estimate)
        lower.append(estimate - low)
        upper.append(high - estimate)

    positions = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.errorbar(
        estimates,
        positions,
        xerr=np.asarray([lower, upper]),
        fmt="o",
        capsize=4,
    )
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(positions, labels=labels)
    ax.set_xlabel("Post-minus-pre systolic episode contrast (mmHg)")
    ax.set_title("Episode contrast sensitivity to within-episode time form")
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
        default=Path("figures/episode_time_form_sensitivity.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("figures/episode_time_form_sensitivity.svg"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the time-form sensitivity analysis and write derived outputs."""

    args = parse_args()
    records = primary.load_snapshot(args.data)
    results = fit_episode_time_form_sensitivity(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_time_form_sensitivity(results, args.output_figure)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
