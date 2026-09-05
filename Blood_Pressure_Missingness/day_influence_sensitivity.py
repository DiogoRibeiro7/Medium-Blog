"""Leave-one-observed-day-out influence analysis for blood-pressure trends.

This module quantifies how strongly any single observed calendar day can move the
main systolic conclusions. It complements HC3 covariance by perturbing the data
itself: each observed day is removed once and the global trend plus the gap-aware
episode model are refitted.

The analysis is descriptive robustness checking, not a causal procedure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

import analysis as primary
import gap_aware_trend_decomposition as gap_aware


def _global_trend(records: Sequence[primary.DailyRecord]) -> dict[str, float]:
    """Return the primary HC3 systolic trend for one record set."""

    return primary.linear_trend(records, "mean_systolic_mmHg")


def _episode_model(records: Sequence[primary.DailyRecord]) -> dict[str, float]:
    """Return the gap-aware centered episode model for one record set."""

    gap = gap_aware.dominant_internal_gap(records)
    before, after = gap_aware.split_observation_episodes(records, gap)
    return gap_aware._episode_centered_model(before, after)


def _ordinary_ols_influence(
    records: Sequence[primary.DailyRecord],
) -> list[dict[str, float | int]]:
    """Return standard OLS influence diagnostics for observed systolic days."""

    observed = primary.observed_records(records)
    x = np.asarray([record.day_index for record in observed], dtype=float)
    y = np.asarray(
        [float(record.mean_systolic_mmHg) for record in observed], dtype=float
    )
    fitted = sm.OLS(y, sm.add_constant(x)).fit()
    influence = fitted.get_influence()
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    dfbetas = np.asarray(influence.dfbetas, dtype=float)

    return [
        {
            "day_index": record.day_index,
            "mean_systolic_mmHg": float(record.mean_systolic_mmHg),
            "cooks_distance": float(cooks[index]),
            "leverage": float(leverage[index]),
            "dfbeta_slope": float(dfbetas[index, 1]),
        }
        for index, record in enumerate(observed)
    ]


def leave_one_day_out(
    records: Sequence[primary.DailyRecord],
) -> list[dict[str, float | int | bool]]:
    """Remove each observed day once and refit the main systolic estimands."""

    observed = primary.observed_records(records)
    if len(observed) < 6:
        raise ValueError("At least six observed days are required for jackknife checks.")

    results: list[dict[str, float | int | bool]] = []
    for removed in observed:
        retained = [
            record
            for record in records
            if record.day_index != removed.day_index
        ]
        global_fit = _global_trend(retained)
        episode_fit = _episode_model(retained)
        results.append(
            {
                "removed_day_index": removed.day_index,
                "removed_mean_systolic_mmHg": float(removed.mean_systolic_mmHg),
                "global_slope_per_30_days": float(global_fit["slope_per_30_days"]),
                "global_ci95_low_per_30_days": float(
                    global_fit["ci95_low_per_30_days"]
                ),
                "global_ci95_high_per_30_days": float(
                    global_fit["ci95_high_per_30_days"]
                ),
                "episode_level_difference_mmHg": float(
                    episode_fit["post_minus_pre_mean_difference_mmHg"]
                ),
                "episode_level_ci95_low_mmHg": float(
                    episode_fit["post_minus_pre_ci95_low_mmHg"]
                ),
                "episode_level_ci95_high_mmHg": float(
                    episode_fit["post_minus_pre_ci95_high_mmHg"]
                ),
                "within_episode_slope_per_30_days": float(
                    episode_fit["common_within_episode_slope_per_30_days"]
                ),
                "within_episode_ci95_low_per_30_days": float(
                    episode_fit["within_slope_ci95_low_per_30_days"]
                ),
                "within_episode_ci95_high_per_30_days": float(
                    episode_fit["within_slope_ci95_high_per_30_days"]
                ),
                "global_interval_excludes_zero": float(
                    global_fit["ci95_high_per_30_days"]
                ) < 0.0,
                "episode_level_interval_excludes_zero": float(
                    episode_fit["post_minus_pre_ci95_high_mmHg"]
                ) < 0.0,
                "within_episode_interval_excludes_zero": float(
                    episode_fit["within_slope_ci95_high_per_30_days"]
                ) < 0.0,
            }
        )
    return results


def summarize_influence(records: Sequence[primary.DailyRecord]) -> dict[str, object]:
    """Build machine-readable influence and jackknife summaries."""

    baseline_global = _global_trend(records)
    baseline_episode = _episode_model(records)
    deletions = leave_one_day_out(records)
    diagnostics = _ordinary_ols_influence(records)

    global_slopes = np.asarray(
        [float(item["global_slope_per_30_days"]) for item in deletions]
    )
    episode_differences = np.asarray(
        [float(item["episode_level_difference_mmHg"]) for item in deletions]
    )
    within_slopes = np.asarray(
        [float(item["within_episode_slope_per_30_days"]) for item in deletions]
    )

    max_cook = max(diagnostics, key=lambda item: float(item["cooks_distance"]))
    max_abs_dfbeta = max(
        diagnostics, key=lambda item: abs(float(item["dfbeta_slope"]))
    )

    return {
        "n_observed_days": len(primary.observed_records(records)),
        "baseline": {
            "global_slope_per_30_days": float(
                baseline_global["slope_per_30_days"]
            ),
            "episode_level_difference_mmHg": float(
                baseline_episode["post_minus_pre_mean_difference_mmHg"]
            ),
            "within_episode_slope_per_30_days": float(
                baseline_episode["common_within_episode_slope_per_30_days"]
            ),
        },
        "leave_one_day_out_summary": {
            "global_slope_min_per_30_days": float(np.min(global_slopes)),
            "global_slope_max_per_30_days": float(np.max(global_slopes)),
            "global_interval_excludes_zero_for_all_deletions": all(
                bool(item["global_interval_excludes_zero"]) for item in deletions
            ),
            "episode_level_difference_min_mmHg": float(
                np.min(episode_differences)
            ),
            "episode_level_difference_max_mmHg": float(
                np.max(episode_differences)
            ),
            "episode_level_interval_excludes_zero_for_all_deletions": all(
                bool(item["episode_level_interval_excludes_zero"])
                for item in deletions
            ),
            "within_episode_slope_min_per_30_days": float(
                np.min(within_slopes)
            ),
            "within_episode_slope_max_per_30_days": float(
                np.max(within_slopes)
            ),
            "within_episode_significant_negative_deletions": [
                int(item["removed_day_index"])
                for item in deletions
                if bool(item["within_episode_interval_excludes_zero"])
            ],
        },
        "most_influential_by_cooks_distance": max_cook,
        "largest_absolute_slope_dfbeta": max_abs_dfbeta,
        "deletions": deletions,
        "ordinary_ols_influence": diagnostics,
        "interpretation": {
            "single_day_deletion_is_causal_test": False,
            "purpose": "assess_small_sample_sensitivity_to_individual_observed_days",
        },
    }


def plot_leave_one_day_out(results: dict[str, object], output_path: Path) -> None:
    """Plot the global systolic slope after deleting each observed day."""

    deletions = results.get("deletions")
    baseline = results.get("baseline")
    if not isinstance(deletions, list) or not isinstance(baseline, dict):
        raise TypeError("Influence results are malformed.")

    days = np.asarray([int(item["removed_day_index"]) for item in deletions])
    slopes = np.asarray(
        [float(item["global_slope_per_30_days"]) for item in deletions]
    )
    low = np.asarray(
        [float(item["global_ci95_low_per_30_days"]) for item in deletions]
    )
    high = np.asarray(
        [float(item["global_ci95_high_per_30_days"]) for item in deletions]
    )
    baseline_slope = float(baseline["global_slope_per_30_days"])

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.errorbar(
        days,
        slopes,
        yerr=np.asarray([slopes - low, high - slopes]),
        fmt="o",
        capsize=3,
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(baseline_slope, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Removed observed day index")
    ax.set_ylabel("Systolic slope per 30 days (mmHg)")
    ax.set_title("Leave-one-observed-day-out global systolic trend")
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
        default=Path("figures/day_influence_sensitivity.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("figures/day_influence_sensitivity.svg"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the influence analysis and write public derived outputs."""

    args = parse_args()
    records = primary.load_snapshot(args.data)
    results = summarize_influence(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    plot_leave_one_day_out(results, args.output_figure)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
