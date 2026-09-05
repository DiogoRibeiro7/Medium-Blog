"""Decompose the blood-pressure time trend around the dominant missing-data gap.

The primary analysis reports a global calendar-time slope across the full public
snapshot. This module asks whether that slope is primarily a smooth within-period
trend or a contrast between observation episodes separated by the longest
unobserved interval.

The split is defined by the unique longest *internal* missing run. It is an
observation-design decomposition, not a change-point test: no measurement exists
inside the gap, so the analysis cannot identify when or why any level difference
arose there.
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

DAYS_PER_REPORTING_PERIOD: Final[float] = 30.0
MIN_EPISODE_DAYS: Final[int] = 4


@dataclass(frozen=True)
class InternalGap:
    """One missing calendar run bounded by observed days on both sides."""

    start_day_index: int
    end_day_index: int
    length_days: int
    left_observed_day_index: int
    right_observed_day_index: int


@dataclass(frozen=True)
class EpisodeFit:
    """One HC3 linear trend fitted within a single observation episode."""

    n_days: int
    mean_systolic_mmHg: float
    slope_per_30_days: float
    ci95_low_per_30_days: float
    ci95_high_per_30_days: float
    p_value: float


def _observed_systolic(record: primary.DailyRecord) -> float:
    """Return a validated systolic daily mean from an observed record."""

    if not record.observed or record.mean_systolic_mmHg is None:
        raise ValueError("Expected an observed record with a systolic mean.")
    value = float(record.mean_systolic_mmHg)
    if not math.isfinite(value):
        raise ValueError("Observed systolic means must be finite.")
    return value


def internal_missing_gaps(records: Sequence[primary.DailyRecord]) -> list[InternalGap]:
    """Return all missing runs bounded by observed records.

    Leading or trailing missing rows are not useful for an episode decomposition,
    because they do not separate two observed periods.
    """

    if not records:
        raise ValueError("At least one calendar record is required.")

    gaps: list[InternalGap] = []
    start: int | None = None
    for position, record in enumerate(records):
        if record.observed:
            if start is not None:
                end = position - 1
                left_position = start - 1
                if left_position >= 0 and records[left_position].observed:
                    gaps.append(
                        InternalGap(
                            start_day_index=records[start].day_index,
                            end_day_index=records[end].day_index,
                            length_days=end - start + 1,
                            left_observed_day_index=records[left_position].day_index,
                            right_observed_day_index=record.day_index,
                        )
                    )
                start = None
        elif start is None:
            start = position

    return gaps


def dominant_internal_gap(records: Sequence[primary.DailyRecord]) -> InternalGap:
    """Return the unique longest internal missing run.

    Raises:
        ValueError: If no internal gap exists or the longest gap is tied, because
            an arbitrary episode split would then be scientifically misleading.
    """

    gaps = internal_missing_gaps(records)
    if not gaps:
        raise ValueError("No internal missing gap separates observed episodes.")

    maximum = max(gap.length_days for gap in gaps)
    winners = [gap for gap in gaps if gap.length_days == maximum]
    if len(winners) != 1:
        raise ValueError("The longest internal missing gap is not unique.")
    return winners[0]


def split_observation_episodes(
    records: Sequence[primary.DailyRecord], gap: InternalGap
) -> tuple[list[primary.DailyRecord], list[primary.DailyRecord]]:
    """Split observed records into the episodes before and after ``gap``."""

    observed = primary.observed_records(records)
    before = [
        record for record in observed if record.day_index < gap.start_day_index
    ]
    after = [record for record in observed if record.day_index > gap.end_day_index]

    if len(before) < MIN_EPISODE_DAYS or len(after) < MIN_EPISODE_DAYS:
        raise ValueError(
            "Each observation episode requires at least "
            f"{MIN_EPISODE_DAYS} observed days."
        )
    if len(before) + len(after) != len(observed):
        raise ValueError("The dominant gap did not partition all observed days.")
    return before, after


def _fit_episode(records: Sequence[primary.DailyRecord]) -> EpisodeFit:
    """Fit one episode-specific HC3 linear trend."""

    x = np.asarray([record.day_index for record in records], dtype=float)
    y = np.asarray([_observed_systolic(record) for record in records], dtype=float)
    fitted = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    interval = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)[1]
    return EpisodeFit(
        n_days=len(records),
        mean_systolic_mmHg=float(np.mean(y)),
        slope_per_30_days=float(fitted.params[1] * DAYS_PER_REPORTING_PERIOD),
        ci95_low_per_30_days=float(interval[0] * DAYS_PER_REPORTING_PERIOD),
        ci95_high_per_30_days=float(interval[1] * DAYS_PER_REPORTING_PERIOD),
        p_value=float(fitted.pvalues[1]),
    )


def _episode_centered_model(
    before: Sequence[primary.DailyRecord], after: Sequence[primary.DailyRecord]
) -> dict[str, float]:
    """Estimate an episode level contrast and a common within-episode slope.

    Calendar time is centered separately within each episode. Consequently the
    episode indicator compares episode mean levels, while the time coefficient is
    identified only from within-episode time variation.
    """

    combined = [*before, *after]
    y = np.asarray([_observed_systolic(record) for record in combined], dtype=float)
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

    design = sm.add_constant(np.column_stack([episode, within_time]))
    fitted = sm.OLS(y, design).fit(cov_type="HC3")
    interval = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)

    return {
        "pre_episode_mean_systolic_mmHg": float(fitted.params[0]),
        "post_minus_pre_mean_difference_mmHg": float(fitted.params[1]),
        "post_minus_pre_ci95_low_mmHg": float(interval[1, 0]),
        "post_minus_pre_ci95_high_mmHg": float(interval[1, 1]),
        "post_minus_pre_p_value": float(fitted.pvalues[1]),
        "common_within_episode_slope_per_30_days": float(
            fitted.params[2] * DAYS_PER_REPORTING_PERIOD
        ),
        "within_slope_ci95_low_per_30_days": float(
            interval[2, 0] * DAYS_PER_REPORTING_PERIOD
        ),
        "within_slope_ci95_high_per_30_days": float(
            interval[2, 1] * DAYS_PER_REPORTING_PERIOD
        ),
        "within_slope_p_value": float(fitted.pvalues[2]),
    }


def _global_slope_decomposition(
    before: Sequence[primary.DailyRecord], after: Sequence[primary.DailyRecord]
) -> dict[str, float | None]:
    """Exactly decompose the global OLS slope into within/between covariance terms.

    For episodes ``g``, total time-pressure cross-deviation satisfies

    ``S_xy = S_xy_within + S_xy_between``.

    Dividing both components by the *global* ``S_xx`` yields two additive
    contributions that sum exactly to the global OLS slope. This is a numerical
    decomposition of the global estimator, not a causal attribution.
    """

    groups = [list(before), list(after)]
    combined = [record for group in groups for record in group]
    x = np.asarray([record.day_index for record in combined], dtype=float)
    y = np.asarray([_observed_systolic(record) for record in combined], dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    total_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    total_xx = float(np.sum((x - x_mean) ** 2))
    if total_xx <= 0.0:
        raise ValueError("Global calendar time has no variation.")

    within_xy = 0.0
    between_xy = 0.0
    for group in groups:
        group_x = np.asarray([record.day_index for record in group], dtype=float)
        group_y = np.asarray(
            [_observed_systolic(record) for record in group], dtype=float
        )
        group_x_mean = float(np.mean(group_x))
        group_y_mean = float(np.mean(group_y))
        within_xy += float(
            np.sum((group_x - group_x_mean) * (group_y - group_y_mean))
        )
        between_xy += float(
            len(group)
            * (group_x_mean - x_mean)
            * (group_y_mean - y_mean)
        )

    global_slope = total_xy / total_xx
    within_contribution = within_xy / total_xx
    between_contribution = between_xy / total_xx
    if not math.isclose(
        global_slope,
        within_contribution + between_contribution,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Slope decomposition failed its additive identity.")

    covariance_fraction = (
        between_xy / total_xy if not math.isclose(total_xy, 0.0, abs_tol=1e-15) else None
    )
    return {
        "global_ols_slope_per_30_days": global_slope
        * DAYS_PER_REPORTING_PERIOD,
        "within_episode_contribution_per_30_days": within_contribution
        * DAYS_PER_REPORTING_PERIOD,
        "between_episode_contribution_per_30_days": between_contribution
        * DAYS_PER_REPORTING_PERIOD,
        "additive_sum_per_30_days": (
            within_contribution + between_contribution
        )
        * DAYS_PER_REPORTING_PERIOD,
        "between_fraction_of_time_pressure_covariance": covariance_fraction,
    }


def analyze_gap_aware_trend(
    records: Sequence[primary.DailyRecord],
) -> dict[str, object]:
    """Run the complete dominant-gap episode decomposition."""

    gap = dominant_internal_gap(records)
    before, after = split_observation_episodes(records, gap)
    global_trend = primary.linear_trend(records, "mean_systolic_mmHg")
    centered = _episode_centered_model(before, after)
    decomposition = _global_slope_decomposition(before, after)

    if not math.isclose(
        float(global_trend["slope_per_30_days"]),
        float(decomposition["global_ols_slope_per_30_days"]),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Gap-aware and primary global slopes disagree.")

    return {
        "dominant_internal_gap": gap.__dict__,
        "n_observed_days": len(before) + len(after),
        "pre_gap_episode": _fit_episode(before).__dict__,
        "post_gap_episode": _fit_episode(after).__dict__,
        "episode_centered_model": centered,
        "global_trend": global_trend,
        "exact_global_slope_decomposition": decomposition,
        "interpretation": {
            "is_change_point_test": False,
            "transition_inside_gap_is_observed": False,
            "can_identify_when_level_difference_arose": False,
            "purpose": "separate_global_trend_into_within_and_between_episode_structure",
        },
    }


def plot_gap_aware_trend(
    records: Sequence[primary.DailyRecord],
    results: dict[str, object],
    output_path: Path,
) -> None:
    """Plot observed daily systolic means and separate episode-specific trends."""

    gap_payload = results.get("dominant_internal_gap")
    if not isinstance(gap_payload, dict):
        raise TypeError("dominant_internal_gap must be a dictionary.")
    gap = InternalGap(**{key: int(value) for key, value in gap_payload.items()})
    before, after = split_observation_episodes(records, gap)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for label, episode in (("Pre-gap episode", before), ("Post-gap episode", after)):
        x = np.asarray([record.day_index for record in episode], dtype=float)
        y = np.asarray([_observed_systolic(record) for record in episode], dtype=float)
        fitted = sm.OLS(y, sm.add_constant(x)).fit()
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.scatter(x, y, label=f"{label} observations")
        ax.plot(grid, fitted.params[0] + fitted.params[1] * grid, label=f"{label} trend")

    ax.axvspan(
        gap.start_day_index,
        gap.end_day_index,
        alpha=0.15,
        label=f"{gap.length_days}-day unobserved gap",
    )
    ax.set_xlabel("Day index from first observation")
    ax.set_ylabel("Daily mean systolic pressure (mmHg)")
    ax.set_title("Global trend spans two observation episodes separated by a long gap")
    ax.legend()
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
        default=Path("figures/gap_aware_trend_decomposition.json"),
        help="Machine-readable gap-aware results.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("figures/gap_aware_trend_decomposition.svg"),
        help="Gap-aware episode figure.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the decomposition and write public derived outputs."""

    args = parse_args()
    records = primary.load_snapshot(args.data)
    results = analyze_gap_aware_trend(records)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_gap_aware_trend(records, results, args.output_figure)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
