"""Diagnose short-lag residual dependence without erasing calendar gaps.

Standard row-order autocorrelation calculations implicitly treat consecutive rows
as equally spaced. That is not appropriate for this tracker: consecutive observed
days can be separated by one day, several days, or the dominant missing interval.

This module therefore fits the established gap-aware common-linear systolic model
and summarizes residual association by *actual calendar-day distance*. The output
is deliberately descriptive. Pair counts are reported and no formal claim of
serial independence or a valid Newey-West/HAC correction is made from this small,
irregular sample.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import statsmodels.api as sm

import analysis as primary
import gap_aware_trend_decomposition as gap_aware

MAX_EXACT_LAG_DAYS: Final[int] = 7
MIN_PAIRS_FOR_CORRELATION: Final[int] = 3


@dataclass(frozen=True)
class ResidualPoint:
    """One observed day and its residual from the gap-aware common-linear model."""

    day_index: int
    episode: int
    residual: float


def _gap_aware_residual_points(
    records: Sequence[primary.DailyRecord],
) -> tuple[list[ResidualPoint], dict[str, float]]:
    """Fit the established gap-aware model and return observed-day residuals."""

    gap = gap_aware.dominant_internal_gap(records)
    before, after = gap_aware.split_observation_episodes(records, gap)
    combined = [*before, *after]

    y = np.asarray(
        [gap_aware._observed_systolic(record) for record in combined], dtype=float
    )
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
    fitted = sm.OLS(y, design).fit()
    established = gap_aware._episode_centered_model(before, after)

    expected = np.asarray(
        [
            float(established["pre_episode_mean_systolic_mmHg"]),
            float(established["post_minus_pre_mean_difference_mmHg"]),
            float(established["common_within_episode_slope_per_30_days"])
            / gap_aware.DAYS_PER_REPORTING_PERIOD,
        ],
        dtype=float,
    )
    if not np.allclose(fitted.params, expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "Temporal-dependence residual model does not match the established "
            "gap-aware common-linear estimand."
        )

    points = [
        ResidualPoint(
            day_index=record.day_index,
            episode=int(episode[position]),
            residual=float(fitted.resid[position]),
        )
        for position, record in enumerate(combined)
    ]
    return points, established


def _pearson_correlation(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Return Pearson correlation when enough non-degenerate pairs exist."""

    if len(pairs) < MIN_PAIRS_FOR_CORRELATION:
        return None
    left = np.asarray([pair[0] for pair in pairs], dtype=float)
    right = np.asarray([pair[1] for pair in pairs], dtype=float)
    if math.isclose(float(np.std(left)), 0.0, abs_tol=1e-15) or math.isclose(
        float(np.std(right)), 0.0, abs_tol=1e-15
    ):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _observed_order_spacing(points: Sequence[ResidualPoint]) -> dict[str, object]:
    """Summarize calendar distances hidden by observed-row ordering."""

    if len(points) < 2:
        raise ValueError("At least two observed residual points are required.")

    calendar_gaps = [
        right.day_index - left.day_index
        for left, right in zip(points[:-1], points[1:])
    ]
    if any(gap <= 0 for gap in calendar_gaps):
        raise ValueError("Observed residual points must be strictly time ordered.")

    counts = Counter(calendar_gaps)
    row_order_pairs = [
        (left.residual, right.residual)
        for left, right in zip(points[:-1], points[1:])
    ]
    unit_pairs = int(counts.get(1, 0))
    return {
        "n_consecutive_observed_pairs": len(calendar_gaps),
        "calendar_gap_days_counts": {
            str(gap): int(counts[gap]) for gap in sorted(counts)
        },
        "n_exactly_one_day_apart": unit_pairs,
        "n_not_one_day_apart": len(calendar_gaps) - unit_pairs,
        "observed_order_is_equally_spaced_daily": len(counts) == 1 and 1 in counts,
        "row_order_lag1_residual_correlation": _pearson_correlation(
            row_order_pairs
        ),
    }


def _exact_calendar_lag_summary(
    points: Sequence[ResidualPoint], lag_days: int
) -> dict[str, object]:
    """Summarize residual pairs exactly ``lag_days`` apart within one episode."""

    if lag_days <= 0:
        raise ValueError("Calendar lag must be positive.")

    by_key = {(point.episode, point.day_index): point for point in points}
    pairs: list[tuple[float, float]] = []
    for point in points:
        later = by_key.get((point.episode, point.day_index + lag_days))
        if later is not None:
            pairs.append((point.residual, later.residual))

    return {
        "lag_days": lag_days,
        "n_pairs": len(pairs),
        "pearson_r": _pearson_correlation(pairs),
    }


def diagnose_temporal_dependence(
    records: Sequence[primary.DailyRecord],
    max_lag_days: int = MAX_EXACT_LAG_DAYS,
) -> dict[str, object]:
    """Return calendar-gap-aware residual-dependence diagnostics."""

    if max_lag_days <= 0:
        raise ValueError("max_lag_days must be positive.")

    points, established = _gap_aware_residual_points(records)
    spacing = _observed_order_spacing(points)
    exact_lags = [
        _exact_calendar_lag_summary(points, lag)
        for lag in range(1, max_lag_days + 1)
    ]

    return {
        "n_observed_days": len(points),
        "residual_model": {
            "name": "gap_aware_common_linear",
            "post_minus_pre_mean_difference_mmHg": float(
                established["post_minus_pre_mean_difference_mmHg"]
            ),
            "common_within_episode_slope_per_30_days": float(
                established["common_within_episode_slope_per_30_days"]
            ),
        },
        "observed_order_spacing": spacing,
        "exact_calendar_lag_residual_correlations": exact_lags,
        "interpretation": {
            "calendar_distance_preserved": True,
            "exact_lag_pairs_restricted_to_same_gap_defined_episode": True,
            "row_order_lag1_is_valid_hac_time_lag": bool(
                spacing["observed_order_is_equally_spaced_daily"]
            ),
            "formal_serial_independence_claimed": False,
            "formal_autocorrelation_test_reported": False,
            "purpose": "describe_residual_dependence_without_collapsing_calendar_gaps",
        },
    }


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
        default=Path("figures/temporal_dependence_diagnostics.json"),
        help="Aggregate diagnostic JSON output.",
    )
    parser.add_argument(
        "--max-lag-days",
        type=int,
        default=MAX_EXACT_LAG_DAYS,
        help="Largest exact calendar lag to summarize.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the diagnostic and write aggregate output."""

    args = parse_args()
    records = primary.load_snapshot(args.data)
    results = diagnose_temporal_dependence(records, max_lag_days=args.max_lag_days)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
