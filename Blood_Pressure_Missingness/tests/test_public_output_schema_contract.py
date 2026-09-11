"""Structural contract for committed public blood-pressure outputs."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

SNAPSHOT_COLUMNS = [
    "day_index",
    "observed",
    "n_readings",
    "n_sessions",
    "mean_systolic_mmHg",
    "mean_diastolic_mmHg",
    "mean_pulse_pressure_mmHg",
    "mean_bpm",
]

METRIC_NAMES = {
    "mean_systolic_mmHg",
    "mean_diastolic_mmHg",
    "mean_pulse_pressure_mmHg",
    "mean_bpm",
}

REQUIRED_JSON_OUTPUTS = {
    "results.json",
    "observation_process_sensitivity.json",
    "gap_aware_trend_decomposition.json",
    "day_influence_sensitivity.json",
    "episode_observation_sensitivity.json",
    "episode_time_form_sensitivity.json",
    "temporal_dependence_diagnostics.json",
}

OPTIONAL_JSON_OUTPUTS = {
    "temperature_covariate.json",
    "temperature_covariate_sensitivity.json",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path.name} must contain one JSON object")
    return payload


def _keys(test: unittest.TestCase, value: Any, expected: set[str]) -> None:
    test.assertIsInstance(value, dict)
    test.assertEqual(set(value), expected)


class PublicOutputSchemaContractTests(unittest.TestCase):
    """Freeze public interface shape without freezing numerical results."""

    def test_snapshot_column_order_is_stable(self) -> None:
        with (DATA_DIR / "analysis_snapshot.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            self.assertEqual(next(reader), SNAPSHOT_COLUMNS)

    def test_required_json_output_set_is_present(self) -> None:
        present = {path.name for path in FIGURES_DIR.glob("*.json")}
        self.assertTrue(REQUIRED_JSON_OUTPUTS <= present)
        self.assertTrue(present <= REQUIRED_JSON_OUTPUTS | OPTIONAL_JSON_OUTPUTS)

    def test_results_schema(self) -> None:
        payload = _load(FIGURES_DIR / "results.json")
        _keys(self, payload, {"source_audit", "calendar", "sampling", "metrics"})
        self.assertEqual(set(payload["metrics"]), METRIC_NAMES)
        for metric in payload["metrics"].values():
            _keys(self, metric, {"means", "sampling_association", "linear_trend"})
            _keys(
                self,
                metric["means"],
                {
                    "reading_weighted_mean",
                    "equal_observed_day_mean",
                    "difference_weighted_minus_equal_day",
                },
            )
            _keys(
                self,
                metric["sampling_association"],
                {"pearson_r", "pearson_p_value", "spearman_rho", "spearman_p_value"},
            )
            _keys(
                self,
                metric["linear_trend"],
                {
                    "slope_per_day",
                    "slope_per_30_days",
                    "ci95_low_per_30_days",
                    "ci95_high_per_30_days",
                    "p_value",
                    "r_squared",
                },
            )

    def test_observation_process_schema(self) -> None:
        payload = _load(FIGURES_DIR / "observation_process_sensitivity.json")
        _keys(self, payload, {"n_observed_days", "trend_estimates", "sampling_adjustment", "interpretation"})
        self.assertIsInstance(payload["trend_estimates"], list)
        self.assertTrue(payload["trend_estimates"])
        expected = {"name", "slope_per_30_days", "ci95_low_per_30_days", "ci95_high_per_30_days", "p_value"}
        for item in payload["trend_estimates"]:
            _keys(self, item, expected)

    def test_gap_aware_schema(self) -> None:
        payload = _load(FIGURES_DIR / "gap_aware_trend_decomposition.json")
        _keys(
            self,
            payload,
            {
                "dominant_internal_gap",
                "n_observed_days",
                "pre_gap_episode",
                "post_gap_episode",
                "episode_centered_model",
                "global_trend",
                "exact_global_slope_decomposition",
                "interpretation",
            },
        )

    def test_day_influence_schema(self) -> None:
        payload = _load(FIGURES_DIR / "day_influence_sensitivity.json")
        _keys(
            self,
            payload,
            {
                "n_observed_days",
                "frozen_episode_definition",
                "baseline",
                "leave_one_day_out_summary",
                "most_influential_by_cooks_distance",
                "largest_absolute_slope_dfbeta",
                "deletions",
            },
        )
        self.assertIsInstance(payload["deletions"], list)
        self.assertTrue(payload["deletions"])
        expected = {
            "removed_day_index",
            "removed_mean_systolic_mmHg",
            "global_slope_per_30_days",
            "global_ci95_low_per_30_days",
            "global_ci95_high_per_30_days",
            "episode_level_difference_mmHg",
            "episode_level_ci95_low_mmHg",
            "episode_level_ci95_high_mmHg",
            "within_episode_slope_per_30_days",
            "within_episode_ci95_low_per_30_days",
            "within_episode_ci95_high_per_30_days",
            "global_interval_excludes_zero",
            "episode_level_interval_excludes_zero",
            "within_episode_interval_excludes_zero",
        }
        for item in payload["deletions"]:
            _keys(self, item, expected)

    def test_episode_observation_schema(self) -> None:
        payload = _load(FIGURES_DIR / "episode_observation_sensitivity.json")
        _keys(
            self,
            payload,
            {
                "dominant_internal_gap",
                "n_observed_days",
                "episode_sizes",
                "sampling_intensity",
                "estimates",
                "sampling_adjustment",
                "interpretation",
            },
        )
        expected = {
            "name",
            "episode_difference_mmHg",
            "episode_ci95_low_mmHg",
            "episode_ci95_high_mmHg",
            "episode_p_value",
            "within_episode_slope_per_30_days",
            "within_slope_ci95_low_per_30_days",
            "within_slope_ci95_high_per_30_days",
            "within_slope_p_value",
        }
        for item in payload["estimates"]:
            _keys(self, item, expected)

    def test_episode_time_form_schema(self) -> None:
        payload = _load(FIGURES_DIR / "episode_time_form_sensitivity.json")
        _keys(self, payload, {"dominant_internal_gap", "n_observed_days", "episode_sizes", "estimates", "interpretation"})
        expected = {
            "name",
            "n_parameters",
            "episode_difference_mmHg",
            "episode_ci95_low_mmHg",
            "episode_ci95_high_mmHg",
            "episode_p_value",
            "aic",
            "bic",
        }
        for item in payload["estimates"]:
            _keys(self, item, expected)

    def test_temporal_dependence_schema(self) -> None:
        payload = _load(FIGURES_DIR / "temporal_dependence_diagnostics.json")
        _keys(self, payload, {"n_observed_days", "residual_model", "observed_order_spacing", "exact_calendar_lag_residual_correlations", "interpretation"})
        expected = {"lag_days", "n_pairs", "pearson_r"}
        for item in payload["exact_calendar_lag_residual_correlations"]:
            _keys(self, item, expected)

    def test_temperature_schema_when_committed(self) -> None:
        path = FIGURES_DIR / "temperature_covariate.json"
        if path.exists():
            payload = _load(path)
            _keys(self, payload, {"weather_source", "matching", "temperature_summary", "model", "privacy"})
            _keys(self, payload["model"], {"formula", "day_weighting", "within_day_temperature", "covariance", "metrics"})
            self.assertEqual(set(payload["model"]["metrics"]), METRIC_NAMES)

        sensitivity_path = FIGURES_DIR / "temperature_covariate_sensitivity.json"
        if sensitivity_path.exists():
            payload = _load(sensitivity_path)
            _keys(self, payload, {"purpose", "specifications", "metrics", "privacy"})
            self.assertEqual(set(payload["metrics"]), METRIC_NAMES)
            expected = {
                "primary_nearest_hour_equal_day_linear",
                "linear_interpolation_equal_day_linear",
                "nearest_hour_reading_weighted_linear",
                "nearest_hour_equal_day_quadratic",
                "robustness",
            }
            for metric in payload["metrics"].values():
                _keys(self, metric, expected)


if __name__ == "__main__":
    unittest.main()
