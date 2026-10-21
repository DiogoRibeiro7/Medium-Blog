"""Regression checks for chronological calibration and paired drift streams."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from temporal_drift import (
    generate_stream, evaluate_stream, run_temporal_experiment,
    aggregate_temporal_coverage, aggregate_temporal_decisions,
)


def test_stable_and_drift_share_prebreakpoint_observations() -> None:
    stable = generate_stream(7, "stable", n_fit=80, n_cal=30, n_deploy=40)
    changed = generate_stream(7, "concept_shift", n_fit=80, n_cal=30, n_deploy=40)
    np.testing.assert_array_equal(stable.x_fit, changed.x_fit)
    np.testing.assert_array_equal(stable.y_cal, changed.y_cal)
    np.testing.assert_array_equal(stable.y_deploy[:20], changed.y_deploy[:20])
    np.testing.assert_array_equal(stable.x_deploy, changed.x_deploy)
    assert np.any(stable.mean_deploy[20:] != changed.mean_deploy[20:])


def test_concept_and_noise_drifts_are_distinct() -> None:
    stable = generate_stream(4, "stable", n_fit=80, n_cal=30, n_deploy=40)
    noise = generate_stream(4, "noise_shift", n_fit=80, n_cal=30, n_deploy=40)
    concept = generate_stream(4, "concept_shift", n_fit=80, n_cal=30, n_deploy=40)
    np.testing.assert_array_equal(stable.mean_deploy, noise.mean_deploy)
    np.testing.assert_array_equal(stable.x_deploy, concept.x_deploy)
    assert np.any(noise.y_deploy[20:] != stable.y_deploy[20:])


def test_no_current_outcome_leakage_into_current_predictions() -> None:
    data = generate_stream(11, "stable", n_fit=80, n_cal=30, n_deploy=40)
    rows, _ = evaluate_stream(data, rolling_window=20, refit_every=10, refit_window=64)
    modified = replace(data, y_deploy=np.concatenate(([999.0], data.y_deploy[1:])))
    other, _ = evaluate_stream(modified, rolling_window=20, refit_every=10, refit_window=64)
    first = rows.loc[rows["time"] == 0, ["policy", "prediction", "radius"]].reset_index(drop=True)
    altered = other.loc[other["time"] == 0, ["policy", "prediction", "radius"]].reset_index(drop=True)
    assert first.equals(altered)
    # Once the altered outcome has become historical training data, a
    # subsequent refit may change predictions; t=0 must never change.
    first_refit = rows.loc[
        (rows["time"] == 10) & (rows["policy"] == "Refit/rolling"), "prediction"
    ].iloc[0]
    altered_refit = other.loc[
        (other["time"] == 10) & (other["policy"] == "Refit/rolling"), "prediction"
    ].iloc[0]
    assert first_refit != pytest.approx(altered_refit)


def test_smoke_end_to_end_regret_and_coverage() -> None:
    intervals, decisions = run_temporal_experiment(
        repeats=1, n_fit=80, n_cal=30, n_deploy=40,
        rolling_window=20, refit_every=10, refit_window=64,
        scenarios=("low_variance_signal",), regimes=("stable", "concept_shift"),
    )
    assert len(intervals) == 2 * 40 * 3
    assert len(decisions) == 2 * 40 * 3 * 4 * 2
    assert intervals["covered"].between(0, 1).all()
    assert (decisions["regret"] >= -1e-10).all()
    coverage = aggregate_temporal_coverage(intervals, bootstrap_draws=10)
    assert coverage["bootstrap_lower_95"].isna().all()
    economic = aggregate_temporal_decisions(decisions)
    assert not economic.empty
