"""Scientific invariants for delayed-feedback prequential evaluation."""

from dataclasses import replace

import numpy as np
import pytest

from gradual_drift import (
    available_label_count,
    generate_delayed_stream,
    evaluate_delayed_stream,
    run_gradual_drift_study,
    aggregate_study,
    paired_delay_penalty,
)


def test_feedback_release_boundary() -> None:
    assert [available_label_count(t, 2) for t in range(6)] == [0, 0, 0, 1, 2, 3]
    assert [available_label_count(t, 0) for t in range(4)] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        available_label_count(2, -1)


def test_regimes_share_seeded_iid_reference() -> None:
    stable = generate_delayed_stream(4, "stable", n_fit=80, n_cal=30, n_live=45)
    drift = generate_delayed_stream(4, "gradual_drift", n_fit=80, n_cal=30, n_live=45)
    np.testing.assert_array_equal(stable.x_live, drift.x_live)
    np.testing.assert_array_equal(stable.y_live[:15], drift.y_live[:15])
    assert drift.mean_live[-1] == pytest.approx(-stable.mean_live[-1])


def test_late_label_does_not_change_earlier_forecasts() -> None:
    stream = generate_delayed_stream(15, n_fit=80, n_cal=30, n_live=45)
    modified_y = stream.y_live.copy()
    modified_y[0] += 100.0
    modified = replace(stream, y_live=modified_y)
    original, _ = evaluate_delayed_stream(
        stream, feedback_delay=3, rolling_window=20,
        refit_every=5, refit_window=64,
    )
    changed, _ = evaluate_delayed_stream(
        modified, feedback_delay=3, rolling_window=20,
        refit_every=5, refit_window=64,
    )
    # j=0 first becomes eligible at t=4.
    columns = ["policy", "prediction", "radius", "available_labels"]
    a = original.loc[original["step"] < 4, columns].reset_index(drop=True)
    b = changed.loc[changed["step"] < 4, columns].reset_index(drop=True)
    assert a.equals(b)
    assert (original.loc[original["step"] == 3, "available_labels"] == 0).all()
    assert (original.loc[original["step"] == 4, "available_labels"] == 1).all()


def test_paired_delays_have_matching_oracles() -> None:
    interval, decision = run_gradual_drift_study(
        repeats=1, delays=(0, 3), regimes=("stable", "gradual_drift"),
        scenarios=("low_variance_signal",), n_fit=80, n_cal=30,
        n_live=45, rolling_window=20, refit_every=5, refit_window=64,
    )
    assert len(interval) == 2 * 2 * 45 * 3
    assert len(decision) == 2 * 2 * 45 * 3 * 4 * 2
    assert interval["covered"].between(0, 1).all()
    assert (decision["regret"] >= -1e-10).all()
    for (regime, step, cost), rows in decision.groupby(["regime", "step", "cost"]):
        assert rows["oracle_profit"].max() == pytest.approx(rows["oracle_profit"].min())
    coverage, economic = aggregate_study(interval, decision, bootstrap_draws=10)
    assert coverage["mc_lower"].isna().all()
    assert not economic.empty
    shift = paired_delay_penalty(decision, comparison_delay=3, bootstrap_draws=10)
    assert shift["mc_lower"].isna().all()


def test_reject_invalid_duplicate_delays() -> None:
    with pytest.raises(ValueError):
        run_gradual_drift_study(repeats=1, delays=(0, 0))
