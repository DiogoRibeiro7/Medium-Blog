"""Mathematical and chronology invariants for sequential detector comparison."""
from dataclasses import replace
import numpy as np
import pytest

from comparative_change_detection import (
    BOCPDTracker, GLRTracker, CUSUMTracker, ComparisonDesign,
    score_maxima, evaluate_policies, run_comparison, summarize_comparison,
)
from gradual_drift import generate_delayed_stream
from horizon_detector_economics import reference_increments


def test_glr_known_positive_shift_likelihood() -> None:
    detector = GLRTracker(minimum=2, maximum=4)
    assert detector.update(2.0) == 0.0
    assert detector.update(2.0) == pytest.approx(4.0)
    detector.reset()
    assert detector.update(-2.0) == 0.0
    assert detector.update(-2.0) == pytest.approx(0.0)


def test_bocpd_run_length_posterior_is_normalised() -> None:
    tracker = BOCPDTracker(hazard=0.1, prior_precision=1.0,
                           short_window=2, warmup=4)
    for value in (0.0, 0.1, -0.2, 3.0, 3.1, 3.2):
        score = tracker.update(value)
        assert 0 <= score <= 1
        assert tracker.posterior.sum() == pytest.approx(1.0)
        assert (tracker.posterior >= 0).all()
        assert len(tracker.posterior) == len(tracker.means)
    tracker.reset()
    assert tracker.observation_count == 0
    assert tracker.posterior.tolist() == [1.0]


def test_cusum_resets_and_score_maxima() -> None:
    detector = CUSUMTracker(allowance=0.2)
    assert detector.update(1.0) == pytest.approx(0.8)
    detector.reset()
    assert detector.value == 0
    design = ComparisonDesign(n_fit=80, n_cal=35, n_live=45,
                              null_streams=19, delays=(3,),
                              refit_window=70)
    maxima = score_maxima(np.array([np.nan, 0.0, 2.0, 2.0]), design)
    assert set(maxima) == {"CUSUM", "GLR", "BOCPD"}
    assert all(np.isfinite(x) and x >= 0 for x in maxima.values())


def test_delayed_outcome_cannot_affect_early_results() -> None:
    design = ComparisonDesign(n_fit=80, n_cal=35, n_live=45,
                              null_streams=19, delays=(3,),
                              refit_window=70, scheduled_every=10)
    stream = generate_delayed_stream(9, n_fit=80, n_cal=35, n_live=45)
    altered_y = stream.y_live.copy()
    altered_y[0] += 500.0
    altered = replace(stream, y_live=altered_y)
    a, _ = evaluate_policies(stream, design, 3,
             {"CUSUM": 0.0, "GLR": 0.0, "BOCPD": 0.0})
    b, _ = evaluate_policies(altered, design, 3,
             {"CUSUM": 0.0, "GLR": 0.0, "BOCPD": 0.0})
    columns = ["step", "policy", "prediction", "available_labels", "alarm"]
    assert a.loc[a.step < 4, columns].reset_index(drop=True).equals(
           b.loc[b.step < 4, columns].reset_index(drop=True))
    released = reference_increments(stream, 3, allowance=0.0)
    assert np.isnan(released[:4]).all()


def test_economic_charge_and_end_to_end_shapes() -> None:
    design = ComparisonDesign(n_fit=80, n_cal=35, n_live=45,
                              null_streams=19, delays=(0,),
                              refit_window=70, scheduled_every=10,
                              action_costs=(0.6,), refit_costs=(0.0, 0.2))
    thresholds, models, economics = run_comparison(
        design, evaluation_repeats=1, seed=21,
        scenarios=("low_variance_signal",),
        regimes=("stable", "gradual_drift"))
    assert len(thresholds) == 3
    assert len(models) == 2 * 45 * 5
    assert len(economics) == 2 * 45 * 5 * 2
    assert (economics["net_regret"] >= -1e-9).all()
    assert np.isfinite(models["squared_error"]).all()
    operational, loss = summarize_comparison(models, economics)
    assert not operational.empty and not loss.empty
    costs = economics.pivot(
        index=["regime", "step", "policy", "action_cost"],
        columns="refit_cost", values="net_regret"
    ).reset_index()
    event = models[["regime", "step", "policy", "refit_event"]]
    joined = costs.merge(event, on=["regime", "step", "policy"])
    np.testing.assert_allclose(joined[0.2] - joined[0.0],
                               0.2 * joined["refit_event"])


def test_reject_bad_hazard_and_glr_windows() -> None:
    with pytest.raises(ValueError):
        BOCPDTracker(hazard=1.0)
    with pytest.raises(ValueError):
        GLRTracker(minimum=5, maximum=3)
