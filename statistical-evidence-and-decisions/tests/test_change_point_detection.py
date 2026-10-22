"""Checks for sequential detection and delayed information."""
from dataclasses import replace
import numpy as np
import pytest
from gradual_drift import generate_delayed_stream, available_label_count
from change_point_detection import (
    DetectorSettings, residual_disagreement, bootstrap_threshold,
    evaluate_detector, run_detector_study, detector_summary
)

def test_feedback_availability() -> None:
    assert [available_label_count(t,3) for t in range(6)] == [0,0,0,0,1,2]
    with pytest.raises(ValueError):
        DetectorSettings(delay=-1)

def test_score_reversal_and_threshold_reproducibility() -> None:
    x=np.array([-2.,-1.,1.,2.])
    np.testing.assert_allclose(residual_disagreement(x,x,1.),0.)
    assert residual_disagreement(-x,x,1.).mean()>0
    scores=np.random.default_rng(5).normal(size=80)
    settings=DetectorSettings(bootstrap_draws=10)
    assert bootstrap_threshold(scores,settings,5)==bootstrap_threshold(scores,settings,5)

def test_delayed_label_cannot_change_earlier_forecasts() -> None:
    data=generate_delayed_stream(11,n_fit=80,n_cal=35,n_live=45)
    changed=data.y_live.copy();changed[0]+=1000
    mutated=replace(data,y_live=changed)
    settings=DetectorSettings(delay=3,scheduled_every=10,cooldown=10,
                              train_window=70,bootstrap_draws=10)
    a,_=evaluate_detector(data,settings)
    b,_=evaluate_detector(mutated,settings)
    cols=["policy","prediction","statistic","alarm","available_labels"]
    assert a.loc[a.step<4,cols].reset_index(drop=True).equals(
        b.loc[b.step<4,cols].reset_index(drop=True))
    assert (a.loc[a.step==4,"available_labels"]==1).all()

def test_smoke_determinism_and_nonnegative_regret() -> None:
    data,decisions=run_detector_study(repeats=1,smoke=True,seed=9)
    assert len(data)==2*2*45*3
    assert len(decisions)==2*2*45*3*4
    assert (decisions.regret>=-1e-9).all()
    assert np.isfinite(data.squared_error).all()
    assert not detector_summary(data,decisions).empty
