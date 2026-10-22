"""Horizon calibration, time ordering and retraining cost invariants."""
from dataclasses import replace
import numpy as np
import pytest
from gradual_drift import generate_delayed_stream
from horizon_detector_economics import (
    MaintenanceDesign, corrected_horizon_quantile, reference_increments,
    horizon_maximum, evaluate_maintenance, run_maintenance_study,
    summarize_maintenance,
)

def test_corrected_rank_and_infinite_threshold() -> None:
    scores=np.arange(1.,20.)
    threshold,rank=corrected_horizon_quantile(scores,0.10)
    assert rank==18 and threshold==18
    threshold,rank=corrected_horizon_quantile(scores,0.01)
    assert rank==20 and np.isinf(threshold)

def test_legal_feedback_release() -> None:
    stream=generate_delayed_stream(11,n_fit=80,n_cal=35,n_live=45)
    original=reference_increments(stream,delay=3)
    assert np.isnan(original[:4]).all()
    altered=stream.y_live.copy();altered[0]+=500
    changed=replace(stream,y_live=altered)
    updated=reference_increments(changed,delay=3)
    np.testing.assert_allclose(original[:4],updated[:4],equal_nan=True)
    assert original[4]!=updated[4]

def test_cost_is_charged_only_for_refit_events() -> None:
    design=MaintenanceDesign(n_fit=80,n_cal=35,n_live=45,calibration_streams=19,
                             delay_values=(0,),refit_every=10,refit_window=70,
                             action_costs=(0.6,),refit_costs=(0.0,0.25))
    stream=generate_delayed_stream(7,n_fit=80,n_cal=35,n_live=45)
    models,economic=evaluate_maintenance(stream,design,0,float("inf"),float("inf"))
    assert models.loc[models.policy=="Scheduled","refit_event"].sum()==4
    assert models.loc[models.policy=="CUSUM matched","refit_event"].sum()==0
    costs=economic.pivot(index=["step","policy","action_cost"],columns="refit_cost",values="net_regret")
    merged=costs.reset_index().merge(models[["step","policy","refit_event"]],on=["step","policy"])
    np.testing.assert_allclose(merged[0.25]-merged[0.0],0.25*merged.refit_event)

def test_calibration_and_evaluation_are_separate_and_reproducible() -> None:
    d=MaintenanceDesign(n_fit=80,n_cal=35,n_live=45,calibration_streams=19,
                        delay_values=(0,),refit_every=10,refit_window=70,
                        action_costs=(0.6,),refit_costs=(0.,0.2))
    a,m,e=run_maintenance_study(d,evaluation_repeats=1,
            scenarios=("low_variance_signal",),regimes=("stable","ar1"),seed=55)
    assert len(a)==2
    assert len(m)==2*45*4
    assert len(e)==2*45*4*2
    assert (e.net_regret>=-1e-9).all()
    assert np.isfinite(m.squared_error).all()
    operational,loss=summarize_maintenance(m,e)
    assert not operational.empty and not loss.empty
    assert operational.probability_any_alarm.between(0,1).all()

def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        MaintenanceDesign(calibration_streams=1)
    with pytest.raises(ValueError):
        horizon_maximum(np.array([np.inf]))
