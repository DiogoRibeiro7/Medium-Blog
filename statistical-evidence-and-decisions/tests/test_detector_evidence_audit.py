"""Paired Monte Carlo and multiplicity validation invariants."""
import numpy as np
import pandas as pd
import pytest
from comparative_change_detection import ComparisonDesign
from detector_evidence_audit import (
 wilson_interval, paired_regret, empirical_null_alarms,
 detector_profiles, first_alarm_step, run_sensitivity, summarize_sensitivity
)

def test_wilson_extremes_and_validation() -> None:
    lo,hi=wilson_interval(0,20)
    assert lo==0 and 0<hi<1
    lo,hi=wilson_interval(20,20)
    assert hi==1 and 0<lo<1
    with pytest.raises(ValueError): wilson_interval(21,20)

def test_paired_regret_sign_and_missing_match() -> None:
    rows=[]
    for i in range(3):
        for policy,regret in [("CUSUM",0.3),("GLR",0.1)]:
            rows.append(dict(scenario="low_variance_signal",regime="stable",
                delay=0,replicate=i,action_cost=0.6,refit_cost=0.1,
                policy=policy,net_regret=regret))
    frame=pd.DataFrame(rows)
    out=paired_regret(frame,candidates=("GLR",),bootstrap_draws=30)
    assert out.iloc[0].delta_regret==pytest.approx(-0.2)
    assert out.iloc[0].independent_streams==3
    with pytest.raises(ValueError):
        paired_regret(frame.iloc[:-1],candidates=("GLR",))

def test_alarm_family_is_union_of_detector_events() -> None:
    rows=[]
    for i in range(3):
        for name in ("CUSUM","GLR","BOCPD"):
            rows.append(dict(scenario="low_variance_signal",regime="stable",
                delay=0,replicate=i,policy=name,
                alarm=int((i==0 and name=="CUSUM") or
                          (i==1 and name=="GLR"))))
    out=empirical_null_alarms(pd.DataFrame(rows))
    family=out.loc[out.policy=="Any detector"].iloc[0]
    assert family.alarm_probability==pytest.approx(2/3)

def test_first_alarm_and_profiles() -> None:
    assert first_alarm_step(np.array([np.nan,0.,2.,3.]),2.)==3
    assert first_alarm_step(np.array([np.nan,0.]),1.) is None
    p=detector_profiles(ComparisonDesign())
    assert len(p)==3 and p["Short memory"].bocpd_hazard > p["Long memory"].bocpd_hazard

def test_sensitivity_small_experiment_and_correction() -> None:
    design=ComparisonDesign(n_fit=80,n_cal=35,n_live=45,
        null_streams=19,delays=(0,),refit_window=70,
        glr_max_window=16)
    calibration,observations=run_sensitivity(
        design,evaluation_repeats=1,profiles=("Baseline",),
        regimes=("stable","gradual_drift"),seed=33)
    assert len(calibration)==1*1*2*3
    assert len(observations)==2*1*2*3
    summary=summarize_sensitivity(observations)
    assert set(summary.detector)=={"CUSUM","GLR","BOCPD","Any detector"}
    assert summary.alarm_probability.between(0,1).all()
    assert summary.wilson_lower_95.isna().all()
    table=calibration.pivot(index="detector",columns="rule",values="threshold")
    assert (table["Bonferroni"] >= table["Individual"]).all()
