"""Tests for preregistered detector evidence accounting."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from evidence_synthesis import (
 PrimaryProtocol, preregistered_verdict, validate_evaluation,
 descriptive_context, _one_sided_wilson_upper,
)

def synthetic_panels(n=5, gain=0.04):
    models=[]; economics=[]
    for regime in ("stable","gradual_drift"):
        for replicate in range(n):
            for policy in ("CUSUM","GLR","BOCPD"):
                for t in range(3):
                    alarm=int(regime=="stable" and replicate==0 and policy=="GLR" and t==1)
                    models.append(dict(scenario="low_variance_signal",regime=regime,
                      delay=0,replicate=replicate,policy=policy,step=t,
                      alarm=alarm,refit_event=alarm,squared_error=0.5))
                    regret=(0.2-(gain if policy=="GLR" else 0.0))
                    if regime=="stable":regret=0.1
                    economics.append(dict(scenario="low_variance_signal",regime=regime,
                      delay=0,replicate=replicate,policy=policy,step=t,
                      action_cost=0.6,refit_cost=0.05,net_regret=regret,
                      expected_net_profit=1.0-regret,oracle_profit=1.0))
    return pd.DataFrame(models),pd.DataFrame(economics)

def test_insufficient_data_never_claims_victory():
    m,e=synthetic_panels()
    results=preregistered_verdict(m,e)
    assert set(results.status)=={"exploratory_insufficient_streams"}
    assert len(results)==2
    assert not descriptive_context(m,e).empty

def test_bonferroni_bounds_and_finite_small_sample():
    upper=_one_sided_wilson_upper(0,100,0.0125)
    assert 0<upper<1
    assert _one_sided_wilson_upper(100,100,0.0125)==1

def test_material_gain_only_when_rule_is_met():
    m,e=synthetic_panels(6,0.04)
    p=replace(PrimaryProtocol(),minimum_streams=3,bootstrap_draws=100,
              max_false_alarm=0.999,meaningful_gain=0.01)
    verdict=preregistered_verdict(m,e,p)
    assert verdict.loc[verdict.candidate=="GLR","status"].iloc[0]=="supported_in_simulation"
    assert verdict.loc[verdict.candidate=="BOCPD","status"].iloc[0]=="material_gain_not_supported"

def test_regret_identity_and_complete_pairing_are_required():
    m,e=synthetic_panels()
    altered=e.copy()
    altered.loc[0,"net_regret"]=0.99
    with pytest.raises(ValueError,match="Regret"):
        validate_evaluation(m,altered)
    missing=e.loc[~((e.regime=="gradual_drift")&(e.replicate==0)&(e.policy=="GLR"))]
    with pytest.raises(ValueError):
        preregistered_verdict(m,missing)

def test_protocol_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        PrimaryProtocol(minimum_streams=0)
    with pytest.raises(ValueError):
        PrimaryProtocol(candidates=("CUSUM",))
