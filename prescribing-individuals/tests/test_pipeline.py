"""The extract and analysis steps on small synthetic inputs shaped like the real sources."""

import numpy as np
import pandas as pd
import pytest

from prescribing import analyse, trials


def _study(nct, sd_active, sd_placebo, n=60, max_age="65 Years"):
    groups = [{"id": "OG000", "title": "Lisdexamfetamine 70 mg", "description": ""},
              {"id": "OG001", "title": "Placebo", "description": "matching placebo"}]
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": "Adult ADHD trial"},
            "conditionsModule": {"conditions": ["ADHD"]},
            "armsInterventionsModule": {"interventions": [{"name": "Lisdexamfetamine"}, {"name": "Placebo"}]},
            "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 2 * n},
                             "designInfo": {"allocation": "RANDOMIZED", "interventionModel": "PARALLEL"}},
            "eligibilityModule": {"minimumAge": "18 Years", "maximumAge": max_age, "stdAges": ["ADULT"]},
        },
        "resultsSection": {
            "baselineCharacteristicsModule": {
                "groups": [{"id": "BG000", "title": "Drug"}, {"id": "BG001", "title": "Placebo"},
                           {"id": "BG002", "title": "Total"}],
                "measures": [{"title": "Age, Categorical", "paramType": "COUNT_OF_PARTICIPANTS",
                              "classes": [{"categories": [
                                  {"title": "Between 18 and 65 years",
                                   "measurements": [{"groupId": "BG002", "value": str(2 * n - 1)}]},
                                  {"title": ">=65 years", "measurements": [{"groupId": "BG002", "value": "1"}]}]}]}],
            },
            "outcomeMeasuresModule": {"outcomeMeasures": [{
                "type": "SECONDARY", "title": "Change From Baseline in Systolic Blood Pressure",
                "paramType": "MEAN", "dispersionType": "STANDARD_DEVIATION", "unitOfMeasure": "mmHg",
                "groups": groups,
                "denoms": [{"units": "Participants", "counts": [{"groupId": "OG000", "value": str(n)},
                                                               {"groupId": "OG001", "value": str(n)}]}],
                "classes": [{"categories": [{"measurements": [
                    {"groupId": "OG000", "value": "2.5", "spread": str(sd_active)},
                    {"groupId": "OG001", "value": "0.3", "spread": str(sd_placebo)}]}]}],
            }]},
            "adverseEventsModule": {
                "eventGroups": [{"id": "EG000", "title": "Lisdexamfetamine 70 mg"}, {"id": "EG001", "title": "Placebo"}],
                "otherEvents": [{"term": "Insomnia", "stats": [
                    {"groupId": "EG000", "numAffected": 12, "numAtRisk": n},
                    {"groupId": "EG001", "numAffected": 3, "numAtRisk": n}]}],
            },
        },
    }


def test_trials_extract_and_variance_ratio():
    studies = [_study("NCT1", 12.0, 9.0), _study("NCT2", 11.0, 10.0), _study("NCT3", 9.5, 9.0, max_age=None)]
    t = trials.extract(studies)
    assert set(t["studies"]["nct"]) == {"NCT1", "NCT2", "NCT3"}
    inv = analyse.trial_inventory(t)
    assert inv["adult_trials"] == 3
    assert inv["max_age_missing"] == 1
    assert inv["participants_65_plus"] == 3
    vr = analyse.variance_ratio_table(t)
    assert len(vr) == 3 and (vr["kind"] == "sbp").all()
    s = analyse.variance_ratio_summary(vr)
    row = s.iloc[0]
    assert row["vr"] > 1
    assert row["md"] == pytest.approx(2.2)
    ae, _ = analyse.adverse_summary(t)
    assert ae.iloc[0]["rr"] == pytest.approx(4.0, rel=0.05)


def _nhanes(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 81, n)
    sbp = 110 + 0.5 * (age - 18) + rng.normal(0, 12, n)
    df = pd.DataFrame({
        "seqn": np.arange(n), "cycle": rng.choice(["I", "P", "L"], n), "age": age,
        "pregnant": False, "stratum": rng.integers(0, 10, n).astype(str), "psu": rng.integers(1, 3, n),
        "w_cycle": rng.uniform(1000, 5000, n),
        "sbp1": sbp + rng.normal(0, 4, n), "sbp2": sbp + rng.normal(0, 4, n), "sbp3": sbp + rng.normal(0, 4, n),
        "dbp": rng.normal(72, 10, n), "bp_med": rng.random(n) < 0.2, "cvd": rng.random(n) < 0.05,
        "rx_available": True, "stimulant": rng.random(n) < 0.02,
    })
    df["mode"] = np.where(df["cycle"] == "I", "auscultatory", "oscillometric")
    df["w_osc"] = np.where(df["mode"] == "oscillometric", df["w_cycle"], 0.0)
    df["w_all"] = df["w_cycle"]
    df["sbp"] = df[["sbp1", "sbp2", "sbp3"]].mean(axis=1)
    return df


def test_nhanes_analyses_run_and_order_by_age():
    df = _nhanes()
    b = analyse.nhanes_baselines(df)
    assert list(b["band"]) == [x[0] for x in analyse.AGE_BANDS]
    assert b["sbp_mean"].iloc[:4].is_monotonic_increasing
    c = analyse.nhanes_crossing(df, mu=2.0, tau=6.0)
    assert (c["hetero"] >= 0).all()
    assert (c["uniform_10"] >= c["uniform_2"]).all()
    r = analyse.nhanes_replicates(df)
    assert r["within_sd"].between(3, 5).all()
    s, _ = analyse.nhanes_stimulants(df)
    assert (s["prevalence"] < 0.1).all()


def test_faers_rors():
    counts = pd.DataFrame([
        {"age_band": "18-44", "drug_group": "g", "event_group": "e", "n_band": 100000, "n_drug": 2000,
         "n_event": 3000, "n_both": 150},
        {"age_band": "65-74", "drug_group": "g", "event_group": "e", "n_band": 80000, "n_drug": 300,
         "n_event": 4000, "n_both": 40},
    ])
    r = analyse.faers_rors(counts)
    assert (r["ror"] > 1).all()
    c = analyse.faers_age_contrast(r, old=("65-74",))
    assert len(c) == 1
