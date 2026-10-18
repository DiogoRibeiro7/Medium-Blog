"""FDA Adverse Event Reporting System through the openFDA API.

For each age band the module counts reports (all drugs), reports mentioning a
stimulant group, reports with an event group, and reports with both. These four
numbers give the reporting odds ratio of the event for the stimulant group
against all other drugs in the same age band.

A report is counted for a drug group if any drug on it matches, whether listed
as suspect or concomitant; openFDA cannot tie the drug role and the drug name to
the same entry in one query. Reports without a harmonised name or brand match
are missed. Both limitations apply equally to every age band.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from . import http
from .paths import DATA

API = "https://api.fda.gov/drug/event.json"

AGE_BANDS = [("6-17", 6, 17), ("18-44", 18, 44), ("45-64", 45, 64), ("65-74", 65, 74), ("75+", 75, 120)]

DRUG_GROUPS = {
    "methylphenidate class": {
        "generic": ["methylphenidate", "dexmethylphenidate", "serdexmethylphenidate"],
        "brand": ["ritalin", "concerta", "focalin", "metadate", "methylin", "daytrana", "quillichew",
                  "cotempla", "aptensio", "jornay", "azstarys", "relexxii"],
    },
    "amphetamine class": {
        "generic": ["amphetamine", "dextroamphetamine", "lisdexamfetamine"],
        "brand": ["adderall", "vyvanse", "dexedrine", "mydayis", "evekeo", "adzenys", "dyanavel",
                  "zenzedi", "procentra", "xelstrym"],
    },
}

EVENT_GROUPS = {
    "blood pressure": ["HYPERTENSION", "BLOOD PRESSURE INCREASED", "BLOOD PRESSURE SYSTOLIC INCREASED",
                       "BLOOD PRESSURE DIASTOLIC INCREASED", "HYPERTENSIVE CRISIS"],
    "heart rate and rhythm": ["TACHYCARDIA", "HEART RATE INCREASED", "PALPITATIONS", "ATRIAL FIBRILLATION",
                              "ARRHYTHMIA", "SUPRAVENTRICULAR TACHYCARDIA", "VENTRICULAR TACHYCARDIA"],
    "ischaemic and serious cardiac": ["MYOCARDIAL INFARCTION", "ACUTE MYOCARDIAL INFARCTION",
                                      "CEREBROVASCULAR ACCIDENT", "ISCHAEMIC STROKE", "CARDIAC ARREST",
                                      "CARDIAC FAILURE", "CORONARY ARTERY DISEASE"],
    "insomnia": ["INSOMNIA", "INITIAL INSOMNIA", "MIDDLE INSOMNIA", "SLEEP DISORDER"],
    "anxiety and agitation": ["ANXIETY", "AGITATION", "NERVOUSNESS", "IRRITABILITY", "RESTLESSNESS"],
    "psychosis": ["PSYCHOTIC DISORDER", "HALLUCINATION", "HALLUCINATION, VISUAL", "HALLUCINATION, AUDITORY",
                  "DELUSION", "PARANOIA"],
    "appetite and weight": ["DECREASED APPETITE", "WEIGHT DECREASED"],
}


def _or(field: str, values: list[str], exact: bool = False) -> str:
    f = f"{field}.exact" if exact else field
    return "(" + " ".join(f'{f}:"{v}"' for v in values) + ")"


def drug_clause(group: str) -> str:
    g = DRUG_GROUPS[group]
    parts = [_or("patient.drug.openfda.generic_name", g["generic"]),
             _or("patient.drug.medicinalproduct", g["generic"] + g["brand"])]
    return "(" + " ".join(parts) + ")"


def age_clause(lo: int, hi: int) -> str:
    return f"(patient.patientonsetageunit:801 AND patient.patientonsetage:[{lo} TO {hi}])"


def event_clause(group: str) -> str:
    return _or("patient.reaction.reactionmeddrapt", EVENT_GROUPS[group], exact=True)


def total(search: str) -> int:
    params = {"search": search, "limit": 1}
    key = os.environ.get("OPENFDA_API_KEY")
    if key:
        params["api_key"] = key
    r = http.get(API, params=params, ok_404=True)
    time.sleep(0.35)
    if r is None:
        return 0
    return int(r.json()["meta"]["results"]["total"])


def build() -> pd.DataFrame:
    rows = []
    meta = None
    for band, lo, hi in AGE_BANDS:
        a = age_clause(lo, hi)
        n_band = total(a)
        print(f"  openFDA {band}: {n_band:,} reports")
        events = {e: total(f"{a} AND {event_clause(e)}") for e in EVENT_GROUPS}
        for g in DRUG_GROUPS:
            dclause = drug_clause(g)
            n_drug = total(f"{a} AND {dclause}")
            for e in EVENT_GROUPS:
                both = total(f"{a} AND {dclause} AND {event_clause(e)}")
                rows.append({"age_band": band, "drug_group": g, "event_group": e, "n_band": n_band,
                             "n_drug": n_drug, "n_event": events[e], "n_both": both})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "faers_counts.csv.gz", index=False, compression="gzip")
    r = http.get(API, params={"limit": 1}).json()
    meta = {"last_updated": r["meta"].get("last_updated"), "total_reports": r["meta"]["results"]["total"]}
    (DATA / "faers_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"openFDA: {len(df)} rows; database last updated {meta['last_updated']}")
    return df


def load() -> tuple[pd.DataFrame, dict]:
    return (pd.read_csv(DATA / "faers_counts.csv.gz"),
            json.loads((DATA / "faers_meta.json").read_text()))
