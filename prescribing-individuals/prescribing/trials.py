"""ClinicalTrials.gov: adult stimulant trials for ADHD with posted results.

The API (v2) returns one JSON record per study. This module keeps the raw pages
(gzipped, not committed) and extracts four tidy tables:

- studies:   design, eligibility ages, enrolment
- ages:      baseline age categories (participants aged 65 and over)
- outcomes:  blood-pressure and heart-rate outcome measures, by arm
- adverse:   adverse events of interest, by arm
"""

from __future__ import annotations

import gzip
import json
import re

import pandas as pd

from . import http
from .paths import DATA, RAW

API = "https://clinicaltrials.gov/api/v2/studies"
DRUG_TERMS = ["methylphenidate", "dexmethylphenidate", "amphetamine", "dextroamphetamine",
              "lisdexamfetamine", "serdexmethylphenidate", "adderall", "vyvanse", "concerta",
              "ritalin", "focalin", "mydayis", "dexedrine", "jornay", "azstarys", "quillichew",
              "aptensio", "adzenys", "evekeo", "dyanavel", "mixed amphetamine salts", "mas xr"]
QUERY_INTR = " OR ".join(f'"{t}"' if " " in t else t for t in DRUG_TERMS[:6] + ["mixed amphetamine salts"])
QUERY_COND = "ADHD OR attention deficit hyperactivity disorder OR attention deficit disorder"

VITAL = re.compile(r"blood pressure|systolic|diastolic|heart rate|pulse|\bSBP\b|\bDBP\b|\bHR\b", re.I)
AE_GROUPS = {
    "insomnia": re.compile(r"insomnia|sleep disorder|initial insomnia|middle insomnia", re.I),
    "blood_pressure": re.compile(r"hypertension|blood pressure increased|blood pressure systolic increased|"
                                 r"blood pressure diastolic increased", re.I),
    "heart_rate": re.compile(r"tachycardia|heart rate increased|palpitations", re.I),
    "appetite": re.compile(r"decreased appetite|appetite decreased|anorexia|weight decreased", re.I),
    "anxiety": re.compile(r"anxiety|nervousness|agitation|irritability", re.I),
}


def fetch() -> list[dict]:
    """Download every matching study with results; cache the raw pages."""
    dest = RAW / "trials" / "studies.json.gz"
    if dest.exists():
        return json.loads(gzip.decompress(dest.read_bytes()))
    params = {"query.cond": QUERY_COND, "query.intr": QUERY_INTR, "aggFilters": "results:with",
              "pageSize": 100, "format": "json", "countTotal": "true"}
    studies, token = [], None
    while True:
        if token:
            params["pageToken"] = token
        r = http.get(API, params=params).json()
        studies.extend(r.get("studies", []))
        print(f"  ClinicalTrials.gov: {len(studies):,} of {r.get('totalCount', '?')} studies")
        token = r.get("nextPageToken")
        if not token:
            break
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(gzip.compress(json.dumps(studies).encode()))
    return studies


def _age_years(s: str | None) -> float | None:
    if not s:
        return None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(year|month|week|day)", s, re.I)
    if not m:
        return None
    v = float(m.group(1))
    unit = m.group(2).lower()
    return v if unit == "year" else v / {"month": 12, "week": 52.18, "day": 365.25}[unit]


def _is_drug(text: str) -> bool:
    t = text.lower()
    return any(d in t for d in DRUG_TERMS)


def _role(title: str, description: str) -> str:
    t = f"{title} {description}".lower()
    has_placebo = "placebo" in t
    has_drug = _is_drug(t)
    if has_placebo and not has_drug:
        return "placebo"
    if has_drug and not has_placebo:
        return "active"
    if has_placebo and has_drug:
        # e.g. "Placebo" arm described as "matching placebo for methylphenidate"
        return "placebo" if title.lower().strip().startswith("placebo") else "mixed"
    return "other"


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract(studies: list[dict]) -> dict[str, pd.DataFrame]:
    srows, arows, orows, erows = [], [], [], []
    for st in studies:
        ps = st.get("protocolSection", {})
        rs = st.get("resultsSection", {})
        nct = ps.get("identificationModule", {}).get("nctId")
        design = ps.get("designModule", {})
        info = design.get("designInfo", {})
        elig = ps.get("eligibilityModule", {})
        conds = " ".join(ps.get("conditionsModule", {}).get("conditions", []))
        interventions = ps.get("armsInterventionsModule", {}).get("interventions", [])
        intr_text = " ".join(i.get("name", "") + " " + " ".join(i.get("otherNames", []) or [])
                             for i in interventions)
        srows.append({
            "nct": nct,
            "title": ps.get("identificationModule", {}).get("briefTitle"),
            "conditions": conds,
            "stimulant_intervention": _is_drug(intr_text),
            "phases": ";".join(design.get("phases", []) or []),
            "allocation": info.get("allocation"),
            "model": info.get("interventionModel"),
            "masking": (info.get("maskingInfo") or {}).get("masking"),
            "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
            "min_age": _age_years(elig.get("minimumAge")),
            "max_age": _age_years(elig.get("maximumAge")),
            "std_ages": ";".join(elig.get("stdAges", []) or []),
            "start": (ps.get("statusModule", {}).get("startDateStruct") or {}).get("date"),
        })

        # baseline age categories
        base = rs.get("baselineCharacteristicsModule", {})
        bgroups = {g["id"]: g.get("title", "") for g in base.get("groups", [])}
        total_id = next((gid for gid, t in bgroups.items() if t.strip().lower() == "total"), None)
        for meas in base.get("measures", []):
            title = meas.get("title", "")
            if not re.search(r"\bage\b", title, re.I):
                continue
            for cls in meas.get("classes", []):
                for cat in cls.get("categories", []):
                    ctitle = (cat.get("title") or cls.get("title") or "").strip()
                    for m in cat.get("measurements", []):
                        if total_id and m.get("groupId") != total_id:
                            continue
                        arows.append({"nct": nct, "measure": title, "param": meas.get("paramType"),
                                      "category": ctitle, "value": _num(m.get("value")),
                                      "spread": _num(m.get("spread")), "group": m.get("groupId"),
                                      "has_total": total_id is not None})

        # vital-sign outcomes
        for om in rs.get("outcomeMeasuresModule", {}).get("outcomeMeasures", []):
            title = om.get("title", "")
            desc = om.get("description", "") or ""
            cats_text = " ".join((c.get("title") or "") + " " + " ".join((k.get("title") or "")
                                 for k in c.get("categories", [])) for c in om.get("classes", []))
            if not (VITAL.search(title) or VITAL.search(cats_text)):
                continue
            groups = {g["id"]: (g.get("title", ""), g.get("description", "") or "") for g in om.get("groups", [])}
            denoms = {}
            for dn in om.get("denoms", []):
                if (dn.get("units") or "").lower().startswith("participant"):
                    for c in dn.get("counts", []):
                        denoms[c["groupId"]] = _num(c.get("value"))
            for ci, cls in enumerate(om.get("classes", [])):
                for ki, cat in enumerate(cls.get("categories", [])):
                    for m in cat.get("measurements", []):
                        gid = m.get("groupId")
                        gt, gd = groups.get(gid, ("", ""))
                        orows.append({
                            "nct": nct, "type": om.get("type"), "title": title, "description": desc,
                            "time_frame": om.get("timeFrame"), "param": om.get("paramType"),
                            "dispersion": om.get("dispersionType"), "unit": om.get("unitOfMeasure"),
                            "class": cls.get("title"), "class_index": ci,
                            "category": cat.get("title"), "category_index": ki,
                            "group": gid, "group_title": gt, "group_description": gd,
                            "role": _role(gt, gd), "n": denoms.get(gid),
                            "value": _num(m.get("value")), "spread": _num(m.get("spread")),
                            "lower": _num(m.get("lowerLimit")), "upper": _num(m.get("upperLimit")),
                        })

        # adverse events of interest
        aem = rs.get("adverseEventsModule", {})
        egroups = {g["id"]: (g.get("title", ""), g.get("description", "") or "") for g in aem.get("eventGroups", [])}
        for kind in ("seriousEvents", "otherEvents"):
            for ev in aem.get(kind, []):
                term = ev.get("term", "")
                hit = [k for k, rx in AE_GROUPS.items() if rx.search(term)]
                if not hit:
                    continue
                for s in ev.get("stats", []):
                    gid = s.get("groupId")
                    gt, gd = egroups.get(gid, ("", ""))
                    erows.append({"nct": nct, "kind": kind, "term": term, "group_key": hit[0],
                                  "group": gid, "group_title": gt, "role": _role(gt, gd),
                                  "affected": _num(s.get("numAffected")), "at_risk": _num(s.get("numAtRisk"))})
    return {
        "studies": pd.DataFrame(srows),
        "ages": pd.DataFrame(arows),
        "outcomes": pd.DataFrame(orows),
        "adverse": pd.DataFrame(erows),
    }


def build() -> dict[str, pd.DataFrame]:
    tables = extract(fetch())
    for name, df in tables.items():
        df.to_csv(DATA / f"trials_{name}.csv.gz", index=False, compression="gzip")
        print(f"ClinicalTrials.gov {name}: {len(df):,} rows")
    return tables


def load() -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(DATA / f"trials_{n}.csv.gz", low_memory=False)
            for n in ("studies", "ages", "outcomes", "adverse")}
