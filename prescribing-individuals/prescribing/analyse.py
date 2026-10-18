"""Analyses of the four sources. Each function returns plain dictionaries and
DataFrames that the report and the figures read; nothing here downloads data."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats
from .cvrisk import pce_risk

AGE_BANDS = [("18-39", 18, 39), ("40-59", 40, 59), ("60-69", 60, 69), ("70-79", 70, 79), ("80+", 80, 200)]
OLDER = ("60-69", "70-79", "80+")


def band_of(age: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    for name, lo, hi in AGE_BANDS:
        out[(age >= lo) & (age <= hi)] = name
    return out


# =========================================================================== NHANES
def nhanes_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["band"] = band_of(d["age"])
    d["bp_med"] = d["bp_med"].fillna(False).astype(bool)
    d["htn"] = (d["sbp"] >= 130) | (d["dbp"] >= 80) | d["bp_med"]
    d["htn140"] = (d["sbp"] >= 140) | (d["dbp"] >= 90) | d["bp_med"]
    d["cvd"] = d["cvd"].fillna(False).astype(bool)
    return d


def _est(d, y, w, domain):
    return stats.survey_mean(d[y].astype(float), d[w], d["stratum"], d["psu"], domain)


def nhanes_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Blood pressure and cardiovascular baselines by age, oscillometric cycles (2017-2023)."""
    d = nhanes_frame(df)
    base = (d["w_osc"] > 0) & ~d["pregnant"].fillna(False).astype(bool) & d["sbp"].notna()
    rows = []
    for name, _, _ in AGE_BANDS:
        dom = base & (d["band"] == name)
        m = _est(d, "sbp", "w_osc", dom)
        h = _est(d, "htn", "w_osc", dom)
        h2 = _est(d, "htn140", "w_osc", dom)
        c = _est(d, "cvd", "w_osc", dom)
        med = stats.weighted_quantile(d.loc[dom, "sbp"], d.loc[dom, "w_osc"], [0.1, 0.5, 0.9])
        rows.append({"band": name, "n": m.n, "sbp_mean": m.est, "sbp_mean_se": m.se,
                     "sbp_p10": med[0], "sbp_p50": med[1], "sbp_p90": med[2],
                     "htn": h.est, "htn_se": h.se, "htn140": h2.est, "htn140_se": h2.se,
                     "cvd": c.est, "cvd_se": c.se})
    return pd.DataFrame(rows)


def nhanes_crossing(df: pd.DataFrame, mu: float, tau: float, shifts=(2.0, 5.0, 10.0),
                    tau_hi: float | None = None) -> pd.DataFrame:
    """Share of each age band moved across 130 mmHg systolic by a stimulant response.

    Population: oscillometric cycles, not pregnant, not already hypertensive
    (systolic < 130, diastolic < 80, no blood-pressure medication). Two models of
    the response: a uniform shift for everyone, and a normal distribution of
    individual responses with mean mu and between-person SD tau (from the trials).
    Shares are of the whole age band, so they are directly comparable across ages.
    """
    d = nhanes_frame(df)
    base = (d["w_osc"] > 0) & ~d["pregnant"].fillna(False).astype(bool) & d["sbp"].notna() & d["dbp"].notna()
    normal = base & ~d["htn"]
    rows = []
    for name, _, _ in AGE_BANDS:
        band = base & (d["band"] == name)
        rec = {"band": name, "n_band": int(band.sum()),
               "share_not_htn": _est(d.assign(x=normal.astype(float)), "x", "w_osc", band).est}
        for s in shifts:
            cross = normal & (d["sbp"] >= 130 - s)
            e = _est(d.assign(x=cross.astype(float)), "x", "w_osc", band)
            rec[f"uniform_{s:g}"] = e.est
            rec[f"uniform_{s:g}_se"] = e.se
        p = np.where(normal, sps.norm.sf(130 - d["sbp"], loc=mu, scale=max(tau, 1e-9)), 0.0)
        e = _est(d.assign(x=p), "x", "w_osc", band)
        rec["hetero"] = e.est
        rec["hetero_se"] = e.se
        below140 = base & ~d["htn140"]
        p140 = np.where(below140, sps.norm.sf(140 - d["sbp"], loc=mu, scale=max(tau, 1e-9)), 0.0)
        e140 = _est(d.assign(x=p140), "x", "w_osc", band)
        rec["hetero_140"] = e140.est
        rec["hetero_140_se"] = e140.se
        rec["share_below_140_untreated"] = _est(d.assign(x=below140.astype(float)), "x", "w_osc", band).est
        if tau_hi is not None:
            # sensitivity: the upper 95% limit of the between-person SD from the trials
            ph = np.where(normal, sps.norm.sf(130 - d["sbp"], loc=mu, scale=max(tau_hi, 1e-9)), 0.0)
            eh = _est(d.assign(x=ph), "x", "w_osc", band)
            rec["hetero_tau_hi"] = eh.est
            rec["hetero_tau_hi_se"] = eh.se
        p_avg = np.where(normal, (d["sbp"] >= 130 - mu).astype(float), 0.0)
        rec["average_only"] = _est(d.assign(x=p_avg), "x", "w_osc", band).est
        rec["share_rise_10_or_more"] = float(sps.norm.sf(10, loc=mu, scale=max(tau, 1e-9)))
        rows.append(rec)
    return pd.DataFrame(rows)


RISK_BANDS = [("40-49", 40, 49), ("50-59", 50, 59), ("60-69", 60, 69), ("70-79", 70, 79)]


# Relative risk of major cardiovascular events per 10 mmHg of systolic pressure implied by
# randomised trials of blood-pressure lowering (Ettehad et al., Lancet 2016: RR 0.80 per
# 10 mmHg reduction), used as a sensitivity check on the Pooled Cohort Equations.
TRIAL_RR_PER_10 = 1 / 0.80


def risk_frame(df: pd.DataFrame, mu: float, tau: float, shifts=(2.0, 5.0, 10.0)) -> pd.DataFrame:
    """One row per adult 40-79 without cardiovascular history, with their 10-year risk
    (r0) and the extra risk from each sustained systolic shift (d2, d5, d10) and from a
    heterogeneous response N(mu, tau^2) (dhet). Empty if the risk inputs are missing."""
    d = nhanes_frame(df)
    need = ["tc", "hdl", "sbp", "smoker", "age"]
    if not all(c in d.columns for c in need):
        return pd.DataFrame()
    d["diabetes"] = d.get("told_diabetes", pd.Series(False, index=d.index)).fillna(False).astype(bool) | \
        (d.get("hba1c", pd.Series(np.nan, index=d.index)) >= 6.5)
    ok = ((d["w_osc"] > 0) & d["age"].between(40, 79) & ~d["cvd"] &
          ~d["pregnant"].fillna(False).astype(bool) & d[["tc", "hdl", "sbp"]].notna().all(axis=1) &
          d["smoker"].notna())
    x = d[ok].copy()
    female = (x["sex"] == "female").to_numpy()
    black = x["black"].fillna(False).astype(bool).to_numpy()

    def risk(sbp):
        return pce_risk(x["age"], female, black, x["tc"], x["hdl"], sbp, x["bp_med"],
                        x["smoker"].astype(float), x["diabetes"].astype(float))

    r0 = risk(x["sbp"].to_numpy())
    x["r0"] = r0
    for sh in shifts:
        x[f"d{sh:g}"] = risk(x["sbp"].to_numpy() + sh) - r0
    nodes, weights = np.polynomial.hermite_e.hermegauss(20)
    weights = weights / weights.sum()
    hetero = np.zeros(len(x))
    for z, w in zip(nodes, weights):
        hetero += w * (risk(x["sbp"].to_numpy() + mu + max(tau, 0.0) * z) - r0)
    x["dhet"] = hetero
    x["rr10"] = (r0 + x["d10"]) / r0
    x["d2_trial_rr"] = r0 * (TRIAL_RR_PER_10 ** 0.2 - 1)
    return x


def nhanes_absolute_risk(df: pd.DataFrame, mu: float, tau: float, shifts=(2.0, 5.0, 10.0)) -> pd.DataFrame:
    """Extra 10-year cardiovascular risk from a sustained rise in systolic pressure, by age.

    Adults 40 to 79 in the oscillometric cycles without cardiovascular history,
    with complete risk-equation inputs. Each person's 10-year risk is computed by
    the Pooled Cohort Equations at their measured systolic pressure and again at
    systolic + shift, keeping treatment status fixed. The heterogeneous scenario
    averages the risk over individual responses ~ N(mu, tau^2) with 20-point
    Gauss-Hermite quadrature. Results are survey-weighted means per age band,
    reported as extra events per 1,000 people over 10 years. extra_2_trial_rr applies
    the trial-based relative risk per 10 mmHg to each person's baseline risk instead.
    """
    x = risk_frame(df, mu, tau, shifts)
    if x.empty:
        return pd.DataFrame()
    rows = []
    for name, lo, hi in RISK_BANDS:
        dom = x["age"].between(lo, hi)
        rec = {"band": name, "n": int(dom.sum())}
        for col, lab in [("r0", "baseline_risk"), ("dhet", "extra_hetero")] + \
                        [(f"d{sh:g}", f"extra_{sh:g}") for sh in shifts] + [("d2_trial_rr", "extra_2_trial_rr")]:
            e = stats.survey_mean(x[col], x["w_osc"], x["stratum"], x["psu"], dom)
            rec[lab] = e.est
            rec[lab + "_se"] = e.se
        rec["rr_per_10"] = float(np.average(x.loc[dom, "rr10"], weights=x.loc[dom, "w_osc"]))
        rows.append(rec)
    return pd.DataFrame(rows)


RISK_CUTS = (0.075, 0.20)   # guideline categories: below 7.5%, 7.5-20%, 20% or more


def nhanes_risk_spread(x: pd.DataFrame) -> pd.DataFrame:
    """How much 10-year risk, and the extra risk of a +5 mmHg rise, vary inside each age band.

    Survey-weighted 10th, 25th, 50th, 75th and 90th percentiles, and the share of each
    band in the guideline risk categories.
    """
    rows = []
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    for name, lo, hi in RISK_BANDS:
        dom = x["age"].between(lo, hi).to_numpy()
        g = x[dom]
        rec = {"band": name, "n": int(dom.sum())}
        for col, lab in (("r0", "risk"), ("d5", "extra5")):
            for qq, v in zip(q, stats.weighted_quantile(g[col], g["w_osc"], q)):
                rec[f"{lab}_p{int(qq * 100)}"] = float(v)
        for lab, ind in (("share_below_7_5", x["r0"] < RISK_CUTS[0]),
                         ("share_7_5_to_20", (x["r0"] >= RISK_CUTS[0]) & (x["r0"] < RISK_CUTS[1])),
                         ("share_20_plus", x["r0"] >= RISK_CUTS[1])):
            e = stats.survey_mean(ind.astype(float), x["w_osc"], x["stratum"], x["psu"], dom)
            rec[lab] = e.est
            rec[lab + "_se"] = e.se
        rows.append(rec)
    return pd.DataFrame(rows)


def age_rule(x: pd.DataFrame, cutoffs=(65, 70), high=RISK_CUTS[1]) -> pd.DataFrame:
    """An age cutoff judged as a classifier of high 10-year cardiovascular risk.

    Among adults 40-79 without cardiovascular history: of the people the rule would
    exclude (age >= cutoff), the share who are not at high risk; of the people at high
    risk, the share the rule would let through (age < cutoff). Survey-weighted.
    """
    hi_risk = (x["r0"] >= high).astype(float)
    rows = []
    for c in cutoffs:
        older = x["age"] >= c
        excl = stats.survey_mean(1 - hi_risk, x["w_osc"], x["stratum"], x["psu"], older)
        miss = stats.survey_mean((~older).astype(float), x["w_osc"], x["stratum"], x["psu"], hi_risk.astype(bool))
        rows.append({"cutoff": c, "high_risk_threshold": high,
                     "excluded_not_high_risk": excl.est, "excluded_not_high_risk_se": excl.se,
                     "n_excluded": excl.n,
                     "high_risk_below_cutoff": miss.est, "high_risk_below_cutoff_se": miss.se,
                     "n_high_risk": miss.n})
    return pd.DataFrame(rows)


def nhanes_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Within-visit SD of replicate systolic readings, by age band (oscillometric cycles)."""
    d = nhanes_frame(df)
    base = (d["w_osc"] > 0) & ~d["pregnant"].fillna(False).astype(bool)
    cols = [c for c in ("sbp1", "sbp2", "sbp3") if c in d.columns]
    rows = []
    for name, _, _ in AGE_BANDS:
        r = d.loc[base & (d["band"] == name), cols].to_numpy(dtype=float)
        sd, n = stats.within_person_sd(r)
        rows.append({"band": name, "within_sd": sd, "n": n,
                     "readings_for_10": stats.readings_needed(10, sd),
                     "readings_for_5": stats.readings_needed(5, sd)})
    return pd.DataFrame(rows)


def nhanes_stimulants(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted prevalence of prescription stimulant use by age band, all cycles with drug data."""
    d = nhanes_frame(df)
    has = d["rx_available"].fillna(False).astype(bool) & d["stimulant"].notna()
    d["stim"] = d["stimulant"].astype(float)
    rows = []
    for name, _, _ in AGE_BANDS:
        dom = has & (d["band"] == name)
        e = _est(d, "stim", "w_all", dom)
        rows.append({"band": name, "n": e.n, "users": int(d.loc[dom, "stim"].sum()),
                     "prevalence": e.est, "se": e.se})
    by_band = pd.DataFrame(rows)
    rows = []
    for cyc in sorted(d.loc[has, "cycle"].unique()):
        for label, lo in (("18-59", 18), ("60+", 60)):
            hi = 59 if lo == 18 else 200
            dom = has & (d["cycle"] == cyc) & (d["age"] >= lo) & (d["age"] <= hi)
            e = _est(d, "stim", "w_cycle", dom)
            rows.append({"cycle": cyc, "ages": label, "n": e.n, "users": int(d.loc[dom, "stim"].sum()),
                         "prevalence": e.est, "se": e.se})
    return by_band, pd.DataFrame(rows)


# =========================================================================== trials
KIND = {
    "sbp": re.compile(r"systolic|\bSBP\b", re.I),
    "dbp": re.compile(r"diastolic|\bDBP\b", re.I),
    "hr": re.compile(r"heart rate|pulse|\bHR\b", re.I),
}


def stimulant_trials(studies: pd.DataFrame) -> pd.DataFrame:
    s = studies.copy()
    s = s[s["stimulant_intervention"].astype(bool)]
    s["adhd"] = s["conditions"].fillna("").str.contains(r"ADHD|attention", case=False)
    return s


def adult_trials(studies: pd.DataFrame) -> pd.DataFrame:
    s = stimulant_trials(studies)
    return s[s["min_age"] >= 17.5]


def mean_baseline_age(ages: pd.DataFrame) -> pd.Series:
    """Mean age in years from the 'Age, Continuous' baseline measure (months converted)."""
    cont = ages[ages["measure"].str.contains("continuous", case=False, na=False) &
                ages["param"].fillna("").str.upper().eq("MEAN")].copy()
    unit = cont["unit"].fillna("years").astype(str).str.lower() if "unit" in cont.columns else "years"
    cont["years"] = np.where(pd.Series(unit, index=cont.index).str.contains("month"), cont["value"] / 12, cont["value"])
    return cont.groupby("nct")["years"].first()


def older_trials(t: dict) -> pd.DataFrame:
    """Trials of older people: minimum age 50 or more, or a mean baseline age of 60 or more."""
    s = stimulant_trials(t["studies"])
    mage = mean_baseline_age(t["ages"])
    s = s.assign(mean_age=s["nct"].map(mage))
    return s[(s["min_age"] >= 50) | (s["mean_age"] >= 60)]


def trial_inventory(t: dict) -> dict:
    allst = stimulant_trials(t["studies"])
    old = older_trials(t)
    s = adult_trials(t["studies"])
    out = {"stimulant_trials_with_results": len(allst),
           "of_which_adhd": int(allst["adhd"].sum()),
           "older_trials": len(old),
           "older_trials_list": "; ".join(f"{r.nct} ({str(r.conditions)[:60]}; min age {r.min_age}, mean age {r.mean_age})"
                                          for r in old.itertuples()),
           "adult_trials": len(s),
           "adult_adhd_trials": int(s["adhd"].sum()),
           "enrolled": float(s["enrollment"].fillna(0).sum()),
           "max_age_missing": int(s["max_age"].isna().sum()),
           "max_age_below_65": int((s["max_age"] < 65).sum()),
           "max_age_65_exactly": int((s["max_age"] == 65).sum()),
           "max_age_above_65": int((s["max_age"] > 65).sum()),
           "min_age_55_plus": int((s["min_age"] >= 55).sum()),
           "randomised_parallel": int(((s["allocation"] == "RANDOMIZED") & (s["model"] == "PARALLEL")).sum())}
    ages = t["ages"]
    ages = ages[ages["nct"].isin(s["nct"])]
    cat = ages["category"].fillna("").astype(str)
    is65 = cat.str.contains(r"(?:>=|≥|>)\s*65|65\s*(?:years)?\s*(?:and|or)\s*(?:over|older|above)", case=False, regex=True)
    with_cat = ages.loc[is65 & ages["param"].fillna("").str.upper().eq("COUNT_OF_PARTICIPANTS")]
    out["trials_reporting_65_category"] = int(with_cat["nct"].nunique())
    out["participants_65_plus"] = float(with_cat.groupby("nct")["value"].sum().sum())
    catrows = ages[ages["nct"].isin(with_cat["nct"]) & ages["param"].fillna("").str.upper().eq("COUNT_OF_PARTICIPANTS") &
                   ages["measure"].str.contains("categorical", case=False, na=False)]
    out["participants_in_those_trials"] = float(catrows["value"].sum())
    adhd_ids = set(s.loc[s["adhd"], "nct"])
    out["adhd_trials_reporting_65_category"] = int(with_cat.loc[with_cat["nct"].isin(adhd_ids), "nct"].nunique())
    out["adhd_participants_65_plus"] = float(with_cat.loc[with_cat["nct"].isin(adhd_ids), "value"].sum())
    out["adhd_participants_in_those_trials"] = float(catrows.loc[catrows["nct"].isin(adhd_ids), "value"].sum())
    out["adhd_max_age_above_65"] = int((s.loc[s["adhd"], "max_age"] > 65).sum())
    out["adhd_max_age_missing"] = int(s.loc[s["adhd"], "max_age"].isna().sum())
    out["older_trials_enrolled"] = float(old["enrollment"].fillna(0).sum())
    mage = mean_baseline_age(ages)
    out["trials_with_mean_age"] = int(len(mage))
    out["mean_age_median"] = float(mage.median()) if len(mage) else float("nan")
    out["mean_age_max"] = float(mage.max()) if len(mage) else float("nan")
    return out


def _outcome_kind(row) -> str | None:
    text = " ".join(str(row[c]) for c in ("title", "class", "category") if pd.notna(row[c]))
    for k, rx in KIND.items():
        if rx.search(text):
            return k
    return None


def _norm(x: pd.Series) -> pd.Series:
    return x.fillna("").astype(str).str.upper().str.replace(" ", "_")


def variance_ratio_table(t: dict, population: str = "adult") -> pd.DataFrame:
    """One active-versus-placebo comparison per trial and vital sign.

    Requirements: randomised parallel adult trial; arithmetic mean change from
    baseline with an observed SD (or an SE converted with sqrt(n)); a placebo arm
    and at least one active arm with five or more participants. When several
    time points are posted, the last one is used; when several active arms are
    posted, the largest is used.
    """
    if population == "adult":
        s = adult_trials(t["studies"])
    elif population == "older":
        s = older_trials(t)
    else:
        s = stimulant_trials(t["studies"])
    s = s[(s["allocation"] == "RANDOMIZED") & (s["model"] == "PARALLEL")]
    o = t["outcomes"]
    o = o[o["nct"].isin(s["nct"])].copy()
    o["dispersion"] = _norm(o["dispersion"])
    o = o[_norm(o["param"]).eq("MEAN")]
    o = o[o["dispersion"].isin(["STANDARD_DEVIATION", "STANDARD_ERROR"])]
    text = (o["title"].fillna("") + " " + o["description"].fillna("")).str.lower()
    o = o[~o["title"].fillna("").str.contains(r"exertion|exercise|peak|maximum|orthostatic|standing", case=False)]
    o["measure"] = np.where(text.loc[o.index].str.contains("change"), "change", "final")
    unit = o["unit"].fillna("").str.lower()
    o = o[unit.str.contains("mm|hg|beat|bpm|min") | unit.eq("")]
    o["kind"] = o.apply(_outcome_kind, axis=1)
    o = o[o["kind"].notna() & o["n"].ge(5) & o["value"].notna() & o["spread"].gt(0)]
    o["sd"] = np.where(o["dispersion"] == "STANDARD_ERROR", o["spread"] * np.sqrt(o["n"]), o["spread"])
    rows = []
    for (nct, kind), g in o.groupby(["nct", "kind"]):
        best = None
        for (title, measure, ci, ki), gg in g.groupby(["title", "measure", "class_index", "category_index"]):
            pla = gg[gg["role"] == "placebo"]
            act = gg[gg["role"] == "active"]
            if pla.empty or act.empty:
                continue
            key = (measure == "change", ci, ki)
            cand = (key, title, measure, pla.iloc[0], act.sort_values("n").iloc[-1])
            if best is None or key > best[0]:
                best = cand
        if best is None:
            continue
        _, title, measure, p, a = best
        y, v = stats.ln_vr(a["sd"], a["n"], p["sd"], p["n"])
        info = s.loc[s["nct"] == nct].iloc[0]
        rows.append({"nct": nct, "population": population, "kind": kind, "measure": measure,
                     "conditions": info["conditions"], "min_age": info["min_age"], "max_age": info["max_age"],
                     "outcome": title, "unit": a["unit"],
                     "active_arm": a["group_title"], "n_active": a["n"], "n_placebo": p["n"],
                     "mean_active": a["value"], "mean_placebo": p["value"],
                     "sd_active": a["sd"], "sd_placebo": p["sd"],
                     "ln_vr": float(y), "ln_vr_var": float(v),
                     "md": a["value"] - p["value"],
                     "md_var": a["sd"] ** 2 / a["n"] + p["sd"] ** 2 / p["n"]})
    return pd.DataFrame(rows)


def _noise_shares(c: pd.DataFrame) -> dict:
    """From change-score trials only: the share expected to show a rise of 10 or more on each arm."""
    if c.empty:
        return {"k_for_shares": 0}
    mp = float(np.average(c["mean_placebo"], weights=c["n_placebo"]))
    sp = float(np.sqrt(np.average(c["sd_placebo"] ** 2, weights=c["n_placebo"])))
    ma = float(np.average(c["mean_active"], weights=c["n_active"]))
    sa = float(np.sqrt(np.average(c["sd_active"] ** 2, weights=c["n_active"])))
    return {"k_for_shares": int(len(c)), "mean_change_placebo": mp, "sd_change_placebo": sp,
            "mean_change_active": ma, "sd_change_active": sa,
            "share_rise10_placebo": float(sps.norm.sf(10, mp, sp)),
            "share_rise10_active": float(sps.norm.sf(10, ma, sa))}


def variance_ratio_summary(vr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in ("sbp", "dbp", "hr"):
        g = vr[vr["kind"] == kind]
        if g.empty:
            continue
        p = stats.dersimonian_laird(g["ln_vr"], g["ln_vr_var"])
        m = stats.dersimonian_laird(g["md"], g["md_var"])
        sd_c = float(np.sqrt(np.average(g["sd_placebo"] ** 2, weights=g["n_placebo"])))
        ratio = float(np.exp(p.est))
        # between-person SD of the response: Var(active) - Var(placebo) = tau^2 when the response is
        # independent of the baseline; this holds for change scores and for final values alike
        tau = sd_c * np.sqrt(max(ratio ** 2 - 1, 0.0))
        lo, hi = np.exp(p.ci[0]), np.exp(p.ci[1])
        rows.append({"kind": kind, "k": p.k, "k_change": int((g["measure"] == "change").sum()),
                     "n_active": float(g["n_active"].sum()),
                     "n_placebo": float(g["n_placebo"].sum()),
                     "vr": ratio, "vr_lo": float(lo), "vr_hi": float(hi), "vr_p": p.p,
                     "vr_i2": p.i2, "vr_tau2": p.tau2,
                     "md": m.est, "md_lo": m.ci[0], "md_hi": m.ci[1], "md_i2": m.i2,
                     "sd_placebo_pooled": sd_c, "tau_response": float(tau),
                     "tau_lo": float(sd_c * np.sqrt(max(lo ** 2 - 1, 0.0))),
                     "tau_hi": float(sd_c * np.sqrt(max(hi ** 2 - 1, 0.0))),
                     "share_trials_vr_above_1": float((g["ln_vr"] > 0).mean()),
                     # share whose true drug effect is a rise of 10 or more, if responses ~ N(md, tau^2)
                     "true_rise10": float(sps.norm.sf(10, m.est, max(tau, 1e-9))),
                     "true_rise10_tau_hi": float(sps.norm.sf(10, m.est,
                                                             max(sd_c * np.sqrt(max(hi ** 2 - 1, 0.0)), 1e-9))),
                     **_noise_shares(g[g["measure"] == "change"])})
    return pd.DataFrame(rows)


def adverse_summary(t: dict, population: str = "adult") -> tuple[pd.DataFrame, pd.DataFrame]:
    s = adult_trials(t["studies"]) if population == "adult" else older_trials(t)
    s = s[(s["allocation"] == "RANDOMIZED") & (s["model"] == "PARALLEL")]
    e = t["adverse"]
    e = e[e["nct"].isin(s["nct"]) & e["role"].isin(["active", "placebo"])]
    # one count per arm and event group: the maximum over matching terms avoids double counting
    arm = (e.groupby(["nct", "group_key", "group", "role"])
             .agg(affected=("affected", "max"), at_risk=("at_risk", "max")).reset_index())
    trial = (arm.groupby(["nct", "group_key", "role"])
               .agg(affected=("affected", "sum"), at_risk=("at_risk", "sum")).reset_index())
    wide = trial.pivot_table(index=["nct", "group_key"], columns="role",
                             values=["affected", "at_risk"]).dropna()
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    rows = []
    for key, g in wide.groupby("group_key"):
        y, v = stats.log_risk_ratio(g["affected_active"], g["at_risk_active"],
                                    g["affected_placebo"], g["at_risk_placebo"])
        p = stats.dersimonian_laird(y, v)
        rows.append({"event": key, "k": p.k, "rr": float(np.exp(p.est)),
                     "rr_lo": float(np.exp(p.ci[0])), "rr_hi": float(np.exp(p.ci[1])), "i2": p.i2,
                     "risk_active": float(g["affected_active"].sum() / g["at_risk_active"].sum()),
                     "risk_placebo": float(g["affected_placebo"].sum() / g["at_risk_placebo"].sum()),
                     "n_active": float(g["at_risk_active"].sum()), "n_placebo": float(g["at_risk_placebo"].sum())})
    return pd.DataFrame(rows), wide


# =========================================================================== FAERS
def faers_rors(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in counts.iterrows():
        a = int(r["n_both"])
        b = int(r["n_drug"] - a)
        c = int(r["n_event"] - a)
        d = int(r["n_band"] - r["n_drug"] - r["n_event"] + a)
        ror = stats.reporting_odds_ratio(a, b, c, d)
        rows.append({**r.to_dict(), "ror": ror.ror, "ror_lo": ror.lo, "ror_hi": ror.hi,
                     "share_of_drug_reports": a / r["n_drug"] if r["n_drug"] else float("nan"),
                     "share_of_other_reports": c / (r["n_band"] - r["n_drug"]) if r["n_band"] > r["n_drug"] else float("nan")})
    return pd.DataFrame(rows)


def faers_age_contrast(rors: pd.DataFrame, old=("65-74", "75+"), young="18-44") -> pd.DataFrame:
    """Ratio of the ROR in older bands to the ROR in 18-44, per drug and event group."""
    rows = []
    for (g, e), x in rors.groupby(["drug_group", "event_group"]):
        y = x[x["age_band"] == young]
        if y.empty or not np.isfinite(y["ror"].iloc[0]):
            continue
        yr = y.iloc[0]
        lv_y = 1 / yr["n_both"] + 1 / (yr["n_drug"] - yr["n_both"]) + \
            1 / (yr["n_event"] - yr["n_both"]) + 1 / (yr["n_band"] - yr["n_drug"] - yr["n_event"] + yr["n_both"])
        for band in old:
            o = x[x["age_band"] == band]
            if o.empty or not np.isfinite(o["ror"].iloc[0]) or o["n_both"].iloc[0] < 3:
                continue
            orr = o.iloc[0]
            lv_o = 1 / orr["n_both"] + 1 / (orr["n_drug"] - orr["n_both"]) + \
                1 / (orr["n_event"] - orr["n_both"]) + 1 / (orr["n_band"] - orr["n_drug"] - orr["n_event"] + orr["n_both"])
            lr = np.log(orr["ror"] / yr["ror"])
            se = np.sqrt(lv_o + lv_y)
            rows.append({"drug_group": g, "event_group": e, "band": band, "ratio": float(np.exp(lr)),
                         "lo": float(np.exp(lr - 1.96 * se)), "hi": float(np.exp(lr + 1.96 * se)),
                         "n_both_old": int(orr["n_both"])})
    return pd.DataFrame(rows)


# =========================================================================== Medicare
def medicare_trend(m: pd.DataFrame) -> pd.DataFrame:
    g = (m.groupby(["year", "drug_class"])
          .agg(claims=("Tot_Clms", "sum"), claims65=("GE65_Tot_Clms", "sum"),
               fills65=("GE65_Tot_30day_Fills", "sum"), benes65=("GE65_Tot_Benes", "sum"),
               benes=("Tot_Benes", "sum"))
          .reset_index())
    g["share65"] = g["claims65"] / g["claims"]
    return g


# =========================================================================== one-patient experiments
def n_of_1_table() -> pd.DataFrame:
    rows = []
    for d in (0.3, 0.5, 1.0):
        for k in range(1, 11):
            rows.append({"d": d, "k": k, "p_wrong": float(stats.p_wrong_direction(d, k))})
    return pd.DataFrame(rows)
