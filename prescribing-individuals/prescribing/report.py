"""Run every analysis on the committed extracts and write the numbers the article quotes."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from . import analyse, faers, medicare, nhanes, trials
from .paths import DATA, RESULTS, ensure

FALLBACK_MU, FALLBACK_TAU = 2.0, 0.0


def _pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def compute() -> dict:
    out: dict = {}
    if (DATA / "trials_studies.csv.gz").exists():
        t = trials.load()
        out["trial_inventory"] = analyse.trial_inventory(t)
        tables = [analyse.variance_ratio_table(t, pop) for pop in ("all", "adult", "older")]
        out["vr_table"] = pd.concat([x for x in tables if len(x)], ignore_index=True) if any(len(x) for x in tables) \
            else pd.DataFrame()
        sums = []
        for pop, x in zip(("all", "adult", "older"), tables):
            if len(x):
                sums.append(analyse.variance_ratio_summary(x).assign(population=pop))
        out["vr_summary"] = pd.concat(sums, ignore_index=True) if sums else pd.DataFrame()
        out["ae_summary"], out["ae_table"] = analyse.adverse_summary(t, "adult")
        out["ae_summary_older"], _ = analyse.adverse_summary(t, "older")
    if (DATA / "nhanes_adults.csv.gz").exists():
        df = nhanes.load()
        out["nhanes_n"] = int(len(df))
        out["nhanes_cycles"] = sorted(df["cycle"].unique().tolist())
        out["baselines"] = analyse.nhanes_baselines(df)
        mu, tau, tau_hi, source = FALLBACK_MU, FALLBACK_TAU, None, "fallback (no trial estimate)"
        vs = out.get("vr_summary")
        if vs is not None and len(vs):
            for pop, label in (("all", "pooled trials, all ages"), ("adult", "pooled adult trials")):
                r = vs[(vs["kind"] == "sbp") & (vs["population"] == pop)]
                if len(r) and r.iloc[0]["k"] >= 3:
                    mu, tau, source = float(r.iloc[0]["md"]), float(r.iloc[0]["tau_response"]), label
                    tau_hi = float(r.iloc[0]["tau_hi"])
                    break
        out["crossing_params"] = {"mu": mu, "tau": tau, "tau_hi": tau_hi, "source": source}
        out["crossing"] = analyse.nhanes_crossing(df, mu, tau, tau_hi=tau_hi)
        out["absolute_risk"] = analyse.nhanes_absolute_risk(df, mu, tau)
        x = analyse.risk_frame(df, mu, tau)
        if len(x):
            out["risk_spread"] = analyse.nhanes_risk_spread(x)
            out["age_rule"] = analyse.age_rule(x)
        out["replicates"] = analyse.nhanes_replicates(df)
        out["stim_by_band"], out["stim_by_cycle"] = analyse.nhanes_stimulants(df)
    if (DATA / "faers_counts.csv.gz").exists():
        counts, meta = faers.load()
        out["faers_meta"] = meta
        out["faers_rors"] = analyse.faers_rors(counts)
        out["faers_contrast"] = analyse.faers_age_contrast(out["faers_rors"])
    if (DATA / "medicare_national.csv.gz").exists():
        out["medicare"] = analyse.medicare_trend(medicare.load())
    out["n_of_1"] = analyse.n_of_1_table()
    return out


def write(out: dict) -> str:
    ensure()
    buf = io.StringIO()
    with redirect_stdout(buf):
        p = print
        pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 200,
                      "display.float_format", lambda v: f"{v:.4g}")
        if "trial_inventory" in out:
            p("== ClinicalTrials.gov: stimulant trials with posted results, all indications")
            for k, v in out["trial_inventory"].items():
                p(f"  {k}: {v}")
            p("\n== Variance ratios, active versus placebo (change from baseline)")
            p(out["vr_summary"].to_string(index=False) if len(out["vr_summary"]) else "  none")
            p("\n  per trial:")
            if len(out["vr_table"]):
                p(out["vr_table"][["population", "nct", "kind", "n_active", "n_placebo", "mean_active",
                                   "mean_placebo", "sd_active", "sd_placebo", "ln_vr"]].to_string(index=False))
            p("\n== Adverse events, active versus placebo (randomised parallel adult trials)")
            p(out["ae_summary"].to_string(index=False) if len(out["ae_summary"]) else "  none")
            p("\n== Adverse events, active versus placebo (randomised parallel trials of older people)")
            p(out["ae_summary_older"].to_string(index=False) if len(out["ae_summary_older"]) else "  none")
        if "baselines" in out:
            p(f"\n== NHANES: {out['nhanes_n']:,} examined adults, cycles {', '.join(out['nhanes_cycles'])}")
            p("  blood pressure and cardiovascular baselines, oscillometric cycles (2017-2023), survey-weighted")
            p(out["baselines"].to_string(index=False))
            cp = out["crossing_params"]
            hi = f", upper 95% limit {cp['tau_hi']:.2f}" if cp.get("tau_hi") is not None else ""
            p(f"\n  stimulant response model: mean {cp['mu']:.2f} mmHg, between-person SD {cp['tau']:.2f} mmHg{hi} "
              f"({cp['source']})")
            p(out["crossing"].to_string(index=False))
            p("\n  extra 10-year cardiovascular risk (Pooled Cohort Equations) from a sustained systolic rise,"
              " adults 40-79 without cardiovascular history")
            p(out["absolute_risk"].to_string(index=False) if len(out["absolute_risk"]) else "  not available")
            if "risk_spread" in out:
                p("\n  spread of 10-year risk (risk_*) and of the extra risk from +5 mmHg (extra5_*) inside each age band")
                p(out["risk_spread"].to_string(index=False))
                p("\n  an age cutoff as a classifier of high 10-year risk (20% or more), adults 40-79 without CVD")
                p(out["age_rule"].to_string(index=False))
            p("\n  within-visit SD of replicate systolic readings and readings needed (each period) for 80% power")
            p(out["replicates"].to_string(index=False))
            p("\n  prescription stimulant use, all cycles with drug data, survey-weighted")
            p(out["stim_by_band"].to_string(index=False))
            p(out["stim_by_cycle"].to_string(index=False))
        if "faers_rors" in out:
            p(f"\n== FAERS via openFDA (database last updated {out['faers_meta'].get('last_updated')}, "
              f"{out['faers_meta'].get('total_reports'):,} reports)")
            cols = ["age_band", "drug_group", "event_group", "n_band", "n_drug", "n_event", "n_both",
                    "share_of_drug_reports", "ror", "ror_lo", "ror_hi"]
            p(out["faers_rors"][cols].to_string(index=False))
            p("\n  ratio of ROR (older band / 18-44)")
            p(out["faers_contrast"].to_string(index=False) if len(out["faers_contrast"]) else "  none")
        if "medicare" in out:
            p("\n== Medicare Part D, national claims")
            p(out["medicare"].to_string(index=False))
        p("\n== Chance an on/off comparison points the wrong way (k pairs, benefit d in noise units)")
        t = out["n_of_1"].pivot(index="k", columns="d", values="p_wrong")
        p(t.to_string())
    text = buf.getvalue()
    (RESULTS / "summary.txt").write_text(text)
    serial = {k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v) for k, v in out.items()}
    (RESULTS / "results.json").write_text(json.dumps(serial, indent=1, default=float))
    return text
