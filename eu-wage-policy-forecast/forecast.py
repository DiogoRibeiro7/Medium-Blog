"""Scenario forecasts with uncertainty, 2026-2029 for Portugal and the 60% reference for the EU.

    python forecast.py

Reads results/empirics.json, results/*_draws.csv, results/coverage.csv, results/eu_reference.csv
(empirics.py) and results/model_floor_grid_raw.csv, results/model_reform.csv (experiments.py).

Portugal. Each Monte Carlo draw combines four sources of uncertainty:
  * wage growth: compensation per employee grows at the average of the published projections plus
    an AR(1) deviation fitted to 1996-2025 (2026 has a smaller spread: half of it is observed);
  * the median: it grows with compensation per employee, plus a share theta of the minimum wage's
    excess growth, plus Portugal's own drift (both drawn from their bootstrap distributions);
  * the bottom: P10/median follows the Kaitz index with one of three elasticities (EU linear,
    EU bite-dependent, Portugal's own), each drawn from its bootstrap distribution;
  * the model: outcomes are read off the steady-state floor grid of a randomly drawn seed.
The four minimum-wage paths share the draws (common random numbers), so the difference between
two paths is measured with less noise than either path.

Employment is reported two ways: the model's own match destruction (an upper bound in its logic),
and published own-wage elasticities applied to the wage gain of the quarter of employees nearest
the floor.

Pay transparency combines the coverage shares (empirics.py) with published effects and with the
model's equal-pay experiments (experiments.py).

EU. For each country below 60% of the median: the change in P10/median and in the low-pay share if
the minimum wage moved to 60% with the median unchanged, from the bite-dependent elasticity.

Writes results/forecast_*.csv and results/forecast_summary.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from inputs import (EVIDENCE, KAITZ_BASE, MW_2029_GRID, MW_HISTORY, MW_PATHS, OWE, PT_GAP_2023,
                    REFERENCE_KAITZ, REPORTING_THRESHOLDS, SD_2026_PP, WAGE_PROJECTIONS)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
N_DRAWS = 4000
SEED = 2027
YEARS = (2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029)
Q = (5, 25, 50, 75, 95)
MODEL_OUTCOMES = ("employment_rate", "share_at_floor", "p25_p50", "job_to_job_per_year", "log_gain_at_move",
                  "counter_raises_per_year", "labour_share", "mean_output")


def central_growth() -> dict[int, float]:
    """Average of the published projections for each year, % growth."""
    out = {}
    for y in (2026, 2027, 2028, 2029):
        vals = [v[y] for v in WAGE_PROJECTIONS.values() if y in v]
        out[y] = float(np.mean(vals))
    return out


def draw_growth(emp: dict, rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Log growth of compensation per employee by year: observed to 2025, simulated 2026-2029."""
    gp = emp["growth"]
    obs = {int(k): v for k, v in gp["observed"].items()}
    c = central_growth()
    g = {y: np.full(N_DRAWS, obs[y]) for y in (2023, 2024, 2025)}
    u = rng.normal(0, SD_2026_PP / 100, N_DRAWS)
    g[2026] = np.log1p(c[2026] / 100) + u
    for y in (2027, 2028, 2029):
        u = gp["rho"] * u + rng.normal(0, gp["sigma"], N_DRAWS)
        g[y] = np.log1p(c[y] / 100) + u
    return g


def simulate_paths(emp: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    """Minimum wage, SES median, SES Kaitz index and base-wage Kaitz index for every path and draw."""
    g = draw_growth(emp, rng)
    spill = pd.read_csv(RESULTS / "spillover_draws.csv")
    idx = rng.integers(0, len(spill), N_DRAWS)
    theta = spill.theta.to_numpy()[idx]
    drift = spill.drift.to_numpy()[idx]
    ses = emp["pt_ses_2022"]
    out = {}
    for name, (label, path) in MW_PATHS.items():
        mw = {y: np.full(N_DRAWS, v) for y, v in MW_HISTORY.items()}
        med = {2022: np.full(N_DRAWS, ses["median"])}
        base = {2025: np.full(N_DRAWS, MW_HISTORY[2025] / KAITZ_BASE[2025])}
        for y in range(2023, 2030):
            if y >= 2027:
                if path[y] is None:   # indexed: g_mw = g_med, so g_med = g_avg + drift / (1 - theta)
                    mw[y] = mw[y - 1] * np.exp(g[y] + drift / (1 - theta))
                else:
                    mw[y] = np.full(N_DRAWS, path[y])
            g_mw = np.log(mw[y] / mw[y - 1])
            g_med = g[y] + theta * (g_mw - g[y]) + drift
            med[y] = med[y - 1] * np.exp(g_med)
            if y >= 2026:
                base[y] = base[y - 1] * np.exp(g_med)
        out[name] = {
            "mw": mw, "median_ses": med, "median_base": base,
            "kaitz_ses": {y: mw[y] / med[y] for y in med},
            "kaitz_base": {y: mw[y] / base[y] for y in base},
        }
    # risk curve: 2027 and 2028 on the agreement, 2029 scanned
    base28 = out["agreement"]["median_base"][2028]
    risk = []
    for x in MW_2029_GRID:
        gm = np.log(x / MW_PATHS["agreement"][1][2028])
        b29 = base28 * np.exp(g[2029] + theta * (gm - g[2029]) + drift)
        k = x / b29
        risk.append({"mw_2029": x, "p_ge_095": float(np.mean(k >= 0.95)), "p_ge_100": float(np.mean(k >= 1.0)),
                     "kaitz_base_q50": float(np.median(k)), "gap_eur_q50": float(np.median(b29 - x))})
    return out, {"theta": theta, "drift": drift, "growth": g, "risk": pd.DataFrame(risk)}


def bottom(paths: dict, emp: dict, rng: np.random.Generator) -> dict:
    """ln(P10/median) in each year from the change in ln Kaitz since 2022, for three elasticities."""
    el = pd.read_csv(RESULTS / "elasticity_draws.csv")
    idx = rng.integers(0, len(el), N_DRAWS)
    e = {c: el[c].to_numpy()[idx] for c in el.columns}
    ses = emp["pt_ses_2022"]
    l0, k0 = np.log(ses["p10_over_median"]), np.log(ses["kaitz"])
    out = {}
    for name, d in paths.items():
        out[name] = {}
        for y, k in d["kaitz_ses"].items():
            lk = np.log(k)
            out[name][y] = {
                "EU linear": l0 + e["lin"] * (lk - k0),
                "EU bite-dependent": l0 + e["quad_b1"] * (lk - k0) + e["quad_b2"] * (lk ** 2 - k0 ** 2),
                "Portugal's own": l0 + e["pt_own"] * (lk - k0),
            }
    return out


def model_outcomes(paths: dict, rng: np.random.Generator) -> dict:
    """Steady-state model outcomes at each draw's SES Kaitz index, from a randomly drawn seed's grid."""
    raw = pd.read_csv(RESULTS / "model_floor_grid_raw.csv")
    seeds = sorted(raw.seed.unique())
    pick = rng.integers(0, len(seeds), N_DRAWS)
    curves = {s: raw[raw.seed == s].sort_values("kaitz") for s in seeds}
    out = {}
    for name, d in paths.items():
        out[name] = {}
        for y in (2026, 2029):
            k = d["kaitz_ses"][y]
            vals = {o: np.empty(N_DRAWS) for o in MODEL_OUTCOMES}
            for j, s in enumerate(seeds):
                sel = pick == j
                cv = curves[s]
                for o in MODEL_OUTCOMES:
                    vals[o][sel] = np.interp(k[sel], cv.kaitz.to_numpy(), cv[o].to_numpy())
            out[name][y] = vals
    return out


def q(a: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(a, Q)]


def fmt(a: np.ndarray, scale: float = 1.0, nd: int = 3, sign: bool = False) -> str:
    # rounding first and adding 0.0 turns -0.0 into 0.0, so last-bit noise cannot flip a printed sign
    lo, mid, hi = np.round(np.percentile(a * scale, [5, 50, 95]), nd) + 0.0
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(mid)} [{f.format(lo)}, {f.format(hi)}]"


def main() -> int:
    rng = np.random.default_rng(SEED)
    emp = json.loads((RESULTS / "empirics.json").read_text())
    paths, aux = simulate_paths(emp, rng)
    bot = bottom(paths, emp, rng)
    mod = model_outcomes(paths, rng)
    ref = "constant_bite"
    L = ["# Forecasts (forecast.py)", "",
         f"{N_DRAWS} draws; quantiles reported as median [5th, 95th percentile].", ""]

    # ---------------------------------------------------------------- 1. the bite of each path
    c = central_growth()
    L.append("## 1. Portugal: the minimum wage against the median, 2026-2029")
    L.append("- Central wage growth (average of published projections, %): " + ", ".join(f"{y} {v:.2f}" for y, v in c.items()))
    L.append(f"- Spillover theta {fmt(aux['theta'])}; drift {fmt(aux['drift'], 100, 2)} pt a year")
    rows = []
    for name, d in paths.items():
        label = MW_PATHS[name][0]
        for y in YEARS:
            for concept, key in (("SES October earnings", "kaitz_ses"), ("base wage (Banco de Portugal)", "kaitz_base")):
                if y in d[key]:
                    rows.append({"path": name, "label": label, "year": y, "concept": concept,
                                 **{f"q{p}": v for p, v in zip(Q, q(d[key][y]))}})
            rows.append({"path": name, "label": label, "year": y, "concept": "minimum wage (euro)",
                         **{f"q{p}": v for p, v in zip(Q, q(d["mw"][y]))}})
        L.append(f"- {label}: minimum wage 2029 {fmt(d['mw'][2029], 1, 0)}; SES Kaitz 2026 {fmt(d['kaitz_ses'][2026])}, "
                 f"2029 {fmt(d['kaitz_ses'][2029])}; base-wage Kaitz 2026 {fmt(d['kaitz_base'][2026])}, "
                 f"2029 {fmt(d['kaitz_base'][2029])}; P(base-wage Kaitz 2029 >= 0.95) {np.mean(d['kaitz_base'][2029] >= 0.95):.2f}; "
                 f"median base wage minus minimum wage, 2029 (euro) {fmt(d['median_base'][2029] - d['mw'][2029], 1, 0)}")
    kp = pd.DataFrame(rows)
    kp.to_csv(RESULTS / "forecast_kaitz_paths.csv", index=False)
    for name in MW_PATHS:
        if name == ref:
            continue
        dk = np.log(paths[name]["kaitz_ses"][2029]) - np.log(paths[ref]["kaitz_ses"][2029])
        p65 = np.mean(paths[name]["kaitz_ses"][2029] > emp["elasticity"]["kaitz_max"])
        L.append(f"- {MW_PATHS[name][0]} against the indexed path, 2029: log Kaitz {fmt(dk, 1, 3, True)}; "
                 f"P(SES Kaitz 2029 above the sample maximum {emp['elasticity']['kaitz_max']:.3f}) {p65:.2f}; "
                 f"P(base-wage Kaitz 2029 >= 1) {np.mean(paths[name]['kaitz_base'][2029] >= 1):.2f}")
    extra = {nm: sum(14 * (paths[nm]["mw"][y] - paths["agreement"]["mw"][y]) for y in (2027, 2028, 2029))
             for nm in ("programme", "front_loaded", "union_2027")}
    L.append("- Extra gross pay 2027-2029 for a full-time minimum-wage worker against the agreement path (14 payments): " +
             "; ".join(f"{MW_PATHS[nm][0]} {fmt(v, 1, 0)} euro" for nm, v in extra.items()))
    risk = aux["risk"]
    risk.to_csv(RESULTS / "forecast_risk_curve.csv", index=False)
    for thr in (0.2, 0.5):
        hit = risk[risk.p_ge_095 >= thr]
        L.append(f"- Risk curve (2027-2028 on the agreement): lowest 2029 minimum wage with P(base-wage Kaitz >= 0.95) >= {thr:.0%}: "
                 f"{int(hit.mw_2029.iloc[0]) if len(hit) else 'none in grid'}")
    for thr in (0.2, 0.5):
        hit = risk[risk.p_ge_100 >= thr]
        L.append(f"- Risk curve: lowest 2029 minimum wage with P(base-wage Kaitz >= 1) >= {thr:.0%}: "
                 f"{int(hit.mw_2029.iloc[0]) if len(hit) else 'none in grid'}")
    for x in (1070, 1100):
        r = risk[risk.mw_2029 == x].iloc[0]
        L.append(f"- 2029 minimum wage {x}: P(base Kaitz >= 0.95) {r.p_ge_095:.2f}, P(>= 1) {r.p_ge_100:.2f}, "
                 f"median gap to the median base wage {r.gap_eur_q50:.0f} euro")
    eur = pd.read_csv(HERE / "data" / "minimum_wage_ratio.csv")
    eur = eur[(eur.geo == "PT") & (eur.indic_se == "MMW_MED_ME_PP") & (eur.nace_r2 == "B-S")].set_index("time").value
    L.append(f"- Cross-check: Eurostat's 12-month ratio to the median, PT, {eur[2022]:.1f}% (2022) and {eur[2026]:.1f}% (2026), "
             f"which scales the SES 2022 index {emp['pt_ses_2022']['kaitz']:.3f} to {emp['pt_ses_2022']['kaitz'] * eur[2026] / eur[2022]:.3f} in 2026")
    L.append(f"- Base-wage Kaitz 2026 nowcast vs the {100 * KAITZ_BASE[2025]:.0f}% of 2025: {fmt(paths[ref]['kaitz_base'][2026])}; "
             f"median base wage 2026 (euro) {fmt(paths[ref]['median_base'][2026], 1, 0)}")

    # ---------------------------------------------------------------- 2. the bottom
    L += ["", "## 2. Portugal: P10/median (SES October earnings) in 2029, by elasticity"]
    ses = emp["pt_ses_2022"]
    L.append(f"- 2022 observed: P10/median {ses['p10_over_median']:.3f}, minimum wage/P10 {ses['mw_paid'] / ses['p10']:.3f}")
    rows = []
    for name in MW_PATHS:
        for model, lv in bot[name][2029].items():
            r = np.exp(lv)
            mw_p10 = paths[name]["kaitz_ses"][2029] / r
            rows.append({"path": name, "elasticity": model, **{f"q{p}": v for p, v in zip(Q, q(r))},
                         **{f"mw_p10_q{p}": v for p, v in zip(Q, q(mw_p10))}})
            L.append(f"- {MW_PATHS[name][0]}, {model}: P10/median {fmt(r)}, minimum wage/P10 {fmt(mw_p10)}; "
                     f"P(minimum wage/P10 >= 0.95) {np.mean(mw_p10 >= 0.95):.2f}")
    pd.DataFrame(rows).to_csv(RESULTS / "forecast_bottom.csv", index=False)
    for name in [nm for nm in MW_PATHS if nm != ref]:
        for model in bot[name][2029]:
            dd = bot[name][2029][model] - bot[ref][2029][model]
            L.append(f"- {MW_PATHS[name][0]} against the indexed path, {model}: log P10/median {fmt(dd, 1, 3, True)}")

    # ---------------------------------------------------------------- 3. model outcomes
    L += ["", "## 3. Portugal: model outcomes (steady state at the 2029 Kaitz index, seeds drawn), change from 2026"]
    rows = []
    for name in MW_PATHS:
        parts = []
        for o in MODEL_OUTCOMES:
            dlt = mod[name][2029][o] - mod[name][2026][o]
            if o == "mean_output":
                dlt = mod[name][2029][o] / mod[name][2026][o] - 1
            rows.append({"path": name, "outcome": o, "level_2026_q50": float(np.median(mod[name][2026][o])),
                         **{f"q{p}": v for p, v in zip(Q, q(dlt))}})
            scale, nd = (100, 2)
            parts.append(f"{o} {fmt(dlt, scale, nd, True)}" + ("%" if o == "mean_output" else (" pt" if o != "log_gain_at_move" else "")))
        L.append(f"- {MW_PATHS[name][0]}: " + "; ".join(parts))
    pd.DataFrame(rows).to_csv(RESULTS / "forecast_model_outcomes.csv", index=False)
    lvl = {o: float(np.median(mod[ref][2026][o])) for o in MODEL_OUTCOMES}
    L.append("- Model levels at the 2026 nowcast: " + ", ".join(f"{o} {v:.3f}" for o, v in lvl.items()))

    # ---------------------------------------------------------------- 4. employment
    L += ["", "## 4. Portugal: employment against the indexed path, 2029"]
    employees = emp["pt_employees_2025"]
    rows = []
    for name in [nm for nm in MW_PATHS if nm != ref]:
        # wage gain of the bottom quarter: log P10 against the indexed path (Portugal's own elasticity, central)
        lp10 = {nm: bot[nm][2029]["Portugal's own"] + np.log(paths[nm]["median_ses"][2029]) for nm in (name, ref)}
        dw = lp10[name] - lp10[ref]
        model_emp = (mod[name][2029]["employment_rate"] - mod[ref][2029]["employment_rate"]) / mod[ref][2029]["employment_rate"]
        line = [f"bottom-quarter wage gain {fmt(dw, 100, 2, True)}%"]
        for lab, owe in OWE.items():
            jobs = owe * dw * 0.25 * employees
            line.append(f"OWE {owe} ({lab}): {fmt(jobs, 1e-3, 1, True)} thousand jobs")
            rows.append({"path": name, "method": f"OWE {owe}", **{f"q{p}": v for p, v in zip(Q, q(jobs))}})
        jobs_m = model_emp * employees
        line.append(f"model match destruction: {fmt(jobs_m, 1e-3, 1, True)} thousand jobs ({fmt(model_emp, 100, 2, True)}%)")
        owe_model = model_emp / (0.25 * dw)
        line.append(f"implied model OWE {fmt(owe_model, 1, 2)}")
        rows.append({"path": name, "method": "model", **{f"q{p}": v for p, v in zip(Q, q(jobs_m))}})
        L.append(f"- {MW_PATHS[name][0]}: " + "; ".join(line))
    pd.DataFrame(rows).to_csv(RESULTS / "forecast_employment.csv", index=False)
    L.append(f"- Employees (national accounts, 2025): {employees / 1e6:.3f} million")

    # ---------------------------------------------------------------- 5. pay transparency
    cov = pd.read_csv(RESULTS / "coverage.csv").set_index("geo")
    s = {2027: float(cov.loc["PT", "share_ge150"]), 2031: float(cov.loc["PT", "share_ge50"])}
    L += ["", "## 5. Portugal: pay transparency"]
    L.append(f"- Employees covered by the reporting duty (business economy): from 2027 (>= {REPORTING_THRESHOLDS[2027]}) "
             f"{100 * s[2027]:.1f}%, from 2031 (>= {REPORTING_THRESHOLDS[2031]}) {100 * s[2031]:.1f}%")
    rows = []
    for y, share in s.items():
        for basis, gap in PT_GAP_2023.items():
            lo = gap * (1 + share * EVIDENCE["dk_gap_share"])
            hi = gap * (1 + share * EVIDENCE["uk_gap_share"])
            rows.append({"from": y, "basis": basis, "gap_2023": gap, "gap_after_dk": lo, "gap_after_uk": hi})
            L.append(f"- Gender gap on {basis} ({100 * gap:.1f}% in 2023) if covered firms close 13% (Denmark) to 19% (UK) of it, "
                     f"coverage from {y}: {100 * hi:.2f}% to {100 * lo:.2f}% ({100 * (hi - gap):+.2f} to {100 * (lo - gap):+.2f} pt)")
        L.append(f"- Male wages in covered firms -1.7% (Denmark) to -3% (UK): economy-wide male mean "
                 f"{100 * share * EVIDENCE['dk_male_wage']:+.2f}% to {100 * share * EVIDENCE['uk_male_wage']:+.2f}% (coverage from {y})")
    pd.DataFrame(rows).to_csv(RESULTS / "forecast_transparency_evidence.csv", index=False)
    mr = pd.read_csv(RESULTS / "model_reform.csv")
    cal = pd.read_csv(RESULTS / "model_calibration.csv")
    pi_c = float(cal.loc[cal.target == -0.02, "visibility"].iloc[0])
    for share, lab in ((round(s[2027], 3), "2027 duty"), (round(s[2031], 3), "2031 duty"), (1.0, "all employers")):
        sub = mr[(mr.m == 0.65) & (np.isclose(mr.cover_share, share)) & (np.isclose(mr.visibility, pi_c))]
        r5 = sub[sub.year == 5].iloc[0]
        L.append(f"- Model, coverage {lab} ({100 * share:.1f}%), year 5: mean {100 * r5.d_mean_wage_mean:+.2f}% "
                 f"(sd {100 * r5.d_mean_wage_std:.2f}), P25 {100 * r5.d_p25_mean:+.2f}%, P50 {100 * r5.d_p50_mean:+.2f}%, "
                 f"P90 {100 * r5.d_p90_mean:+.2f}%, counter raises {100 * r5.d_counter_raises_per_year_mean:+.2f} pt/yr, "
                 f"job-to-job {100 * r5.d_job_to_job_per_year_mean:+.2f} pt/yr, employment {100 * r5.d_employment_rate_mean:+.2f} pt")

    # ---------------------------------------------------------------- 6. EU reference value
    eu = pd.read_csv(RESULTS / "eu_reference.csv")
    el = pd.read_csv(RESULTS / "elasticity_draws.csv")
    emp_e = emp["elasticity"]
    L += ["", f"## 6. EU: moving each minimum wage to {100 * REFERENCE_KAITZ:.0f}% of the median (median unchanged)"]
    rows = []
    for r in eu.itertuples():
        if r.kaitz_oecd >= REFERENCE_KAITZ or not np.isfinite(r.kaitz_ses_2022):
            continue
        k0 = np.log(r.kaitz_ses_2022)
        k1 = k0 + r.dlog_to_reference
        dl = el.quad_b1.to_numpy() * (k1 - k0) + el.quad_b2.to_numpy() * (k1 ** 2 - k0 ** 2)
        dl_point = emp_e["quad_b1"] * (k1 - k0) + emp_e["quad_b2"] * (k1 ** 2 - k0 ** 2)
        lp = r.low_pay_share_2022 * np.exp(el.lowpay.to_numpy() * r.dlog_to_reference)
        lp_point = r.low_pay_share_2022 * np.exp(emp_e["lowpay"] * r.dlog_to_reference)
        beyond = float(np.exp(k1)) > emp_e["kaitz_max"]
        rows.append({"geo": r.geo, "kaitz_oecd": r.kaitz_oecd, "dlog_kaitz": r.dlog_to_reference,
                     "kaitz_ses_2022": r.kaitz_ses_2022, "kaitz_ses_after": float(np.exp(k1)), "beyond_sample": beyond,
                     "p10_med_2022": r.p10_over_median_2022, "dlog_p10_med": float(dl_point),
                     "dlog_p10_med_q5": float(np.percentile(dl, 5)), "dlog_p10_med_q95": float(np.percentile(dl, 95)),
                     "low_pay_2022": r.low_pay_share_2022, "low_pay_after": float(lp_point),
                     "low_pay_after_q5": float(np.percentile(lp, 5)), "low_pay_after_q95": float(np.percentile(lp, 95))})
    eur = pd.DataFrame(rows).sort_values("dlog_kaitz", ascending=False)
    eur.to_csv(RESULTS / "forecast_eu.csv", index=False)
    for r in eur.itertuples():
        L.append(f"- {r.geo}: Kaitz {100 * r.kaitz_oecd:.1f} -> 60 (+{100 * r.dlog_kaitz:.1f} log pt); SES Kaitz "
                 f"{r.kaitz_ses_2022:.3f} -> {r.kaitz_ses_after:.3f}{' (beyond the sample)' if r.beyond_sample else ''}; "
                 f"log P10/median {100 * r.dlog_p10_med:+.1f} [{100 * r.dlog_p10_med_q5:+.1f}, {100 * r.dlog_p10_med_q95:+.1f}] pt; "
                 f"low-pay share {r.low_pay_2022:.1f}% -> {r.low_pay_after:.1f}% [{r.low_pay_after_q5:.1f}, {r.low_pay_after_q95:.1f}]")
    L.append(f"- Countries whose move to 60% would take the SES Kaitz index beyond the sample maximum "
             f"({emp_e['kaitz_max']:.3f}): {', '.join(eur[eur.beyond_sample].geo) or 'none'}")

    (RESULTS / "forecast_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
