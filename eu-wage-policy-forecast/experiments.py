"""Model experiments for the forecast.

    python experiments.py

1. Floor grid. Steady states of the economy without the equal-pay rule for minimum wages m from
   0.40 to 1.10, eight seeds each. forecast.py reads every outcome as a function of the Kaitz index
   and evaluates it along the forecast paths.
2. Calibration of the equal-pay rule. At Portugal's floor (m = 0.65: about 22% of workers at the
   minimum and a Kaitz index of 0.63, close to the 2022 SES value of 0.64 and to the Quadros de
   Pessoal share at the minimum), the rule is switched on for every firm and the probability that a
   raise must be extended to the peers ("visibility") is set so that the average wage three to five
   years later is 1%, 2% or 3% below the same economy without the rule. 2% is the US estimate of
   Cullen and Pakzad-Hurson (2023); 1% and 3% bracket it.
3. Coverage scenarios. With the central visibility, the rule is switched on for the largest firms
   covering the share of employees above 150 persons (the 2027 reporting duty), above 50 (2031) and
   all employees (the right to information, which reaches every employer), and followed for eight
   years. Sensitivity: the 50+ coverage with the 1% and 3% visibilities, and full coverage at a
   low floor (m = 0.45), where the minimum wage does not pin the bottom.

Writes results/model_floor_grid*.csv, results/model_calibration.csv, results/model_reform*.csv and
results/model_summary.txt.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from inputs import REPORTING_THRESHOLDS, TRANSPARENCY_TARGET_YEARS, TRANSPARENCY_TARGETS
from model import Params, simulate, simulate_reform, with_

HERE = Path(__file__).parent
RESULTS = HERE / "results"

BASE = Params()
M_PT = 0.65                                     # Portugal's floor in model units
M_GRID = np.round(np.arange(0.40, 1.1001, 0.025), 3)
GRID_SEEDS = range(8)
CAL_SEEDS = range(6)
REFORM_SEEDS = range(10)
YEARS = 8
WORKERS = max(1, min(4, os.cpu_count() or 1))


def _grid_point(args: tuple[float, int]) -> dict:
    m, seed = args
    return {"m": m, "seed": seed, **simulate(with_(BASE, m=m, seed=seed))}


def _reform(args: tuple[Params, int]) -> pd.DataFrame:
    prm, years = args
    return simulate_reform(prm, years).assign(seed=prm.seed, cover_share=prm.cover_share,
                                              visibility=prm.visibility, m=prm.m)


def run_reforms(prms: list[Params], years: int = YEARS) -> pd.DataFrame:
    with ProcessPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(_reform, [(p, years) for p in prms]))
    return pd.concat(parts, ignore_index=True)


def reform_effects(raw: pd.DataFrame) -> pd.DataFrame:
    """Reform minus baseline, by seed and year: relative change for levels, difference for rates."""
    keys = ["seed", "cover_share", "visibility", "m", "year"]
    r = raw[raw.run == "reform"].set_index(keys)
    b = raw[raw.run == "baseline"].set_index(keys)
    rel = ["mean_wage", "p10", "p25", "p50", "p75", "p90", "mean_bottom25"]
    pts = ["employment_rate", "labour_share", "job_to_job_per_year", "counter_raises_per_year",
           "counters_refused_per_year", "share_at_floor"]
    out = pd.concat([(r[rel] / b[rel] - 1).add_prefix("d_"),
                     (r[pts] - b[pts]).add_prefix("d_"),
                     (r["log_gain_at_move"] - b["log_gain_at_move"]).rename("d_log_gain_at_move"),
                     (r["mean_wage_covered"] / b["mean_wage"] - 1).rename("d_mean_wage_covered_vs_all"),
                     r["covered_emp_share"]], axis=1)
    return out.reset_index()


def calibrate(target: float) -> tuple[float, float]:
    """Visibility at which full coverage lowers the mean wage by `target` over TARGET_YEARS (bisection in log)."""
    lo, hi = np.log(1e-3), np.log(1.0)

    def effect(log_pi: float) -> float:
        prms = [with_(BASE, m=M_PT, cover_share=1.0, visibility=float(np.exp(log_pi)), seed=s) for s in CAL_SEEDS]
        eff = reform_effects(run_reforms(prms, years=max(TRANSPARENCY_TARGET_YEARS)))
        return float(eff[eff.year.isin(TRANSPARENCY_TARGET_YEARS)].d_mean_wage.mean())

    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if effect(mid) > target:        # too little effect: more visibility
            lo = mid
        else:
            hi = mid
    pi = float(np.exp(0.5 * (lo + hi)))
    return round(pi, 4), effect(np.log(round(pi, 4)))


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    L = ["# Model experiments (model.py, experiments.py)", "",
         f"Baseline: {BASE}", f"Portugal's floor in model units: m = {M_PT}", ""]

    # 1. floor grid
    with ProcessPoolExecutor(WORKERS) as ex:
        grid = pd.DataFrame(list(ex.map(_grid_point, [(m, s) for m in M_GRID for s in GRID_SEEDS])))
    grid.to_csv(RESULTS / "model_floor_grid_raw.csv", index=False)
    g = grid.drop(columns="seed").groupby("m").mean().reset_index()
    g.to_csv(RESULTS / "model_floor_grid.csv", index=False)
    L.append(f"## 1. Floor grid: steady states, {len(GRID_SEEDS)} seeds, mean over seeds")
    for r in g.itertuples():
        if round(r.m / 0.05, 6) == round(r.m / 0.05):
            L.append(f"- m={r.m:.2f}: kaitz {r.kaitz:.3f}, at floor {100 * r.share_at_floor:.1f}%, employment {100 * r.employment_rate:.2f}%, "
                     f"P10/P50 {r.p10_p50:.3f}, P25/P50 {r.p25_p50:.3f}, bottom-quarter mean/P50 {r.mean_bottom25 / r.p50:.3f}, "
                     f"job-to-job {100 * r.job_to_job_per_year:.2f}%/yr, log gain at move {r.log_gain_at_move:.3f}, "
                     f"counter raises {100 * r.counter_raises_per_year:.2f}%/yr, labour share {100 * r.labour_share:.1f}%, "
                     f"mean output {r.mean_output:.4f}")

    # 2. calibration of visibility
    cal = []
    for target in TRANSPARENCY_TARGETS:
        pi, achieved = calibrate(target)
        cal.append({"target": target, "visibility": pi, "achieved": achieved})
    cal = pd.DataFrame(cal)
    cal.to_csv(RESULTS / "model_calibration.csv", index=False)
    L += ["", f"## 2. Visibility calibrated at m = {M_PT}, full coverage, mean wage effect averaged over years "
          f"{TRANSPARENCY_TARGET_YEARS[0]}-{TRANSPARENCY_TARGET_YEARS[-1]} ({len(CAL_SEEDS)} seeds)"]
    for r in cal.itertuples():
        L.append(f"- target {100 * r.target:+.0f}%: visibility {r.visibility:.4f} (achieved {100 * r.achieved:+.2f}%)")

    # 3. coverage scenarios
    cov = pd.read_csv(RESULTS / "coverage.csv").set_index("geo")
    s150 = round(float(cov.loc["PT", "share_ge150"]), 3)
    s50 = round(float(cov.loc["PT", "share_ge50"]), 3)
    pi_c = float(cal.loc[cal.target == -0.02, "visibility"].iloc[0])
    pi_lo = float(cal.loc[cal.target == -0.01, "visibility"].iloc[0])
    pi_hi = float(cal.loc[cal.target == -0.03, "visibility"].iloc[0])
    scen = []
    for share in (s150, s50, 1.0):
        scen += [with_(BASE, m=M_PT, cover_share=share, visibility=pi_c, seed=s) for s in REFORM_SEEDS]
    for pi in (pi_lo, pi_hi):
        scen += [with_(BASE, m=M_PT, cover_share=s50, visibility=pi, seed=s) for s in REFORM_SEEDS]
    scen += [with_(BASE, m=0.45, cover_share=1.0, visibility=pi_c, seed=s) for s in REFORM_SEEDS]
    raw = run_reforms(scen)
    raw.to_csv(RESULTS / "model_reform_raw.csv", index=False)
    eff = reform_effects(raw)
    eff.to_csv(RESULTS / "model_reform_effects.csv", index=False)
    summ = eff.drop(columns="seed").groupby(["m", "cover_share", "visibility", "year"]).agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(RESULTS / "model_reform.csv", index=False)
    labels = {s150: f"employees in firms >= {REPORTING_THRESHOLDS[2027]} ({100 * s150:.1f}%)",
              s50: f"employees in firms >= {REPORTING_THRESHOLDS[2031]} ({100 * s50:.1f}%)", 1.0: "all employees"}
    L += ["", f"## 3. Equal-pay rule switched on, effects against the same economy without it "
          f"(mean over {len(REFORM_SEEDS)} seeds, sd in brackets)"]
    for (m, share, pi), gsub in summ.groupby(["m", "cover_share", "visibility"]):
        L.append(f"### m={m}, coverage {labels.get(share, share)}, visibility {pi}, covered share of employment "
                 f"{100 * gsub.covered_emp_share_mean.iloc[0]:.1f}% in year 1")
        for r in gsub[gsub.year.isin([1, 2, 3, 4, 5, 8])].itertuples():
            L.append(f"- year {r.year}: mean {100 * r.d_mean_wage_mean:+.2f}% ({100 * r.d_mean_wage_std:.2f}); "
                     f"P10 {100 * r.d_p10_mean:+.2f}%; P25 {100 * r.d_p25_mean:+.2f}%; P50 {100 * r.d_p50_mean:+.2f}%; "
                     f"P75 {100 * r.d_p75_mean:+.2f}%; P90 {100 * r.d_p90_mean:+.2f}% ({100 * r.d_p90_std:.2f}); "
                     f"employment {100 * r.d_employment_rate_mean:+.2f} pt ({100 * r.d_employment_rate_std:.2f}); "
                     f"job-to-job {100 * r.d_job_to_job_per_year_mean:+.2f} pt; counter raises {100 * r.d_counter_raises_per_year_mean:+.2f} pt; "
                     f"refused counters {100 * r.d_counters_refused_per_year_mean:+.2f} pt; "
                     f"log gain at move {r.d_log_gain_at_move_mean:+.3f}; labour share {100 * r.d_labour_share_mean:+.2f} pt")
    (RESULTS / "model_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
