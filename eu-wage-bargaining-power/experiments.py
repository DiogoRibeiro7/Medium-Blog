"""Run the model experiments and write every number to results/.

    python experiments.py            # about 3 minutes on two cores

1. Lever fingerprints: each labour-law lever moved alone from a common baseline, averaged
   over seeds, as percentage changes of wage quantiles, employment, the labour share,
   job-to-job moves and average match output.
2. Floor grid: for minimum wages from 0.30 to 0.95 (in units of the no-floor median wage),
   the effect of a stronger bargaining weight and of more rival offers on each quantile,
   plotted against the Kaitz index the floor produces.
"""

from __future__ import annotations

import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from model import Params, simulate, with_

RESULTS = Path(__file__).parent / "results"
BASE = Params(m=0.45)
SEEDS_LEVERS = range(10)
SEEDS_GRID = range(6)

LEVERS: dict[str, dict] = {
    "baseline": {},
    "minimum wage 0.45 -> 0.60": {"m": 0.60},
    "bargaining weight 0.10 -> 0.25": {"beta": 0.25},
    "rival offers halved (non-competes)": {"lambda1": 0.025},
    "rival offers +50% (mobility)": {"lambda1": 0.075},
    "outside option 0.45 -> 0.55": {"b0": 0.55},
    "equal pay, broad job bands (5)": {"transparency": True, "n_bands": 5},
    "equal pay, narrow job bands (20)": {"transparency": True, "n_bands": 20},
}
OUTCOMES = ["p10", "p25", "p50", "p75", "p90", "mean_wage", "employment_rate", "labour_share",
            "job_to_job_per_year", "mean_output", "share_at_floor", "kaitz", "passthrough_firm",
            "counter_raises_per_year", "counters_refused_per_year"]

FLOORS = np.round(np.arange(0.30, 0.951, 0.05), 2)
GRID_VARIANTS = {"base": {}, "beta 0.10 -> 0.20": {"beta": 0.20}, "lambda1 0.05 -> 0.075": {"lambda1": 0.075}}


def _run(args):
    label, kw, seed = args
    r = simulate(with_(BASE, seed=seed, **kw))
    r.update(label=label, seed=seed)
    return r


def _run_grid(args):
    m, variant, seed = args
    r = simulate(with_(BASE, m=float(m), seed=seed, **GRID_VARIANTS[variant]))
    r.update(m=float(m), variant=variant, seed=seed)
    return r


def lever_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Percentage change against the baseline of the same seed, then mean and sd over seeds."""
    base = raw[raw.label == "baseline"].set_index("seed")
    rows = []
    for label, g in raw.groupby("label", sort=False):
        g = g.set_index("seed")
        for k in OUTCOMES:
            if k in ("kaitz", "passthrough_firm"):
                d = g[k] - base.loc[g.index, k]          # level differences of ratios and elasticities
                unit = "ratio"
            elif k in ("share_at_floor", "counter_raises_per_year", "counters_refused_per_year", "employment_rate",
                       "labour_share", "job_to_job_per_year"):
                d = 100 * (g[k] - base.loc[g.index, k])  # percentage-point differences for rates and shares
                unit = "points"
            else:
                d = 100 * (g[k] / base.loc[g.index, k] - 1)
                unit = "%"
            rows.append({"lever": label, "outcome": k, "level": g[k].mean(), "change": d.mean(),
                         "change_sd_over_seeds": d.std(ddof=1), "unit": unit})
    return pd.DataFrame(rows)


def grid_table(raw: pd.DataFrame) -> pd.DataFrame:
    base = raw[raw.variant == "base"].set_index(["m", "seed"])
    rows = []
    for (m, variant), g in raw.groupby(["m", "variant"]):
        g = g.set_index(["m", "seed"])
        b = base.loc[g.index]
        row = {"m": m, "variant": variant, "kaitz_base": b["kaitz"].mean(), "share_at_floor_base": b["share_at_floor"].mean(),
               "employment_base": b["employment_rate"].mean()}
        for k in ("p10", "p25", "p50", "p75", "p90", "mean_wage"):
            d = 100 * (g[k] / b[k] - 1)
            row[f"d_{k}"] = d.mean()
            row[f"d_{k}_sd"] = d.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    jobs = [(label, kw, s) for (label, kw), s in itertools.product(LEVERS.items(), SEEDS_LEVERS)]
    gjobs = [(m, v, s) for m, v, s in itertools.product(FLOORS, GRID_VARIANTS, SEEDS_GRID)]
    with Pool() as pool:
        raw = pd.DataFrame(pool.map(_run, jobs))
        graw = pd.DataFrame(pool.map(_run_grid, gjobs))
    raw.to_csv(RESULTS / "model_levers_raw.csv", index=False)
    graw.to_csv(RESULTS / "model_floor_grid_raw.csv", index=False)
    lev = lever_table(raw)
    lev.to_csv(RESULTS / "model_levers.csv", index=False)
    grid = grid_table(graw)
    grid.to_csv(RESULTS / "model_floor_grid.csv", index=False)

    lines = ["# Model experiments (model.py, experiments.py)", "",
             f"Baseline: {BASE}", f"Seeds: levers {len(SEEDS_LEVERS)}, floor grid {len(SEEDS_GRID)}", ""]
    b = raw[raw.label == "baseline"]
    lines.append("Baseline levels (mean over seeds): " + ", ".join(f"{k}={b[k].mean():.3f}" for k in OUTCOMES))
    lines.append("")
    lines.append("## Lever fingerprints: change against baseline (mean over seeds, sd in brackets)")
    for label in LEVERS:
        if label == "baseline":
            continue
        g = lev[lev.lever == label].set_index("outcome")
        sfx = {"%": "%", "points": " pt", "ratio": ""}
        parts = [f"{k} {g.loc[k, 'change']:+.3f}{sfx[g.loc[k, 'unit']]} ({g.loc[k, 'change_sd_over_seeds']:.3f})"
                 for k in OUTCOMES]
        lines.append(f"- {label}: " + "; ".join(parts))
    lines.append("")
    lines.append("## Floor grid: effect of each lever on quantiles, by minimum wage")
    for _, r in grid[grid.variant != "base"].sort_values(["variant", "m"]).iterrows():
        lines.append(f"- {r.variant} | m={r.m:.2f} kaitz={r.kaitz_base:.3f} at_floor={100 * r.share_at_floor_base:.1f}% "
                     f"emp={100 * r.employment_base:.1f}% | dP10 {r.d_p10:+.2f}% dP25 {r.d_p25:+.2f}% dP50 {r.d_p50:+.2f}% "
                     f"dP75 {r.d_p75:+.2f}% dP90 {r.d_p90:+.2f}% dmean {r.d_mean_wage:+.2f}%")
    lines.append("")
    lines.append("## Floor grid: base levels")
    gb = graw[graw.variant == "base"].groupby("m").mean(numeric_only=True)
    for m, r in gb.iterrows():
        lines.append(f"- m={m:.2f}: kaitz={r.kaitz:.3f} share_at_floor={100 * r.share_at_floor:.1f}% "
                     f"employment={100 * r.employment_rate:.1f}% p50/p10={r.p50_p10:.3f} mean_output={r.mean_output:.3f} "
                     f"labour_share={100 * r.labour_share:.1f}%")
    (RESULTS / "model_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
