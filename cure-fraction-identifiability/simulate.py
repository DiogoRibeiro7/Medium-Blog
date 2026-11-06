"""Simulation study: when does a mixture cure model estimate the cure fraction?

    python simulate.py [--reps 200] [--workers 2]

Data-generating process (one dataset):

- n units; each is "cured" (never has the event) with probability pi;
- the others have an event time T0 from a latency distribution scaled so that its 95th percentile
  is 10 time units;
- units enter at a uniform time over a calendar window of length tau and are followed until the
  window closes, so individual follow-up is uniform on (0, tau): administrative censoring with
  staggered entry, the pattern of the bust data in the companion analysis.

The follow-up ratio R = tau / q95(T0) measures how much of the latency distribution the longest
follow-up covers. Grid: R in {0.25, 0.5, 0.75, 1, 1.5, 2, 3}, n in {250, 1000, 4000},
pi in {0.2, 0.5}, latency lognormal (sigma 0.5 or 1.0) or Weibull (shape 1.5).

For each dataset: the Kaplan-Meier plateau, cure and no-cure fits with lognormal and Weibull
latency, the Maller-Zhou alpha_n statistic and three model-based diagnostics. Writes
``results/sim_raw.csv`` (one row per dataset), ``results/sim_cells.csv`` (one row per cell) and
``results/sim_summary.txt``.
"""

from __future__ import annotations

import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from models import fit, km_plateau, maller_zhou_alpha

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

Q95 = 10.0
Z95 = 1.6448536269514722
RATIOS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
SIZES = (250, 1000, 4000)
PIS = (0.2, 0.5)
LATENCIES = ("lognormal-0.5", "lognormal-1.0", "weibull-1.5")
BAD = 0.10  # an estimate this far from the truth counts as a failure


def draw_latency(latency: str, size: int, rng: np.random.Generator) -> np.ndarray:
    fam, par = latency.split("-")
    par = float(par)
    if fam == "lognormal":
        mu = np.log(Q95) - Z95 * par
        return np.exp(mu + par * rng.standard_normal(size))
    scale = Q95 / (-np.log(0.05)) ** (1 / par)
    return scale * rng.weibull(par, size)


def latency_tail(latency: str, t: float) -> float:
    """P(T0 > t) for the true latency distribution."""
    from scipy.stats import norm

    fam, par = latency.split("-")
    par = float(par)
    if fam == "lognormal":
        mu = np.log(Q95) - Z95 * par
        return float(norm.sf((np.log(t) - mu) / par))
    scale = Q95 / (-np.log(0.05)) ** (1 / par)
    return float(np.exp(-((t / scale) ** par)))


def simulate_dataset(pi: float, latency: str, n: int, ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    tau = ratio * Q95
    cured = rng.random(n) < pi
    T0 = draw_latency(latency, n, rng)
    follow = rng.uniform(0, tau, n)
    t = np.where(cured, follow, np.minimum(T0, follow))
    e = (~cured & (T0 <= follow)).astype(int)
    return np.maximum(t, 1e-3), e


def analyse(t: np.ndarray, e: np.ndarray) -> dict:
    fits = {(fam, c): fit(t, e, fam, c) for fam in ("lognormal", "weibull") for c in (True, False)}
    horizon = 2 * t.max()
    ln, wb = fits[("lognormal", True)], fits[("weibull", True)]
    ln0, wb0 = fits[("lognormal", False)], fits[("weibull", False)]
    best_cure = min((ln, wb), key=lambda f: f.aic)
    best_none = min((ln0, wb0), key=lambda f: f.aic)
    return {
        "events": int(e.sum()),
        "max_time": float(t.max()),
        "km_plateau": km_plateau(t, e),
        "mz_alpha": maller_zhou_alpha(t, e),
        "pi_ln": ln.pi,
        "pi_ln_lo": ln.pi_lo,
        "pi_ln_hi": ln.pi_hi,
        "pi_wb": wb.pi,
        "pi_wb_lo": wb.pi_lo,
        "pi_wb_hi": wb.pi_hi,
        "aic_ln": ln.aic,
        "aic_wb": wb.aic,
        "aic_ln0": ln0.aic,
        "aic_wb0": wb0.aic,
        "cure_preferred": int(best_cure.aic < best_none.aic),
        "pi_best": best_cure.pi,
        "family_gap": abs(ln.pi - wb.pi),
        "extrap_gap": float(best_cure.survival(horizon)[0] - best_none.survival(horizon)[0]),
        "converged": int(all(f.converged for f in fits.values())),
    }


def run_cell(args: tuple) -> list[dict]:
    pi, latency, n, ratio, reps, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    tail_at_tau = latency_tail(latency, ratio * Q95)
    for r in range(reps):
        t, e = simulate_dataset(pi, latency, n, ratio, rng)
        out = analyse(t, e)
        rows.append({"pi": pi, "latency": latency, "n": n, "ratio": ratio, "rep": r, "tail_at_tau": tail_at_tau, **out})
    return rows


def cells_summary(raw: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        pi = g.pi.iloc[0]
        out = {"reps": len(g), "tail_at_tau": g.tail_at_tau.iloc[0], "event_share": (g.events / g.n).mean()}
        for k in ("km_plateau", "pi_ln", "pi_wb", "pi_best"):
            err = g[k] - pi
            out[f"{k}_bias"] = err.mean()
            out[f"{k}_rmse"] = np.sqrt((err**2).mean())
            out[f"{k}_fail"] = (err.abs() > BAD).mean()
        out["ln_cover"] = ((g.pi_ln_lo <= pi) & (pi <= g.pi_ln_hi)).mean()
        out["wb_cover"] = ((g.pi_wb_lo <= pi) & (pi <= g.pi_wb_hi)).mean()
        out["ln_width"] = (g.pi_ln_hi - g.pi_ln_lo).median()
        out["wb_width"] = (g.pi_wb_hi - g.pi_wb_lo).median()
        out["km_above_truth"] = (g.km_plateau >= pi - 0.02).mean()
        out["cure_preferred"] = g.cure_preferred.mean()
        out["mz_sufficient"] = (g.mz_alpha < 0.05).mean()
        out["family_gap"] = g.family_gap.median()
        out["extrap_gap"] = g.extrap_gap.median()
        out["converged"] = g.converged.mean()
        return pd.Series(out)

    return raw.groupby(["latency", "pi", "n", "ratio"]).apply(agg).reset_index()


RULES = {
    "Maller-Zhou says sufficient (alpha_n < 0.05)": lambda r: r.mz_alpha < 0.05,
    "latency families agree (gap < 0.05)": lambda r: r.family_gap < 0.05,
    "estimate near the Kaplan-Meier bound (within 0.05)": lambda r: (r.km_plateau - r.pi_best).abs() < 0.05,
    "AIC prefers a cure model": lambda r: r.cure_preferred == 1,
    "families agree and AIC prefers a cure model": lambda r: (r.family_gap < 0.05) & (r.cure_preferred == 1),
}


def diagnostic_rules(raw: pd.DataFrame) -> pd.DataFrame:
    """How well each rule separates datasets where the AIC-best cure estimate is off by more than BAD."""
    fail = (raw.pi_best - raw.pi).abs() > BAD
    rows = []
    for name, rule in RULES.items():
        ok = rule(raw)
        rows.append(
            {
                "rule": name,
                "pass_share": ok.mean(),
                "fail_if_pass": fail[ok].mean(),
                "fail_if_flagged": fail[~ok].mean(),
                "failures_flagged": (fail & ~ok).sum() / fail.sum(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summary-only", action="store_true", help="rebuild the summaries from results/sim_raw.csv")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    if args.summary_only:
        raw = pd.read_csv(RES / "sim_raw.csv")
        cells = cells_summary(raw)
        cells.to_csv(RES / "sim_cells.csv", index=False, float_format="%.6g")
        write_summary(raw, cells)
        return
    grid = list(itertools.product(PIS, LATENCIES, SIZES, RATIOS))
    jobs = [(pi, lat, n, ratio, args.reps, 1000 + i) for i, (pi, lat, n, ratio) in enumerate(grid)]
    with Pool(args.workers) as pool:
        parts = pool.map(run_cell, jobs, chunksize=1)
    raw = pd.DataFrame([r for part in parts for r in part])
    raw.to_csv(RES / "sim_raw.csv", index=False, float_format="%.6g")
    cells = cells_summary(raw)
    cells.to_csv(RES / "sim_cells.csv", index=False, float_format="%.6g")
    write_summary(raw, cells)


def write_summary(raw: pd.DataFrame, cells: pd.DataFrame) -> None:
    L = []
    say = L.append
    say(f"datasets: {len(raw)}; cells: {len(cells)}; replicates per cell: {int(cells.reps.min())}; all fits converged in {raw.converged.mean():.3f} of datasets")
    say(f"failure = estimate more than {BAD:.2f} from the true cure fraction; coverage of nominal 90% Wald intervals\n")
    for lat in LATENCIES:
        for pi in PIS:
            say(f"{lat}, pi = {pi}")
            g = cells[(cells.latency == lat) & (cells.pi == pi)]
            for _, r in g.iterrows():
                say(
                    f"  n={int(r.n):5d} R={r.ratio:4.2f} (P(T0>tau)={r.tail_at_tau:.3f}): "
                    f"bias KM {r.km_plateau_bias:+.3f} LN {r.pi_ln_bias:+.3f} WB {r.pi_wb_bias:+.3f} best {r.pi_best_bias:+.3f}; "
                    f"fail KM {r.km_plateau_fail:.2f} LN {r.pi_ln_fail:.2f} WB {r.pi_wb_fail:.2f} best {r.pi_best_fail:.2f}; "
                    f"cover LN {r.ln_cover:.2f} WB {r.wb_cover:.2f}; width LN {r.ln_width:.3f}; "
                    f"cure preferred {r.cure_preferred:.2f}; MZ says sufficient {r.mz_sufficient:.2f}; family gap {r.family_gap:.3f}"
                )
    short = raw.ratio <= 0.75
    in_bound = raw.km_plateau >= raw.pi - 0.02
    cov_ln = (raw.pi_ln_lo <= raw.pi) & (raw.pi <= raw.pi_ln_hi)
    cov_wb = (raw.pi_wb_lo <= raw.pi) & (raw.pi <= raw.pi_wb_hi)
    cov_best = cov_ln.where(raw.aic_ln <= raw.aic_wb, cov_wb)
    say(
        f"\nFollow-up ratios up to 0.75: the identified set [0, Kaplan-Meier plateau] contains the truth (within 0.02) in {in_bound[short].mean():.3f} of datasets; "
        f"the 90% interval of the AIC-chosen cure model contains it in {cov_best[short].mean():.3f}"
    )
    rules = diagnostic_rules(raw)
    rules.to_csv(RES / "sim_rules.csv", index=False, float_format="%.6g")
    say(f"\nDiagnostic rules over all datasets (failure rate of the AIC-best cure estimate: {((raw.pi_best - raw.pi).abs() > BAD).mean():.3f})")
    for _, r in rules.iterrows():
        say(f"  {r.rule}: passes {r.pass_share:.3f} of datasets; failure rate if passed {r.fail_if_pass:.3f}, if flagged {r.fail_if_flagged:.3f}; share of failures flagged {r.failures_flagged:.3f}")
    text = "\n".join(L) + "\n"
    (RES / "sim_summary.txt").write_text(text)
    print(text[:3000])


if __name__ == "__main__":
    main()
