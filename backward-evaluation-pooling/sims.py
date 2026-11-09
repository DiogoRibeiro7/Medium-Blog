"""Simulations: lasting regimes versus temporary episodes (PREREGISTRATION.md, secondary).

    python sims.py --reps 300 --workers 2

AR(1), 720 rows, change at row 480, one-step forecasts for rows 480-719 with every scheme in
``pooling.py``. Episodes revert after 24 rows; the crash-and-rebound episode is two shocks of
-12 and +6 innovation standard deviations in rows 480 and 481.

Writes ``results/sims.csv.gz`` (one row per replication x scenario x scheme x phase),
``results/sims_weights.csv`` and ``results/sims_summary.txt``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backward-evaluation"))

from backward_eval import embed_ar  # noqa: E402
from pooling import SCHEMES, criteria, run_fast  # noqa: E402

ROWS, AT, P, EPISODE = 720, 480, 1, 24
BLOCK, GAP, MAX_BACK = 60, 6, 8
Z_CRIT = float(norm.ppf(1 - 0.05 / MAX_BACK))
PHASES = {"0-59": (0, 60), "60-239": (60, 240), "all": (0, 240)}
OUT = Path(__file__).parent / "results"

REGIMES = ("variance_up", "variance_down", "mean_shift", "persistence_up")
EPISODES = ("burst", "level_episode", "crash")
LABELS = {
    "stationary": "No change",
    "variance_up": "Regime: volatility rises (sd 1 to 2)",
    "variance_down": "Regime: volatility falls (sd 2 to 1)",
    "mean_shift": "Regime: level shifts (mean 0 to 1.5)",
    "persistence_up": "Regime: persistence rises (phi 0.2 to 0.8)",
    "burst": "Episode: volatility burst (sd 3 for 24 rows)",
    "level_episode": "Episode: level jump (mean 3 for 24 rows)",
    "crash": "Episode: crash and rebound (shocks of -12 and +6 sd)",
}


def paths(name: str, n: int, at: int):
    mu, phi, sd = np.zeros(n), np.full(n, 0.5), np.ones(n)
    shocks: dict[int, float] = {}
    if name == "variance_up":
        sd[at:] = 2.0
    elif name == "variance_down":
        sd[:at] = 2.0
    elif name == "mean_shift":
        mu[at:] = 1.5
    elif name == "persistence_up":
        phi[:at], phi[at:] = 0.2, 0.8
    elif name == "burst":
        sd[at : at + EPISODE] = 3.0
    elif name == "level_episode":
        mu[at : at + EPISODE] = 3.0
    elif name == "crash":
        shocks = {at: -12.0, at + 1: 6.0}
    return mu, phi, sd, shocks


def simulate(mu, phi, sd, shocks, rng, burn=200):
    n = len(mu)
    e = rng.normal(size=n + burn)
    for k, v in shocks.items():
        e[burn + k] = v
    y = np.empty(n + burn)
    prev, prev_mu = mu[0] + rng.normal() * sd[0] / np.sqrt(1 - phi[0] ** 2), mu[0]
    for t in range(n + burn):
        j = max(t - burn, 0)
        prev = mu[j] + phi[j] * (prev - prev_mu) + sd[j] * e[t]
        prev_mu = mu[j]
        y[t] = prev
    return y[burn:]


def one(args):
    name, seed = args
    rng = np.random.default_rng(seed)
    mu, phi, sd, shocks = paths(name, ROWS + P, AT + P)
    y = simulate(mu, phi, sd, shocks, rng)
    X, z, _ = embed_ar(y, P)
    origins = np.arange(AT, ROWS)
    outs, sel = run_fast(X, z, origins, BLOCK, MAX_BACK, Z_CRIT, GAP, reselect=12, folds=5, seed=seed)
    rows = []
    for k, o in outs.items():
        for ph, (a, b) in PHASES.items():
            rows.append({"scenario": name, "seed": seed, "scheme": k, "phase": ph,
                         "nlpd": float(o.nlpd[a:b].mean()), "mse": float(o.sqerr[a:b].mean()),
                         "cover90": float(o.cover90[a:b].mean())})
    w = [{"scenario": name, "seed": seed, "since": s.origin - AT, "rows": s.rows, "stopped": s.stopped,
          "z_star": s.z_star if s.z_star is not None else np.nan} for s in sel]
    return rows, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    names = ["stationary", *REGIMES, *EPISODES]
    tasks = [(n, args.seed + 100_000 * i + r) for i, n in enumerate(names) for r in range(args.reps)]
    rows, ws = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r, w in ex.map(one, tasks, chunksize=4):
            rows += r
            ws += w
    df = pd.DataFrame(rows)
    wf = pd.DataFrame(ws)
    df.to_csv(OUT / "sims.csv.gz", index=False, float_format="%.6g")
    wsum = wf.groupby(["scenario", "since"]).agg(share_stopped=("stopped", "mean"), median_rows=("rows", "median"),
                                                 median_z_star=("z_star", "median")).reset_index()
    wsum["median_weight_when_stopped"] = norm.cdf(wsum["median_z_star"] - Z_CRIT)
    wsum.to_csv(OUT / "sims_weights.csv", index=False, float_format="%.4g")

    lines = [f"Regimes versus episodes: AR(1), {ROWS} rows, change at row {AT}, forecasts for rows {AT}-{ROWS - 1}, "
             f"{args.reps} replications per scenario; blocks {BLOCK}, gap {GAP}, z_crit {Z_CRIT:.3f}",
             "Gain = mean log loss of the expanding window minus that of the scheme (nats per forecast; "
             "positive = better than expanding), mean over replications (standard error).", ""]
    gains: dict[str, dict[str, float]] = {}
    for name in names:
        sub = df[df.scenario == name]
        lines.append(f"[{name}] {LABELS[name]}")
        for ph in PHASES:
            s = sub[sub.phase == ph].pivot(index="seed", columns="scheme", values="nlpd")
            parts = []
            for k in SCHEMES:
                g = s["E"] - s[k]
                parts.append(f"{k} {g.mean():+.4f} ({g.std(ddof=1) / np.sqrt(len(g)):.4f})")
                if ph == "all":
                    gains.setdefault(name, {})[k] = float(g.mean())
            lines.append(f"  gain, rows {ph} after the change: " + "; ".join(parts))
        c = sub[sub.phase == "all"].groupby("scheme")["cover90"].mean()
        lines.append("  90% coverage: " + "; ".join(f"{k} {c[k]:.3f}" for k in SCHEMES))
        w = wf[wf.scenario == name]
        st = w[w.stopped]
        wt = pd.Series(norm.cdf(st.z_star - Z_CRIT))
        lines.append(f"  walk stopped in {100 * w.stopped.mean():.1f}% of selections; when it stopped: median z* "
                     f"{st.z_star.median():.2f}, median EWP weight {wt.median():.3f}, share of weights above 0.9 "
                     f"{(wt > 0.9).mean():.3f}")
        early = w[(w.since >= 0) & (w.since < 60)]
        lines.append(f"  first 60 rows after the change: stopped in {100 * early.stopped.mean():.1f}% of selections, "
                     f"median EWP weight when stopped {np.median(norm.cdf(early[early.stopped].z_star - Z_CRIT)) if early.stopped.any() else float('nan'):.3f}")
        lines.append("")

    for label, units in (("regimes and episodes", [*REGIMES, *EPISODES]), ("regimes only", list(REGIMES)),
                         ("episodes only", list(EPISODES))):
        cr = criteria(gains, units)
        lines.append(f"Criteria across {label}: B gains in {cr['units_B_gains']}, loses in {cr['units_B_loses']}")
        lines.append(f"  C1 gains kept: sum G(B) {cr['sum_G_B_where_B_gains']:+.4f}, sum G(EWP) {cr['sum_G_EWP_there']:+.4f} "
                     f"-> {cr['C1_gains_kept']}")
        lines.append(f"  C2 losses removed: sum G(B) {cr['sum_G_B_where_B_loses']:+.4f}, sum G(EWP) {cr['sum_G_EWP_there_']:+.4f} "
                     f"-> {cr['C2_losses_removed']}")
        lines.append(f"  C3 weighting matters: mean G(EWP) {cr['mean_G_EWP']:+.4f}; fixed pools "
                     + ", ".join(f"{k} {v:+.4f}" for k, v in cr["mean_G_fixed"].items())
                     + f"; best {cr['best_fixed']} -> {cr['C3_weighting_matters']}")
    (OUT / "sims_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
