"""Simulations S1 of PREREGISTRATION.md: regimes versus episodes, with the end-of-episode check.

    python sims.py --model gaussian --reps 300 --workers 2
    python sims.py --model t --reps 300 --workers 2

The scenarios and the simulator are those of ``../backward-evaluation-pooling/sims.py``
(imported). AR(1), 720 rows, change at row 480, one-step forecasts for rows 480-719; blocks 60,
gap 6, recent span 12, threshold 1.645 (variants with r = 24 and c = 0, 2.498 are reported too).
Gaussian models run every scheme; Student-t(5) models every scheme except AveW.

Writes ``results/sims_<model>.csv.gz`` (replication x scenario x scheme x phase),
``results/sims_<model>_decisions.csv`` (share of decisions using the backward window, by time
since the change) and ``results/sims_<model>_summary.txt``.
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import common
from backward_eval import ARStudentT, embed_ar
from episode import criteria, run_fast, run_generic

_spec = importlib.util.spec_from_file_location("pooling_sims", common.ROOT / "backward-evaluation-pooling" / "sims.py")
psims = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psims)

ROWS, AT, P = psims.ROWS, psims.AT, psims.P
BLOCK, GAP, MAX_BACK, Z_CRIT = psims.BLOCK, psims.GAP, psims.MAX_BACK, psims.Z_CRIT
PHASES = psims.PHASES
REGIMES, EPISODES, LABELS = psims.REGIMES, psims.EPISODES, psims.LABELS
NAMES = ["stationary", *REGIMES, *EPISODES]
RS, CS = (12, 24), (0.0, 1.645, Z_CRIT)


def one(args):
    name, seed, model = args
    rng = np.random.default_rng(seed)
    mu, phi, sd, shocks = psims.paths(name, ROWS + P, AT + P)
    y = psims.simulate(mu, phi, sd, shocks, rng)
    X, z, _ = embed_ar(y, P)
    origins = np.arange(AT, ROWS)
    kw = dict(reselect=12, folds=5, seed=seed, rs=RS, cs=CS, primary=(12, 1.645))
    if model == "gaussian":
        outs, sel = run_fast(X, z, origins, BLOCK, MAX_BACK, Z_CRIT, GAP, **kw)
    else:
        outs, sel = run_generic(X, z, origins, ARStudentT, BLOCK, MAX_BACK, Z_CRIT, GAP, avew=False, **kw)
    rows = []
    for k, o in outs.items():
        for ph, (a, b) in PHASES.items():
            rows.append({"scenario": name, "seed": seed, "scheme": k, "phase": ph,
                         "nlpd": float(o.nlpd[a:b].mean()), "mse": float(o.sqerr[a:b].mean()),
                         "cover90": float(o.cover90[a:b].mean())})
    dec = []
    for d in sel.decisions:
        st = d.stopped_at is not None
        dec.append({"scenario": name, "seed": seed, "since": d.origin - AT, "stopped": st,
                    "z_star": d.z_star if st else np.nan, "z_end12": d.z_end[12], "z_end24": d.z_end[24]})
    return rows, dec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("gaussian", "t"), default="gaussian")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5151)
    args = ap.parse_args()
    common.OUT.mkdir(exist_ok=True)
    tasks = [(n, args.seed + 100_000 * i + r, args.model) for i, n in enumerate(NAMES) for r in range(args.reps)]
    rows, decs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r, d in ex.map(one, tasks, chunksize=4):
            rows += r
            decs += d
    df = pd.DataFrame(rows)
    dc = pd.DataFrame(decs)
    tag = args.model
    df.to_csv(common.OUT / f"sims_{tag}.csv.gz", index=False, float_format="%.6g")
    dc["uses_B"] = dc.stopped
    dc["uses_C"] = dc.stopped & ~(dc.z_end12 > 1.645)
    share = dc.groupby(["scenario", "since"]).agg(share_B=("uses_B", "mean"), share_C=("uses_C", "mean"),
                                                 median_z_end=("z_end12", "median")).reset_index()
    share.to_csv(common.OUT / f"sims_{tag}_decisions.csv", index=False, float_format="%.4g")

    schemes = [k for k in df.scheme.unique()]
    order = [k for k in ("E", "B", "C", "P0.25", "PC0.25", "AveW") if k in schemes] + \
        sorted(k for k in schemes if k.startswith("C["))
    lines = [f"S1 simulations, {tag} models: AR(1), {ROWS} rows, change at row {AT}, forecasts for rows {AT}-{ROWS - 1}, "
             f"{args.reps} replications per scenario; blocks {BLOCK}, gap {GAP}, z_crit {Z_CRIT:.3f}; "
             "end-of-episode check r = 12, c = 1.645 (variants C[r,c])",
             "Gain = mean log loss of the expanding window minus that of the scheme (nats per forecast), "
             "mean over replications (standard error).", ""]
    gains: dict[str, dict[str, float]] = {}
    for name in NAMES:
        sub = df[df.scenario == name]
        lines.append(f"[{name}] {LABELS[name]}")
        for ph in PHASES:
            s = sub[sub.phase == ph].pivot(index="seed", columns="scheme", values="nlpd")
            parts = []
            for k in order:
                g = s["E"] - s[k]
                parts.append(f"{k} {g.mean():+.4f} ({g.std(ddof=1) / np.sqrt(len(g)):.4f})")
                if ph == "all":
                    gains.setdefault(name, {})[k] = float(g.mean())
            lines.append(f"  gain, rows {ph} after the change: " + "; ".join(parts))
        sh = share[share.scenario == name]
        early = sh[sh.since < 60]
        mid = sh[(sh.since >= 36) & (sh.since < 60)]
        late = sh[sh.since >= 60]
        lines.append(f"  decisions using the backward window: rows 0-59 B {early.share_B.mean():.3f} C {early.share_C.mean():.3f}; "
                     f"rows 36-59 B {mid.share_B.mean():.3f} C {mid.share_C.mean():.3f}; "
                     f"rows 60-239 B {late.share_B.mean():.3f} C {late.share_C.mean():.3f}")
        lines.append("")
    for label, units in (("episodes", list(EPISODES)), ("regimes", list(REGIMES))):
        cr = criteria(gains, units, "C")
        lines.append(f"S1 across {label}: B gains in {cr['B_gains_on']}, loses in {cr['B_loses_on']}")
        lines.append(f"  losses removed: sum G(B) {cr['sum_B_where_B_loses']:+.4f}, sum G(C) {cr['sum_C_there']:+.4f} -> {cr['H1']}")
        lines.append(f"  gains kept: sum G(B) {cr['sum_B_where_B_gains']:+.4f}, sum G(C) {cr['sum_C_there_']:+.4f} -> {cr['H2']}")
        lines.append("  mean gain: " + "; ".join(f"{k} {cr['mean'][k]:+.4f}" for k in order))
    (common.OUT / f"sims_{tag}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
