"""Part 2 of the study: does the backward window rule improve forward forecasts?

    python rule_sims.py --reps 300 --workers 2

Each replication simulates 720 rows with (where the scenario has one) a break at row 480,
then forecasts rows 480-719 one step ahead with every window scheme in ``schemes.py``.
The forecasts are scored the ordinary way, forward, so the backward rule is judged by
the test it is meant to improve, not by itself.

Writes ``results/rule_sims.csv.gz`` (one row per replication x scenario x scheme x phase),
``results/rule_windows.csv`` (median and quartiles of the window chosen by each
data-driven rule, by scenario and months since the break) and ``results/rule_summary.txt``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from backward_eval import embed_ar
from dgp import LABELS, scenarios, simulate
from schemes import CumOLS, run_all, run_window_scheme, selected_starts

ROWS, BREAK_ROW, P = 720, 480, 1
BLOCK, GAP, MAX_BACK = 60, 6, 8
Z_CRIT = float(norm.ppf(1 - 0.05 / MAX_BACK))  # Bonferroni, 5% over the blocks examined
ROLLING = (60, 120, 240, 360)
EXTRA_Z = (1.0, 1.645, 2.0, 3.0)  # the threshold as a dial
PHASES = {"0-59": (0, 60), "60-119": (60, 120), "120-239": (120, 240), "all": (0, 240)}
OUT = Path(__file__).parent / "results"


def one(args: tuple[str, int]) -> tuple[list[dict], list[dict]]:
    name, seed = args
    rng = np.random.default_rng(seed)
    mu, phi, sigma = scenarios(ROWS + P, BREAK_ROW + P)[name]()
    y = simulate(mu, phi, sigma, rng)
    X, z, _ = embed_ar(y, P)
    origins = np.arange(BREAK_ROW, ROWS)
    has_break = name not in ("stationary", "drift")
    # reference that knows the break date: rows since the break, but never fewer than 24
    oracle = np.minimum(BREAK_ROW, origins - 24) if has_break else np.zeros(len(origins), dtype=int)
    outs, details = run_all(
        X, z, origins, ROLLING, BLOCK, MAX_BACK, Z_CRIT, GAP, reselect=12, folds=5, seed=seed,
        avew_min=BLOCK, avew_step=12, roc_min=BLOCK // 2,
        oracle_starts=oracle if name != "drift" else None, extra_z=EXTRA_Z,
    )
    # a coarser variant of the rule: 120-row blocks, looking back at most 4 blocks
    starts, det = selected_starts(X, z, origins, "backward", 12, 120, 4, float(norm.ppf(1 - 0.05 / 4)),
                                  GAP, 5, seed, 30)
    outs["backward_b120"] = run_window_scheme(CumOLS(X, z), origins, starts)
    details["backward_b120"] = det

    rows = []
    for scheme, o in outs.items():
        for ph, (a, b) in PHASES.items():
            rows.append({
                "scenario": name, "seed": seed, "scheme": scheme, "phase": ph,
                "nlpd": float(o.nlpd[a:b].mean()), "mse": float(o.sqerr[a:b].mean()),
                "cover90": float(o.cover90[a:b].mean()),
                "window": float(np.nanmean(o.window[a:b])) if np.isfinite(o.window[a:b]).any() else np.nan,
            })
    win = []
    for rule, det in details.items():
        for o, w, stop in det:
            win.append({"scenario": name, "seed": seed, "rule": rule, "origin": o, "rows": w,
                        "since_break": o - BREAK_ROW, "stopped": stop is not None})
    return rows, win


def window_summary(wf: pd.DataFrame) -> pd.DataFrame:
    g = wf.groupby(["scenario", "rule", "since_break"])
    return pd.DataFrame({
        "q25": g.rows.quantile(0.25), "median": g.rows.median(), "q75": g.rows.quantile(0.75),
        "share_stopped": g.stopped.mean(), "n": g.rows.size(),
    }).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    names = list(scenarios(ROWS + P, BREAK_ROW + P))
    tasks = [(n, args.seed + 100_000 * i + r) for i, n in enumerate(names) for r in range(args.reps)]
    all_rows, all_win = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rows, win in ex.map(one, tasks, chunksize=4):
            all_rows += rows
            all_win += win
    df = pd.DataFrame(all_rows)
    wf = pd.DataFrame(all_win)
    df.to_csv(OUT / "rule_sims.csv.gz", index=False, float_format="%.6g")
    window_summary(wf).to_csv(OUT / "rule_windows.csv", index=False)

    lines = [
        f"Forward evaluation of window schemes, AR(1), {ROWS} rows, break at row {BREAK_ROW}, "
        f"origins {BREAK_ROW}-{ROWS - 1}, {args.reps} replications per scenario",
        f"Backward rule: {BLOCK}-row blocks, gap {GAP}, up to {MAX_BACK} blocks back, z > {Z_CRIT:.3f}, "
        "re-chosen every 12 origins. backward_b120: 120-row blocks, up to 4 back.",
        "Excess NLPD = scheme minus reference (oracle where defined, expanding for 'drift'), nats per "
        "forecast, mean over replications with its standard error.",
        "",
    ]
    for name in names:
        sub = df[df.scenario == name]
        ref = "oracle" if name != "drift" else "expanding"
        lines.append(f"[{name}] {LABELS[name]} (reference: {ref})")
        for ph in PHASES:
            s = sub[sub.phase == ph]
            base = s[s.scheme == ref].set_index("seed")["nlpd"]
            parts = []
            for scheme in s.scheme.unique():
                v = s[s.scheme == scheme].set_index("seed")["nlpd"] - base
                parts.append((scheme, v.mean(), v.std(ddof=1) / np.sqrt(len(v))))
            parts.sort(key=lambda t: t[1])
            lines.append(f"  phase {ph}: " + "; ".join(f"{k} {m:+.3f} ({se:.3f})" for k, m, se in parts))
        s = sub[sub.phase == "all"].groupby("scheme")[["mse", "cover90", "window"]].mean()
        lines.append("  all origins, mean MSE / coverage / window: " + "; ".join(
            f"{k} {r.mse:.3f}/{r.cover90:.3f}/{'' if np.isnan(r.window) else f'{r.window:.0f}'}"
            for k, r in s.iterrows()))
        w = wf[(wf.scenario == name) & (wf.rule == "backward")]
        lines.append(f"  backward rule: stopped early in {100 * w.stopped.mean():.1f}% of selections; "
                     f"median window {w.rows.median():.0f} rows; median window at 120+ rows after the break "
                     f"{w[w.since_break >= 120].rows.median():.0f}")
        lines.append("")
    (OUT / "rule_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
