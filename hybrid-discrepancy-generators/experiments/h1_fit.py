"""H1. Fit in the observed regime, and how much data each approach needs.

For each replicate history, models are trained on the first n days and
scored on 40 held-out days in two ways:

* prediction: given each held-out half hour's calendar, offered calls and
  agents, predict its service level and abandonment; scored by the RMSE of
  the day-level (offered-weighted) service level and abandonment across the
  40 days, and by the interval-level RMSE of the service level;
* generation (at 90 and 810 days): sample 40 days and measure how easily a
  classifier tells them from the held-out days (ROC AUC).

The learned corrections are also evaluated along a grid of load per agent
that runs beyond the training data, to show what each learner does there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (COLUMNS, EVAL_DAYS, WORLD, CartSynthesizer, Timer,
                    fit_hybrids, parse, rng, save, seeds)
from gentradeoffs.evaluation import discriminator_auc
from gentradeoffs.mechanistic import QueueModel, service_mean
from gentradeoffs.world import simulate_table
from hybridgen.discrepancy import DirectRegression, features
from hybridgen.eta import eta_day, eta_table


def day_scores(pred: pd.DataFrame, obs: pd.DataFrame) -> dict[str, float]:
    w = obs["offered"].to_numpy(float)
    out = {}
    for col, key in (("sl", "day_rmse_sl"), ("abandon_rate", "day_rmse_ab")):
        p = (pred[col].fillna(0).to_numpy() * w)
        o = (obs[col].fillna(0).to_numpy() * w)
        g = obs["day"].to_numpy()
        pd_ = pd.Series(p).groupby(g).sum() / pd.Series(w).groupby(g).sum()
        od_ = pd.Series(o).groupby(g).sum() / pd.Series(w).groupby(g).sum()
        out[key] = float(np.sqrt(np.mean((pd_ - od_) ** 2)))
    m = obs["sl"].notna().to_numpy()
    out["interval_rmse_sl"] = float(np.sqrt(np.average(
        (pred["sl"].to_numpy()[m] - obs["sl"].to_numpy()[m]) ** 2,
        weights=w[m])))
    return out


def correction_curve(gens: dict, train: pd.DataFrame,
                     g: np.random.Generator) -> dict:
    """delta_sl along load per agent, holding the rest at typical values.

    Rows of a mid-morning half hour on a Wednesday without a campaign, with
    30 agents and offered calls stepped so that the load per agent runs
    from 0.5 to 2.0. Erlang A's expectation for each row is computed for a
    day that is empty apart from that half hour.
    """
    s = service_mean(train)
    loads = np.linspace(0.5, 2.0, 31)
    offered = np.round(loads * 1800.0 * 30 / s)
    base = gens["Erlang A"]
    frame = pd.DataFrame({"day": np.arange(loads.size), "dow": 2,
                          "interval": 6, "campaign": 0,
                          "offered": offered, "agents": 30})
    rows = []
    for i, n in enumerate(offered):
        counts = np.zeros(24, np.int64)
        counts[6] = n
        staff = np.full(24, 30, np.int64)
        rows.append(eta_day(base.queue, g, counts, staff)[6])
    eta = pd.DataFrame(rows, columns=["eta_sl", "eta_abandon_rate",
                                      "eta_awt"], index=frame.index)
    out = {"load": features(frame, s)["load"].to_numpy(),
           "eta_sl": eta["eta_sl"].to_numpy()}
    for name, gen in gens.items():
        if hasattr(gen, "delta") and gen.delta.scale == "additive":
            out[gen.delta.learner] = gen.delta.predict(frame, eta)["sl"] \
                .to_numpy()
    return out


def main() -> None:
    args = parse(__doc__)
    reps = 2 if args.quick else 6
    lengths = [10, 30] if args.quick else [10, 30, 90, 270, 810]
    gen_lengths = [30] if args.quick else [90, 810]
    sim_days = 80 if args.quick else 600
    pred_rows, gen_rows, residual_bins = [], [], []
    curve = {}
    load_range = {}
    for r, seq in enumerate(seeds("h1", reps)):
        n_max = max(lengths)
        history = simulate_table(rng(seq, 0), n_max + EVAL_DAYS)
        holdout = history[history.day >= n_max].copy()
        for i, n in enumerate(lengths):
            train = history[history.day < n]
            with Timer() as t:
                gens = fit_hybrids(train, seq, greybox=n in gen_lengths,
                                   sim_days=sim_days)
            s = service_mean(train)
            eta_h = eta_table(gens["Erlang A"].queue, rng(seq, 1, i), holdout)
            preds = {"Erlang A": eta_h.rename(columns=lambda c: c[4:])}
            if i == 0:
                # Reality's own queue (true handle-time and patience laws)
                # replayed on the held-out inputs: what a perfect mechanism
                # would score given the same information.
                oracle = QueueModel("true queue", WORLD.service_mean,
                                    service="lognormal",
                                    service_cv=WORLD.service_cv,
                                    patience="true", world=WORLD)
                oracle_pred = eta_table(oracle, rng(seq, 8), holdout) \
                    .rename(columns=lambda c: c[4:])
            preds["True queue given the inputs"] = oracle_pred
            preds["Boosting on the data"] = DirectRegression(s).fit(
                train, rng(seq, 2, i)).predict(holdout)
            for name, gen in gens.items():
                if hasattr(gen, "delta"):
                    p, info = gen.predict_given(rng(seq, 3, i), holdout)
                    preds[name] = p
                    pred_rows.append({"rep": r, "n_days": n,
                                      "model": name + " (impossible)",
                                      "share_impossible":
                                          info["share_impossible"]})
            for name, p in preds.items():
                pred_rows.append({"rep": r, "n_days": n, "model": name,
                                  **day_scores(p, holdout)})
            if n in gen_lengths:
                cart = CartSynthesizer(seed=r).fit(train[COLUMNS])
                synths = {"CART": cart.sample(rng(seq, 4, i), len(holdout))}
                for j, (name, gen) in enumerate(gens.items()):
                    out = gen.sample_days(rng(seq, 5, i, j), EVAL_DAYS)
                    if isinstance(out, tuple):
                        synths[name] = out[0]
                        if out[1]:
                            gen_rows.append({"rep": r, "n_days": n,
                                             "model": name + " (impossible)",
                                             **out[1]})
                    else:
                        synths[name] = out
                for name, syn in synths.items():
                    gen_rows.append({"rep": r, "n_days": n, "model": name,
                                     "auc": discriminator_auc(
                                         holdout, syn[COLUMNS], seed=r)})
                gb = gens["Erlang A + estimated redials"]
                gen_rows.append({"rep": r, "n_days": n,
                                 "model": "estimated redial probability",
                                 "p_hat": gb.estimate.p_hat})
            if r == 0 and n == 90:
                curve = correction_curve(gens, train, rng(seq, 6))
                ld = features(train, s)["load"]
                load_range = {"p1": float(ld.quantile(0.01)),
                              "p99": float(ld.quantile(0.99)),
                              "max": float(ld.max())}
                eta_t = eta_table(gens["Erlang A"].queue, rng(seq, 7), train)
                resid = (train["sl"] - eta_t["eta_sl"]).to_numpy()
                bins = pd.cut(ld, np.arange(0.4, 2.05, 0.1))
                agg = pd.DataFrame({"bin": bins, "r": resid,
                                    "w": train["offered"]}).dropna()
                for b, grp in agg.groupby("bin", observed=True):
                    residual_bins.append({
                        "load": float(b.mid),
                        "mean_residual": float(np.average(grp.r,
                                                          weights=grp.w)),
                        "rows": int(len(grp))})
            print(f"rep {r} n={n} done in {t.seconds:.0f}s fit", flush=True)
    pred = pd.DataFrame(pred_rows)
    print(pred.groupby(["model", "n_days"])[["day_rmse_sl", "day_rmse_ab",
                                             "interval_rmse_sl",
                                             "share_impossible"]]
          .mean().round(4).to_string())
    gen = pd.DataFrame(gen_rows)
    print(gen.groupby(["model", "n_days"]).mean(numeric_only=True)
          .round(3).to_string())
    save("h1_fit", {"prediction": pred_rows, "generation": gen_rows,
                    "curve": curve, "train_load_range": load_range,
                    "residual_bins": residual_bins, "reps": reps,
                    "lengths": lengths})


if __name__ == "__main__":
    main()
