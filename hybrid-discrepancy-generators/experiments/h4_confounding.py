"""H4. A parameter and a discrepancy that the data cannot tell apart.

Erlang A is calibrated with its patience rate set to each value on a grid
of mean patience, instead of being estimated, and a discrepancy is learned
on top with boosting and with a Gaussian process. For each value the
held-out fit (day-level RMSE of the service level, prediction given the
inputs) and the service level forecast for a 30% surge (generation) are
recorded, with and without the learned correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (EVAL_DAYS, POLICY, TRAIN_DAYS, WORLD, calendar_of, parse,
                    rng, sample, save, seeds)
from gentradeoffs.evaluation import pooled
from gentradeoffs.mechanistic import (DemandModel, MechanisticGenerator,
                                      erlang_models, service_mean,
                                      theta_ratio)
from gentradeoffs.world import simulate_table
from h1_fit import day_scores
from hybridgen.discrepancy import Discrepancy, apply
from hybridgen.eta import eta_table
from hybridgen.hybrid import OutputHybrid

SURGE = 1.3
PATIENCE = (100.0, 140.0, 190.0, 250.0, 330.0, 450.0)


def main() -> None:
    args = parse(__doc__)
    reps = 1 if args.quick else 3
    grid = PATIENCE[::2] if args.quick else PATIENCE
    n_surge = 15 if args.quick else 60
    rows = []
    for r, seq in enumerate(seeds("h4", reps)):
        history = simulate_table(rng(seq, 0), TRAIN_DAYS + EVAL_DAYS)
        train = history[history.day < TRAIN_DAYS]
        holdout = history[history.day >= TRAIN_DAYS]
        surge = simulate_table(rng(seq, 1), n_surge, multiplier=SURGE)
        cal = calendar_of(surge)
        demand = DemandModel.fit(train)
        s = service_mean(train)
        rows.append({"rep": r, "patience": 1 / theta_ratio(train),
                     "model": "ratio estimate"})
        for i, pat in enumerate(grid):
            queue = erlang_models(train)["Erlang A"].with_(theta=1 / pat)
            base = MechanisticGenerator(queue, demand, POLICY, WORLD)
            eta_t = eta_table(queue, rng(seq, 2, i), train)
            eta_h = eta_table(queue, rng(seq, 3, i), holdout)
            fits = {"Erlang A alone": base}
            pred = {"Erlang A alone": eta_h.rename(columns=lambda c: c[4:])}
            for j, learner in enumerate(("boosting", "Gaussian process")):
                d = Discrepancy(learner, s).fit(train, eta_t, rng(seq, 4, i, j))
                fits[f"+ {learner}"] = OutputHybrid(base, d)
                pred[f"+ {learner}"] = apply(eta_h, d.predict(holdout, eta_h))[0]
            for j, (name, gen) in enumerate(fits.items()):
                syn = sample(gen, rng(seq, 5, i, j), n_surge,
                             multiplier=SURGE, calendar=cal)
                rows.append({"rep": r, "patience": pat, "model": name,
                             **day_scores(pred[name], holdout),
                             "surge_sl": pooled(syn)["sl"],
                             "surge_sl_reality": pooled(surge)["sl"]})
            print(f"rep {r} mean patience {pat:.0f}s done", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby(["model", "patience"])[["day_rmse_sl", "surge_sl",
                                                "surge_sl_reality"]]
          .mean().round(4).to_string())
    save("h4_confounding", {"rows": rows, "grid": list(grid), "reps": reps,
                            "surge": SURGE,
                            "true_mean_patience": WORLD.mean_patience()})


if __name__ == "__main__":
    main()
