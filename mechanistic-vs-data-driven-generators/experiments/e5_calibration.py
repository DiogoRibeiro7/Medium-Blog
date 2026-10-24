"""E5. Calibrating a misspecified parameter.

Erlang A has one patience parameter, the exponential rate theta. Reality's
patience is not exponential, so theta has no true value; each way of
estimating it converges to its own limit. Three estimators are compared:

* ratio: abandonments divided by total waiting time (the exponential MLE);
* SMM (abandonment): theta whose simulated abandonment share matches;
* SMM (waiting): theta whose simulated mean wait per call matches.

Both SMM estimators replay the observed days three times per evaluation.

Each is computed on the first ``n`` days of independent histories. The
references are one over reality's mean patience, and the theta that makes
Erlang A reproduce the abandonment share of a demand surge. Finally Erlang A
is run under that surge with each value of theta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (POLICY, WORLD, parse, rng, save, seeds)
from gentradeoffs.evaluation import pooled
from gentradeoffs.mechanistic import (DemandModel, MechanisticGenerator,
                                      QueueModel, service_mean, theta_ratio,
                                      theta_smm)
from gentradeoffs.world import simulate_table

SURGE = 1.3
SIMS = 3      # simulated copies of the data per SMM evaluation


def estimates(table: pd.DataFrame, g: np.random.Generator) -> dict[str, float]:
    s = service_mean(table)
    return {"ratio": theta_ratio(table),
            "SMM (abandonment)": theta_smm(table, g, "abandon", s, sims=SIMS),
            "SMM (waiting)": theta_smm(table, g, "wait", s, sims=SIMS)}


def main() -> None:
    args = parse(__doc__)
    reps = 2 if args.quick else 12
    lengths = [10, 30] if args.quick else [10, 30, 90, 270]
    long_days = 60 if args.quick else 1000
    surge_days = 20 if args.quick else 120
    rows = []
    for r, seq in enumerate(seeds("e5", reps)):
        history = simulate_table(rng(seq, 0), max(lengths))
        for i, n in enumerate(lengths):
            est = estimates(history[history.day < n], rng(seq, 1, i))
            for name, theta in est.items():
                rows.append({"rep": r, "n_days": n, "estimator": name,
                             "theta": theta})
        print(f"rep {r} done", flush=True)
    frame = pd.DataFrame(rows)
    ref_seq = seeds("e5-reference", 1)[0]
    long = simulate_table(rng(ref_seq, 0), long_days)
    limits = estimates(long, rng(ref_seq, 1))
    surge = simulate_table(rng(ref_seq, 2), surge_days, multiplier=SURGE)
    theta_surge = theta_smm(surge, rng(ref_seq, 3), "abandon",
                            service_mean(long), sims=SIMS)
    references = {"one over mean patience": 1 / WORLD.mean_patience(),
                  f"SMM (abandonment) at m={SURGE}": theta_surge}
    # What each value does to a surge forecast.
    demand = DemandModel.fit(long)
    truth = pooled(surge)
    forecasts = {"Reality": truth}
    candidates = {**{f"{k} (long-run limit)": v for k, v in limits.items()},
                  **references}
    for j, (name, theta) in enumerate(candidates.items()):
        q = QueueModel(name, service_mean(long), patience="exp", theta=theta)
        gen = MechanisticGenerator(q, demand, POLICY, WORLD)
        forecasts[name] = pooled(gen.sample_days(rng(ref_seq, 4, j),
                                                 surge_days,
                                                 multiplier=SURGE))
    summary = (frame.groupby(["estimator", "n_days"])["theta"]
               .agg(["mean", "std"]))
    summary["mean_patience_implied"] = 1 / summary["mean"]
    print(summary.to_string())
    print({k: round(v, 6) for k, v in limits.items()})
    print({k: round(v, 6) for k, v in references.items()})
    print({k: {m: round(x, 3) for m, x in v.items()}
           for k, v in forecasts.items()})
    save("e5_calibration", {"rows": rows, "limits": limits,
                            "limit_days": long_days,
                            "references": references, "surge": SURGE,
                            "sims": SIMS,
                            "surge_forecasts": forecasts, "reps": reps})


if __name__ == "__main__":
    main()
