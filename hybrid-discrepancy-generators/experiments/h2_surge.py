"""H2. Fresh demand m times normal, unannounced to the planners.

Every hybrid is run as a generator with its demand scaled by m, the
company's staffing rule and the evaluation days' calendar, and compared with
reality over 60 days per value of m. To show where Erlang A's error lives,
it is also replayed on reality's own surge days, given their offered calls
and staffing (prediction given the inputs rather than generation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (TRAIN_DAYS, calendar_of, fit_hybrids, parse, rng, sample,
                    save, seeds)
from gentradeoffs.evaluation import pooled
from gentradeoffs.mechanistic import predict_given, rates
from gentradeoffs.world import simulate_table

LEARNERS = ("boosting", "Gaussian process", "linear")


def main() -> None:
    args = parse(__doc__)
    reps = 1 if args.quick else 5
    mults = [1.0, 1.3] if args.quick else [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    n_eval = 15 if args.quick else 60
    sim_days = 80 if args.quick else 600
    rows = []
    for r, seq in enumerate(seeds("h2", reps)):
        train = simulate_table(rng(seq, 0), TRAIN_DAYS)
        gens = fit_hybrids(train, seq, learners=LEARNERS, sim_days=sim_days)
        p_hat = gens["Erlang A + estimated redials"].estimate.p_hat
        for i, m in enumerate(mults):
            real = simulate_table(rng(seq, 1, i), n_eval, multiplier=m)
            cal = calendar_of(real)
            rows.append({"rep": r, "m": m, "model": "Reality", **pooled(real),
                         "p_hat": p_hat})
            for j, (name, gen) in enumerate(gens.items()):
                syn = sample(gen, rng(seq, 2, i, j), n_eval, multiplier=m,
                             calendar=cal)
                rows.append({"rep": r, "m": m, "model": name, **pooled(syn)})
            g = rng(seq, 3, i)
            tot = sum(predict_given(gens["Erlang A"].queue, g,
                                    d["offered"].to_numpy().astype(np.int64),
                                    d["agents"].to_numpy().astype(np.int64),
                                    reps=2).sum(axis=0)
                      for _, d in real.groupby("day"))
            est = rates(tot)
            rows.append({"rep": r, "m": m,
                         "model": "Erlang A given reality's inputs",
                         "sl": est["sl"], "abandon": est["abandon"]})
            print(f"rep {r} m={m} reality SL {pooled(real)['sl']:.3f}",
                  flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby(["m", "model"])["sl"].mean().unstack().round(3)
          .T.to_string())
    save("h2_surge", {"rows": rows, "mults": mults, "reps": reps,
                      "eval_days": n_eval})


if __name__ == "__main__":
    main()
