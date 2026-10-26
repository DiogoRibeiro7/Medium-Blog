"""H5. Adding or removing k agents in every half hour.

Reality replays the same callers with k more agents on top of the staffing
rule. Every generator simulates the evaluation calendar with the same
random numbers for every k, so that its answer differs across k only
through the intervention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (EVAL_DAYS, POLICY, TRAIN_DAYS, WORLD, calendar_of,
                    fit_hybrids, parse, rng, sample, save, seeds)
from gentradeoffs.mechanistic import rates
from gentradeoffs.world import play, simulate_table

LEARNERS = ("boosting", "Gaussian process")


def main() -> None:
    args = parse(__doc__)
    reps = 1 if args.quick else 5
    ks = [-4, 0, 4] if args.quick else [-6, -4, -2, 0, 2, 4, 6]
    n_eval = 15 if args.quick else EVAL_DAYS
    sim_days = 80 if args.quick else 600
    rows = []
    for r, seq in enumerate(seeds("h5", reps)):
        train = simulate_table(rng(seq, 0), TRAIN_DAYS)
        gens = fit_hybrids(train, seq, learners=LEARNERS, sim_days=sim_days)
        evaluation, days = simulate_table(rng(seq, 1), n_eval,
                                          keep_days=True)
        cal = calendar_of(evaluation)
        for k in ks:
            tot = np.zeros(6)
            for day in days:
                o, _ = play(day, WORLD, POLICY, extra=k)
                tot += o.sum(axis=0)
            rows.append({"rep": r, "k": k, "model": "Reality", **rates(tot)})
            for j, (name, gen) in enumerate(gens.items()):
                syn = sample(gen, rng(seq, 2, j), n_eval, calendar=cal,
                             extra=k)
                off = syn["offered"].to_numpy(float)
                rows.append({"rep": r, "k": k, "model": name,
                             "sl": float(np.nansum(syn["sl"] * off)
                                         / off.sum()),
                             "abandon": float(np.nansum(
                                 syn["abandon_rate"] * off) / off.sum())})
            print(f"rep {r} k={k:+d} done", flush=True)
    frame = pd.DataFrame(rows)
    base = frame[frame.k == 0].set_index(["rep", "model"])[["sl", "abandon"]]
    frame = frame.join(base, on=["rep", "model"], rsuffix="_base")
    frame["sl_effect"] = frame["sl"] - frame["sl_base"]
    frame["abandon_effect"] = frame["abandon"] - frame["abandon_base"]
    print(frame.groupby(["model", "k"])["sl_effect"].mean().unstack()
          .round(3).to_string())
    save("h5_intervention", {"rows": frame.to_dict("records"), "ks": ks,
                             "reps": reps})


if __name__ == "__main__":
    main()
