"""H3. Sizing the redial slot from the interval table.

* By training length: the indirect-inference estimate of the redial
  probability on independent histories (truth: 0.6), with the analyst's
  assumed mean delay of 600 s (the truth).
* Placebo: the same estimate on a world in which nobody redials.
* Wrong shape: the analyst assumes a mean delay of 300 s or 1,200 s; the
  estimate, and what the resulting generator predicts for a 30% surge.
* The binding function for one history, for the figure.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from common import (POLICY, WORLD, calendar_of, int_seed, parse, rng, sample,
                    save, seeds)
from gentradeoffs.evaluation import pooled
from gentradeoffs.world import simulate_table
from hybridgen.hybrid import (auxiliary_coefficient, day_factor_generator,
                              estimate_redials)

SURGE = 1.3


def main() -> None:
    args = parse(__doc__)
    reps = 2 if args.quick else 8
    lengths = [30] if args.quick else [30, 90, 270, 810]
    delay_lengths = [30] if args.quick else [90, 810]
    sim_days = 80 if args.quick else 600
    n_eval = 15 if args.quick else 60
    est_rows, aux_rows, binding = [], [], {}
    placebo_world = replace(WORLD, retry_prob=0.0)
    for r, seq in enumerate(seeds("h3", reps)):
        n_max = max(lengths)
        history = simulate_table(rng(seq, 0), n_max)
        placebo = simulate_table(rng(seq, 1), n_max, world=placebo_world)
        surge = simulate_table(rng(seq, 2), n_eval, multiplier=SURGE)
        cal = calendar_of(surge)
        for i, n in enumerate(lengths):
            train = history[history.day < n]
            e = estimate_redials(train, int_seed(seq, 3, i), 600.0, POLICY,
                                 WORLD, sim_days=sim_days)
            est_rows.append({"rep": r, "n_days": n, "world": "redials 0.6",
                             "delay": 600.0, "p_hat": e.p_hat})
            aux_rows.append({"rep": r, "n_days": n, "world": "redials 0.6",
                             "beta": e.beta_observed})
            if r == 0 and n == (90 if 90 in lengths else lengths[0]):
                binding = {"grid": e.grid, "beta_simulated": e.beta_simulated,
                           "beta_observed": e.beta_observed}
            if n in delay_lengths:
                pt = placebo[placebo.day < n]
                ep = estimate_redials(pt, int_seed(seq, 4, i), 600.0, POLICY,
                                      placebo_world, sim_days=sim_days)
                est_rows.append({"rep": r, "n_days": n, "world": "no redials",
                                 "delay": 600.0, "p_hat": ep.p_hat})
                aux_rows.append({"rep": r, "n_days": n, "world": "no redials",
                                 "beta": auxiliary_coefficient(pt)})
                for k, delay in enumerate((300.0, 600.0, 1200.0)):
                    if delay == 600.0:
                        p_hat = e.p_hat
                    else:
                        p_hat = estimate_redials(
                            train, int_seed(seq, 5, i, k), delay, POLICY,
                            WORLD, sim_days=sim_days).p_hat
                        est_rows.append({"rep": r, "n_days": n,
                                         "world": "redials 0.6",
                                         "delay": delay, "p_hat": p_hat})
                    gen = day_factor_generator(train, POLICY, WORLD, p=p_hat,
                                               delay=delay)
                    syn = sample(gen, rng(seq, 6, i, k), n_eval,
                                 multiplier=SURGE, calendar=cal)
                    est_rows.append({"rep": r, "n_days": n,
                                     "world": "surge forecast",
                                     "delay": delay, "p_hat": p_hat,
                                     "sl": pooled(syn)["sl"],
                                     "sl_reality": pooled(surge)["sl"]})
            print(f"rep {r} n={n}: p_hat {e.p_hat:.3f}", flush=True)
    frame = pd.DataFrame(est_rows)
    print(frame.groupby(["world", "delay", "n_days"])[["p_hat", "sl",
                                                       "sl_reality"]]
          .agg(["mean", "std"]).round(3).to_string())
    aux = pd.DataFrame(aux_rows)
    print(aux.groupby(["world", "n_days"])["beta"].agg(["mean", "std"])
          .round(3).to_string())
    save("h3_redials", {"estimates": est_rows, "auxiliary": aux_rows,
                        "binding": binding, "reps": reps, "surge": SURGE,
                        "true_p": WORLD.retry_prob,
                        "true_delay": WORLD.retry_delay_mean})


if __name__ == "__main__":
    main()
