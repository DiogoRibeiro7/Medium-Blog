"""E3. Interventions: add ``k`` agents to every interval.

The true effect comes from replaying the same callers through the company's
staffing rule with ``k`` more agents (common random numbers), so that the
only difference between the two runs is the intervention.

Mechanistic generators replay each evaluation day's offered calls with the
day's staffing plus ``k``. Data-driven synthesizers are asked in two ways:

* adjusted: condition on calendar, offered calls and agents plus ``k``;
* naive: condition on calendar and agents plus ``k`` only, and let the
  synthesizer fill in the offered calls, as a joint model would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (EVAL_DAYS, POLICY, TRAIN_DAYS, WORLD, fit_generators,
                    parse, rng, save, seeds)
from gentradeoffs.evaluation import pooled
from gentradeoffs.mechanistic import days_inputs, predict_given, rates
from gentradeoffs.world import play, simulate_table

CONDITIONAL = ("Gaussian copula", "CART", "Gaussian mixture")


def main() -> None:
    args = parse(__doc__)
    reps = 1 if args.quick else 5
    ks = [-4, 0, 4] if args.quick else [-6, -4, -2, 0, 2, 4, 6]
    n_eval = 20 if args.quick else EVAL_DAYS
    out = []
    support = []
    for r, seq in enumerate(seeds("e3", reps)):
        train = simulate_table(rng(seq, 0), TRAIN_DAYS)
        gens = fit_generators(train, seed=r,
                              which=("Erlang C", "Erlang A") + CONDITIONAL)
        evaluation, days = simulate_table(rng(seq, 1), n_eval,
                                          keep_days=True)
        inputs = list(days_inputs(evaluation))
        # Agents relative to what the training data shows for similar load.
        support.append({"rep": r,
                        "train_agents_min": train["agents"].min(),
                        "train_agents_max": train["agents"].max()})
        for i, k in enumerate(ks):
            tot = np.zeros(6)
            for day in days:
                o, _ = play(day, WORLD, POLICY, extra=k)
                tot += o.sum(axis=0)
            out.append({"rep": r, "k": k, "method": "Reality", **rates(tot)})
            for j, name in enumerate(("Erlang C", "Erlang A")):
                q = gens[name].queue
                g = rng(seq, 2, i, j)
                tot = sum(predict_given(q, g, c, np.maximum(s + k, 1),
                                        reps=2).sum(axis=0)
                          for c, s in inputs)
                out.append({"rep": r, "k": k, "method": name, **rates(tot)})
            given = evaluation[["dow", "interval", "campaign", "offered",
                                "agents"]].copy()
            given["agents"] = np.maximum(given["agents"] + k, 1)
            for j, name in enumerate(CONDITIONAL):
                for mode, cols in (("adjusted", list(given.columns)),
                                   ("naive", ["dow", "interval", "campaign",
                                              "agents"])):
                    draws = [pooled(gens[name].sample_given(
                        rng(seq, 3, i, j, len(mode), d), given[cols]))
                        for d in range(2)]
                    out.append({"rep": r, "k": k,
                                "method": f"{name} ({mode})",
                                "sl": np.mean([x["sl"] for x in draws]),
                                "abandon": np.mean([x["abandon"]
                                                    for x in draws])})
            print(f"rep {r} k={k:+d} done", flush=True)
    frame = pd.DataFrame(out)
    base = (frame[frame.k == 0].set_index(["rep", "method"])[["sl", "abandon"]])
    frame = frame.join(base, on=["rep", "method"], rsuffix="_base")
    frame["sl_effect"] = frame["sl"] - frame["sl_base"]
    frame["abandon_effect"] = frame["abandon"] - frame["abandon_base"]
    print(frame.groupby(["k", "method"])["sl_effect"].mean().unstack()
          .round(3).T.to_string())
    save("e3_intervention", {"rows": frame.to_dict("records"), "ks": ks,
                             "support": support, "reps": reps})


if __name__ == "__main__":
    main()
