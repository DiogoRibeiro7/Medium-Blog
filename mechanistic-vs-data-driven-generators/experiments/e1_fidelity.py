"""E1. Fidelity in the observed regime.

Each generator is fitted on the first ``n`` days of a simulated history and
asked for 960 rows (40 days' worth). A boosted classifier then tries to tell
them from 40 held-out real days. The same is done for an independent real
sample, which gives the floor that no generator can beat at this sample
size. Repeated for several training lengths and replicate histories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (EVAL_DAYS, Timer, fit_generators, parse, rng, save,
                    seeds)
from gentradeoffs.evaluation import discriminator_auc, pooled
from gentradeoffs.world import COLUMNS, simulate_table


def load_per_agent(frame: pd.DataFrame) -> np.ndarray:
    return (frame["offered"] * frame["aht"]
            / (1800.0 * frame["agents"])).to_numpy()


def main() -> None:
    args = parse(__doc__)
    reps = 2 if args.quick else 6
    lengths = [30, 90] if args.quick else [30, 90, 270, 810]
    rows, scatter = [], {}
    for r, seq in enumerate(seeds("e1", reps)):
        n_max = max(lengths)
        history = simulate_table(rng(seq, 0), n_max + EVAL_DAYS)
        holdout = history[history.day >= n_max].reset_index(drop=True)
        other = simulate_table(rng(seq, 1), EVAL_DAYS)
        floor = discriminator_auc(holdout, other, seed=r)
        rows.append({"rep": r, "n_days": None, "generator": "Real (independent)",
                     "auc": floor, **pooled(other)})
        rows.append({"rep": r, "n_days": None, "generator": "Real (holdout)",
                     "auc": np.nan, **pooled(holdout)})
        for i, n in enumerate(lengths):
            train = history[history.day < n]
            with Timer() as t:
                gens = fit_generators(train, seed=r)
            for j, (name, gen) in enumerate(gens.items()):
                synth = gen.sample(rng(seq, 2, i, j), len(holdout))
                auc = discriminator_auc(holdout, synth, seed=r)
                rows.append({"rep": r, "n_days": n, "generator": name,
                             "auc": auc, **pooled(synth),
                             "n_components": getattr(gen, "n_components",
                                                     None)})
                if r == 0 and n == 90:
                    scatter[name] = {"load": load_per_agent(synth),
                                     "sl": synth["sl"].to_numpy()}
            if r == 0 and n == 90:
                scatter["Real (holdout)"] = {
                    "load": load_per_agent(holdout),
                    "sl": holdout["sl"].to_numpy()}
                gmm_k = gens["Gaussian mixture"].n_components
            print(f"rep {r} n={n} fitted in {t.seconds:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    summary = (frame.dropna(subset=["auc"])
               .groupby(["generator", "n_days"], dropna=False)["auc"]
               .agg(["mean", "std", "count"]).reset_index())
    print(summary.to_string(index=False))
    save("e1_fidelity", {"rows": frame.to_dict("records"),
                         "scatter": scatter, "gmm_components_rep0_n90": gmm_k,
                         "columns": COLUMNS, "lengths": lengths,
                         "reps": reps})


if __name__ == "__main__":
    main()
