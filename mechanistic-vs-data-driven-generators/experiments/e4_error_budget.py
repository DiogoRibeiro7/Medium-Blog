"""E4. Where the error comes from: simulation budget against real data.

The quantity estimated is the long-run service level of the centre in its
usual regime (offered-weighted share of calls answered within 20 s). Its
true value is taken from 4,000 simulated days of reality.

Panel a fixes the real data at 90 days and grows the synthetic sample.
Panel b fixes the synthetic sample at a size where Monte Carlo error is
small and grows the real data, with the plain average over the real days as
a reference. Each point is repeated over independent histories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (fit_generators, parse, rng, save, seeds)
from gentradeoffs.evaluation import pooled
from gentradeoffs.world import simulate_table

GENERATORS = ("Erlang A", "Gaussian copula", "CART", "SMOTE",
              "Gaussian mixture")


def mixture_size(n_days: int) -> int:
    """Components chosen by BIC in a pilot run (8 up to 90 days of data, the
    largest candidate, 24, from 270 days); fixed here to save refitting."""
    return 8 if n_days <= 90 else 24


def cumulative_sl(table: pd.DataFrame, rows_per_step: list[int]) -> list[float]:
    w = table["offered"].to_numpy(float)
    num = np.cumsum(table["sl"].to_numpy() * w)
    den = np.cumsum(w)
    return [float(num[n - 1] / den[n - 1]) for n in rows_per_step]


def main() -> None:
    args = parse(__doc__)
    reps = 3 if args.quick else 16
    truth_days = 200 if args.quick else 4000
    budgets = [1, 5, 25] if args.quick else [1, 2, 5, 10, 25, 50, 100, 250,
                                             500]
    lengths = [10, 30] if args.quick else [10, 30, 90, 270, 810]
    big_mech = 25 if args.quick else 300
    big_rows = 24 * (50 if args.quick else 500)
    truth = pooled(simulate_table(rng(seeds("e4-truth", 1)[0]), truth_days))
    print(f"true service level {truth['sl']:.4f}", flush=True)
    a_rows, b_rows = [], []
    for r, seq in enumerate(seeds("e4", reps)):
        history = simulate_table(rng(seq, 0), max(lengths + [90]))
        # Panel a: 90 real days, growing synthetic sample.
        train = history[history.day < 90]
        gens = fit_generators(train, seed=r, gmm_components=mixture_size(90),
                              which=GENERATORS)
        for j, name in enumerate(GENERATORS):
            g = rng(seq, 1, j)
            if name.startswith("Erlang"):
                synth = gens[name].sample_days(g, max(budgets))
            else:
                synth = gens[name].sample(g, 24 * max(budgets))
            for n_syn, est in zip(budgets, cumulative_sl(
                    synth, [24 * b for b in budgets])):
                a_rows.append({"rep": r, "generator": name,
                               "synthetic_days": n_syn, "estimate": est})
        # Panel b: growing real data, large synthetic sample.
        for i, n in enumerate(lengths):
            train = history[history.day < n]
            b_rows.append({"rep": r, "generator": "Empirical average",
                           "real_days": n, "estimate": pooled(train)["sl"]})
            gens = fit_generators(train, seed=r,
                                  gmm_components=mixture_size(n),
                                  which=GENERATORS)
            for j, name in enumerate(GENERATORS):
                g = rng(seq, 2, i, j)
                if name.startswith("Erlang"):
                    synth = gens[name].sample_days(g, big_mech)
                else:
                    synth = gens[name].sample(g, big_rows)
                b_rows.append({"rep": r, "generator": name, "real_days": n,
                               "estimate": pooled(synth)["sl"]})
        print(f"rep {r} done", flush=True)
    a = pd.DataFrame(a_rows)
    b = pd.DataFrame(b_rows)
    for frame in (a, b):
        frame["error"] = frame["estimate"] - truth["sl"]
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))  # noqa: E731
    print(a.groupby(["generator", "synthetic_days"])["error"]
          .agg(["mean", rmse]).round(4).to_string())
    print(b.groupby(["generator", "real_days"])["error"]
          .agg(["mean", rmse]).round(4).to_string())
    save("e4_error_budget", {"truth": truth, "panel_a": a_rows,
                             "panel_b": b_rows, "reps": reps,
                             "truth_days": truth_days})


if __name__ == "__main__":
    main()
