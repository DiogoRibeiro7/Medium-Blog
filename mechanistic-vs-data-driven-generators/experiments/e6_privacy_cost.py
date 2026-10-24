"""E6. Closeness to the training rows, and what each generator costs to run.

Distance to closest record (DCR): for every synthetic row, the distance to
the nearest training row after scaling each column by its training standard
deviation. The yardstick is the same distance for real held-out rows, which
were never seen in training. Reported: the median DCR of synthetic rows
relative to that of held-out rows, the share of synthetic rows closer to a
training row than 95% of held-out rows are, and the share that are exact
copies of a training row.

Also timed: fitting, and producing 1,000 rows. Timings depend on the
machine and are not part of the reproducibility check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (EVAL_DAYS, NAMES, TRAIN_DAYS, Timer, make_generator, parse,
                    rng, save, seeds)
from gentradeoffs.evaluation import distance_to_closest_record
from gentradeoffs.world import COLUMNS, simulate_table


def main() -> None:
    args = parse(__doc__)
    reps = 1 if args.quick else 5
    rows, timing = [], []
    for r, seq in enumerate(seeds("e6", reps)):
        history = simulate_table(rng(seq, 0), TRAIN_DAYS + EVAL_DAYS)
        train = history[history.day < TRAIN_DAYS].dropna()
        holdout = history[history.day >= TRAIN_DAYS].dropna()
        scale = train[COLUMNS].std().to_numpy()
        scale = np.where(scale > 0, scale, 1.0)
        ref = distance_to_closest_record(train, holdout, scale)
        p5 = np.quantile(ref, 0.05)
        for j, name in enumerate(NAMES):
            with Timer() as t_fit:
                gen = make_generator(name, train, seed=r)
            with Timer() as t_gen:
                synth = gen.sample(rng(seq, 1, j), 1000).dropna()
            d = distance_to_closest_record(train, synth, scale)
            rows.append({"rep": r, "generator": name,
                         "median_dcr_ratio": float(np.median(d)
                                                   / np.median(ref)),
                         "share_closer_than_p5": float(np.mean(d < p5)),
                         "share_exact_copies": float(np.mean(d < 1e-9))})
            if r == 0:
                timing.append({"generator": name,
                               "fit_seconds": t_fit.seconds,
                               "seconds_per_1000_rows": t_gen.seconds})
        print(f"rep {r} done", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby("generator")[["median_dcr_ratio",
                                      "share_closer_than_p5",
                                      "share_exact_copies"]].mean()
          .round(3).to_string())
    print(pd.DataFrame(timing).round(3).to_string(index=False))
    save("e6_privacy_cost", {"rows": rows, "timing": timing, "reps": reps})


if __name__ == "__main__":
    main()
