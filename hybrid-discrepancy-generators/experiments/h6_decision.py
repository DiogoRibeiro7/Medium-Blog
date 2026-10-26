"""H6. Staffing for a launch whose surge is uniform on 1.1-1.3.

As in the earlier article: management adds k agents to every half hour on
launch days and wants the day's service level to reach 70% on at least 80%
of them. Each generator draws its own launch days, with a surge drawn
uniformly on the stated range for every day, and picks the smallest k that
meets the requirement; the choice is then scored on 300 launch days of
reality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (POLICY, TRAIN_DAYS, WORLD, fit_hybrids, parse, rng,
                    save, seeds)
from gentradeoffs.queue import OFFERED, WITHIN
from gentradeoffs.world import calendar, draw_day, play, simulate_table
from hybridgen.hybrid import OutputHybrid

TARGET = 0.70
COVERAGE = 0.80
SURGE = (1.1, 1.3)
LEARNERS = ("boosting",)


def day_sl(gen, g, dow, camp, m, k) -> float:
    if isinstance(gen, OutputHybrid):
        frame = gen.sample_days(g, 1, multiplier=m, calendar=[(dow, camp)],
                                extra=k)[0]
        off = frame["offered"].to_numpy(float)
        return float(np.nansum(frame["sl"] * off) / off.sum())
    out, _ = gen.day(g, dow, camp, multiplier=m, extra=k)
    return out[:, WITHIN].sum() / out[:, OFFERED].sum()


def main() -> None:
    args = parse(__doc__)
    ks = list(range(0, 13, 4)) if args.quick else list(range(0, 13))
    n_days = 30 if args.quick else 300
    sim_days = 80 if args.quick else 600
    seq = seeds("h6", 1)[0]
    train = simulate_table(rng(seq, 0), TRAIN_DAYS)
    gens = fit_hybrids(train, seq, learners=LEARNERS, sim_days=sim_days)
    models = {name: (gen.generator if hasattr(gen, "generator") else gen)
              for name, gen in gens.items()}
    g = rng(seq, 1)
    cal = calendar(g, n_days, WORLD)
    days = [draw_day(g, dow, camp, WORLD, POLICY,
                     multiplier=g.uniform(*SURGE)) for dow, camp in cal]
    curves = {"Reality": {}}
    for k in ks:
        sl = [(lambda o: o[:, WITHIN].sum() / o[:, OFFERED].sum())(
            play(d, WORLD, POLICY, extra=k)[0]) for d in days]
        curves["Reality"][k] = float(np.mean(np.array(sl) >= TARGET))
    for j, (name, gen) in enumerate(models.items()):
        curves[name] = {}
        for k in ks:
            gk = rng(seq, 2, j, k)
            sl = [day_sl(gen, gk, dow, camp, gk.uniform(*SURGE), k)
                  for dow, camp in cal]
            curves[name][k] = float(np.mean(np.array(sl) >= TARGET))
        print(f"{name} done", flush=True)
    chosen = {}
    for name, c in curves.items():
        ok = [k for k, v in sorted(c.items()) if v >= COVERAGE]
        k = ok[0] if ok else None
        chosen[name] = {"k": k, "coverage_in_reality":
                        curves["Reality"].get(k) if k is not None else None}
    print(pd.DataFrame(curves).round(3).to_string())
    print(pd.DataFrame(chosen).T.to_string())
    save("h6_decision", {"curves": {n: {str(k): v for k, v in c.items()}
                                    for n, c in curves.items()},
                         "chosen": chosen, "target": TARGET,
                         "coverage": COVERAGE, "surge": SURGE,
                         "days": n_days,
                         "p_hat": gens["Erlang A + estimated redials"]
                         .estimate.p_hat})


if __name__ == "__main__":
    main()
