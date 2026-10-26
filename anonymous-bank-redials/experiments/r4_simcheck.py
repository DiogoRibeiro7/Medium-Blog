"""R4. Which statistic survives clustered demand? A check where p is known.

Worlds are simulated from the centre calibrated to the real table, with a
known redial probability (0, 0.15 or 0.3) and either independent half-hour
demand or demand that clusters as much as the joint estimator finds in the
real data. Each world gives several 247-day histories. On each history the
redial probability is estimated four ways:

* lag slope, binding function simulated without clustering (the estimator
  of the earlier article);
* lag-minus-lead slope and controlled slope, same binding;
* joint estimation of the redial probability and the clustering.

The analyst's mean redial delay is the true one (ten minutes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (calibrated_centre, load_table, parse, real_statistics,
                    rng, save, seed)
from anonbank.indirect import JointEstimator, binding_functions
from anonbank.sim import Clustering, Delay, Simulator
from anonbank.stats import Statistics

SINGLE = ("lag", "lag_minus_lead", "controlled")
P_TRUE = (0.0, 0.15, 0.3)
ALL_STATS = ("lag", "lead", "lag_minus_lead", "controlled", "acf1",
             "dispersion")


def main() -> None:
    args = parse(__doc__)
    hist = 3 if args.quick else 16
    n_days = 60 if args.quick else 247
    bind_days = 120 if args.quick else 988
    joint_days = 60 if args.quick else 494
    tab = load_table()
    centre, k = calibrated_centre(tab, args.quick)
    base = Simulator(centre, delay=Delay(600.0))
    bindings = binding_functions(base, SINGLE, seed("r4-bind"), bind_days)
    joint = JointEstimator.fit(base, seed("r4-joint"), joint_days)
    real = real_statistics(tab)
    g = rng("r4-weights")
    idx = g.integers(0, real.n_days, size=(200, real.n_days))
    boot = np.array([real.vector(joint.statistics, i) for i in idx])
    weights = 1.0 / boot.var(axis=0)
    rho, sigma = joint.point(real, weights)[1:]
    worlds = {"independent demand": Clustering(),
              "clustered demand": Clustering(float(rho), float(sigma))}
    rows, stats = [], []
    for w_i, (wname, cl) in enumerate(worlds.items()):
        for p_i, p in enumerate(P_TRUE):
            truth = Simulator(centre, p=p, delay=Delay(600.0), clustering=cl)
            for h in range(hist):
                off, ab, dow = truth.days(rng("r4-world", w_i, p_i, h),
                                          n_days)
                st = Statistics(off, ab, dow)
                stats.append({"world": wname, "p_true": p, "history": h,
                              **{n: st.get(n) for n in ALL_STATS}})
                for name in SINGLE:
                    rows.append({"world": wname, "p_true": p, "history": h,
                                 "estimator": name,
                                 "p_hat": float(bindings[name].invert(
                                     st.get(name)))})
                th = joint.point(st, weights)
                rows.append({"world": wname, "p_true": p, "history": h,
                             "estimator": "joint", "p_hat": float(th[0]),
                             "rho_hat": float(th[1])})
            print(f"{wname} p={p} done", flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["world", "p_true", "estimator"])["p_hat"] \
        .agg(["mean", "std"]).round(3)
    print(summary.to_string())
    save("r4_simcheck", {"rows": rows, "statistics": stats,
                         "service_multiplier": k,
                         "clustering_rho": rho, "clustering_sigma": sigma,
                         "bindings": {n: {"grid": b.grid, "values": b.values}
                                      for n, b in bindings.items()},
                         "joint_r2": dict(zip(joint.statistics, joint.r2)),
                         "histories": hist, "days": n_days})


if __name__ == "__main__":
    main()
