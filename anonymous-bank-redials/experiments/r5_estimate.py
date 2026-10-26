"""R5. The redial probability estimated from the real interval table.

Every estimator of R4 on the real table, under four assumptions about the
delay between hanging up and calling again: exponential with a mean of 5,
10 or 20 minutes, and the empirical delays of identified redials (from R2,
when available). 90% intervals from a day-cluster bootstrap. The
estimates are set beside the excess calling after abandonment measured
directly from customer IDs (R2), raw and weighted for the callbacks the
IDs cannot see.
"""

from __future__ import annotations

import numpy as np

from common import (RESULTS, calibrated_centre, load, load_table, parse,
                    real_statistics, rng, save, seed)
from anonbank.indirect import JointEstimator, binding_functions, estimate
from anonbank.sim import Delay, Simulator

SINGLE = ("lag", "lag_minus_lead", "controlled")


def delays() -> dict[str, Delay]:
    out = {"exponential, mean 5 min": Delay(300.0),
           "exponential, mean 10 min": Delay(600.0),
           "exponential, mean 20 min": Delay(1200.0)}
    if (RESULTS / "r2_ids.json").exists():
        q = np.array(load("r2_ids")["delay_quantiles_min"]) * 60.0
        # Sample from the empirical quantile function of identified redials.
        fine = np.interp(np.linspace(0, 1, 2001), np.linspace(0, 1, q.size),
                         q)
        out["empirical, from caller IDs"] = Delay(empirical=fine)
    return out


def main() -> None:
    args = parse(__doc__)
    bind_days = 120 if args.quick else 988
    joint_days = 60 if args.quick else 494
    reps = 100 if args.quick else 1000
    joint_reps = 40 if args.quick else 300
    tab = load_table()
    centre, k = calibrated_centre(tab, args.quick)
    real = real_statistics(tab)
    rows = []
    bindings_out = {}
    for d_i, (dname, delay) in enumerate(delays().items()):
        sim = Simulator(centre, delay=delay)
        bindings = binding_functions(sim, SINGLE, seed("r5-bind", d_i),
                                     bind_days)
        for name, b in bindings.items():
            est = estimate(real, b, rng("r5-boot", d_i), reps)
            rows.append({"delay": dname, "estimator": name, **est})
        bindings_out[dname] = {n: {"grid": b.grid, "values": b.values}
                               for n, b in bindings.items()}
        joint = JointEstimator.fit(sim, seed("r5-joint", d_i), joint_days)
        est = joint.estimate(real, rng("r5-jboot", d_i), joint_reps)
        rows.append({"delay": dname, "estimator": "joint",
                     **{k2: v for k2, v in est.items()
                        if not isinstance(v, dict)},
                     "observed": est["observed"], "fitted": est["fitted"],
                     "r2": dict(zip(joint.statistics, joint.r2))})
        print(dname, [(r["estimator"], round(r["p_hat"], 3),
                       round(r["p_lo"], 3), round(r["p_hi"], 3))
                      for r in rows if r["delay"] == dname], flush=True)
    ids = None
    if (RESULTS / "r2_ids.json").exists():
        r2 = load("r2_ids")
        ids = {}
        for vol in ("10", "50", "None"):
            rows_ = r2["curves"][vol]
            ids[vol] = {}
            for kind, col in (("raw", "again"),
                              ("weighted", "again_weighted")):
                curve = {(r["window_min"], r["group"]): r.get(col)
                         for r in rows_}
                wins = (10, 30, 60, 180) if kind == "raw" else (10, 30, 60)
                ids[vol][kind] = {str(w): curve[(w, "abandoned")]
                                  - curve[(w, "served")] for w in wins}
    save("r5_estimate", {"rows": rows, "service_multiplier": k,
                         "id_excess": ids, "bindings": bindings_out})


if __name__ == "__main__":
    main()
