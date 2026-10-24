"""E7. A staffing decision under an uncertain surge.

A launch is expected to raise fresh demand by a factor somewhere between
1.1 and 1.3 for a run of days; the planners' forecast does not include it.
Management adds ``k`` agents to every interval on launch days and wants the
day's service level to reach ``TARGET`` on at least ``COVERAGE`` of them.
Each method picks the smallest ``k`` that meets the requirement in its own
synthetic launch days, and the choice is then scored in reality.

* Erlang A (point): calibrated model, surge set to the midpoint 1.2.
* Erlang A (randomised): every synthetic day draws its own surge, patience
  rate, service-time variability, redial behaviour and demand shock from
  wide ranges (domain randomisation), and the requirement is checked on the
  pooled days.
* Erlang A (robust): the same ranges, but each draw is a scenario simulated
  over 100 days, and the requirement has to hold in 90% of scenarios.
* Gaussian copula: offered calls set to the surge times the training mean,
  staffing drawn given the offered calls, ``k`` added, outcomes drawn given
  everything. Rows are drawn independently, so its "days" are assembled
  from rows that never shared a day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (POLICY, TRAIN_DAYS, WORLD, expected_offered,
                    make_generator, parse, rng, save, seeds)
from gentradeoffs.evaluation import day_level, pooled
from gentradeoffs.mechanistic import (MechanisticGenerator, QueueModel,
                                      service_mean, theta_ratio)
from gentradeoffs.queue import N_INTERVALS, OFFERED, WITHIN
from gentradeoffs.world import calendar, draw_day, play, simulate_table

TARGET = 0.70
COVERAGE = 0.80
SURGE = (1.1, 1.3)
ROBUST_SHARE = 0.90


def coverage(day_sl: np.ndarray) -> float:
    return float(np.mean(day_sl >= TARGET))


def smallest_k(curve: dict[int, float]) -> int | None:
    ok = [k for k, c in sorted(curve.items()) if c >= COVERAGE]
    return ok[0] if ok else None


def main() -> None:
    args = parse(__doc__)
    ks = list(range(0, 13, 4)) if args.quick else list(range(0, 13))
    n_days = 30 if args.quick else 300
    n_scen = 4 if args.quick else 20
    scen_days = 20 if args.quick else 100
    seq = seeds("e7", 1)[0]
    train = simulate_table(rng(seq, 0), TRAIN_DAYS)
    erlang_a = make_generator("Erlang A", train)
    copula = make_generator("Gaussian copula", train)
    theta, s_mean = theta_ratio(train), service_mean(train)
    ab_hat = pooled(train)["abandon"]

    # Reality: the same launch days replayed with every k.
    g = rng(seq, 1)
    cal = calendar(g, n_days, WORLD)
    days = [draw_day(g, dow, camp, WORLD, POLICY,
                     multiplier=g.uniform(*SURGE)) for dow, camp in cal]
    curves = {"Reality": {}}
    for k in ks:
        sl = []
        for d in days:
            out, _ = play(d, WORLD, POLICY, extra=k)
            sl.append(out[:, WITHIN].sum() / out[:, OFFERED].sum())
        curves["Reality"][k] = coverage(np.array(sl))

    # Erlang A at the midpoint.
    curves["Erlang A (point)"] = {}
    for k in ks:
        g = rng(seq, 2, k)
        sl = []
        for dow, camp in cal:
            out, _ = erlang_a.day(g, dow, camp, multiplier=np.mean(SURGE),
                                  extra=k)
            sl.append(out[:, WITHIN].sum() / out[:, OFFERED].sum())
        curves["Erlang A (point)"][k] = coverage(np.array(sl))

    def scenario(g: np.random.Generator) -> MechanisticGenerator:
        """One draw of everything the analyst is unsure about."""
        p = g.uniform(0.0, 0.9)
        q = QueueModel("randomised", s_mean, service="lognormal",
                       service_cv=g.uniform(0.5, 1.5), patience="exp",
                       theta=theta * np.exp(g.uniform(-np.log(2), np.log(2))),
                       retry_prob=p, retry_delay_mean=g.uniform(300, 1200))
        demand = erlang_a.demand.with_(
            log_rate=erlang_a.demand.log_rate + np.log(1 - p * ab_hat),
            day_cv=g.uniform(0.0, 0.2), interval_sigma=0.05)
        return MechanisticGenerator(q, demand, POLICY, WORLD)

    def day_sl(gen, g, dow, camp, k) -> float:
        out, _ = gen.day(g, dow, camp, multiplier=g.uniform(*SURGE), extra=k)
        return out[:, WITHIN].sum() / out[:, OFFERED].sum()

    # Domain randomisation: a fresh draw of the uncertain ingredients for
    # every synthetic day, and the requirement checked on the mixture.
    curves["Erlang A (randomised)"] = {}
    for k in ks:
        g = rng(seq, 3, k)
        sl = [day_sl(scenario(g), g, dow, camp, k)
              for _ in range(2) for dow, camp in cal]
        curves["Erlang A (randomised)"][k] = coverage(np.array(sl))

    # Robust version: the same ranges, but each draw is a whole scenario
    # simulated over many days, and the requirement must hold in
    # ``ROBUST_SHARE`` of the scenarios rather than on average.
    per_scenario = np.zeros((n_scen, len(ks)))
    for s_i in range(n_scen):
        g = rng(seq, 6, s_i)
        gen = scenario(g)
        for k_i, k in enumerate(ks):
            gk = rng(seq, 7, s_i, k)
            sl = [day_sl(gen, gk, dow, camp, k)
                  for dow, camp in cal[:scen_days]]
            per_scenario[s_i, k_i] = coverage(np.array(sl))
    curves["Erlang A (robust)"] = {
        k: float(np.quantile(per_scenario[:, k_i], 1 - ROBUST_SHARE))
        for k_i, k in enumerate(ks)}

    # Gaussian copula, two-stage conditioning.
    frame = pd.DataFrame([{"day": i, "dow": dow, "interval": j,
                           "campaign": int(camp)}
                          for i, (dow, camp) in enumerate(cal)
                          for j in range(N_INTERVALS)])
    g = rng(seq, 4)
    surge = g.uniform(*SURGE, size=n_days)[frame["day"].to_numpy()]
    frame["offered"] = np.round(surge * expected_offered(train, frame))
    staffed = copula.sample_given(g, frame[["dow", "interval", "campaign",
                                            "offered"]])
    curves["Gaussian copula"] = {}
    for k in ks:
        given = staffed[["dow", "interval", "campaign", "offered",
                         "agents"]].copy()
        given["agents"] = given["agents"] + k
        synth = copula.sample_given(rng(seq, 5, k), given)
        synth["day"] = frame["day"].to_numpy()
        curves["Gaussian copula"][k] = coverage(day_level(synth)["sl"]
                                                .to_numpy())

    chosen = {name: smallest_k(c) for name, c in curves.items()}
    scored = {name: {"k": k,
                     "coverage_in_reality": (curves["Reality"].get(k)
                                             if k is not None else None),
                     "extra_agent_hours_per_day": (k * N_INTERVALS * 0.5
                                                   if k is not None
                                                   else None)}
              for name, k in chosen.items()}
    print(pd.DataFrame(curves).round(3).to_string())
    print(pd.DataFrame(scored).T.to_string())
    save("e7_decision", {"curves": {n: {str(k): v for k, v in c.items()}
                                    for n, c in curves.items()},
                         "chosen": scored, "target": TARGET,
                         "per_scenario": per_scenario, "ks": ks,
                         "robust_share": ROBUST_SHARE,
                         "coverage": COVERAGE, "surge": SURGE,
                         "days": n_days})


if __name__ == "__main__":
    main()
