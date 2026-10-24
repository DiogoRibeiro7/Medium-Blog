"""E2. Regimes the data never covered: fresh demand scaled by ``m``.

Reality is run with fresh demand multiplied by ``m`` and the planners not
told; their intraday revision reacts to what they see. Each generator is
asked the same question in the only form it can take:

* mechanistic generators simulate their own demand model times ``m``, the
  company's staffing rule and their queue;
* data-driven synthesizers are conditioned on the calendar and on offered
  calls equal to ``m`` times the training mean for that calendar cell, and
  sample staffing and outcomes.

Part b adds redials to Erlang A with probability ``p`` (fresh demand
deflated so that offered calls still match the training data) to show a
mechanism that is invisible in the observed regime. Part c switches the
queue's three assumptions to the true ones one at a time.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import (EVAL_DAYS, POLICY, TRAIN_DAYS, WORLD, expected_offered,
                    fit_generators, parse, rng, save, seeds)
from gentradeoffs.evaluation import discriminator_auc, pooled
from gentradeoffs.mechanistic import (DemandModel, MechanisticGenerator,
                                      service_cv, service_mean, theta_ratio)
from gentradeoffs.world import simulate_table

CONDITIONAL = ("Gaussian copula", "CART", "Gaussian mixture")


def deflated(demand: DemandModel, p: float, abandon: float) -> DemandModel:
    """Fresh demand implied by offered calls when a share ``p`` of the
    ``abandon`` share of calls redial."""
    return demand.with_(log_rate=demand.log_rate + np.log(1 - p * abandon))


def calendar_of(table: pd.DataFrame) -> list[tuple[int, bool]]:
    first = table.groupby("day")[["dow", "campaign"]].first()
    return [(int(d), bool(c)) for d, c in first.to_numpy()]


def main() -> None:
    args = parse(__doc__)
    reps = 1 if args.quick else 5
    mults = [1.0, 1.3] if args.quick else [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    n_eval = 20 if args.quick else 60
    redial_p = [0.0, 0.3, 0.6, 0.9]
    out_rows, redial_rows, ablation_rows, ranges = [], [], [], []
    for r, seq in enumerate(seeds("e2", reps)):
        history = simulate_table(rng(seq, 0), TRAIN_DAYS + EVAL_DAYS)
        train = history[history.day < TRAIN_DAYS]
        holdout = history[history.day >= TRAIN_DAYS]
        gens = fit_generators(train, seed=r,
                              which=("Erlang C", "Erlang A") + CONDITIONAL)
        # How far did day-level demand move in the training data?
        day_ratio = (train.assign(exp=expected_offered(train, train))
                     .groupby("day")[["offered", "exp"]].sum())
        ranges.append(np.quantile(day_ratio["offered"] / day_ratio["exp"],
                                  [0.0, 0.05, 0.95, 1.0]))
        erlang_a = gens["Erlang A"]
        ab_hat = pooled(train)["abandon"]
        redial_gens = {
            p: MechanisticGenerator(
                erlang_a.queue.with_(name=f"Erlang A + redials p={p}",
                                     retry_prob=p,
                                     retry_delay_mean=WORLD.retry_delay_mean),
                deflated(erlang_a.demand, p, ab_hat), POLICY, WORLD)
            for p in redial_p}
        for p, g in redial_gens.items():
            synth = g.sample_days(rng(seq, 5, int(10 * p)), EVAL_DAYS)
            redial_rows.append({"rep": r, "p": p, "m": "in-sample",
                                "auc": discriminator_auc(holdout, synth,
                                                         seed=r),
                                **pooled(synth)})
        # Ablation: the analyst's queue with true components switched in.
        mean_s, cv_s, th = (service_mean(train), service_cv(train),
                            theta_ratio(train))
        ablation = {}
        for arr, serv, pat, red in itertools.product(
                ("poisson", "overdispersed"), ("exp", "lognormal"),
                ("exp", "true"), (0, 1)):
            q = erlang_a.queue.with_(
                name=f"{arr}/{serv}/{pat}/{red}", service=serv,
                service_cv=cv_s, patience=pat, theta=th, world=WORLD,
                retry_prob=WORLD.retry_prob * red,
                retry_delay_mean=WORLD.retry_delay_mean)
            demand = deflated(erlang_a.demand, WORLD.retry_prob * red, ab_hat)
            if arr == "overdispersed":
                demand = demand.with_(day_cv=WORLD.day_cv,
                                      interval_sigma=WORLD.interval_sigma)
            ablation[(arr, serv, pat, red)] = MechanisticGenerator(
                q, demand, POLICY, WORLD)
        for i, m in enumerate(mults):
            real = simulate_table(rng(seq, 1, i), n_eval, multiplier=m)
            cal = calendar_of(real)
            truth = pooled(real)
            out_rows.append({"rep": r, "m": m, "generator": "Reality",
                             **truth})
            for j, name in enumerate(("Erlang C", "Erlang A")):
                synth = gens[name].sample_days(rng(seq, 2, i, j), n_eval,
                                               multiplier=m, calendar=cal)
                out_rows.append({"rep": r, "m": m, "generator": name,
                                 **pooled(synth)})
            given = real[["dow", "interval", "campaign"]].copy()
            given["offered"] = np.round(m * expected_offered(train, given))
            for j, name in enumerate(CONDITIONAL):
                draws = [pooled(gens[name].sample_given(rng(seq, 3, i, j, k),
                                                        given))
                         for k in range(2)]
                out_rows.append({"rep": r, "m": m, "generator": name,
                                 "sl": np.mean([d["sl"] for d in draws]),
                                 "abandon": np.mean([d["abandon"]
                                                     for d in draws])})
            for p, g in redial_gens.items():
                synth = g.sample_days(rng(seq, 4, i, int(10 * p)), n_eval,
                                      multiplier=m, calendar=cal)
                redial_rows.append({"rep": r, "p": p, "m": m, "auc": np.nan,
                                    **pooled(synth)})
            if m in (1.0, 1.3):
                for key, g in ablation.items():
                    synth = g.sample_days(rng(seq, 6, i, *map(hash_key, key)),
                                          n_eval, multiplier=m,
                                          calendar=cal)
                    est = pooled(synth)
                    ablation_rows.append({
                        "rep": r, "m": m, "arrivals": key[0],
                        "service": key[1], "patience": key[2],
                        "redials": key[3],
                        "sl": est["sl"], "abandon": est["abandon"],
                        "sl_error": est["sl"] - truth["sl"],
                        "abandon_error": est["abandon"] - truth["abandon"]})
            print(f"rep {r} m={m}: reality SL {truth['sl']:.3f}", flush=True)
    frame = pd.DataFrame(out_rows)
    print(frame.groupby(["m", "generator"])[["sl", "abandon"]].mean()
          .unstack().round(3).to_string())
    red = pd.DataFrame(redial_rows)
    print(red.groupby(["m", "p"])[["sl", "abandon", "auc"]].mean()
          .round(3).to_string())
    abl = pd.DataFrame(ablation_rows)
    print(abl.groupby(["m", "arrivals", "service", "patience", "redials"])[
        ["sl_error", "abandon_error"]].mean().round(3).to_string())
    save("e2_surge", {"rows": out_rows, "redials": redial_rows,
                      "ablation": ablation_rows,
                      "training_day_ratio_quantiles": np.mean(ranges, axis=0),
                      "mults": mults, "reps": reps})


def hash_key(x) -> int:
    codes = {"exp": 0, "lognormal": 1, "true": 2, "poisson": 3,
             "overdispersed": 4}
    return codes[x] if isinstance(x, str) else int(x)


if __name__ == "__main__":
    main()
