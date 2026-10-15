"""Run the whole depot-leasing experiment and print every number the article quotes.

Steps
  1. build the instance and draw 104 weeks of history and 50,000 test weeks
  2. plan against the mean, against zone-wise quantiles, and with SAA on the history
  3. score every design on the same test weeks (paired comparisons)
  4. re-solve SAA on independent histories of 10 to 208 weeks (decision stability,
     in-sample noise, Mak-Morton-Wood optimality gap)

Outputs
  results/run_log.txt        everything printed below
  results/summary.json       the numbers behind the article and the figure
  results/weekly_costs.npz   weekly cost of the mean-forecast and SAA designs on
                             the test weeks (used by make_figure.py)

`--quick` runs a reduced version and writes *_quick.* files instead.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

import config
from depotsaa import (SEED, build_instance, ci95, draw_weeks, evaluate, names,
                      solve_two_stage)


class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)

    def save(self, path):
        path.write_text("\n".join(self.lines) + "\n")


def main(quick: bool = False):
    t_start = time.perf_counter()
    log = Log()
    n_test = config.QUICK_N_TEST if quick else config.N_TEST
    sizes = config.QUICK_SIZES if quick else config.SIZES
    reps = config.QUICK_REPLICATIONS if quick else config.REPLICATIONS
    suffix = "_quick" if quick else ""

    inst, rng = build_instance(SEED)
    log(f"versions: {({p: md.version(p) for p in ['highspy', 'linopy', 'numpy', 'pandas', 'xarray']})}")
    log(f"zones={inst.n_zones} candidate depots={inst.n_sites} seed={SEED}"
        + ("  [quick mode]" if quick else ""))
    log(f"mean weekly demand, all zones = {inst.mu.sum():.1f} pallets")
    log(f"capacity of all candidates    = {inst.cap.sum():.0f} pallets/week")
    log(f"fixed cost range              = EUR {inst.fixed.min():.0f}..{inst.fixed.max():.0f} per week")
    log(f"transport cost range          = EUR {inst.cost.min():.1f}..{inst.cost.max():.1f} per pallet")
    log("depots: " + ", ".join(f"D{j + 1}(cap {inst.cap[j]:.0f}, EUR {inst.fixed[j]:.0f})"
                               for j in range(inst.n_sites)))

    history = draw_weeks(inst, config.N_HISTORY, rng)
    test = draw_weeks(inst, n_test, np.random.default_rng(SEED + 1))
    tot, h = test.sum(1), history.sum(1)
    log(f"\ntest weeks ({n_test:,}): total demand mean {tot.mean():.0f}, sd {tot.std():.0f}, "
        f"p10 {np.percentile(tot, 10):.0f}, p90 {np.percentile(tot, 90):.0f}, "
        f"p99 {np.percentile(tot, 99):.0f}")
    log(f"history ({config.N_HISTORY}): total demand mean {h.mean():.0f}, sd {h.std():.0f}, "
        f"min {h.min():.0f}, max {h.max():.0f}")

    # ------------------------------------------------------------- plans --
    plans = []
    y_ev, obj_ev, *_ = solve_two_stage(inst, history.mean(0, keepdims=True))
    y_evo, obj_evo, *_ = solve_two_stage(inst, inst.mu[None, :])
    log(f"\nEV  (sample mean of history): open {names(y_ev)}, planned {obj_ev:,.0f}")
    log(f"EV  (true mean, an oracle)  : open {names(y_evo)}, planned {obj_evo:,.0f}")
    plans.append(("Mean forecast", y_ev, obj_ev))
    quantile_rows = []
    for q in config.QUANTILES:
        dq = np.percentile(history, q, axis=0)
        y_q, obj_q, *_ = solve_two_stage(inst, dq[None, :])
        log(f"P{q} (zone-wise quantiles)   : open {names(y_q)}, planned {obj_q:,.0f}; "
            f"sum of zone P{q}s {dq.sum():.0f} vs P{q} of the total {np.percentile(h, q):.0f}")
        plans.append((f"Zone P{q} forecast", y_q, obj_q))
        quantile_rows.append(dict(q=q, sum_of_zone_quantiles=float(dq.sum()),
                                  quantile_of_total=float(np.percentile(h, q))))

    y_saa, obj_saa, t_saa, nv, nc = solve_two_stage(inst, history)
    log(f"SAA ({config.N_HISTORY} weeks)             : open {names(y_saa)}, in-sample "
        f"{obj_saa:,.0f}; {nv:,} variables, {nc:,} constraints, {t_saa:.1f}s")
    link = None
    if not quick:
        y_l, obj_l, t_l, _, nc_l = solve_two_stage(inst, history, link=True)
        log(f"SAA with x <= d*y link rows : same design {bool((y_l == y_saa).all())}, "
            f"objective {obj_l:,.0f}, {nc_l:,} constraints, {t_l:.1f}s")
        link = dict(same_design=bool((y_l == y_saa).all()), objective=obj_l,
                    constraints=nc_l, seconds=t_l, seconds_without=t_saa)
    plans.append((f"SAA, {config.N_HISTORY} weeks", y_saa, obj_saa))

    # ------------------------------------------------------------ scoring --
    cache: dict = {}
    log(f"\nscored on {n_test:,} new weeks (EUR/week; +/- is a 95% half-width):")
    rows = []
    for label, y, planned in plans:
        r = evaluate(inst, y, test, cache)
        m, hw = ci95(r["total"])
        fill = 1 - r["outsourced"].sum() / test.sum()
        log(f"  {label:18s} {names(y):17s} cap {r['capacity']:5.0f} | planned "
            f"{planned:7,.0f} | realised {m:7,.0f} +/- {hw:3.0f} | gap "
            f"{(planned - m) / m:+6.1%} | fixed {r['fixed']:6,.0f} transport "
            f"{r['transport'].mean():6,.0f} outsourcing "
            f"{inst.penalty * r['outsourced'].mean():6,.0f} | own fill {fill:5.1%} | "
            f"weeks outsourcing {np.mean(r['outsourced'] > 1e-6):5.1%} | "
            f"p95 week {np.percentile(r['total'], 95):7,.0f}")
        rows.append(dict(label=label, design=names(y), capacity=r["capacity"],
                         planned=planned, realised=m, halfwidth=hw,
                         fixed=r["fixed"], transport=float(r["transport"].mean()),
                         outsourcing=float(inst.penalty * r["outsourced"].mean()),
                         own_fill=float(fill),
                         share_weeks_outsourcing=float(np.mean(r["outsourced"] > 1e-6)),
                         p95_week=float(np.percentile(r["total"], 95))))

    ev_r, saa_r = evaluate(inst, y_ev, test, cache), evaluate(inst, y_saa, test, cache)
    diff = ev_r["total"] - saa_r["total"]
    m, hw = ci95(diff)
    log(f"\nvalue of the stochastic solution (EV design - SAA design, paired): "
        f"{m:,.0f} +/- {hw:,.0f} EUR/week = {m / saa_r['total'].mean():.1%}; "
        f"x52 = {52 * m:,.0f} EUR/year")
    log(f"  SAA design cheaper in {np.mean(diff > 0):.1%} of weeks, dearer in "
        f"{np.mean(diff < 0):.1%}")
    out_weeks = ev_r["outsourced"] > 1e-6
    log(f"  EV design: mean cost of weeks it outsources {ev_r['total'][out_weeks].mean():,.0f}, "
        f"of weeks it does not {ev_r['total'][~out_weeks].mean():,.0f}")
    vss = dict(mean=m, halfwidth=hw, share_of_saa=m / saa_r["total"].mean(),
               per_year=52 * m, share_weeks_ev_cheaper=float(np.mean(diff < 0)))

    # ---------------------------------------------------- stability study --
    log(f"\nre-solving SAA on {reps} independent histories of each length:")
    study = []
    for S in sizes:
        for rep in range(reps):
            hist = draw_weeks(inst, S, np.random.default_rng([SEED, S, rep]))
            y, obj, secs, *_ = solve_two_stage(inst, hist)
            study.append(dict(S=S, rep=rep, design=names(y), insample=obj,
                              outsample=evaluate(inst, y, test, cache)["total"].mean(),
                              secs=secs))
    df = pd.DataFrame(study)
    for S, g in df.groupby("S"):
        common, cnt = Counter(g.design).most_common(1)[0]
        log(f"  {S:3d} weeks: in-sample {g.insample.mean():7,.0f} (sd {g.insample.std():5,.0f}) | "
            f"realised {g.outsample.mean():7,.0f} (worst {g.outsample.max():7,.0f}) | "
            f"{g.design.nunique()} distinct designs, {common} in {cnt}/{reps} | "
            f"median solve {g.secs.median():5.2f}s")
        if g.design.nunique() > 1:
            for dsg, n in Counter(g.design).most_common():
                rr = g[g.design == dsg]
                log(f"        {dsg:24s} x{n}: realised {rr.outsample.iloc[0]:7,.0f}")

    g = df[df.S == max(sizes)]
    lb_m, lb_h = ci95(g.insample)
    best = min(cache, key=lambda k: cache[k]["total"].mean())
    ub_m, ub_h = ci95(cache[best]["total"])
    lb, ub = lb_m - lb_h, ub_m + ub_h
    log(f"\nstatistical lower bound ({max(sizes)}-week SAA, {reps} replications): "
        f"{lb_m:,.0f} - {lb_h:,.0f} = {lb:,.0f}")
    log(f"upper bound from the best design scored, {names(np.array(best))}: "
        f"{ub_m:,.0f} + {ub_h:,.0f} = {ub:,.0f}")
    log(f"conservative optimality gap: {ub - lb:,.0f} EUR/week = {(ub - lb) / ub:.2%}")
    log(f"designs scored in total: {len(cache)}")
    log(f"\ntotal runtime {time.perf_counter() - t_start:.0f}s")

    summary = dict(
        quick=quick, seed=SEED, n_test=n_test, n_history=config.N_HISTORY,
        plans=rows, quantiles=quantile_rows, vss=vss, link_rows=link,
        study=study, bounds=dict(lower=lb, upper=ub, gap=ub - lb, gap_share=(ub - lb) / ub,
                                 best_design=names(np.array(best))),
    )
    (config.RESULTS / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(config.RESULTS / f"weekly_costs{suffix}.npz",
                        mean_forecast=ev_r["total"].astype(np.float32),
                        saa=saa_r["total"].astype(np.float32))
    log.save(config.RESULTS / f"run_log{suffix}.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="reduced smoke run")
    main(ap.parse_args().quick)
