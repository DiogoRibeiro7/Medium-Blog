"""Experiment 3: the PINN in every dimension, for both problems.

One file per run in results/pinn/, so the sweep can be stopped and resumed; runs whose
file exists are skipped. The training budget is the same in every dimension.
"""

import argparse
import dataclasses
import json

import config
from pinncross.pinn import PinnConfig, train_pinn
from pinncross.problems import Problem

FULL = PinnConfig()
QUICK = PinnConfig(width=16, depth=2, adam_steps=60, adam_batch=128,
                   lbfgs_steps=20, lbfgs_batch=256, eval_points=2000)


def main(quick: bool, problems, dims, seeds, lbfgs_refresh: int = 0):
    cfg = QUICK if quick else FULL
    folder = config.RESULTS / ("pinn_quick" if quick else "pinn")
    suffix = ""
    if lbfgs_refresh:
        # diagnostic variant: fresh L-BFGS points every K iterations, saved separately
        cfg = dataclasses.replace(cfg, lbfgs_refresh=lbfgs_refresh)
        folder = config.RESULTS / ("pinn_variants_quick" if quick else "pinn_variants")
        suffix = f"-refresh{lbfgs_refresh}"
    folder.mkdir(exist_ok=True)
    for seed in seeds:
        for name in problems:
            for d in dims or config.DIMS[name]:
                if d not in config.DIMS[name]:
                    continue
                path = folder / f"{name}-d{d}-s{seed}{suffix}.json"
                if path.exists():
                    continue
                r = train_pinn(Problem(name, d, config.G), cfg, seed=seed,
                               log_every=0 if quick else 100)
                path.write_text(json.dumps(r, indent=1))
                print(f"{name} d={d} seed={seed}: relE={r['rel_energy_error']:.2e} "
                      f"L2={r['l2_error']:.2e} cpu={r['cpu_s']:.0f}s "
                      f"(Adam only: relE={r['adam_only']['rel_energy_error']:.2e})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--problems", nargs="+", default=["oscillator", "coupled"])
    ap.add_argument("--dims", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--lbfgs-refresh", type=int, default=0)
    a = ap.parse_args()
    main(a.quick, a.problems, a.dims, a.seeds, a.lbfgs_refresh)
