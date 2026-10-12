"""Experiment 4: step-size sweep on the tanh network (digits), several seeds.

For every (eta, seed): 6000 full-batch GD steps; the "plateau" sharpness is the
median over the evaluations in the last 1000 steps. Also recorded: adaptive
(rescaling-invariant) sharpness at the end, training loss, test accuracy and
test MSE.

Output: results/e4_sweep.csv
"""

import argparse

import numpy as np
import pandas as pd

from config import INIT_GAIN, N_TRAIN, RESULTS, SIZES
from eos.data import digits_split
from eos.mlp import Problem, accuracy, init_params, mse
from eos.sharpness import adaptive_sharpness
from eos.train import run_gd

ETAS = [0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5]
SEEDS = [0, 1, 2, 3]


def main(quick: bool = False):
    steps, every, window = (600, 100, 200) if quick else (6000, 100, 1000)
    etas = ETAS[::4] if quick else ETAS
    seeds = SEEDS[:1] if quick else SEEDS
    data = digits_split(N_TRAIN)
    rows = []
    for seed in seeds:
        params = init_params(seed, SIZES, INIT_GAIN)
        prob = Problem.build(SIZES, "tanh", data.x_train, data.y_train, params)
        theta0 = prob.flatten(params)
        for eta in etas:
            tr = run_gd(prob, theta0, eta, steps, every)
            row = {"seed": seed, "eta": eta, "ceiling": 2 / eta, "diverged": tr.diverged,
                   "initial_sharpness": tr.sharpness[0]}
            if not tr.diverged:
                last = tr.sharpness[tr.eval_steps >= steps - window]
                a, _ = adaptive_sharpness(prob, tr.params)
                row.update({
                    "plateau_sharpness": float(np.median(last)),
                    "plateau_min": float(last.min()), "plateau_max": float(last.max()),
                    "final_sharpness": float(tr.sharpness[-1]),
                    "adaptive_sharpness": a,
                    "train_loss": float(tr.loss[-1]),
                    "train_acc": accuracy(prob, tr.params, data.x_train, data.labels_train),
                    "test_acc": accuracy(prob, tr.params, data.x_test, data.labels_test),
                    "test_mse": mse(prob, tr.params, data.x_test, data.y_test),
                    "loss_increases_frac": float(np.mean(np.diff(tr.loss) > 0)),
                })
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / ("e4_sweep_quick.csv" if quick else "e4_sweep.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
