"""Experiment 3: sharpness and loss along full-batch gradient descent on a
tanh network (digits), for several step sizes.

Output: results/e3_traces.npz
"""

import argparse

import numpy as np

from config import INIT_GAIN, N_TRAIN, RESULTS, SIZES
from eos.data import digits_split
from eos.mlp import Problem, accuracy, init_params
from eos.train import run_gd

ETAS = [0.05, 0.1, 0.15, 0.3]


def main(quick: bool = False):
    steps, every = (400, 50) if quick else (8000, 50)
    data = digits_split(N_TRAIN)
    params = init_params(0, SIZES, INIT_GAIN)
    prob = Problem.build(SIZES, "tanh", data.x_train, data.y_train, params)
    theta0 = prob.flatten(params)
    out = {}
    for eta in ETAS:
        tr = run_gd(prob, theta0, eta, steps, every)
        out[f"loss_{eta}"] = tr.loss
        out[f"steps_{eta}"] = tr.eval_steps
        out[f"sharp_{eta}"] = tr.sharpness
        out[f"testacc_{eta}"] = accuracy(prob, tr.params, data.x_test, data.labels_test)
        last = tr.sharpness[tr.eval_steps >= steps - 2000]
        print(f"eta={eta}: 2/eta={2/eta:.2f}  final loss={tr.loss[-1]:.5f}  "
              f"sharpness last 2000 steps: median={np.median(last):.3f} "
              f"range=[{last.min():.3f}, {last.max():.3f}]  test acc={out[f'testacc_{eta}']:.4f}", flush=True)
    np.savez(RESULTS / ("e3_traces_quick.npz" if quick else "e3_traces.npz"), etas=np.array(ETAS), **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
