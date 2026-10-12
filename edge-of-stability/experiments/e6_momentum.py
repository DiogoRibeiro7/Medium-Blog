"""Experiment 6: the ceiling moves with momentum.

Heavy-ball update m <- beta m + g, theta <- theta - eta m. On a quadratic with
curvature lambda the iteration is stable iff eta * lambda < 2 + 2 beta, so the
sharpness ceiling becomes (2 + 2 beta) / eta (Cohen et al., 2021, eq. 1).

Output: results/e6_momentum.npz
"""

import argparse

import numpy as np

from config import INIT_GAIN, N_TRAIN, RESULTS, SIZES
from eos.data import digits_split
from eos.mlp import Problem, init_params
from eos.train import run_gd

BETA = 0.9
ETAS = [0.15, 0.3, 0.5]


def main(quick: bool = False):
    steps, every = (400, 50) if quick else (6000, 50)
    data = digits_split(N_TRAIN)
    params = init_params(0, SIZES, INIT_GAIN)
    prob = Problem.build(SIZES, "tanh", data.x_train, data.y_train, params)
    theta0 = prob.flatten(params)
    out = {}
    for eta in ETAS:
        tr = run_gd(prob, theta0, eta, steps, every, momentum=BETA)
        ceiling = (2 + 2 * BETA) / eta
        out[f"loss_{eta}"] = tr.loss
        out[f"steps_{eta}"] = tr.eval_steps
        out[f"sharp_{eta}"] = tr.sharpness
        last = tr.sharpness[tr.eval_steps >= steps - 2000] if not tr.diverged else np.array([np.nan])
        print(f"beta={BETA} eta={eta}: ceiling (2+2b)/eta={ceiling:.3f}, plain 2/eta={2/eta:.3f}, diverged={tr.diverged}, "
              f"final loss={tr.loss[-1]:.5f}, sharpness last 2000: median={np.median(last):.3f} "
              f"[{np.min(last):.3f}, {np.max(last):.3f}]", flush=True)
    np.savez(RESULTS / ("e6_momentum_quick.npz" if quick else "e6_momentum.npz"), etas=np.array(ETAS), beta=BETA, **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
