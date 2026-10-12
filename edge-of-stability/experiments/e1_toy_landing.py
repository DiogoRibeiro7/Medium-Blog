"""Experiment 1: where gradient descent lands in the two-parameter model
f(u, v) = u v, compared with where gradient flow lands.

Outputs
  results/e1_paths.json    illustrative trajectories (flow and GD) and time series
  results/e1_landing.csv   random starts x step sizes: flow-endpoint sharpness vs GD outcome
"""

import argparse
import json

import numpy as np
import pandas as pd

from config import RESULTS
from eos import toy

ETA_PATH = 0.5
STARTS = {"A": (2.5, 0.1), "B": (2.0, 0.4)}
ETAS_SCATTER = [0.3, 0.5, 0.7]


def paths(quick: bool = False):
    out = {"eta": ETA_PATH, "stable_bound": toy.stable_imbalance_bound(ETA_PATH), "starts": {}}
    for name, (u0, v0) in STARTS.items():
        _, uf, vf = toy.gradient_flow(u0, v0, t_max=60.0, n=600)
        u, v = toy.gd_trajectory(u0, v0, ETA_PATH, 400)
        r = toy.residual(u, v)
        d = toy.imbalance(u, v)
        pred = d[0] * np.concatenate([[1.0], np.cumprod(1.0 - ETA_PATH**2 * r[:-1] ** 2)])
        s = np.array([toy.sharpness(a, b) for a, b in zip(u, v)])
        ue, ve = toy.flow_endpoint(u0, v0)
        out["starts"][name] = {
            "u0": u0, "v0": v0,
            "flow_u": uf.tolist(), "flow_v": vf.tolist(),
            "flow_endpoint": [ue, ve], "flow_endpoint_sharpness": ue * ue + ve * ve,
            "gd_u": u.tolist(), "gd_v": v.tolist(),
            "residual": r.tolist(), "imbalance": d.tolist(), "imbalance_predicted": pred.tolist(),
            "sharpness": s.tolist(),
            "gd_final_sharpness": float(toy.sharpness(u[-1], v[-1])),
            "gd_final_imbalance": float(d[-1]),
            "max_abs_identity_error": float(np.max(np.abs(d - pred))),
        }
        print(f"start {name}=({u0},{v0}): flow sharpness {ue*ue+ve*ve:.3f}, "
              f"GD(eta={ETA_PATH}) sharpness {toy.sharpness(u[-1], v[-1]):.3f}, 2/eta={2/ETA_PATH}, "
              f"identity max error {np.max(np.abs(d - pred)):.2e}")
    with open(RESULTS / ("e1_paths_quick.json" if quick else "e1_paths.json"), "w") as fh:
        json.dump(out, fh)


def landing(n_starts: int, steps: int, quick: bool = False):
    rng = np.random.default_rng(1)
    u0 = np.exp(rng.uniform(np.log(0.05), np.log(4.0), n_starts))
    v0 = np.exp(rng.uniform(np.log(0.05), np.log(4.0), n_starts))
    rows = []
    for eta in ETAS_SCATTER:
        for a, b in zip(u0, v0):
            ue, ve = toy.flow_endpoint(a, b)
            u, v = toy.gd_trajectory(a, b, eta, steps)
            diverged = not np.isfinite(u[-1])
            if diverged:
                s_gd, r_end = np.nan, np.nan
            else:
                s_gd, r_end = toy.sharpness(u[-1], v[-1]), toy.residual(u[-1], v[-1])
            rows.append({
                "eta": eta, "u0": a, "v0": b,
                "flow_sharpness": ue * ue + ve * ve,
                "gd_sharpness": s_gd, "final_residual": r_end, "diverged": diverged,
                "converged": (not diverged) and abs(r_end) < 1e-9,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / ("e1_landing_quick.csv" if quick else "e1_landing.csv"), index=False)
    for eta, g in df.groupby("eta"):
        ceiling = 2 / eta
        allowed = g.flow_sharpness < ceiling
        conv = g[g.converged]
        print(f"eta={eta}: n={len(g)}, flow endpoint allowed={allowed.mean():.3f}, diverged={g.diverged.mean():.3f}, "
              f"converged={g.converged.mean():.3f}, max GD sharpness={conv.gd_sharpness.max():.4f} (2/eta={ceiling:.4f}), "
              f"allowed starts: max|GD-flow|={np.max(np.abs(conv[conv.flow_sharpness < ceiling].gd_sharpness - conv[conv.flow_sharpness < ceiling].flow_sharpness)):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    paths(q)
    landing(200 if q else 3000, 2000 if q else 20000, q)
