"""Experiment 2: gradient descent on the balanced two-parameter model for step
sizes beyond the stability threshold: 2-cycle, period doubling, chaos, crisis.

On u = v = s, gradient descent is the cubic map g(s) = s (1 + eta - eta s^2).
Closed forms checked here against numerics:
  * the minimiser s = 1 loses stability at eta = 1 (multiplier 1 - 2 eta);
  * the 2-cycle has multiplier 9 - 2 (1 + eta)^2, stable for 1 < eta < sqrt(5) - 1;
  * the mean loss over the 2-cycle is (1 - 1/eta^2) / 4;
  * the bounded invariant interval is destroyed at eta = 2 (boundary crisis).

Outputs
  results/e2_bifurcation.npz   orbit residuals and Lyapunov exponents on an eta grid
  results/e2_summary.json      period-doubling points, Feigenbaum ratios, checks
"""

import argparse
import json

import numpy as np
from scipy.optimize import brentq

from config import RESULTS
from eos import toy

S0 = 1.01


def cycle_multiplier(eta: float, p: int) -> float:
    pts = toy.orbit(eta, S0, 200000, p)
    s = pts[0]
    for _ in range(100):  # Newton on g^p(s) - s = 0
        x, d = s, 1.0
        for _ in range(p):
            d *= toy.balanced_map_derivative(x, eta)
            x = toy.balanced_map(x, eta)
        step = (x - s) / (d - 1.0)
        s -= step
        if abs(step) < 1e-15:
            break
    x, m = s, 1.0
    for _ in range(p):
        m *= toy.balanced_map_derivative(x, eta)
        x = toy.balanced_map(x, eta)
    return m


def doubling_points(n_doublings: int):
    pts = [1.0]
    width = 0.3
    for k in range(1, n_doublings + 1):
        p = 2**k
        lo = pts[-1] + 1e-9
        grid = np.linspace(lo, lo + width, 300)
        with np.errstate(all="ignore"):
            vals = np.array([cycle_multiplier(e, p) + 1.0 for e in grid])
        i = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0][0]
        with np.errstate(all="ignore"):
            root = brentq(lambda e: cycle_multiplier(e, p) + 1.0, grid[i], grid[i + 1], xtol=1e-15)
        width = (root - pts[-1]) * 0.6
        pts.append(root)
    return np.array(pts)


def main(quick: bool = False):
    n_grid = 400 if quick else 4000
    etas = np.linspace(0.9, 2.05, n_grid)
    keep, burn = 200, (2000 if quick else 20000)
    resid = np.full((n_grid, keep), np.nan)
    lyap = np.full(n_grid, np.nan)
    for i, e in enumerate(etas):
        with np.errstate(all="ignore"):
            pts = toy.orbit(e, S0, burn, keep)
            if pts.size:
                resid[i] = pts**2 - 1.0
                lyap[i] = toy.lyapunov_exponent(e, S0, burn, 4000 if quick else 20000)
    np.savez(RESULTS / ("e2_bifurcation_quick.npz" if quick else "e2_bifurcation.npz"), etas=etas, residual=resid, lyapunov=lyap)

    bps = doubling_points(4 if quick else 6)
    gaps = np.diff(bps)
    ratios = gaps[:-1] / gaps[1:]
    delta = 4.669201609102990
    eta_inf = bps[-1] + gaps[-1] / (delta - 1.0)

    # closed-form checks
    checks = {}
    for e in [1.05, 1.1, 1.2]:
        pts = toy.orbit(e, S0, 200000, 2)
        mult = np.prod(toy.balanced_map_derivative(pts, e))
        mean_loss = np.mean(0.5 * (pts**2 - 1.0) ** 2)
        checks[str(e)] = {
            "multiplier_numeric": float(mult), "multiplier_formula": 9 - 2 * (1 + e) ** 2,
            "mean_loss_numeric": float(mean_loss), "mean_loss_formula": (1 - 1 / e**2) / 4,
            "cycle_points": pts.tolist(),
        }
    crisis = {}
    for e in [1.99, 1.999, 2.001, 2.01]:
        with np.errstate(all="ignore"):
            crisis[str(e)] = "bounded" if toy.orbit(e, S0, 200000, 10).size else "escapes"
    # imbalanced starts beyond the threshold: does the imbalance die out?
    rng = np.random.default_rng(3)
    balance = {}
    for e in [1.05, 1.15, 1.25, 1.5]:
        n_ok = n_bal = 0
        for _ in range(200 if quick else 1000):
            u0, v0 = rng.uniform(0.5, 1.6, 2)
            with np.errstate(all="ignore"):
                u, v = toy.gd_trajectory(u0, v0, e, 20000)
            if np.isfinite(u[-1]):
                n_ok += 1
                n_bal += abs(u[-1] ** 2 - v[-1] ** 2) < 1e-8
        balance[str(e)] = {"bounded": n_ok, "balanced_at_end": int(n_bal)}

    summary = {
        "doubling_points": bps.tolist(),
        "sqrt5_minus_1": float(np.sqrt(5) - 1),
        "feigenbaum_ratios": ratios.tolist(),
        "eta_infinity_estimate": float(eta_inf),
        "two_cycle_checks": checks,
        "crisis": crisis,
        "imbalanced_starts": balance,
        "lyapunov_max": float(np.nanmax(lyap)),
        "fraction_positive_lyapunov_1.31_to_2": float(np.nanmean(lyap[(etas > 1.31) & (etas < 2)] > 0)),
    }
    with open(RESULTS / ("e2_summary_quick.json" if quick else "e2_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def wide_starts(quick: bool = False):
    """Boundedness beyond the threshold from the wider start distribution of experiment 1
    (both coordinates log-uniform on [0.05, 4]). Output: results/e2_wide_starts.json"""
    rng = np.random.default_rng(5)
    n, steps = (100, 2000) if quick else (400, 5000)
    out = {}
    for e in [1.05, 1.25, 1.5]:
        u = np.exp(rng.uniform(np.log(0.05), np.log(4.0), n))
        v = np.exp(rng.uniform(np.log(0.05), np.log(4.0), n))
        with np.errstate(all="ignore"):
            for _ in range(steps):
                r = u * v - 1.0
                u, v = u - e * r * v, v - e * r * u
                u = np.where(np.abs(u) > 1e8, np.nan, u)
                v = np.where(np.abs(v) > 1e8, np.nan, v)
        b = np.isfinite(u) & np.isfinite(v)
        out[str(e)] = {"starts": n, "bounded": int(b.sum()),
                       "balanced_among_bounded": int(np.sum(np.abs(u[b] ** 2 - v[b] ** 2) < 1e-8))}
    print(json.dumps(out, indent=2))
    with open(RESULTS / ("e2_wide_starts_quick.json" if quick else "e2_wide_starts.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part", choices=["all", "main", "wide"], default="all")
    a = ap.parse_args()
    if a.part in ("all", "main"):
        main(a.quick)
    if a.part in ("all", "wide"):
        wide_starts(a.quick)
