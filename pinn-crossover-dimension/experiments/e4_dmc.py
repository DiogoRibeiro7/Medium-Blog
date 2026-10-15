"""Experiment 4: diffusion Monte Carlo energies by dimension, extrapolated to tau -> 0.

The time-step check (three values of tau at d = 2 and 8, both problems) shows the
energy bias falling as tau^2, as the symmetric split predicts, and large on the
coupled problem: about 2.6% at tau = 0.04 in d = 2. Every main estimate is therefore
the standard Richardson extrapolation from two runs at tau and tau/2 covering the same
imaginary time,

    E = (4 E(tau/2) - E(tau)) / 3,   se = sqrt((4 se(tau/2))^2 + se(tau)^2) / 3,

and its CPU time is the sum of both runs. Three run lengths per dimension trace the
error-versus-CPU curve. Each run uses its own seed.
"""

import argparse
import json

import numpy as np

import config
from pinncross.dmc import run_dmc
from pinncross.problems import Problem


def extrapolated(P, walkers, tau, steps, seed):
    a = run_dmc(P, walkers, tau, steps, int(5 / tau), seed=seed)
    b = run_dmc(P, walkers, tau / 2, 2 * steps, int(5 / tau) * 2, seed=seed + 50_000)
    E = (4.0 * b["E_potential"] - a["E_potential"]) / 3.0
    se = float(np.sqrt((4.0 * b["se_potential"]) ** 2 + a["se_potential"] ** 2) / 3.0)
    return {
        "problem": P.name, "d": P.d, "g": P.g, "method": "dmc", "tau": tau,
        "steps": steps, "n_walkers": walkers, "E": E, "se": se, "E0": P.E0,
        "rel_energy_error": abs(E - P.E0) / P.E0, "rel_se": se / P.E0,
        "cpu_s": a["cpu_s"] + b["cpu_s"], "wall_s": a["wall_s"] + b["wall_s"],
        "runs": [a, b],
    }


def main(quick: bool, redo_tau_check: bool):
    path = config.RESULTS / f"e4_dmc{config.quick_suffix(quick)}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    out = {"tau_check": [], "runs": []}
    walkers = 500 if quick else config.DMC_WALKERS
    if old.get("tau_check") and not redo_tau_check and not quick:
        out["tau_check"] = old["tau_check"]
    else:
        seed = 0
        for name in ("oscillator", "coupled"):
            for d in (2, 8):
                for tau in config.DMC_TAU_CHECK:
                    seed += 1
                    steps = 300 if quick else 40000
                    r = run_dmc(Problem(name, d, config.G), walkers, tau, steps,
                                int(5 / tau), seed=seed)
                    out["tau_check"].append(r)
                    print(f"tau check {name} d={d} tau={tau}: E={r['E_potential']:.5f} "
                          f"± {r['se_potential']:.5f} (E0={r['E0']}) cpu={r['cpu_s']:.0f}s",
                          flush=True)
                    path.write_text(json.dumps(out, indent=1))
    seed = 100
    for name in ("oscillator", "coupled"):
        for d in config.DIMS[name]:
            for steps in ([200] if quick else config.DMC_STEPS):
                seed += 1
                r = extrapolated(Problem(name, d, config.G), walkers, config.DMC_TAU, steps, seed)
                out["runs"].append(r)
                print(f"{name} d={d} steps={steps}: relE={r['rel_energy_error']:.2e} "
                      f"(se {r['rel_se']:.1e}) cpu={r['cpu_s']:.1f}s", flush=True)
                path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--redo-tau-check", action="store_true")
    a = ap.parse_args()
    main(a.quick, a.redo_tau_check)
