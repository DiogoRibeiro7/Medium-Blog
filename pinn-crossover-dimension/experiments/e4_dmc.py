"""Experiment 4: diffusion Monte Carlo energies by dimension.

First a time-step check (three values of tau at d = 2 and 8, both problems), then the
main runs at config.DMC_TAU with three run lengths per dimension, which trace the
error-versus-CPU-time curve. Each run uses its own seed.
"""

import argparse
import json

import config
from pinncross.dmc import run_dmc
from pinncross.problems import Problem


def main(quick: bool):
    path = config.RESULTS / f"e4_dmc{config.quick_suffix(quick)}.json"
    out = {"tau_check": [], "runs": []}
    walkers = 500 if quick else config.DMC_WALKERS
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
    for name in ("oscillator", "coupled"):
        for d in config.DIMS[name]:
            for steps in ([200] if quick else config.DMC_STEPS):
                seed += 1
                r = run_dmc(Problem(name, d, config.G), walkers, config.DMC_TAU, steps,
                            int(5 / config.DMC_TAU), seed=seed)
                out["runs"].append(r)
                print(f"{name} d={d} steps={steps}: relE={r['rel_energy_error']:.2e} "
                      f"(se {r['rel_se']:.1e}) cpu={r['cpu_s']:.1f}s", flush=True)
                path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
