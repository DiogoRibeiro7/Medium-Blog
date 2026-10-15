"""Experiment 5: diffusion Monte Carlo given the same CPU time as the PINN.

Experiment 4 gives the error-versus-time law of Monte Carlo from short runs. Here the
comparison is measured instead of inferred from that law: for the dimensions where the
grids no longer reach the PINN's accuracy within the memory budget (and d = 5 as a
reference), one extrapolated DMC estimate is run with its CPU time set to the PINN's
training time for seed 0 in that dimension. Run after experiment 3.
"""

import argparse
import json

import config
from e4_dmc import extrapolated
from pinncross.problems import Problem

CASES = [("oscillator", 8), ("coupled", 6), ("coupled", 7), ("coupled", 8)]


def main(quick: bool):
    q = config.quick_suffix(quick)
    e4 = json.loads((config.RESULTS / f"e4_dmc{q}.json").read_text())
    folder = config.RESULTS / ("pinn_quick" if quick else "pinn")
    path = config.RESULTS / f"e5_dmc_at_pinn_budget{q}.json"
    out = json.loads(path.read_text()) if path.exists() else []
    done = {(r["problem"], r["d"]) for r in out}
    walkers = 500 if quick else config.DMC_WALKERS
    for k, (name, d) in enumerate(CASES):
        if (name, d) in done:
            continue
        pf = folder / f"{name}-d{d}-s0.json"
        if not pf.exists():
            continue
        budget = json.loads(pf.read_text())["cpu_s"]
        ref = max((r for r in e4["runs"] if r["problem"] == name and r["d"] == d),
                  key=lambda r: r["steps"])
        per_step = ref["cpu_s"] / ref["steps"]
        steps = max(200, int(budget / per_step))
        if quick:
            steps = 200
        r = extrapolated(Problem(name, d, config.G), walkers, config.DMC_TAU, steps,
                         seed=900 + k)
        r["pinn_cpu_s"] = budget
        out.append(r)
        path.write_text(json.dumps(out, indent=1))
        print(f"{name} d={d}: DMC relE={r['rel_energy_error']:.2e} (se {r['rel_se']:.1e}) "
              f"cpu={r['cpu_s']:.0f}s vs PINN cpu={budget:.0f}s", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
