"""Experiment 1: how fast each grid method converges, and how much the box size matters.

* Oscillator: exact grid answers in any dimension from the separable 1D eigenproblem.
* Coupled problem: actual d-dimensional solves in d = 2 and 3, to show the same
  convergence orders hold when nothing separates.
* Box sensitivity: for each method and target accuracy, the smallest N per axis that
  reaches it, for several box half-widths L.
"""

import argparse
import json

import numpy as np

import config
from pinncross.grids import (oscillator_n_needed, separable_oscillator,
                             solve_ground_state)
from pinncross.problems import Problem


def main(quick: bool):
    out = {"L": config.L, "oscillator": [], "coupled": [], "box_sensitivity": []}

    # oscillator, any d (the energy error does not depend on d; the L2 error does)
    sweep = {"fd2": [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
             "fd4": [8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192],
             "sinc": list(range(4, 41, 2))}
    for method, ns in sweep.items():
        for n in ns[: 4 if quick else None]:
            r = separable_oscillator(method, n, config.L, 1)
            r["l2_error_nodal_by_d"] = {
                d: separable_oscillator(method, n, config.L, d)["l2_error_nodal"]
                for d in range(1, 9)}
            out["oscillator"].append(r)

    # coupled problem, real multi-dimensional solves
    csweep = {"fd2": [8, 12, 16, 24, 32, 48, 64, 96, 128],
              "fd4": [8, 12, 16, 20, 24, 32, 40, 48, 64],
              "sinc": [6, 8, 10, 12, 14, 16, 18, 20, 24, 28]}
    for d in (2, 3):
        P = Problem("coupled", d, config.G)
        for method, ns in csweep.items():
            for n in ns[: 3 if quick else None]:
                if d == 3 and n**3 > 1.2e6:
                    continue
                r = solve_ground_state(P, method, n, config.L)
                out["coupled"].append(r)
                print(f"coupled d={d} {method:4s} N={n:3d} relE={r['rel_energy_error']:.2e} "
                      f"L2={r['l2_error_nodal']:.2e} it={r['iterations']}", flush=True)

    # box convergence: energy against box half-width at fixed spacing h ~ 0.2, sinc grid.
    # The grid experiments use, for each problem, the smallest box whose energy is within
    # 1e-10 (relative) of the largest box; that rule uses no knowledge of the answer.
    out["box_convergence"] = []
    for name, d in (("oscillator", 1), ("coupled", 2), ("coupled", 3)):
        P = Problem(name, d, config.G)
        Ls = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
        if quick:
            Ls = Ls[-3:]
        for L in Ls:
            n = int(round(2 * L / 0.2)) - 1
            if d == 3 and n > 50:
                n = 50
            r = solve_ground_state(P, "sinc", n, L, track_memory=False)
            out["box_convergence"].append({"problem": name, "d": d, "L": L, "n": n,
                                           "E": r["E"], "E0": P.E0})
            print(f"box {name} d={d} L={L} N={n} E-E0={r['E'] - P.E0:.3e}", flush=True)

    # box sensitivity on the oscillator: N per axis needed for each target, by box size
    targets = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
    for method in ("fd2", "fd4", "sinc"):
        for L in (3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
            row = {"method": method, "L": L}
            for t in targets:
                nmax = {"fd2": 20000, "fd4": 3000, "sinc": 200}[method]
                row[f"{t:g}"] = oscillator_n_needed(method, L, t, nmax=nmax)
            out["box_sensitivity"].append(row)
            print(row, flush=True)

    path = config.RESULTS / f"e1_convergence{config.quick_suffix(quick)}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
