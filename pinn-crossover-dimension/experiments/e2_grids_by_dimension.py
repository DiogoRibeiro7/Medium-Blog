"""Experiment 2: real d-dimensional grid solves, for every method, up to the memory budget.

For each problem, dimension and method, N per axis increases through config.GRID_N
until the relative energy error is below 1e-8, the predicted memory exceeds
config.MEMORY_BUDGET, or the last solve took more than CPU_CAP seconds of CPU. Each
solve records the error, CPU and wall time, LOBPCG iterations and the peak memory
traced by tracemalloc. Results are appended to
results/e2_grids.json as they finish, so the script can be stopped and restarted.
"""

import argparse
import json

import config
from pinncross.grids import solve_ground_state
from pinncross.problems import Problem

VECTORS = 20   # conservative number of length-N^d arrays LOBPCG and the operator hold
CPU_CAP = 600  # stop refining once one solve takes more than 10 min of CPU (about one PINN run)


def run_one_method(rows, path, P, method, L, budget):
    """Increase N until the error, memory or CPU stopping rule is met."""
    name, d = P.name, P.d
    for n in config.GRID_N[method]:
        if VECTORS * 8.0 * n**d > budget:
            print(f"{name} d={d} {method} L={L} N={n}: over memory budget", flush=True)
            return
        key = (name, d, method, n, L)
        prev = [r for r in rows if (r["problem"], r["d"], r["method"], r["n"], r["L"]) == key]
        if prev:
            r = prev[0]
        else:
            r = solve_ground_state(P, method, n, L)
            rows.append(r)
            path.write_text(json.dumps(rows, indent=1))
        print(f"{name} d={d} {method:4s} L={L} N={n:4d} unknowns={r['unknowns']:.2e} "
              f"relE={r['rel_energy_error']:.2e} L2={r['l2_error_nodal']:.2e} "
              f"cpu={r['cpu_s']:.1f}s peak={(r['peak_bytes'] or 0) / 1e9:.2f}GB "
              f"it={r['iterations']}", flush=True)
        if r["rel_energy_error"] < 1e-8:
            return
        if r["cpu_s"] > CPU_CAP:
            print(f"{name} d={d} {method} L={L}: last solve over {CPU_CAP:.0f} s CPU, stop",
                  flush=True)
            return


def main(quick: bool, problems, dims, methods):
    path = config.RESULTS / f"e2_grids{config.quick_suffix(quick)}.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    budget = 2e8 if quick else config.MEMORY_BUDGET
    for name in problems:
        for d in dims or config.DIMS[name]:
            if d not in config.DIMS[name]:
                continue
            P = Problem(name, d, config.G)
            for method in methods:
                for L in config.GRID_L[name]:
                    run_one_method(rows, path, P, method, L, budget)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--problems", nargs="+", default=["oscillator", "coupled"])
    ap.add_argument("--dims", nargs="+", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=["fd2", "fd4", "sinc"])
    a = ap.parse_args()
    main(a.quick, a.problems, a.dims, a.methods)
