"""Combine the saved results into the comparisons the article quotes.

For every problem and dimension, the PINN's measured energy error is the target. For
each grid method the script finds the cheapest saved solve that reaches it:

* ``measured``   - a d-dimensional solve in results/e2_grids.json reached the target;
* ``separable``  - oscillator only: the exact grid answer from one 1D eigenproblem
                   gives the N needed (minimised over box sizes); the memory is that N^d
                   times the number of grid-sized vectors measured in experiment 2;
* ``estimated``  - coupled problem only, when no solve within the memory budget reached
                   the target: N is taken from the largest lower dimension where one did,
                   which the oscillator shows is how the per-axis requirement behaves
                   (the relative energy error does not grow with d at fixed N), and the
                   memory follows from it. Flagged in every table and figure.

For diffusion Monte Carlo, the statistical error falls as cpu^(-1/2); the CPU time
needed to bring the standard error down to the target is (c / target)^2 with
c = median(rel_se * sqrt(cpu)) over the three run lengths.

Writes results/summary.json and results/summary.md.
"""

import argparse
import json
from statistics import median

import numpy as np

import config
from pinncross.grids import oscillator_n_needed

METHODS = ("fd2", "fd4", "sinc")
BOXES = (3.5, 4.0, 4.5, 5.0, 5.5, 6.0)


def load(quick: bool):
    q = config.quick_suffix(quick)
    e2 = json.loads((config.RESULTS / f"e2_grids{q}.json").read_text())
    e4 = json.loads((config.RESULTS / f"e4_dmc{q}.json").read_text())
    folder = config.RESULTS / ("pinn_quick" if quick else "pinn")
    pinn = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    return e2, e4, pinn


def vectors_per_unknown(e2) -> float:
    big = [r["peak_bytes"] / (8.0 * r["unknowns"]) for r in e2
           if r["peak_bytes"] and r["unknowns"] >= 1e6]
    if not big:
        big = [r["peak_bytes"] / (8.0 * r["unknowns"]) for r in e2 if r["peak_bytes"]]
    return float(median(big))


def pinn_table(pinn):
    out = {}
    for r in pinn:
        out.setdefault((r["problem"], r["d"]), []).append(r)
    table = {}
    for key, runs in out.items():
        rel = [r["rel_energy_error"] for r in runs]
        table[key] = {
            "seeds": [r["seed"] for r in runs],
            "rel_energy_error": rel,
            "rel_energy_error_median": float(median(rel)),
            "rel_se_median": float(median(r["rel_se"] for r in runs)),
            "l2_error": [r["l2_error"] for r in runs],
            "l2_error_median": float(median(r["l2_error"] for r in runs)),
            "cpu_s": [r["cpu_s"] for r in runs],
            "cpu_s_median": float(median(r["cpu_s"] for r in runs)),
            "wall_s_median": float(median(r["wall_s"] for r in runs)),
            "adam_only_rel_energy_error_median":
                float(median(r["adam_only"]["rel_energy_error"] for r in runs)),
            "n_params": runs[0]["n_params"],
        }
    return table


def grid_match(e2, name, d, method, target, kvec, lower_dims_hits):
    rows = [r for r in e2 if r["problem"] == name and r["d"] == d and r["method"] == method]
    hit = [r for r in rows if r["rel_energy_error"] <= target]
    if hit:
        best = min(hit, key=lambda r: r["cpu_s"])
        return {"status": "measured", "n": best["n"], "L": best["L"],
                "unknowns": best["unknowns"], "cpu_s": best["cpu_s"],
                "memory_bytes": best["peak_bytes"],
                "rel_energy_error": best["rel_energy_error"]}
    if name == "oscillator":
        best = None
        for L in BOXES:
            n = oscillator_n_needed(method, L, target,
                                    nmax={"fd2": 40000, "fd4": 4000, "sinc": 300}[method])
            if n is not None and (best is None or n < best[0]):
                best = (n, L)
        if best is None:
            return {"status": "not reached"}
        n, L = best
        return {"status": "separable", "n": n, "L": L, "unknowns": float(n) ** d,
                "cpu_s": None, "memory_bytes": kvec * 8.0 * float(n) ** d}
    # coupled: use the per-axis N from the largest lower dimension that reached it
    if lower_dims_hits:
        dd, n = max(lower_dims_hits)
        return {"status": "estimated", "n": n, "from_d": dd, "unknowns": float(n) ** d,
                "cpu_s": None, "memory_bytes": kvec * 8.0 * float(n) ** d}
    return {"status": "not reached"}


def dmc_time(e4, name, d, target):
    runs = [r for r in e4["runs"] if r["problem"] == name and r["d"] == d]
    if not runs:
        return None
    c = median(r["rel_se"] * np.sqrt(r["cpu_s"]) for r in runs)
    longest = max(runs, key=lambda r: r["cpu_s"])
    return {"c": float(c), "cpu_s_to_target": float((c / target) ** 2),
            "longest_cpu_s": longest["cpu_s"], "longest_rel_error": longest["rel_energy_error"],
            "longest_rel_se": longest["rel_se"]}


def main(quick: bool):
    e2, e4, pinn = load(quick)
    kvec = vectors_per_unknown(e2)
    ptab = pinn_table(pinn)
    summary = {"vectors_per_unknown": kvec, "memory_budget": config.MEMORY_BUDGET,
               "rows": []}
    for name in ("oscillator", "coupled"):
        hits_by_method = {m: [] for m in METHODS}
        for d in config.DIMS[name]:
            if (name, d) not in ptab:
                continue
            p = ptab[(name, d)]
            target = p["rel_energy_error_median"]
            row = {"problem": name, "d": d, "pinn": p, "target": target, "grids": {}}
            for m in METHODS:
                g = grid_match(e2, name, d, m, target, kvec, hits_by_method[m])
                row["grids"][m] = g
                if g["status"] == "measured":
                    hits_by_method[m].append((d, g["n"]))
            row["dmc"] = dmc_time(e4, name, d, target)
            summary["rows"].append(row)

    # largest dimension each grid method reaches at fixed accuracy, oscillator (exact)
    caps = {"budget_5GB": config.MEMORY_BUDGET, "laptop_16GB": config.LAPTOP_MEMORY,
            "workstation_256GB": config.WORKSTATION_MEMORY}
    reach = []
    for target in (1e-3, 1e-4, 1e-5, 1e-6):
        for m in METHODS:
            n = min(x for x in (oscillator_n_needed(m, L, target,
                    nmax={"fd2": 40000, "fd4": 4000, "sinc": 300}[m]) for L in BOXES)
                    if x is not None)
            row = {"target": target, "method": m, "n": n}
            for k, cap in caps.items():
                d = 0
                while kvec * 8.0 * float(n) ** (d + 1) <= cap and d < 60:
                    d += 1
                row[k] = d
            reach.append(row)
    summary["oscillator_reach"] = reach

    path = config.RESULTS / f"summary{config.quick_suffix(quick)}.json"
    path.write_text(json.dumps(summary, indent=1, default=float))

    # markdown table
    lines = ["| problem | d | PINN energy error | PINN CPU | FD2 to match | FD4 to match | "
             "sinc to match | DMC CPU to match |",
             "|---|---|---|---|---|---|---|---|"]

    def fmt_grid(g):
        if g["status"] == "not reached":
            return "not reached"
        mem = g["memory_bytes"] / 1e9
        tag = {"measured": "", "separable": " (exact, 1D)", "estimated": " (est.)"}[g["status"]]
        cpu = f", {g['cpu_s']:.1f} s" if g.get("cpu_s") is not None else ""
        return f"N={g['n']}, {mem:.3g} GB{cpu}{tag}"

    for r in summary["rows"]:
        p = r["pinn"]
        dmc = r["dmc"]
        dmc_s = f"{dmc['cpu_s_to_target'] / 3600:.2g} h" if dmc else "-"
        lines.append(
            f"| {r['problem']} | {r['d']} | {p['rel_energy_error_median']:.1e} | "
            f"{p['cpu_s_median'] / 60:.1f} min | {fmt_grid(r['grids']['fd2'])} | "
            f"{fmt_grid(r['grids']['fd4'])} | {fmt_grid(r['grids']['sinc'])} | {dmc_s} |")
    (config.RESULTS / f"summary{config.quick_suffix(quick)}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"vectors per unknown: {kvec:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
