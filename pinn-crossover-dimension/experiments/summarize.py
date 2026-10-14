"""Combine the saved results into the comparisons the article quotes.

For every problem and dimension, the PINN's measured energy error is the target. For
each grid method the script finds the cheapest saved solve that reaches it:

* ``measured``   - a d-dimensional solve in results/e2_grids.json reached the target;
* ``separable``  - oscillator only: the exact grid answer from one 1D eigenproblem
                   gives the N needed (minimised over box sizes); the memory is that N^d
                   times the number of grid-sized vectors measured in experiment 2;
* ``estimated``  - coupled problem only, when no solve within the memory budget reached
                   the target: N is the smallest that reached the same target in the
                   highest lower dimension where any solve did (on the oscillator the
                   relative energy error does not depend on d at fixed N; on the coupled
                   problem the requirement grows slowly, so this understates what the
                   grid needs). Flagged in every table and figure.

For diffusion Monte Carlo, the statistical error falls as cpu^(-1/2); the CPU time
needed to bring the standard error down to the target is (c / target)^2 with
c = median(rel_se * sqrt(cpu)) over the three run lengths.

The same matching is repeated on the relative L2 error of the wavefunction (nodal error
for the grids, which for the sinc grid equals the continuous error of its interpolant).
Because the PINN's energy is a Rayleigh quotient, its energy error is roughly the square
of its wavefunction error, so matching on energy asks more of the grids than matching
on the wavefunction does.

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


def stable_hits(rows, key, target):
    """Solves from which the error stays below target for every larger N at the same box.

    At coarse N the discretisation error and the box-truncation error can cancel by
    accident; a solve that reaches the target only that way is not counted."""
    out = []
    for L in sorted({r["L"] for r in rows}):
        seq = sorted((r for r in rows if r["L"] == L), key=lambda r: r["n"])
        for i in range(len(seq)):
            if all(r[key] <= target for r in seq[i:]):
                out.append(seq[i])
                break
    return out


def grid_match(e2, name, d, method, target, kvec, lower_dims_hits, metric="energy"):
    """Cheapest grid answer reaching ``target`` in the given metric (energy or L2)."""
    key = "rel_energy_error" if metric == "energy" else "l2_error_nodal"
    rows = [r for r in e2 if r["problem"] == name and r["d"] == d and r["method"] == method]
    hit = stable_hits(rows, key, target)
    if hit:
        best = min(hit, key=lambda r: (r["peak_bytes"], r["cpu_s"]))
        return {"status": "measured", "n": best["n"], "L": best["L"],
                "unknowns": best["unknowns"], "cpu_s": best["cpu_s"],
                "memory_bytes": best["peak_bytes"], "error": best[key]}
    if name == "oscillator":
        best = None
        for L in BOXES:
            n = oscillator_n_needed(method, L, target,
                                    nmax={"fd2": 40000, "fd4": 4000, "sinc": 300}[method],
                                    metric=key, d=d if metric == "l2" else 1)
            if n is not None and (best is None or n < best[0]):
                best = (n, L)
        if best is None:
            return {"status": "not reached"}
        n, L = best
        return {"status": "separable", "n": n, "L": L, "unknowns": float(n) ** d,
                "cpu_s": None, "memory_bytes": kvec * 8.0 * float(n) ** d}
    # coupled: the smallest N per axis that reached the same target in the highest lower
    # dimension where any solve reached it. The per-axis requirement of the coupled
    # problem grows slowly with d, so this tends to understate what the grid needs.
    for dd in range(d - 1, 1, -1):
        lower = stable_hits([r for r in e2 if r["problem"] == name and r["d"] == dd
                             and r["method"] == method], key, target)
        if lower:
            n = min(r["n"] for r in lower)
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
        hits_l2 = {m: [] for m in METHODS}
        for d in config.DIMS[name]:
            if (name, d) not in ptab:
                continue
            p = ptab[(name, d)]
            target = p["rel_energy_error_median"]
            if target > 0.1:
                # the PINN failed in this dimension (median energy error above 10%):
                # there is no accuracy to match
                summary["rows"].append({"problem": name, "d": d, "pinn": p,
                                        "pinn_failed": True, "target": target})
                continue
            target_l2 = p["l2_error_median"]
            row = {"problem": name, "d": d, "pinn": p, "target": target,
                   "target_l2": target_l2, "grids": {}, "grids_l2": {}}
            for m in METHODS:
                g = grid_match(e2, name, d, m, target, kvec, hits_by_method[m])
                row["grids"][m] = g
                if g["status"] == "measured":
                    hits_by_method[m].append((d, g["n"]))
                g2 = grid_match(e2, name, d, m, target_l2, kvec, hits_l2[m], metric="l2")
                row["grids_l2"][m] = g2
                if g2["status"] == "measured":
                    hits_l2[m].append((d, g2["n"]))
            row["dmc"] = dmc_time(e4, name, d, target)
            row["pinn_failed"] = False
            summary["rows"].append(row)

    q = config.quick_suffix(quick)
    e5p = config.RESULTS / f"e5_dmc_at_pinn_budget{q}.json"
    summary["dmc_at_pinn_budget"] = json.loads(e5p.read_text()) if e5p.exists() else []
    for r in summary["dmc_at_pinn_budget"]:
        r.pop("runs", None)
    vfolder = config.RESULTS / ("pinn_variants_quick" if quick else "pinn_variants")
    summary["pinn_variants"] = [
        {k: v for k, v in json.loads(f.read_text()).items() if k != "history"}
        for f in sorted(vfolder.glob("*.json"))] if vfolder.exists() else []

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
        if r["pinn_failed"]:
            lines.append(f"| {r['problem']} | {r['d']} | {p['rel_energy_error_median']:.1e} "
                         f"(failed) | {p['cpu_s_median'] / 60:.1f} min | - | - | - | - |")
            continue
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
