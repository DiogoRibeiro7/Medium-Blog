"""Draw every figure from the saved results (nothing is recomputed except psi0 in fig 1)."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from pinncross import style  # noqa: E402
from pinncross.problems import Problem  # noqa: E402

style.apply()
COL = {"fd2": style.ORANGE, "fd4": style.YELLOW, "sinc": style.BLUE,
       "pinn": style.AQUA, "dmc": style.INK2}
NAME = {"fd2": "2nd-order finite differences", "fd4": "4th-order finite differences",
        "sinc": "Spectral (sinc) grid", "pinn": "PINN", "dmc": "Diffusion Monte Carlo"}
PROB = {"oscillator": "Harmonic oscillator (separable)",
        "coupled": "Coupled quartic (non-separable)"}
CREDIT = "Diogo Ribeiro"


def credit(fig):
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=8, color=style.MUTED)


def load(quick):
    q = config.quick_suffix(quick)
    e1 = json.loads((config.RESULTS / f"e1_convergence{q}.json").read_text())
    e2 = json.loads((config.RESULTS / f"e2_grids{q}.json").read_text())
    e4 = json.loads((config.RESULTS / f"e4_dmc{q}.json").read_text())
    s = json.loads((config.RESULTS / f"summary{q}.json").read_text())
    folder = config.RESULTS / ("pinn_quick" if quick else "pinn")
    pinn = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    return e1, e2, e4, s, pinn


def fig1_problems(q):
    """|psi0|^2 in two dimensions for both problems."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    t = np.linspace(-2.6, 2.6, 401)
    xx, yy = np.meshgrid(t, t, indexing="xy")
    pts = np.stack([xx, yy], -1)
    for ax, name in zip(axes, ("oscillator", "coupled")):
        P = Problem(name, 2, config.G)
        dens = np.exp(-2.0 * P.phi(pts))
        ax.contourf(xx, yy, dens / dens.max(), levels=np.linspace(0.02, 1, 11),
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                        "b", ["#eef4fc"] + style.SEQ))
        ax.contour(xx, yy, dens / dens.max(), levels=[0.05, 0.25, 0.5, 0.75],
                   colors=[style.INK2], linewidths=0.6)
        ax.set_aspect("equal")
        ax.set_title(PROB[name])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(False)
    fig.suptitle("Ground-state density $|\\psi_0|^2$ in two dimensions", x=0.01, y=1.04,
                 ha="left", fontweight="bold")
    credit(fig)
    fig.savefig(config.FIGURES / f"fig1_problems{q}.png")
    plt.close(fig)


def fig2_convergence(e1, e2, q):
    """Energy error against points per axis: 1D oscillator (exact) and coupled d = 3."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    ax = axes[0]
    for m in ("fd2", "fd4", "sinc"):
        rows = sorted((r for r in e1["oscillator"] if r["method"] == m), key=lambda r: r["n"])
        ax.loglog([r["n"] for r in rows], [max(r["rel_energy_error"], 1e-16) for r in rows],
                  "-o", ms=3, color=COL[m], label=NAME[m])
    ax.set_title("Oscillator (any d; exact grid answer)")
    ax.set_xlabel("points per axis, N")
    ax.set_ylabel("relative energy error")
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    for m in ("fd2", "fd4", "sinc"):
        rows = sorted((r for r in e1["coupled"] if r["method"] == m and r["d"] == 3),
                      key=lambda r: r["n"])
        if rows:
            ax.loglog([r["n"] for r in rows], [max(r["rel_energy_error"], 1e-16) for r in rows],
                      "-o", ms=3, color=COL[m], label=NAME[m])
    ax.set_title("Coupled problem, d = 3 (full 3D solves)")
    ax.set_xlabel("points per axis, N")
    ax.set_ylim(1e-12, 1)
    for a in axes:
        a.axhline(1e-4, color=style.MUTED, lw=0.8, ls=":")
    credit(fig)
    fig.savefig(config.FIGURES / f"fig2_convergence{q}.png")
    plt.close(fig)


def fig3_memory(s, pinn, q):
    """Memory each grid method needs to match the PINN's energy error, by dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    rss = [r.get("max_rss_bytes") for r in pinn if r.get("max_rss_bytes")]
    for ax, name in zip(axes, ("oscillator", "coupled")):
        rows = [r for r in s["rows"] if r["problem"] == name and not r["pinn_failed"]]
        for m in ("fd2", "fd4", "sinc"):
            ds, mem, solid = [], [], []
            for r in rows:
                g = r["grids"][m]
                if g["status"] == "not reached":
                    continue
                ds.append(r["d"])
                mem.append(g["memory_bytes"])
                solid.append(g["status"] == "measured")
            if not ds:
                continue
            ax.semilogy(ds, mem, "-", color=COL[m], label=NAME[m])
            ax.scatter([d for d, f in zip(ds, solid) if f], [x for x, f in zip(mem, solid) if f],
                       color=COL[m], s=28, zorder=3)
            ax.scatter([d for d, f in zip(ds, solid) if not f],
                       [x for x, f in zip(mem, solid) if not f],
                       facecolor=style.SURFACE, edgecolor=COL[m], s=28, zorder=3)
        if rss:
            ax.axhline(max(rss), color=COL["pinn"], lw=1.6, ls="--",
                       label="PINN (whole process, peak)")
        for cap, lab in ((config.MEMORY_BUDGET, "5 GB (this machine)"),
                         (config.LAPTOP_MEMORY, "16 GB laptop"),
                         (config.WORKSTATION_MEMORY, "256 GB workstation")):
            ax.axhline(cap, color=style.MUTED, lw=0.8, ls=":")
            ax.text(rows[0]["d"] if rows else 1, cap * 1.25, lab, fontsize=8,
                    color=style.INK2)
        ax.set_title(PROB[name])
        ax.set_xlabel("dimension d")
        ax.set_xticks(config.DIMS[name])
    axes[0].set_ylabel("memory to match the PINN's energy error (bytes)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.text(0.01, -0.03, "Filled: measured d-dimensional solve. Hollow: oscillator from the exact "
             "1D eigenproblem; coupled problem estimated from the N\nthat worked in a lower "
             "dimension. Target = the PINN's median relative energy error in that dimension.",
             fontsize=8.5, color=style.INK2, ha="left", va="top")
    credit(fig)
    fig.savefig(config.FIGURES / f"fig3_memory_to_match{q}.png")
    plt.close(fig)


def fig4_error_vs_time(e2, e4, pinn, s, q):
    """Energy error against CPU time for every method, two dimensions per problem."""
    panels = {"oscillator": (5, 8), "coupled": (5, 7)}
    e5 = s.get("dmc_at_pinn_budget", [])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4), sharey=True)
    for i, name in enumerate(("oscillator", "coupled")):
        for j, d in enumerate(panels[name]):
            ax = axes[i, j]
            for m in ("fd2", "fd4", "sinc"):
                rows = [r for r in e2 if r["problem"] == name and r["d"] == d and r["method"] == m]
                if not rows:
                    continue
                # lower envelope over box sizes: best error reached for a given cost
                rows.sort(key=lambda r: r["cpu_s"])
                xs, ys, best = [], [], np.inf
                for r in rows:
                    if r["rel_energy_error"] < best:
                        best = r["rel_energy_error"]
                        xs.append(max(r["cpu_s"], 1e-3))
                        ys.append(max(best, 1e-16))
                ax.loglog(xs, ys, "-o", ms=3, color=COL[m], label=NAME[m])
            pr = [r for r in pinn if r["problem"] == name and r["d"] == d]
            if pr:
                ax.scatter([r["cpu_s"] for r in pr], [r["rel_energy_error"] for r in pr],
                           marker="D", s=40, color=COL["pinn"], zorder=4, label="PINN (each seed)")
            dr = [r for r in e4["runs"] if r["problem"] == name and r["d"] == d]
            if dr:
                ax.errorbar([r["cpu_s"] for r in dr], [r["rel_energy_error"] for r in dr],
                            yerr=[r["rel_se"] for r in dr], fmt="s", ms=5, color=COL["dmc"],
                            capsize=2, label="Diffusion Monte Carlo (± 1 s.e.)")
                c = np.median([r["rel_se"] * np.sqrt(r["cpu_s"]) for r in dr])
                t = np.logspace(-1, 4.5, 50)
                ax.loglog(t, c / np.sqrt(t), ls=":", color=COL["dmc"], lw=1)
            for r in e5:
                if r["problem"] == name and r["d"] == d:
                    ax.errorbar([r["cpu_s"]], [r["rel_energy_error"]], yerr=[r["rel_se"]],
                                fmt="s", ms=7, mfc=style.SURFACE, color=COL["dmc"], capsize=2,
                                label="DMC given the PINN's CPU time")
            ax.set_title(f"{PROB[name].split(' (')[0]}, d = {d}")
            ax.set_ylim(1e-9, 1)
            ax.set_xlim(1e-3, 3e4)
            if i == 1:
                ax.set_xlabel("CPU time (s)")
            if j == 0:
                ax.set_ylabel("relative energy error")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.text(0.01, -0.02, "Grid curves: lowest error reached for a given CPU time over the "
             "points per axis and box sizes tried. Dotted line: Monte Carlo standard error "
             "falling as cpu^(-1/2).", fontsize=8.5, color=style.INK2, ha="left", va="top")
    credit(fig)
    fig.savefig(config.FIGURES / f"fig4_error_vs_cpu{q}.png")
    plt.close(fig)


def fig5_pinn_by_dimension(s, q):
    """PINN energy error by dimension, Adam only vs Adam + L-BFGS, and DMC at equal CPU."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, name in zip(axes, ("oscillator", "coupled")):
        rows = [r for r in s["rows"] if r["problem"] == name]
        ok = [r for r in rows if not r["pinn_failed"]]
        ds = [r["d"] for r in rows]
        for r in rows:
            ax.scatter([r["d"]] * len(r["pinn"]["rel_energy_error"]),
                       r["pinn"]["rel_energy_error"], color=COL["pinn"], s=14, alpha=0.6)
        ax.semilogy(ds, [r["pinn"]["rel_energy_error_median"] for r in rows], "-o",
                    color=COL["pinn"], label="PINN, Adam then L-BFGS")
        ax.semilogy(ds, [r["pinn"]["adam_only_rel_energy_error_median"] for r in rows], "--o",
                    color=COL["pinn"], alpha=0.55, label="PINN, Adam only")
        dmc = [(r["d"], r["dmc"]["c"] / np.sqrt(r["pinn"]["cpu_s_median"])) for r in ok
               if r["dmc"]]
        if dmc:
            ax.semilogy(*zip(*dmc), "-s", color=COL["dmc"],
                        label="DMC standard error at the PINN's CPU time")
        ax.set_title(PROB[name])
        ax.set_xlabel("dimension d")
        ax.set_xticks(ds)
    axes[0].set_ylabel("relative energy error")
    axes[0].legend(loc="upper left", fontsize=8.5)
    credit(fig)
    fig.savefig(config.FIGURES / f"fig5_pinn_by_dimension{q}.png")
    plt.close(fig)


def main(quick):
    q = config.quick_suffix(quick)
    e1, e2, e4, s, pinn = load(quick)
    fig1_problems(q)
    fig2_convergence(e1, e2, q)
    fig3_memory(s, pinn, q)
    fig4_error_vs_time(e2, e4, pinn, s, q)
    fig5_pinn_by_dimension(s, q)
    print("figures written to", config.FIGURES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
