"""Draw every figure from the files in results/ (no recomputation)."""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
import pandas as pd

from config import FIGURES, RESULTS
from eos import style, toy
from eos.style import AQUA, BLUE, INK, INK2, MUTED, ORANGE, SEQ, SERIES, YELLOW

style.apply()


def label_end(ax, x, y, text, color, dx=0, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK2,
                fontsize=10, ha=ha, va="center")


# ---------------------------------------------------------------- figure 1
def fig_toy_geometry():
    d = json.load(open(RESULTS / "e1_paths.json"))
    eta = d["eta"]
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15], hspace=0.55, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    uu, vv = np.meshgrid(np.linspace(0.0, 2.9, 400), np.linspace(-1.25, 2.9, 500))
    ax.contourf(uu, vv, np.log10(toy.loss(uu, vv) + 1e-4), levels=14, cmap="Greys", alpha=0.11)
    # level sets of the imbalance: gradient-flow paths lie on these
    for c in [-6, -4, -2, 0, 2, 4, 6]:
        ax.contour(uu, vv, uu**2 - vv**2, levels=[c], colors=[MUTED], linewidths=0.6, linestyles=":")
    # curve of minimisers, split into stable and unstable arcs for this eta
    uc = np.geomspace(0.3, 2.9, 600)
    vc = 1 / uc
    stable = uc**2 + vc**2 < 2 / eta
    ax.plot(np.where(stable, uc, np.nan), np.where(stable, vc, np.nan), color=AQUA, lw=5, solid_capstyle="round",
            label=f"minimisers GD can stop at (sharpness < 2/η = {2/eta:.0f})")
    ax.plot(np.where(~stable, uc, np.nan), np.where(~stable, vc, np.nan), color=ORANGE, lw=5, alpha=0.55,
            label="minimisers GD cannot stop at")
    starts = [("A", BLUE, "o"), ("B", INK2, "s")]
    for name, col, mk in starts:
        s = d["starts"][name]
        ax.plot(s["flow_u"], s["flow_v"], color=col, lw=1.6, ls="--")
        ax.plot(*s["flow_endpoint"], marker="x", ms=10, mew=2.2, color=col)
        u, v = np.array(s["gd_u"][:60]), np.array(s["gd_v"][:60])
        ax.plot(u, v, color=col, lw=0.7, alpha=0.9, marker=mk, ms=2.5 if name == "B" else 4)
        ax.plot(s["gd_u"][-1], s["gd_v"][-1], marker=mk, ms=11, color=col, mec="white", mew=1.5)
        off = (6, -12) if name == "A" else (6, -14)
        ax.annotate(f"start {name}", (s["u0"], s["v0"]), xytext=off, textcoords="offset points",
                    color=INK2, fontsize=10, ha="left")
    ax.set_xlim(0, 2.9)
    ax.set_ylim(-1.25, 2.9)
    ax.set_aspect("equal")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(f"Flow (dashed, ends at ×) vs GD, η = {eta}")
    ax.legend(loc="upper right", fontsize=9)

    n = 40
    bound = d["stable_bound"]
    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    a1.axhline(0, color=MUTED, lw=0.8)
    a2.axhspan(-bound, bound, color=AQUA, alpha=0.15, lw=0)
    a3.axhline(2 / eta, color=ORANGE, lw=1.2, ls="--")
    for name, col, mk in starts:
        s = d["starts"][name]
        t = np.arange(n)
        a1.plot(t, np.array(s["residual"])[:n], color=col, marker=mk, ms=3.5, lw=1, label=f"start {name}")
        a2.plot(t, np.array(s["imbalance"])[:n], color=col, marker=mk, ms=4, lw=0)
        a2.plot(t, np.array(s["imbalance_predicted"])[:n], color=col, lw=0.9)
        a3.plot(t, np.array(s["sharpness"])[:n], color=col, marker=mk, ms=3.5, lw=1)
    a1.set_title("Residual r = uv − 1", fontsize=11)
    a1.legend(loc="lower right", fontsize=9, ncol=2)
    a2.set_title("Imbalance D = u² − v²  (markers: measured, line: D₀ ∏(1 − η²rₖ²); band: stoppable)", fontsize=11)
    a3.annotate("2/η", (n - 1, 2 / eta), xytext=(0, 6), textcoords="offset points", color=INK2, ha="right", fontsize=10)
    a3.set_title("Top Hessian eigenvalue", fontsize=11)
    a3.set_xlabel("GD step")
    fig.savefig(FIGURES / "fig1_two_parameter_model.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_toy_landing():
    df = pd.read_csv(RESULTS / "e1_landing.csv")
    etas = sorted(df.eta.unique())
    fig, axes = plt.subplots(1, len(etas), figsize=(13, 4.4), sharey=True)
    for ax, eta in zip(axes, etas):
        g = df[df.eta == eta]
        c = 2 / eta
        conv = g[g.converged]
        ok = conv.flow_sharpness < c
        ax.plot([2, 40], [2, 40], color=MUTED, lw=0.8)
        ax.axhline(c, color=ORANGE, ls="--", lw=1.2)
        ax.axvline(c, color=ORANGE, ls=":", lw=0.8)
        ax.scatter(conv.flow_sharpness[ok], conv.gd_sharpness[ok], s=8, color=BLUE, alpha=0.45, lw=0,
                   label="flow endpoint allowed")
        ax.scatter(conv.flow_sharpness[~ok], conv.gd_sharpness[~ok], s=8, color=ORANGE, alpha=0.6, lw=0,
                   label="flow endpoint forbidden")
        forb = g[g.flow_sharpness >= c]
        ax.text(0.97, 0.97, f"forbidden starts that diverged: {forb.diverged.mean():.0%}\n"
                            f"converged runs ending above 2/η: {(conv.gd_sharpness > c).sum()}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)
        ax.annotate("2/η", (2.1, c), xytext=(0, 5), textcoords="offset points", color=INK2, fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(2, 30)
        ax.set_ylim(1.9, 12)
        ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30]))
        ax.yaxis.set_major_locator(FixedLocator([2, 3, 4, 6, 10]))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_locator(NullLocator())
        ax.set_title(f"η = {eta}")
        ax.set_xlabel("sharpness where gradient flow stops")
    axes[0].set_ylabel("sharpness where GD stops")
    axes[0].legend(loc="upper right", fontsize=9, markerscale=2, bbox_to_anchor=(1.0, 0.82))
    fig.savefig(FIGURES / "fig2_where_gd_lands.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_bifurcation():
    b = np.load(RESULTS / "e2_bifurcation.npz")
    summ = json.load(open(RESULTS / "e2_summary.json"))
    etas, res, lyap = b["etas"], b["residual"], b["lyapunov"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ee = np.repeat(etas, res.shape[1])
    a1.scatter(ee, res.ravel(), s=0.15, color=BLUE, alpha=0.35, lw=0, rasterized=True)
    marks = [(1.0, "η = 1\nminimum loses\nstability"), (np.sqrt(5) - 1, "η = √5 − 1\nperiod 4"),
             (summ["eta_infinity_estimate"], "η∞ ≈ 1.3023\nchaos"), (2.0, "η = 2\ncrisis:\ndivergence")]
    for i, (x, txt) in enumerate(marks):
        for ax in (a1, a2):
            ax.axvline(x, color=ORANGE, lw=0.9, ls="--")
        if i == 2:
            a1.annotate(txt, (x, -0.8), xytext=(-4, 0), textcoords="offset points", fontsize=9, color=INK2,
                        va="top", ha="right")
        else:
            a1.annotate(txt, (x, 1.0), xytext=(4, 0), textcoords="offset points", fontsize=9, color=INK2, va="top")
    a1.set_ylim(-1.05, 1.05)
    a1.set_ylabel("residual uv − 1 on the attractor")
    a1.set_title("Gradient descent on the balanced model beyond the edge: 2-cycle, period doubling, chaos")
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.plot(etas, lyap, color=INK2, lw=0.7)
    a2.set_ylim(-2.0, 1.2)
    a2.set_ylabel("Lyapunov exponent")
    a2.set_xlabel("step size η")
    a2.set_xlim(0.9, 2.05)
    fig.savefig(FIGURES / "fig3_period_doubling.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_network_traces():
    z = np.load(RESULTS / "e3_traces.npz")
    etas = list(z["etas"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for eta, col in zip(etas, SERIES):
        st, sh = z[f"steps_{eta}"], z[f"sharp_{eta}"]
        a1.plot(st, sh, color=col, lw=1.8)
        a1.axhline(2 / eta, color=col, lw=1.0, ls="--")
        label_end(a1, st[-1], sh[-1], f"η = {eta}", col, dx=6, dy={0.05: 7, 0.15: -8}.get(float(eta), 0))
        loss = z[f"loss_{eta}"]
        a2.plot(np.arange(len(loss)), loss, color=col, lw=0.9)
        label_end(a2, len(loss) - 1, loss[-1], f"η = {eta}", col, dx=6)
    a1.set_ylim(0, 25)
    a1.set_xlabel("GD step")
    a1.set_title("Top Hessian eigenvalue (dashed: 2/η)")
    a1.set_xlim(0, 9400)
    a2.set_yscale("log")
    a2.set_xlabel("GD step")
    a2.set_title("Training loss (MSE)")
    a2.set_xlim(0, 9400)
    fig.savefig(FIGURES / "fig4_network_edge_of_stability.png")
    plt.close(fig)

    # zoom on the loss for the run that sharpens into its ceiling
    eta = 0.15
    loss = z[f"loss_{eta}"]
    st, sh = z[f"steps_{eta}"], z[f"sharp_{eta}"]
    fig, (b1, b2) = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True)
    lo, hi = 2000, 5000
    b1.plot(np.arange(lo, hi), loss[lo:hi], color=AQUA, lw=0.9)
    b1.set_yscale("log")
    b1.set_title(f"η = {eta}: training loss, steps {lo}–{hi}")
    m = (st >= lo) & (st < hi)
    b2.axhline(2 / eta, color=ORANGE, ls="--", lw=1.0)
    b2.plot(st[m], sh[m], color=AQUA, marker="o", ms=3, lw=1)
    b2.annotate("2/η", (lo, 2 / eta), xytext=(4, -14), textcoords="offset points", color=INK2, ha="left")
    b2.set_title("Top Hessian eigenvalue (every 50 steps)")
    b2.set_xlabel("GD step")
    fig.savefig(FIGURES / "fig5_loss_spikes_zoom.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def fig_sweep():
    df = pd.read_csv(RESULTS / "e4_sweep.csv")
    ok = df[~df.diverged]
    etas = sorted(ok.eta.unique())
    cmap = {e: plt.get_cmap("Blues")(0.35 + 0.6 * i / max(1, len(etas) - 1)) for i, e in enumerate(etas)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    lim = [3.5, 75]
    a1.plot(lim, lim, color=ORANGE, lw=1.2, ls="--")
    a1.annotate("sharpness = 2/η", (60, 60), xytext=(-6, -18), textcoords="offset points", ha="right",
                color=INK2, fontsize=10)
    for e in etas:
        g = ok[ok.eta == e]
        a1.scatter(g.ceiling, g.plateau_sharpness, s=36, color=cmap[e], edgecolor="white", lw=0.6, zorder=3)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(*lim)
    a1.set_ylim(3.5, 30)
    for axis in (a1.xaxis, a1.yaxis):
        axis.set_major_locator(FixedLocator([4, 5, 7, 10, 15, 20, 30, 40, 70]))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
    a1.set_xlabel("ceiling 2/η")
    a1.set_ylabel("sharpness, median of last 1000 steps")
    a1.set_title("Where the ceiling binds, sharpness is 2/η")
    for e in etas:
        g = ok[ok.eta == e]
        a2.scatter(g.plateau_sharpness, g.test_mse, s=36, color=cmap[e], edgecolor="white", lw=0.6, zorder=3)
    for e in [etas[0], etas[-1]]:
        g = ok[ok.eta == e]
        a2.annotate(f"η = {e}", (g.plateau_sharpness.median(), g.test_mse.max()), xytext=(0, 10),
                    textcoords="offset points", ha="center", color=INK2, fontsize=10)
    a2.set_xscale("log")
    a2.xaxis.set_major_locator(FixedLocator([4, 5, 7, 10, 15, 20]))
    a2.xaxis.set_major_formatter(ScalarFormatter())
    a2.xaxis.set_minor_locator(NullLocator())
    a2.set_xlabel("sharpness, median of last 1000 steps")
    a2.set_ylabel("test MSE")
    a2.set_title("Test MSE vs sharpness (light: η = 0.03, dark: η = 0.5)")
    from scipy.stats import pearsonr, spearmanr
    rho = spearmanr(ok.plateau_sharpness, ok.test_mse).statistic
    dm = ok.assign(s=ok.plateau_sharpness - ok.groupby("eta").plateau_sharpness.transform("mean"),
                   m=ok.test_mse - ok.groupby("eta").test_mse.transform("mean"))
    r_within = pearsonr(dm.s, dm.m).statistic
    a2.text(0.97, 0.04, f"pooled over η: Spearman ρ = {rho:.2f}\nwithin η (seeds only): Pearson r = {r_within:+.2f}",
            transform=a2.transAxes, ha="right", va="bottom", fontsize=10, color=INK2)
    fig.savefig(FIGURES / "fig6_step_size_sweep.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 7
def fig_rescaling():
    full = pd.read_csv(RESULTS / "e5_rescaling.csv")
    div = full[full.diverged]
    df = full[~full.diverged]
    eta = 0.05
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4))
    a1.axhline(2 / eta, color=ORANGE, ls="--", lw=1.2)
    a1.plot(full.alpha, full.sharpness_start, color=BLUE, lw=1.4)
    a1.plot(df.alpha, df.sharpness_start, color=BLUE, marker="o", ms=4, lw=0)
    a1.plot(div.alpha, div.sharpness_start, color=INK, marker="x", ms=7, mew=1.8, lw=0, label="GD diverged")
    a1.annotate("2/η", (full.alpha.min(), 2 / eta), xytext=(2, 5), textcoords="offset points", color=INK2)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("rescaling factor α")
    a1.set_title("Sharpness")
    a1.legend(loc="upper center", fontsize=9)
    a2.plot(full.alpha, full.adaptive_start, color=AQUA, marker="o", ms=4, lw=1.4)
    a2.set_xscale("log")
    spread = full.adaptive_start.max() / full.adaptive_start.min() - 1
    a2.set_ylim(0, full.adaptive_start.max() * 1.6)
    a2.text(0.5, 0.85, f"max/min − 1 = {spread:.1e}", transform=a2.transAxes, ha="center", color=INK2, fontsize=10)
    a2.set_xlabel("rescaling factor α")
    a2.set_title("Adaptive sharpness")
    ratio = df.loss_max / df.loss_start
    stable = df.sharpness_start < 2 / eta
    a3.axvline(2 / eta, color=ORANGE, ls="--", lw=1.2)
    a3.scatter(df.sharpness_start[stable], ratio[stable], color=BLUE, s=28, label="below 2/η", zorder=3)
    a3.scatter(df.sharpness_start[~stable], ratio[~stable], color=ORANGE, s=28, marker="s", label="above 2/η", zorder=3)
    a3.set_xscale("log")
    a3.set_yscale("log")
    a3.set_xlabel("sharpness after rescaling")
    a3.set_title("Worst loss ÷ starting loss, next 2000 steps")
    a3.legend(loc="upper left")
    for ax in (a1, a2, a3):
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    a1.yaxis.set_major_locator(FixedLocator([10, 15, 20, 30, 40, 60, 100, 200]))
    a3.xaxis.set_major_locator(FixedLocator([10, 15, 20, 30, 40, 60, 100, 200]))
    for axis in (a1.yaxis, a3.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(plt.NullFormatter())
    a3.xaxis.set_minor_locator(NullLocator())
    a1.yaxis.set_minor_locator(NullLocator())
    fig.savefig(FIGURES / "fig7_rescaling.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 8
def fig_imbalance():
    z = np.load(RESULTS / "e5_tracking.npz")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    n = 40
    for eta, col, mk in [(0.05, BLUE, "o"), (0.25, ORANGE, "s")]:
        q = z[f"Q0_{eta}"]
        loss = z[f"loss_{eta}"]
        a1.plot(np.arange(n), loss[:n], color=col, lw=1.2, marker=mk, ms=3.5, label=f"η = {eta}")
        cum = np.concatenate([[0], np.cumsum(z[f"inc_{eta}"][:n])])
        a2.plot(np.arange(n + 1), q[: n + 1] - q[0], color=col, lw=1.8, label=f"η = {eta}: measured")
        a2.plot(np.arange(n + 1), cum, color=INK, lw=0, marker=mk, ms=3, mfc="none",
                label=f"η = {eta}: Σ η²(‖g₀‖² − ‖g₁‖²)")
        tot = q[-1] - q[0]
        a2.annotate(f"after {len(q) - 1} steps: {tot:+.4f}", (n, q[n] - q[0]), xytext=(-4, 10 if eta == 0.25 else -14),
                    textcoords="offset points", ha="right", color=INK2, fontsize=10)
    a1.set_yscale("log")
    a1.set_title("Training loss, first 40 steps (ReLU network)")
    a1.set_xlabel("GD step")
    a1.legend(loc="upper right")
    a2.set_title("Change in layer imbalance Q₀")
    a2.set_xlabel("GD step")
    a2.legend(loc="center right", fontsize=9)
    fig.savefig(FIGURES / "fig8_layer_imbalance.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 9
def fig_momentum():
    z = np.load(RESULTS / "e6_momentum.npz")
    beta = float(z["beta"])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for eta, col in zip(z["etas"], [BLUE, ORANGE, AQUA]):
        st, sh = z[f"steps_{eta}"], z[f"sharp_{eta}"]
        if len(sh) < 3:
            continue
        ax.plot(st, sh, color=col, lw=1.8)
        ax.axhline((2 + 2 * beta) / eta, color=col, ls="--", lw=1.0)
        ax.axhline(2 / eta, color=col, ls=":", lw=1.0)
        ax.annotate(f"η = {eta}", (st[-1], sh[-1]), xytext=(6, 8 if eta == 0.3 else 0), textcoords="offset points",
                    color=INK2, va="center")
    ax.set_ylabel("top Hessian eigenvalue")
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 28)
    ax.set_xlabel("step")
    ax.set_title(f"Heavy-ball momentum β = {beta}: dashed (2 + 2β)/η, dotted 2/η")
    fig.savefig(FIGURES / "fig9_momentum.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    which = sys.argv[1:] or ["1", "2", "3", "4", "6", "7", "8", "9"]
    table = {"1": fig_toy_geometry, "2": fig_toy_landing, "3": fig_bifurcation, "4": fig_network_traces,
             "6": fig_sweep, "7": fig_rescaling, "8": fig_imbalance, "9": fig_momentum}
    for k in which:
        if k in table:
            table[k]()
            print("figure", k, "done")
