"""Figures for Part 2 of the actuarial-tariff-calibration series. Run after mechanism_numbers.py (reads figure_data.json).

fig1_two_curves.png        The concentration curve of expected claims and the Lorenz curve of the premium for the true
                           frequency raised to the powers 0.5, 1 and 2 (population pool)
fig2_convex_minorant.png   The ordered Lorenz curve of the claim classifier on 3,000 test policies with its greatest
                           convex minorant, and the isotonic recalibration of three models on the full test sample
fig3_mcb_dsc.png           Miscalibration against discrimination in the population, before and after recalibration,
                           with lines of equal expected deviance
fig4_model_choice.png      Paired Gini and deviance differences over 400 test samples for two pairs of models

Run:  python figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tariffcal import lorenz, lower_convex_hull  # noqa: E402
from mechanism_numbers import N_POOL, RNG_POOL, rebalance, simulate_policies, true_frequency  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DATA = json.loads((HERE / "figure_data.json").read_text())

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#7a5cd6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})
CREDIT = "Figure: Diogo Ribeiro. Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/actuarial-tariff-calibration"


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def figure_two_curves() -> None:
    Xp, ep = simulate_policies(N_POOL, RNG_POOL)
    mu = true_frequency(Xp)
    total = (ep * mu).sum()
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    cw, cc = lorenz(mu, ep, ep * mu)
    ax.plot(cw, cc, color=INK, lw=2.6)
    ax.annotate("expected claims, ordered by any of the three\npremiums: one concentration curve", (0.80, 0.62),
                xytext=(0.30, 0.78), color=INK2, fontsize=9, arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8})
    cols = {0.5: AQUA, 2.0: ORANGE}
    curves = {}
    for g in (0.5, 2.0):
        r = rebalance(mu ** g, ep, total)
        curves[g] = lorenz(r, ep, ep * r)
        ax.plot(*curves[g], color=cols[g], lw=2)
    xa = 0.40
    ax.annotate("Lorenz curve of the premium truth^0.5:\nless spread than the claims", (xa, np.interp(xa, *curves[0.5])),
                xytext=(0.05, 0.55), color=INK2, fontsize=9, arrowprops={"arrowstyle": "-", "color": AQUA, "lw": 1})
    xb = 0.70
    ax.annotate("Lorenz curve of the premium truth^2:\nmore spread than the claims", (xb, np.interp(xb, *curves[2.0])),
                xytext=(0.45, 0.05), color=INK2, fontsize=9, arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1})
    ax.text(0.03, 0.93, "The premium of the true frequency (power 1) has a Lorenz curve\nidentical to the concentration curve: "
            "auto-calibration.", color=INK2, fontsize=9, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("cumulative share of policy-years, lowest premium first")
    ax.set_ylabel("cumulative share")
    ax.set_title("One ranking, three premiums (population)")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.11)
    credit(fig)
    fig.savefig(OUT / "fig1_two_curves.png")
    plt.close(fig)


def figure_convex_minorant() -> None:
    sm = DATA["small_sample"]
    r, e, N = (np.array(sm[k], dtype=float) for k in ("clf", "e", "N"))
    cw, cc = lorenz(r, e, N)
    hx, hy = lower_convex_hull(cw, cc)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.plot(cw, cc, color=BLUE, lw=1.4)
    ax.plot(hx, hy, color=ORANGE, lw=2)
    ax.plot(hx, hy, "o", color=ORANGE, ms=4, mec="white", mew=0.8)
    ax.text(0.05, 0.62, f"ordered Lorenz curve of the claim\nclassifier, {r.size:,} policies, {int(N.sum())} claims",
            color=INK2, fontsize=9)
    ax.annotate("its greatest convex minorant =\nthe curve of the isotonic recalibration", (hx[len(hx) // 2], hy[len(hx) // 2]),
                xytext=(0.45, 0.06), color=INK2, fontsize=9, arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("cumulative share of policy-years")
    ax.set_ylabel("cumulative share of claims")
    ax.set_title("Pool-adjacent-violators is a convex hull")

    ax = axes[1]
    lims = [0.02, 3.0]
    ax.plot(lims, lims, color=MUTED, lw=1, ls="--")
    names = {"gbm": ("Poisson boosting", BLUE), "clf": ("claim classifier", ORANGE),
             "nooffset": ("boosting without offset", VIOLET)}
    for k, (lab, col) in names.items():
        d = DATA["corp"][k]
        x, y = np.array(d["rate"]), np.array(d["recal"])
        ok = y > 0
        ax.step(x[ok], y[ok], where="post", color=col, lw=1.8, label=lab)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.025, 1.9, "above the diagonal: the model charges less\nthan the claims at that prediction", color=INK2,
            fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 3.0)
    ax.set_ylim(0.02, 3.0)
    ax.set_xlabel("predicted claims per policy-year (log scale)")
    ax.set_ylabel("observed claims per policy-year at that prediction (log)")
    ax.set_title("Isotonic recalibration on the 500,000-policy test sample")
    ax.grid(which="minor", visible=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.12)
    credit(fig)
    fig.savefig(OUT / "fig2_convex_minorant.png")
    plt.close(fig)


def figure_mcb_dsc() -> None:
    pop = DATA["population"]
    rec = DATA["recalibrated"]
    fam = DATA["family"]
    dsc_max = pop["truth"]["DSC"]
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    for ex in (1, 2, 3, 4, 6, 8):
        xs = np.linspace(0, 8, 50)
        ax.plot(xs, dsc_max - (ex - xs), color=GRID, lw=1.2, zorder=0)
        xl = ex - (dsc_max - 35.38)
        if xl < 7.8:
            ax.text(xl + 0.06, 35.38, f"excess {ex}", color=MUTED, fontsize=8, ha="left", va="center")
    pts = [("truth", "true frequency", INK, (8, -26)), ("glm", "GLM", BLUE, (-62, -6)),
           ("gbm", "boosting", BLUE, (8, 4)), ("clf", "classifier", ORANGE, (8, 4)),
           ("nooffset", "no offset", VIOLET, (8, 4)), ("truth^1.25", "truth^1.25", AQUA, (8, -26))]
    for k, lab, col, off in pts:
        p = pop[k]
        ax.plot(p["MCB"], p["DSC"], "o", color=col, ms=9, mec="white", mew=1.2, zorder=3)
        ax.annotate(f"{lab}\nGini {p['gini']:.4f}", (p["MCB"], p["DSC"]), xytext=off, textcoords="offset points",
                    color=INK2, fontsize=8.5)
        rk = k if k in rec else None
        if rk:
            q = rec[rk]
            ax.annotate("", xy=(q["MCB"], q["DSC"]), xytext=(p["MCB"], p["DSC"]),
                        arrowprops={"arrowstyle": "->", "color": col, "lw": 1.1, "alpha": 0.8})
            ax.plot(q["MCB"], q["DSC"], "s", color=col, ms=6, mec="white", mew=1, zorder=3)
    p8 = fam["0.8"]
    ax.plot(p8["MCB"], p8["DSC"], "o", color=AQUA, ms=9, mec="white", mew=1.2, zorder=3)
    ax.annotate(f"truth^0.8\nGini {p8['gini']:.4f}", (p8["MCB"], p8["DSC"]), xytext=(6, -22), textcoords="offset points",
                color=INK2, fontsize=8.5)
    ax.text(4.2, 32.45, "circles: as fitted; squares: after isotonic recalibration\n"
            "on a 200,000-policy sample (steps of at least 1,000 policy-years)", color=INK2, fontsize=8.5)
    ax.set_xlim(-0.8, 8.0)
    ax.set_ylim(32.3, 35.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("miscalibration MCB, per 1,000 policy-years")
    ax.set_ylabel("discrimination DSC, per 1,000 policy-years")
    ax.set_title("Population: deviance = uncertainty - discrimination + miscalibration")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.11)
    credit(fig)
    fig.savefig(OUT / "fig3_mcb_dsc.png")
    plt.close(fig)


def figure_model_choice() -> None:
    G, D = DATA["gini"], DATA["deviance"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=False, gridspec_kw={"wspace": 0.25})
    cases = [(("gbm", "truth^1.25"), "boosting minus truth^1.25", ORANGE),
             (("gbm", "glm"), "boosting minus GLM", BLUE)]
    for ax, ((a, b), title, col) in zip(axes, cases):
        dg = np.array(G[a]) - np.array(G[b])
        dd = np.array(D[a]) - np.array(D[b])
        ax.axvline(0, color=MUTED, lw=1)
        ax.axhline(0, color=MUTED, lw=1)
        ax.plot(dg, dd, "o", color=col, ms=4, alpha=0.55, mec="none")
        ax.set_xlabel("Gini difference (positive: the Gini prefers boosting)")
        ax.set_ylabel("deviance difference per 1,000 policy-years\n(negative: the deviance prefers boosting)")
        g_share = np.mean(dg > 0)
        d_share = np.mean(dd < 0)
        ax.set_title(f"{title}\nGini prefers boosting in {100 * g_share:.0f}% of samples, "
                     f"deviance in {100 * d_share:.0f}%", fontsize=10.5)
        m = max(abs(dg).max(), 1e-3) * 1.1
        ax.set_xlim(-m, m)
    fig.suptitle("400 test samples of 170,000 policies: which model each criterion chooses", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig4_model_choice.png")
    plt.close(fig)


def main() -> None:
    figure_two_curves()
    figure_convex_minorant()
    figure_mcb_dsc()
    figure_model_choice()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
