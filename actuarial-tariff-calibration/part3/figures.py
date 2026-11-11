"""Figures for Part 3 of the actuarial-tariff-calibration series. Run after audit_numbers.py (reads figure_data.json).

fig1_claim_amounts.png      The 26,444 claim amounts: the fixed IRSA-IDA amounts and the tail
fig2_segment_balance.png    Premium over claims by bonus-malus band and driver age band, three models, 95% intervals
fig3_premium_deciles.png    Premium over claims by premium decile (auto-calibration), three models
fig4_gini_by_target.png     Bootstrap distribution of the Gini of each model on counts, capped losses and total losses
fig5_repeated_splits.png    Ten refitted splits: Gini against deviance on counts, and the Gini on total losses

Run:  python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
COL = {"GLM-A": ORANGE, "GLM-B": VIOLET, "GBM": BLUE}
LAB = {"GLM-A": "GLM-A (bonus-malus linear)", "GLM-B": "GLM-B (bonus-malus in bands)", "GBM": "Poisson boosting"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})
CREDIT = "Figure: Diogo Ribeiro. Data: freMTPL2 (CASdatasets). Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/actuarial-tariff-calibration"


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def figure_claim_amounts() -> None:
    a = np.array(DATA["amounts"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.25, "width_ratios": [1.3, 1]})
    ax = axes[0]
    bins = np.logspace(0, np.log10(a.max() * 1.1), 120)
    heights, _, _ = ax.hist(a, bins=bins, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("claim amount, euros (log scale)")
    ax.set_ylabel("number of claims")
    ax.set_title("26,444 claim amounts")
    vals, cnts = np.unique(np.round(a, 2), return_counts=True)
    top = np.argsort(-cnts, kind="stable")[:3]
    ax.annotate(f"three fixed amounts (IRSA-IDA convention):\n{', '.join(f'{vals[i]:,.0f}' for i in sorted(top))} euros,\n"
                f"{100 * cnts[top].sum() / a.size:.1f}% of all claims", (1250, heights.max() * 0.97), xytext=(4000, heights.max() * 0.72),
                color=INK2, fontsize=9, ha="left", va="center", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    s = np.sort(a)[::-1]
    share = np.cumsum(s) / s.sum()
    k = np.arange(1, s.size + 1)
    ax.plot(k, share, color=BLUE, lw=2)
    ax.set_xscale("log")
    for n_, lab in ((1, "largest claim"), (41, "41 claims above 100,000")):
        ax.plot(n_, share[n_ - 1], "o", color=ORANGE, ms=7, mec="white", mew=1.2)
    ax.text(1.4, share[0] - 0.03, f"largest claim: {100 * share[0]:.1f}% of payments", color=INK2, fontsize=9, va="top")
    ax.text(41 * 1.4, share[40] - 0.03, f"41 claims above 100,000:\n{100 * share[40]:.1f}% of payments", color=INK2,
            fontsize=9, va="top", ha="left")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of largest claims (log scale)")
    ax.set_ylabel("share of all payments")
    ax.set_title("How concentrated the payments are")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig1_claim_amounts.png")
    plt.close(fig)


def figure_segments() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, var, title in zip(axes, ("bonus-malus", "driver age"), ("By bonus-malus band", "By driver age band")):
        ax.axhline(1, color=MUTED, lw=1)
        for j, k in enumerate(("GLM-A", "GLM-B", "GBM")):
            segs = DATA["segments"][k][var]
            x = np.arange(len(segs)) + (j - 1) * 0.22
            r = np.array([s["ratio"] for s in segs])
            lo = np.array([s["lo"] for s in segs])
            hi = np.array([s["hi"] for s in segs])
            ax.vlines(x, lo, hi, color=COL[k], lw=1.6, alpha=0.8)
            ax.plot(x, r, "o", color=COL[k], ms=6, mec="white", mew=1, label=LAB[k])
        ax.set_xticks(np.arange(len(segs)), [s["segment"] for s in segs])
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1, 1.4, 2, 3], ["0.5", "0.7", "1", "1.4", "2", "3"])
        ax.minorticks_off()
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("premium / claims, relative to the model's level (log scale)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Test sample: what each tariff charges a segment for each euro of its claims (95% Poisson intervals)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.12)
    credit(fig)
    fig.savefig(OUT / "fig2_segment_balance.png")
    plt.close(fig)


def figure_deciles() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.axhline(1, color=MUTED, lw=1)
    x = np.arange(1, 11)
    for k in ("GLM-A", "GLM-B", "GBM"):
        v = np.array(DATA["deciles"][k])
        ax.plot(x, v, "o-", color=COL[k], lw=2, ms=6, mec="white", mew=1, label=LAB[k])
    ax.set_xticks(x)
    ax.set_xlabel("premium decile (equal exposure), cheapest first")
    ax.set_ylabel("premium / claims, relative to the model's level")
    ax.set_title("Test sample: balance by premium level (auto-calibration)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig3_premium_deciles.png")
    plt.close(fig)


def figure_targets() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    targets = [("N", "claim counts"), ("Lc", "losses, each claim\ncapped at 100,000"), ("L", "total losses")]
    for i, (t, lab) in enumerate(targets):
        for j, k in enumerate(("GLM-A", "GBM")):
            v = np.array(DATA["boot_gini"][f"{k}|{t}"])
            pos = i + (j - 0.5) * 0.3
            parts = ax.violinplot(v, positions=[pos], widths=0.26, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(COL[k])
                pc.set_edgecolor("none")
                pc.set_alpha(0.45)
            ax.plot(pos, DATA["point_gini"][f"{k}|{t}"], "o", color=COL[k], ms=7, mec="white", mew=1.2,
                    label=LAB[k] if i == 0 else None)
    ax.set_xticks(range(3), [t[1] for t in targets])
    ax.set_ylabel("Gini index of the frequency model's ranking")
    ax.set_title("Test sample: the same two rankings scored on three targets (500 bootstrap resamples)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig4_gini_by_target.png")
    plt.close(fig)


def figure_splits() -> None:
    sp = DATA["splits"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    for k in ("GLM-A", "GLM-B", "GBM"):
        g = [s[k]["gini"] for s in sp]
        dv = [s[k]["D"] for s in sp]
        ax.plot(g, dv, "o", color=COL[k], ms=7, mec="white", mew=1, label=LAB[k])
    for s in sp:
        ax.plot([s[k]["gini"] for k in ("GLM-A", "GLM-B", "GBM")], [s[k]["D"] for k in ("GLM-A", "GLM-B", "GBM")],
                color=GRID, lw=1, zorder=0)
    ax.set_xlabel("Gini on claim counts")
    ax.set_ylabel("Poisson deviance per 1,000 policy-years")
    ax.set_title("Ten splits: the models agree with each other")
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    x = np.arange(1, len(sp) + 1)
    for k in ("GLM-A", "GBM"):
        ax.plot(x, [s[k]["gini_L"] for s in sp], "o-", color=COL[k], lw=1.6, ms=6, mec="white", mew=1, label=LAB[k] + ", total losses")
        ax.plot(x, [s[k]["gini"] for s in sp], "s--", color=COL[k], lw=1, ms=4, alpha=0.7, label=LAB[k] + ", counts")
    ax.set_xticks(x)
    ax.set_xlabel("split")
    ax.set_ylabel("Gini index")
    ax.set_title("Ten splits: the Gini on total losses does not")
    ax.legend(loc="upper left", fontsize=8.5, ncol=1)
    ax.set_ylim(0.2, 0.62)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig5_repeated_splits.png")
    plt.close(fig)


def main() -> None:
    figure_claim_amounts()
    figure_segments()
    figure_deciles()
    figure_targets()
    figure_splits()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
