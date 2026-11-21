"""Figures for the bid-rigging-screens article; reads the data and results/ written by screens.py.

    python figures.py     # figures/fig1_*.png ... fig5_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import screens as S

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SOURCE = ("Data: Imhof and Wallimann (2021), Mendeley Data doi:10.17632/x8jwmbcpgn.1 (CC BY 4.0). "
          "Analysis: Diogo Ribeiro.")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
})


def finish(fig, name: str, note: str = "") -> None:
    fig.text(0.01, 0.01, (note + "  " if note else "") + SOURCE, fontsize=7.5, color=INK2, ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1(data) -> None:
    """Median CV by period in the three data sets, with the Swiss 80% threshold."""
    thr = float(pd.read_csv(RES / "thresholds.csv").rule.iloc[0].split("<= ")[1].split(" ")[0])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharex=True)
    bins = np.logspace(np.log10(0.05), np.log10(200), 46)
    for ax, k in zip(axes, ("CH", "JP", "IT")):
        d = data[k]
        for y, colr, lab in ((0, BLUE, "competitive"), (1, ORANGE, "cartel")):
            v = d[d.y == y].median3cvbid.clip(0.05, 199)
            w = np.ones(len(v)) / len(v)
            ax.hist(v, bins=bins, weights=w, histtype="stepfilled", alpha=0.28, color=colr, linewidth=0)
            ax.hist(v, bins=bins, weights=w, histtype="step", color=colr, linewidth=1.8, label=lab)
        ax.axvline(thr, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_title(f"{S.NAMES[k]} ({100 * d.y.mean():.0f}% cartel)")
        ax.set_xlabel("median coefficient of variation of the three bids (%)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("share of coalitions in the period")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].annotate(f"Swiss rule: {thr:.2f}", xy=(thr, axes[0].get_ylim()[1] * 0.92), xytext=(6, 0),
                     textcoords="offset points", fontsize=8.5, color=INK)
    fig.suptitle("Cartel coalitions bid closer together in all three data sets, but on different scales",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    finish(fig, "fig1_median_cv_by_period", "Dashed line: the median CV that flags 80% of Swiss cartel coalitions.")


def fig2() -> None:
    uf = pd.read_csv(RES / "unseen_firms.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, k in zip(axes, ("JP", "CH")):
        u = uf[uf.country == k]
        xs, rf, cv, lo, hi = [0], [u.rf_random_same_size.mean()], [np.nan], [u.rf_random_same_size.quantile(.1)], [u.rf_random_same_size.quantile(.9)]
        cvlo, cvhi = [np.nan], [np.nan]
        for j in (1, 2, 3):
            c = f"rf_k{j}"
            if c in u and u[c].notna().sum() >= 10:
                v = u.dropna(subset=[c])
                xs.append(j); rf.append(v[c].mean()); cv.append(v[f"cv_k{j}"].mean())
                lo.append(v[c].quantile(.1)); hi.append(v[c].quantile(.9))
                cvlo.append(v[f"cv_k{j}"].quantile(.1)); cvhi.append(v[f"cv_k{j}"].quantile(.9))
        xs = np.array(xs)
        ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(xs, rf, "-o", color=BLUE, markersize=7, label="random forest, 36 screens")
        ax.fill_between(xs[1:], cvlo[1:], cvhi[1:], color=ORANGE, alpha=0.15, linewidth=0)
        ax.plot(xs[1:], cv[1:], "-o", color=ORANGE, markersize=7, label="median CV alone, no training")
        ax.set_xticks(xs, ["random split\n(same size)"] + [f"{j} unseen\nfirm{'s' if j > 1 else ''}" for j in xs[1:]])
        ax.set_ylim(0.80, 1.005)
        ax.set_title(S.NAMES[k] + (" (17-firm market with identifiers)" if k == "CH" else " (all coalitions)"))
        ax.set_ylabel("AUC on held-out coalitions")
        ax.text(xs[-1] - 0.08, rf[-1] - 0.012, f"{rf[-1]:.3f}", color=BLUE, ha="right", fontsize=9)
        ax.text(xs[-1] - 0.08, cv[-1] - 0.014, f"{cv[-1]:.3f}", color=ORANGE, ha="right", fontsize=9)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Inside one market the forest keeps most of its edge; it fades as the coalition's firms become new",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    finish(fig, "fig2_unseen_firms", "30% of firms held out in each of 50 draws; bands span the 10th to 90th percentile of the draws.")


def fig3() -> None:
    loo = pd.read_csv(RES / "transfer_leave_one_out.csv")
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    order = ["CH", "JP", "IT"]
    series = [("rf_raw", "random forest, 32 screens", BLUE),
              ("logit_ranks_four_medians", "logit on within-country ranks of four medians", AQUA),
              ("median_cv_no_training", "median CV alone, no training", ORANGE)]
    for i, k in enumerate(order):
        r = loo[loo.test == k].iloc[0]
        vals = [r[c] for c, _, _ in series]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, linewidth=6, solid_capstyle="round", zorder=1)
        for c, lab, colr in series:
            ax.scatter(r[c], i, s=90, color=colr, edgecolor=SURFACE, linewidth=2, zorder=3, label=lab if i == 0 else None)
        ax.text(r["rf_raw"], i + 0.27, f"{r['rf_raw']:.2f}", ha="center", fontsize=8.5, color=BLUE)
        a, b = r["logit_ranks_four_medians"], r["median_cv_no_training"]
        ax.text(min(a, b) - 0.004, i + 0.05, f"{a:.2f} logit | {b:.2f} CV", ha="right", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(3), [f"{S.NAMES[k]} held out" for k in order])
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlim(0.5, 0.95)
    ax.axvline(0.5, color=INK2, linewidth=0.8)
    ax.set_xlabel("AUC in the held-out country (model trained on the other two)")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8.8)
    ax.set_title("In a country it has not seen, the forest falls below a screen that needs no training",
                 loc="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    finish(fig, "fig3_transfer_between_countries")


def fig4() -> None:
    il = pd.read_csv(RES / "investigation_lists.csv")
    il = il[(il.top_share == 0.10) & (il.country != "CH")]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(2)
    w = 0.34
    for j, (scorer, colr) in enumerate((("median CV", ORANGE), ("forest, other countries", BLUE))):
        v = [il[(il.country == k) & (il.scorer == scorer)].precision.iloc[0] for k in ("JP", "IT")]
        bars = ax.bar(x + (j - 0.5) * (w + 0.02), v, width=w, color=colr, label=scorer, edgecolor=SURFACE, linewidth=2)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{100 * val:.0f}%", ha="center", fontsize=9, color=INK)
    for i, k in enumerate(("JP", "IT")):
        br = il[il.country == k].base_rate.iloc[0]
        ax.plot([i - 0.42, i + 0.42], [br, br], color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(i + 0.43, br, f"cartel share {100 * br:.0f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [S.NAMES[k] for k in ("JP", "IT")])
    ax.set_xlim(-0.6, 1.95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("cartel coalitions among those flagged")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Precision of the 10% of coalitions ranked most suspicious", loc="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    finish(fig, "fig4_investigation_lists", "Forest trained on the other two countries, on within-country ranks.")


def fig5(data) -> None:
    d = data["CH"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=False)
    rng = np.random.default_rng(0)
    for ax, col, title in ((axes[0], "min3cvbid", "Minimum CV across the shared tenders"),
                           (axes[1], "median3cvbid", "Median CV across the shared tenders")):
        for y, colr, lab in ((0, BLUE, "competitive"), (1, ORANGE, "cartel")):
            g = d[d.y == y]
            jit = np.exp(rng.uniform(-0.06, 0.06, len(g)))
            ax.scatter(g.n_tenders * jit, g[col].clip(lower=0.02), s=10, alpha=0.45, color=colr, linewidth=0, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tenders in which the three firms bid together")
        ax.set_title(title)
    axes[0].set_ylabel("coefficient of variation of the three bids (%)")
    axes[0].legend(frameon=False, loc="lower left", markerscale=2)
    fig.suptitle("Switzerland: an extreme across tenders falls as tenders accumulate; a median does not",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    finish(fig, "fig5_tender_counts")


if __name__ == "__main__":
    data = S.load()
    fig1(data); fig2(); fig3(); fig4(); fig5(data)
    print("figures written:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
