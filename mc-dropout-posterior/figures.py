"""Figures 3, 4 and 6 from the result tables of 02_concentration.py and
03_uci_benchmark.py (figures 1, 2 and 5 are drawn by their own scripts)."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcd import (AQUA, BLUE, FIGURES, INK, INK2, MAGENTA, ORANGE, RESULTS, UCI_LABEL,
                 UCI_SETS, YELLOW, style_axes)

DARK_ORANGE = "#b4441a"
FIGURES.mkdir(parents=True, exist_ok=True)


def fig3():
    path = RESULTS / "02_concentration.csv"
    if not path.exists():
        return
    from matplotlib.ticker import FuncFormatter, NullFormatter
    df = pd.read_csv(path)
    df = df[df.rep < 2]
    series = [("gp", "Gaussian-process posterior", BLUE),
              ("mc_0.5", "MC dropout, rate 0.5", DARK_ORANGE),
              ("mc_0.05", "MC dropout, rate 0.05", ORANGE),
              ("mc_tuned", "MC dropout, rate tuned", MAGENTA),
              ("bootstrap", "bootstrap of the fitted network", YELLOW),
              ("ensemble", "deep ensemble (5)", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="white",
                             gridspec_kw={"width_ratios": [1.25, 1]})
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for ax, col, ylab, title in ((axes[0], "sd", "epistemic sd at a test point (mean, RMSD units)",
                                  "A. Spread of the regression function"),
                                 (axes[1], "rmse", "test RMSE of the mean prediction",
                                  "B. Accuracy of the same models")):
        style_axes(ax)
        ends = []
        for key, label, colr in series:
            g = df[df.method == key].groupby("n")[col]
            m, lo, hi = g.mean(), g.min(), g.max()
            ns = np.array(m.index, float)
            ax.plot(ns, m.values, color=colr, lw=2, marker="o", ms=4)
            if col == "sd":
                ax.vlines(ns, lo.values, hi.values, color=colr, lw=1)
                ends.append([ns[-1], m.values[-1], label])
        if col == "sd":
            ends.sort(key=lambda e: e[1])
            ypos = []
            for x, yv, label in ends:  # nudge labels apart on the log scale
                yl = yv
                if ypos and np.log(yl) - np.log(ypos[-1]) < 0.11:
                    yl = ypos[-1] * np.exp(0.11)
                ypos.append(yl)
                ax.text(x * 1.15, yl, label, color=INK, va="center", fontsize=8.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("subset size n (80% fitted, 20% validation)", color=INK2)
        ax.set_ylabel(ylab, color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(plain)
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_yticks([0.4, 0.6, 1, 1.5, 2, 3, 4, 6])
    axes[0].set_xlim(350, df.n.max() * 9)
    axes[1].set_xlim(350, df.n.max() * 1.6)
    axes[1].text(0.98, 0.98, "colours as in panel A", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8, color=INK2)
    fig.text(0.01, 0.01, "UCI protein data, split 0; 4,573 test rows; mean of two replicate subsets per n "
             "(vertical bars in A: range). Every network has one hidden layer of 100 units; GP up to n = 16,000.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "fig3_concentration.png", dpi=170)


def load_uci_runs():
    parts = [pd.read_csv(RESULTS / f"03_splits_{n}.csv") for n in UCI_SETS
             if (RESULTS / f"03_splits_{n}.csv").exists()]
    return pd.concat(parts, ignore_index=True) if parts else None


def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


def fig4(df):
    sets = [s for s in UCI_SETS if s in set(df.dataset)]
    labels = [f"{UCI_LABEL[s]} ({(df.dataset == s).sum()})" for s in sets]
    y = np.arange(len(sets))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor="white", sharey=True)
    ax = axes[0]
    style_axes(ax)
    for yi, s in zip(y, sets):
        g = df[df.dataset == s]
        m, se = mean_se(g.share_mc)
        ax.errorbar(m, yi, xerr=1.96 * se, fmt="o", color=ORANGE, ms=6, capsize=0, lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the predictive variance from the dropout spread", color=INK2)
    ax.set_title("A. What the MC samples add to the noise term 1/tau", loc="left", fontsize=10.5, color=INK)

    ax = axes[1]
    style_axes(ax)
    ax.axvline(0, color=INK2, lw=0.8)
    for yi, s in zip(y, sets):
        g = df[df.dataset == s]
        m1, se1 = mean_se(g.ll_mc - g.ll_tau_only)
        m2, se2 = mean_se(g.ll_mc - g.ll_val_var)
        ax.errorbar(m1, yi + 0.12, xerr=1.96 * se1, fmt="o", color=ORANGE, ms=6, lw=1.5)
        ax.errorbar(m2, yi - 0.12, xerr=1.96 * se2, fmt="s", color=BLUE, ms=5, lw=1.5)
    ax.set_xlabel("test log-likelihood of MC dropout minus a Gaussian without the spread", color=INK2)
    ax.set_title("B. Does the spread improve the benchmark score?", loc="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.045, "Reference MC dropout protocol on the published UCI splits (number of splits in brackets); "
             "mean and 95% interval over splits. Right of zero in B: the spread helps.", fontsize=8, color=INK2)
    fig.text(0.01, 0.012, "B, circles: Gaussian centred on the MC mean with the same tuned variance 1/tau; "
             "squares: Gaussian with the validation residual variance of the selected network.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "fig4_benchmark_score.png", dpi=170)


def fig6(df):
    sets = [s for s in UCI_SETS if s in set(df.dataset)]
    cov = np.array([100, 90, 80, 70, 60, 50])
    scores = [("mc", "MC dropout sd", ORANGE), ("ens", "deep-ensemble sd", AQUA),
              ("knn", "distance to training rows", BLUE), ("oracle", "realised error (ceiling)", INK2)]
    ncol = 4
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow + 0.6), facecolor="white", sharex=True)
    axes = np.atleast_2d(axes)
    for ax, s in zip(axes.ravel(), sets):
        style_axes(ax)
        g = df[df.dataset == s]
        for key, label, colr in scores:
            vals = np.array([g[f"rc_{key}_{c}"].mean() for c in cov])
            ax.plot(cov, vals, color=colr, lw=2 if key != "oracle" else 1.2,
                    ls="-" if key != "oracle" else "--", marker="o" if key != "oracle" else None, ms=3)
        ax.axhline(1, color=INK2, lw=0.6)
        ax.set_title(UCI_LABEL[s], loc="left", fontsize=10, color=INK)
        ax.set_ylim(0, 1.15)
        ax.invert_xaxis()
    for ax in axes.ravel()[len(sets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("% of test rows kept", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE kept / RMSE all", color=INK2)
    handles = [plt.Line2D([], [], color=c, lw=2, ls="-" if k != "oracle" else "--") for k, _, c in scores]
    fig.legend(handles, [l for _, l, _ in scores], loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.035))
    fig.text(0.01, 0.008, "Abstain on the test rows with the highest score; all scores rank the errors of the same "
             "MC dropout predictions. Mean over splits.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / "fig6_risk_coverage.png", dpi=170)


if __name__ == "__main__":
    fig3()
    df = load_uci_runs()
    if df is not None:
        fig4(df)
        fig6(df)
    print("figures written")
