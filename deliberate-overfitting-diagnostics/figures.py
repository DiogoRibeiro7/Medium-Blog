"""Figures for the article. Run after exp1_memorization.py, exp2_noise_floor.py and exp3_bugs.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import FIG, RES
from exp2_noise_floor import power_law

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
NEU = "#8a8984"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots 1-4
CREDIT = "Diogo Ribeiro · data: simulation, UCI Adult, California housing (1990 census)"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
})


def _credit(fig, y=0.005):
    fig.text(0.995, y, CREDIT, ha="right", va="bottom", fontsize=7.5, color=INK2)


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- figure 1
def fig1_memorisation():
    d = pd.read_csv(RES / "exp1a_memorisation_grid.csv")
    acc = d[d["train_acc"].notna()]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    etas = np.sort(acc["eta"].unique())
    tr = acc.groupby("eta")["train_acc"].min()
    ax.plot(etas, tr.loc[etas], color=INK, lw=2.4)
    ax.text(0.505, tr.iloc[-1] + 0.012, "Training accuracy: every model, real or shuffled labels",
            ha="right", va="bottom", fontsize=9, color=INK)
    ax.plot(etas, 1 - etas, color=INK2, lw=1.2, ls=":")
    ax.text(0.32, 0.735, "Best possible test accuracy\n(1 − Bayes error), dotted", fontsize=8.5, color=INK2,
            va="bottom")
    colours = [S1, S2, S3, S4]
    real = acc[acc["labels"] == "real"].groupby(["model", "eta"])["test_acc"].mean().unstack(0)
    order = real.loc[0.0].sort_values(ascending=False).index
    offsets = {}
    for i, (m, c) in enumerate(zip(order, colours)):
        ax.plot(etas, real.loc[etas, m], color=c, marker="o", ms=5, mec=SURFACE, mew=1.2)
        offsets[m] = real.loc[0.0, m]
    ys = np.array(sorted(offsets.values(), reverse=True))
    ylab = ys.copy()
    for i in range(1, len(ylab)):
        ylab[i] = min(ylab[i], ylab[i - 1] - 0.03)
    for (m, c), yl in zip(zip(order, colours), ylab):
        ax.text(-0.012, yl, m.split(" (")[0], ha="right", va="center", fontsize=8.5, color=INK)
        ax.plot([-0.009, -0.003], [yl, offsets[m]], color=c, lw=1)
    sh = acc[acc["labels"] == "shuffled"].groupby("eta")["test_acc"]
    ax.fill_between(etas, sh.min().loc[etas], sh.max().loc[etas], color=NEU, alpha=0.25, lw=0)
    ax.text(0.25, 0.465, "Test accuracy with shuffled training labels (all four models)", fontsize=8.5,
            color=INK2, ha="center")
    ax.set_xlim(-0.13, 0.51)
    ax.set_ylim(0.42, 1.06)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("Share of labels flipped at random = Bayes error of the problem")
    ax.set_ylabel("Accuracy")
    ax.set_title("Every model memorises every version of the data; only test accuracy tracks the signal",
                 loc="left", fontsize=11.5)
    fig.text(0.01, 0.005, "Simulation: 2,000 training rows, 10 features (5 informative), 20,000 test rows; "
             "test lines are means of 3 replications.", fontsize=7.5, color=INK2, va="bottom")
    _credit(fig, 0.035)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
    _save(fig, "fig1_memorisation_grid.png")


# --------------------------------------------------------------------------- figure 2
def fig2_feature_sets():
    d = pd.read_csv(RES / "exp1c_feature_set_ranking.csv").sort_values("tree_train_acc")
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ys = np.arange(len(d))
    base = d["majority_baseline"].iloc[0]
    for y, r in zip(ys, d.itertuples()):
        ax.plot([r.reg_xgb_test_acc, r.tree_train_acc], [y, y], color=GRID, lw=6, solid_capstyle="round")
        ax.scatter(r.tree_train_acc, y, s=95, facecolor=SURFACE, edgecolor=INK, lw=1.6, zorder=3)
        ax.scatter(r.reg_xgb_test_acc, y, s=105, color=S1, edgecolor=SURFACE, lw=1.5, zorder=4)
        ax.text(1.008, y, f"{100 * r.tree_train_acc:.1f}%  /  {100 * r.reg_xgb_test_acc:.1f}%",
                va="center", fontsize=8.5, color=INK2)
    ax.axvline(base, color=INK2, lw=0.9, ls=":")
    ax.text(base + 0.003, -0.55, f"Always predict \"<=50K\": {100 * base:.1f}%", fontsize=8, color=INK2,
            ha="left", va="center")
    ax.set_yticks(ys, d["feature_set"])
    ax.set_ylim(-0.85, len(d) - 0.5)
    ax.set_xlim(0.72, 1.0)
    ax.set_xlabel("Accuracy")
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=9,
                      label="Training accuracy of a tree grown to purity"),
               Line2D([], [], marker="o", ls="", color=S1, markersize=10,
                      label="Held-out accuracy of a regularised XGBoost")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=2, frameon=False,
               fontsize=8.8)
    fig.text(0.01, 0.965, "Ranking feature sets by how well an overfit model fits them, UCI Adult",
             fontsize=12, fontweight="bold", va="top")
    fig.text(0.01, 0.005, "34,189 training rows, 14,653 test rows. Right-hand figures: training / held-out.",
             fontsize=7.5, color=INK2, va="bottom")
    _credit(fig, 0.035)
    fig.subplots_adjust(left=0.33, right=0.86, top=0.86, bottom=0.17)
    _save(fig, "fig2_feature_set_ranking.png")


# --------------------------------------------------------------------------- figure 3
def fig3_power():
    d = pd.read_csv(RES / "exp1b_detection_power.csv")
    g = d.groupby("eta").agg(mem=("memorised", "mean"),
                             leaves=("p_leaves", lambda p: (p <= 0.05).mean()),
                             hold=("p_holdout", lambda p: (p <= 0.05).mean()))
    n_rep = d.groupby("eta").size().iloc[0]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = g.index.to_numpy()
    for col, c, lab, dy in (("mem", INK, "\"It memorised, so the signal is there\"", 0.03),
                            ("hold", S1, "Held-out accuracy vs permuted labels", 0.0),
                            ("leaves", S2, "Tree leaves: real vs shuffled labels", 0.0)):
        p = g[col].to_numpy()
        se = np.sqrt(np.clip(p * (1 - p), 1e-4, None) / n_rep)
        ax.fill_between(x, np.clip(p - 1.96 * se, 0, 1), np.clip(p + 1.96 * se, 0, 1), color=c, alpha=0.12,
                        lw=0)
        ax.plot(x, p, color=c, marker="o", ms=5, mec=SURFACE, mew=1.2)
    ax.text(0.40, 1.03, "\"It memorised, so the signal is there\"", color=INK, fontsize=9, ha="center")
    ax.text(0.381, 0.66, "Held-out accuracy\nvs permuted labels", color=INK, fontsize=9)
    ax.text(0.302, 0.25, "Tree leaves at purity:\nreal vs 39 shuffled copies",
            color=INK, fontsize=9)
    ax.axhline(0.05, color=INK2, lw=0.9, ls=":")
    ax.text(0.302, 0.065, "5% false-alarm target", ha="left", fontsize=8, color=INK2)
    ax.set_xlim(0.295, 0.505)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel("Bayes error of the problem (0.5 = labels unrelated to the features)")
    ax.set_ylabel("Share of data sets where the rule says \"signal\"")
    ax.set_title("Three ways to ask whether the features carry signal", loc="left")
    fig.text(0.01, 0.005, f"{n_rep} simulated data sets per point, 2,000 rows each; bands are 95% intervals.",
             fontsize=7.5, color=INK2, va="bottom")
    _credit(fig, 0.035)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.17)
    _save(fig, "fig3_detection_power.png")


# --------------------------------------------------------------------------- figure 4
def fig4_bug_matrix():
    m = pd.read_csv(RES / "exp3_detection_matrix.csv")
    info = pd.read_csv(RES / "exp3_check_details.csv").set_index("bug")
    bugs = list(dict.fromkeys(m["bug"]))
    checks = list(dict.fromkeys(m["check"]))
    piv = m.pivot(index="bug", columns="check", values="flagged").loc[bugs, checks]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.grid(False)
    ax.axvspan(0.5, 1.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    for i, b in enumerate(bugs):
        for j, c in enumerate(checks):
            if piv.loc[b, c]:
                ax.scatter(j, i, s=260, color=S2 if b != "No bug" else NEU, edgecolor=SURFACE, lw=1.5, zorder=3)
            else:
                ax.scatter(j, i, s=40, facecolor=SURFACE, edgecolor=NEU, lw=1.0, zorder=3)
        r = info.loc[b]
        ax.text(len(checks) - 0.35, i, f"{100 * r.reported_test_acc:.1f}%  →  {100 * r.clean_acc:.1f}%",
                va="center", fontsize=8.8, color=INK2)
    ax.text(len(checks) + 0.45, -0.95, "Reported →\non clean data", fontsize=8.3, color=INK2, va="center",
            ha="center")
    short = {"C1": "Overfit\none batch", "C2": "Overfit full\ntraining set", "C3": "Real vs\nshuffled\nlabels",
             "C4": "Held-out vs\npermuted\nlabels", "C5": "Single-\ncolumn\naudit",
             "C6": "Train-test\nduplicate\naudit", "C7": "Label\naudit"}
    labels = [short[c.split()[0]] for c in checks]
    ax.set_xticks(range(len(checks)), labels, fontsize=8.3)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(bugs)), bugs)
    ax.set_xlim(-0.6, len(checks) + 1.4)
    ax.set_ylim(len(bugs) - 0.5, -1.3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.975, "Which planted bug each check catches (filled = the check fails)", fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.01, 0.005, "UCI Adult pipeline, 70/30 split. Shaded column: the \"overfit the training data\" "
             "check. Accuracy of the regularised model as the pipeline reports it, then on clean rows.",
             fontsize=7.5, color=INK2, va="bottom")
    _credit(fig, 0.035)
    fig.subplots_adjust(left=0.33, right=0.99, top=0.8, bottom=0.1)
    _save(fig, "fig4_bug_detection_matrix.png")


# --------------------------------------------------------------------------- figure 5
def fig5_adult_floor():
    d = pd.read_csv(RES / "exp2a_adult_duplicate_floor.csv")
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = d["step"].to_numpy()
    upper = np.minimum(d["lookup_cv_error"], d["xgb_cv_error"])
    ax.fill_between(x, d["plug_in_floor"], upper, color=S1, alpha=0.12, lw=0)
    ax.plot(x, d["plug_in_floor"], color=INK, marker="o", ms=5, mec=SURFACE, mew=1.2)
    ax.plot(x, d["xgb_cv_error"], color=S1, marker="o", ms=5, mec=SURFACE, mew=1.2)
    ax.plot(x, d["lookup_cv_error"], color=S2, marker="o", ms=4, mec=SURFACE, mew=1.0, lw=1.4)
    ax.annotate("Training error of a tree grown to purity\n= plug-in duplicate floor", xy=(x[7], d["plug_in_floor"].iloc[7]),
                xytext=(x[3] - 0.4, 0.085), fontsize=9, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(x[-1], d["xgb_cv_error"].iloc[-1] - 0.008, "Regularised XGBoost, 5-fold CV", fontsize=9,
            color=INK, ha="right", va="top")
    ax.text(x[1] + 0.1, d["lookup_cv_error"].iloc[1] + 0.006, "Cell lookup table, 5-fold CV", fontsize=9,
            color=INK, va="bottom")
    ax.text(x[10] - 0.3, 0.075, "Bayes error lies in here\n(in expectation)", fontsize=8.5, color=INK2,
            ha="center", va="center")
    ax.set_xticks(x, ["+ " + a.replace("_", " ") if k else a.replace("_", " ")
                      for k, a in enumerate(d["added"])], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("Error rate")
    ax.set_ylim(-0.01, 0.27)
    ax.set_title("UCI Adult: the floor an overfit model reports collapses as columns are added", loc="left")
    fig.text(0.01, 0.005, "48,842 rows. Columns added left to right; each point uses every column up to it.",
             fontsize=7.5, color=INK2, va="bottom")
    _credit(fig, 0.035)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.3)
    _save(fig, "fig5_adult_duplicate_floor.png")


# --------------------------------------------------------------------------- figure 6
def fig6_continuous_floors():
    ch = pd.read_csv(RES / "exp2b_cover_hart.csv")
    lc = pd.read_csv(RES / "exp2c_learning_curves.csv")
    fits = pd.read_csv(RES / "exp2c_asymptotes.csv").set_index("data")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6))
    ax = axes[0]
    sub = ch[ch["eta"] == 0.2].groupby(["noise_dims", "n"])[["cover_hart_lower", "xgb_test_error"]].mean()
    for nd, c in zip((0, 20), (S1, S2)):
        s = sub.loc[nd]
        ax.plot(s.index, s["cover_hart_lower"], color=c, marker="o", ms=5, mec=SURFACE, mew=1.2)
        ax.plot(s.index, s["xgb_test_error"], color=c, lw=1.3, ls="--")
        yb = s["cover_hart_lower"].iloc[-1] - (0.009 if nd == 0 else 0.0)
        yx = s["xgb_test_error"].iloc[-1] + 0.004
        ax.text(s.index[-1] * 1.2, yb, f"bound, {nd} irrelevant columns", fontsize=8.2, va="center", color=INK)
        ax.text(s.index[-1] * 1.2, yx, f"XGBoost test error, {nd} irrelevant", fontsize=8.2,
                va="center", color=INK2)
    ax.axhline(0.2, color=INK, lw=1.2, ls=":")
    ax.text(500, 0.205, "True Bayes error 0.20", fontsize=8.3, color=INK, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(400, 1.2e6)
    ax.set_xlabel("Training rows")
    ax.set_ylabel("Error rate")
    ax.set_title("Cover-Hart lower bound\nfrom 1-NN error", loc="left", fontsize=11)
    for ax, name, true in ((axes[1], "Simulation (Bayes error 0.20)", 0.2), (axes[2], "UCI Adult (14 columns)", None)):
        g = lc[lc["data"] == name].groupby("n")[["test_error", "train_error", "overfit_train_error"]].mean()
        f = fits.loc[name]
        grid = np.geomspace(g.index.min(), g.index.max() * 30, 200)
        ax.plot(grid, power_law(grid, f.asymptote, f.b, f.c), color=S1, lw=1.2, ls="--")
        ax.plot(g.index, g["test_error"], color=S1, marker="o", ms=5, mec=SURFACE, mew=1.2)
        ax.plot(g.index, g["train_error"], color=S2, marker="o", ms=5, mec=SURFACE, mew=1.2)
        ax.plot(g.index, g["overfit_train_error"], color=NEU, marker="o", ms=4, mec=SURFACE, mew=1.0, lw=1.4)
        ax.axhline(f.asymptote, color=S1, lw=0.9, ls=":")
        if true is not None:
            ax.axhline(true, color=INK, lw=1.2, ls=":")
            ax.text(g.index.max() * 30, true - 0.008, "True Bayes error 0.20", fontsize=8.3, color=INK,
                    va="top", ha="right")
            ax.text(g.index[3] * 1.2, g["train_error"].iloc[2] - 0.02, "Train (regularised)", fontsize=8.5,
                    color=INK, va="top")
        else:
            ax.text(g.index[1], g["train_error"].iloc[1] - 0.008, "Train (regularised)", fontsize=8.5,
                    color=INK, va="top")
        ax.text(g.index[1], g["test_error"].iloc[1] + 0.01, "Test", fontsize=8.5, color=INK)
        ax.text(g.index[-1], g["overfit_train_error"].iloc[-1] + 0.008, "Train, tree grown to purity",
                fontsize=8.5, color=INK2, ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("Training rows")
        ax.set_title(("Learning curves, simulation" if true else "Learning curves, UCI Adult")
                     + f"\nfitted asymptote {f.asymptote:.3f} (dotted)", loc="left", fontsize=11)
    fig.suptitle("Continuous features: estimating the floor without exact duplicates", x=0.01, ha="left",
                 fontweight="bold", fontsize=12.5)
    fig.text(0.01, 0.005, "Left: means of 4 replications. Middle and right: regularised XGBoost with early "
             "stopping, means of 3 replications; dashed line is a + b·n^(-c) fitted to the test errors.",
             fontsize=7.5, color=INK2, va="bottom")
    _credit(fig, 0.035)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.2, wspace=0.28)
    _save(fig, "fig6_continuous_floors.png")


def main():
    for f in (fig1_memorisation, fig2_feature_sets, fig3_power, fig4_bug_matrix, fig5_adult_floor,
              fig6_continuous_floors):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e.filename)


if __name__ == "__main__":
    main()
