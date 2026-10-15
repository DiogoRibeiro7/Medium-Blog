"""Figures for Part 6. Writes PNG files to ./figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import (HORIZON, MODELS, calibration_table, gain_curve, km_at, rule_of_thumb, run_all)
from sksurv.nonparametric import kaplan_meier_estimator

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#dddcd6"
TRUTH = "#a9a8a2"
VIOLET = "#4a3aa7"
AQUA = "#1baf7a"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": RULE, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "ytick.left": False,
})
PERCENT = plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
LABELS = {"Cox": "Cox", "random survival forest": "random survival\nforest",
          "discrete-time logistic": "discrete-time\nlogistic", "discrete-time boosting": "discrete-time\nboosting",
          "26-week classifier": "26-week\nclassifier", "rule of thumb": "rule of thumb"}


def header(fig, main, sub, y=0.95):
    fig.add_artist(plt.Line2D([0.04, 0.085], [y + 0.012, y + 0.012], color=VIOLET,
                              linewidth=3, transform=fig.transFigure))
    fig.text(0.04, y - 0.005, main, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.04, y - 0.005 - 0.36 / fig.get_figheight(), sub, fontsize=11, color=INK_2,
             ha="left", va="top")


def credit(fig, text="Data: UnempDur (Ecdat), McCall (1996). Figure by the author"):
    fig.text(0.96, 0.012, text, fontsize=8.5, color=INK_2, ha="right", va="bottom")


# ---------------------------------------------------------------------------
# Figure 1: two clocks
# ---------------------------------------------------------------------------


def figure_data(data):
    X, time, event = data["X"], data["time"], data["event"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.25)
    header(fig, "Claimants and non-claimants leave unemployment on different clocks",
           "3,343 spells of unemployment, by whether the worker filed a claim for unemployment insurance (UI).")
    for ui, colour, name in ((1, VIOLET, "filed a UI claim"), (0, ORANGE, "no UI claim")):
        m = X[:, 1] == ui
        tt, ss = kaplan_meier_estimator(event[m].astype(bool), time[m])
        axes[0].step(np.r_[0, 2 * tt], np.r_[1, ss], where="post", color=colour, linewidth=2.2)
        s26 = km_at(event[m], time[m], HORIZON)
        x_lab, y_lab = (27, s26 + 0.03) if ui else (3, 0.03)
        axes[0].text(x_lab, y_lab, f"{name}: {100 * s26:.0f}%\nstill jobless at 26 weeks",
                     color=colour, fontsize=10, va="bottom")
        k = np.arange(1, 21)
        hz = np.array([((time[m] == j) & event[m]).sum() / (time[m] >= j).sum() for j in k])
        axes[1].plot(2 * k, hz, color=colour, linewidth=1.6, marker="o", markersize=5)
    axes[0].axvline(26, color=INK_2, linewidth=0.9, linestyle=(0, (3, 3)))
    axes[0].set_xlim(0, 56)
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks([0, 8, 16, 26, 36, 46, 56])
    axes[0].yaxis.set_major_formatter(PERCENT)
    axes[0].set_xlabel("Weeks since losing the job")
    axes[0].set_title("Still jobless (Kaplan-Meier)", loc="left", fontsize=11.5, color=INK)
    axes[1].set_xlim(0, 41)
    axes[1].set_ylim(0, 0.32)
    axes[1].set_xticks([2, 10, 18, 26, 34, 40])
    axes[1].yaxis.set_major_formatter(PERCENT)
    axes[1].set_xlabel("Weeks since losing the job (end of each two-week interval)")
    axes[1].set_title("Share re-employed in each two-week interval", loc="left", fontsize=11.5, color=INK)
    axes[1].text(8, 0.27, "no UI claim: re-employment rates\nhighest in the first six weeks", color=ORANGE,
                 fontsize=10, va="top")
    axes[1].text(20, 0.16, "after about 18 weeks the\ntwo groups look alike", color=INK_2, fontsize=10)
    credit(fig)
    fig.savefig(OUT / "fig1_two_clocks.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: scores across repeats
# ---------------------------------------------------------------------------


def figure_scores(results):
    by_rep = results.groupby(["model", "repeat"]).mean(numeric_only=True)
    panels = (("Harrell C", "Harrell's C (higher better)"),
              ("AUC at 26 weeks", "AUC at 26 weeks (higher better)"),
              ("integrated Brier", "Integrated Brier (lower better)"),
              ("still jobless, top 20%", "Jobless at 26 weeks, top 20%"))
    order = list(MODELS) + ["26-week classifier", "rule of thumb"]
    colours = {"Cox": VIOLET, "random survival forest": INK_2, "discrete-time logistic": INK_2,
               "discrete-time boosting": ORANGE, "26-week classifier": AQUA, "rule of thumb": INK}
    fig, axes = plt.subplots(1, 4, figsize=(14, 6.4), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.78, bottom=0.12, wspace=0.14)
    header(fig, "Small differences on every score, and the order changes with the score",
           "Ten repeats of five-fold cross-validation: each dot is one repeat's average over its five test folds.")
    ypos = {m: len(order) - 1 - i for i, m in enumerate(order)}
    for ax, (metric, title) in zip(axes, panels):
        for m in order:
            if (m, 0) not in by_rep.index or np.isnan(by_rep.loc[m][metric]).all():
                continue
            v = by_rep.loc[m][metric].to_numpy()
            ax.scatter(v, np.full(len(v), ypos[m]), s=22, color=colours[m], alpha=0.75, linewidth=0)
            ax.plot([v.mean(), v.mean()], [ypos[m] - 0.3, ypos[m] + 0.3], color=colours[m], linewidth=2.2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True, color=RULE, linewidth=0.7)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.locator_params(axis="x", nbins=4)
    for m in ("26-week classifier", "rule of thumb"):
        axes[2].text(0.04, ypos[m], "no survival curve", transform=axes[2].get_yaxis_transform(),
                     fontsize=9, color=TRUTH, va="center")
    axes[3].xaxis.set_major_formatter(PERCENT)
    axes[0].set_yticks([ypos[m] for m in order])
    axes[0].set_yticklabels([LABELS[m].replace("\n", " ") for m in order])
    axes[0].tick_params(axis="y", length=0)
    credit(fig)
    fig.savefig(OUT / "fig2_scores.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: calibration at 26 weeks
# ---------------------------------------------------------------------------


def figure_calibration(data, pooled):
    event, time = data["event"], data["time"]
    names = list(MODELS) + ["26-week classifier"]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 4.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.17, wspace=0.1)
    header(fig, "Predicted against observed: share still jobless at 26 weeks",
           "Out-of-fold predictions from one repeat, in tenths of predicted risk; observed share by Kaplan-Meier.")
    for ax, name in zip(axes, names):
        p = pooled[name] if name == "26-week classifier" else pooled[name][:, HORIZON - 1]
        pred, obs = calibration_table(p, event, time)
        ax.plot([0, 0.65], [0, 0.65], color=TRUTH, linewidth=1.2)
        ax.plot(pred, obs, color=VIOLET, marker="o", markersize=5, linewidth=1.4)
        ax.set_title(LABELS[name].replace("\n", " "), loc="left", fontsize=10.5, color=INK)
        ax.text(0.02, 0.6, f"mean gap {np.mean(np.abs(pred - obs)):.3f}", fontsize=9.5, color=INK_2)
        ax.set_xlim(0, 0.65)
        ax.set_ylim(0, 0.65)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.xaxis.set_major_formatter(PERCENT)
        ax.yaxis.set_major_formatter(PERCENT)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    credit(fig)
    fig.savefig(OUT / "fig3_calibration.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: targeting
# ---------------------------------------------------------------------------


def figure_targeting(data, pooled):
    X, event, time = data["X"], data["event"], data["time"]
    shares = np.linspace(0.05, 0.6, 23)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.subplots_adjust(left=0.09, right=0.76, top=0.78, bottom=0.13)
    header(fig, "Whom to offer intensive support: the simplest ranking does best up to about 40%",
           "Share still jobless at 26 weeks among the workers ranked highest, by the share selected "
           "(out-of-fold predictions).")
    lines = [("random survival forest", INK_2, 1.0, (0, (2, 2))), ("discrete-time logistic", INK_2, 1.0, (0, (5, 2))),
             ("Cox", VIOLET, 2.0, "-"), ("discrete-time boosting", ORANGE, 2.0, "-"),
             ("26-week classifier", AQUA, 2.0, "-")]
    ends = {}
    for name, colour, width, style in lines:
        p = pooled[name] if name == "26-week classifier" else pooled[name][:, HORIZON - 1]
        g = gain_curve(p, event, time, shares)
        ax.plot(shares, g, color=colour, linewidth=width, linestyle=style)
        ends[name] = (g[-1], colour)
    g = gain_curve(rule_of_thumb(X).astype(float), event, time, shares)
    ax.plot(shares, g, color=INK, linewidth=2.6)
    base = km_at(event, time, HORIZON)
    ax.axhline(base, color=TRUTH, linewidth=1.5)
    ax.text(0.61, base, f"random selection: {100 * base:.0f}%", color=INK_2, va="center", fontsize=10)
    ax.axvline(0.2, color=INK_2, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(0.205, 0.35, "capacity: one\nworker in five", color=INK_2, fontsize=10, va="bottom")
    key = (("rule of thumb:\nUI claimants, oldest first", INK), ("26-week classifier", AQUA),
           ("discrete-time boosting", ORANGE), ("Cox", VIOLET),
           ("random survival forest (dotted)\nand discrete-time logistic (dashed)", INK_2))
    y = 0.60
    for text, colour in key:
        ax.text(0.61, y, text, color=colour, fontsize=10, va="top")
        y -= 0.033 if "\n" not in text else 0.05
    ax.set_xlim(0.05, 0.6)
    ax.set_ylim(0.3, 0.62)
    ax.xaxis.set_major_formatter(PERCENT)
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_xlabel("Share of workers selected")
    ax.set_ylabel("Still jobless at 26 weeks among those selected")
    credit(fig)
    fig.savefig(OUT / "fig4_targeting.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    data, results, pooled = run_all()
    figure_data(data)
    figure_scores(results)
    figure_calibration(data, pooled)
    figure_targeting(data, pooled)
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
