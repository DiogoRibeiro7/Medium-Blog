"""Draws the five figures from results/. Run after analysis.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
FIGURES.mkdir(exist_ok=True)

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b2"
SOURCE = "Source: nycflights13 (CC0); code at github.com/DiogoRibeiro7/Medium-Blog/ml-pipeline-tests"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

NAMES = {
    "reviewed": "Reviewed pipeline",
    "leak_dep_delay": "Own departure delay\nused as a feature",
    "unlagged_congestion": "Congestion window\nnot lagged to T",
    "global_encoding": "Route and tail rates\nfrom all labels",
    "random_split": "Shuffled split\ninstead of time split",
    "drop_incomplete": "Cancelled and diverted\nflights dropped",
    "utc_hour": "Hour taken from\nthe UTC timestamp",
}


def _clean(ax, keep_bottom=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if not keep_bottom:
        ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fig_reported_realized(d):
    res = d["results"]
    keys = [k for k in NAMES if k in res] + [k for k in res if k not in NAMES]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for i, k in enumerate(keys):
        y = len(keys) - 1 - i
        rep, real = res[k]["reported"], res[k]["realized"]
        ax.plot([real, rep], [y, y], color=MUTED, linewidth=2.5, solid_capstyle="round", zorder=1)
        ax.plot(rep, y, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=2)
        ax.plot(real, y, "o", color=ORANGE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=2)
    labels = [NAMES.get(k, "Shipped pipeline\n(all six defects)") for k in keys]
    ax.set_yticks(range(len(keys)), list(reversed(labels)), fontsize=9)
    ax.axvline(0.5, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.503, -0.55, "chance", fontsize=8, color=INK_2, va="center")
    ax.set_ylim(-0.8, len(keys) - 0.5)
    ax.set_xlim(0.48, 0.95)
    ax.set_xlabel("AUC")
    ax.grid(axis="x", color=MUTED, linewidth=0.6)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.plot([], [], "o", color=BLUE, label="reported on the pipeline's own test set")
    ax.plot([], [], "o", color=ORANGE, label="realized in December, features as production computes them")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2)
    ax.set_title("What each defect reports, and what it delivers", loc="left", fontsize=11)
    fig.text(0.01, 0.01, SOURCE, fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "fig1_reported_realized.png", dpi=200)
    plt.close(fig)


LAYERS = [("data validation", "Data\nvalidation"), ("unit tests", "Unit tests\n(fixtures)"),
          ("property-based", "Property\n(Hypothesis)"), ("future perturbation", "Future\nperturbation"),
          ("null labels", "Null\nlabels"), ("split audit", "Split\naudit"),
          ("train/serve parity", "Train/serve\nparity"), ("metric gate", "AUC\ngate")]


def fig_detection(d):
    rows = [(n, res) for n, res in d["detection"] if n != "reviewed"]
    fig, ax = plt.subplots(figsize=(10, 4.9))
    for i, (name, res) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, (layer, _) in enumerate(LAYERS):
            failed = any(not passed for lay, _, passed in res if lay == layer)
            ax.add_patch(Rectangle((j + 0.06, y + 0.1), 0.88, 0.8, facecolor=BLUE if failed else SURFACE,
                                   edgecolor=BLUE if failed else MUTED, linewidth=1.2))
            ax.text(j + 0.5, y + 0.5, "caught" if failed else "passed", ha="center", va="center",
                    color="white" if failed else INK_2, fontsize=8.5)
    ax.set_xlim(0, len(LAYERS))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([j + 0.5 for j in range(len(LAYERS))], [lab for _, lab in LAYERS], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks([len(rows) - 1 - i + 0.5 for i in range(len(rows))], [NAMES[n] for n, _ in rows], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which test layer fails on each defect", loc="left", fontsize=11, pad=34)
    fig.text(0.01, 0.01, "The reviewed pipeline passes every layer. The AUC gate passes every defect. " + SOURCE,
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "fig4_detection.png", dpi=200)
    plt.close(fig)


def fig_leak_predictions():
    leak = np.load(RESULTS / "leak_december_predictions.npy")
    rev = np.load(RESULTS / "reviewed_december_predictions.npy")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=False)
    bins = np.linspace(0, 1, 41)
    for ax, p, title, colr in ((axes[0], rev, "Reviewed model", BLUE),
                               (axes[1], leak, "Model trained with its own departure delay", ORANGE)):
        ax.hist(p, bins=bins, color=colr, edgecolor=SURFACE, linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("predicted probability of disruption, December flights")
        ax.grid(axis="y", color=MUTED, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("flights")
    fig.text(0.01, 0.01, "At prediction time the departure delay is unknown. The model learned in training that "
             "a missing departure delay means a cancellation. " + "Source: nycflights13.", fontsize=7.8, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "fig2_leak_predictions.png", dpi=200)
    plt.close(fig)


def fig_split(d):
    s = d["split"]
    order = [("random rows", "Rows shuffled\n(80/20)"), ("random days", "Whole days\nshuffled"),
             ("months ahead", "Train Jan-Aug,\ntest Oct-Nov")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (k, lab) in enumerate(order):
        y = len(order) - 1 - i
        ax.plot([0.5, s[k]], [y, y], color=MUTED, linewidth=2)
        ax.plot(s[k], y, "o", color=BLUE, markersize=10, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(s[k] + 0.006, y, f"{s[k]:.3f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(order)), [lab for _, lab in reversed(order)], fontsize=9)
    ax.set_xlim(0.5, 0.82)
    ax.set_xlabel("test AUC of the same reviewed model")
    ax.grid(axis="x", color=MUTED, linewidth=0.6)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.set_title("The same model, three ways of holding out data", loc="left", fontsize=11)
    fig.text(0.01, 0.01, SOURCE, fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "fig3_split.png", dpi=200)
    plt.close(fig)


def fig_slices(d):
    sl = sorted(d["slices"], key=lambda r: r[3])
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for i, (col, val, n, a, lo, hi) in enumerate(sl):
        ax.plot([lo, hi], [i, i], color=BLUE if col == "carrier" else AQUA, linewidth=2, solid_capstyle="round")
        ax.plot(a, i, "o", color=BLUE if col == "carrier" else AQUA, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
    ax.set_yticks(range(len(sl)), [f"{'airline' if c == 'carrier' else 'airport'} {v} (n={n:,})"
                                   for c, v, n, *_ in sl], fontsize=8.5)
    ax.axvline(d["overall"], color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(d["overall"] + 0.003, len(sl) - 0.6, f"all flights {d['overall']:.3f}", fontsize=8, color=INK_2)
    ax.set_xlabel("December AUC, 95% interval from resampling days")
    ax.grid(axis="x", color=MUTED, linewidth=0.6)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.set_title("Performance by airline and airport, reviewed model", loc="left", fontsize=11)
    fig.text(0.01, 0.01, "Slices with at least 1,000 December flights. " + SOURCE, fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "fig5_slices.png", dpi=200)
    plt.close(fig)


def main():
    d = json.loads((RESULTS / "figure_data.json").read_text())
    fig_reported_realized(d)
    fig_detection(d)
    fig_leak_predictions()
    fig_split(d)
    fig_slices(d)


if __name__ == "__main__":
    main()
