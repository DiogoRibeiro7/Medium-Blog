"""Figures from results/summary.json (written by loops_numbers.py).

python figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import loops_core as L  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9890", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": MUTED,
    }
)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def families(summary):
    out = {}
    for key, m in summary["models"].items():
        out.setdefault(m["family"], []).append((m["params_m"], key))
    return {f: [k for _, k in sorted(v)] for f, v in sorted(out.items())}


def fig_rate_by_size(summary):
    fams = families(summary)
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.6), sharey=True, squeeze=False)
    shown = [("greedy", BLUE, "-", "Greedy"), ("T0.5", AQUA, "--", "Sampling, T = 0.5"),
             ("T1", ORANGE, "-", "Sampling, T = 1")]  # fmt: skip
    for ax, (fam, keys) in zip(axes[0], fams.items(), strict=True):
        x = [summary["models"][k]["params_m"] for k in keys]
        for rule, col, ls, lab in shown:
            y = [100 * summary["models"][k][rule]["rate"] for k in keys]
            ax.plot(x, y, color=col, ls=ls, lw=2, marker="o", ms=5, label=lab)
        h = [100 * summary["models"][k]["human"]["rate"] for k in keys]
        ax.plot(x, h, color=MUTED, lw=1, ls=":", label="The books themselves")
        ax.set_xscale("log")
        ax.set_title(fam, loc="left", fontsize=11)
        ax.set_xlabel("Parameters (millions)")
        style(ax)
    axes[0][0].set_ylabel("Continuations ending in a loop (%)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_rate_by_size.png", dpi=200)
    plt.close(fig)


def fig_reinforcement(summary):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for key, m in summary["models"].items():
        r = m.get("reinforcement")
        if not r:
            continue
        y = [r["pre_p"], *r["cycle_p"]]
        ax.plot(range(len(y)), y, marker="o", ms=3, lw=1.2, label=key)
    ax.set_xticks(range(7), ["before", "1", "2", "3", "4", "5", "6"])
    ax.set_xlabel("Cycle of the loop")
    ax.set_ylabel("Mean probability of the chosen token")
    style(ax)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_reinforcement.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    fig_rate_by_size(summary)
    fig_reinforcement(summary)
    print(f"figures written to {FIGURES}; rules {', '.join(L.RULES)}; {np.size(list(summary['models']))} models")


if __name__ == "__main__":
    main()
