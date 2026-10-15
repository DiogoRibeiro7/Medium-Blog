"""Draw Figure 1 from results/summary.json and results/weekly_costs.npz.

(a) the weekly cost each planning method reports against the cost of its depots
    on the test weeks; (b) the distribution of weekly cost for the mean-forecast
    and SAA designs on the same weeks.

`--quick` reads the *_quick files and writes figures/fig1_quick.png.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main(quick: bool = False):
    suffix = "_quick" if quick else ""
    summary = json.loads((config.RESULTS / f"summary{suffix}.json").read_text())
    weekly = np.load(config.RESULTS / f"weekly_costs{suffix}.npz")
    rows = summary["plans"]
    ev_weekly, saa_weekly = weekly["mean_forecast"], weekly["saa"]
    n_test = summary["n_test"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200,
                               gridspec_kw=dict(width_ratios=[1.05, 1]))
    fig.patch.set_facecolor(SURFACE)

    style(a)
    ypos = np.arange(len(rows))[::-1]
    for yy, r in zip(ypos, rows):
        a.plot([r["planned"] / 1e3, r["realised"] / 1e3], [yy, yy], color=MUTED,
               lw=1.2, zorder=1)
        a.scatter(r["planned"] / 1e3, yy, s=70, facecolor=SURFACE, edgecolor=INK2,
                  lw=2, zorder=3)
        a.scatter(r["realised"] / 1e3, yy, s=70, color=INK, edgecolor=SURFACE,
                  lw=2, zorder=4)
        a.text(max(r["planned"], r["realised"]) / 1e3 + 0.5, yy, r["design"],
               va="center", ha="left", fontsize=8, color=INK2)
    a.set_yticks(ypos, [r["label"] for r in rows], fontsize=9, color=INK)
    a.set_xlim(44, 74)
    a.set_xlabel("EUR thousand per week", fontsize=9, color=INK2)
    a.scatter([], [], s=70, facecolor=SURFACE, edgecolor=INK2, lw=2,
              label="cost the plan reports")
    a.scatter([], [], s=70, color=INK, label=f"cost on {n_test:,} new weeks")
    a.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    a.set_title("(a) Planned against realised weekly cost", loc="left",
                fontsize=10.5, color=INK, pad=10)

    style(b)
    b.grid(axis="y", color=GRID, linewidth=1)
    for w, col, lab in [(saa_weekly, BLUE, "SAA depots (4)"),
                        (ev_weekly, ORANGE, "Mean-forecast depots (3)")]:
        xs = np.sort(w) / 1e3
        ps = np.arange(1, len(xs) + 1) / len(xs)
        b.plot(xs, ps, color=col, lw=2, solid_capstyle="round", label=lab)
    b.set_xlim(35, 110)
    b.set_ylim(0, 1.01)
    b.set_xlabel("weekly cost, EUR thousand", fontsize=9, color=INK2)
    b.set_ylabel("share of weeks at or below", fontsize=9, color=INK2)
    b.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    b.set_title(f"(b) Distribution of weekly cost on the same {n_test:,} weeks",
                loc="left", fontsize=10.5, color=INK, pad=10)
    for w, col, dy in [(ev_weekly, ORANGE, -0.06), (saa_weekly, BLUE, 0.05)]:
        q = np.percentile(w, 95) / 1e3
        b.scatter(q, 0.95, s=40, color=col, edgecolor=SURFACE, lw=2, zorder=5)
        b.text(q + 1.2, 0.95 + dy, f"95th pct {q:.1f}k", fontsize=8, color=INK2,
               va="center")
    ev_cheaper = summary["vss"]["share_weeks_ev_cheaper"]
    b.text(66, 0.50, f"The curves cross near EUR 49k: the three-depot\n"
           f"design is the cheaper one in {ev_cheaper:.0%} of weeks,\n"
           f"and the dearer one on average.", fontsize=8, color=INK2, va="center")

    fig.text(0.01, -0.02, "Synthetic instance: 10 candidate depots, 30 customer zones, "
             "lognormal demand with common, regional and zone shocks. "
             "Figure: Diogo Ribeiro.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    path = config.FIGURES / ("fig1_quick.png" if quick else "fig1_planned_vs_realised.png")
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure written to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
