"""Figures for Part 5. Writes PNG files to ./figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from competing_risks import (CONTRACT, METHODS, SEGMENTS, aalen_johansen, kaplan_meier, repairs_in_service,
                             run_all, segment_name, step_at, true_values)

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


def header(fig, main, sub, y=0.95):
    fig.add_artist(plt.Line2D([0.04, 0.085], [y + 0.012, y + 0.012], color=VIOLET,
                              linewidth=3, transform=fig.transFigure))
    fig.text(0.04, y - 0.005, main, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.04, y - 0.005 - 0.36 / fig.get_figheight(), sub, fontsize=11, color=INK_2,
             ha="left", va="top")


def credit(fig, text="Simulated data. Figure by the author"):
    fig.text(0.96, 0.012, text, fontsize=8.5, color=INK_2, ha="right", va="bottom")


# ---------------------------------------------------------------------------
# Figure 1: where the pumps are, and what one minus Kaplan-Meier claims
# ---------------------------------------------------------------------------


def figure_states(fleet):
    grid = np.linspace(0, 10, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.12, wspace=0.08)
    header(fig, "One minus Kaplan-Meier counts breakdowns that replaced pumps never had",
           "Share of pumps broken down, still in service and replaced as planned, by age (Aalen-Johansen), "
           "with one minus Kaplan-Meier for breakdown.")
    for ax, prog, title in ((axes[0], 0, "Sites outside the programme"), (axes[1], 1, "Sites in the modernisation programme")):
        m = fleet["prog"] == prog
        t, c1, c2, _ = aalen_johansen(fleet["time"][m], fleet["status"][m])
        b = step_at(t, c1, grid)
        r = step_at(t, c2, grid)
        kt, ks = kaplan_meier(fleet["time"][m], (fleet["status"][m] == 1).astype(int))
        net = 1 - step_at(kt, ks, grid, 1.0)
        ax.fill_between(grid, 0, b, step="post", color=ORANGE, alpha=0.85, linewidth=0)
        ax.fill_between(grid, 1 - r, 1, step="post", color=AQUA, alpha=0.75, linewidth=0)
        ax.step(grid, net, where="post", color=VIOLET, linewidth=2.2)
        ax.axvline(CONTRACT, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_title(title, loc="left", fontsize=11.5, color=INK)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel("Age of pump (years)")
        ax.yaxis.set_major_formatter(PERCENT)
        b5, net5 = float(step_at(t, c1, CONTRACT)), 1 - float(step_at(kt, ks, CONTRACT, 1.0))
        ax.text(CONTRACT + 0.12, b5 / 2, f"broken down\nby year 5: {100 * b5:.0f}%", color=SURFACE,
                fontsize=10, va="center", fontweight="bold")
        ax.annotate(f"one minus Kaplan-Meier\nfor breakdown: {100 * net5:.0f}%", xy=(CONTRACT, net5),
                    xytext=(0.3, 0.3), color=VIOLET, fontsize=10, va="center",
                    arrowprops={"arrowstyle": "-", "color": VIOLET, "linewidth": 0.9})
        ax.text(0.3, 0.97, "replaced as planned", color=INK, fontsize=10, va="top")
        ax.text(0.3, 0.55, "still in service", color=INK_2, fontsize=10, va="center")
    axes[0].set_ylabel("Share of pumps installed")
    credit(fig)
    fig.savefig(OUT / "fig1_states.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: same breakdown hazard, fewer breakdowns
# ---------------------------------------------------------------------------


def nelson_aalen_cause(time, status, cause):
    order = np.argsort(time, kind="stable")
    t, s = time[order], status[order]
    uniq, first = np.unique(t, return_index=True)
    d = np.add.reduceat((s == cause).astype(int), first)
    n = len(t) - first
    keep = d > 0
    return uniq[keep], np.cumsum(d[keep] / n[keep])


def figure_hazard_vs_incidence(fleet):
    grid = np.linspace(0, 10, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.25)
    header(fig, "The same breakdown hazard, fewer breakdowns",
           "Pumps at programme sites are replaced four times as often; their hazard of breaking down "
           "while in service is unchanged.")
    for prog, colour, name in ((0, VIOLET, "other sites"), (1, ORANGE, "programme sites")):
        m = fleet["prog"] == prog
        tt, H = nelson_aalen_cause(fleet["time"][m], fleet["status"][m], 1)
        axes[0].step(grid, step_at(tt, H, grid), where="post", color=colour, linewidth=2)
        t, c1, _, _ = aalen_johansen(fleet["time"][m], fleet["status"][m])
        axes[1].step(grid, step_at(t, c1, grid), where="post", color=colour, linewidth=2)
        axes[1].text(9.8, float(step_at(t, c1, 9.8)) + (0.025 if prog == 0 else -0.06), name,
                     color=colour, ha="right", fontsize=10.5)
    axes[0].text(0.3, 1.3, "other sites (violet) and programme sites (orange)\nlie close together until few pumps remain",
                 color=INK_2, fontsize=10, va="top")
    axes[0].set_title("Cumulative hazard of breakdown (Nelson-Aalen)", loc="left", fontsize=11.5, color=INK)
    axes[0].set_ylim(0, 1.4)
    axes[1].set_title("Share of pumps that break down (Aalen-Johansen)", loc="left", fontsize=11.5, color=INK)
    axes[1].set_ylim(0, 0.6)
    axes[1].yaxis.set_major_formatter(PERCENT)
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel("Age of pump (years)")
    credit(fig)
    fig.savefig(OUT / "fig2_hazard_vs_incidence.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: repairs per pump in service, and per contract
# ---------------------------------------------------------------------------


def figure_repairs(fleet):
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.subplots_adjust(left=0.08, right=0.66, top=0.78, bottom=0.13)
    header(fig, "A pump that has left service needs no more repairs",
           "Standard-duty pumps: expected repairs per pump in service (mean cumulative function) and "
           "per contract.")
    for prog, colour, name in ((0, VIOLET, "other sites"), (1, ORANGE, "programme sites")):
        m = (fleet["prog"] == prog) & (fleet["duty"] == 0)
        r = repairs_in_service(fleet, m, horizon=CONTRACT)
        ages = np.r_[0.0, r["ages"]]
        ax.step(ages, np.r_[0.0, r["mcf"]], where="post", color=colour, linewidth=1.4,
                linestyle=(0, (4, 2)))
        ax.step(ages, np.r_[0.0, r["expected"]], where="post", color=colour, linewidth=2.4)
        ax.text(5.1, float(r["expected"][-1]), f"per contract, {name}: {r['expected'][-1]:.2f}",
                color=colour, va="center", fontsize=10.5)
        if prog == 0:
            ax.text(5.1, float(r["mcf"][-1]) + 0.15, f"per pump in service,\nboth site types: about {r['mcf'][-1]:.1f}",
                    color=INK_2, va="center", fontsize=10.5)
    ax.set_xlim(0, CONTRACT)
    ax.set_ylim(0, 5.6)
    ax.set_xticks(np.arange(0, CONTRACT + 1))
    ax.set_xlabel("Age of pump (years)")
    ax.set_ylabel("Expected number of repairs")
    credit(fig)
    fig.savefig(OUT / "fig3_repairs.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: contract prices across simulated fleets
# ---------------------------------------------------------------------------


def figure_prices(reps):
    colours = {"dashboard": AQUA, "one cause at a time": ORANGE, "competing risks": VIOLET}
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.8, bottom=0.17)
    header(fig, "Pricing the contract: one mistake undercharges, the other can double the price",
           "Five-year service contract, 500 simulated fleets: middle 95% of estimates (line), "
           "mean (dot), true price (grey).")
    labels = []
    for i, seg in enumerate(SEGMENTS):
        y = len(SEGMENTS) - 1 - i
        tv = true_values(*seg)["price"]
        ax.plot([tv, tv], [y - 0.38, y + 0.38], color=TRUTH, linewidth=5, solid_capstyle="butt")
        for k, m in enumerate(METHODS):
            v = reps[seg][m]
            lo, hi = np.percentile(v, [2.5, 97.5])
            yy = y + 0.22 * (1 - k)
            ax.plot([lo, hi], [yy, yy], color=colours[m], linewidth=2.6, solid_capstyle="round")
            ax.scatter([v.mean()], [yy], s=36, color=colours[m], zorder=3, edgecolor=SURFACE, linewidth=0.8)
        labels.append(segment_name(*seg).replace(", ", "\n"))
    ax.set_yticks(range(len(SEGMENTS)))
    ax.set_yticklabels(labels[::-1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color=RULE, linewidth=0.7)
    ax.set_xlim(1000, 10000)
    ax.set_xticks(range(2000, 10001, 2000))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Break-even contract price per pump (EUR)")
    for k, m in enumerate(METHODS):
        fig.text(0.27 + 0.2 * k, 0.065, m, color=colours[m], fontsize=10.5)
    credit(fig)
    fig.savefig(OUT / "fig4_contract_prices.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fleet, (reps, _, _, _) = run_all()
    figure_states(fleet)
    figure_hazard_vs_incidence(fleet)
    figure_repairs(fleet)
    figure_prices(reps)
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
