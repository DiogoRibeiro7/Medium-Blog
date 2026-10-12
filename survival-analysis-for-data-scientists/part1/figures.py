"""Figures for Part 1. Writes PNG files to ./figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kaplan_meier import (HORIZON, LAUNCH_TO_SNAPSHOT, MARGIN, SEED, censored_as_active,
                          drop_censored, kaplan_meier, simulate_customers,
                          snapshot_and_replications, true_rmst, true_survival)

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


def header(fig, main, sub, y=0.95):
    """Title block of this series: a short violet rule, then title and subtitle."""
    fig.add_artist(plt.Line2D([0.04, 0.085], [y + 0.012, y + 0.012], color=VIOLET,
                              linewidth=3, transform=fig.transFigure))
    fig.text(0.04, y - 0.005, main, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.04, y - 0.005 - 0.36 / fig.get_figheight(), sub, fontsize=11, color=INK_2,
             ha="left", va="top")


def credit(fig, text="Simulated data. Figure by the author"):
    fig.text(0.96, 0.012, text, fontsize=8.5, color=INK_2, ha="right", va="bottom")


# ---------------------------------------------------------------------------
# Figure 1: from calendar time to time since signup
# ---------------------------------------------------------------------------


def figure_timelines():
    customers, _, _ = snapshot_and_replications()
    rng = np.random.default_rng(SEED + 7)
    pick = np.sort(rng.choice(len(customers["time"]), 16, replace=False))
    signup = customers["signup"][pick]
    time = customers["time"][pick]
    event = customers["event"][pick]
    order = np.argsort(signup)
    signup, time, event = signup[order], time[order], event[order]
    rows = np.arange(len(pick))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    fig.subplots_adjust(left=0.04, right=0.975, top=0.78, bottom=0.14, wspace=0.08)
    header(fig, "Censoring comes from when customers joined, not from what they did",
           "Sixteen customers from the snapshot. Left: on the calendar. Right: the same "
           "customers aligned at their signup date.")
    for ax, start in ((axes[0], signup), (axes[1], np.zeros_like(signup))):
        for y, s, t, e in zip(rows, start, time, event):
            ax.plot([s, s + t], [y, y], color=VIOLET if e else INK_2, linewidth=2.2,
                    solid_capstyle="butt")
            if e:
                ax.scatter([s + t], [y], marker="x", s=60, color=VIOLET, linewidth=2, zorder=3)
            else:
                ax.scatter([s + t], [y], s=50, facecolor=SURFACE, edgecolor=INK_2,
                           linewidth=1.6, zorder=3)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_ylim(-1, len(rows))
    axes[0].axvline(LAUNCH_TO_SNAPSHOT, color=ORANGE, linewidth=1.6)
    axes[0].text(LAUNCH_TO_SNAPSHOT - 0.6, len(rows) - 0.4, "snapshot", color=ORANGE,
                 fontsize=10.5, ha="right", va="center")
    axes[0].set_xlim(0, 38)
    axes[0].set_xlabel("Months since the product launched")
    axes[1].set_xlim(0, 38)
    axes[1].set_xlabel("Months since the customer signed up")
    axes[1].scatter([], [], marker="x", s=60, color=VIOLET, linewidth=2, label="churned")
    axes[1].scatter([], [], s=50, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.6,
                    label="still active at the snapshot (censored)")
    axes[1].legend(loc="lower right", fontsize=10.5)
    credit(fig)
    fig.savefig(OUT / "fig1_timelines.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: four estimates of the retention curve
# ---------------------------------------------------------------------------


def figure_curves():
    customers, _, _ = snapshot_and_replications()
    t, e = customers["time"], customers["event"]
    km = kaplan_meier(t, e)
    grid = np.linspace(0, 36, 721)

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    fig.subplots_adjust(left=0.08, right=0.80, top=0.80, bottom=0.12)
    header(fig, "Two shortcuts miss the retention curve in opposite directions",
           "Share of customers still subscribed by months since signup: Kaplan-Meier and two "
           "shortcuts on one snapshot of 5,000 customers.")
    ax.plot(grid, true_survival(grid), color=TRUTH, linewidth=6, solid_capstyle="round",
            zorder=1)
    steps_t = np.r_[0, km["time"]]
    ax.fill_between(steps_t, np.r_[1, km["lower"]], np.r_[1, km["upper"]], step="post",
                    color=VIOLET, alpha=0.18, linewidth=0, zorder=2)
    ax.step(steps_t, np.r_[1, km["surv"]], where="post", color=VIOLET, linewidth=1.8, zorder=3)
    ax.plot(grid, drop_censored(t, e, grid), color=ORANGE, linewidth=2.0, zorder=3)
    ax.plot(grid, censored_as_active(t, e, grid), color=AQUA, linewidth=2.0, zorder=3)

    end = 36
    labels = [
        (true_survival(end), "truth", INK_2, 0.035),
        (float(km["surv"][-1]), "Kaplan-Meier,\nwith 95% band", VIOLET, -0.065),
        (float(drop_censored(t, e, end)[0]), "drop the\ncensored", ORANGE, 0.045),
        (float(censored_as_active(t, e, end)[0]), "count the censored\nas still active", AQUA, 0.0),
    ]
    for y, text, colour, dy in labels:
        ax.text(end + 0.8, y + dy, text, color=colour, fontsize=10.5, ha="left", va="center")
    for x in (12, 24):
        ax.axvline(x, color=RULE, linewidth=1.0, zorder=0)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Still subscribed")
    credit(fig)
    fig.savefig(OUT / "fig2_retention_curves.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: the acquisition budget across 1,000 snapshots
# ---------------------------------------------------------------------------


def figure_budget():
    _, rep, _ = snapshot_and_replications()
    methods = [("Kaplan-Meier", 4, VIOLET), ("Fixed horizon\n(customers with 24 months)", 5, INK_2),
               ("Count the censored\nas still active", 7, AQUA), ("Drop the censored", 6, ORANGE)]
    truth = MARGIN * true_rmst(HORIZON)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(left=0.25, right=0.96, top=0.76, bottom=0.15)
    header(fig, "How much can we pay to acquire a customer?",
           f"Break-even acquisition cost (EUR {MARGIN:g} margin x expected paid months in the "
           "first 24), across 1,000 snapshots of 5,000 customers.")
    for row, (name, col, colour) in enumerate(methods):
        values = MARGIN * rep[:, col]
        lo, mid, hi = np.percentile(values, [2.5, 50, 97.5])
        y = len(methods) - 1 - row
        ax.plot([lo, hi], [y, y], color=colour, linewidth=6, solid_capstyle="round", alpha=0.35)
        ax.scatter([mid], [y], s=90, color=colour, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(hi + 3, y, f"{lo:.0f} to {hi:.0f}", fontsize=10, color=INK_2, va="center")
    ax.axvline(truth, color=INK, linewidth=1.2, zorder=0)
    ax.text(truth + 1.5, len(methods) - 0.45, f"true break-even: EUR {truth:.0f}",
            fontsize=10.5, ha="left", va="center")
    ax.set_yticks(range(len(methods)), [m[0] for m in methods][::-1])
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=INK)
    ax.set_ylim(-0.6, len(methods) - 0.2)
    ax.set_xlim(100, 270)
    ax.set_xlabel("EUR per customer (dot: median; bar: middle 95% of snapshots)")
    ax.grid(axis="x", color=RULE, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    credit(fig)
    fig.savefig(OUT / "fig3_acquisition_budget.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: censoring that depends on the customer
# ---------------------------------------------------------------------------


def figure_informative():
    rng = np.random.default_rng(SEED + 11)
    grid = np.linspace(0, 36, 721)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    fig.subplots_adjust(left=0.08, right=0.78, top=0.80, bottom=0.12)
    header(fig, "More data does not fix censoring that depends on the customer",
           "Kaplan-Meier on 200,000 customers when a pause in the subscription is recorded "
           "as the end of the record.")
    ax.plot(grid, true_survival(grid), color=TRUTH, linewidth=6, solid_capstyle="round", zorder=1)
    curves = [(0.0, "pauses not recorded", VIOLET),
              (0.5, "pause rate 0.5 x churn rate", AQUA),
              (1.0, "pause rate 1 x churn rate", ORANGE)]
    for rate, label, colour in curves:
        c = simulate_customers(rng, 200_000, pause_rate=rate)
        km = kaplan_meier(c["time"], c["event"])
        ax.step(np.r_[0, km["time"]], np.r_[1, km["surv"]], where="post", color=colour,
                linewidth=1.8, zorder=3)
        y_end = float(km["surv"][km["time"] <= 36][-1])
        ax.text(36.8, y_end + (0.03 if rate == 1.0 else 0.0),
                label, color=colour, fontsize=10, ha="left", va="center")
    ax.text(36.8, true_survival(36) - 0.055, "truth", color=INK_2, fontsize=10, ha="left",
            va="center")
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Still subscribed")
    credit(fig)
    fig.savefig(OUT / "fig4_informative_censoring.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    figure_timelines()
    figure_curves()
    figure_budget()
    figure_informative()
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
