"""Figures for Part 3. Writes PNG files to ./figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cox_model import (CHANNELS, COLUMNS, HORIZON, MARGIN_ANNUAL, MARGIN_MONTHLY, RENEWAL, SIZES,
                       baseline_cumhaz, design, ph_test, predicted_survival, predicted_value,
                       run_all, true_survival, true_value)

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

MINUS = "−"


def header(fig, main, sub, y=0.95):
    fig.add_artist(plt.Line2D([0.04, 0.085], [y + 0.012, y + 0.012], color=VIOLET,
                              linewidth=3, transform=fig.transFigure))
    fig.text(0.04, y - 0.005, main, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.04, y - 0.005 - 0.36 / fig.get_figheight(), sub, fontsize=11, color=INK_2,
             ha="left", va="top")


def credit(fig, text="Simulated data. Figure by the author"):
    fig.text(0.96, 0.012, text, fontsize=8.5, color=INK_2, ha="right", va="bottom")


def signed(v, digits=0):
    return f"{'+' if v >= 0 else MINUS}{abs(v):.{digits}f}"


# ---------------------------------------------------------------------------
# Figure 1: one hazard ratio for annual billing bends both curves
# ---------------------------------------------------------------------------


def figure_misfit(tab, fits):
    X, time, event = tab["X"], tab["time"], tab["event"]
    base_all = baseline_cumhaz(fits["cox"], X, time, event)[0]
    base_s = baseline_cumhaz(fits["strat"], X[:, :4], time, event)
    grid = np.linspace(0, 24, 1201)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.12)
    header(fig, "One hazard ratio for annual billing bends both retention curves",
           "Search customers with 10 to 49 employees: true retention, and the fits of two Cox models "
           "to one simulated table of 8,000 customers.")
    for ax, a, title in ((axes[0], 1, "Annual billing: can only leave at renewal"),
                         (axes[1], 0, "Monthly billing: can leave at any time")):
        x = design(np.array([0]), np.array([1]), np.array([a]))[0]
        ax.plot(grid, true_survival(grid, 0, 1, a), color=TRUTH, linewidth=6,
                solid_capstyle="butt", zorder=1)
        ax.step(grid, predicted_survival(grid, x, base_all, fits["cox"]["beta"]), where="post",
                color=ORANGE, linewidth=2, zorder=2)
        ax.step(grid, predicted_survival(grid, x[:4], base_s[a], fits["strat"]["beta"]),
                where="post", color=VIOLET, linewidth=2, zorder=3)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("Months since signup")
        ax.set_ylim(0.4, 1.07)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    axes[0].set_ylabel("Still a customer")
    ax = axes[0]
    ax.text(0.3, 1.035, "true (grey): nobody leaves before month 12", color=INK_2, fontsize=10,
            va="center")
    ax.text(0.6, 0.79, "plan as a covariate:\nsmooth decline that\ncannot happen", color=ORANGE,
            fontsize=10, va="center")
    ax.text(18.0, 0.70, "stratified by plan", color=VIOLET, fontsize=10, va="center", ha="center")
    ax = axes[1]
    ax.text(13.0, 0.86, "plan as a covariate:\nextra drops at months\n12 and 24", color=ORANGE,
            fontsize=10, va="center")
    ax.text(1.0, 0.56, "true (grey) and\nstratified (violet)\nalmost coincide", color=VIOLET,
            fontsize=10, va="center")
    credit(fig)
    fig.savefig(OUT / "fig2_annual_billing_misfit.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: scaled Schoenfeld residuals
# ---------------------------------------------------------------------------


def figure_schoenfeld(tab, fits):
    X, time, event = tab["X"], tab["time"], tab["event"]
    test = ph_test(fits["cox"], X, time, event)
    t, sc = test["times"], test["scaled"]
    renew = np.isclose(t, 12.0) | np.isclose(t, 24.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.22)
    header(fig, "Residuals that should be flat over time, and one column that is not",
           "Coefficient plus scaled Schoenfeld residual, Cox model with billing plan as a covariate, "
           "averaged over two-month bands of churn time.")
    for ax, j, title in ((axes[0], 4, "Annual billing"), (axes[1], 0, "Social channel")):
        beta = fits["cox"]["beta"][j]
        edges = np.arange(0.0, 37.0, 2.0)
        mids, means, half = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            k = (t >= a) & (t < b) & ~renew
            if k.sum() < 10:
                continue
            mids.append((a + b) / 2)
            means.append(sc[k, j].mean())
            half.append(1.959964 * sc[k, j].std(ddof=1) / np.sqrt(k.sum()))
        ax.axhline(beta, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.errorbar(mids, means, yerr=half, fmt="o", color=VIOLET, markersize=6, elinewidth=1.4,
                    capsize=0, zorder=3)
        for m in (12.0, 24.0):
            k = np.isclose(t, m)
            if k.sum():
                v = sc[k, j].mean()
                h = 1.959964 * sc[k, j].std(ddof=1) / np.sqrt(k.sum())
                ax.errorbar([m], [v], yerr=[h], fmt="D", color=ORANGE, markersize=8,
                            elinewidth=1.6, capsize=0, zorder=4)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK)
        ax.set_xlim(0, 36)
        ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
        ax.set_xlabel("Month of churn")
    axes[0].set_ylabel("Log hazard ratio implied at that time")
    ax = axes[0]
    ax.set_ylim(-3.5, 7.5)
    ax.text(18.0, 4.7, "renewal months (diamonds):\nannual customers leave in bulk", color=ORANGE,
            fontsize=10, va="center", ha="center")
    ax.text(0.8, -3.05, "other months: annual customers cannot leave", color=VIOLET, fontsize=10,
            va="center")
    ax.text(35.5, fits["cox"]["beta"][4] + 0.35, f"dashed line: fitted constant {signed(fits['cox']['beta'][4], 2)}",
            color=INK_2, fontsize=10, ha="right")
    ax = axes[1]
    ax.set_ylim(-2.2, 2.5)
    ax.text(0.8, -1.9, f"dashed line: fitted constant {signed(fits['cox']['beta'][0], 2)}",
            color=INK_2, fontsize=10, va="center")
    ax.text(1.0, 2.35, "no trend: the hazard ratio of social customers\nis the same early and late",
            color=VIOLET, fontsize=10, va="top")
    credit(fig)
    fig.savefig(OUT / "fig1_schoenfeld_residuals.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: the same hazard ratio is worth different amounts
# ---------------------------------------------------------------------------


def figure_months(tab, fits):
    X, time, event = tab["X"], tab["time"], tab["event"]
    base = baseline_cumhaz(fits["strat"], X[:, :4], time, event)
    beta = fits["strat"]["beta"]
    rows = []
    for a in (0, 1):
        for s in range(3):
            x_s = design(np.array([0]), np.array([s]), np.array([a]))[0][:4]
            x_r = design(np.array([2]), np.array([s]), np.array([a]))[0][:4]
            est = predicted_value(x_r, base[a], beta, a) - predicted_value(x_s, base[a], beta, a)
            tru = true_value(2, s, a) - true_value(0, s, a)
            rate = MARGIN_ANNUAL / RENEWAL if a else MARGIN_MONTHLY
            rows.append((f"{'annual' if a else 'monthly'}, {SIZES[s]} employees", tru, est, tru / rate))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.27, right=0.80, top=0.78, bottom=0.13)
    header(fig, "One hazard ratio, four-fold differences in value",
           "Extra margin over 24 months from a referral rather than a search customer. "
           "True hazard ratio: 0.70 in every row.")
    ypos = np.arange(len(rows))[::-1].astype(float)
    ypos[:3] += 0.6
    for y, (label, tru, est, months) in zip(ypos, rows):
        ax.plot([0, tru], [y, y], color=RULE, linewidth=6, solid_capstyle="butt", zorder=1)
        ax.scatter([tru], [y], s=140, color=TRUTH, zorder=2, edgecolor=SURFACE, linewidth=1.2)
        ax.scatter([est], [y], s=55, color=VIOLET, zorder=3, edgecolor=SURFACE, linewidth=1.0)
        ax.text(41.5, y, f"EUR {tru:.2f}\n({months:.2f} paid months)", color=INK_2, fontsize=9.5,
                va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color=RULE, linewidth=0.7)
    ax.set_xlim(0, 40)
    ax.set_xlabel("EUR per customer, first 24 months")
    ax.text(41.5, ypos[0] + 0.85, "true value", color=INK_2, fontsize=9.5, va="center",
            fontweight="bold")
    ax.scatter([0.8], [ypos[-1] - 0.85], s=140, color=TRUTH, clip_on=False)
    ax.text(1.6, ypos[-1] - 0.85, "true", color=INK_2, fontsize=9.5, va="center")
    ax.scatter([6.3], [ypos[-1] - 0.85], s=55, color=VIOLET, clip_on=False)
    ax.text(7.1, ypos[-1] - 0.85, "estimated from one table (stratified Cox model)", color=VIOLET,
            fontsize=9.5, va="center")
    ax.set_ylim(ypos[-1] - 1.3, ypos[0] + 0.6)
    credit(fig)
    fig.savefig(OUT / "fig3_value_of_one_hazard_ratio.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: errors in segment values across repeated tables
# ---------------------------------------------------------------------------


def figure_segments(reps):
    errors = reps[1]
    fig, ax = plt.subplots(figsize=(11, 7.4))
    fig.subplots_adjust(left=0.2, right=0.78, top=0.82, bottom=0.1)
    header(fig, "Pooling across segments halves the error in the thinnest segment",
           "Error in the value of a customer from each lead segment, 500 simulated tables: "
           "middle 95% of errors (line) and mean error (dot).")
    methods = (("km", AQUA, "Kaplan-Meier per segment\nand plan"),
               ("cox", ORANGE, "Cox, plan as a covariate"),
               ("strat", VIOLET, "Cox, stratified by plan"))
    labels, ypos = [], []
    y = 0.0
    for c in range(3):
        for s in range(3):
            for k, (name, colour, _) in enumerate(methods):
                e = errors[name][:, c, s]
                lo, hi = np.percentile(e, [2.5, 97.5])
                yy = y - 0.26 * (k - 1)
                ax.plot([lo, hi], [yy, yy], color=colour, linewidth=2.2, solid_capstyle="round")
                ax.scatter([e.mean()], [yy], s=30, color=colour, zorder=3, edgecolor=SURFACE,
                           linewidth=0.8)
            labels.append(f"{CHANNELS[c]}, {SIZES[s]}")
            ypos.append(y)
            y -= 1.0
        y -= 0.35
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color=RULE, linewidth=0.7)
    ax.set_xlim(-21, 21)
    ax.set_xlabel("Estimated minus true value per customer (EUR)")
    for k, (name, colour, label) in enumerate(methods):
        ax.text(22.5, ypos[0] + 0.3 - 0.75 * k, label, color=colour, fontsize=10, va="top")
    credit(fig)
    fig.savefig(OUT / "fig4_segment_value_errors.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    tab, fits, reps = run_all()
    figure_misfit(tab, fits)
    figure_schoenfeld(tab, fits)
    figure_months(tab, fits)
    figure_segments(reps)
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
