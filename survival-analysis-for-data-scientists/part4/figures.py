"""Figures for Part 4. Writes PNG files to ./figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extrapolation import (MODELS, RULES, TIME_RATIO, TRUTHS, WINDOWS, analyse_table, kaplan_meier,
                           km_at, run_all, survival, true_value)

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
WINDOW_FILL = "#efeee9"

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
PERCENT = plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")


def header(fig, main, sub, y=0.95):
    fig.add_artist(plt.Line2D([0.04, 0.085], [y + 0.012, y + 0.012], color=VIOLET,
                              linewidth=3, transform=fig.transFigure))
    fig.text(0.04, y - 0.005, main, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.04, y - 0.005 - 0.36 / fig.get_figheight(), sub, fontsize=11, color=INK_2,
             ha="left", va="top")


def credit(fig, text="Simulated data. Figure by the author"):
    fig.text(0.96, 0.012, text, fontsize=8.5, color=INK_2, ha="right", va="bottom")


def eur(v):
    return f"EUR {v:.0f}"


# ---------------------------------------------------------------------------
# Figure 1: six models, one year of data
# ---------------------------------------------------------------------------


def figure_six_models(tab):
    res = analyse_table(tab, 12.0)
    x = tab["referral"]
    km = kaplan_meier(tab["time"][x == 0], tab["event"][x == 0])
    grid = np.linspace(0, 60, 601)
    obs = np.linspace(0, 12, 400)
    truth = TRUTHS["gamma mixture"](grid)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.8, bottom=0.09, wspace=0.08, hspace=0.32)
    header(fig, "Six models fitted to one year of data, extrapolated to five",
           "Paid-channel customers in one simulated table at 12 months: Kaplan-Meier (black), each model "
           "(violet), true retention (grey).")
    for ax, (i, m) in zip(axes.ravel(), enumerate(MODELS)):
        ax.axvspan(0, 12, color=WINDOW_FILL, zorder=0, linewidth=0)
        ax.plot(grid, truth, color=TRUTH, linewidth=5, solid_capstyle="butt", zorder=1)
        ax.plot(grid, survival(res["fits"][m], grid), color=VIOLET, linewidth=2, zorder=2)
        ax.step(obs, km_at(km, obs), where="post", color=INK, linewidth=1.3, zorder=3)
        ax.set_title(m, loc="left", fontsize=11.5, color=INK)
        ax.text(59, 0.93, f"{chr(0x0394)}AIC {res['delta'][i]:.1f}\n60-month value {eur(res['values'][i])}",
                ha="right", va="top", fontsize=9.5, color=VIOLET if res["delta"][i] <= 10 else INK_2)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 12, 24, 36, 48, 60])
        ax.yaxis.set_major_formatter(PERCENT)
    for ax in axes[1]:
        ax.set_xlabel("Months since signup")
    for ax in axes[:, 0]:
        ax.set_ylabel("Still a customer")
    axes[0, 0].text(6, 0.08, "observed\nwindow", ha="center", fontsize=9.5, color=INK_2)
    fig.text(0.97, 0.835, f"true 60-month value {eur(true_value('gamma mixture'))}", ha="right",
             fontsize=10, color=INK_2)
    credit(fig)
    fig.savefig(OUT / "fig1_six_models.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: a constant time ratio and a drifting hazard ratio
# ---------------------------------------------------------------------------


def figure_time_ratio(tab):
    res = analyse_table(tab, 12.0)
    gm = res["fits"]["gamma mixture"]
    ratio = float(np.exp(gm["theta"][-1]))
    time, event, x = tab["time"], tab["event"], tab["referral"]
    obs = np.linspace(0, 12, 400)
    km0 = kaplan_meier(time[x == 0], event[x == 0])
    km1 = kaplan_meier(time[x == 1], event[x == 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.25)
    header(fig, "Referral stretches time: a constant time ratio, a drifting hazard ratio",
           "Every referral customer churns at 0.7 times the rate they otherwise would: lifetimes "
           f"are {TIME_RATIO:.2f} times as long.")
    ax = axes[0]
    ax.step(obs, km_at(km0, obs), where="post", color=AQUA, linewidth=2)
    ax.step(obs, km_at(km1, obs), where="post", color=VIOLET, linewidth=2)
    ax.plot(obs, km_at(km0, obs / ratio), color=ORANGE, linewidth=1.6, linestyle=(0, (4, 2)))
    ax.text(11.8, float(km_at(km0, 12.0)) - 0.03, "paid channels", color=AQUA, ha="right", va="top")
    ax.text(11.8, float(km_at(km1, 12.0)) + 0.045, "referral", color=VIOLET, ha="right")
    ax.text(0.4, 0.62, f"paid-channel curve with time\nstretched by the estimated\nratio {ratio:.2f} (dashed)",
            color=ORANGE, fontsize=10, va="top")
    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Still a customer (Kaplan-Meier)")
    ax.set_title("Observed retention, one simulated table", loc="left", fontsize=11.5, color=INK)

    ax = axes[1]
    grid = np.linspace(0, 60, 601)
    S = TRUTHS["gamma mixture"]
    eps = 1e-6
    h0 = (S(grid) - S(grid + eps)) / eps / S(grid)
    h1 = (S(grid / TIME_RATIO) - S((grid + eps) / TIME_RATIO)) / eps / S(grid / TIME_RATIO)
    ax.axhline(0.7, color=VIOLET, linewidth=1.6)
    ax.plot(grid, h1 / h0, color=TRUTH, linewidth=5, solid_capstyle="butt")
    ax.text(59, 0.715, "each customer: 0.70", color=VIOLET, ha="right", va="bottom", fontsize=10)
    ax.text(59, float((h1 / h0)[-1]) - 0.015, "referral against paid,\nwhole population",
            color=INK_2, ha="right", va="top", fontsize=10)
    ax.set_xlim(0, 60)
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Hazard ratio")
    ax.set_title("The same effect as a hazard ratio (true values)", loc="left", fontsize=11.5, color=INK)
    credit(fig)
    fig.savefig(OUT / "fig2_time_ratio.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: three businesses with the same first year
# ---------------------------------------------------------------------------


def figure_three_truths():
    grid = np.linspace(0.0, 60, 1201)
    colours = {"gamma mixture": VIOLET, "Weibull": ORANGE, "loyal core": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.42)
    header(fig, "Three businesses with the same first year",
           "True retention and hazard of paid-channel customers under three processes "
           "that differ by less than half a point in months 0 to 12.")
    ax = axes[0]
    ax.axvspan(0, 12, color=WINDOW_FILL, zorder=0, linewidth=0)
    for name, S in TRUTHS.items():
        ax.plot(grid, S(grid), color=colours[name], linewidth=2.2)
        ax.text(61.5, float(S(60.0)) + {"gamma mixture": 0.015, "Weibull": -0.012}.get(name, 0.0),
                f"{name}\n{eur(true_value(name))}", color=colours[name], ha="left", va="center",
                fontsize=10, clip_on=False, linespacing=1.1)
    ax.text(6, 0.08, "first\nyear", ha="center", color=INK_2, fontsize=10)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Still a customer")
    ax.set_title("Retention, and value over 60 months", loc="left", fontsize=11.5, color=INK)
    ax = axes[1]
    ax.axvspan(0, 12, color=WINDOW_FILL, zorder=0, linewidth=0)
    g = grid[1:]
    for name, S in TRUTHS.items():
        h = (S(g) - S(g + 1e-6)) / 1e-6 / S(g)
        ax.plot(g, h, color=colours[name], linewidth=2.2)
    ax.text(40, 0.045, "Weibull", color=ORANGE, fontsize=10)
    ax.text(40, 0.024, "gamma mixture", color=VIOLET, fontsize=10)
    ax.text(25, 0.006, "loyal core", color=AQUA, fontsize=10)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 0.1)
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Churn per month among those still subscribed")
    ax.set_title("Hazard", loc="left", fontsize=11.5, color=INK)
    credit(fig)
    fig.savefig(OUT / "fig3_three_businesses.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: decision rules by window
# ---------------------------------------------------------------------------


def figure_rules(reps):
    rules = (("best AIC", VIOLET), ("AIC average", AQUA), ("observed + lowest tail", ORANGE))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 6.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.17, wspace=0.08)
    header(fig, "Every rule is unreliable after one year; waiting fixes two of the three businesses",
           "Error in the 60-month value of a paid-channel customer across 100 simulated tables per "
           "business and window: middle 95% (line), mean (dot).")
    for ax, truth in zip(axes, TRUTHS):
        tv = true_value(truth)
        ax.axhline(0, color=INK_2, linewidth=0.9)
        for k, (rule, colour) in enumerate(rules):
            for w in WINDOWS:
                err = reps[(truth, w)]["rules"][rule] - tv
                lo, hi = np.percentile(err, [2.5, 97.5])
                xx = w + (k - 1) * 2.2
                ax.plot([xx, xx], [lo, hi], color=colour, linewidth=2.4, solid_capstyle="round")
                ax.scatter([xx], [err.mean()], s=36, color=colour, zorder=3, edgecolor=SURFACE,
                           linewidth=0.8)
        ax.set_title(f"true process: {truth}", loc="left", fontsize=11.5, color=INK)
        ax.set_xticks(WINDOWS)
        ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS])
        ax.set_xlim(7, 41)
        ax.set_xlabel("Months of data")
        ax.set_ylim(-140, 100)
    axes[0].set_ylabel("Estimated minus true value (EUR)")
    for k, (rule, colour) in enumerate(rules):
        fig.text(0.08 + 0.25 * k, 0.035, rule, color=colour, fontsize=10.5)
    credit(fig)
    fig.savefig(OUT / "fig4_decision_rules.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    tab, reps = run_all()
    figure_six_models(tab)
    figure_time_ratio(tab)
    figure_three_truths()
    figure_rules(reps)
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
