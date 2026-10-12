"""Figures for Part 2. Writes PNG files to ./figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hazards_logrank import (ALPHA, BETA, HORIZON, LOOKS, MARGIN, SEED, kaplan_meier,
                             nelson_aalen, observe, occurrence_exposure, run_all,
                             simulate_experiment, true_hazard, true_rmst, true_survival)

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
    fig.add_artist(plt.Line2D([0.04, 0.085], [y + 0.012, y + 0.012], color=VIOLET,
                              linewidth=3, transform=fig.transFigure))
    fig.text(0.04, y - 0.005, main, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.04, y - 0.005 - 0.36 / fig.get_figheight(), sub, fontsize=11, color=INK_2,
             ha="left", va="top")


def credit(fig, text="Simulated data. Figure by the author"):
    fig.text(0.96, 0.012, text, fontsize=8.5, color=INK_2, ha="right", va="bottom")


def example_experiment():
    rng = np.random.default_rng(SEED)
    return simulate_experiment(rng, "steady")


# ---------------------------------------------------------------------------
# Figure 1: cumulative hazard and monthly churn rate
# ---------------------------------------------------------------------------


def figure_hazards():
    time, event, arm = observe(example_experiment(), 12.0)
    grid = np.linspace(0, 12, 400)
    edges = np.arange(0.0, 13.0, 1.0)
    mids = edges[:-1] + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.25)
    header(fig, "Two views of the same churn: accumulated, and month by month",
           "One experiment with 2,250 customers per arm, analysed 12 months after it started. "
           "Thick grey lines: the true values.")
    ax = axes[0]
    for a, colour, name in ((0, AQUA, "control"), (1, VIOLET, "new onboarding")):
        truth_arm = "control" if a == 0 else "steady"
        ax.plot(grid, ALPHA * np.log1p((0.8 if a else 1.0) * grid / BETA), color=TRUTH,
                linewidth=5, solid_capstyle="round", zorder=1)
        na = nelson_aalen(time[arm == a], event[arm == a])
        ax.step(np.r_[0, na["time"]], np.r_[0, na["cumhaz"]], where="post", color=colour,
                linewidth=1.8, zorder=3)
        ax.text(11.8, float(na["cumhaz"][-1]) + (0.05 if a == 0 else -0.11), name,
                color=colour, fontsize=10.5, va="center", ha="right")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 0.7)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel("Months since signup")
    ax.set_title("Nelson-Aalen cumulative hazard", loc="left", fontsize=11.5, color=INK)

    ax = axes[1]
    for a, colour, shift in ((0, AQUA, -0.12), (1, VIOLET, 0.12)):
        truth_arm = "control" if a == 0 else "steady"
        ax.plot(grid, true_hazard(grid, truth_arm), color=TRUTH, linewidth=5,
                solid_capstyle="round", zorder=1)
        rates, lo, hi = occurrence_exposure(time[arm == a], event[arm == a], edges)
        ax.errorbar(mids + shift, rates, yerr=[rates - lo, hi - rates], fmt="o", color=colour,
                    markersize=6, elinewidth=1.4, capsize=0, zorder=3)
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0, 0.1)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Months since signup")
    ax.set_title("Churn per customer-month, by month (95% intervals)", loc="left",
                 fontsize=11.5, color=INK)
    ax.text(11.0, 0.099, "fewer customers have reached\nthe later months: wider intervals",
            fontsize=9.5, color=INK_2, ha="right", va="top")
    credit(fig)
    fig.savefig(OUT / "fig1_hazards.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: the population hazard ratio drifts
# ---------------------------------------------------------------------------


def figure_hazard_ratio():
    rng = np.random.default_rng(SEED + 5)
    n = 400_000
    rate = rng.gamma(ALPHA, 1.0 / BETA, n)
    life = rng.exponential(1.0, n) / rate
    arm = rng.integers(0, 2, n)
    life = np.where(arm == 1, life / 0.8, life)
    time = np.minimum(life, 36.0)
    event = (life <= 36.0).astype(int)
    edges = np.arange(0.0, 37.0, 3.0)
    mids = edges[:-1] + 1.5
    r0, _, _ = occurrence_exposure(time[arm == 0], event[arm == 0], edges)
    r1, _, _ = occurrence_exposure(time[arm == 1], event[arm == 1], edges)
    grid = np.linspace(0, 36, 400)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.09, right=0.80, top=0.78, bottom=0.13)
    header(fig, "Every customer churns 20% more slowly, yet the hazard ratio drifts towards 1",
           "Hazard ratio by months since signup, 400,000 simulated customers followed "
           "for 36 months.")
    ax.axhline(0.8, color=ORANGE, linewidth=1.6)
    ax.text(36.6, 0.8, "each customer's\nhazard ratio: 0.80", color=ORANGE, fontsize=10.5,
            va="center")
    ax.plot(grid, true_hazard(grid, "steady") / true_hazard(grid), color=TRUTH, linewidth=5,
            solid_capstyle="round")
    ax.scatter(mids, r1 / r0, s=55, color=VIOLET, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    ax.text(36.6, float(true_hazard(36, "steady") / true_hazard(36)),
            "population hazard\nratio: true (grey)\nand estimated by\nquarter (dots)",
            color=INK_2, fontsize=10.5, va="center")
    ax.axhline(1.0, color=INK_2, linewidth=0.9)
    ax.set_xlim(0, 36)
    ax.set_ylim(0.7, 1.05)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Hazard ratio, new onboarding / control")
    credit(fig)
    fig.savefig(OUT / "fig2_hazard_ratio_drift.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: power by look
# ---------------------------------------------------------------------------


def figure_power(reps):
    looks = np.array(LOOKS)

    def rate(name, key):
        return [np.mean([r[key] < 0.05 for r in reps[name][look]]) for look in LOOKS]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    fig.subplots_adjust(left=0.09, right=0.75, top=0.78, bottom=0.13)
    header(fig, "The log-rank test keeps learning; the month-1 comparison stops",
           "Share of 1,000 simulated experiments with a significant result (5% level), "
           "by the month at which the experiment is analysed.")
    series = [("steady", "lr_p", VIOLET, "o", "log-rank, real effect"),
              ("steady", "m1_p", ORANGE, "o", "month-1 test, real effect"),
              ("control", "lr_p", VIOLET, "s", "log-rank, no effect"),
              ("control", "m1_p", ORANGE, "s", "month-1 test, no effect")]
    for name, key, colour, marker, label in series:
        y = rate(name, key)
        hollow = name == "control"
        ax.plot(looks, y, color=colour, linewidth=1.2 if hollow else 2.2,
                linestyle=(0, (3, 2)) if hollow else "-")
        ax.scatter(looks, y, s=55, marker=marker, zorder=3, linewidth=1.5,
                   facecolor=SURFACE if hollow else colour, edgecolor=colour)
        dy = {"log-rank, no effect": 0.03, "month-1 test, no effect": -0.03}.get(label, 0.0)
        ax.text(12.4, y[-1] + dy, label, color=colour, fontsize=10.5, va="center")
    ax.axhline(0.05, color=INK_2, linewidth=0.8)
    ax.set_xlim(2.5, 12)
    ax.set_xticks(LOOKS)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Months after the experiment started (enrolment runs for the first 3)")
    ax.set_ylabel("Experiments with a significant result")
    credit(fig)
    fig.savefig(OUT / "fig3_power_by_look.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: crossing curves
# ---------------------------------------------------------------------------


def figure_crossing(reps):
    grid = np.linspace(0, 12, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.3)
    header(fig, "Halving early churn can still lose customer-months",
           "A treatment that halves churn for two months and raises it by half afterwards. "
           "Left: true retention. Right: what 1,000 experiments report.")
    ax = axes[0]
    ax.plot(grid, true_survival(grid), color=AQUA, linewidth=2.4)
    ax.plot(grid, true_survival(grid, "delay"), color=VIOLET, linewidth=2.4)
    ax.fill_between(grid, true_survival(grid), true_survival(grid, "delay"),
                    where=true_survival(grid, "delay") >= true_survival(grid),
                    color=VIOLET, alpha=0.15, linewidth=0)
    ax.fill_between(grid, true_survival(grid), true_survival(grid, "delay"),
                    where=true_survival(grid, "delay") < true_survival(grid),
                    color=ORANGE, alpha=0.2, linewidth=0)
    ax.text(11.8, float(true_survival(12)) + 0.065, "control", color=AQUA, fontsize=10.5,
            va="center", ha="right")
    ax.text(11.8, float(true_survival(12, "delay")) - 0.03, "treatment", color=VIOLET,
            fontsize=10.5, va="center", ha="right")
    diff = true_rmst(HORIZON, "delay") - true_rmst(HORIZON)
    ax.text(0.4, 0.62, f"area gained before month 4\nis smaller than the area lost after:\n"
            f"{diff:+.2f} paid months per customer\nin the first 12 (EUR {MARGIN * diff:+.2f})",
            fontsize=10, color=INK, va="top")
    ax.set_xlim(0, 12)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Still subscribed")

    ax = axes[1]
    looks = np.array(LOOKS)
    m1 = [np.mean([(r["m1_p"] < 0.05) and (r["m1_diff"] < 0) for r in reps["delay"][lk]])
          for lk in LOOKS]
    fav = [np.mean([(r["lr_p"] < 0.05) and (r["lr_z"] < 0) for r in reps["delay"][lk]])
           for lk in LOOKS]
    ag = [np.mean([(r["lr_p"] < 0.05) and (r["lr_z"] > 0) for r in reps["delay"][lk]])
          for lk in LOOKS]
    for y, colour, style, label, dy in (
            (m1, ORANGE, "-", "month-1 retention test\nfavours the treatment", -0.02),
            (fav, VIOLET, "-", "log-rank favours\nthe treatment", 0.0),
            (ag, VIOLET, (0, (3, 2)), "log-rank finds it\nharmful", 0.0)):
        ax.plot(looks, y, color=colour, linestyle=style, linewidth=2)
        ax.scatter(looks, y, s=45, color=colour, zorder=3)
    ax.text(4.6, 0.95, "month-1 retention test\nfavours the treatment", color=ORANGE,
            fontsize=10, va="top")
    ax.text(3.1, 0.40, "log-rank\nfavours it", color=VIOLET, fontsize=10, va="center")
    ax.text(10.6, 0.55, "log-rank finds\nit harmful", color=VIOLET, fontsize=10, va="center",
            ha="center")
    ax.set_xlim(2.5, 12.5)
    ax.set_xticks(LOOKS)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Months after the experiment started")
    ax.set_ylabel("Experiments with a significant result")
    credit(fig)
    fig.savefig(OUT / "fig4_crossing_hazards.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    _, reps = run_all()
    figure_hazards()
    figure_hazard_ratio()
    figure_power(reps)
    figure_crossing(reps)
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
