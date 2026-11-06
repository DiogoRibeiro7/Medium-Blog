"""Figures for the article, from ``results/``.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import fit, km_at, kaplan_meier
from simulate import Q95, latency_tail, simulate_dataset

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP3 = ["#86b6ef", "#2a78d6", "#0d366b"]  # ordinal: small -> large sample
INK, INK2, MUTED, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea"
CREDIT = "Diogo Ribeiro"
LAT_NAMES = {"lognormal-0.5": "lognormal, light tail", "lognormal-1.0": "lognormal, heavy tail", "weibull-1.5": "Weibull, shape 1.5"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    }
)


def note(fig, text: str, y: float = 0.005) -> None:
    fig.text(0.01, y, f"{text}   |   {CREDIT}", ha="left", va="bottom", fontsize=8.2, color=MUTED)


def suptitle(fig, text: str, y: float = 1.02) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=y)


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pct(ax, axis="y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def log_ratio_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([0.25, 0.5, 1, 2, 3])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "3"])


def fig1_one_window_many_answers() -> None:
    """One simulated dataset: what the window shows, what each model says beyond it."""
    rng = np.random.default_rng(20261106)
    pi_true, lat, ratio = 0.3, "lognormal-1.0", 0.5
    t, e = simulate_dataset(pi_true, lat, 1000, ratio, rng)
    tau = ratio * Q95
    grid = np.linspace(0.05, 4 * Q95, 400)
    ln, wb = fit(t, e, "lognormal", True), fit(t, e, "weibull", True)
    ln0 = fit(t, e, "lognormal", False)
    true_curve = pi_true + (1 - pi_true) * np.array([latency_tail(lat, g) for g in grid])
    times, surv = kaplan_meier(t, e)
    bound = float(km_at(t, e, [np.sort(t)[-10]])[0])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axvspan(tau, grid.max(), color=SHADE, lw=0, zorder=0)
    ax.text(tau + 0.6, 0.98, "beyond the longest follow-up", va="top", fontsize=9, color=INK2)
    ax.step(np.r_[0, times], np.r_[1, surv], where="post", color=INK, lw=2, label="observed (Kaplan-Meier)", zorder=5)
    ax.plot(grid, true_curve, color=MUTED, lw=1.6, ls=(0, (2, 2)), label=f"truth: {pi_true:.0%} never have the event", zorder=2)
    ax.plot(grid, ln.survival(grid), color=BLUE, lw=2, label=f"cure model, lognormal latency: {ln.pi:.0%}", zorder=3)
    ax.plot(grid, wb.survival(grid), color=ORANGE, lw=2, label=f"cure model, Weibull latency: {wb.pi:.0%}", zorder=3)
    ax.plot(grid, ln0.survival(grid), color=AQUA, lw=2, ls=(0, (5, 2)), label="no-cure lognormal: 0%", zorder=3)
    x_br = grid.max() * 0.985
    ax.plot([x_br, x_br], [0, bound], color=INK, lw=3, solid_capstyle="butt", zorder=6)
    ax.annotate(f"what the data alone\nsay: between 0%\nand {bound:.0%}", (x_br, bound / 2), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=9, color=INK, bbox={"boxstyle": "square,pad=0.2", "fc": SHADE, "ec": "none"})
    ax.set_xlim(0, grid.max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time (the 95th percentile of the event times of units that do have the event is 10)")
    ax.set_ylabel("share without the event")
    pct(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9.2)
    suptitle(fig, "One window, three answers: what a cure model reports depends on its latency distribution")
    fig.tight_layout()
    note(fig, "One simulated dataset: n = 1,000, 30% never have the event, lognormal latency, follow-up up to half of the 95th percentile.", y=-0.12)
    save(fig, "fig1_one_window_many_answers.png")


def fig2_failure_by_follow_up() -> None:
    c = pd.read_csv(RES / "sim_cells.csv")
    c = c[c.n == 1000]
    series = [("km_plateau_fail", "Kaplan-Meier plateau", INK, "-"), ("pi_ln_fail", "cure model, lognormal latency", BLUE, "-"), ("pi_wb_fail", "cure model, Weibull latency", ORANGE, "-"), ("pi_best_fail", "cure model chosen by AIC", AQUA, (0, (4, 2)))]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharex=True, sharey=True)
    for j, lat in enumerate(LAT_NAMES):
        for i, pi in enumerate((0.2, 0.5)):
            ax = axes[i, j]
            g = c[(c.latency == lat) & (c.pi == pi)].sort_values("ratio")
            for col, lab, colr, ls in series:
                ax.plot(g.ratio, g[col], color=colr, ls=ls, marker="o", ms=4.5, mec=SURFACE, mew=1, label=lab)
            ax.set_title(f"{LAT_NAMES[lat]}, {pi:.0%} cured", fontsize=10)
            log_ratio_axis(ax)
            ax.set_ylim(-0.03, 1.03)
            pct(ax)
    fig.supxlabel("follow-up ratio: longest follow-up divided by the 95th percentile of the event times", fontsize=10, color=INK2, y=0.07)
    fig.supylabel("share of datasets with an error above 0.10", fontsize=10, color=INK2, x=0.005)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9.2)
    suptitle(fig, "With short follow-up, only the right latency family recovers the cure fraction, and not always")
    fig.tight_layout(rect=(0.01, 0.08, 1, 1))
    note(fig, "200 simulated datasets per point, n = 1,000; staggered entry, so individual follow-up is uniform up to the longest.", y=-0.06)
    save(fig, "fig2_failure_by_follow_up.png")


def fig3_coverage() -> None:
    c = pd.read_csv(RES / "sim_cells.csv")
    c = c[(c.latency == "lognormal-1.0") & (c.pi == 0.2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (col, title) in zip(axes, [("ln_cover", "Correct latency family (lognormal)"), ("wb_cover", "Wrong latency family (Weibull)")]):
        ax.axhline(0.9, color=MUTED, lw=1, ls=(0, (3, 3)))
        for n, colr in zip((250, 1000, 4000), RAMP3):
            g = c[c.n == n].sort_values("ratio")
            ax.plot(g.ratio, g[col], color=colr, marker="o", ms=5, mec=SURFACE, mew=1, label=f"n = {n:,}")
        ax.set_title(title)
        log_ratio_axis(ax)
        ax.set_xlim(0.22, 4.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("follow-up ratio")
        pct(ax)
    axes[0].set_ylabel("share of 90% intervals containing the truth")
    axes[1].text(0.24, 0.92, "nominal 90%", va="bottom", fontsize=8.8, color=INK2)
    axes[1].legend(loc="center left", bbox_to_anchor=(0.02, 0.55), title="sample size", title_fontsize=9)
    suptitle(fig, "With the wrong latency family, more data buys narrower intervals around the wrong value")
    fig.tight_layout()
    note(fig, "Wald intervals on the logit scale; data with heavy-tailed lognormal latency and 20% cured; 200 datasets per point.", y=-0.04)
    save(fig, "fig3_coverage.png")


def fig4_checks_by_follow_up() -> None:
    c = pd.read_csv(RES / "sim_cells.csv")
    c = c[c.n == 1000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    lat_colors = {"lognormal-0.5": BLUE, "lognormal-1.0": ORANGE, "weibull-1.5": AQUA}
    for ax, (col, title) in zip(axes, [("mz_sufficient", "Maller-Zhou test says follow-up is sufficient"), ("cure_preferred", "AIC prefers a cure model")]):
        for lat, colr in lat_colors.items():
            g = c[(c.latency == lat)].groupby("ratio")[col].mean()
            ax.plot(g.index, g.values, color=colr, marker="o", ms=5, mec=SURFACE, mew=1, label=LAT_NAMES[lat])
        ax.set_title(title)
        log_ratio_axis(ax)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("follow-up ratio")
        pct(ax)
    axes[0].set_ylabel("share of datasets")
    axes[1].legend(loc="lower right", title="latency of the simulated data", title_fontsize=9)
    suptitle(fig, "A sufficient-follow-up test that says yes half the time when follow-up is far too short")
    fig.tight_layout()
    note(fig, "n = 1,000; averages over cure fractions of 20% and 50%; 400 datasets per point.", y=-0.04)
    save(fig, "fig4_checks_by_follow_up.png")


def fig5_rules() -> None:
    r = pd.read_csv(RES / "sim_rules.csv")
    order = list(r.rule)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    y = np.arange(len(order))[::-1]
    for yi, (_, row) in zip(y, r.iterrows()):
        ax.plot([row.fail_if_pass, row.fail_if_flagged], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.plot([row.fail_if_pass], [yi], "o", ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=3, label="estimate passes the check" if yi == y[0] else None)
        ax.plot([row.fail_if_flagged], [yi], "o", ms=10, color=ORANGE, mec=SURFACE, mew=2, zorder=3, label="estimate is flagged" if yi == y[0] else None)
        ax.annotate(f"passes {row.pass_share:.0%}, flags {row.failures_flagged:.0%} of bad estimates", (max(row.fail_if_pass, row.fail_if_flagged), yi), xytext=(12, 0), textcoords="offset points", va="center", fontsize=8.8, color=INK2)
    raw = pd.read_csv(RES / "sim_raw.csv", usecols=["pi", "pi_best"])
    overall = float(((raw.pi_best - raw.pi).abs() > 0.10).mean())
    ax.axvline(overall, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(overall + 0.005, len(order) - 0.45, "all datasets", fontsize=8.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9.5)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of estimates off by more than 0.10")
    pct(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9.2)
    suptitle(fig, "Two cheap checks together separate trustworthy cure-fraction estimates from the rest", y=1.09)
    fig.tight_layout()
    note(fig, "All 25,200 simulated datasets; the estimate is the cure model (lognormal or Weibull latency) chosen by AIC.", y=-0.05)
    save(fig, "fig5_rules.png")


def fig6_eu_busts() -> None:
    a = pd.read_csv(RES / "application.csv")
    panels = [("gva", "relative", "Value added, relative to other member states"), ("emp", "relative", "Employment, relative to other member states")]
    bands = ["<0.30", "0.30-0.50", "0.50-0.80", ">0.80"]
    names = {"<0.30": "under 0.30", "0.30-0.50": "0.30 to 0.50", "0.50-0.80": "0.50 to 0.80", ">0.80": "over 0.80"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, (m, v, title) in zip(axes, panels):
        g = a[(a.measure == m) & (a.version == v)].set_index("band").loc[bands]
        y = np.arange(len(bands))[::-1]
        for yi, (band, r) in zip(y, g.iterrows()):
            ax.barh(yi, r.km_bound, height=0.55, color=SHADE, edgecolor=MUTED, lw=0.8, zorder=1, label="what the data alone allow" if yi == y[0] else None)
            for dy, (pi, lo, hi, colr, lab) in zip((0.12, -0.12), [(r.pi_ln, r.pi_ln_lo, r.pi_ln_hi, BLUE, "cure model, lognormal latency"), (r.pi_wb, r.pi_wb_lo, r.pi_wb_hi, ORANGE, "cure model, Weibull latency")]):
                ax.plot([lo, hi], [yi + dy, yi + dy], color=colr, lw=2, zorder=2, solid_capstyle="round")
                ax.plot([pi], [yi + dy], "o", ms=8, color=colr, mec=SURFACE, mew=1.5, zorder=3, label=lab if yi == y[0] else None)
            mark = f"n = {int(r.n)}" + (", passes both checks" if r.passes_two_checks else "")
            ax.annotate(mark, (1.0, yi + 0.3), xytext=(-2, 0), textcoords="offset points", ha="right", va="center", fontsize=8.6, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([names[b] for b in bands])
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        pct(ax, "x")
    axes[0].set_ylabel("depth of the bust (log points)")
    fig.supxlabel("share of busts that never regain their peak", fontsize=10, color=INK2, y=0.1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=9.2)
    suptitle(fig, "EU industry busts: the data bound the never-recover share; the latency family picks a point inside")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    note(fig, "Busts with troughs in the single-market and euro stages, 1995-2025. Grey bar: 0 up to Kaplan-Meier at the last time with 10 busts at risk. Lines: 90% Wald intervals.", y=-0.06)
    save(fig, "fig6_eu_busts.png")


def main() -> None:
    fig1_one_window_many_answers()
    fig2_failure_by_follow_up()
    fig3_coverage()
    fig4_checks_by_follow_up()
    fig5_rules()
    fig6_eu_busts()


if __name__ == "__main__":
    main()
