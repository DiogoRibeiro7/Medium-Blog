"""Figures for the article, from ``results/``.

    python figures.py

fig1  share of decisions that train on the backward window after a change, plain rule vs checked rule
fig2  gain of each scheme in each simulated scenario, Gaussian and Student-t models
fig3  gain of each scheme on each test series (pre-registered scoring)
fig4  the yearly decisions on the test series: expanding, backward window, switched by the check
fig5  rejection strength against persistence in the two groups of test series (S2)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import common  # noqa: E402

RES, FIG = common.OUT, common.HERE / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog"
STYLE = {
    "B": ("plain backward rule", RED, "o"),
    "C": ("backward rule + end-of-episode check", VIOLET, "D"),
    "P0.25": ("fixed pool, 0.25 on the backward window", BLUE, "s"),
    "PC0.25": ("fixed pool, 0.25 on the checked window", "#86b6ef", "s"),
    "AveW": ("pool over window lengths", AQUA, "^"),
}
REGIMES = ["variance_up", "variance_down", "mean_shift", "persistence_up"]
EPISODES = ["burst", "level_episode", "crash"]
SCEN = {"stationary": "No change", "variance_up": "Volatility rises", "variance_down": "Volatility falls",
        "mean_shift": "Level shifts", "persistence_up": "Persistence rises", "burst": "Volatility burst",
        "level_episode": "Level jump", "crash": "Crash and rebound"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def credit(fig):
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color=INK2)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def sim_gains(model: str) -> pd.DataFrame:
    df = pd.read_csv(RES / f"sims_{model}.csv.gz")
    df = df[df.phase == "all"].pivot_table(index=["scenario", "seed"], columns="scheme", values="nlpd")
    g = (-(df.sub(df["E"], axis=0))).groupby("scenario").mean()
    return g


def fig1():
    dg = pd.read_csv(RES / "sims_gaussian_decisions.csv")
    dt = pd.read_csv(RES / "sims_t_decisions.csv")
    scen = ["burst", "crash", "variance_up"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, s in zip(axes, scen):
        for d, ls, tag in ((dg, "-", "Gaussian"), (dt, (0, (4, 2)), "Student-t")):
            q = d[d.scenario == s].sort_values("since")
            ax.plot(q.since, q.share_B, color=RED, lw=2, ls=ls, marker="o", ms=4, label=f"plain rule, {tag} models")
            ax.plot(q.since, q.share_C, color=VIOLET, lw=2, ls=ls, marker="D", ms=4, label=f"checked rule, {tag} models")
        if s == "burst":
            ax.axvspan(0, 24, color="#d8d7d2", lw=0, zorder=0)
        elif s == "crash":
            ax.axvspan(0, 2, color="#d8d7d2", lw=0, zorder=0)
        ax.axvline(60, color=INK2, lw=0.8, ls=":")
        ax.text(63, 0.62, "episode leaves\nthe anchor block" if s != "variance_up" else "anchor block entirely\nafter the change",
                fontsize=8, color=INK2, va="top")
        ax.set_title(SCEN[s] + (" (lasting)" if s in REGIMES else " (temporary)"), loc="left")
        ax.set_xlabel("periods since the change (one decision every 12)")
        ax.set_xlim(-4, 232)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("share of decisions that train\non the backward window")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("After an episode the check sends the rule back to all the data; after a lasting change it stays out of the way",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig1_decisions_after_change.png")


def fig2():
    gg, gt = sim_gains("gaussian"), sim_gains("t")
    order = [*REGIMES, *EPISODES, "stationary"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, g, title, keys in ((axes[0], gg, "Gaussian models", ["B", "C", "P0.25", "AveW"]),
                               (axes[1], gt, "Student-t(5) models", ["B", "C", "P0.25", "PC0.25"])):
        y = np.arange(len(order))[::-1]
        off = np.linspace(-0.27, 0.27, len(keys))
        for k, dy in zip(keys, off):
            lab, col, mk = STYLE[k]
            ax.scatter(g.loc[order, k], y + dy, color=col, marker=mk, s=38, label=lab, zorder=3)
        for yi, s in zip(y, order):
            ax.plot([g.loc[s, "B"], g.loc[s, "C"]], [yi + off[0], yi + off[1]], color=INK2, lw=0.8, zorder=2)
        ax.axvline(0, color=INK, lw=0.8)
        ax.axhline(y[len(REGIMES)] + 0.5, color=GRID, lw=1)
        ax.axhline(y[len(REGIMES) + len(EPISODES)] + 0.5, color=GRID, lw=1)
        ax.set_yticks(y, [SCEN[s] for s in order])
        ax.set_title(title, loc="left")
        ax.set_xlabel("gain over the expanding window (nats per forecast)")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Simulations: the check keeps the gains after lasting changes, and removes most episode losses only with Gaussian models",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig2_simulated_gains.png")


def fig3():
    g = pd.read_csv(RES / "forward_gains.csv")
    g = g[(g.set == "test") & (g.errors == "t") & (g.scoring == "no_covid")]
    keys = [k for k in common.TEST if k in set(g.series)]
    tab = g.pivot(index="series", columns="scheme", values="gain").loc[keys]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(keys))[::-1]
    schemes = ["B", "C", "P0.25", "AveW"]
    off = np.linspace(-0.27, 0.27, len(schemes))
    for k, dy in zip(schemes, off):
        lab, col, mk = STYLE[k]
        ax.scatter(tab[k], y + dy, color=col, marker=mk, s=44, label=lab, zorder=3)
    for yi, key in zip(y, keys):
        ax.plot([tab.loc[key, "B"], tab.loc[key, "C"]], [yi + off[0], yi + off[1]], color=INK2, lw=0.8, zorder=2)
    split = y[[common.GROUP[k] for k in keys].index("B")] + 0.5
    ax.axhline(split, color=GRID, lw=1.2)
    ax.text(ax.get_xlim()[1] if False else 0.098, split + 0.15, "real activity", fontsize=8, color=INK2, ha="right")
    ax.text(0.098, split - 0.45, "rates and exchange rates", fontsize=8, color=INK2, ha="right")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [common.SHORT[k] for k in keys])
    ax.set_xlim(-0.03, 0.1)
    ax.set_xlabel("gain over the expanding window (nats per month, 1990–2026, March–December 2020 not scored)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("The pre-registered test: AR(3), Student-t(5) errors, five-year blocks, ten series outside the US", loc="left")
    credit(fig)
    fig.tight_layout()
    save(fig, "fig3_test_gains.png")


def fig4():
    d = pd.read_csv(RES / "forward_decisions.csv")
    d = d[d.set == "test"].copy()
    d["year"] = d.date.str[:4].astype(int)
    keys = [k for k in common.TEST if k in set(d.series)]
    years = np.arange(1990, d.year.max() + 1)
    colors = {0: "#efeeea", 1: "#f4b6a6", 2: VIOLET}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, errors, title in ((axes[0], "t", "Student-t(5) errors (primary)"), (axes[1], "gaussian", "Gaussian errors")):
        q = d[d.errors == errors]
        for i, k in enumerate(keys):
            r = q[q.series == k].set_index("year")
            for yr in years:
                if yr not in r.index:
                    continue
                row = r.loc[yr]
                state = 0 if not row.stopped else (2 if row.check_switched else 1)
                ax.add_patch(plt.Rectangle((yr - 0.45, len(keys) - 1 - i - 0.4), 0.9, 0.8, color=colors[state], lw=0))
        ax.axvspan(2019.55, 2020.45, color="none", ec=INK2, lw=0.8, ls=":")
        ax.set_xlim(1989.3, years[-1] + 0.7)
        ax.set_ylim(-0.7, len(keys) - 0.3)
        ax.set_yticks(range(len(keys))[::-1], [common.SHORT[k] for k in keys])
        ax.set_xticks(range(1990, years[-1] + 1, 5))
        ax.set_title(title, loc="left")
        ax.axhline(len(keys) - 5 - 0.5, color=INK2, lw=0.6)
    handles = [Patch(color=colors[0], label="walk did not stop: all data"),
               Patch(color=colors[1], label="walk stopped: backward window"),
               Patch(color=colors[2], label="walk stopped, check says the change ended: all data")]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.suptitle("Every yearly decision on the ten test series (dotted box: January 2020, the last decision before COVID-19)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, "fig4_decision_map.png")


def fig5():
    df = pd.read_csv(RES / "premise_triples.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, grp, title in ((axes[0], "A", "Real activity (five series)"), (axes[1], "B", "Rates and exchange rates (five series)")):
        d = df[df.group == grp]
        rho, p = spearmanr(d.R, d.P)
        pc = d.P.clip(-3, 3)
        col = np.where(d.P > 0, BLUE, ORANGE)
        ax.scatter(d.R, pc, c=col, s=34, edgecolor=SURFACE, lw=0.6, zorder=3)
        ax.axvline(common.Z_CRIT, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.axhline(0, color=INK, lw=0.8)
        strong = d[d.R > common.Z_CRIT]
        ax.text(0.02, 0.97, f"Spearman ρ = {rho:+.2f} (p = {p:.2f}), {len(d)} triples\n"
                f"rejections above 2.498: {int((strong.P > 0).sum())} of {len(strong)} persisted",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xscale("symlog", linthresh=1.0)
        tk = [-3, -1, 0, 1, 2.5, 5, 10, 20]
        ax.set_xticks(tk, [f"{v:g}" for v in tk])
        ax.minorticks_off()
        ax.set_xlabel("rejection strength R: z of block k's model on block k − 1 (symmetric log scale)")
        ax.set_title(title, loc="left")
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("persistence P (nats per month, clipped at ±3)\n> 0: the change lasted into block k + 1")
    fig.suptitle("The split seen in US data appears again outside the US: in rates and currencies, stronger rejections went with more lasting changes",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, "fig5_split_by_group.png")


def main() -> None:
    import sys
    only = set(sys.argv[1:])
    for f in (fig1, fig2, fig3, fig4, fig5):
        if not only or f.__name__ in only:
            f()


if __name__ == "__main__":
    main()
