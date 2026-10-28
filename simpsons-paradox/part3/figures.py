"""Figures for Part 3 of the Simpson's paradox series.

fig1_covid_age.png        China and Italy: case fatality by age, and the age of the cases
fig2_covid_time.png       Italy against China over time: total, direct and indirect effects
fig3_sat.png              SAT scores and teacher salaries, with the share of students taking the test
fig4_penguins.png         bill length and depth, pooled and within species
fig5_chance.png           how often reversals appear without a cause: random tables, samples, and baseball

Run:  python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from real_data_numbers import (
    AGES,
    SEED,
    baseball,
    cfr_effects,
    load_covid,
    load_penguins,
    load_sat,
    ols,
    random_tables,
)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#7a5cd6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
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
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 170,
})


def figure_covid_age() -> None:
    cov = load_covid("covid_countries.csv")
    ita = load_covid("covid_italy_time.csv")
    ch_c, ch_d = cov[("China", "17 February 2020")]
    it_c, it_d = ita[("Italy", "9 March 2020")]
    x = np.arange(9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    w = 0.38
    ax1.bar(x - w / 2, 100 * ch_d / ch_c, width=w, color=BLUE, label="China, 17 February")
    ax1.bar(x + w / 2, 100 * it_d / it_c, width=w, color=ORANGE, label="Italy, 9 March")
    ax1.set_xticks(x)
    ax1.set_xticklabels(AGES)
    ax1.set_xlabel("age")
    ax1.set_ylabel("case fatality rate (%)")
    ax1.set_title("Within every age band, Italy's rate is lower or equal")
    ax1.legend(loc="upper left", fontsize=9)
    for xi, (lab, tot, col) in enumerate((("China overall", 100 * ch_d.sum() / ch_c.sum(), BLUE),
                                          ("Italy overall", 100 * it_d.sum() / it_c.sum(), ORANGE))):
        ax1.axhline(tot, color=col, lw=1.2, ls=(0, (4, 2)))
        ax1.text(0.0 if xi == 0 else 2.4, tot + 0.35, f"{lab} {tot:.1f}%", fontsize=8.5, color=col)
    ax2.bar(x - w / 2, 100 * ch_c / ch_c.sum(), width=w, color=BLUE, label="China")
    ax2.bar(x + w / 2, 100 * it_c / it_c.sum(), width=w, color=ORANGE, label="Italy")
    ax2.set_xticks(x)
    ax2.set_xticklabels(AGES)
    ax2.set_xlabel("age")
    ax2.set_ylabel("share of confirmed cases (%)")
    ax2.set_title("But Italy's cases were much older")
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_covid_age.png")
    plt.close(fig)


def figure_covid_time() -> None:
    cov = load_covid("covid_countries.csv")
    ita = load_covid("covid_italy_time.csv")
    ch_c, ch_d = cov[("China", "17 February 2020")]
    order = sorted(ita, key=lambda k: (["March", "April", "May"].index(k[1].split()[1]), int(k[1].split()[0])))
    labels, tot, dire, ind = [], [], [], []
    for key in order:
        c, d = ita[key]
        t, da, ia, _ = cfr_effects(ch_c, ch_d, c, d)
        labels.append(key[1].replace(" 2020", "").replace("March", "Mar").replace("April", "Apr"))
        tot.append(100 * t)
        dire.append(100 * da)
        ind.append(100 * ia)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.plot(x, tot, "o-", color=INK, lw=2, ms=5, label="total effect (crude CFR difference)")
    ax.plot(x, ind, "o-", color=ORANGE, lw=2, ms=5, label="indirect effect, through the age of cases")
    ax.plot(x, dire, "o-", color=BLUE, lw=2, ms=5, label="direct effect, at China's age mix")
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Italy minus China (percentage points)")
    ax.set_title("The age of the cases explains the early gap; later, Italy's own rates rose")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_covid_time.png")
    plt.close(fig)


def figure_sat() -> None:
    d = load_sat()
    q = np.quantile(d["frac"], [1 / 3, 2 / 3])
    g = np.digitize(d["frac"], q)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    cols = (AQUA, VIOLET, ORANGE)
    names = (f"fewer than {q[0]:.0f}% take the SAT", f"{q[0]:.0f}% to {q[1]:.0f}%", f"more than {q[1]:.0f}%")
    for k in range(3):
        m = g == k
        ax.scatter(d["salary"][m], d["sat"][m], s=34, color=cols[k], alpha=0.85, lw=0, label=names[k])
    xs = np.linspace(d["salary"].min() - 1, d["salary"].max() + 1, 2)
    X = np.column_stack([np.ones(len(d["sat"])), d["salary"]])
    b, _, _ = ols(d["sat"], X)
    ax.plot(xs, b[0] + b[1] * xs, color=INK, lw=2, label=f"all states: {b[1]:+.1f} points per $1,000")
    X = np.column_stack([np.ones(len(d["sat"])), d["salary"], d["frac"]])
    b2, _, _ = ols(d["sat"], X)
    for k in range(3):
        f = d["frac"][g == k].mean()
        ax.plot(xs, b2[0] + b2[1] * xs + b2[2] * f, color=cols[k], lw=1.6, ls=(0, (4, 2)))
    ax.text(xs[1] - 0.2, b2[0] + b2[1] * xs[1] + b2[2] * d["frac"][g == 0].mean() + 8,
            f"holding the share taking fixed: {b2[1]:+.1f} per $1,000", fontsize=8.5, color=INK2, ha="right")
    ax.set_xlabel("average teacher salary, 1994-95 (thousand dollars)")
    ax.set_ylabel("average total SAT score")
    ax.set_title("Higher-paying states score lower, and more of their students sit the test")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sat.png")
    plt.close(fig)


def figure_penguins() -> None:
    p = load_penguins()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    cols = {"Adelie": BLUE, "Chinstrap": VIOLET, "Gentoo": ORANGE}
    for s, col in cols.items():
        m = p["species"] == s
        ax.scatter(p["length"][m], p["depth"][m], s=16, color=col, alpha=0.6, lw=0, label=s)
        X = np.column_stack([np.ones(m.sum()), p["length"][m]])
        b, _, _ = ols(p["depth"][m], X)
        xs = np.array([p["length"][m].min(), p["length"][m].max()])
        ax.plot(xs, b[0] + b[1] * xs, color=col, lw=2.2)
    X = np.column_stack([np.ones(len(p["length"])), p["length"]])
    b, _, _ = ols(p["depth"], X)
    xs = np.array([p["length"].min(), p["length"].max()])
    ax.plot(xs, b[0] + b[1] * xs, color=INK, lw=2.2, ls=(0, (5, 2)), label=f"all penguins: {b[1]:+.3f} mm per mm")
    ax.set_xlabel("bill length (mm)")
    ax.set_ylabel("bill depth (mm)")
    ax.set_title("Longer bills are shallower overall, and deeper within every species")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_penguins.png")
    plt.close(fig)


def figure_chance() -> None:
    # the same random stream, in the same order, as section 4 of real_data_numbers.py, so the plotted values
    # are the ones printed there
    rng = np.random.default_rng(SEED)
    _, _, sampled = random_tables(rng)
    bb = baseball(rng)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ns = [n for n, _, _ in sampled]
    ax1.plot(ns, [100 * r for _, r, _ in sampled], "o-", color=BLUE, lw=2.2)
    ax1.axhline(100 / 60, color=INK2, lw=1, ls=(0, (4, 2)))
    ax1.text(ns[-1], 100 / 60 + 0.05, "random population tables: 1/60", fontsize=8.5, color=INK2, ha="right")
    ax1.set_xscale("log")
    ax1.set_xlabel("sample size (log scale)")
    ax1.set_ylabel("share of samples with a reversal (%)")
    ax1.set_title("Spurious reversals in samples from populations without one")
    ax1.set_ylim(0, 2.0)

    per = 1000 / bb["pairs"]
    ax2.plot(bb["years"], bb["observed"] * per, "o-", color=ORANGE, lw=2, ms=4, label="observed")
    ax2.plot(bb["years"], bb["swap"].mean(axis=0) * per, "-", color=BLUE, lw=1.6,
             label="observed averages, at-bat totals swapped at random")
    ax2.plot(bb["years"], bb["constant"].mean(axis=0) * per, "-", color=MUTED, lw=1.6,
             label="true averages constant, binomial noise")
    ax2.set_xlabel("first season of the pair")
    ax2.set_ylabel("reversals per 1,000 pairs of players")
    ax2.set_title("Baseball: real changes in averages, not noise alone")
    ax2.set_ylim(0, None)
    ax2.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_chance.png")
    plt.close(fig)

if __name__ == "__main__":
    figure_covid_age()
    figure_covid_time()
    figure_sat()
    figure_penguins()
    figure_chance()
    print("figures written to", OUT)
