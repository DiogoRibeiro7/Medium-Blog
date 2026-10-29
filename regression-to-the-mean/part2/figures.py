"""Figures for Part 2 of the regression-to-the-mean series.

fig1_sources.png          how the correlation between occasions decays under three models, and a random walk
fig2_selection.png        expected regression by selection share and correlation; averaging repeated measurements
fig3_nonnormal.png        conditional means for skewed, heavy-tailed and two-group populations
fig4_series.png           New York temperature anomalies after extreme days, and the DAX in returns and levels
fig5_designs.png          five designs for a treatment given to the screened, and the anorexia trial

Run:  python figures.py
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mechanism_numbers import (
    SCREENING,
    SEED,
    expected_regression,
    nonnormal_pairs,
    ols,
    read,
    screening_designs,
    spearman_brown,
    three_sources,
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
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 170,
})


def figure_sources() -> None:
    models, walk = three_sources(np.random.default_rng(SEED + 2))
    lags = np.arange(1, 12)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = (BLUE, ORANGE, AQUA)
    theory = (lambda k: 0.5 + 0 * k, lambda k: 0.7 ** k, lambda k: 0.4 + 0.4 * 0.7 ** k)
    labels = ("stable trait + error", "transient AR(1) state", "trait + state + error")
    for (name, z), col, th, lab in zip(models.items(), cols, theory, labels):
        ax1.plot(lags, [np.corrcoef(z[:, 0], z[:, k])[0, 1] for k in lags], "o", color=col, ms=5, label=lab)
        kk = np.linspace(1, 11, 100)
        ax1.plot(kk, th(kk), color=col, lw=1.4)
    ax1.set_xlabel("occasions apart (k)")
    ax1.set_ylabel("correlation between occasions")
    ax1.set_title("What persists, and for how long")
    ax1.set_ylim(-0.05, 0.8)
    ax1.legend(loc="upper right", fontsize=9)
    ks = np.arange(1, 12)
    fwd = [np.polyfit(walk[:, 0], walk[:, k], 1)[0] for k in ks]
    bwd = [np.polyfit(walk[:, k], walk[:, 0], 1)[0] for k in ks]
    ax2.plot(ks, fwd, "o-", color=BLUE, ms=5, label="later on earlier")
    ax2.plot(ks, bwd, "s-", color=ORANGE, ms=5, label="earlier on later")
    ax2.axhline(1, color=MUTED, lw=1)
    ax2.set_xlabel("steps apart (k)")
    ax2.set_ylabel("regression slope")
    ax2.set_title("A random walk: no mean to return to")
    ax2.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_sources.png")
    plt.close(fig)


def figure_selection() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    shares = np.geomspace(0.001, 0.5, 200)
    for rho, col in zip((0.3, 0.5, 0.7, 0.9), (ORANGE, VIOLET, AQUA, BLUE)):
        ax1.plot(100 * shares, [expected_regression(rho, q)[1] for q in shares], color=col, lw=2, label=f"correlation {rho}")
    rng = np.random.default_rng(SEED + 3)
    n = 2_000_000
    x = rng.normal(0, 1, n)
    y = 0.6 * x + math.sqrt(1 - 0.36) * rng.normal(0, 1, n)
    for q in (0.2, 0.05):
        m = x > np.quantile(x, 1 - q)
        ax1.plot(100 * q, x[m].mean() - y[m].mean(), "D", color=INK, ms=6)
    ax1.plot(100 * shares, [expected_regression(0.6, q)[1] for q in shares], color=INK, lw=1, ls=(0, (4, 2)),
             label="correlation 0.6 (diamonds: simulation)")
    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_xticks([50, 20, 10, 5, 1, 0.1], ["50%", "20%", "10%", "5%", "1%", "0.1%"])
    ax1.set_xlabel("share of the population selected on the first measurement")
    ax1.set_ylabel("expected fall on the second (SD)")
    ax1.set_title("The more selective, the larger the fall")
    ax1.legend(loc="upper left", fontsize=8.5)
    lam = stats.norm.pdf(stats.norm.isf(0.1)) / 0.1
    ks = [1, 2, 3, 5, 10]
    sel = [lam * math.sqrt(0.6 / spearman_brown(0.6, k)) for k in ks]
    kept = [spearman_brown(0.6, k) * s for k, s in zip(ks, sel)]
    xpos = np.arange(len(ks))
    ax2.bar(xpos - 0.18, sel, width=0.36, color=MUTED, label="selected group on the selection variable")
    ax2.bar(xpos + 0.18, kept, width=0.36, color=BLUE, label="same group on a new measurement")
    ax2.set_xticks(xpos, [f"{k}" for k in ks])
    ax2.set_xlabel("measurements averaged before selecting the top 10% (reliability 0.6 each)")
    ax2.set_ylabel("mean deviation (SD of one measurement)")
    ax2.set_title("Averaging before selecting shrinks the fall")
    ax2.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_selection.png")
    plt.close(fig)


def figure_nonnormal() -> None:
    pairs = nonnormal_pairs(np.random.default_rng(SEED + 4))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, name, title in zip(axes, ("lognormal (skewed)", "Student t with 3 df (heavy tails)",
                                      "normal component, heavy-tailed noise", "two groups"),
                               ("Skewed shared component", "Heavy-tailed shared component", "Heavy-tailed noise",
                                "Two groups, means -2 and +2")):
        x, y = pairs[name][0], pairs[name][1]
        lo, hi = np.quantile(x, [0.001, 0.999])
        edges = np.linspace(lo, hi, 41)
        mids, means = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (x >= a) & (x < b)
            if m.sum() >= 200:
                mids.append(x[m].mean())
                means.append(y[m].mean())
        r = np.corrcoef(x, y)[0, 1]
        xs = np.array([lo, hi])
        ax.plot(xs, xs, color=MUTED, lw=1, ls=(0, (4, 2)), label="no regression")
        ax.plot(xs, y.mean() + r * (xs - x.mean()), color=ORANGE, lw=1.6, label=f"straight line, slope {r:.2f}")
        ax.plot(mids, means, "o", color=BLUE, ms=4, label="mean of the second, by first")
        ax.set_title(title)
        ax.set_xlabel("first measurement")
        ax.set_ylabel("second measurement")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_nonnormal.png")
    plt.close(fig)


def figure_series() -> None:
    w = [r for r in read("weather.csv") if r["origin"] == "JFK" and r["temp"] not in ("", "NA")]
    daily = {}
    for r in w:
        daily.setdefault((int(r["month"]), int(r["day"])), []).append(float(r["temp"]))
    keys = sorted(daily)
    temp = np.array([np.mean(daily[k]) for k in keys])
    doy = np.array([datetime.date(2013, m, d).timetuple().tm_yday for m, d in keys], float)
    X = np.column_stack([np.ones(len(doy))] + [f(2 * np.pi * h * doy / 365) for h in (1, 2) for f in (np.sin, np.cos)])
    beta, _ = ols(X, temp)
    anom = temp - X @ beta
    r1 = np.corrcoef(anom[:-1], anom[1:])[0, 1]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))
    for sel, col, lab in ((anom > np.quantile(anom, 0.95), ORANGE, "hottest 5% of days"),
                          (anom < np.quantile(anom, 0.05), BLUE, "coldest 5% of days")):
        idx = np.where(sel)[0]
        path = [anom[sel].mean()] + [np.mean([anom[i + k] for i in idx if i + k < len(anom)]) for k in range(1, 6)]
        ax1.plot(range(6), path, "o-", color=col, lw=2, ms=5, label=lab)
        ax1.plot(range(6), [path[0] * r1 ** k for k in range(6)], color=col, lw=1, ls=(0, (4, 2)))
    ax1.axhline(0, color=MUTED, lw=1)
    ax1.set_xlabel("days later")
    ax1.set_ylabel("temperature anomaly at JFK, 2013 (F)")
    ax1.set_title("Extreme days, then the days after")
    ax1.legend(loc="upper right", fontsize=8.5, title="dashed: AR(1) from the lag-1 correlation", title_fontsize=7.5)
    dax = np.array([float(r["DAX"]) for r in read("EuStockMarkets.csv")])
    ret = 100 * np.diff(np.log(dax))
    dec = np.digitize(ret[:-1], np.quantile(ret[:-1], np.linspace(0.1, 0.9, 9)))
    ax2.plot([ret[:-1][dec == d].mean() for d in range(10)], [ret[1:][dec == d].mean() for d in range(10)], "o-", color=VIOLET, ms=5)
    ax2.axhline(ret.mean(), color=MUTED, lw=1, ls=(0, (4, 2)))
    ax2.set_xlabel("today's DAX return, mean by decile (%)")
    ax2.set_ylabel("next day's return, mean (%)")
    ax2.set_title("Returns: regression is complete")
    ax3.scatter(dax[:-20], dax[20:], s=4, color=MUTED, alpha=0.5, lw=0)
    lim = [dax.min() * 0.95, dax.max() * 1.02]
    ax3.plot(lim, lim, color=INK, lw=1, ls=(0, (4, 2)))
    ax3.axhline(dax.mean(), color=ORANGE, lw=1.2, label=f"mean of the period, {dax.mean():.0f}")
    ax3.set_xlabel("DAX close today")
    ax3.set_ylabel("DAX close 20 trading days later")
    ax3.set_title("Levels: nothing to regress to")
    ax3.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_series.png")
    plt.close(fig)


def figure_designs() -> None:
    res = screening_designs(np.random.default_rng(SEED + 7), 1_000, 20_000)
    effect = SCREENING[4]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    names = {"naive": "before-after,\nscreening value", "predicted": "subtract predicted\nregression",
             "second baseline": "second baseline", "rct": "randomised\ncontrol", "rd": "regression\ndiscontinuity"}
    order = ["naive", "predicted", "second baseline", "rct", "rd"]
    ax1.boxplot([res[k] for k in order], orientation="horizontal", widths=0.5, showfliers=False,
                medianprops=dict(color=INK), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    ax1.axvline(effect, color=ORANGE, lw=1.6, label=f"true effect {effect:+.0f}")
    ax1.set_yticks(range(1, 6), [names[k] for k in order], fontsize=8.5)
    ax1.set_xlabel("estimated effect of treatment (units of the measurement)")
    ax1.set_title("Screened and treated: five ways to estimate the effect")
    ax1.legend(loc="lower right", fontsize=9)
    an = read("anorexia.csv")
    treat = np.array([r["Treat"] for r in an])
    pre = np.array([float(r["Prewt"]) for r in an])
    post = np.array([float(r["Postwt"]) for r in an])
    for g, col, lab in (("Cont", MUTED, "control"), ("CBT", BLUE, "cognitive behavioural"), ("FT", ORANGE, "family therapy")):
        m = treat == g
        ax2.scatter(pre[m], post[m] - pre[m], s=22, color=col, lw=0, alpha=0.85, label=lab)
        b = np.polyfit(pre[m], post[m] - pre[m], 1)
        xs = np.array([pre[m].min(), pre[m].max()])
        ax2.plot(xs, b[1] + b[0] * xs, color=col, lw=1.6)
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_xlabel("weight before treatment (lb)")
    ax2.set_ylabel("change in weight (lb)")
    ax2.set_title("Anorexia trial: change depends on the starting weight")
    ax2.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_designs.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_sources()
    figure_selection()
    figure_nonnormal()
    figure_series()
    figure_designs()
    print("figures written to", OUT)
