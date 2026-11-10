"""Figures for the article, from ``results/`` and ``data/``.

    python figures.py

fig1  two example series: daily electricity load and daily page views
fig2  which model family forecasts best (mean log loss of the expanding window)
fig3  mean gain over the expanding window of every window scheme, by family and domain
fig4  F1: pooling over windows against the backward rules, series by series
fig5  F2: losses removed and gains kept by the end-of-episode check, Gaussian and Student-t
fig6  F3: what heavy tails do to the plain rule and to the check
fig7  window decisions over time: share of series where the walk stopped and the check switched
fig8  log loss over time relative to the expanding window, electricity around March 2020
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
from models import FAMILIES  # noqa: E402

RES, FIG = common.OUT, common.HERE / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog"
STYLE = {
    "B": ("plain backward rule", RED, "o"),
    "C0": ("backward rule + end-of-episode check", VIOLET, "D"),
    "P0.25": ("fixed pool, 0.25 on the backward window", BLUE, "s"),
    "AveW": ("pool over window lengths", AQUA, "^"),
}
FAM_LABEL = {"AR": "autoregression", "SLM": "seasonal linear", "ETS": "exponential smoothing", "GBM": "gradient boosting"}
DOM_LABEL = {"elec": "Electricity load, 25 countries", "web": "Wikipedia page views, 304 articles"}

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


def gains() -> pd.DataFrame:
    return pd.read_csv(RES / "series_gains.csv")


def wide(g, dom, fam, dist, col="G"):
    return g[(g.domain == dom) & (g.family == fam) & (g.dist == dist)].pivot_table(index="series", columns="scheme", values=col)


def fig1():
    el = pd.read_csv(common.HERE / "data" / "electricity_daily.csv", parse_dates=["date"]).set_index("date")
    web = pd.read_csv(common.HERE / "data" / "wikipedia_daily.csv.gz", parse_dates=["date"]).set_index("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4))
    s = el["IT"] / 1000
    axes[0].plot(s.index, s, color=BLUE, lw=0.7)
    axes[0].axvspan(pd.Timestamp("2020-03-09"), pd.Timestamp("2020-05-18"), color="#d8d7d2", lw=0, zorder=0)
    axes[0].set_ylabel("GWh per day")
    axes[0].set_title("Italy, daily electricity load (shaded: national lockdown, March–May 2020)", loc="left")
    w = web["country:Ukraine"]
    axes[1].plot(w.index, w, color=ORANGE, lw=0.7)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("page views per day (log scale)")
    axes[1].set_title("English Wikipedia article 'Ukraine', daily page views", loc="left")
    for ax in axes:
        ax.grid(color=GRID, lw=0.5)
    credit(fig)
    fig.tight_layout()
    save(fig, "fig1_example_series.png")


def fig2():
    g = gains()
    e = g[g.scheme == "E"].groupby(["domain", "family", "dist"]).ll.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, dom in zip(axes, common.DOMAINS):
        x = np.arange(len(FAMILIES))
        for k, (dist, col, lab) in enumerate((("gauss", ORANGE, "Gaussian"), ("t", BLUE, "Student-t(5)"))):
            v = [e[(dom, f, dist)] for f in FAMILIES]
            ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=col, label=lab)
        ax.set_xticks(x, [FAM_LABEL[f] for f in FAMILIES])
        ax.set_title(DOM_LABEL[dom], loc="left")
        ax.set_ylabel("mean log loss, expanding window\n(nats per day; lower is better)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Which model family forecasts best, before any window rule", x=0.01, ha="left", fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, "fig2_model_families.png")


def fig3():
    g = gains()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, dom in zip(axes, common.DOMAINS):
        y = np.arange(len(FAMILIES))[::-1]
        off = np.linspace(-0.27, 0.27, 4)
        for (k, (lab, col, mk)), dy in zip(STYLE.items(), off):
            means, ses = [], []
            for f in FAMILIES:
                x = wide(g, dom, f, "t")[k]
                means.append(x.mean())
                ses.append(x.std(ddof=1) / np.sqrt(len(x)))
            ax.errorbar(means, y + dy, xerr=1.96 * np.array(ses), fmt=mk, color=col, ms=6, lw=1, label=lab)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y, [FAM_LABEL[f] for f in FAMILIES])
        ax.set_title(DOM_LABEL[dom], loc="left")
        ax.set_xlabel("mean gain over the expanding window (nats per day, ±1.96 se across series)")
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Window rules against the expanding window, Student-t distributions", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, "fig3_window_rules.png")


def fig4():
    g = gains()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (k, lab) in zip(axes, (("B", "pool over windows minus plain backward rule"),
                                   ("C0", "pool over windows minus checked rule"))):
        pos, data, cols = [], [], []
        for i, dom in enumerate(common.DOMAINS):
            for j, f in enumerate(FAMILIES):
                x = wide(g, dom, f, "t")
                data.append((x["AveW"] - x[k]).to_numpy())
                pos.append(i * 5 + j)
                cols.append(BLUE if dom == "elec" else ORANGE)
        bp = ax.boxplot(data, positions=pos, widths=0.6, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for p, d, c in zip(pos, data, cols):
            ax.scatter(np.full(len(d), p) + np.random.default_rng(0).uniform(-0.18, 0.18, len(d)), d, s=4, color=c, alpha=0.5)
            ax.scatter([p], [d.mean()], marker="D", color=INK, s=22, zorder=4)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(pos, [FAM_LABEL[f].replace(" ", "\n") for _ in common.DOMAINS for f in FAMILIES], fontsize=8)
        ax.set_title(lab, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.text(1.5, 1.0, "electricity", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=BLUE)
        ax.text(6.5, 1.0, "page views", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=ORANGE)
    axes[0].set_ylabel("difference in gain, series by series\n(nats per day; > 0: pooling better; diamond = mean)")
    fig.suptitle("F1: pooling over window lengths against the backward rules, Student-t distributions", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "fig4_f1_pooling.png")


def fig5():
    c = pd.read_csv(RES / "cells.csv")
    g = gains()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, dist, title in ((axes[0], "gauss", "Gaussian distributions (the F2 criterion)"),
                            (axes[1], "t", "Student-t distributions (for comparison)")):
        labels, removed, kept = [], [], []
        for dom in common.DOMAINS:
            for f in FAMILIES:
                x = wide(g, dom, f, dist)
                neg, pos = x["B"] < 0, x["B"] > 0
                removed.append(100 * (1 - x.loc[neg, "C0"].sum() / x.loc[neg, "B"].sum()) if neg.any() else np.nan)
                kept.append(100 * x.loc[pos, "C0"].sum() / x.loc[pos, "B"].sum() if pos.any() else np.nan)
                labels.append(f"{'elec' if dom == 'elec' else 'web'} {f}")
        xx = np.arange(len(labels))
        ax.bar(xx - 0.2, removed, width=0.38, color=VIOLET, label="share of the plain rule's losses removed")
        ax.bar(xx + 0.2, kept, width=0.38, color=AQUA, label="share of the plain rule's gains kept")
        ax.axhline(50, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.text(len(labels) - 0.5, 52, "pre-registered bar: 50%", ha="right", fontsize=8, color=INK2)
        ax.set_xticks(xx, labels, rotation=0, fontsize=8)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("percent")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    _ = c
    fig.suptitle("F2: what the end-of-episode check (c = 0) does to the plain rule's losses and gains", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig5_f2_check.png")


def fig6():
    g = gains()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels, a_m, a_s, b_g, b_t = [], [], [], [], []
    for dom in common.DOMAINS:
        for f in FAMILIES:
            xt, xg = wide(g, dom, f, "t"), wide(g, dom, f, "gauss")
            d = xt["B"] - xg.loc[xt.index, "B"]
            a_m.append(d.mean())
            a_s.append(d.std(ddof=1) / np.sqrt(len(d)))
            b_g.append((xg["C0"] - xg["B"]).mean())
            b_t.append((xt["C0"] - xt["B"]).mean())
            labels.append(f"{'elec' if dom == 'elec' else 'web'} {f}")
    xx = np.arange(len(labels))
    axes[0].bar(xx, a_m, yerr=1.96 * np.array(a_s), color=BLUE, capsize=3)
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_title("Plain rule's gain with Student-t minus with Gaussian", loc="left")
    axes[0].set_ylabel("nats per day (±1.96 se across series)")
    axes[1].bar(xx - 0.2, b_g, width=0.38, color=ORANGE, label="Gaussian")
    axes[1].bar(xx + 0.2, b_t, width=0.38, color=BLUE, label="Student-t(5)")
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_title("What the check adds to the plain rule (mean gain)", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(xx, labels, fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle("F3: heavy tails and the end-of-episode check", x=0.01, ha="left", fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig6_f3_tails.png")


def fig7():
    d = pd.read_csv(RES / "decisions.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, dom in zip(axes, common.DOMAINS):
        q = d[d.domain == dom]
        for dist, ls in (("t", "-"), ("gauss", (0, (4, 2)))):
            r = q[q.dist == dist].groupby("date").agg(stop=("stopped_at", lambda s: (s > 0).mean()),
                                                     sw=("switch_c0", "mean"))
            ax.plot(r.index, r.stop, color=RED, ls=ls, lw=1.6, label=f"walk stopped ({'Student-t' if dist == 't' else 'Gaussian'})")
            ax.plot(r.index, r.sw, color=VIOLET, ls=ls, lw=1.6, label=f"check switched back ({'Student-t' if dist == 't' else 'Gaussian'})")
        ax.axvline(pd.Timestamp("2020-03-01"), color=INK2, lw=0.8, ls=":")
        ax.set_title(DOM_LABEL[dom], loc="left")
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("share of series (decision every 91 days)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Window decisions over time (dotted line: March 2020)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig7_decisions.png")


def fig8():
    pm = pd.read_csv(RES / "periods_mean.csv", parse_dates=["period"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
    for ax, dom in zip(axes, common.DOMAINS):
        q = pm[(pm.domain == dom) & (pm.dist == "t") & (pm.family == "SLM")].pivot(index="period", columns="scheme", values="ll")
        for k, (lab, col, mk) in STYLE.items():
            ax.plot(q.index, q["E"] - q[k], color=col, marker=mk, ms=3, lw=1.3, label=lab)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axvline(pd.Timestamp("2020-03-01"), color=INK2, lw=0.8, ls=":")
        ax.set_title(DOM_LABEL[dom] + ", seasonal linear model", loc="left")
        ax.set_ylabel("gain over the expanding window\n(nats per day, mean across series, per 91 days)")
        ax.grid(color=GRID, lw=0.5)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("When the window rules win and lose, period by period (Student-t)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    credit(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig8_over_time.png")


def main() -> None:
    only = set(sys.argv[1:])
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        if not only or f.__name__ in only:
            f()


if __name__ == "__main__":
    main()
