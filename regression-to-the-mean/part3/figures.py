"""Figures for Part 3 of the regression-to-the-mean series.

fig1_bp_change.png        change from the first systolic reading to the mean of the next two, and noise by level and age
fig2_bp_threshold.png     chance of still being at or above 140 mmHg, in the same sitting and at a next visit
fig3_components.png       reliability of one season from the sampling model against the year-to-year correlation
fig4_rookies.png          rookie seasons and the season after, with Rookie of the Year winners marked
fig5_teams.png            team surplus split into run differential and the Pythagorean residual, two seasons running
fig6_calibration.png      predicted against observed regression for held-out selections, and three predictions for OPS

Run:  python figures.py
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from real_data_numbers import bp_conditional_curve, jk, jk_replicates, main, wcov, wmean

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

with contextlib.redirect_stdout(io.StringIO()):
    RES = main()


def bp_arrays():
    a = RES["nhanes"]
    x1, x2, x3, m23 = (a[c].to_numpy(float) for c in ("BPSys1", "BPSys2", "BPSys3", "m23"))
    return a, x1, x2, x3, m23, a.w4.to_numpy()


def figure_bp_change() -> None:
    a, x1, x2, x3, m23, w = bp_arrays()
    reps = jk_replicates(a.SDMVSTRA.to_numpy(), a.SDMVPSU.to_numpy(), w)
    edges = np.arange(88, 196, 4)
    idx = np.digitize(x1, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    ch = m23 - x1

    def binned(ww):
        num = np.bincount(idx[ok], weights=(ww * ch)[ok], minlength=len(edges) - 1)
        den = np.bincount(idx[ok], weights=ww[ok], minlength=len(edges) - 1)
        return num / den

    est, se = jk(binned, w, reps)
    counts = np.bincount(idx[ok], minlength=len(edges) - 1)
    centre = np.array([wmean(x1[idx == i], w[idx == i]) if counts[i] else np.nan for i in range(len(edges) - 1)])
    keep = counts >= 30
    mu1, mu23, b = wmean(x1, w), wmean(m23, w), RES["slope_bp"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    grid = np.linspace(88, 194, 200)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(mu23 - mu1, color=MUTED, lw=1.4, ls="--")
    ax.text(191, mu23 - mu1 + 0.35, "drift alone", color=INK2, fontsize=9, ha="right")
    ax.plot(grid, mu23 + b * (grid - mu1) - grid, color=INK2, lw=2)
    ax.annotate("drift + regression\n(one line, slope %.3f)" % b, xy=(104, mu23 + b * (104 - mu1) - 104),
                xytext=(96, -5.2), color=INK2, fontsize=9, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.errorbar(centre[keep], est[keep], yerr=1.96 * se[keep], fmt="o", ms=5, color=BLUE, ecolor=BLUE,
                elinewidth=1.2, capsize=0)
    ax.axvline(140, color=MUTED, lw=0.8, ls=":")
    ax.text(141, 3.4, "140", color=INK2, fontsize=9)
    ax.set_xlim(86, 196)
    ax.set_ylim(-11, 4.2)
    ax.set_xlabel("first systolic reading (mmHg)")
    ax.set_ylabel("mean of readings 2 and 3 minus reading 1 (mmHg)")
    ax.set_title("Change from the first reading, by first reading")

    age = a.Age.to_numpy()
    bins = [(80, 115), (115, 130), (130, 145), (145, 230)]
    xs = [107, 122, 137, 158]
    for (lo, hi), col, lab in zip(((18, 40), (40, 60), (60, 81)), (BLUE, ORANGE, VIOLET), ("18–39", "40–59", "60–80")):
        k = (age >= lo) & (age < hi)
        ys = []
        for blo, bhi in bins:
            kk = k & (m23 >= blo) & (m23 < bhi)
            ys.append(math.sqrt(wcov(x2[kk] - x3[kk], x2[kk] - x3[kk], w[kk]) / 2) if kk.sum() >= 40 else np.nan)
        bx.plot(xs, ys, "-o", color=col, lw=2, ms=6)
        last = max(i for i, v in enumerate(ys) if not np.isnan(v))
        bx.text(xs[last] + 2.5, ys[last], lab, color=INK2, fontsize=9, va="center")
    bx.set_xlim(100, 172)
    bx.set_ylim(3.0, 5.2)
    bx.set_xlabel("mean of readings 2 and 3 (mmHg)")
    bx.set_ylabel("SD of one reading around the sitting mean (mmHg)")
    bx.set_title("Noise in one reading, by level and age")
    fig.text(0.01, 0.005, "NHANES 2009–2012, adults with three readings (n = 10,622), examination weights. "
             "Bars: 95% intervals from the stratified jackknife; bins with at least 30 adults.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig1_bp_change.png")
    plt.close(fig)


def figure_bp_threshold() -> None:
    a, x1, x2, x3, m23, w = bp_arrays()
    reps = jk_replicates(a.SDMVSTRA.to_numpy(), a.SDMVPSU.to_numpy(), w)
    values = np.arange(120, 182, 2)
    idx = np.searchsorted(values, x1)
    ok = np.isin(x1, values)
    hit = (m23 >= 140).astype(float)

    def share(ww):
        num = np.bincount(idx[ok], weights=(ww * hit)[ok], minlength=len(values))
        den = np.bincount(idx[ok], weights=ww[ok], minlength=len(values))
        return num / den

    est, se = jk(share, w, reps)
    counts = np.bincount(idx[ok], minlength=len(values))
    keep = counts >= 15
    grid = np.linspace(118, 182, 300)
    within = RES["within_avg"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    same, nxt = bp_conditional_curve(x1, m23, w, 140, RES["sigma_visit"], grid)
    ax.errorbar(values[keep], est[keep], yerr=1.96 * se[keep], fmt="o", ms=5, color=INK, ecolor=INK2,
                elinewidth=1, capsize=0, zorder=3, label="observed, same sitting")
    ax.plot(grid, same, color=BLUE, lw=2, label="normal model, same sitting")
    ax.plot(grid, nxt, color=ORANGE, lw=2, label="next visit, day-to-day SD 8.3 (scenario)")
    for s_day, ls in ((4.0, (0, (1, 1.5))), (12.0, (0, (4, 2)))):
        _, nx2 = bp_conditional_curve(x1, m23, w, 140, math.sqrt(within ** 2 + s_day ** 2), grid)
        ax.plot(grid, nx2, color=ORANGE, lw=1.3, ls=ls, alpha=0.7, label=f"next visit, day-to-day SD {s_day:.0f}")
    ax.axvline(140, color=MUTED, lw=0.8, ls=":")
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(118, 182)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("first systolic reading (mmHg)")
    ax.set_ylabel("share with mean of readings 2–3 at or above 140")
    ax.set_title("Chance of still being at or above 140, given the first reading")
    fig.text(0.01, 0.005, "NHANES 2009–2012, examination weights; 95% jackknife intervals where at least 15 adults "
             "share the first reading.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig2_bp_threshold.png")
    plt.close(fig)


def figure_components() -> None:
    rel, yty = RES["reliability"], RES["year_to_year"]
    names = {"OPS": "OPS", "BABIP": "BABIP", "K": "strikeout rate", "BBr": "walk rate", "HRr": "home-run rate"}
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    g = np.linspace(0.3, 1.0, 50)
    ax.plot(g, g, color=MUTED, lw=1.2)
    ax.plot(g, 0.9 * g, color=MUTED, lw=1.0, ls="--")
    ax.text(0.985, 0.33, "solid line: r = reliability\n(true rates the same in both seasons)\n\n"
            "dashed line: r = 0.9 × reliability\n(true rates correlated 0.9)", color=INK2, fontsize=8.5,
            ha="right", va="bottom")
    place = {"OPS": (0.56, 0.60, "right"), "BABIP": (0.46, 0.355, "left"), "K": (0.805, 0.845, "right"),
             "BBr": (0.745, 0.785, "right"), "HRr": (0.685, 0.725, "right")}
    for k, lab in names.items():
        col = ORANGE if k == "BABIP" else BLUE
        ax.plot(rel[k], yty[k], "o", ms=9, color=col, mec="white", mew=1.5, zorder=3)
        tx, ty, ha = place[k]
        ax.annotate(f"{lab}  {rel[k]:.2f} → {yty[k]:.2f}", xy=(rel[k], yty[k]), xytext=(tx, ty),
                    color=INK2, fontsize=9, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=5))
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("reliability of one season, from the sampling model alone")
    ax.set_ylabel("correlation with the next season")
    ax.set_title("A second season is predictable from the noise in the first")
    fig.text(0.01, 0.005, "Regular seasons (300+ PA), 1955–2025 without 2020; 11,830 consecutive pairs. "
             "Lahman Baseball Database.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig3_components.png")
    plt.close(fig)


def figure_rookies() -> None:
    rk = RES["rookies"]
    r = rk[rk.PA_n >= 100]
    win = r.roy == 1
    beta = RES["rookie_line"]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.plot(r.OPS_d[~win], r.OPS_d_n[~win], "o", ms=3.2, color=MUTED, alpha=0.45, mec="none")
    ax.plot(r.OPS_d[win], r.OPS_d_n[win], "o", ms=6, color=ORANGE, mec="white", mew=0.8)
    g = np.linspace(-0.3, 0.32, 50)
    ax.plot(g, g, color=MUTED, lw=1.0, ls="--")
    ax.text(0.27, 0.29, "no regression", color=INK2, fontsize=8.5, rotation=36, ha="right")
    ax.plot(g, beta[0] + beta[1] * g, color=INK2, lw=2)
    ax.annotate("line: rookies who did not win\n(at age 24)", xy=(0.24, beta[0] + beta[1] * 0.24),
                xytext=(0.30, -0.22), color=INK2, fontsize=9, ha="right", va="top",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    x0, y0 = r.OPS_d[win].mean(), r.OPS_d_n[win].mean()
    pred = (beta[0] + beta[1] * r.OPS_d[win] + beta[2] * (r.age[win] - 24)).mean()
    ax.plot([x0], [x0], "s", ms=9, color=INK, mfc="white", mew=1.6, zorder=4)
    ax.annotate("", xy=(x0, y0), xytext=(x0, x0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=4)
    ax.plot([x0], [y0], "s", ms=9, color=INK, zorder=4)
    ax.plot([x0], [pred], "_", ms=16, mew=2.4, color=BLUE, zorder=4)
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    ax.text(x0 + 0.014, y0 + 0.006, f"winners' next season {y0:+.3f}", color=INK, fontsize=9, va="center", bbox=box)
    ax.text(x0 + 0.014, pred - 0.012, f"predicted {pred:+.3f}", color=INK2, fontsize=9, va="top", bbox=box)
    ax.text(x0 - 0.014, x0 + 0.004, f"winners' rookie season {x0:+.3f}", color=INK, fontsize=9, ha="right", bbox=box)
    ax.set_xlim(-0.3, 0.32)
    ax.set_ylim(-0.3, 0.32)
    ax.set_xlabel("rookie OPS minus the season mean of regulars")
    ax.set_ylabel("next-season OPS minus the season mean of regulars")
    ax.set_title("Rookie of the Year winners (orange) and the sophomore season")
    fig.text(0.01, 0.005, f"Rookie seasons with 300+ PA, 1955–2024, and 100+ PA the next season "
             f"({len(r)} players, {int(win.sum())} winners). Lahman Baseball Database.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig4_rookies.png")
    plt.close(fig)


def figure_teams() -> None:
    tp = RES["teams"]
    groups = (("top quarter\nby wins", tp.dev >= tp.dev.quantile(0.75)),
              ("top tenth by\nPythagorean residual", tp.luck >= tp.luck.quantile(0.9)),
              ("bottom quarter\nby wins", tp.dev <= tp.dev.quantile(0.25)))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    width = 0.34
    for i, (lab, sel) in enumerate(groups):
        g = tp[sel]
        for j, (pe, lu, tag) in enumerate(((g.pdev.mean(), g.luck.mean(), "season"),
                                           (g.pdev_n.mean(), g.luck_n.mean(), "next"))):
            x = i + (j - 0.5) * (width + 0.04)
            parts = [(162 * pe, BLUE), (162 * lu, ORANGE)]
            pos, neg = 0.0, 0.0
            for v, col in parts:
                if v >= 0:
                    ax.bar(x, v, width, bottom=pos, color=col, edgecolor="white", linewidth=2)
                    pos += v
                else:
                    ax.bar(x, v, width, bottom=neg, color=col, edgecolor="white", linewidth=2)
                    neg += v
            total = 162 * (pe + lu)
            ax.text(x, (pos if total >= 0 else neg) + (0.5 if total >= 0 else -0.5), f"{total:+.1f}", ha="center",
                    va="bottom" if total >= 0 else "top", color=INK, fontsize=9)
            ax.text(x, -18.6, "selected\nseason" if j == 0 else "next\nseason", ha="center", va="top", color=INK2,
                    fontsize=8.5)
        ax.text(i, 19.5, lab, ha="center", va="top", color=INK, fontsize=9.5)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-21.5, 20)
    ax.set_ylabel("wins above .500 over 162 games")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="run differential (Pythagorean expectation)"),
                       Patch(color=ORANGE, label="residual: wins beyond the runs")],
              loc="lower left", bbox_to_anchor=(0.0, 0.08), fontsize=9)
    ax.set_title("What a winning season is made of, and what survives to the next")
    fig.text(0.01, 0.005, "Major league teams 1961–2025 without 2020; 1,656 consecutive franchise seasons. "
             "Lahman Baseball Database.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig5_teams.png")
    plt.close(fig)


def figure_calibration() -> None:
    cal = RES["calibration"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1, 1.15]})
    cols = {"blood pressure": BLUE, "batting": ORANGE, "teams": AQUA}
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2)
    for dom, col in cols.items():
        p = [c[2] for c in cal if c[0] == dom]
        o = [c[3] for c in cal if c[0] == dom]
        ax.plot(p, o, "o", ms=7, color=col, mec="white", mew=1, label=dom)
    notes = {"OPS, top 5%": "OPS, top 1% and top 5%", "OPS, bottom 10%": "OPS, bottom 10%",
             "home-run rate, top 10%": "home-run rate, top 10%", "BABIP, bottom 10%": "BABIP, bottom 10%",
             "top 10% by residual": "luckiest tenth of teams"}
    for dom, lab, p, o, n in cal:
        if lab in notes:
            ha = "left" if p < 0.5 else "right"
            ax.annotate(notes[lab], (p, o), xytext=(8 if ha == "left" else -8, 0),
                        textcoords="offset points", ha=ha, va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted share of the deviation kept")
    ax.set_ylabel("observed share kept")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Held-out selections, predicted against observed")

    rows = RES["ops_rows"]
    labels = [r[0] for r in rows]
    order = ["bottom 1%", "bottom 10%", "bottom 25%", "top 25%", "top 10%", "top 5%", "top 1%"]
    rows = sorted(rows, key=lambda r: order.index(r[0]))
    xs = np.arange(len(rows))
    bx.plot(xs, [r[2] for r in rows], "-", color=BLUE, lw=2)
    bx.plot(xs, [r[3] for r in rows], "-o", color=VIOLET, lw=2, ms=5)
    bx.plot(xs, [r[4] for r in rows], "-o", color=ORANGE, lw=2, ms=5)
    bx.plot(xs, [r[1] for r in rows], "o", color=INK, ms=8, zorder=3)
    bx.text(6.15, rows[-1][2], "one slope", color=INK2, fontsize=9, va="center")
    bx.text(6.15, rows[-1][3], "player-specific\nnoise", color=INK2, fontsize=9, va="center")
    bx.text(6.15, rows[-1][4] + 0.012, "Tweedie", color=INK2, fontsize=9, va="center")
    bx.text(5, rows[5][1] + 0.012, "observed", color=INK, fontsize=9, ha="center")
    bx.set_xticks(xs)
    bx.set_xticklabels([r[0].replace(" ", "\n") for r in rows], fontsize=9)
    bx.set_xlim(-0.4, 7.4)
    bx.set_ylim(0.45, 0.8)
    bx.set_xlabel("OPS selection in the first season (1990–2024)")
    bx.set_ylabel("share of the deviation kept the next season")
    bx.set_title("OPS: the shape of the distribution sets the extremes")
    del labels
    fig.text(0.01, 0.005, "Predictions fitted on 2009–2010 (blood pressure) or 1955–1989 (baseball) and applied to "
             "later selections. Kept share: (mean next − overall mean) / (mean first − overall mean).",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig6_calibration.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_bp_change()
    figure_bp_threshold()
    figure_components()
    figure_rookies()
    figure_teams()
    figure_calibration()
