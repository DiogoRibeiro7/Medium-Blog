"""Figures for Part 4 of the Simpson's paradox series.

fig1_ab_segments.png      Hillstrom's experiment: segment effects around the aggregate, for e-mail against none
                          and for the two e-mails against each other
fig2_ramp.png             pooling a ramped test: the bias multiplier, and the re-enactment on real outcomes
fig3_monthly_kpi.png      monthly on-time rate, raw and mix-adjusted, and month-to-month changes split
fig4_carrier_gaps.png     carrier differences split into origin mix, hour mix and the within-cell part
fig5_fairness.png         calibration by decile, and metric gaps as strata are added
fig6_scanner.png          aggregate against standardised differences for every comparison in the scan

Run:  python figures.py
"""

from __future__ import annotations

import datetime
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from practice_numbers import (
    BIG,
    SEED,
    SHOW,
    carrier_gap,
    counts_by,
    difference,
    flights_scan,
    kitagawa,
    load_compas,
    load_flights,
    load_hillstrom,
    priors_band,
    ramp_draws,
    ramp_multiplier,
    ramp_reenactment,
    stratified_metric,
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


def figure_ab_segments(h) -> None:
    tm, tw, tc = (h["segment"] == a for a in ("Mens E-Mail", "Womens E-Mail", "No E-Mail"))
    covs = (("history_segment", "spend in past year"), ("zip_code", "area"), ("channel", "channel"),
            ("newbie", "new customer"), ("mens", "bought men's"), ("womens", "bought women's"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    labels, est, lo, hi = [], [], [], []
    for var, nice in covs:
        v = np.asarray(h[var]).astype(str)
        for level in sorted(set(v)):
            m = v == level
            d, se = difference(h["visit"][tm & m].sum(), (tm & m).sum(), h["visit"][tc & m].sum(), (tc & m).sum())
            name = level.split(") ")[-1].replace("Surburban", "Suburban").replace("$", r"\$") if var in ("history_segment", "zip_code") else level
            if var in ("newbie", "mens", "womens"):
                name = "yes" if level == "1" else "no"
            labels.append(f"{nice}: {name}")
            est.append(100 * d); lo.append(100 * (d - 1.96 * se)); hi.append(100 * (d + 1.96 * se))
    ypos = np.arange(len(labels))[::-1]
    agg = 100 * (h["visit"][tm].mean() - h["visit"][tc].mean())
    ax1.axvline(agg, color=INK2, lw=1.2, ls=(0, (4, 2)))
    ax1.hlines(ypos, lo, hi, color=BLUE, lw=2)
    ax1.plot(est, ypos, "o", color=BLUE, ms=5)
    ax1.set_yticks(ypos, labels, fontsize=8.5)
    ax1.set_xlabel("men's e-mail minus no e-mail, visits (percentage points)")
    ax1.text(agg + 0.1, ypos[0] + 0.6, f"all customers {agg:+.2f}", fontsize=8.5, color=INK2)
    ax1.set_title("E-mail against none: every segment close to the aggregate")
    ax1.set_ylim(-0.8, len(labels) + 0.3)

    labels2, est2, lo2, hi2 = [], [], [], []
    for var, nice in (("mens", "bought men's"), ("womens", "bought women's"), ("newbie", "new customer")):
        for level in (0, 1):
            m = h[var] == level
            d, se = difference(h["visit"][tm & m].sum(), (tm & m).sum(), h["visit"][tw & m].sum(), (tw & m).sum())
            labels2.append(f"{nice}: {'yes' if level else 'no'}")
            est2.append(100 * d); lo2.append(100 * (d - 1.96 * se)); hi2.append(100 * (d + 1.96 * se))
    ypos2 = np.arange(len(labels2))[::-1]
    agg2 = 100 * (h["visit"][tm].mean() - h["visit"][tw].mean())
    ax2.axvline(0, color=MUTED, lw=1)
    ax2.axvline(agg2, color=INK2, lw=1.2, ls=(0, (4, 2)))
    ax2.hlines(ypos2, lo2, hi2, color=ORANGE, lw=2)
    ax2.plot(est2, ypos2, "o", color=ORANGE, ms=5)
    ax2.set_yticks(ypos2, labels2, fontsize=8.5)
    ax2.set_xlabel("men's minus women's e-mail, visits (percentage points)")
    ax2.text(agg2 + 0.1, ypos2[0] + 0.45, f"all customers {agg2:+.2f}", fontsize=8.5, color=INK2)
    ax2.set_title("Two e-mails compared: real heterogeneity")
    ax2.set_ylim(-0.6, len(labels2) - 0.1)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ab_segments.png")
    plt.close(fig)


def figure_ramp(h) -> None:
    # the same random stream, in the same order, as section 1 of practice_numbers.py
    rng = np.random.default_rng(SEED)
    ramp_reenactment(h, rng)
    nv, nz, mv, mz = ramp_draws(h, rng, 2000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a = np.geomspace(0.005, 0.5, 200)
    ax1.plot(a, [ramp_multiplier(x) for x in a], color=BLUE, lw=2.2)
    ax1.plot([0.01], [ramp_multiplier(0.01)], "o", color=ORANGE, ms=7)
    ax1.annotate(f"1% on day 1: {ramp_multiplier(0.01):+.3f}", (0.01, ramp_multiplier(0.01)), (0.012, -0.58),
                 fontsize=9, color=INK2)
    ax1.axhline(0, color=MUTED, lw=1)
    ax1.set_xscale("log")
    ax1.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["0.5%", "1%", "2%", "5%", "10%", "20%", "50%"])
    ax1.set_xlabel("treatment share on day 1 (50% on day 2, equal traffic)")
    ax1.set_ylabel("bias / (day-1 baseline - day-2 baseline)")
    ax1.set_title("How much a baseline shift leaks into the pooled result")

    bins = np.linspace(-2, 5, 71)
    ax2.hist(100 * nv, bins=bins, color=ORANGE, alpha=0.75, label="pooled across days")
    ax2.hist(100 * mv, bins=bins, color=BLUE, alpha=0.75, label="Mantel-Haenszel by day")
    full = 100 * (h["visit"][h["segment"] == "Mens E-Mail"].mean() - h["visit"][h["segment"] == "Womens E-Mail"].mean())
    ax2.axvline(0, color=MUTED, lw=1)
    ax2.axvline(full, color=INK, lw=1.4, ls=(0, (4, 2)))
    ax2.text(full + 0.08, ax2.get_ylim()[1] * 0.92, f"full experiment {full:+.2f}", fontsize=8.5, color=INK2)
    ax2.set_xlabel("men's minus women's e-mail, visits (percentage points)")
    ax2.set_ylabel("re-enactments (of 2,000)")
    ax2.set_title("Re-enacted on real outcomes: pooling loses the win")
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_ramp.png")
    plt.close(fig)


def figure_monthly(fl) -> None:
    y = fl["ontime"]
    cells = (fl["carrier"], fl["origin"], fl["hour_band"])
    year = counts_by(y, *cells)
    w_year = {k: n / len(y) for k, (_, n) in year.items()}
    raw, adj, cms = [], [], []
    for m in range(1, 13):
        mm = fl["month"] == m
        cm = counts_by(y[mm], *(c[mm] for c in cells))
        common = [k for k in cm if k in w_year]
        wsum = sum(w_year[k] for k in common)
        raw.append(100 * y[mm].mean())
        adj.append(100 * sum(w_year[k] * cm[k][0] / cm[k][1] for k in common) / wsum)
        cms.append(cm)
    mix, within = [], []
    for c1, c2 in zip(cms[:-1], cms[1:]):
        common = [k for k in c1 if k in c2]
        n1 = sum(c1[k][1] for k in common); n2 = sum(c2[k][1] for k in common)
        p1 = np.array([c1[k][0] / c1[k][1] for k in common]); p2 = np.array([c2[k][0] / c2[k][1] for k in common])
        w1 = np.array([c1[k][1] / n1 for k in common]); w2 = np.array([c2[k][1] / n2 for k in common])
        mx, wi = kitagawa(p2, w2, p1, w1)
        mix.append(100 * mx); within.append(100 * wi)
    months = [datetime.date(2013, m, 1).strftime("%b") for m in range(1, 13)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(months, raw, "o-", color=INK2, lw=2, ms=4, label="as reported")
    ax1.plot(months, adj, "o--", color=ORANGE, lw=1.6, ms=4, label="at the year's carrier x airport x hour mix")
    ax1.set_ylabel("flights arriving on time (%)")
    ax1.set_title("A monthly KPI that mix adjustment barely moves")
    ax1.legend(loc="lower left", fontsize=9)
    x = np.arange(11)
    ax2.bar(x - 0.2, within, color=BLUE, width=0.4, label="within cells (rates)")
    ax2.bar(x + 0.2, mix, color=ORANGE, width=0.4, label="mix of cells")
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_xticks(x, [f"{a}-{b}" for a, b in zip(months[:-1], months[1:])], rotation=45, ha="right", fontsize=8.5)
    ax2.set_ylabel("change in on-time rate (points)")
    ax2.set_title("Month-to-month changes are almost all within cells")
    ax2.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_monthly_kpi.png")
    plt.close(fig)


def figure_carriers(fl) -> None:
    rows = [carrier_gap(fl, a, b) for a, b in itertools.combinations(BIG, 2) if (a, b) in SHOW]
    rows.sort(key=lambda r: r["total"])
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for i, r in enumerate(rows):
        pos, neg = 0.0, 0.0
        for key, col in (("origin", VIOLET), ("hour_band", AQUA), ("within", BLUE)):
            v = 100 * r[key]
            if v >= 0:
                ax.barh(i, v, left=pos, color=col, height=0.6)
                pos += v
            else:
                ax.barh(i, v, left=neg, color=col, height=0.6)
                neg += v
        ax.plot(100 * r["total"], i, "D", color=INK, ms=6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [f"{r['a']} minus {r['b']}" for r in rows])
    ax.set_xlabel("difference in on-time arrival rate (percentage points)")
    ax.set_title("Carrier gaps: how much is where and when they fly")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (VIOLET, AQUA, BLUE)]
    handles.append(plt.Line2D([], [], marker="D", color=INK, ls=""))
    ax.legend(handles, ["mix: origin airport", "mix: departure hour", "within airport x hour cells",
                        "reported difference"], loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_carrier_gaps.png")
    plt.close(fig)


def figure_fairness(cp) -> None:
    two = np.isin(cp["race"], ["African-American", "Caucasian"])
    for k in ("race", "sex", "age_cat", "decile", "priors", "charge", "recid"):
        cp[k] = cp[k][two]
    black = cp["race"] == "African-American"
    high = (cp["decile"] >= 5).astype(int)
    recid = cp["recid"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for grp, col, lab, off in ((black, ORANGE, "Black defendants", -0.12), (~black, BLUE, "white defendants", 0.12)):
        r, lo, hi = [], [], []
        for d in range(1, 11):
            m = grp & (cp["decile"] == d)
            ci = stats.binomtest(int(recid[m].sum()), int(m.sum())).proportion_ci()
            r.append(100 * recid[m].mean()); lo.append(100 * ci.low); hi.append(100 * ci.high)
        x = np.arange(1, 11) + off
        ax1.vlines(x, lo, hi, color=col, lw=1.6)
        ax1.plot(x, r, "o-", color=col, ms=4, lw=1.4, label=lab)
    ax1.set_xticks(range(1, 11))
    ax1.set_xlabel("COMPAS decile score")
    ax1.set_ylabel("rearrested within two years (%)")
    ax1.set_title("Recidivism by score: close, not identical")
    ax1.legend(loc="upper left", fontsize=9)

    coarse = priors_band(cp["priors"], [1, 4])
    fine = priors_band(cp["priors"], [1, 2, 3, 4, 6, 10])
    levels = [("aggregate", None), ("priors", coarse), ("age", cp["age_cat"]),
              ("priors x age", np.char.add(np.char.add(coarse, " | "), cp["age_cat"])),
              ("finer priors\nx age", np.char.add(np.char.add(fine, " | "), cp["age_cat"])),
              ("finer priors x age\nx sex x charge", np.char.add(np.char.add(np.char.add(
                  np.char.add(fine, " | "), cp["age_cat"]), np.char.add(" | ", cp["sex"])), np.char.add(" | ", cp["charge"])))]
    cols = {"base rate": MUTED, "selection rate": VIOLET, "FPR": ORANGE, "FNR": AQUA, "PPV": BLUE}
    names = {"base rate": "base rate", "selection rate": "labelled higher risk", "FPR": "false positive rate",
             "FNR": "false negative rate", "PPV": "positive predictive value"}
    for metric, col in cols.items():
        vals = []
        for _, s in levels:
            if s is None:
                vals.append(100 * stratified_metric(metric, recid, high, black, np.zeros(len(recid), int))[0])
            else:
                vals.append(100 * stratified_metric(metric, recid, high, black, s)[0])
        ax2.plot(range(len(levels)), vals, "o-", color=col, lw=1.8, ms=4, label=names[metric])
    ax2.axhline(0, color=INK2, lw=1)
    ax2.set_xticks(range(len(levels)), [l for l, _ in levels], fontsize=8)
    ax2.set_ylabel("Black minus white (percentage points)")
    ax2.set_title("Group gaps as strata are added")
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.43), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_fairness.png")
    plt.close(fig)


def figure_scanner(fl) -> None:
    found = flights_scan(fl)
    agg = np.array([100 * f["agg"] for f in found])
    std = np.array([100 * f["std"] for f in found])
    flip = np.array([f["flip"] for f in found])
    ev = np.array([f["flip"] and abs(f["agg"]) > 1.96 * f["agg_se"] and abs(f["std"]) > 1.96 * f["std_se"] for f in found])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, lim in ((ax1, 26), (ax2, 4.5)):
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(0, color=MUTED, lw=1)
        ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1.2, zorder=0)
        ax.scatter(agg[~flip], std[~flip], s=9, color=MUTED, alpha=0.6, lw=0, label="same sign")
        ax.scatter(agg[flip & ~ev], std[flip & ~ev], s=26, color=ORANGE, lw=0, label="opposite signs")
        ax.scatter(agg[ev], std[ev], s=60, facecolor="none", edgecolor=INK, lw=1.4,
                   label="opposite signs, both beyond 1.96 SE")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("aggregate difference (points)")
        ax.set_ylabel("standardised difference (points)")
    offsets = {"origin": (0.2, 0.3), "carrier": (-1.9, -0.55), "hour_band": (0.15, -0.5)}
    for f, a_, s_, e in zip(found, agg, std, ev):
        if e:
            dx, dy = offsets[f["stratifier"]]
            ax2.annotate(f"{f['a']}-{f['b']} by {f['stratifier'].replace('_', ' ')}", (a_, s_), (a_ + dx, s_ + dy),
                         fontsize=8, color=INK2)
    ax1.set_title(f"{len(found)} comparisons in the flight data")
    ax2.set_title("Close to zero, where reversals live")
    ax1.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_scanner.png")
    plt.close(fig)


if __name__ == "__main__":
    h = load_hillstrom()
    figure_ab_segments(h)
    figure_ramp(h)
    fl = load_flights()
    figure_monthly(fl)
    figure_carriers(fl)
    figure_fairness(load_compas())
    figure_scanner(fl)
    print("figures written to", OUT)
