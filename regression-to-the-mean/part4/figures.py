"""Figures for Part 4 of the regression-to-the-mean series.

fig1_winners_arms.png     the winner's lead in the half it was chosen on and in the other half, by number of arms
fig2_winners_kept.png     share of the winner's lead that survives, against reliability, with two predictions
fig3_nsw.png              earnings of NSW trainees, controls and comparison groups; estimates against the experiment
fig4_alert_path.png       carrier late shares around an alert day, observed and predicted
fig5_segments.png         where alerts come from under two rules, by the size of the segment

Run:  python figures.py
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from practice_numbers import main

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


def figure_winners_arms() -> None:
    label, R, wl = RES["sims"][0]
    ks = sorted(wl.k.unique())
    A = [wl[wl.k == k].yA.mean() for k in ks]
    B = [wl[wl.k == k].yB.mean() for k in ks]
    T = [wl[wl.k == k].yt.mean() for k in ks]
    N = [wl[wl.k == k].npm.mean() for k in ks]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(ks, np.expm1(A) * 100, "-o", color=ORANGE, lw=2.4, ms=7, label="lead in the half it was chosen on")
    ax.plot(ks, np.expm1(B) * 100, "-o", color=BLUE, lw=2.4, ms=7, label="lead in the other half")
    ax.plot(ks, np.expm1(T) * 100, "--", color=INK2, lw=1.6, label="true lead (known in a simulation)")
    ax.plot(ks, np.expm1(N) * 100, ":", color=VIOLET, lw=2.2, marker="s", ms=5,
            label="predicted from the first half, nonparametric prior")
    ax.set_xticks(ks)
    ax.set_xlabel("arms in the test")
    ax.set_ylabel("winner's click rate over the other arms' (%)")
    ax.set_ylim(0, 38)
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("The winner's lead shrinks when it is measured again")
    fig.text(0.01, 0.005, f"First simulated archive ({len(wl):,} tests, heavy-tailed effects, estimated one-arm reliability "
             f"{R:.2f} on half the traffic); assumptions in the text.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig1_winners_arms.png")
    plt.close(fig)


def figure_winners_kept() -> None:
    rows = []
    names = ("first archive:\nheavy-tailed effects", "second archive:\nnormal effects", "third archive:\nlarger heavy-tailed effects")
    for (label, R, wl), name in zip(RES["sims"], names):
        rows.append((name, R, wl.yB.mean() / wl.yA.mean(), wl.yt.mean() / wl.yA.mean(),
                     wl.lin.mean() / wl.yA.mean(), wl.npm.mean() / wl.yA.mean()))
    R, wl = RES["routes"]
    rows.append(("best carrier on a\nNew York route (real)", R, wl.pB.mean() / wl.pA.mean(), np.nan,
                 wl.lin.mean() / wl.pA.mean(), wl.npm.mean() / wl.pA.mean()))
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, R, obs, true, lin, npm) in zip(ys, rows):
        ax.plot([0, 1], [y, y], color=GRID, lw=1)
        ax.plot([R], [y], "|", ms=22, mew=2.5, color=MUTED)
        ax.plot([lin], [y + 0.13], "D", ms=7, color=AQUA)
        ax.plot([npm], [y - 0.13], "s", ms=7, color=VIOLET)
        if not np.isnan(true):
            ax.plot([true], [y], "o", ms=10, mfc="none", mec=INK, mew=1.6)
        ax.plot([obs], [y], "o", ms=8, color=BLUE)
        ax.text(-0.02, y, lab, ha="right", va="center", fontsize=9, color=INK2)
    ax.plot([], [], "o", color=BLUE, ms=8, label="observed in the other half")
    ax.plot([], [], "o", mfc="none", mec=INK, mew=1.6, ms=10, label="true (simulations only)")
    ax.plot([], [], "D", color=AQUA, ms=7, label="predicted, normal prior")
    ax.plot([], [], "s", color=VIOLET, ms=7, label="predicted, nonparametric prior")
    ax.plot([], [], "|", color=MUTED, ms=14, mew=2.5, label="estimated reliability of one arm")
    ax.legend(loc="upper right", fontsize=8.5, ncol=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of the winner's lead that survives")
    ax.set_title("How much of a winner's lead is real")
    fig.text(0.01, 0.005, "Predictions use only the half in which the winner was chosen. Routes: 96 New York routes with "
             "2+ carriers, odd days against even days, 2013 (nycflights13).", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(left=0.27)
    fig.savefig(OUT / "fig2_winners_kept.png")
    plt.close(fig)


def figure_nsw() -> None:
    nsw = RES["nsw"]
    dw, psid, cps, comps = nsw["dw"], nsw["psid"], nsw["cps"], nsw["comps"]
    tr, ct = dw[dw.treat == 1], dw[dw.treat == 0]
    years = [1974, 1975, 1978]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1, 1.1]})
    lines = [
        ("NSW trainees", tr, ORANGE, "-", 2.4),
        ("NSW controls (randomised)", ct, BLUE, "-", 2.4),
        ("CPS, 1975 earnings under 2,000", cps[cps.re75 < 2000], AQUA, "--", 1.8),
        ("PSID, 1975 earnings under 2,000", psid[psid.re75 < 2000], VIOLET, "--", 1.8),
    ]
    for lab, g, col, ls, lw in lines:
        y = [g.re74.mean() / 1000, g.re75.mean() / 1000, g.re78.mean() / 1000]
        ax.plot(years, y, ls, color=col, lw=lw, marker="o", ms=6)
        ax.text(1978.25, y[-1], lab, color=INK2, fontsize=9, va="center")
    ax.axvspan(1975.6, 1977.4, color=GRID, alpha=0.6, lw=0)
    ax.text(1976.5, 0.4, "programme\nperiod", ha="center", color=INK2, fontsize=8.5)
    ax.set_xticks(years)
    ax.set_xlim(1973.6, 1982.2)
    ax.set_ylim(0, 7.6)
    ax.set_ylabel("mean annual earnings (thousands of 1982 dollars)")
    ax.set_title("Earnings before and after, by group")

    pp = (tr.re78 - tr.re75).mean()
    exp = tr.re78.mean() - ct.re78.mean()
    se = np.hypot(tr.re78.std(ddof=1) / np.sqrt(len(tr)), ct.re78.std(ddof=1) / np.sqrt(len(ct)))
    items = [("trainees, before and after", pp, ORANGE)]
    for lab, key in (("PSID, everyone", ("PSID", "everyone")), ("CPS, everyone", ("CPS", "everyone")),
                     ("CPS, no 1975 earnings", ("CPS", "no 1975 earnings")),
                     ("CPS, 1975 under 2,000", ("CPS", "1975 earnings under 2,000")),
                     ("PSID, no 1975 earnings", ("PSID", "no 1975 earnings")),
                     ("PSID, 1975 under 2,000", ("PSID", "1975 earnings under 2,000"))):
        items.append((f"diff.-in-diff., {lab}", comps[key][5], MUTED))
    ys = np.arange(len(items))[::-1]
    bx.axvspan((exp - 1.96 * se) / 1000, (exp + 1.96 * se) / 1000, color=BLUE, alpha=0.12, lw=0)
    bx.axvline(exp / 1000, color=BLUE, lw=2)
    bx.text(exp / 1000 + 0.08, len(items) - 0.35, f"experiment {exp:,.0f}", color=BLUE, fontsize=9)
    bx.axvline(0, color=INK2, lw=0.8)
    for y, (lab, v, col) in zip(ys, items):
        bx.plot([v / 1000], [y], "o", ms=8, color=col if col != MUTED else INK2)
        bx.text(-1.35, y, lab, ha="right", va="center", color=INK2, fontsize=9)
        bx.text(v / 1000 + (0.15 if v >= 0 else -0.15), y + 0.28, f"{v:+,.0f}", ha="center", color=INK, fontsize=8.5)
    bx.set_yticks([])
    bx.set_xlim(-1.3, 5.6)
    bx.set_ylim(-0.7, len(items) - 0.1)
    bx.set_xlabel("estimated effect of training on 1978 earnings (thousands of 1982 dollars)")
    bx.set_title("Estimates against the experimental benchmark")
    bx.spines["left"].set_visible(False)
    fig.text(0.01, 0.005, "Dehejia–Wahba sample of the NSW demonstration (185 trainees, 260 controls) and the PSID and "
             "CPS comparison groups of LaLonde (1986). Band: 95% interval of the experimental estimate.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(wspace=0.55)
    fig.savefig(OUT / "fig3_nsw.png")
    plt.close(fig)


def figure_alert_path() -> None:
    obs, pred, allv, base = RES["alert_path"]
    days = np.arange(-2, 8)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axhline(base, color=MUTED, lw=1.2, ls="--")
    ax.plot(days, obs, "-", color=ORANGE, lw=2.4, marker="o", ms=6, label="carrier, observed")
    ax.plot(days, pred, "-", color=BLUE, lw=2.2, marker="s", ms=5, label="carrier, predicted from January–June")
    ax.plot(days, allv, "-", color=MUTED, lw=1.6, marker="o", ms=4, label="all flights, same days")
    ax.plot([], [], color=MUTED, lw=1.2, ls="--", label="carrier's trailing 28-day mean before the alert")
    ax.legend(loc="upper right", fontsize=9)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(-2.4, 7.4)
    ax.set_ylim(0.15, 0.78)
    ax.set_xticks(days)
    ax.set_xlabel("days from the alert")
    ax.set_ylabel("share of flights arriving 15+ minutes late or not at all")
    ax.set_title("What follows an alert on a carrier's daily late share")
    fig.text(0.01, 0.005, "Nine carriers with 20+ flights a day from New York, alerts July–December 2013 (n = 63). "
             "nycflights13.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig4_alert_path.png")
    plt.close(fig)


def figure_segments() -> None:
    tiers, seg = RES["segments"]
    order = ["under 10", "10 to 40", "40 or more"]
    rules = (("late share of 75% or more, July-December", ORANGE, "late share ≥ 75%"),
             ("z of 3 or more, July-December", BLUE, "binomial z ≥ 3 against the day"),
             ("rescaled z of 3 or more, July-December", VIOLET, "z ≥ 3 rescaled by the series' own spread"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.9))
    x = np.arange(3)
    w = 0.2
    flight = [tiers.loc[t, "flight_share"] for t in order]
    bars = [(-1.5 * w, flight, MUTED, "share of flights")]
    for i, (key, col, lab) in enumerate(rules):
        bars.append(((i - 0.5) * w, [seg[key][0].get(t, 0) for t in order], col, f"share of alerts, {lab}"))
    for off, vals, col, lab in bars:
        ax.bar(x + off, vals, w, color=col, edgecolor="white", linewidth=1.5, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\nflights a day" for t in order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Where the alerts come from")
    offs = (-0.22, 0.0, 0.22)
    for (key, col, lab), off in zip(rules, offs):
        tab = seg[key][1]
        for i, t in enumerate(order):
            n_, xs, ns, al = tab.get(t)
            bx.annotate("", xy=(i + off, ns), xytext=(i + off, xs), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            bx.plot([i + off + 0.07], [al], "_", ms=9, mew=2, color=INK2)
    for key, col, lab in rules:
        bx.plot([], [], color=col, lw=2, label=f"{lab}: alert day → next day")
    bx.plot([], [], "_", ms=9, mew=2, color=INK2, label="all flights on the alert day")
    bx.legend(loc="upper right", fontsize=8)
    bx.set_xticks(range(3))
    bx.set_xticklabels([f"{t}\nflights a day" for t in order])
    bx.set_xlim(-0.5, 2.5)
    bx.set_ylim(0, 1.3)
    bx.set_ylabel("late share")
    bx.set_title("Alert day and the day after")
    fig.text(0.01, 0.005, "34 carrier-airport series, New York (nycflights13); alerts July–December 2013. The rescaled rule "
             "divides z by each series' standard deviation of z in January–June.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig5_segments.png")
    plt.close(fig)

if __name__ == "__main__":
    figure_winners_arms()
    figure_winners_kept()
    figure_nsw()
    figure_alert_path()
    figure_segments()
