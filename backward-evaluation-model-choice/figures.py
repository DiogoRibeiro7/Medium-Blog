"""Figures for the article, from ``results/``.

    python figures.py

fig1  what each selection procedure fits and scores at a decision date (schematic)
fig2  H1 on the test set: L(FWD) - L(BWD) for every article, by list
fig3  every procedure against the forward backtest, by data set and candidate set
fig4  why: how well each score ranks next quarter's losses, and what each procedure picks
fig5  what the backward score rewards: how much worse each family looks on the past, and the one year it paid
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import common  # noqa: E402
from choice import SETS  # noqa: E402

RES, FIG = common.OUT, common.HERE / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog"
PROC = {
    "BWD": ("backward evaluation", RED, "o"),
    "BWD1": ("backward, nearest past year only", MAGENTA, "v"),
    "HYB": ("average of forward and backward scores", VIOLET, "D"),
    "ENS": ("equal-weight pool of the candidates", AQUA, "^"),
    "FIX": ("one model chosen on the development set", YELLOW, "s"),
    "ORACLE": ("best candidate in hindsight", INK2, "*"),
}
LISTS = {"us_state": "US states", "city": "cities", "solar_system": "Solar System bodies",
         "programming_language": "programming languages", "instrument": "musical instruments",
         "sport": "sports", "mammal": "mammals"}
DOM = {"dev_elec": "Development: electricity load, 25 countries",
       "dev_web": "Development: Wikipedia, 304 earlier articles",
       "test_web": "Test: Wikipedia, 245 new articles"}
SHORT = {"dev_elec": "dev: electricity", "dev_web": "dev: Wikipedia", "test_web": "test: Wikipedia"}
SET_LABEL = {"A": "set A: four model families", "B": "set B: four autoregression orders",
             "A8": "set A8: families × Gaussian or Student-t"}
CAND_LABEL = {"AR7_t": "autoregression", "SLM_t": "seasonal linear", "ETS_t": "exponential smoothing",
              "GBM_t": "gradient boosting"}
CAND_COLOR = {"SLM_t": BLUE, "ETS_t": ORANGE, "GBM_t": GREEN, "AR7_t": YELLOW}

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


def wide(loss, dom, st):
    return loss[(loss.domain == dom) & (loss.set == st)].pivot_table(index=["series", "group"], columns="proc",
                                                                     values="L").reset_index()


def fig1():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    L, gap = 364, 28
    o = 6 * L
    rows = [
        ("Forward backtest (FWD)", BLUE,
         ("fit: expanding window, refitted every 91 days", 0, o),
         ("scored: the last year, one day ahead", o - L, o)),
        ("Backward evaluation (BWD)", RED,
         ("fit: once, on the last year", o - L, o),
         ("scored: everything before the 28-day gap", 28, o - L - gap)),
        ("Backward, nearest year (BWD1)", MAGENTA,
         ("fit: once, on the last year", o - L, o),
         ("scored: only the year before the gap", o - 2 * L - gap, o - L - gap)),
    ]
    for i, (name, c, fit, score) in enumerate(rows):
        y = 2 - i
        ax.text(-60, y, name, ha="right", va="center", fontsize=10, fontweight="bold")
        for (lab, a, b), yy, alpha in ((fit, y + 0.16, 0.35), (score, y - 0.16, 1.0)):
            ax.barh(yy, b - a, left=a, height=0.24, color=c, alpha=alpha, lw=0)
            if b - a > 2.5 * L:
                ax.text((a + b) / 2, yy, lab, ha="center", va="center", fontsize=8.5,
                        color="white" if alpha == 1.0 else INK)
            else:
                ax.text(a - 25, yy, lab, ha="right", va="center", fontsize=8.5, color=INK)
    ax.axvline(o, color=INK, lw=1.2)
    ax.text(o + 20, 2.5, "decision date", va="center", fontsize=9)
    ax.axvspan(o, o + 91, color=GRID, lw=0, zorder=0)
    ax.text(o + 110, 0.0, "next 91 days:\nthe chosen model\nis deployed", ha="left", va="center", fontsize=8.5,
            color=INK2)
    ax.set_xlim(-1450, o + 560)
    ax.set_ylim(-0.55, 2.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(np.arange(0, o + 1, L))
    ax.set_xticklabels([f"{k}" for k in range(-6, 1)])
    ax.set_xlabel("years before the decision date (every candidate is scored the same way; the lowest score wins)")
    ax.set_title("What each procedure fits and what it scores at a decision date", loc="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig1_procedures.png")


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def fig2(loss):
    w = wide(loss, "test_web", "A")
    w["d"] = w.FWD - w.BWD
    order = w.groupby("group").d.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 5.6))
    rng = np.random.default_rng(0)
    lo, hi = np.quantile(w.d, [0.005, 0.995])
    pad = 0.1 * (hi - lo)
    for i, g in enumerate(order):
        x = w[w.group == g].d.to_numpy()
        xc = np.clip(x, lo - pad, hi + pad)
        ax.scatter(xc, i + rng.uniform(-0.22, 0.22, len(x)), s=16, color=BLUE, alpha=0.55, lw=0)
        m, h = ci(x)
        ax.errorbar(m, i, xerr=h, fmt="D", ms=8, color=INK, mfc=SURFACE, mew=1.6, capsize=4, zorder=5)
    m, h = ci(w.d)
    ax.errorbar(m, len(order) + 0.4, xerr=h, fmt="D", ms=9, color=INK, mfc=INK, capsize=4)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(list(range(len(order))) + [len(order) + 0.4])
    ax.set_yticklabels([f"{LISTS[g]} ({(w.group == g).sum()})" for g in order] + [f"all {len(w)} articles"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(lo - pad * 1.2, hi + pad * 1.2)
    ax.set_xlabel("L(forward backtest) − L(backward evaluation), nats per day  "
                  "(right of zero: backward evaluation chose better)")
    ax.set_title("Test set, set A: does choosing by backward evaluation beat the forward backtest?", loc="left")
    ax.text(0.995, 0.985, "dots: articles (clipped at the 0.5% and 99.5% quantiles)\ndiamonds: mean ± 95% interval",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig2_h1_by_list.png")


def fig3(loss):
    procs = ["BWD", "BWD1", "HYB", "FIX", "ENS", "ORACLE"]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.6), sharey=True)
    for i, dom in enumerate(common.DOMAINS):
        for j, st in enumerate(SETS):
            ax = axes[i, j]
            w = wide(loss, dom, st)
            fix = loss[loss.domain.isin(("dev_elec", "dev_web")) & (loss.set == st) & loss.proc.str.startswith("CAND:")]
            w["FIX"] = w[fix.groupby("proc").L.mean().idxmin()]
            for k, p in enumerate(procs):
                m, h = ci(w[p] - w.FWD)
                lab, c, mk = PROC[p]
                ax.errorbar(m, k, xerr=h, fmt=mk, ms=8, color=c, capsize=3, label=lab if (i, j) == (0, 0) else None)
            ax.axvline(0, color=INK2, lw=1)
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(SET_LABEL[st], fontsize=10)
            if j == 0:
                ax.set_ylabel(DOM[dom].replace(": ", ":\n"), fontsize=9)
            ax.set_yticks(range(len(procs)))
            ax.set_yticklabels(procs)
            ax.invert_yaxis() if (i, j) == (0, 0) else None
            ax.tick_params(labelsize=8.5)
            ax.xaxis.set_major_locator(MaxNLocator(5))
    for ax in axes[-1]:
        ax.set_xlabel("L(procedure) − L(forward backtest), nats per day\n(left of zero: better than the forward backtest)",
                      fontsize=8.5)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.43, 0.012))
    fig.suptitle("Every procedure against the forward backtest (mean ± 95% interval across series)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    credit(fig)
    save(fig, "fig3_all_procedures.png")


def fig4():
    q = pd.read_csv(RES / "decision_quality.csv")
    ch = pd.read_csv(RES / "choices.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    procs = [("FWD", "forward backtest", BLUE, "o"), ("HYB", "average of both", VIOLET, "D"),
             ("BWD1", "backward, nearest year", MAGENTA, "v"), ("BWD", "backward evaluation", RED, "s")]
    for k, dom in enumerate(common.DOMAINS):
        r = q[(q.domain == dom) & (q.set == "A")].set_index("proc")
        for j, (p, lab, c, mk) in enumerate(procs):
            a.scatter(r.loc[p, "spearman"], k + (j - 1.5) * 0.12, color=c, marker=mk, s=60,
                      label=lab if k == 0 else None, zorder=3)
    a.set_yticks(range(3))
    a.set_yticklabels([SHORT[d] for d in common.DOMAINS])
    a.invert_yaxis()
    a.set_xlim(-0.05, 1)
    a.axvline(0, color=INK2, lw=1)
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.set_xlabel("mean rank correlation of the scores with next quarter's losses")
    a.set_title("How well each score ranks the candidates (set A)", loc="left", fontsize=10.5)
    a.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    cands = list(SETS["A"])
    labels, y = [], 0
    for dom in common.DOMAINS:
        for p in ("FWD", "BWD"):
            c = ch[(ch.domain == dom) & (ch.proc == p)].set_index("choice").share.reindex(cands).fillna(0)
            left = 0.0
            for cand in cands:
                v = c[cand]
                if v > 0:
                    b.barh(y, v - 0.004, left=left, color=CAND_COLOR[cand], height=0.7)
                    if v >= 0.07:
                        b.text(left + v / 2, y, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8, color="white")
                left += v
            labels.append(f"{SHORT[dom]}, {p}")
            y += 1
        y += 0.4
    ys = [0, 1, 2.4, 3.4, 4.8, 5.8]
    b.set_yticks(ys)
    b.set_yticklabels(labels, fontsize=8.5)
    b.invert_yaxis()
    b.set_xlim(0, 1)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1])
    b.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    b.set_xlabel("share of decisions")
    b.set_title("Which family each procedure picks (set A)", loc="left", fontsize=10.5)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=CAND_COLOR[c], label=CAND_LABEL[c]) for c in cands], frameon=False, fontsize=8.5,
             ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig4_scores_and_choices.png")


def fig5():
    dec = pd.read_csv(RES / "decisions.csv.gz")
    d = dec[dec.set == "A"]
    cands = list(SETS["A"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    width = 0.25
    for k, dom in enumerate(common.DOMAINS):
        f, bw = d[(d.domain == dom) & (d.proc == "FWD")], d[(d.domain == dom) & (d.proc == "BWD")]
        vals = [bw[f"score:{c}"].mean() - f[f"score:{c}"].mean() for c in cands]
        xs = np.arange(len(cands)) + (k - 1) * width
        a.bar(xs, vals, width=width - 0.03, color=[BLUE, ORANGE, GREEN][k], label=SHORT[dom])
    a.set_xticks(range(len(cands)))
    a.set_xticklabels([CAND_LABEL[c].replace(" ", "\n", 1) for c in cands], fontsize=9)
    a.set_ylabel("backward score − forward score, nats per day")
    a.set_title("How much worse each family looks on the past", loc="left", fontsize=10.5)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    for k, (dom, mk) in enumerate(zip(common.DOMAINS, "o^s")):
        x = d[d.domain == dom]
        f = x[x.proc == "FWD"].set_index(["series", "date"]).realised_choice
        bw = x[x.proc == "BWD"].set_index(["series", "date"]).realised_choice
        g = (bw - f).rename("d").reset_index()
        g["year"] = pd.to_datetime(g.date).dt.year
        r = g.groupby("year").d.agg(["mean", "std", "count"])
        h = 1.96 * r["std"] / np.sqrt(r["count"])
        b.errorbar(r.index + (k - 1) * 0.12, r["mean"], yerr=h, fmt=mk + "-", ms=6, lw=1.2, capsize=2,
                   color=[BLUE, ORANGE, GREEN][k], label=SHORT[dom])
    b.axhline(0, color=INK2, lw=1)
    b.set_xlabel("year of the decision")
    b.set_ylabel("next-quarter loss, backward pick − forward pick")
    b.set_title("When the backward pick did better: below zero (mean ± 95% interval)", loc="left", fontsize=10.5)
    b.grid(axis="y", color=GRID, lw=0.8)
    b.set_axisbelow(True)
    b.xaxis.set_major_locator(MaxNLocator(integer=True))
    b.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig5_what_backward_rewards.png")


def main() -> None:
    loss = pd.read_csv(RES / "series_losses.csv")
    fig1()
    fig2(loss)
    fig3(loss)
    fig4()
    fig5()


if __name__ == "__main__":
    main()
