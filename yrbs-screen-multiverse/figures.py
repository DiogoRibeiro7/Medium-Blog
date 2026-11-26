"""The article's figures, from results/ (written by multiverse_numbers.py).

python figures.py      # figures/fig1_... to figures/fig6_...
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import yrbs_core as Y  # noqa: E402

RES = Y.HERE / "results"
FIG = Y.HERE / "figures"
Z = 1.959963984540054

INK = "#191919"
INK2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
GIRLS = "#257de0"  # categorical slot 1
BOYS = "#f47230"  # categorical slot 2
THIRD = "#00b894"  # categorical slot 3
MUTED = "#b8b8b8"
SOURCE = "Data: CDC Youth Risk Behavior Survey, national high-school files. Analysis: Diogo Ribeiro."

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": AXIS,
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
        "grid.linewidth": 1,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)

FORK_LABELS = {
    "measure": "Which question (wave)",
    "coding": "How heavy use is coded",
    "outcome": "Which outcome",
    "controls": "Which controls",
    "sample": "Who is analysed",
    "weighting": "Survey weights",
}
LEVEL_LABELS = {
    "linear": "hours or frequency as a number",
    "high_vs_rest": "3+ h (several a day+) vs everyone else",
    "high_vs_light": "3+ h (several a day+) vs light users",
    "top_vs_rest": "5+ h (hourly+) vs everyone else",
    "top_vs_light": "5+ h (hourly+) vs light users",
    "sad": "sad or hopeless",
    "consider": "considered suicide",
    "plan": "made a plan",
    "attempt": "attempted",
    "poor_mh": "poor mental health",
    "none": "none",
    "demographic": "sex, grade, race/ethnicity",
    "demographic_identity": "+ sexual identity",
    "pathways": "+ sleep, bullying, activity",
    "all": "everyone",
    "girls": "girls",
    "boys": "boys",
    "survey": "weighted",
    "unweighted": "unweighted",
    "tv 2017": "TV hours, 2017",
    "games 2017": "games/computer hours, 2017",
    "tv 2019": "TV hours, 2019",
    "games 2019": "games/computer hours, 2019",
    "screen 2021": "all screen time, 2021",
    "social 2023": "social media frequency, 2023",
}


def source(fig, y=-0.01):
    fig.text(0.0, y, SOURCE, fontsize=7.5, color=INK2, ha="left", va="top")


def load():
    specs = json.loads((RES / "specs.json").read_text())
    summary = json.loads((RES / "summary.json").read_text())
    return specs, summary


def fig1_trend(summary):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
    waves = sorted(int(w) for w in summary["trend"])
    for lab, col in (("girls", GIRLS), ("boys", BOYS)):
        p = np.array([summary["trend"][str(w)][lab][0] for w in waves]) * 100
        se = np.array([summary["trend"][str(w)][lab][1] for w in waves]) * 100
        a.errorbar(
            waves, p, yerr=Z * se, color=col, lw=2, marker="o", ms=7, mec="white", mew=2, capsize=0, elinewidth=1
        )
        a.text(waves[-1] + 0.4, p[-1], lab, color=INK, va="center", fontsize=9)
    a.set_xticks(waves)
    a.set_ylim(0, 65)
    a.set_xlim(waves[0] - 0.8, waves[-1] + 2.2)
    a.set_ylabel("% sad or hopeless for two weeks or more")
    a.set_title("Persistent sadness, high-school students")

    keys = list(summary["heavy"])
    names = [LEVEL_LABELS[k] for k in keys]
    vals_g = [summary["heavy"][k]["girls"][1] * 100 for k in keys]
    vals_b = [summary["heavy"][k]["boys"][1] * 100 for k in keys]
    yy = np.arange(len(keys))[::-1]
    b.barh(yy + 0.18, vals_g, height=0.34, color=GIRLS, label="girls")
    b.barh(yy - 0.18, vals_b, height=0.34, color=BOYS, label="boys")
    for y0, vg, vb in zip(yy, vals_g, vals_b, strict=True):
        b.text(vg + 1, y0 + 0.18, f"{vg:.0f}%", va="center", fontsize=8, color=INK)
        b.text(vb + 1, y0 - 0.18, f"{vb:.0f}%", va="center", fontsize=8, color=INK)
    b.set_yticks(yy)
    b.set_yticklabels(names)
    b.grid(axis="y", visible=False)
    b.grid(axis="x", visible=True)
    b.set_xlim(0, 100)
    b.set_xlabel("% in the heaviest category (5+ hours a day; for social media, hourly or more)")
    b.set_title("Heavy use, by the question asked that year")
    b.legend(loc="lower right")
    source(fig, -0.05)
    fig.savefig(FIG / "fig1_trend_and_questions.png")
    plt.close(fig)


def fig2_dose(summary):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.25})
    for ax, key, xt, title in (
        (a, "social 2023", list(Y.SOCIAL_LABELS), "2023: how often do you use social media?"),
        (b, "screen 2021", ["<1", "1", "2", "3", "4", "5+"], "2021: screen time on a school day (hours)"),
    ):
        for lab, col, off in (("girls", GIRLS, -0.08), ("boys", BOYS, 0.08)):
            rows = summary["dose"][key][lab]
            x = np.arange(len(rows)) + off
            p = np.array([r[1] for r in rows]) * 100
            se = np.array([r[2] for r in rows]) * 100
            ax.errorbar(x, p, yerr=Z * se, color=col, lw=2, marker="o", ms=7, mec="white", mew=2, elinewidth=1)
            ax.text(x[-1] + 0.25, p[-1], lab, va="center", fontsize=9, color=INK)
        ax.set_xticks(np.arange(len(xt)))
        ax.set_xticklabels(xt, rotation=35 if len(xt) > 6 else 0, ha="right" if len(xt) > 6 else "center", fontsize=8.5)
        ax.set_ylim(0, 75)
        ax.set_xlim(-0.5, len(xt) - 0.1)
        ax.set_title(title)
    a.set_ylabel("% sad or hopeless (95% CI)")
    source(fig, -0.12)
    fig.savefig(FIG / "fig2_dose_response.png")
    plt.close(fig)


def fig3_curve(specs):
    beta = np.array([s["beta"] for s in specs])
    scale = np.array([s["beta"] / s["b"] if s["b"] != 0 else 0 for s in specs])
    se = np.array([s["se_design"] for s in specs]) * np.abs(scale)
    t = np.array([s["b"] / s["se_design"] for s in specs])
    order = np.argsort(beta)
    n = len(beta)
    forks = list(Y.FORKS)
    levels = []
    for k in forks:
        vals = [s["measure"] + " " + str(s["wave"]) if k == "measure" else s[k] for s in specs]
        uniq = list(dict.fromkeys(vals))
        levels.append((k, uniq, np.array(vals)))
    rows = sum(len(u) for _, u, _ in levels)
    fig = plt.figure(figsize=(10, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.06)
    a = fig.add_subplot(gs[0])
    x = np.arange(n)
    sig = t[order] > Z
    a.vlines(x[~sig], (beta - Z * se)[order][~sig], (beta + Z * se)[order][~sig], color=GRID, lw=0.6)
    a.vlines(x[sig], (beta - Z * se)[order][sig], (beta + Z * se)[order][sig], color="#cce3fd", lw=0.6)
    a.scatter(x[~sig], beta[order][~sig], s=2, color=MUTED, lw=0, label="not significant at 5%")
    a.scatter(x[sig], beta[order][sig], s=2, color=GIRLS, lw=0, label="positive and significant at 5%")
    a.axhline(0, color=INK2, lw=1)
    med = float(np.median(beta))
    a.axhline(med, color=INK, lw=1, ls=(0, (1, 2)))
    a.text(n * 0.01, med, f" median {med:.3f}", va="bottom", fontsize=8.5, color=INK)
    a.set_xlim(-n * 0.005, n * 1.005)
    a.set_xticks([])
    a.set_ylabel("standardised coefficient (95% CI)")
    a.set_title(f"{n:,} analyses of the same question: does more use go with worse mental health?")
    a.legend(loc="upper left", markerscale=5, bbox_to_anchor=(0.0, 0.93))
    b = fig.add_subplot(gs[1], sharex=a)
    ticks, labs = [], []
    r = rows
    for k, uniq, vals in levels:
        for lv in uniq:
            r -= 1
            on = vals[order] == lv
            b.scatter(x[on], np.full(on.sum(), r), s=1.2, marker="|", color=INK2, lw=0.6)
            ticks.append(r)
            labs.append(LEVEL_LABELS.get(lv, lv))
        b.axhline(r - 0.5, color=GRID, lw=1)
        b.text(
            n * 1.01, r + (len(uniq) - 1) / 2, FORK_LABELS[k], va="center", fontsize=8.5, color=INK, fontweight="bold"
        )
    b.set_yticks(ticks)
    b.set_yticklabels(labs, fontsize=7.8)
    b.set_ylim(-0.6, rows - 0.4)
    b.grid(False)
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    b.set_xlabel("specifications, sorted by the size of the association")
    source(fig, 0.06)
    fig.savefig(FIG / "fig3_specification_curve.png")
    plt.close(fig)


def fig4_forks(summary):
    phi = summary["shapley"]
    order = sorted(phi, key=phi.get)
    by = summary["by_level"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.75})
    v = [phi[k] * 100 for k in order]
    a.barh(range(len(order)), v, height=0.5, color=GIRLS)
    for i, val in enumerate(v):
        a.text(val + 0.5, i, f"{val:.0f}%", va="center", fontsize=8.5, color=INK)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([FORK_LABELS[k] for k in order])
    a.grid(axis="y", visible=False)
    a.grid(axis="x", visible=True)
    a.set_xlabel("% of the variance of the 3,120 coefficients")
    a.set_title("Which choice moves the answer")
    a.set_xlim(0, max(v) * 1.25)
    yy, labs = [], []
    y = 0
    for k in reversed(order):
        for lv, (med, _share, _n) in by[k].items():
            b.scatter(med, -y, s=40, color=GIRLS, edgecolor="white", linewidth=1.5, zorder=3)
            yy.append(-y)
            labs.append(f"{LEVEL_LABELS.get(lv, lv)}")
            y += 1
        b.axhline(-y + 0.5, color=GRID, lw=1)
        y += 0.4
    b.set_yticks(yy)
    b.set_yticklabels(labs, fontsize=7.8)
    b.grid(axis="y", visible=False)
    b.grid(axis="x", visible=True)
    b.axvline(0, color=INK2, lw=1)
    b.set_xlabel("median standardised coefficient at each option")
    b.set_title("Median association under each option")
    source(fig, -0.06)
    fig.savefig(FIG / "fig4_which_choice_matters.png")
    plt.close(fig)


def fig5_null(summary):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.3})
    nl = summary["null_all"]
    a.hist(nl["null_median"], bins=50, color=MUTED, edgecolor="white", linewidth=0.5)
    a.axvline(nl["observed"], color=GIRLS, lw=2)
    a.text(nl["observed"], a.get_ylim()[1] * 0.9, f"  observed {nl['observed']:.3f}", color=INK, fontsize=8.5)
    a.set_xlabel("median standardised coefficient")
    a.set_title("If there were no association anywhere")
    a.set_ylabel("simulated curves")
    a.grid(axis="y", visible=False)
    b.hist(nl["null_count"], bins=50, color=MUTED, edgecolor="white", linewidth=0.5)
    b.axvline(nl["observed_count"], color=GIRLS, lw=2)
    b.text(nl["observed_count"], b.get_ylim()[1] * 0.9, f"  observed {nl['observed_count']:,}", color=INK, fontsize=8.5)
    b.set_xlabel("specifications positive and significant at 5%")
    b.set_title("...and how many would come out significant")
    b.grid(axis="y", visible=False)
    source(fig, -0.12)
    fig.savefig(FIG / "fig5_joint_null.png")
    plt.close(fig)


def fig6_comparators(summary):
    cmp_ = summary["comparators"]
    labs = list(dict.fromkeys(k.split("|")[0] for k in cmp_))
    fig, a = plt.subplots(figsize=(8, 3.6))
    for i, lab in enumerate(labs):
        y0 = len(labs) - 1 - i
        for sx, col, off in (("girls", GIRLS, 0.12), ("boys", BOYS, -0.12)):
            beta, bb, se = cmp_[f"{lab}|{sx}"]
            s = beta / bb if bb else 0
            a.errorbar(
                beta,
                y0 + off,
                xerr=Z * se * abs(s),
                fmt="o",
                color=col,
                ms=7,
                mec="white",
                mew=2,
                elinewidth=1,
                label=sx if i == 0 else None,
            )
    a.set_yticks(range(len(labs)))
    a.set_yticklabels(labs[::-1])
    a.axvline(0, color=INK2, lw=1)
    a.grid(axis="y", visible=False)
    a.grid(axis="x", visible=True)
    a.set_xlabel("standardised association with persistent sadness, 2023 (95% CI)")
    a.set_title("Social media frequency beside other things measured in the same survey")
    a.legend(loc="lower right")
    source(fig, -0.12)
    fig.savefig(FIG / "fig6_comparators.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    specs, summary = load()
    fig1_trend(summary)
    fig2_dose(summary)
    fig3_curve(specs)
    fig4_forks(summary)
    fig5_null(summary)
    fig6_comparators(summary)


if __name__ == "__main__":
    main()
